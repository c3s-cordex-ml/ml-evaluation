from __future__ import annotations

import os
from typing import Any
import click
import xarray as xr
import yaml
from ml_evaluation.definitions import (
    compute_fast_metrics,
    compute_slow_metrics,
    compute_multivar_metrics,
    compute_cdextremes_metrics,
)
import ml_evaluation.evaluation.indices as indices
from ml_evaluation.io import load_dataset, write_zarr, dask_default_params
from ml_evaluation.transformations import (
    align_dims,
    convert_threshold_units,
)


def _get_output_path(base: str, default_name: str) -> str:
    """
    Construct the output path based on a base path and a default filename.

    If the base path is an existing directory, the default filename is appended.
    If the base path ends with '.zarr', it is used directly as the output path.
    Otherwise, the base path is treated as a prefix, and the default filename
    is appended.

    Parameters
    ----------
    base : str
        The base output directory, file prefix, or final Zarr store path.
    default_name : str
        The default filename to append if `base` is a directory or prefix.

    Returns
    -------
    str
        The constructed file path or Zarr store path.
    """
    if base.endswith(".zarr"):
        return base
    if base.endswith("/") or os.path.isdir(base):
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, default_name)
    return f"{base}_{default_name}"


def _combine_seasons(
    season_metrics: list[dict[str, Any]],
    seasons_list: list[str | None],
) -> dict[str, Any]:
    """
    Combine list of metric dicts (one per season) into a single dict.

    Each value in the dict is concatenated along a new 'season' dimension.

    Parameters
    ----------
    season_metrics : list[dict[str, Any]]
        List of metric dictionaries, one for each season.
    seasons_list : list[str | None]
        List of season names corresponding to each dictionary in season_metrics.

    Returns
    -------
    dict[str, Any]
        Combined dictionary where each metric has a 'season' dimension.
    """
    if not season_metrics:
        return {}

    # Convert None or other non-string seasons to string for coords
    season_coords: list[str] = [
        str(s) if s is not None else "all" for s in seasons_list
    ]

    keys = season_metrics[0].keys()
    combined: dict[str, Any] = {}
    for key in keys:
        items = [m[key] for m in season_metrics]
        # Check if they are xarray objects
        if all(isinstance(item, (xr.Dataset, xr.DataArray)) for item in items):
            concat_dim = xr.DataArray(season_coords, dims="season", name="season")
            combined[key] = xr.concat(
                items, dim=concat_dim, coords="minimal", compat="override"
            )
        else:
            # In case some item is a float/int/scalar, convert to DataArray
            da_items: list[xr.DataArray | xr.Dataset] = []
            for item in items:
                if isinstance(item, (xr.Dataset, xr.DataArray)):
                    da_items.append(item)
                else:
                    da_items.append(xr.DataArray(float(item)))
            concat_dim = xr.DataArray(season_coords, dims="season", name="season")
            combined[key] = xr.concat(
                da_items, dim=concat_dim, coords="minimal", compat="override"
            )
    return combined


def run_fast(
    x0: str,
    x1: str,
    var: str,
    output: str,
    seasons: str | list[str] | None = None,
    align_leap_days: str | None = None,
    force_load: bool = False,
    use_dask: bool = False,
) -> None:
    """
    Run the fast-track evaluation workflow.

    Computes Mean Bias, RMSE, and Pearson Correlation between x0 and x1
    for the specified variable and optionally filters by season.
    If seasons is a list, loops over all selected seasons and integrates
    season as a dimension of the output Zarr.

    Parameters
    ----------
    x0 : str
        Path to ground-truth NetCDF file.
    x1 : str
        Path to predictions NetCDF file.
    var : str
        Variable name to evaluate.
    output : str
        Directory, file prefix, or final Zarr store path.
    seasons : str | list[str] | None, default None
        Optional season/month standard name(s) (e.g. DJF, MAM).
    align_leap_days : str | None, default None
        Option to align datasets to standard calendar by filling missing leap days.
    force_load : bool, default False
        Option to load data into memory for quick testing of small datasets.
    use_dask : bool, default False
        Option to use Dask for parallel processing.

    Returns
    -------
    None
    """
    if use_dask:
        from dask.distributed import Client

        _client = Client(**dask_default_params)
        click.echo("Using dask")

    click.echo(f"Running fast-track evaluation for variable '{var}'...")
    click.echo(f"Loading reference: {x0}")
    x0_ds = load_dataset(
        x0, align_leap_days=align_leap_days, chunks="auto" if use_dask else None
    )
    click.echo(f"Loading prediction: {x1}")
    x1_ds = load_dataset(
        x1, align_leap_days=align_leap_days, chunks="auto" if use_dask else None
    )

    x0_ds = x0_ds[[var]]
    x1_ds = x1_ds[[var]]

    # Check for different pr units in target vs prediction
    if var == "pr":
        if x1_ds["pr"].max() > 1 and x0_ds["pr"].max() < 1:
            click.echo("WARNING: pr converted from mm/day to kg/m2/s")
            x1_ds["pr"] = x1_ds["pr"] / 86400.0

    x0_ds, x1_ds, member_dim = align_dims(x0_ds, x1_ds)

    # Force loading when testing on smaller datasets so quick to run
    # Works well on 6 years 3 members
    if force_load:
        x0_ds.load()
        x1_ds.load()

    scale, shift = convert_threshold_units(x0_ds, var)

    # Determine if seasons is a list
    is_list = isinstance(seasons, list)
    seasons_list: list[str | None] = seasons if is_list else [seasons]  # type: ignore

    season_metrics_field = []
    season_metrics_series = []
    season_metrics_float = []
    season_metrics_pr_field = []
    season_metrics_psd = []

    for season in seasons_list:
        if is_list:
            click.echo(f"Processing season: {season}")

        m_field, m_series, m_float, m_pr_field, m_psd = compute_fast_metrics(
            x0_ds, x1_ds, var, season, scale, shift
        )

        # Store for combination later
        if is_list:
            season_metrics_field.append(m_field)
            season_metrics_series.append(m_series)
            season_metrics_float.append(m_float)
            season_metrics_psd.append(m_psd)
            if var == "pr":
                season_metrics_pr_field.append(m_pr_field)
        else:
            metrics_field = m_field
            metrics_series = m_series
            metrics_float = m_float
            metrics_psd = m_psd
            if var == "pr":
                metrics_pr_field = m_pr_field

    # Save/Combine metrics
    click.echo("Combining and saving metrics...")
    if is_list:
        combined_field = _combine_seasons(season_metrics_field, seasons_list)
        combined_series = _combine_seasons(season_metrics_series, seasons_list)
        combined_float = _combine_seasons(season_metrics_float, seasons_list)
        combined_psd = _combine_seasons(season_metrics_psd, seasons_list)
        if var == "pr":
            combined_pr_field = _combine_seasons(season_metrics_pr_field, seasons_list)
    else:
        combined_field = metrics_field
        combined_series = metrics_series
        combined_float = metrics_float
        combined_psd = metrics_psd
        if var == "pr":
            combined_pr_field = metrics_pr_field

    metrics_path_field = _get_output_path(output, "metrics_fast_field.zarr")
    write_zarr(combined_field, metrics_path_field, mode="w", var=var)
    metrics_path_series = _get_output_path(output, "metrics_fast_series.zarr")
    write_zarr(combined_series, metrics_path_series, mode="w", var=var)
    metrics_path_float = _get_output_path(output, "metrics_fast_float.zarr")
    write_zarr(combined_float, metrics_path_float, mode="w", var=var)

    if var == "pr":
        metrics_pr_path = _get_output_path(output, "metrics_pr_field.zarr")
        write_zarr(combined_pr_field, metrics_pr_path, mode="w", var=var)

    metrics_psd_path = _get_output_path(output, "metrics_psd.zarr")
    write_zarr(combined_psd, metrics_psd_path, mode="w", var=var)

    click.echo(f"Saved all fast-track metrics to {metrics_path_field} etc...")
    click.echo("Fast-track evaluation complete.")


def run_slow(
    x0: str,
    x1: str,
    var: str,
    output: str,
    seasons: str | list[str] | None = None,
    align_leap_days: str | None = None,
    force_load: bool = False,
    use_dask: bool = False,
) -> None:
    """
    Run the slow-track evaluation workflow.

    Computes Power Spectral Density (PSD), Radially Averaged Log Spectral Distance (RALSD),
    and Spatial Wasserstein Distance.
    If seasons is a list, loops over all selected seasons and integrates
    season as a dimension of the output Zarr.

    Parameters
    ----------
    x0 : str
        Path to ground-truth NetCDF file.
    x1 : str
        Path to predictions NetCDF file.
    var : str
        Variable name to evaluate.
    output : str
        Directory, file prefix, or final Zarr store path.
    seasons : str | list[str] | None, default None
        Optional season/month standard name(s) (e.g. DJF, MAM).
    align_leap_days : str | None, default None
        Option to align datasets to standard calendar by filling missing leap days.
    force_load : bool, default False
        Option to load data into memory for quick testing of small datasets.
    use_dask : bool, default False
        Option to use Dask for parallel processing.

    Returns
    -------
    None
    """
    if use_dask:
        from dask.distributed import Client

        _client = Client(**dask_default_params)
        click.echo("Using dask")

    click.echo(f"Running slow-track evaluation for variable '{var}'...")
    click.echo(f"Loading reference: {x0}")
    x0_ds = load_dataset(
        x0, align_leap_days=align_leap_days, chunks="auto" if use_dask else None
    )
    click.echo(f"Loading prediction: {x1}")
    x1_ds = load_dataset(
        x1, align_leap_days=align_leap_days, chunks="auto" if use_dask else None
    )

    x0_ds = x0_ds[[var]]
    x1_ds = x1_ds[[var]]

    # Check for different pr units in target vs prediction
    if var == "pr":
        if x1_ds["pr"].max() > 1 and x0_ds["pr"].max() < 1:
            click.echo("WARNING: pr converted from mm/day to kg/m2/s")
            x1_ds["pr"] = x1_ds["pr"] / 86400.0

    x0_ds, x1_ds, member_dim = align_dims(x0_ds, x1_ds)

    # Force loading when testing on smaller datasets so quick to run
    # Works well on 6 years 3 members
    if force_load:
        x0_ds.load()
        x1_ds.load()

    scale, shift = convert_threshold_units(x0_ds, var)

    is_list = isinstance(seasons, list)
    seasons_list: list[str | None] = seasons if is_list else [seasons]  # type: ignore

    season_metrics = []
    season_metrics_tasmax = []
    season_metrics_tasmin = []
    season_metrics_hist = []

    for season in seasons_list:
        if is_list:
            click.echo(f"Processing season: {season}")

        m_slow, m_hist, m_tasmax, m_tasmin = compute_slow_metrics(
            x0_ds, x1_ds, var, season, scale, shift, member_dim
        )

        if is_list:
            season_metrics.append(m_slow)
            season_metrics_hist.append(m_hist)
            if var == "tasmax":
                season_metrics_tasmax.append(m_tasmax)
            if var == "tasmin":
                season_metrics_tasmin.append(m_tasmin)
        else:
            metrics = m_slow
            metrics_hist = m_hist
            if var == "tasmax":
                metrics_tasmax = m_tasmax
            if var == "tasmin":
                metrics_tasmin = m_tasmin

    # Save/Combine metrics
    click.echo("Combining and saving metrics...")
    if is_list:
        combined_metrics = _combine_seasons(season_metrics, seasons_list)
        combined_hist = _combine_seasons(season_metrics_hist, seasons_list)
        if var == "tasmax":
            combined_tasmax = _combine_seasons(season_metrics_tasmax, seasons_list)
        if var == "tasmin":
            combined_tasmin = _combine_seasons(season_metrics_tasmin, seasons_list)
    else:
        combined_metrics = metrics
        combined_hist = metrics_hist
        if var == "tasmax":
            combined_tasmax = metrics_tasmax
        if var == "tasmin":
            combined_tasmin = metrics_tasmin

    metrics_path = _get_output_path(output, "metrics_slow_field.zarr")
    write_zarr(combined_metrics, metrics_path, mode="w", var=var)

    if var == "tasmax":
        metrics_path = _get_output_path(output, "metrics_tasmax_field.zarr")
        write_zarr(combined_tasmax, metrics_path, mode="w", var=var)

    if var == "tasmin":
        metrics_path = _get_output_path(output, "metrics_tasmin_field.zarr")
        write_zarr(combined_tasmin, metrics_path, mode="w", var=var)

    metrics_path = _get_output_path(output, "metrics_hist.zarr")
    write_zarr(combined_hist, metrics_path, mode="w", var=var)

    click.echo(f"Saved all slow-track metrics to {metrics_path} etc...")
    click.echo("Slow-track evaluation complete.")


def run_multivar(
    x0: str,
    x1: str,
    vars: list[str],
    output: str,
    seasons: str | list[str] | None = None,
    align_leap_days: str | None = None,
    force_load: bool = False,
    use_dask: bool = False,
) -> None:
    """
    Run the multi-variate evaluation workflow.

    Computes Pearson and Spearman Correlation between two variables
    for reference and prediction datasets, along with their biases.
    If seasons is a list, loops over all selected seasons and integrates
    season as a dimension of the output Zarr.

    Parameters
    ----------
    x0 : str
        Path to ground-truth NetCDF file.
    x1 : str
        Path to predictions NetCDF file.
    vars : list[str]
        List of exactly two variables to evaluate.
    output : str
        Directory, file prefix, or final Zarr store path.
    seasons : str | list[str] | None, default None
        Optional season/month standard name(s) (e.g. DJF, MAM).
    align_leap_days : str | None, default None
        Option to align datasets to standard calendar by filling missing leap days.
    force_load : bool, default False
        Option to load data into memory for quick testing of small datasets.
    use_dask : bool, default False
        Option to use Dask for parallel processing.

    Returns
    -------
    None
    """
    if len(vars) != 2:
        raise ValueError(
            "Multi-variate evaluation requires exactly 2 variables in 'vars'."
        )

    var = vars[0]
    var2 = vars[1]

    if use_dask:
        from dask.distributed import Client

        _client = Client(**dask_default_params)
        click.echo("Using dask")

    click.echo(f"Running multi-variate evaluation for variables '{var}', '{var2}'...")
    click.echo(f"Loading reference: {x0}")
    x0_ds = load_dataset(
        x0, align_leap_days=align_leap_days, chunks="auto" if use_dask else None
    )
    click.echo(f"Loading prediction: {x1}")
    x1_ds = load_dataset(
        x1, align_leap_days=align_leap_days, chunks="auto" if use_dask else None
    )

    # Check for different pr units in target vs prediction
    if var == "pr" or var2 == "pr":
        if x1_ds["pr"].max() > 1 and x0_ds["pr"].max() < 1:
            click.echo("WARNING: pr converted from mm/day to kg/m2/s")
            x1_ds["pr"] = x1_ds["pr"] / 86400.0

    x0_ds, x1_ds, member_dim = align_dims(x0_ds, x1_ds)

    # Force loading when testing on smaller datasets so quick to run
    # Works well on 6 years 3 members
    if force_load:
        x0_ds.load()
        x1_ds.load()

    scale, shift = convert_threshold_units(x0_ds, var)

    is_list = isinstance(seasons, list)
    seasons_list: list[str | None] = seasons if is_list else [seasons]  # type: ignore

    season_metrics = []

    for season in seasons_list:
        if is_list:
            click.echo(f"Processing season: {season}")

        m_multivar = compute_multivar_metrics(x0_ds, x1_ds, var, var2, season)

        if is_list:
            season_metrics.append(m_multivar)
        else:
            metrics = m_multivar

    click.echo("Combining and saving metrics...")
    if is_list:
        combined_metrics = _combine_seasons(season_metrics, seasons_list)
    else:
        combined_metrics = metrics

    metrics_path = _get_output_path(output, "metrics_mv_field.zarr")
    write_zarr(combined_metrics, metrics_path, mode="w", var=var + var2)

    click.echo(f"Saved all multi-variate metrics to {metrics_path} etc...")
    click.echo("Multi-variate evaluation complete.")


def run_cdquantiles(
    x0: str,
    var: str,
    output: str,
    align_leap_days: str | None = None,
    force_load: bool = False,
    use_dask: bool = False,
) -> None:
    """
    Run the calendar day extreme threshold computation workflow.

    Computes target 90th and 10th calendar percentiles for use
    in fn cdextremes().

    Parameters
    ----------
    x0 : str
        Path to ground-truth NetCDF file.
    var : str
        Variable name to evaluate.
    output : str
        Directory for netcdf threshold fields.
    align_leap_days : str | None, default None
        Option to align datasets to standard calendar by filling missing leap days.
    force_load : bool, default False
        Option to load data into memory for quick testing of small datasets.
    use_dask : bool, default False
        Option to use Dask for parallel processing.

    Returns
    -------
    None
    """
    if use_dask:
        from dask.distributed import Client

        _client = Client(**dask_default_params)
        click.echo("Using dask")

    click.echo(
        f"Running calendar day extreme threshold computation for variable '{var}'..."
    )
    click.echo(f"Loading reference: {x0}")
    x0_ds = load_dataset(
        x0, align_leap_days=align_leap_days, chunks="auto" if use_dask else None
    )

    x0_ds = x0_ds[[var]]

    # Force loading when testing on smaller datasets so quick to run
    # Works well on 6 years
    if force_load:
        x0_ds.load()

    # Compute Bias of freq gt 90th cal_day percentile: field
    click.echo("Computing Target 90th cal_day percentile: field... SLOW...")

    targ_var90p_quantile = indices.calday_quantile(x0_ds, var=var, q=0.9, padding=2)
    targ_var90p_quantile_path = _get_output_path(output, "targ_quantile_var90p.nc")
    targ_var90p_quantile.to_netcdf(targ_var90p_quantile_path)
    click.echo(f"Saved Target 90th cal_day percentile to {targ_var90p_quantile_path}")

    # Compute time Bias of freq timesteps lt 10th cal_day percentile: field
    click.echo("Computing Target 10th cal_day percentile: field... SLOW...")

    targ_var10p_quantile = indices.calday_quantile(x0_ds, var=var, q=0.1, padding=2)
    targ_var10p_quantile_path = _get_output_path(output, "targ_quantile_var10p.nc")
    targ_var10p_quantile.to_netcdf(targ_var10p_quantile_path)
    click.echo(f"Saved Target 10th cal_day percentile to {targ_var10p_quantile_path}")


def run_cdextremes(
    x0: str,
    x1: str,
    var: str,
    output: str,
    thresh_dir: str,
    seasons: str | list[str] | None = None,
    align_leap_days: str | None = None,
    force_load: bool = False,
    use_dask: bool = False,
) -> None:
    """
    Run the calendar day extremes evaluation workflow.

    Computes bias of frequencies and spell lengths of extremes relative to
    calendar day percentile thresholds.
    If seasons is a list, loops over all selected seasons and integrates
    season as a dimension of the output Zarr.

    Parameters
    ----------
    x0 : str
        Path to ground-truth NetCDF file.
    x1 : str
        Path to predictions NetCDF file.
    var : str
        Variable name to evaluate.
    output : str
        Directory, file prefix, or final Zarr store path.
    thresh_dir : str
        Directory for netcdf threshold files (from fn cdquantiles)
        targ_quantile_var90p.nc, targ_quantile_var10p.nc.
    seasons : str | list[str] | None, default None
        Optional season/month standard name(s) (e.g. DJF, MAM).
    align_leap_days : str | None, default None
        Option to align datasets to standard calendar by filling missing leap days.
    force_load : bool, default False
        Option to load data into memory for quick testing of small datasets.
    use_dask : bool, default False
        Option to use Dask for parallel processing.

    Returns
    -------
    None
    """
    if use_dask:
        from dask.distributed import Client

        _client = Client(**dask_default_params)
        click.echo("Using dask")

    click.echo(
        f"Running evaluation wrt calendar day thresholds for variable '{var}'..."
    )
    click.echo(f"Loading reference: {x0}")
    x0_ds = load_dataset(
        x0, align_leap_days=align_leap_days, chunks="auto" if use_dask else None
    )
    click.echo(f"Loading prediction: {x1}")
    x1_ds = load_dataset(
        x1, align_leap_days=align_leap_days, chunks="auto" if use_dask else None
    )

    x0_ds = x0_ds[[var]]
    x1_ds = x1_ds[[var]]

    # Check for different pr units in target vs prediction
    if var == "pr":
        if x1_ds["pr"].max() > 1 and x0_ds["pr"].max() < 1:
            click.echo("WARNING: pr converted from mm/day to kg/m2/s")
            x1_ds["pr"] = x1_ds["pr"] / 86400.0

    x0_ds, x1_ds, member_dim = align_dims(x0_ds, x1_ds)

    # Force loading when testing on smaller datasets so quick to run
    # Works well on 6 years 3 members
    if force_load:
        x0_ds.load()
        x1_ds.load()

    calday_90p_path = _get_output_path(thresh_dir, "targ_quantile_var90p.nc")
    calday_10p_path = _get_output_path(thresh_dir, "targ_quantile_var10p.nc")

    if not os.path.isfile(calday_90p_path) and not os.path.isfile(calday_10p_path):
        click.echo(
            f"Target threshold files do not exist: {calday_90p_path}; {calday_10p_path}. Use track cdquantiles to compute these for variable {var}."
        )
        raise ValueError(
            f"Target threshold files do not exist: {calday_90p_path}; {calday_10p_path}. Use track cdquantiles to compute these for variable {var}."
        )

    targ_var90p_quantile = None
    if os.path.isfile(calday_90p_path):
        targ_var90p_quantile = load_dataset(
            calday_90p_path, chunks="auto" if use_dask else None
        )
        click.echo(f"Loaded threshold file: {calday_90p_path}")
        if len(targ_var90p_quantile[var].dims) < 3:
            click.echo(
                f"WARNING: expected at least 3 dimensions but got: {targ_var90p_quantile[var].dims}"
            )
    else:
        click.echo(
            f"Target threshold file does not exist: {calday_90p_path}. Use track cdquantiles to compute this for variable {var}."
        )

    targ_var10p_quantile = None
    if os.path.isfile(calday_10p_path):
        targ_var10p_quantile = load_dataset(
            calday_10p_path, chunks="auto" if use_dask else None
        )
        click.echo(f"Loaded threshold file: {calday_10p_path}")
        if len(targ_var10p_quantile[var].dims) < 3:
            click.echo(
                f"WARNING: expected at least 3 dimensions but got: {targ_var10p_quantile[var].dims}"
            )
    else:
        click.echo(
            f"Target threshold file does not exist: {calday_10p_path}. Use track cdquantiles to compute this for variable {var}."
        )

    is_list = isinstance(seasons, list)
    seasons_list: list[str | None] = seasons if is_list else [seasons]  # type: ignore

    season_metrics = []

    for season in seasons_list:
        if is_list:
            click.echo(f"Processing season: {season}")

        m_cdextremes = compute_cdextremes_metrics(
            x0_ds, x1_ds, var, season, targ_var90p_quantile, targ_var10p_quantile
        )

        if is_list:
            season_metrics.append(m_cdextremes)
        else:
            metrics = m_cdextremes

    click.echo("Combining and saving metrics...")
    if is_list:
        combined_metrics = _combine_seasons(season_metrics, seasons_list)
    else:
        combined_metrics = metrics

    metrics_path = _get_output_path(output, "metrics_cdextremes.zarr")
    write_zarr(combined_metrics, metrics_path, mode="w", var=var)

    click.echo(f"Saved all calday percentile metrics to {metrics_path}")
    click.echo("Calday percentile evaluation complete.")


def validate_evaluations(config_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Validate the evaluations configuration.

    Ensures the 'evaluations' key exists, all blocks have a valid type,
    required fields are present, file paths exist, and default values are set.

    Parameters
    ----------
    config_data : dict[str, Any]
        Raw configuration data loaded from YAML.

    Returns
    -------
    list[dict[str, Any]]
        List of validated evaluation blocks.

    Raises
    ------
    ValueError
        If validation fails.
    """
    if "evaluations" not in config_data:
        raise ValueError("Missing top-level 'evaluations' key in configuration.")

    blocks = config_data["evaluations"]
    if not isinstance(blocks, list):
        raise ValueError("'evaluations' must be a list.")

    valid_types = {"fast", "slow", "multivar", "cdquantiles", "cdextremes"}
    validated_blocks = []

    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise ValueError(f"Evaluation block at index {idx} must be a dictionary.")

        if "type" not in block:
            raise ValueError(
                f"Evaluation block at index {idx} is missing the 'type' key."
            )

        eval_type = block["type"]
        if eval_type not in valid_types:
            raise ValueError(
                f"Evaluation block at index {idx} has invalid type '{eval_type}'. "
                f"Must be one of {sorted(list(valid_types))}."
            )

        # Validate required fields
        if eval_type in {"fast", "slow"}:
            required = {"x0", "x1", "var", "output"}
        elif eval_type == "multivar":
            required = {"x0", "x1", "vars", "output"}
        elif eval_type == "cdquantiles":
            required = {"x0", "var", "output"}
        elif eval_type == "cdextremes":
            required = {"x0", "x1", "var", "output", "thresh_dir"}
        else:
            required = set()

        for req in required:
            if req not in block:
                raise ValueError(
                    f"Evaluation block of type '{eval_type}' at index {idx} is missing required field '{req}'."
                )

        # Validate path existence
        for path_field in {"x0", "x1", "thresh_dir"}:
            if path_field in block:
                path_val = block[path_field]
                if not isinstance(path_val, str):
                    raise ValueError(
                        f"Field '{path_field}' at index {idx} must be a string representing a path."
                    )
                if not os.path.exists(path_val):
                    # For thresh_dir, check if it will be generated by a preceding cdquantiles task
                    is_generated = False
                    if path_field == "thresh_dir":
                        for prev_block in blocks[:idx]:
                            if (
                                isinstance(prev_block, dict)
                                and prev_block.get("type") == "cdquantiles"
                                and prev_block.get("output") == path_val
                            ):
                                is_generated = True
                                break
                    if not is_generated:
                        raise ValueError(
                            f"Path '{path_val}' specified in '{path_field}' at index {idx} does not exist."
                        )

        # Validate vars for multivar
        if eval_type == "multivar":
            vars_val = block["vars"]
            if (
                not isinstance(vars_val, list)
                or len(vars_val) != 2
                or not all(isinstance(v, str) for v in vars_val)
            ):
                raise ValueError(
                    f"Evaluation block of type 'multivar' at index {idx} must have 'vars' as a list of exactly 2 strings."
                )

        # Validate seasons if present
        if "seasons" in block:
            seasons_val = block["seasons"]
            if seasons_val is not None:
                if isinstance(seasons_val, list):
                    if not all(isinstance(s, str) for s in seasons_val):
                        raise ValueError(
                            f"Field 'seasons' in block at index {idx} must be a list of strings."
                        )
                elif not isinstance(seasons_val, str):
                    raise ValueError(
                        f"Field 'seasons' in block at index {idx} must be a string or a list of strings."
                    )

        # Build validated block with defaults
        validated = {
            "type": eval_type,
            "x0": block["x0"],
            "output": block["output"],
            "align_leap_days": block.get("align_leap_days", None),
            "force_load": bool(block.get("force_load", False)),
            "use_dask": bool(block.get("use_dask", False)),
        }

        if eval_type in {"fast", "slow", "cdextremes"}:
            validated["var"] = block["var"]
            validated["x1"] = block["x1"]
            validated["seasons"] = block.get("seasons", None)

        if eval_type == "multivar":
            validated["vars"] = block["vars"]
            validated["x1"] = block["x1"]
            validated["seasons"] = block.get("seasons", None)

        if eval_type == "cdquantiles":
            validated["var"] = block["var"]

        if eval_type == "cdextremes":
            validated["thresh_dir"] = block["thresh_dir"]

        validated_blocks.append(validated)

    return validated_blocks


@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the YAML configuration file.",
)
def main(config: str) -> None:
    """
    Run evaluation workflows sequentially from a YAML configuration file.

    Parameters
    ----------
    config : str
        Path to the YAML configuration file.

    Returns
    -------
    None
    """
    click.echo(f"Loading configuration from {config}...")
    with open(config, "r") as f:
        try:
            config_data = yaml.safe_load(f)
        except Exception as e:
            click.echo(f"Error: Failed to parse YAML file: {e}")
            raise click.ClickException(f"Failed to parse YAML file: {e}")

    try:
        evaluations = validate_evaluations(config_data)
    except ValueError as e:
        click.echo(f"Configuration Validation Error: {e}")
        raise click.ClickException(str(e))

    num_evals = len(evaluations)
    click.echo(f"Found {num_evals} evaluation task(s) to run.")

    for idx, eval_block in enumerate(evaluations, 1):
        eval_type = eval_block["type"]
        click.echo("=" * 80)
        click.echo(f"STARTING EVALUATION TASK {idx}/{num_evals}: type='{eval_type}'")
        click.echo("=" * 80)

        try:
            if eval_type == "fast":
                run_fast(
                    x0=eval_block["x0"],
                    x1=eval_block["x1"],
                    var=eval_block["var"],
                    output=eval_block["output"],
                    seasons=eval_block["seasons"],
                    align_leap_days=eval_block["align_leap_days"],
                    force_load=eval_block["force_load"],
                    use_dask=eval_block["use_dask"],
                )
            elif eval_type == "slow":
                run_slow(
                    x0=eval_block["x0"],
                    x1=eval_block["x1"],
                    var=eval_block["var"],
                    output=eval_block["output"],
                    seasons=eval_block["seasons"],
                    align_leap_days=eval_block["align_leap_days"],
                    force_load=eval_block["force_load"],
                    use_dask=eval_block["use_dask"],
                )
            elif eval_type == "multivar":
                run_multivar(
                    x0=eval_block["x0"],
                    x1=eval_block["x1"],
                    vars=eval_block["vars"],
                    output=eval_block["output"],
                    seasons=eval_block["seasons"],
                    align_leap_days=eval_block["align_leap_days"],
                    force_load=eval_block["force_load"],
                    use_dask=eval_block["use_dask"],
                )
            elif eval_type == "cdquantiles":
                run_cdquantiles(
                    x0=eval_block["x0"],
                    var=eval_block["var"],
                    output=eval_block["output"],
                    align_leap_days=eval_block["align_leap_days"],
                    force_load=eval_block["force_load"],
                    use_dask=eval_block["use_dask"],
                )
            elif eval_type == "cdextremes":
                run_cdextremes(
                    x0=eval_block["x0"],
                    x1=eval_block["x1"],
                    var=eval_block["var"],
                    output=eval_block["output"],
                    thresh_dir=eval_block["thresh_dir"],
                    seasons=eval_block["seasons"],
                    align_leap_days=eval_block["align_leap_days"],
                    force_load=eval_block["force_load"],
                    use_dask=eval_block["use_dask"],
                )
        except Exception as e:
            click.echo(f"ERROR executing task {idx}: {e}")
            raise click.ClickException(f"Task {idx} failed: {e}")

        click.echo("=" * 80)
        click.echo(f"FINISHED EVALUATION TASK {idx}/{num_evals}: type='{eval_type}'")
        click.echo("=" * 80)
        click.echo("")

    click.echo("All evaluation tasks completed successfully.")


if __name__ == "__main__":
    main()
