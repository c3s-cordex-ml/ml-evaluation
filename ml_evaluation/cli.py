from __future__ import annotations

import os
import click
from ml_evaluation.evaluation.diagnostics import (
    bias_index,
    rmse_index,
    wasserstein_distance,
    psd,
    ralsd,
)
import ml_evaluation.evaluation.indices as indices
from ml_evaluation.io import load_dataset, write_zarr


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
    if os.path.isdir(base):
        return os.path.join(base, default_name)
    if base.endswith(".zarr"):
        return base
    return f"{base}_{default_name}"


@click.group()
def main() -> None:
    """
    Evaluation CLI for Machine Learning Regional Climate Models.

    Provides commands to run fast-track and slow-track evaluation workflows
    to compare downscaled climate predictions against reference data.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pass


@main.command()
@click.option(
    "--x0",
    required=True,
    type=click.Path(exists=True),
    help="Path to ground-truth NetCDF file.",
)
@click.option(
    "--x1",
    required=True,
    type=click.Path(exists=True),
    help="Path to predictions NetCDF file.",
)
@click.option("--var", required=True, type=str, help="Variable name to evaluate.")
@click.option(
    "--output",
    required=True,
    type=str,
    help="Directory or file path for the output Zarr results.",
)
@click.option(
    "--season",
    type=str,
    default=None,
    help="Optional season/month to filter (e.g., DJF, MAM, JJA, SON, JAN).",
)
@click.option(
    "--align-leap-days",
    type=bool,
    default=True,
    help="Option to align datasets to standard calendar by filling missing leap days.",
)
def fast(
    x0: str,
    x1: str,
    var: str,
    output: str,
    season: str | None,
    align_leap_days: bool,
) -> None:
    """
    Run the fast-track evaluation workflow.

    Computes Mean Bias, RMSE, and Pearson Correlation between x0 and x1
    for the specified variable and optionally filters by season.

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
    season : str | None
        Optional season or month standard name.
    align_leap_days : bool
        Option to align datasets to standard calendar by filling missing leap days.

    Returns
    -------
    None
    """
    click.echo(f"Running fast-track evaluation for variable '{var}'...")
    click.echo(f"Loading reference: {x0}")
    x0_ds = load_dataset(x0, align_leap_days=align_leap_days)
    click.echo(f"Loading prediction: {x1}")
    x1_ds = load_dataset(x1, align_leap_days=align_leap_days)

    # Compute Bias of Mean
    click.echo("Computing Bias of Mean...")
    bias_ds = bias_index(x0_ds, x1_ds, var=var, season=season, index_fn=indices.mean)

    # Compute Bias of 98th percentile
    click.echo("Computing Bias of 98%tile...")
    bias98p_ds = bias_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=indices.season_quantile,
        q=0.98,
    )

    # Compute Bias of Annual max
    click.echo("Computing Bias of Ann Max...")
    biasannmax_ds = bias_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=indices.ann_extrema,
        maxmin="max",
        cmean=True,
    )

    # Compute Bias of Annual min
    click.echo("Computing Bias of Ann Min...")
    biasannmin_ds = bias_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=indices.ann_extrema,
        maxmin="min",
        cmean=True,
    )

    # Compute Bias of Annual Freq > 35
    click.echo("Computing Bias of Ann Freq>35...")
    biasanngt35_ds = bias_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=indices.ann_freq_wrt_array,
        threshold=35.0,
        relation="gt",
        cmean=True,
    )

    # Compute Bias of Annual Spell < 1
    click.echo("Computing Bias of Ann Spell<1...")
    biasannsplt1_ds = bias_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=indices.ann_maxspell_wrt_array,
        threshold=1.0,
        relation="lt",
        cmean=True,
    )

    # Compute RMSE Map
    click.echo("Computing RMSE Field...")
    rmsef_ds = rmse_index(x0_ds, x1_ds, var=var, season=season, dim="time")

    # Compute RMSE
    click.echo("Computing RMSE All Dims...")
    rmse_ds = rmse_index(x0_ds, x1_ds, var=var, season=season)

    # Save all fast-track metrics
    click.echo("Combining and saving metrics...")
    metrics = {
        "mean_bias": bias_ds,
        "98ptile_bias": bias98p_ds,
        "annmax_bias": biasannmax_ds,
        "annmin_bias": biasannmin_ds,
        "anngt35_bias": biasanngt35_ds,
        "annsplt1_bias": biasannsplt1_ds,
        "rmsefld": rmsef_ds,
        "rmse": rmse_ds,
    }

    metrics_path = _get_output_path(output, "metrics.zarr")
    write_zarr(metrics, metrics_path, mode="w", var=var)
    click.echo(f"Saved all fast-track metrics to {metrics_path}")

    click.echo("Fast-track evaluation complete.")


@main.command()
@click.option(
    "--x0",
    required=True,
    type=click.Path(exists=True),
    help="Path to ground-truth NetCDF file.",
)
@click.option(
    "--x1",
    required=True,
    type=click.Path(exists=True),
    help="Path to predictions NetCDF file.",
)
@click.option("--var", required=True, type=str, help="Variable name to evaluate.")
@click.option(
    "--output",
    required=True,
    type=str,
    help="Directory or file path for the output Zarr results.",
)
@click.option(
    "--season",
    type=str,
    default=None,
    help="Optional season/month to filter (e.g., DJF, MAM, JJA, SON, JAN).",
)
@click.option(
    "--align-leap-days",
    type=bool,
    default=True,
    help="Option to align datasets to standard calendar by filling missing leap days.",
)
def slow(
    x0: str,
    x1: str,
    var: str,
    output: str,
    season: str | None,
    align_leap_days: bool,
) -> None:
    """
    Run the slow-track evaluation workflow.

    Computes Power Spectral Density (PSD), Radially Averaged Log Spectral Distance (RALSD),
    and Spatial Wasserstein Distance.

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
    season : str | None
        Optional season or month standard name.
    align_leap_days : bool
        Option to align datasets to standard calendar by filling missing leap days.

    Returns
    -------
    None
    """
    click.echo(f"Running slow-track evaluation for variable '{var}'...")
    click.echo(f"Loading reference: {x0}")
    x0_ds = load_dataset(x0, align_leap_days=align_leap_days)
    click.echo(f"Loading prediction: {x1}")
    x1_ds = load_dataset(x1, align_leap_days=align_leap_days)

    # Compute PSD and RALSD
    click.echo("Computing Power Spectral Density (PSD)...")
    psd_x0, psd_x1 = psd(x0_ds, x1_ds, var=var, season=season)

    click.echo("Computing Radially Averaged Log Spectral Distance (RALSD)...")
    ralsd_val = ralsd(psd_x0, psd_x1)
    click.echo(f"RALSD: {ralsd_val}")

    # Compute Spatial Wasserstein Distance
    click.echo("Computing Spatial Wasserstein Distance (this may take a while)...")
    wd = wasserstein_distance(x0_ds, x1_ds, var=var, season=season, spatial=True)

    # Consolidate and save all slow metrics
    click.echo("Combining and saving metrics...")
    metrics = {
        "psd_x0": psd_x0,
        "psd_x1": psd_x1,
        "ralsd": ralsd_val,
        "wasserstein_distance": wd,
    }

    metrics_path = _get_output_path(output, "metrics.zarr")
    write_zarr(metrics, metrics_path, mode="w", var=var)
    click.echo(f"Saved all slow-track metrics to {metrics_path}")

    click.echo("Slow-track evaluation complete.")


if __name__ == "__main__":
    main()
