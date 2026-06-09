from __future__ import annotations

import os
import click
import xarray as xr
from ml_evaluation.evaluation.diagnostics_plus import (
    mean_bias_index,
    error_ann_index,
    pearsoncorr_ann_index,
    wasserstein_distance,
    psd,
    ralsd,
)


def _get_output_path(base: str, suffix: str) -> str:
    """
    Construct the output file path based on a base path and a suffix.

    If the base path is an existing directory, the suffix is appended as a filename.
    Otherwise, the base path is treated as a file prefix, and the suffix is appended.

    Parameters
    ----------
    base : str
        The base output directory or file prefix.
    suffix : str
        The suffix to append (e.g. 'mean_bias.nc').

    Returns
    -------
    str
        The constructed file path.
    """
    if os.path.isdir(base):
        return os.path.join(base, suffix)
    return f"{base}_{suffix}"


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
    help="Directory or file prefix for output NetCDF results.",
)
@click.option(
    "--season",
    type=str,
    default=None,
    help="Optional season/month to filter (e.g., DJF, MAM, JJA, SON, JAN).",
)
def fast(x0: str, x1: str, var: str, output: str, season: str | None) -> None:
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
        Directory or file prefix for output NetCDF results.
    season : str | None
        Optional season or month standard name.

    Returns
    -------
    None
    """
    click.echo(f"Running fast-track evaluation for variable '{var}'...")
    click.echo(f"Loading reference: {x0}")
    x0_ds = xr.open_dataset(x0)
    click.echo(f"Loading prediction: {x1}")
    x1_ds = xr.open_dataset(x1)

    # Align calendars to 365 day calendar
    # - this will not alter derived quantities such as x.time.dt.dayofyear
    x0_ds = x0_ds.sel(time=~((x0_ds.time.dt.month == 2) & (x0_ds.time.dt.day == 29)))
    x1_ds = x1_ds.sel(time=~((x1_ds.time.dt.month == 2) & (x1_ds.time.dt.day == 29)))

    # Compute Mean Bias
    click.echo("Computing Mean Bias...")
    mb = mean_bias_index(x0_ds, x1_ds, var=var, season=season)
    mb_path = _get_output_path(output, "mean_bias.nc")
    mb.to_netcdf(mb_path)
    click.echo(f"Saved Mean Bias to {mb_path}")

    # Compute RMSE
    click.echo("Computing RMSE...")
    rmse_ds = error_ann_index(x0_ds, x1_ds, var=var, typeerror="rmse", season=season)
    rmse_path = _get_output_path(output, "rmse.nc")
    rmse_ds.to_netcdf(rmse_path)
    click.echo(f"Saved RMSE to {rmse_path}")

    # Compute Pearson Correlation
    click.echo("Computing Pearson Correlation...")
    corr = pearsoncorr_ann_index(x0_ds, x1_ds, var=var, season=season)
    corr_path = _get_output_path(output, "correlation.nc")
    corr.to_netcdf(corr_path)
    click.echo(f"Saved Pearson Correlation to {corr_path}")

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
    help="Directory or file prefix for output NetCDF results.",
)
@click.option(
    "--season",
    type=str,
    default=None,
    help="Optional season/month to filter (e.g., DJF, MAM, JJA, SON, JAN).",
)
def slow(x0: str, x1: str, var: str, output: str, season: str | None) -> None:
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
        Directory or file prefix for output NetCDF results.
    season : str | None
        Optional season or month standard name.

    Returns
    -------
    None
    """
    click.echo(f"Running slow-track evaluation for variable '{var}'...")
    click.echo(f"Loading reference: {x0}")
    x0_ds = xr.open_dataset(x0)
    click.echo(f"Loading prediction: {x1}")
    x1_ds = xr.open_dataset(x1)

    # Align calendars to 365 day calendar
    # - this will not alter derived quantities such as x.time.dt.dayofyear
    x0_ds = x0_ds.sel(time=~((x0_ds.time.dt.month == 2) & (x0_ds.time.dt.day == 29)))
    x1_ds = x1_ds.sel(time=~((x1_ds.time.dt.month == 2) & (x1_ds.time.dt.day == 29)))

    # Compute PSD and RALSD
    click.echo("Computing Power Spectral Density (PSD)...")
    psd_x0, psd_x1 = psd(x0_ds, x1_ds, var=var, season=season)

    psd_x0_path = _get_output_path(output, "psd_x0.nc")
    psd_x1_path = _get_output_path(output, "psd_x1.nc")
    psd_x0.to_netcdf(psd_x0_path)
    psd_x1.to_netcdf(psd_x1_path)
    click.echo(f"Saved PSD for reference to {psd_x0_path}")
    click.echo(f"Saved PSD for prediction to {psd_x1_path}")

    click.echo("Computing Radially Averaged Log Spectral Distance (RALSD)...")
    ralsd_val = ralsd(psd_x0, psd_x1)
    click.echo(f"RALSD: {ralsd_val}")

    # Save RALSD value to a tiny text file
    ralsd_path = _get_output_path(output, "ralsd.txt")
    with open(ralsd_path, "w") as f:
        f.write(f"{ralsd_val}\n")
    click.echo(f"Saved RALSD metric to {ralsd_path}")

    # Compute Spatial Wasserstein Distance
    click.echo("Computing Spatial Wasserstein Distance (this may take a while)...")
    wd = wasserstein_distance(x0_ds, x1_ds, var=var, season=season, spatial=True)
    if isinstance(wd, xr.Dataset) or isinstance(wd, xr.DataArray):
        wd_path = _get_output_path(output, "wasserstein_distance.nc")
        wd.to_netcdf(wd_path)
        click.echo(f"Saved Spatial Wasserstein Distance to {wd_path}")
    else:
        click.echo(f"Wasserstein distance returned scalar: {wd}")
        wd_path = _get_output_path(output, "wasserstein_distance.txt")
        with open(wd_path, "w") as f:
            f.write(f"{wd}\n")
        click.echo(f"Saved Wasserstein Distance metric to {wd_path}")

    click.echo("Slow-track evaluation complete.")


if __name__ == "__main__":
    main()
