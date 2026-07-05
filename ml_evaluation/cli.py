from __future__ import annotations

import os
import click
from ml_evaluation.evaluation.diagnostics import (
    bias_index,
    rmse_index,
    corr_index,
    psd,
    ralsd,
    calc_pdf_metric_xr,
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

    # Compute Bias (over time) of Mean: field
    click.echo("Computing time Bias of Mean daily data: field...")
    biasf_ds = bias_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=indices.mean,
    )

    # Compute RMSE of daily data (over time): field
    click.echo("Computing RMSE of daily data (over time): field...")
    rmsef_ds = rmse_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        dim="time",
    )

    # Compute RMSE of daily data (over time, lat, lon): float
    click.echo("Computing RMSE of daily data (over time, lat, lon): float..")
    rmse_ds = rmse_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        dim=("time", "lat", "lon"),
    )

    # Compute Pearson Correlation of daily data (over lat, lon): series
    click.echo("Computing Pearson Correlation of daily data (over lat, lon): series...")
    pcorrs_ds = corr_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        dim=("lat", "lon"),
    )

    # Compute Spearman Correlation of daily data (over lat, lon): series
    click.echo(
        "Computing Spearman Correlation of daily data (over lat, lon): series..."
    )
    scorrs_ds = corr_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        dim=("lat", "lon"),
        srank=True,
    )

    # Compute Bias of 99th percentile (over time): field
    click.echo("Computing Bias of 99th percentile (over time): field...")
    biasf_p99_ds = bias_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=indices.season_quantile,
        q=0.99,
    )

    # Compute Bias of 1th percentile (over time): field
    click.echo("Computing Bias of 1th percentile (over time): field...")
    biasf_p1_ds = bias_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=indices.season_quantile,
        q=0.01,
    )

    # Compute Bias of lag-1 day autocorrelation (over time): field
    click.echo("Computing Bias of lag-1 day autocorrelation (over time): field...")
    biasf_acorr_ds = bias_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=indices.lag1_corr,
    )

    # Compute Bias of annual max (over time): field
    # **precip Rx1day fast track; temperature TXx, TNx slow track?
    click.echo("Computing Bias of annual max (over time): field...")
    biasf_max1day_ds = bias_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=indices.ann_extrema,
        maxmin="max",
        cmean=True,
    )

    # Compute Bias of annual min (over time): field
    # **temperature TXn, TNn slow track? Precip not useful
    click.echo("Computing Bias of annual min (over time): field...")
    biasf_min1day_ds = bias_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=indices.ann_extrema,
        maxmin="min",
        cmean=True,
    )

    # Compute Bias of freq ge 1mm/day (over time): field
    # **Precipitation only metric
    click.echo("Computing Bias of freq ge 1mm/day (over time): field...")
    biasf_r1_ds = bias_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=indices.ann_freq_wrt_array,
        threshold=1.0,
        relation="ge",
        cmean=True,
    )

    # Compute Bias of sdii (over time): field
    # **Precipitation only metric
    click.echo("Computing Bias of sdii (over time): field...")
    biasf_sdii_ds = bias_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=indices.ann_sdii,
        wet_threshold=1.0,
        cmean=True,
    )

    # Compute PSD and RALSD
    click.echo("Computing Power Spectral Density (PSD) (over lat, lon)...")
    psd_x0, psd_x1 = psd(x0_ds, x1_ds, var=var, season=season)

    click.echo("Computing Radially Averaged Log Spectral Distance (RALSD)...")
    ralsd_val = ralsd(psd_x0, psd_x1, var)
    click.echo(f"RALSD: {ralsd_val}")

    print(type(psd_x0))
    print(psd_x0)
    print(type(ralsd_val))
    print(ralsd_val)

    # Compute Var90p, Var10p e.g. TX90p
    # **Needs Code adjusted for leap-year calendar optiont**

    # Compute temporal spectral distance
    # **Needs Code**

    # Save all fast-track metrics
    click.echo("Combining and saving metrics...")
    metrics = {
        "bias_mean_field": biasf_ds,
        "rmse_field": rmsef_ds,
        "rmse_float": rmse_ds,
        "pcorr_series": pcorrs_ds,
        "scorr_series": scorrs_ds,
        "bias_p99_field": biasf_p99_ds,
        "bias_p1_field": biasf_p1_ds,
        "bias_acorr_field": biasf_acorr_ds,
        "bias_max1day_field": biasf_max1day_ds,
        "bias_min1day_field": biasf_min1day_ds,
        "bias_r1_field": biasf_r1_ds,
        "bias_sdii_field": biasf_sdii_ds,
        "ralsd": ralsd_val,
    }

    metrics_psd = {
        "psd_x0": psd_x0,
        "psd_x1": psd_x1,
    }

    # tmp: also store as .nc files to check
    for name, ds in metrics.items():
        filepath = os.path.join(output, f"{name}.nc")
        ds.to_netcdf(filepath)
        print(f"Saved {name} -> {filepath}")
    for name, ds in metrics_psd.items():
        filepath = os.path.join(output, f"{name}.nc")
        ds.to_netcdf(filepath)
        print(f"Saved {name} -> {filepath}")

    metrics_path = _get_output_path(output, "metrics_fast.zarr")
    write_zarr(metrics, metrics_path, mode="w", var=var)

    metrics_psd_path = _get_output_path(output, "metrics_psd.zarr")
    write_zarr(metrics_psd, metrics_psd_path, mode="w", var=var)

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

    # Compute lat, lon Pearson Correlation: field
    click.echo("Computing time Pearson Correlation of daily data: series...")
    pcorrf_ds = corr_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        dim=("time"),
    )

    # Compute lat, lon Spearman Correlation: field
    click.echo("Computing time Spearman Correlation of daily data: series...")
    scorrf_ds = corr_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        dim=("time"),
        srank=True,
    )

    # Compute time Bias of std(ann mean): field
    click.echo("Computing time Bias of std of annual means: field...")
    biasf_annstd_ds = bias_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=indices.interannual_var,
    )

    # Compute Bias of 90th percentile (over time): field
    click.echo("Computing Bias of 90th percentile (over time): field...")
    biasf_p90_ds = bias_index(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=indices.season_quantile,
        q=0.90,
    )

    # Compute Wasserstein Distance (over time): field
    click.echo("Computing Wasserstein Distance (over time): field...")
    wdf_ds = calc_pdf_metric_xr(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=None,
        dim=("time",),
        typedistance="wd",
    )

    # Compute Wasserstein Distance hist method (over time): field
    click.echo("Computing Wasserstein Distance hist method (over time): field...")
    wdhf_ds = calc_pdf_metric_xr(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=None,
        dim=("time",),
        bins_arr=None,
        nbins=100,
        bin_target_only=True,
        typedistance="wdh",
    )

    # Compute FCSS (over time): field
    click.echo("Computing FCSS (over time): field...")
    fcssf_ds = calc_pdf_metric_xr(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=None,
        dim=("time",),
        bins_arr=None,
        nbins=100,
        bin_target_only=True,
        typedistance="fcss",
    )

    # Compute PSS (over time): field
    click.echo("Computing PSS (over time): field...")
    pssf_ds = calc_pdf_metric_xr(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=None,
        dim=("time",),
        bins_arr=None,
        nbins=100,
        bin_target_only=True,
        typedistance="pss",
    )

    # Compute LHD (over time): field
    click.echo("Computing LHD (over time): field...")
    lhdf_ds = calc_pdf_metric_xr(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=None,
        dim=("time",),
        bins_arr=None,
        nbins=100,
        bin_target_only=True,
        typedistance="lhd",
    )

    # Compute FCSS (over time, lat, lon): float
    click.echo("Computing FCSS (over time, lat, lon): float...")
    fcss_ds = calc_pdf_metric_xr(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=None,
        dim=("time", "lat", "lon"),
        bins_arr=None,
        nbins=100,
        bin_target_only=False,
        typedistance="fcss",
    )

    # Compute Wasserstein Distance hist method (over time, lat, lon): field
    click.echo(
        "Computing Wasserstein Distance hist method (over time, lat, lon): field..."
    )
    wdhf_ds = calc_pdf_metric_xr(
        x0_ds,
        x1_ds,
        var=var,
        season=season,
        index_fn=None,
        dim=("time", "lat", "lon"),
        bins_arr=None,
        nbins=100,
        bin_target_only=True,
        typedistance="wdh",
    )

    # Consolidate and save all slow metrics
    click.echo("Combining and saving metrics...")
    metrics = {
        "pcorr_field": pcorrf_ds,
        "scorr_field": scorrf_ds,
        "bias_annstd_field": biasf_annstd_ds,
        "bias_p90_field": biasf_p90_ds,
        "wd_field": wdf_ds,
        "wdh_field": wdhf_ds,
        "fcss_field": fcssf_ds,
        "pss_field": pssf_ds,
        "lhd_field": lhdf_ds,
        "fcss_float": fcss_ds,
        "wdh_float": wdhf_ds,
    }

    # tmp: also store as .nc files to check
    for name, ds in metrics.items():
        filepath = os.path.join(output, f"{name}.nc")
        ds.to_netcdf(filepath)
        print(f"Saved {name} -> {filepath}")

    metrics_path = _get_output_path(output, "metrics_slow.zarr")
    write_zarr(metrics, metrics_path, mode="w", var=var)
    click.echo(f"Saved all slow-track metrics to {metrics_path}")

    click.echo("Slow-track evaluation complete.")


if __name__ == "__main__":
    main()
