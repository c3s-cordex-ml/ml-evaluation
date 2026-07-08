from __future__ import annotations

import os
import pathlib
import numpy as np
import pandas as pd
import xarray as xr
import pytest
from click.testing import CliRunner
from ml_evaluation.cli import main


@pytest.fixture
def temp_nc_files(tmp_path: pathlib.Path) -> tuple[str, str]:
    """
    Generate temporary NetCDF reference and prediction files for CLI tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.

    Returns
    -------
    tuple[str, str]
        Paths to reference and prediction NetCDF files.
    """
    times = pd.date_range("2026-01-01", periods=5, freq="D")
    lats = [45.0, 46.0]
    lons = [10.0, 11.0]

    # Generate reference data
    data_x0 = np.array(
        [
            [[295.0, 296.0], [297.0, 298.0]],
            [[301.0, 302.0], [303.0, 304.0]],
            [[298.0, 299.0], [300.0, 301.0]],
            [[305.0, 306.0], [307.0, 308.0]],
            [[296.0, 297.0], [298.0, 299.0]],
        ]
    )

    # Generate prediction data
    data_x1 = data_x0 + 0.5

    x0 = xr.Dataset(
        {"tasmax": (["time", "lat", "lon"], data_x0)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    x1 = xr.Dataset(
        {"tasmax": (["time", "lat", "lon"], data_x1)},
        coords={"time": times, "lat": lats, "lon": lons},
    )

    x0_path = tmp_path / "ref.nc"
    x1_path = tmp_path / "pred.nc"

    x0.to_netcdf(x0_path)
    x1.to_netcdf(x1_path)

    return str(x0_path), str(x1_path)


def test_cli_help() -> None:
    """
    Test that the CLI helper information prints correctly.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert (
        "Evaluation CLI for Machine Learning Regional Climate Models" in result.output
    )


def test_cli_fast_workflow(
    temp_nc_files: tuple[str, str], tmp_path: pathlib.Path
) -> None:
    """
    Test the CLI fast-track workflow execution.

    Verifies that running the 'fast' command produces output Zarr stores.

    Parameters
    ----------
    temp_nc_files : tuple[str, str]
        Fixture providing paths to reference and prediction NetCDF files.
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.

    Returns
    -------
    None
    """
    x0, x1 = temp_nc_files
    output_prefix = str(tmp_path / "eval_out_")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "fast",
            "--x0",
            x0,
            "--x1",
            x1,
            "--var",
            "tasmax",
            "--output",
            output_prefix,
        ],
    )

    assert result.exit_code == 0
    assert "Running fast-track evaluation" in result.output

    # Check if files were created
    assert os.path.exists(f"{output_prefix}metrics_fast.zarr")

    # Load and verify one of them
    ds = xr.open_zarr(f"{output_prefix}metrics_fast.zarr")
    assert "value" in ds
    assert (
        float(ds["value"].sel(diagnostic="bias_mean_field", lat=45.0, lon=10.0)) == 0.5
    )


def test_cli_slow_workflow(
    temp_nc_files: tuple[str, str], tmp_path: pathlib.Path
) -> None:
    """
    Test the CLI slow-track workflow execution.

    Verifies that running the 'slow' command produces spectral and distance files.

    Parameters
    ----------
    temp_nc_files : tuple[str, str]
        Fixture providing paths to reference and prediction NetCDF files.
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.

    Returns
    -------
    None
    """
    x0, x1 = temp_nc_files
    output_prefix = str(tmp_path / "eval_out_")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "slow",
            "--x0",
            x0,
            "--x1",
            x1,
            "--var",
            "tasmax",
            "--output",
            output_prefix,
        ],
    )

    assert result.exit_code == 0
    assert "Running slow-track evaluation" in result.output

    # Check if files were created
    assert os.path.exists(f"{output_prefix}_metrics_slow.zarr")

    # Load and verify
    ds = xr.open_zarr(f"{output_prefix}_metrics_slow.zarr")
    assert "value" in ds
    assert set(ds["diagnostic"].values) == {
        "pcorr_field",
        "scorr_field",
        "bias_annstd_field",
        "bias_p90_field",
        "wd_field",
        "wdh_field",
        "fcss_field",
        "pss_field",
        "lhd_field",
        "fcss_float",
        "wdh_float",
    }
