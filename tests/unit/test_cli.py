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
    output_prefix = str(tmp_path / "eval_out")

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
    assert os.path.exists(f"{output_prefix}_metrics_fast_field.zarr")


def test_cli_slow_workflow(
    temp_nc_files: tuple[str, str], tmp_path: pathlib.Path
) -> None:
    """
    Test the CLI slow-track workflow execution.

    Verifies that running the 'slow' command produces output Zarr stores.

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
    output_prefix = str(tmp_path / "eval_out")

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
    assert os.path.exists(f"{output_prefix}_metrics_slow_field.zarr")


@pytest.fixture
def temp_multivar_nc_files(tmp_path: pathlib.Path) -> tuple[str, str]:
    """
    Generate temporary NetCDF files with two variables for multi-variate CLI tests.

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

    data_v1 = np.random.rand(len(times), len(lats), len(lons))
    data_v2 = np.random.rand(len(times), len(lats), len(lons))

    x0 = xr.Dataset(
        {
            "tasmax": (["time", "lat", "lon"], data_v1),
            "tasmin": (["time", "lat", "lon"], data_v2),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    x1 = xr.Dataset(
        {
            "tasmax": (["time", "lat", "lon"], data_v1 + 0.1),
            "tasmin": (["time", "lat", "lon"], data_v2 + 0.1),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )

    x0_path = tmp_path / "ref_mv.nc"
    x1_path = tmp_path / "pred_mv.nc"

    x0.to_netcdf(x0_path)
    x1.to_netcdf(x1_path)

    return str(x0_path), str(x1_path)


def test_cli_fast_workflow_with_dask(
    temp_nc_files: tuple[str, str], tmp_path: pathlib.Path
) -> None:
    """
    Test the CLI fast-track workflow execution with Dask.

    Parameters
    ----------
    temp_nc_files : tuple[str, str]
        Fixture providing NetCDF reference and prediction files.
    tmp_path : pathlib.Path
        Fixture providing a temporary directory.

    Returns
    -------
    None
    """
    x0, x1 = temp_nc_files
    output_prefix = str(tmp_path / "eval_out_dask")

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
            "--use-dask",
            "True",
        ],
    )

    assert result.exit_code == 0
    assert "Using dask" in result.output
    assert os.path.exists(f"{output_prefix}_metrics_fast_field.zarr")


def test_cli_slow_workflow_with_dask(
    temp_nc_files: tuple[str, str], tmp_path: pathlib.Path
) -> None:
    """
    Test the CLI slow-track workflow execution with Dask.

    Parameters
    ----------
    temp_nc_files : tuple[str, str]
        Fixture providing NetCDF reference and prediction files.
    tmp_path : pathlib.Path
        Fixture providing a temporary directory.

    Returns
    -------
    None
    """
    x0, x1 = temp_nc_files
    output_prefix = str(tmp_path / "eval_out_slow_dask")

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
            "--use-dask",
            "True",
        ],
    )

    assert result.exit_code == 0
    assert "Using dask" in result.output
    assert os.path.exists(f"{output_prefix}_metrics_slow_field.zarr")


def test_cli_multivar_workflow(
    temp_multivar_nc_files: tuple[str, str], tmp_path: pathlib.Path
) -> None:
    """
    Test the CLI multi-variate workflow with and without Dask.

    Parameters
    ----------
    temp_multivar_nc_files : tuple[str, str]
        Fixture providing multi-variable reference and prediction NetCDF files.
    tmp_path : pathlib.Path
        Fixture providing a temporary directory.

    Returns
    -------
    None
    """
    x0, x1 = temp_multivar_nc_files
    output_prefix = str(tmp_path / "eval_out_mv")

    runner = CliRunner()
    # Test without Dask
    result = runner.invoke(
        main,
        [
            "multivar",
            "--x0",
            x0,
            "--x1",
            x1,
            "--var",
            "tasmax",
            "--var2",
            "tasmin",
            "--output",
            output_prefix,
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(f"{output_prefix}_metrics_mv_field.zarr")

    # Test with Dask
    output_prefix_dask = str(tmp_path / "eval_out_mv_dask")
    result_dask = runner.invoke(
        main,
        [
            "multivar",
            "--x0",
            x0,
            "--x1",
            x1,
            "--var",
            "tasmax",
            "--var2",
            "tasmin",
            "--output",
            output_prefix_dask,
            "--use-dask",
            "True",
        ],
    )
    assert result_dask.exit_code == 0
    assert "Using dask" in result_dask.output
    assert os.path.exists(f"{output_prefix_dask}_metrics_mv_field.zarr")


def test_cli_calendar_day_workflow(
    temp_nc_files: tuple[str, str], tmp_path: pathlib.Path
) -> None:
    """
    Test the CLI calendar day workflow (cdquantiles and cdextremes) with and without Dask.

    Parameters
    ----------
    temp_nc_files : tuple[str, str]
        Fixture providing reference and prediction NetCDF files.
    tmp_path : pathlib.Path
        Fixture providing a temporary directory.

    Returns
    -------
    None
    """
    x0, x1 = temp_nc_files
    thresh_dir = str(tmp_path / "thresh")
    os.makedirs(thresh_dir, exist_ok=True)
    output_prefix = str(tmp_path / "eval_out_cd")

    runner = CliRunner()
    # 1. Run cdquantiles with Dask
    result_q = runner.invoke(
        main,
        [
            "cdquantiles",
            "--x0",
            x0,
            "--var",
            "tasmax",
            "--output",
            thresh_dir,
            "--use-dask",
            "True",
        ],
    )
    assert result_q.exit_code == 0
    assert "Using dask" in result_q.output
    assert os.path.exists(os.path.join(thresh_dir, "targ_quantile_var90p.nc"))
    assert os.path.exists(os.path.join(thresh_dir, "targ_quantile_var10p.nc"))

    # 2. Run cdextremes with Dask
    result_e = runner.invoke(
        main,
        [
            "cdextremes",
            "--x0",
            x0,
            "--x1",
            x1,
            "--var",
            "tasmax",
            "--output",
            output_prefix,
            "--thresh-dir",
            thresh_dir,
            "--use-dask",
            "True",
        ],
    )
    assert result_e.exit_code == 0
    assert "Using dask" in result_e.output
    assert os.path.exists(f"{output_prefix}_metrics_cdextremes.zarr")
