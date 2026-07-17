from __future__ import annotations

import os
import pathlib
from typing import Any
import numpy as np
import pandas as pd
import xarray as xr
import pytest
import yaml
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


def _run_with_config(tmp_path: pathlib.Path, config_dict: dict[str, Any]) -> Any:
    """
    Helper to write configuration dictionary to a YAML file and run the CLI.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory path.
    config_dict : dict[str, Any]
        Configuration dictionary to write.

    Returns
    -------
    click.testing.Result
        Result of the Click runner invocation.
    """
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)
    runner = CliRunner()
    return runner.invoke(main, ["--config", str(config_file)])


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
        "Run evaluation workflows sequentially from a YAML configuration file"
        in result.output
    )


def test_cli_fast_workflow(
    temp_nc_files: tuple[str, str], tmp_path: pathlib.Path
) -> None:
    """
    Test the CLI fast-track workflow execution.

    Verifies that running the 'fast' command via config produces output Zarr stores.

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

    config = {
        "evaluations": [
            {
                "type": "fast",
                "x0": x0,
                "x1": x1,
                "var": "tasmax",
                "output": output_prefix,
            }
        ]
    }

    result = _run_with_config(tmp_path, config)
    assert result.exit_code == 0
    assert "Running fast-track evaluation" in result.output

    # Check if files were created
    assert os.path.exists(f"{output_prefix}_metrics_fast_field.zarr")


def test_cli_slow_workflow(
    temp_nc_files: tuple[str, str], tmp_path: pathlib.Path
) -> None:
    """
    Test the CLI slow-track workflow execution.

    Verifies that running the 'slow' command via config produces spectral and distance files.

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

    config = {
        "evaluations": [
            {
                "type": "slow",
                "x0": x0,
                "x1": x1,
                "var": "tasmax",
                "output": output_prefix,
            }
        ]
    }

    result = _run_with_config(tmp_path, config)
    assert result.exit_code == 0
    assert "Running slow-track evaluation" in result.output

    # Check if files were created
    assert os.path.exists(f"{output_prefix}_metrics_slow_field.zarr")

    # Load and verify
    ds = xr.open_zarr(f"{output_prefix}_metrics_slow_field.zarr")
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
    }


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

    config = {
        "evaluations": [
            {
                "type": "fast",
                "x0": x0,
                "x1": x1,
                "var": "tasmax",
                "output": output_prefix,
                "use_dask": True,
            }
        ]
    }

    result = _run_with_config(tmp_path, config)
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

    config = {
        "evaluations": [
            {
                "type": "slow",
                "x0": x0,
                "x1": x1,
                "var": "tasmax",
                "output": output_prefix,
                "use_dask": True,
            }
        ]
    }

    result = _run_with_config(tmp_path, config)
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

    # Test without Dask
    config_no_dask = {
        "evaluations": [
            {
                "type": "multivar",
                "x0": x0,
                "x1": x1,
                "vars": ["tasmax", "tasmin"],
                "output": output_prefix,
            }
        ]
    }
    result = _run_with_config(tmp_path, config_no_dask)
    assert result.exit_code == 0
    assert os.path.exists(f"{output_prefix}_metrics_mv_field.zarr")

    # Test with Dask
    output_prefix_dask = str(tmp_path / "eval_out_mv_dask")
    config_dask = {
        "evaluations": [
            {
                "type": "multivar",
                "x0": x0,
                "x1": x1,
                "vars": ["tasmax", "tasmin"],
                "output": output_prefix_dask,
                "use_dask": True,
            }
        ]
    }
    result_dask = _run_with_config(tmp_path, config_dask)
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

    # 1. Run cdquantiles with Dask
    config_q = {
        "evaluations": [
            {
                "type": "cdquantiles",
                "x0": x0,
                "var": "tasmax",
                "output": thresh_dir,
                "use_dask": True,
            }
        ]
    }
    result_q = _run_with_config(tmp_path, config_q)
    assert result_q.exit_code == 0
    assert "Using dask" in result_q.output
    assert os.path.exists(os.path.join(thresh_dir, "targ_quantile_var90p.nc"))
    assert os.path.exists(os.path.join(thresh_dir, "targ_quantile_var10p.nc"))

    # 2. Run cdextremes with Dask
    config_e = {
        "evaluations": [
            {
                "type": "cdextremes",
                "x0": x0,
                "x1": x1,
                "var": "tasmax",
                "output": output_prefix,
                "thresh_dir": thresh_dir,
                "use_dask": True,
            }
        ]
    }
    result_e = _run_with_config(tmp_path, config_e)
    assert result_e.exit_code == 0
    assert "Using dask" in result_e.output
    assert os.path.exists(f"{output_prefix}_metrics_cdextremes.zarr")


def test_cli_seasons_list_workflow(tmp_path: pathlib.Path) -> None:
    """
    Test that supplying a list of seasons loops and includes season in output Zarr dimensions.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.

    Returns
    -------
    None
    """
    # Create mock dataset with dates in Jan and Feb to support "JAN" and "FEB" filtering
    times = pd.to_datetime(
        ["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02", "2026-02-03"]
    )
    lats = [45.0, 46.0]
    lons = [10.0, 11.0]

    data_x0 = np.array(
        [
            [[295.0, 296.0], [297.0, 298.0]],
            [[301.0, 302.0], [303.0, 304.0]],
            [[298.0, 299.0], [300.0, 301.0]],
            [[305.0, 306.0], [307.0, 308.0]],
            [[296.0, 297.0], [298.0, 299.0]],
        ]
    )
    data_x1 = data_x0 + 0.5

    x0 = xr.Dataset(
        {"tasmax": (["time", "lat", "lon"], data_x0)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    x1 = xr.Dataset(
        {"tasmax": (["time", "lat", "lon"], data_x1)},
        coords={"time": times, "lat": lats, "lon": lons},
    )

    x0_path = str(tmp_path / "ref_seasons.nc")
    x1_path = str(tmp_path / "pred_seasons.nc")
    x0.to_netcdf(x0_path)
    x1.to_netcdf(x1_path)

    output_prefix = str(tmp_path / "eval_out_seasons")

    config = {
        "evaluations": [
            {
                "type": "fast",
                "x0": x0_path,
                "x1": x1_path,
                "var": "tasmax",
                "output": output_prefix,
                "seasons": ["JAN", "FEB"],
            }
        ]
    }

    result = _run_with_config(tmp_path, config)
    assert result.exit_code == 0
    assert "Running fast-track evaluation" in result.output
    assert "Processing season: JAN" in result.output
    assert "Processing season: FEB" in result.output

    # Verify that the output Zarr file exists and has 'season' dimension
    metrics_path = f"{output_prefix}_metrics_fast_field.zarr"
    assert os.path.exists(metrics_path)
    ds = xr.open_zarr(metrics_path)
    assert "season" in ds.dims
    assert "season" in ds.coords
    np.testing.assert_array_equal(ds["season"].values, ["JAN", "FEB"])
