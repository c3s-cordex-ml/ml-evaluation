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


@pytest.fixture
def temp_signal_nc_files(tmp_path: pathlib.Path) -> dict[str, str]:
    """
    Generate historical/future reference and prediction NetCDF files.

    Two disjoint 2-year daily windows (a "historical" and a "future" period) for
    a reference (RCM-truth) and a warmer, wetter model prediction, for both
    tasmax and pr. The prediction future warming exceeds the reference warming so
    the signal bias is non-zero. Returns a mapping of the four role paths.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.

    Returns
    -------
    dict[str, str]
        Paths keyed by ``x0_hist``, ``x0_fut``, ``x1_hist``, ``x1_fut``.
    """
    rng = np.random.default_rng(0)
    lats = [45.0, 46.0]
    lons = [10.0, 11.0]

    def make(start: str, n: int, warm: float, wet: float) -> xr.Dataset:
        times = pd.date_range(start, periods=n, freq="D")
        shape = (n, len(lats), len(lons))
        tasmax = 295.0 + warm + rng.normal(0.0, 3.0, shape)
        pr = np.abs(rng.gamma(1.0, wet, shape))
        return xr.Dataset(
            {
                "tasmax": (["time", "lat", "lon"], tasmax),
                "pr": (["time", "lat", "lon"], pr),
            },
            coords={"time": times, "lat": lats, "lon": lons},
        )

    n = 730  # two years of daily data per window
    files = {
        "x0_hist": make("1981-01-01", n, warm=0.0, wet=2.0),
        "x0_fut": make("2041-01-01", n, warm=2.0, wet=2.4),
        "x1_hist": make("1981-01-01", n, warm=0.3, wet=2.1),
        "x1_fut": make("2041-01-01", n, warm=3.0, wet=2.9),
    }
    paths = {}
    for role, ds in files.items():
        p = tmp_path / f"{role}.nc"
        ds.to_netcdf(p)
        paths[role] = str(p)
    return paths


@pytest.mark.parametrize(
    "var,keys",
    [
        ("tasmax", ["mean", "p01", "p99"]),
        ("pr", ["r01", "sdii", "rx1day"]),
    ],
)
def test_cli_signal_workflow(
    temp_signal_nc_files: dict[str, str],
    tmp_path: pathlib.Path,
    var: str,
    keys: list[str],
) -> None:
    """
    Test the CLI climate-change-signal workflow.

    Verifies the four signal Zarr stores are written with the expected
    diagnostic names and axes for both a temperature and a precip variable.

    Parameters
    ----------
    temp_signal_nc_files : dict[str, str]
        Fixture providing the four role NetCDF paths.
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.
    var : str
        Variable under test.
    keys : list[str]
        Expected index keys for this variable.

    Returns
    -------
    None
    """
    p = temp_signal_nc_files
    output_prefix = str(tmp_path / f"sig_{var}")

    # The command is source-agnostic: one hist/fut pair for a single source.
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "signal",
            "--hist", p["x0_hist"],
            "--fut", p["x0_fut"],
            "--var", var,
            "--output", output_prefix,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Running climate-change-signal workflow" in result.output

    for suffix, axis in [
        ("field", "lat"),
        ("series", "year"),
        ("month", "month"),
        ("float", None),
    ]:
        store = f"{output_prefix}_metrics_signal_{suffix}.zarr"
        assert os.path.exists(store), store
        ds = xr.open_zarr(store)
        assert "value" in ds
        if axis is not None:
            assert axis in ds["value"].dims

    # Field store carries delta + reldelta per index -- and no cross-source bias.
    field = xr.open_zarr(f"{output_prefix}_metrics_signal_field.zarr")
    diagnostics = set(field["diagnostic"].values)
    for key in keys:
        assert f"delta_{key}_field" in diagnostics
        assert f"reldelta_{key}_field" in diagnostics
    assert not any("signalbias" in d for d in diagnostics)

    # Month store spans all 12 months and carries the same delta/reldelta signal.
    month = xr.open_zarr(f"{output_prefix}_metrics_signal_month.zarr")
    assert month.sizes["month"] == 12
    month_diags = set(month["diagnostic"].values)
    for key in keys:
        assert f"delta_{key}_month" in month_diags
        assert f"reldelta_{key}_month" in month_diags


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
