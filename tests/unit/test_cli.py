from __future__ import annotations

import json
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


@pytest.mark.parametrize(
    ("var", "expected_metrics"),
    [
        ("pr", {"rmse", "bias_mean", "ralsd", "bias_sdii", "bias_max1day", "lhd"}),
        (
            "tasmax",
            {"rmse", "bias_mean", "ralsd", "bias_max1day", "pss", "bias_annstd"},
        ),
    ],
)
def test_cli_benchmark_workflow(
    var: str, expected_metrics: set[str], tmp_path: pathlib.Path
) -> None:
    rng = np.random.default_rng(3)
    times = pd.date_range("2001-01-01", periods=730, freq="D")
    coords = {"time": times, "lat": [45.0, 46.0], "lon": [10.0, 11.0]}
    if var == "pr":
        target = rng.gamma(1.5, 2.0, size=(730, 2, 2))
        units = "mm/day"
    else:
        target = 295.0 + rng.normal(0.0, 3.0, size=(730, 2, 2))
        units = "K"

    reference = xr.Dataset(
        {var: (["time", "lat", "lon"], target, {"units": units})},
        coords=coords,
    )
    prediction_values = target * 1.1 + 0.1 if var == "pr" else target + 0.5
    prediction = xr.Dataset(
        {var: (["time", "lat", "lon"], prediction_values, {"units": units})},
        coords=coords,
    )
    x0_path = tmp_path / f"{var}_ref.nc"
    x1_path = tmp_path / f"{var}_pred.nc"
    reference.to_netcdf(x0_path)
    prediction.to_netcdf(x1_path)
    output_prefix = str(tmp_path / f"benchmark_{var}")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "--x0",
            str(x0_path),
            "--x1",
            str(x1_path),
            "--var",
            var,
            "--output",
            output_prefix,
        ],
    )

    assert result.exit_code == 0, result.output
    json_path = f"{output_prefix}_metrics_benchmark.json"
    assert os.path.exists(json_path)
    with open(json_path) as f:
        metrics = json.load(f)
    assert metrics.pop("attrs") == {"n_members": 1, "members": None}
    assert set(metrics) == expected_metrics
    assert all(v is not None and np.isfinite(v) for v in metrics.values())


@pytest.mark.parametrize("max_members", [1, 2, 5])
def test_cli_benchmark_ensemble(max_members: int, tmp_path: pathlib.Path) -> None:
    """A random subset of members gives scalars and is recorded."""
    rng = np.random.default_rng(4)
    n_members = 3
    times = pd.date_range("2001-01-01", periods=730, freq="D")
    coords = {"time": times, "lat": [45.0, 46.0], "lon": [10.0, 11.0]}
    target = 295.0 + rng.normal(0.0, 3.0, size=(730, 2, 2))
    members = target + rng.normal(0.5, 1.0, size=(n_members, 730, 2, 2))
    reference = xr.Dataset(
        {"tasmax": (["time", "lat", "lon"], target, {"units": "K"})}, coords=coords
    )
    prediction = xr.Dataset(
        {"tasmax": (["member", "time", "lat", "lon"], members, {"units": "K"})},
        coords=coords,
    )
    x0_path = tmp_path / "tasmax_ref.nc"
    x1_path = tmp_path / "tasmax_pred.nc"
    reference.to_netcdf(x0_path)
    prediction.to_netcdf(x1_path)
    output_prefix = str(tmp_path / "benchmark_tasmax")

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "--x0",
            str(x0_path),
            "--x1",
            str(x1_path),
            "--var",
            "tasmax",
            "--output",
            output_prefix,
            "--max-members",
            str(max_members),
        ],
    )

    assert result.exit_code == 0, result.output
    with open(f"{output_prefix}_metrics_benchmark.json") as f:
        metrics = json.load(f)
    attrs = metrics.pop("attrs")
    assert set(attrs) == {"n_members", "members"}
    assert attrs["n_members"] == n_members
    assert len(attrs["members"]) == min(max_members, n_members)
    assert all(0 <= m < n_members for m in attrs["members"])
    assert set(metrics) == {
        "rmse",
        "bias_mean",
        "ralsd",
        "bias_max1day",
        "pss",
        "bias_annstd",
    }
    assert all(v is not None and np.isfinite(v) for v in metrics.values())


def _run_benchmark_members(
    tmp_path: pathlib.Path,
    members_offset_K: list[float],
    extra_args: list[str] | None = None,
    noise_K: float = 1.0,
) -> dict:
    """
    Run ``benchmark`` on a synthetic tasmax ensemble and return the JSON.

    Each member is the target plus a constant offset (K) plus Gaussian noise.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the input/output files.
    members_offset_K : list[float]
        Constant offset (K) of each member with respect to the target.
    extra_args : list[str] | None, optional
        Additional CLI arguments.
    noise_K : float, optional
        Standard deviation (K) of the member noise.

    Returns
    -------
    dict
        The metrics JSON content.
    """
    rng = np.random.default_rng(7)
    times = pd.date_range("2001-01-01", periods=730, freq="D")
    coords = {"time": times, "lat": [45.0, 46.0], "lon": [10.0, 11.0]}
    target = 295.0 + rng.normal(0.0, 3.0, size=(730, 2, 2))
    offsets = np.asarray(members_offset_K)[:, None, None, None]
    members = target + offsets
    if noise_K > 0:
        members = members + rng.normal(0.0, noise_K, size=members.shape)
    xr.Dataset(
        {"tasmax": (["time", "lat", "lon"], target, {"units": "K"})}, coords=coords
    ).to_netcdf(tmp_path / "ref.nc")
    xr.Dataset(
        {"tasmax": (["member", "time", "lat", "lon"], members, {"units": "K"})},
        coords=coords,
    ).to_netcdf(tmp_path / "pred.nc")
    output_prefix = str(tmp_path / "benchmark_tasmax")
    args = [
        "benchmark",
        "--x0",
        str(tmp_path / "ref.nc"),
        "--x1",
        str(tmp_path / "pred.nc"),
        "--var",
        "tasmax",
        "--output",
        output_prefix,
    ] + (extra_args or [])
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    with open(f"{output_prefix}_metrics_benchmark.json") as f:
        return json.load(f)


def test_cli_benchmark_member_protocol(tmp_path: pathlib.Path) -> None:
    """By default, a random subset of 5 members is used and recorded."""
    metrics = _run_benchmark_members(tmp_path, [0.5] * 7)
    attrs = metrics.pop("attrs")
    assert set(attrs) == {"n_members", "members"}
    assert attrs["n_members"] == 7
    subset = attrs["members"]
    assert len(subset) == 5 and subset == sorted(subset) and len(set(subset)) == 5
    assert all(0 <= m < 7 for m in subset)
    assert set(metrics) == {
        "rmse",
        "bias_mean",
        "ralsd",
        "bias_max1day",
        "pss",
        "bias_annstd",
    }
    assert all(v is not None and np.isfinite(v) for v in metrics.values())


def test_cli_benchmark_member_selection_reproducible(tmp_path: pathlib.Path) -> None:
    """Repeated runs select the same members; fewer members than the cap use all."""
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
    run_a = _run_benchmark_members(tmp_path / "a", [0.5] * 7)
    run_b = _run_benchmark_members(tmp_path / "b", [0.5] * 7)
    assert run_a == run_b
    run_c = _run_benchmark_members(tmp_path / "c", [0.5] * 3)
    assert run_c["attrs"]["members"] == [0, 1, 2]


def test_cli_benchmark_member_mean_before_abs(tmp_path: pathlib.Path) -> None:
    """
    Opposite member biases cancel: the member mean is taken before abs().

    Two members at +2 K and two at -2 K (no noise): the member-mean annual
    maximum equals the target one, so bias_max1day is 0 (and not 2 K, which
    would be the mean of the absolute per-member biases). The ensemble mean
    equals the target, so rmse and bias_mean are 0 too.
    """
    metrics = _run_benchmark_members(
        tmp_path, [2.0, 2.0, -2.0, -2.0], ["--max-members", "4"], noise_K=0.0
    )
    assert metrics["attrs"]["members"] == [0, 1, 2, 3]
    assert metrics["bias_max1day"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["bias_mean"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)


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
            "--hist",
            p["x0_hist"],
            "--fut",
            p["x0_fut"],
            "--var",
            var,
            "--output",
            output_prefix,
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
