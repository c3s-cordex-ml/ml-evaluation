from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from ml_evaluation.transformations import align_calendar, standardize_precip_units


def test_align_calendar_no_align() -> None:
    """
    Test align_calendar function when align_leap_days is None.

    Returns
    -------
    None
    """
    times = xr.date_range("2020-02-27", "2020-03-02", calendar="standard", freq="D")
    ds = xr.Dataset(
        {"temp": (["time"], np.arange(len(times), dtype=float))},
        coords={"time": times},
    )
    aligned = align_calendar(ds, align_leap_days=None)
    xr.testing.assert_identical(ds, aligned)


def test_align_calendar_align_leap_days() -> None:
    """
    Test align_calendar function when align_leap_days is "fill".

    Converts a no-leap calendar dataset to a standard calendar and
    verifies that February 29th is added and forward-filled.

    Returns
    -------
    None
    """
    times = xr.date_range("2020-02-27", "2020-03-02", calendar="noleap", freq="D")
    ds = xr.Dataset(
        {"temp": (["time"], np.arange(len(times), dtype=float))},
        coords={"time": times},
    )
    aligned = align_calendar(ds, align_leap_days="fill")

    # The aligned dataset should now have a standard calendar with 5 timestamps (including Feb 29)
    assert len(aligned.time) == 5
    assert "2020-02-29" in aligned.time.dt.strftime("%Y-%m-%d").values

    # Temp value for Feb 29 should be forward-filled from Feb 28
    temp_values = aligned["temp"].values
    expected = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(temp_values, expected)


def _precip_ds(values_mmday: np.ndarray, units: str | None, flux: bool) -> xr.Dataset:
    """Build a small ``pr`` dataset, stored as a flux or in mm/day."""
    data = values_mmday / 86400.0 if flux else values_mmday
    attrs = {"long_name": "precipitation"}
    if units is not None:
        attrs["units"] = units
    return xr.Dataset({"pr": (["time"], data, attrs)})


# Daily precipitation in mm/day, including a NaN that must survive.
PR_MMDAY = np.array([0.0, 0.4, 5.0, 32.0, np.nan])


@pytest.mark.parametrize("units", ["kg m-2 s-1", "kg/m2/s", " KG M-2 S-1 "])
def test_standardize_precip_units_labelled_flux(units: str) -> None:
    out = standardize_precip_units(_precip_ds(PR_MMDAY, units, flux=True), "pr")
    np.testing.assert_allclose(out["pr"].values, PR_MMDAY)
    assert out["pr"].attrs["units"] == "mm/day"
    assert out["pr"].attrs["long_name"] == "precipitation"


@pytest.mark.parametrize("units", ["mm/day", "mm day-1", "mm"])
def test_standardize_precip_units_labelled_daily(units: str) -> None:
    out = standardize_precip_units(_precip_ds(PR_MMDAY, units, flux=False), "pr")
    np.testing.assert_allclose(out["pr"].values, PR_MMDAY)
    assert out["pr"].attrs["units"] == "mm/day"


def test_standardize_precip_units_unlabelled_flux_inferred(
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = standardize_precip_units(_precip_ds(PR_MMDAY, None, flux=True), "pr")
    assert "assumed kg m-2 s-1" in capsys.readouterr().out
    np.testing.assert_allclose(out["pr"].values, PR_MMDAY)
    assert out["pr"].attrs["units"] == "mm/day"


def test_standardize_precip_units_unlabelled_daily_inferred(
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = standardize_precip_units(_precip_ds(PR_MMDAY, None, flux=False), "pr")
    assert "assumed mm/day" in capsys.readouterr().out
    np.testing.assert_allclose(out["pr"].values, PR_MMDAY)


def test_standardize_precip_units_mismatched_pair_agree() -> None:
    target = standardize_precip_units(_precip_ds(PR_MMDAY, "mm/day", False), "pr")
    prediction = standardize_precip_units(_precip_ds(PR_MMDAY, None, True), "pr")
    np.testing.assert_allclose(target["pr"].values, prediction["pr"].values)


def test_standardize_precip_units_does_not_modify_input() -> None:
    ds = _precip_ds(PR_MMDAY, "kg m-2 s-1", flux=True)
    standardize_precip_units(ds, "pr")
    assert ds["pr"].attrs["units"] == "kg m-2 s-1"
    np.testing.assert_allclose(ds["pr"].values, PR_MMDAY / 86400.0)


def test_standardize_precip_units_unrecognised_units_raise() -> None:
    with pytest.raises(ValueError, match="Unrecognised units"):
        standardize_precip_units(_precip_ds(PR_MMDAY, "m/s", flux=False), "pr")


def test_standardize_precip_units_all_nan_unlabelled_raise() -> None:
    all_nan = np.full(3, np.nan)
    with pytest.raises(ValueError, match="all values are NaN"):
        standardize_precip_units(_precip_ds(all_nan, None, flux=False), "pr")


def test_standardize_precip_units_non_precip_units_raise() -> None:
    """The caller must only pass precipitation; other units are rejected."""
    ds = xr.Dataset({"tasmax": (["time"], np.array([280.0, 290.0]), {"units": "K"})})
    with pytest.raises(ValueError, match="Unrecognised units"):
        standardize_precip_units(ds, "tasmax")


def _grid_ds(transpose: bool = False, member: bool = False) -> xr.Dataset:
    """
    Daily tasmax (K) on a 3x4 projected grid with 2-D lat/lon, dims (time, y, x).

    If `transpose`, the spatial dims are stored as (x, y), data and lat/lon alike;
    if `member`, an ensemble dimension of 2 members is added.
    """
    rng = np.random.default_rng(seed=42)
    times = xr.date_range("2041-01-01", periods=5, freq="D")
    y, x = np.arange(3.0), np.arange(4.0)
    lat2d = 45.0 + np.add.outer(y, 0.1 * x)
    lon2d = 7.0 + np.add.outer(0.1 * y, x)
    ds = xr.Dataset(
        {"tasmax": (["time", "y", "x"], 280.0 + rng.random((5, 3, 4)))},
        coords={
            "time": times,
            "y": y,
            "x": x,
            "lat": (["y", "x"], lat2d),
            "lon": (["y", "x"], lon2d),
        },
    )
    if member:
        ds = ds.expand_dims(member=[0, 1])
    if transpose:
        ds = ds.transpose(..., "x", "y")
    return ds


@pytest.mark.parametrize("member", [False, True])
def test_align_dims_accepts_transposed_grid(member: bool) -> None:
    """A prediction stored as (x, y) matches a (y, x) reference, untransposed."""
    from ml_evaluation.transformations import align_dims

    x0 = _grid_ds()
    x1 = _grid_ds(transpose=True, member=member)
    assert x1.lat.dims == ("x", "y")

    _, x1_aligned, _ = align_dims(x0, x1)

    # the data keep their stored order (no costly transpose of the data)
    assert x1_aligned.tasmax.dims[-2:] == ("x", "y")
    # values follow their labels: same as the untransposed prediction
    expected = _grid_ds(member=member).tasmax
    if member:
        expected = expected.transpose("member", ...)
    xr.testing.assert_equal(x1_aligned.tasmax.transpose(*expected.dims), expected)


def test_align_dims_mismatched_grid_raises() -> None:
    """Different lat/lon values (not just a different order) still raise."""
    from ml_evaluation.transformations import align_dims

    x0 = _grid_ds()
    x1 = _grid_ds().assign_coords(lon=_grid_ds().lon + 5.0)
    with pytest.raises(ValueError):
        align_dims(x0, x1)
