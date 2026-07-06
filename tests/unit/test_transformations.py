from __future__ import annotations

import numpy as np
import xarray as xr
from ml_evaluation.transformations import align_calendar


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
