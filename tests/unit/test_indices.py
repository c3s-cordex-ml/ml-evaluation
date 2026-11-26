# SPDX-FileCopyrightText: 2026 c3s-cordex-ml
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import pytest
from ml_evaluation.evaluation.indices import (
    mean,
    season_quantile,
    calday_quantile,
    ann_extrema,
    ann_freq_wrt_array,
    ann_maxspell_wrt_array,
    ann_spellfreq_wrt_array,
    ann_sdii,
    lag1_corr,
    interannual_var,
)


@pytest.fixture
def dummy_dataset() -> xr.Dataset:
    """
    Generate a dummy climate dataset for testing indices.

    Creates a dataset containing daily maximum temperature data ('tasmax')
    over 5 days for a 2x2 grid.

    Parameters
    ----------
    None

    Returns
    -------
    xr.Dataset
        Dummy dataset for testing.
    """
    times = pd.date_range("2026-01-01", periods=5, freq="D")
    lats = [45.0, 46.0]
    lons = [10.0, 11.0]

    # 5 days, 2 latitudes, 2 longitudes
    # Temperature values in Kelvin: around 295K (22C) to 305K (32C)
    data = np.array(
        [
            [[295.0, 296.0], [297.0, 298.0]],
            [[301.0, 302.0], [303.0, 304.0]],
            [[298.0, 299.0], [300.0, 301.0]],
            [[305.0, 306.0], [307.0, 308.0]],
            [[296.0, 297.0], [298.0, 299.0]],
        ]
    )

    ds = xr.Dataset(
        {"tasmax": (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    return ds


def test_mean_index(dummy_dataset: xr.Dataset) -> None:
    """
    Test the mean index calculation over the time dimension.

    Parameters
    ----------
    dummy_dataset : xr.Dataset
        Fixture providing the dummy dataset.

    Returns
    -------
    None
    """
    result = mean(dummy_dataset, "tasmax")
    assert "time" not in result.dims
    assert result["tasmax"].shape == (2, 2)
    # Expected mean for point (0, 0) is (295+301+298+305+296)/5 = 299.0
    assert float(result["tasmax"].sel(lat=45.0, lon=10.0)) == 299.0


def test_plus_indices(dummy_dataset: xr.Dataset) -> None:
    """
    Test generalised versions of indices.

    Parameters
    ----------
    dummy_dataset : xr.Dataset
        Fixture providing the dummy dataset.

    Returns
    -------
    None
    """
    # Test season_quantile
    q_val = season_quantile(dummy_dataset, "tasmax", 0.5)
    assert float(q_val["tasmax"].sel(lat=45.0, lon=10.0)) == 298.0

    # Test ann_extrema
    ext_max = ann_extrema(dummy_dataset, "tasmax", maxmin="max", cmean=True)
    assert float(ext_max["tasmax"].sel(lat=45.0, lon=10.0)) == 305.0

    # Test ann_freq_wrt_array
    freq = ann_freq_wrt_array(
        dummy_dataset, "tasmax", threshold=300.0, relation="gt", cmean=True
    )
    assert float(freq["tasmax"].sel(lat=45.0, lon=10.0)) == 2.0

    # Test ann_maxspell_wrt_array
    spell = ann_maxspell_wrt_array(
        dummy_dataset, "tasmax", threshold=300.0, relation="lt", cmean=True
    )
    # The condition is tasmax < 300. For (0,0):
    # 295 (<300: T)
    # 301 (<300: F)
    # 298 (<300: T)
    # 305 (<300: F)
    # 296 (<300: T)
    # Max consecutive spell length is 1.
    assert float(spell["tasmax"].sel(lat=45.0, lon=10.0)) == 1.0

    # Test ann_sdii
    sdii_val = ann_sdii(dummy_dataset, "tasmax", wet_threshold=300.0, cmean=True)
    # Values above 300: 301 and 305. Mean: 303.0
    assert float(sdii_val["tasmax"].sel(lat=45.0, lon=10.0)) == 303.0

    # Test interannual_var
    ivar = interannual_var(dummy_dataset, "tasmax")
    # A single year, so standard deviation of the single year mean across years is 0.0
    assert float(ivar["tasmax"].sel(lat=45.0, lon=10.0)) == 0.0

    # Test lag1_corr
    lcorr = lag1_corr(dummy_dataset, "tasmax")
    assert "tasmax" in lcorr


def test_ann_spellfreq_wrt_array() -> None:
    """
    Test counting of days in spells of at least ``min_length`` event days.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Event days (value 1 > 0.5): runs of length 2, 6, 3 and 7 within 2026
    pattern = [1, 1, 0] + [1] * 6 + [0] + [1] * 3 + [0] + [1] * 7 + [0]
    times = pd.date_range("2026-01-01", periods=len(pattern), freq="D")
    ds = xr.Dataset(
        {"tas": (["time", "lat", "lon"], np.array(pattern, float)[:, None, None])},
        coords={"time": times, "lat": [45.0], "lon": [10.0]},
    )

    res = ann_spellfreq_wrt_array(ds, "tas", threshold=0.5, min_length=6, cmean=True)
    assert float(res["tas"].squeeze()) == 13.0  # 6 + 7

    res = ann_spellfreq_wrt_array(ds, "tas", threshold=0.5, min_length=3, cmean=True)
    assert float(res["tas"].squeeze()) == 16.0  # 6 + 3 + 7

    # min_length=1 reduces to the plain event-day count
    res = ann_spellfreq_wrt_array(ds, "tas", threshold=0.5, min_length=1, cmean=True)
    freq = ann_freq_wrt_array(ds, "tas", threshold=0.5, cmean=True)
    assert float(res["tas"].squeeze()) == float(freq["tas"].squeeze()) == 18.0

    with pytest.raises(ValueError):
        ann_spellfreq_wrt_array(ds, "tas", threshold=0.5, min_length=0)


def test_calday_quantile_members() -> None:
    """Window dim must not clash with a 'member' dim; pool_dim pools members."""
    times = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    rng = np.random.default_rng(0)
    data = rng.random((3, len(times), 2, 2))
    ds = xr.Dataset(
        {"tasmax": (["member", "time", "lat", "lon"], data)},
        coords={
            "member": [0, 1, 2],
            "time": times,
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0],
        },
    )

    # Per-member thresholds (member dim kept)
    per_member = calday_quantile(ds, "tasmax", q=0.9)
    assert dict(per_member["tasmax"].sizes) == {
        "dayofyear": 366,
        "member": 3,
        "lat": 2,
        "lon": 2,
    }
    expected_m1 = calday_quantile(ds.isel(member=1, drop=True), "tasmax", q=0.9)
    xr.testing.assert_allclose(
        per_member["tasmax"].sel(member=1, drop=True), expected_m1["tasmax"]
    )

    # Pooled threshold (member dim reduced)
    pooled = calday_quantile(ds, "tasmax", q=0.9, pool_dim="member")
    assert dict(pooled["tasmax"].sizes) == {"dayofyear": 366, "lat": 2, "lon": 2}

    # Check one value against a direct computation: Jan 3 of both years,
    # +/- 2 days, all members, pooled together
    doy3 = times.dayofyear == 3
    idx = np.concatenate([np.where(doy3)[0] + s for s in range(-2, 3)])
    expected = np.quantile(data[:, idx, 0, 0], 0.9)
    np.testing.assert_allclose(pooled["tasmax"].sel(dayofyear=3).values[0, 0], expected)


def test_season_quantile_members() -> None:
    """pool_dim pools members into one quantile; otherwise kept per member."""
    times = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    rng = np.random.default_rng(0)
    data = rng.random((3, len(times), 2, 2))
    ds = xr.Dataset(
        {"tasmax": (["member", "time", "lat", "lon"], data)},
        coords={
            "member": [0, 1, 2],
            "time": times,
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0],
        },
    )

    # Per-member quantiles (member dim kept)
    per_member = season_quantile(ds, "tasmax", q=0.9)
    assert dict(per_member["tasmax"].sizes) == {"member": 3, "lat": 2, "lon": 2}

    # Pooled quantile (member dim reduced), checked against a direct computation
    pooled = season_quantile(ds, "tasmax", q=0.9, pool_dim="member")
    assert dict(pooled["tasmax"].sizes) == {"lat": 2, "lon": 2}
    np.testing.assert_allclose(
        pooled["tasmax"].values[0, 0], np.quantile(data[:, :, 0, 0], 0.9)
    )

    # Absent pool_dim is ignored
    no_pool = season_quantile(
        ds.isel(member=0, drop=True), "tasmax", q=0.9, pool_dim="member"
    )
    assert dict(no_pool["tasmax"].sizes) == {"lat": 2, "lon": 2}
