from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import pytest
from ml_evaluation.evaluation.indices import (
    mean,
    season_quantile,
    ann_extrema,
    ann_freq_wrt_array,
    ann_maxspell_wrt_array,
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
