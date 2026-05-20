from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import pytest
from ml_evaluation.evaluation.indices import mean, quantile, su, txx, txn


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


def test_quantile_index(dummy_dataset: xr.Dataset) -> None:
    """
    Test the quantile index calculation over the time dimension.

    Parameters
    ----------
    dummy_dataset : xr.Dataset
        Fixture providing the dummy dataset.

    Returns
    -------
    None
    """
    result = quantile(dummy_dataset, "tasmax", 0.5)
    assert "time" not in result.dims
    # Median for point (0, 0): sorted [295, 296, 298, 301, 305] -> 298.0
    assert float(result["tasmax"].sel(lat=45.0, lon=10.0)) == 298.0


def test_su_index(dummy_dataset: xr.Dataset) -> None:
    """
    Test the summer days (su) index calculation.

    Parameters
    ----------
    dummy_dataset : xr.Dataset
        Fixture providing the dummy dataset.

    Returns
    -------
    None
    """
    # Count days above 300K (27C)
    result = su(dummy_dataset, "tasmax", threshold=300.0)
    # Values at (0, 0): [295, 301, 298, 305, 296] -> 2 values above 300.0
    assert float(result["tasmax"].sel(lat=45.0, lon=10.0)) == 2.0


def test_txx_index(dummy_dataset: xr.Dataset) -> None:
    """
    Test the mean annual maximum (txx) index calculation.

    Parameters
    ----------
    dummy_dataset : xr.Dataset
        Fixture providing the dummy dataset.

    Returns
    -------
    None
    """
    result = txx(dummy_dataset, "tasmax")
    # Annual max for single year 2026: 305.0 at (0, 0)
    assert float(result["tasmax"].sel(lat=45.0, lon=10.0)) == 305.0


def test_txn_index(dummy_dataset: xr.Dataset) -> None:
    """
    Test the mean annual minimum (txn) index calculation.

    Parameters
    ----------
    dummy_dataset : xr.Dataset
        Fixture providing the dummy dataset.

    Returns
    -------
    None
    """
    result = txn(dummy_dataset, "tasmax")
    # Annual min for single year 2026: 295.0 at (0, 0)
    assert float(result["tasmax"].sel(lat=45.0, lon=10.0)) == 295.0
