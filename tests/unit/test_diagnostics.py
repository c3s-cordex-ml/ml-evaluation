from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import pytest
from ml_evaluation.evaluation.diagnostics import (
    mean_bias_index,
    error_ann_index,
    pearsoncorr_ann_index,
    ralsd,
    psd,
)


@pytest.fixture
def test_datasets() -> tuple[xr.Dataset, xr.Dataset]:
    """
    Generate a pair of dummy datasets (x0, x1) for testing diagnostics.

    Parameters
    ----------
    None

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Pair of reference and prediction datasets.
    """
    times = pd.date_range("2026-01-01", periods=5, freq="D")
    lats = [45.0, 46.0]
    lons = [10.0, 11.0]

    # Reference data
    data_x0 = np.array(
        [
            [[295.0, 296.0], [297.0, 298.0]],
            [[301.0, 302.0], [303.0, 304.0]],
            [[298.0, 299.0], [300.0, 301.0]],
            [[305.0, 306.0], [307.0, 308.0]],
            [[296.0, 297.0], [298.0, 299.0]],
        ]
    )

    # Prediction data (with a +1.0 bias everywhere)
    data_x1 = data_x0 + 1.0

    x0 = xr.Dataset(
        {"tasmax": (["time", "lat", "lon"], data_x0)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    x1 = xr.Dataset(
        {"tasmax": (["time", "lat", "lon"], data_x1)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    return x0, x1


def test_mean_bias_diagnostic(test_datasets: tuple[xr.Dataset, xr.Dataset]) -> None:
    """
    Test mean bias calculation between x0 and x1.

    Parameters
    ----------
    test_datasets : tuple[xr.Dataset, xr.Dataset]
        Fixture providing the reference and prediction datasets.

    Returns
    -------
    None
    """
    x0, x1 = test_datasets
    bias = mean_bias_index(x0, x1, var="tasmax")
    # Expected bias is +1.0 everywhere
    assert float(bias["tasmax"].sel(lat=45.0, lon=10.0)) == 1.0


def test_error_ann_index_rmse(test_datasets: tuple[xr.Dataset, xr.Dataset]) -> None:
    """
    Test RMSE calculation between x0 and x1.

    Parameters
    ----------
    test_datasets : tuple[xr.Dataset, xr.Dataset]
        Fixture providing the reference and prediction datasets.

    Returns
    -------
    None
    """
    x0, x1 = test_datasets
    rmse_res = error_ann_index(x0, x1, var="tasmax", typeerror="rmse", dim="time")
    # Expected RMSE is 1.0 everywhere
    assert float(rmse_res["tasmax"].sel(lat=45.0, lon=10.0)) == 1.0


def test_pearsoncorr_ann_index(test_datasets: tuple[xr.Dataset, xr.Dataset]) -> None:
    """
    Test Pearson correlation index calculation between x0 and x1.

    Parameters
    ----------
    test_datasets : tuple[xr.Dataset, xr.Dataset]
        Fixture providing the reference and prediction datasets.

    Returns
    -------
    None
    """
    x0, x1 = test_datasets
    corr = pearsoncorr_ann_index(x0, x1, var="tasmax", dim="time")
    # Since x1 is perfectly linear to x0, correlation should be 1.0
    assert float(corr["tasmax"].sel(lat=45.0, lon=10.0)) == pytest.approx(1.0)


def test_psd_and_ralsd(test_datasets: tuple[xr.Dataset, xr.Dataset]) -> None:
    """
    Test PSD and RALSD diagnostic computations.

    Parameters
    ----------
    test_datasets : tuple[xr.Dataset, xr.Dataset]
        Fixture providing the reference and prediction datasets.

    Returns
    -------
    None
    """
    x0, x1 = test_datasets
    psd_x0, psd_x1 = psd(x0, x1, var="tasmax")
    assert psd_x0.shape == psd_x1.shape
    distance = ralsd(psd_x0, psd_x1)
    assert isinstance(distance, float)
