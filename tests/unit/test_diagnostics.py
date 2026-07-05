from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import pytest
from ml_evaluation.evaluation.diagnostics import (
    ralsd,
    psd,
    calc_pdf_metric_xr,
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
    distance = ralsd(psd_x0, psd_x1, var="tasmax")
    assert isinstance(distance, float)


def test_calc_pdf_metric_xr(test_datasets: tuple[xr.Dataset, xr.Dataset]) -> None:
    """
    Test PDF metric calculations (like Perkins Skill Score and FCSS).

    Parameters
    ----------
    test_datasets : tuple[xr.Dataset, xr.Dataset]
        Fixture providing the reference and prediction datasets.

    Returns
    -------
    None
    """
    x0, x1 = test_datasets

    # Test Perkins Skill Score (pss)
    pss_res = calc_pdf_metric_xr(
        x0,
        x1,
        var="tasmax",
        dim=("time",),
        typedistance="pss",
        nbins=10,
    )
    assert "tasmax" in pss_res
    assert pss_res["tasmax"].shape == (2, 2)  # lat, lon

    # Test Fractional Contribution Skill Score (fcss)
    fcss_res = calc_pdf_metric_xr(
        x0,
        x1,
        var="tasmax",
        dim=None,
        typedistance="fcss",
        nbins=10,
    )
    assert "tasmax" in fcss_res
    assert isinstance(fcss_res["tasmax"].values, np.ndarray)
