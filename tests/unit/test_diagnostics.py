from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import pytest
from ml_evaluation.evaluation.diagnostics import (
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


@pytest.mark.parametrize(
    "metric, shape",
    [
        # non-square grid: mixed order used to raise a "Shape mismatch" error
        ("sal", (3, 6, 9)),
        ("fss", (3, 6, 9)),
        # single square field (no time): SAL used to silently compare the
        # prediction with its transpose
        ("sal", (8, 8)),
    ],
)
def test_pixelwise_metrics_ignore_dim_order(
    metric: str, shape: tuple[int, ...]
) -> None:
    """
    SAL and FSS give the same result whether the prediction is stored as
    (..., y, x) like the target, or as (..., x, y).
    """
    from ml_evaluation.evaluation.diagnostics import (
        compute_sal_xr,
        compute_sfss2D_xr,
    )

    rng = np.random.default_rng(seed=42)
    dims = ("time", "y", "x")[-len(shape) :]
    # precipitation-like fields (mm/day), non-symmetric so a transpose matters
    target = xr.DataArray(rng.gamma(0.5, 4.0, shape), dims=dims)
    prediction = xr.DataArray(rng.gamma(0.5, 4.0, shape), dims=dims)

    def run(pred: xr.DataArray):
        if metric == "sal":
            return compute_sal_xr(
                pred, target, "pr", eThreshFix=1.0, lat_name="y", lon_name="x"
            )
        return compute_sfss2D_xr(
            pred, target, "pr", 1.0, window=3, lat_name="y", lon_name="x"
        )

    same_order = run(prediction)
    mixed_order = run(prediction.transpose(..., "x", "y"))

    same_order = same_order if isinstance(same_order, tuple) else (same_order,)
    mixed_order = mixed_order if isinstance(mixed_order, tuple) else (mixed_order,)
    for expected, result in zip(same_order, mixed_order):
        xr.testing.assert_allclose(result.transpose(*expected.dims), expected)
