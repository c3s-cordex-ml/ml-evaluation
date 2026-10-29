# SPDX-FileCopyrightText: 2026 c3s-cordex-ml
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ml_evaluation.viz.plot_hexagon as ph
from ml_evaluation.evaluation import indices
from ml_evaluation.viz.plot_hexagon import daily_sigma, variability_scale

# The truth fields one metrics directory serves. targ_std_field carries the
# scale; the P01/P99 pair is present so a test can prove it is *not* used as a
# stand-in when targ_std_field is missing.
FIELDS = {
    "targ_std_field": [[1.0, 3.0], [1.0, 3.0]],  # domain mean 2.0
    "targ_p1_field": [[1.0, 2.0], [1.0, 2.0]],
    "targ_p99_field": [[5.0, 5.0], [5.0, 5.0]],
}


def _dataset(diags: dict[str, list]) -> xr.Dataset:
    keys = list(diags)
    data = np.stack([np.asarray(diags[k])[None] for k in keys])
    return xr.Dataset(
        {"value": (("diagnostic", "member", "y", "x"), data)},
        coords={"diagnostic": keys, "member": [0]},
    )


def _cell(tmp_path, monkeypatch, diags) -> str:
    """A metrics directory whose stores are served from memory."""
    d = tmp_path / "tas"
    (d / "metrics_fast_field.zarr").mkdir(parents=True)
    monkeypatch.setattr(ph.xr, "open_zarr", lambda path, *a, **k: _dataset(diags))
    return str(d)


@pytest.fixture
def cell(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> str:
    return _cell(tmp_path, monkeypatch, FIELDS)


def test_season_std_is_the_daily_spread() -> None:
    """The index takes the spread of the daily values themselves."""
    time = pd.date_range("2001-01-01", periods=730, freq="D")
    values = np.where(np.arange(730) % 2 == 0, 1.0, 3.0)  # daily std 1, mean 2
    ds = xr.Dataset({"tas": (("time",), values)}, coords={"time": time})

    assert float(indices.season_std(ds, "tas")["tas"]) == pytest.approx(1.0)


def test_season_std_honours_the_season_filter() -> None:
    """Filtering happens before the spread is taken."""
    time = pd.date_range("2001-01-01", periods=365, freq="D")
    values = np.where(time.month == 1, 5.0, 0.0)
    ds = xr.Dataset({"tas": (("time",), values)}, coords={"time": time})

    # January alone is constant; the whole year is not
    january = indices.season_std(ds, "tas", season="JAN")["tas"]
    assert float(january) == pytest.approx(0.0)
    assert float(indices.season_std(ds, "tas")["tas"]) > 0.0


def test_daily_sigma_is_the_domain_mean_of_the_truth_std(cell: str) -> None:
    assert daily_sigma(cell) == pytest.approx(2.0)


def test_no_proxy_when_targ_std_field_is_absent(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store predating the diagnostic yields NaN, never a P01/P99 stand-in.

    The percentile fields are present and would give 1.75 if approximated; the
    scale must still come back NaN so the wedge is left blank rather than
    coloured from an approximation the caller never asked for.
    """
    stale = {k: v for k, v in FIELDS.items() if k != "targ_std_field"}
    cell = _cell(tmp_path, monkeypatch, stale)
    assert np.isnan(daily_sigma(cell))
    assert np.isnan(variability_scale(cell, "rmse"))


def test_only_unit_carrying_metrics_get_sigma(cell: str) -> None:
    """Everything in the variable's units shares one scale; the rest get none."""
    for key in ("rmse", "bias_mean", "bias_p01", "bias_p99", "bias_rx1day"):
        assert variability_scale(cell, key) == pytest.approx(2.0)
    for key in ("bias_acorr", "bias_r01", "ralsd"):
        assert np.isnan(variability_scale(cell, key))


def _wedge_values(tmp_path, monkeypatch, *, scale, sigma=2.0, errors=None):
    """Draw a two-row scorecard and return the model row's wedge colours."""
    errors = errors or {"REF": 0.01, "M": 0.1}
    for model in errors:
        (tmp_path / model / "tas").mkdir(parents=True, exist_ok=True)
    paths = {m: str(tmp_path / m) for m in errors}

    monkeypatch.setattr(
        ph, "_read_metric", lambda d, k, **kw: errors[pathlib.Path(d).parent.name]
    )
    monkeypatch.setattr(
        ph,
        "variability_scale",
        lambda d, k, **kw: sigma if k in ph.SCALE_IS_SIGMA else np.nan,
    )
    monkeypatch.setattr(ph.plt, "savefig", lambda *a, **k: None)

    drawn: list = []
    monkeypatch.setattr(
        ph, "_wedge", lambda ax, cx, cy, i, r, facecolor, **kw: drawn.append(facecolor)
    )
    ph.plot_multivar_skill_scorecard(
        paths, str(tmp_path / "fig"), ["tas"], reference="REF", scale=scale
    )
    clip = ph.DEFAULT_CLIP[scale]
    axes = [k for k, _ in ph.scorecard_axes("tas")]

    # colour = lambda v: ph.plt.get_cmap("RdBu")(
    #    (max(-clip, min(clip, v)) + clip) / (2 * clip)
    # )
    # Added to address quality check error
    # E731 Do not assign a `lambda` expression, use a `def`
    def colour(v):
        return ph.plt.get_cmap("RdBu")((max(-clip, min(clip, v)) + clip) / (2 * clip))

    return dict(zip(axes, drawn[6:])), colour  # drawn[:6] is the reference row


def test_sigma_wedge_reads_as_percent_of_variability(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tenfold error ratio that is small against sigma stays a small value.

    Reference errs by 0.01, model by 0.1, observed sigma 2.0: the relative rule
    calls that -900% (saturated), the variability rule -4.5%.
    """
    wedges, colour = _wedge_values(tmp_path, monkeypatch, scale="variability")
    assert wedges["bias_mean"] == pytest.approx(colour(-4.5))

    wedges, colour = _wedge_values(tmp_path, monkeypatch, scale="relative")
    assert wedges["bias_mean"] == pytest.approx(colour(-900.0))


def test_scaleless_wedges_map_full_relative_change_onto_the_bar(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RALSD and friends fill the bar at +/-100% relative, not at 1%.

    A doubled error is exactly -100% relative, which must land on the clip; the
    sigma wedges of the same cell, where that same doubling is 0.5% of the
    observed variability, must stay near white.
    """
    clip = ph.DEFAULT_CLIP["variability"]
    wedges, colour = _wedge_values(
        tmp_path, monkeypatch, scale="variability", errors={"REF": 0.01, "M": 0.02}
    )
    assert wedges["ralsd"] == pytest.approx(colour(-clip))
    assert wedges["bias_mean"] == pytest.approx(colour(-0.5))


def test_stale_store_leaves_sigma_wedges_blank(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Without targ_std_field the unit-carrying wedges are blank, and it is said."""
    wedges, colour = _wedge_values(
        tmp_path, monkeypatch, scale="variability", sigma=np.nan
    )
    nan_colour = "0.9"
    for key in ("rmse", "bias_mean", "bias_p01", "bias_p99"):
        assert wedges[key] == nan_colour
    assert wedges["ralsd"] != nan_colour  # the relative wedges still draw
    out = capsys.readouterr().out
    assert "targ_std_field" in out and "ml-eval fast" in out
