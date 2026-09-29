from __future__ import annotations

import pathlib

import numpy as np
import pytest
import xarray as xr

import ml_evaluation.viz.plot_hexagon as ph
import ml_evaluation.viz.plot_var_skill_scorecard as pv
from ml_evaluation.viz.plot_hexagon import read_hexagon_metrics, select_member
from ml_evaluation.viz.plot_var_skill_scorecard import read_var_skill_metrics

# One store's worth of diagnostics; member m holds (m + 1) x the base value, so a
# member selection is visible in the returned scalar.
BASE = {
    "metrics_fast_field.zarr": {
        "bias_mean_field": 1.0,
        "bias_p1_field": 2.0,
        "bias_p99_field": 3.0,
        "bias_acorr_field": 4.0,
        "bias_max1day_field": 5.0,
        "rmse_field": 6.0,
    },
    "metrics_fast_float.zarr": {"rmse_float": 10.0, "ralsd_float": 20.0},
    "metrics_fast_pr_field.zarr": {
        "bias_r1_field": 7.0,
        "bias_sdii_field": 8.0,
        "bias_p98wet_field": 9.0,
    },
}
NMEMBER = 3


def _dataset(diags: dict[str, float]) -> xr.Dataset:
    keys = list(diags)
    data = np.stack(
        [
            np.stack([np.full((2, 2), diags[k] * (m + 1)) for m in range(NMEMBER)])
            for k in keys
        ]
    )
    return xr.Dataset(
        {"value": (("diagnostic", "member", "y", "x"), data)},
        coords={"diagnostic": keys, "member": np.arange(NMEMBER)},
    )


@pytest.fixture
def cell(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A metrics directory whose stores are served from memory.

    The readers only require the store *directory* to exist; ``open_zarr`` is
    monkeypatched so the test needs no zarr writer.
    """
    d = tmp_path / "pr"
    for store in BASE:
        (d / store).mkdir(parents=True)

    def fake_open_zarr(path, *a, **k):
        return _dataset(BASE[pathlib.Path(path).name])

    monkeypatch.setattr(ph.xr, "open_zarr", fake_open_zarr)
    monkeypatch.setattr(pv.xr, "open_zarr", fake_open_zarr)
    return str(d)


def test_select_member_modes() -> None:
    """Positional pick, mean over members, and the two no-op cases."""
    da = xr.DataArray(
        np.array([1.0, 2.0, 3.0]), dims="member", coords={"member": [0, 1, 2]}
    )
    assert float(select_member(da, "member", 0)) == 1.0
    assert float(select_member(da, "member", 2)) == 3.0
    assert float(select_member(da, "member", -1)) == 2.0  # mean over members
    assert select_member(da, None, 0).equals(da)  # no dimension asked for
    plain = xr.DataArray(np.array([1.0, 2.0]), dims="x")
    assert select_member(plain, "member", 1).equals(plain)  # dimension absent


def test_read_hexagon_metrics_member_selection(cell: str) -> None:
    """The hexagon axes score one realisation, not a silent member average."""
    m0 = read_hexagon_metrics(cell, "pr")
    m2 = read_hexagon_metrics(cell, "pr", member_num=2)
    mean = read_hexagon_metrics(cell, "pr", member_num=-1)

    assert m0["rmse"] == 10.0  # default is member 0
    assert m2["rmse"] == 30.0  # member 2 holds 3x
    assert mean["rmse"] == 20.0  # (1 + 2 + 3) / 3
    assert m0["bias_r01"] == 7.0 and m2["bias_r01"] == 21.0
    assert m0 != mean


def test_read_var_skill_metrics_member_selection(cell: str) -> None:
    """The per-variable scorecard columns take the same selector."""

    def value(**kw):
        return read_var_skill_metrics(cell, "pr", ["bias_mean"], **kw)["bias_mean"][0]

    assert value() == 1.0
    assert value(member_num=1) == 2.0
    assert value(member_num=-1) == 2.0  # (1 + 2 + 3) / 3


def test_cv_uses_the_selected_member(cell: str) -> None:
    """The CV line is computed from the same member as the value above it."""
    _, cv = read_var_skill_metrics(cell, "pr", ["bias_mean"], member_num=2)["bias_mean"]
    # a uniform field has zero spread whichever member is picked
    assert cv == 0.0


def test_missing_member_falls_back_to_first(
    cell: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Asking for a member that was never written warns and uses the first."""
    assert read_hexagon_metrics(cell, "pr", member_num=9)["rmse"] == 10.0
    assert "doesn't exist" in capsys.readouterr().out


def test_scorecard_entry_points_accept_member_num(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both figure entry points thread the selector down to the reader.

    Guards the signature as well as the body: the scorecard used to read
    ``member_num`` without declaring it.
    """
    import ml_evaluation.viz.plot_var_skill_scorecard as pvs
    from ml_evaluation.viz.plot_hexagon import plot_multivar_skill_scorecard

    seen: list[int] = []
    for model in ("A", "B"):
        (tmp_path / model / "pr").mkdir(parents=True)
    paths = {m: str(tmp_path / m) for m in ("A", "B")}

    monkeypatch.setattr(
        ph,
        "_read_metric",
        lambda d, k, *, member_dim, member_num: (seen.append(member_num) or 1.0),
    )
    monkeypatch.setattr(ph.plt, "savefig", lambda *a, **k: None)
    plot_multivar_skill_scorecard(
        paths, str(tmp_path / "fig"), ["pr"], reference="A", member_num=2
    )
    assert seen and set(seen) == {2}

    seen.clear()
    monkeypatch.setattr(
        pvs,
        "read_var_skill_metrics",
        lambda d, var, metrics, *, member_dim, member_num: (
            seen.append(member_num) or {"rmse": (1.0, 0.1)}
        ),
    )
    monkeypatch.setattr(pvs, "render_var_skill_scorecard", lambda *a, **k: "fig.png")
    pvs.plot_var_skill_scorecard(
        paths, str(tmp_path / "fig"), ["pr"], metrics=["rmse"], member_num=-1
    )
    assert seen and set(seen) == {-1}
