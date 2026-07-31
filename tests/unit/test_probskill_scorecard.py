from __future__ import annotations

import pathlib

import numpy as np
import pytest
import xarray as xr

import ml_evaluation.viz.plot_multivar_probskill_scorecard as pp

# What `ml-eval prob` writes: three domain scalars, SERAT only for an ensemble.
STORE = {
    "crps_mean_float": 2.0,
    "bs50p_float": 0.20,
    "se_ssrat_float": 0.5,
}


@pytest.fixture
def cell(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """A metrics directory served from memory (no zarr writer needed)."""

    def make(values: dict[str, float] | None = None):
        d = tmp_path / "pr"
        (d / pp.PROB_STORE).mkdir(parents=True, exist_ok=True)
        served = STORE if values is None else values

        def fake_open_zarr(path, *a, **k):
            keys = list(served)
            return xr.Dataset(
                {"value": (("diagnostic",), np.array([served[k] for k in keys]))},
                coords={"diagnostic": keys},
            )

        monkeypatch.setattr(pp.xr, "open_zarr", fake_open_zarr)
        return str(d)

    return make


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Capture the rendered figure instead of writing it."""
    calls: list[dict] = []
    real = pp.plt.subplots

    def fake_savefig(self, path, *a, **k):
        calls.append({"path": path})

    monkeypatch.setattr(pp.plt.Figure, "savefig", fake_savefig)
    monkeypatch.setattr(pp.plt, "subplots", real)
    return calls


def test_prob_columns_are_three_wedges() -> None:
    """The triangle is CRPS / BS / SERAT unless the caller says otherwise."""
    assert pp.prob_columns() == [("crps", "CRPS"), ("bs", "BS"), ("serat", "SERAT")]
    relabelled = pp.prob_columns([("crps", "MAE"), "bs", "serat"])
    assert relabelled[0] == ("crps", "MAE")  # a pair overrides the wedge label
    with pytest.raises(ValueError, match="needs exactly 3 wedges"):
        pp.prob_columns(["crps", "bs"])
    with pytest.raises(ValueError, match="unknown metric"):
        pp.prob_columns(["crps", "bs", "ralsd"])


def test_serat_is_folded_around_one() -> None:
    """|log10(SERAT)|: 1 is perfect, and equal under/over-dispersion tie."""
    folded = pp._comparable(np.array([1.0, 0.5, 2.0, 10.0]), "serat")
    assert folded[0] == pytest.approx(0.0)          # perfectly dispersed
    assert folded[1] == pytest.approx(folded[2])    # half vs double: same badness
    assert folded[3] == pytest.approx(1.0)
    # an error metric passes through untouched
    assert list(pp._comparable(np.array([1.0, 2.0]), "crps")) == [1.0, 2.0]


def test_normalise_is_relative_to_the_column_max() -> None:
    """Every column lands in [0, 1] with 1 = worst of the plotted models."""
    out = pp._normalise(np.array([1.0, 4.0, 2.0]), "crps")
    assert list(out) == [0.25, 1.0, 0.5]
    # NaNs survive as NaN, and an all-NaN column does not raise
    out = pp._normalise(np.array([np.nan, 2.0]), "crps")
    assert np.isnan(out[0]) and out[1] == 1.0
    assert np.isnan(pp._normalise(np.array([np.nan]), "crps")).all()
    # a column of zeros is all-equal, not a division by zero
    assert list(pp._normalise(np.array([0.0, 0.0]), "crps")) == [0.0, 0.0]


def test_read_prob_metrics(cell) -> None:
    """The three scalars come back in raw units."""
    got = pp.read_prob_metrics(cell())
    assert got == {"crps": 2.0, "bs": 0.20, "serat": 0.5}


def test_missing_serat_is_nan_not_an_error(cell) -> None:
    """A single-member run has no SERAT; that is a blank wedge, not a failure."""
    got = pp.read_prob_metrics(cell({"crps_mean_float": 1.0, "bs50p_float": 0.1}))
    assert got["crps"] == 1.0 and got["bs"] == 0.1
    assert np.isnan(got["serat"])


def test_missing_store_is_all_nan(tmp_path: pathlib.Path) -> None:
    """No `ml-eval prob` output at all -> every wedge NaN, no exception."""
    got = pp.read_prob_metrics(str(tmp_path))
    assert all(np.isnan(v) for v in got.values())


def test_render_places_one_triangle_per_cell(tmp_path: pathlib.Path) -> None:
    """Rows x columns of triangles, three wedges each, values annotated."""
    per_model = {
        "RESDIFF": {"pr": {"crps": 2.0, "bs": 0.2, "serat": 0.5},
                    "tas": {"crps": 1.0, "bs": 0.1, "serat": 1.0}},
        "FLOW-MATCHING": {"pr": {"crps": 1.0, "bs": 0.1, "serat": 1.0},
                          "tas": {"crps": 2.0, "bs": 0.2, "serat": 2.0}},
    }
    path = pp.render_multivar_probskill_scorecard(
        per_model, str(tmp_path / "fig"), variables=["pr", "tas"], season="annual"
    )
    assert path.endswith("probskill_scorecard_annual.png")
    assert pathlib.Path(path).exists()


def test_render_marks_a_missing_wedge(tmp_path: pathlib.Path) -> None:
    """A deterministic model still gets CRPS and BS, with SERAT left blank."""
    per_model = {
        "UNET-MULTI": {"pr": {"crps": 3.0, "bs": 0.3, "serat": np.nan}},
        "RESDIFF": {"pr": {"crps": 1.0, "bs": 0.1, "serat": 1.2}},
    }
    path = pp.render_multivar_probskill_scorecard(
        per_model, str(tmp_path / "fig"), variables=["pr"]
    )
    assert pathlib.Path(path).exists()


def test_plot_skips_sources_without_prob_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model that never ran `ml-eval prob` drops out; the figure still draws."""
    for model in ("RESDIFF", "UNET-MULTI"):
        (tmp_path / model / "pr").mkdir(parents=True)

    def fake_read(d, metrics=None, *, member_dim="member", sel=None, label=""):
        if "UNET-MULTI" in d:
            return {"crps": np.nan, "bs": np.nan, "serat": np.nan}
        return {"crps": 1.0, "bs": 0.1, "serat": 1.1}

    monkeypatch.setattr(pp, "read_prob_metrics", fake_read)
    seen = {}
    monkeypatch.setattr(
        pp, "render_multivar_probskill_scorecard",
        lambda per_model, out, **k: seen.update(per_model) or "fig.png",
    )

    out = pp.plot_multivar_probskill_scorecard(
        {"RESDIFF": str(tmp_path / "RESDIFF"), "UNET-MULTI": str(tmp_path / "UNET-MULTI")},
        str(tmp_path / "fig"),
        ["pr"],
    )
    assert out == "fig.png"
    assert list(seen) == ["RESDIFF"]


@pytest.fixture
def swept_cell(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A metrics dir whose store holds an ensemble-size sweep.

    What `ml-eval prob --n-members "2,4,10"` writes: the same diagnostics, each
    stacked along ``n_members``. SERAT climbs toward 1 with the ensemble size
    (a small sub-ensemble is under-dispersive); CRPS is deliberately flat.
    """
    sizes = [2, 4, 10]
    values = np.array(
        [
            [2.0, 2.0, 2.0],                    # crps
            [0.20, 0.20, 0.20],                 # bs
            [1 - 1 / n for n in sizes],         # serat
        ]
    )
    d = tmp_path / "pr"
    (d / pp.PROB_STORE).mkdir(parents=True, exist_ok=True)

    def fake_open_zarr(path, *a, **k):
        return xr.Dataset(
            {"value": (("diagnostic", "n_members"), values)},
            coords={"diagnostic": list(STORE), "n_members": sizes},
        )

    monkeypatch.setattr(pp.xr, "open_zarr", fake_open_zarr)
    return str(d)


def test_sweep_selection_picks_one_ensemble_size(swept_cell: str) -> None:
    """`sel` reads one size out of a store that holds several."""
    assert pp.read_prob_metrics(swept_cell, sel={"n_members": 2})["serat"] == 0.5
    assert pp.read_prob_metrics(swept_cell, sel={"n_members": 4})["serat"] == 0.75


def test_sweep_without_selection_takes_the_full_ensemble(swept_cell: str, capsys) -> None:
    """No `sel` -> the largest size, said out loud -- never the mean across sizes.

    Averaging a diagnostic over ensemble sizes would be a number that describes
    no ensemble at all, so the fallback has to be a deliberate selection.
    """
    assert pp.read_prob_metrics(swept_cell)["serat"] == 1 - 1 / 10
    assert "using the largest (n_members=10)" in capsys.readouterr().out


def test_unscored_size_is_an_error_not_a_blank_wedge(swept_cell: str) -> None:
    """A size that was never scored must not degrade into a NaN cell."""
    with pytest.raises(pp.SweepSelectionError, match=r"available: \[2, 4, 10\]"):
        pp.read_prob_metrics(swept_cell, sel={"n_members": 7})


def test_row_sel_makes_one_store_several_rows(
    swept_cell: str, tmp_path: pathlib.Path, captured: list[dict]
) -> None:
    """The ensemble-size figure: rows share a path and differ only by `sel`."""
    base = str(tmp_path)
    sizes = [2, 4, 10]
    paths = {f"RESDIFF (n={n})": base for n in sizes}
    row_sel = {f"RESDIFF (n={n})": {"n_members": n} for n in sizes}

    seen: dict = {}
    render = pp.render_multivar_probskill_scorecard

    def spy(per_model, out, **kwargs):
        seen.update(per_model)
        return render(per_model, out, **kwargs)

    pp.render_multivar_probskill_scorecard = spy
    try:
        pp.plot_multivar_probskill_scorecard(
            paths, str(tmp_path / "fig"), ["pr"], row_sel=row_sel
        )
    finally:
        pp.render_multivar_probskill_scorecard = render

    assert list(seen) == [f"RESDIFF (n={n})" for n in sizes]
    serats = [seen[f"RESDIFF (n={n})"]["pr"]["serat"] for n in sizes]
    assert serats == [0.5, 0.75, 0.9]


def test_plot_returns_none_when_nothing_ran(tmp_path: pathlib.Path) -> None:
    """No prob stores anywhere -> None and a message, not a traceback."""
    (tmp_path / "RESDIFF" / "pr").mkdir(parents=True)
    assert pp.plot_multivar_probskill_scorecard(
        {"RESDIFF": str(tmp_path / "RESDIFF")}, str(tmp_path / "fig"), ["pr"]
    ) is None
