from __future__ import annotations

import pathlib

import numpy as np
import pytest

import ml_evaluation.viz.plot_var_series as ps

# The two periods one signal run stores on a single year axis.
HIST = np.arange(1981, 2001)
FUT = np.arange(2041, 2061)


def _series(years: np.ndarray, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic ramp, so a baseline mean is exactly predictable."""
    return years, (years - 1981).astype(float) * 0.1 + offset


def test_series_indices() -> None:
    """The index catalog matches what ``ml-eval signal`` writes per variable."""
    assert ps.series_indices("tasmax") == ["mean", "p01", "p99"]
    assert ps.series_indices("pr") == ["sdii", "r01", "rx1day"]


def test_check_index() -> None:
    """An unstored index is rejected; None falls back to the first one."""
    assert ps._check_index("pr", None) == "sdii"
    assert ps._check_index("tasmax", "p99") == "p99"
    with pytest.raises(ValueError, match="not stored for 'tasmax'"):
        ps._check_index("tasmax", "sdii")


def test_break_gaps() -> None:
    """A jump in the year axis becomes a NaN, so the line breaks at the hole."""
    years = np.array([1998, 1999, 2000, 2041, 2042])
    y, v = ps._break_gaps(years, np.arange(5, dtype=float))
    assert np.isnan(v).sum() == 1
    gap = int(np.flatnonzero(np.isnan(v))[0])
    assert y[gap - 1] == 2000.0 and y[gap + 1] == 2041.0
    # contiguous years are left untouched
    y2, v2 = ps._break_gaps(*(_series(HIST)))
    assert not np.isnan(v2).any() and y2.size == HIST.size


def test_merge_series_primary_wins() -> None:
    """Grafting a baseline run keeps the plotted run's values on shared years."""
    primary = (np.array([2000, 2041]), np.array([1.0, 2.0]))
    extra = (np.array([1999, 2000]), np.array([9.0, 9.0]))
    years, values = ps._merge_series(primary, extra)
    assert list(years) == [1999, 2000, 2041]
    assert list(values) == [9.0, 1.0, 2.0]


def test_baseline_mean_window() -> None:
    """The baseline mean uses only the closed window, ignoring NaNs."""
    years = np.arange(1981, 1985)
    values = np.array([1.0, 3.0, np.nan, 100.0])
    assert ps._baseline_mean(years, values, (1981, 1983)) == 2.0
    assert np.isnan(ps._baseline_mean(years, values, (1990, 2000)))


def test_select_years() -> None:
    """Both a closed window and an explicit year list restrict the panel."""
    years, values = _series(np.arange(1981, 1991))
    y, _ = ps._select_years(years, values, (1983, 1985))
    assert list(y) == [1983, 1984, 1985]
    y, _ = ps._select_years(years, values, [1981, 1990])
    assert list(y) == [1981, 1990]


def test_plot_series_values_and_signal(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both modes render a PNG; the baseline mode names the file an anomaly.

    The zarr reader is monkeypatched with synthetic series so the drawing path
    (gap breaking, reference/driver styling, legend) is exercised without
    on-disk stores. Paths are real directories, so the ``<path>/<var>``
    resolution is exercised for real.
    """
    rows = {
        "RCM truth": tmp_path / "rcm",
        "GCM": tmp_path / "gcm",
        "UNet-single": tmp_path / "unet",
    }
    for base in rows.values():
        (base / "tasmax").mkdir(parents=True)

    offsets = {"rcm": 0.0, "gcm": 5.0, "unet": -2.0}
    years = np.concatenate([HIST, FUT])

    def fake_reader(metrics_dir: str, index: str) -> tuple[np.ndarray, np.ndarray]:
        return _series(years, offsets[pathlib.Path(metrics_dir).parent.name])

    monkeypatch.setattr(ps, "read_series", fake_reader)
    spec = {k: str(v) for k, v in rows.items()}

    raw = ps.plot_var_series(
        spec,
        str(tmp_path / "fig"),
        "tasmax",
        "mean",
        reference="RCM truth",
        driver="GCM",
        units="K",
    )
    assert pathlib.Path(raw).is_file()
    assert pathlib.Path(raw).name == "series_tasmax_mean_annual.png"

    anom = ps.plot_var_series(
        spec,
        str(tmp_path / "fig"),
        "tasmax",
        "mean",
        reference="RCM truth",
        driver="GCM",
        baseline=(1981, 2000),
        units="K",
    )
    assert pathlib.Path(anom).name == "series_tasmax_mean_anom_annual.png"


def test_plot_series_baseline_from_other_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A future-only run is anomalised against a baseline in a separate run.

    Without ``baseline_paths`` the baseline window is empty and the source is
    dropped; with it, the historical years are merged in, so the anomaly is
    defined *and* those years are drawn.
    """
    fut_dir = tmp_path / "fut"
    hist_dir = tmp_path / "hist"
    for base in (fut_dir, hist_dir):
        (base / "tasmax").mkdir(parents=True)

    def fake_reader(metrics_dir: str, index: str) -> tuple[np.ndarray, np.ndarray]:
        period = pathlib.Path(metrics_dir).parent.name
        return _series(FUT if period == "fut" else HIST)

    monkeypatch.setattr(ps, "read_series", fake_reader)
    spec = {"UNet-single": str(fut_dir)}
    out = str(tmp_path / "fig")

    assert (
        ps.plot_var_series(spec, out, "tasmax", "mean", baseline=(1981, 2000)) is None
    )

    captured: dict = {}

    def spy(per_source, var, index, out_dir, **kwargs):
        captured["per_source"] = per_source
        captured["kwargs"] = kwargs
        return "fake.png"

    monkeypatch.setattr(ps, "render_series", spy)
    assert (
        ps.plot_var_series(
            spec,
            out,
            "tasmax",
            "mean",
            baseline=(1981, 2000),
            baseline_paths={"UNet-single": str(hist_dir)},
        )
        == "fake.png"
    )

    years, values = captured["per_source"]["UNet-single"]
    # both periods are on the axis, and the anomaly is relative to the mean of
    # the historical ramp (0.0 .. 1.9 -> 0.95)
    assert years[0] == 1981 and years[-1] == 2060 and years.size == 40
    assert values[0] == pytest.approx(-0.95)
    assert captured["kwargs"]["is_anomaly"] is True


def test_plot_series_scale_and_missing_source(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``scale`` converts units; a source with no store is skipped, not fatal."""
    good = tmp_path / "good"
    (good / "pr").mkdir(parents=True)

    def fake_reader(metrics_dir: str, index: str) -> tuple[np.ndarray, np.ndarray]:
        if pathlib.Path(metrics_dir).parent.name != "good":
            return np.array([], dtype=int), np.array([], dtype=float)
        return HIST, np.full(HIST.size, 2.0)

    monkeypatch.setattr(ps, "read_series", fake_reader)
    captured: dict = {}

    def spy(per_source, var, index, out_dir, **kwargs):
        captured.update(per_source)
        return "fake.png"

    monkeypatch.setattr(ps, "render_series", spy)
    ps.plot_var_series(
        {"good": str(good), "absent": str(tmp_path / "absent")},
        str(tmp_path / "fig"),
        "pr",
        "rx1day",
        scale=86400.0,
    )
    assert list(captured) == ["good"]
    assert captured["good"][1][0] == pytest.approx(172800.0)


def test_plot_series_unknown_reference(tmp_path: pathlib.Path) -> None:
    """A reference that is neither a label nor a path is an error."""
    with pytest.raises(KeyError, match="reference 'RCM'"):
        ps.plot_var_series(
            {"UNet": str(tmp_path)}, str(tmp_path), "tasmax", "mean", reference="RCM"
        )
    with pytest.raises(ValueError, match="reference_path= given without"):
        ps.plot_var_series(
            {"UNet": str(tmp_path)},
            str(tmp_path),
            "tasmax",
            "mean",
            reference_path=str(tmp_path),
        )


def test_render_series_colour_budget(tmp_path: pathlib.Path) -> None:
    """More models than categorical colours is refused rather than cycled."""
    per_source = {f"m{i}": _series(HIST) for i in range(len(ps.SERIES_COLORS) + 1)}
    with pytest.raises(ValueError, match="categorical colours"):
        ps.render_series(per_source, "tasmax", "mean", str(tmp_path / "fig"))


# ------------------------------------------------------------ ensemble spread --


def test_member_spread_minmax_and_std() -> None:
    """Mean line plus a band, from several members' series."""
    years = np.array([2041, 2042])
    members = [
        (years, np.array([1.0, 2.0])),
        (years, np.array([3.0, 4.0])),
        (years, np.array([2.0, 3.0])),
    ]
    y, mean, lo, hi = ps.member_spread(members, "minmax")
    assert list(y) == [2041, 2042]
    assert list(mean) == [2.0, 3.0]
    assert list(lo) == [1.0, 2.0] and list(hi) == [3.0, 4.0]

    _, mean, lo, hi = ps.member_spread(members, "std")
    sd = np.std([1.0, 3.0, 2.0])
    assert lo[0] == pytest.approx(2.0 - sd) and hi[0] == pytest.approx(2.0 + sd)


def test_member_spread_aligns_ragged_members() -> None:
    """Members covering different years are unioned, not truncated."""
    a = (np.array([2041, 2042]), np.array([1.0, 2.0]))
    b = (np.array([2042, 2043]), np.array([4.0, 6.0]))
    y, mean, lo, hi = ps.member_spread([a, b], "minmax")
    assert list(y) == [2041, 2042, 2043]
    assert mean[0] == 1.0 and mean[1] == 3.0 and mean[2] == 6.0
    assert lo[1] == 2.0 and hi[1] == 4.0


def test_member_spread_edge_cases() -> None:
    """One member collapses the band; nothing readable gives empty arrays."""
    years = np.array([2041, 2042])
    y, mean, lo, hi = ps.member_spread([(years, np.array([1.0, 2.0]))])
    assert list(lo) == list(hi) == list(mean)
    y, mean, lo, hi = ps.member_spread([])
    assert y.size == mean.size == lo.size == hi.size == 0
    with pytest.raises(ValueError, match="spread must be one of"):
        ps.member_spread([(years, np.array([1.0, 2.0]))], "iqr")


def test_plot_var_series_draws_a_member_band(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A label given several paths becomes a mean line with a shaded band."""
    years = np.arange(1981, 1985)
    monkeypatch.setattr(
        ps,
        "read_series",
        lambda d, index: (years, np.full(years.size, float(d[-1]))),
    )
    monkeypatch.setattr(ps.os.path, "isdir", lambda p: True)
    monkeypatch.setattr(ps, "metrics_dir", lambda base, var: str(base))

    seen = {}
    monkeypatch.setattr(
        ps,
        "render_series",
        lambda per_source, *a, **kw: seen.update(
            per_source=per_source, bands=kw.get("bands")
        )
        or "fig.png",
    )
    ps.plot_var_series(
        {"RESDIFF": ["m1", "m3"], "RCM": "r2"},
        str(tmp_path),
        "pr",
        "rx1day",
        reference="RCM",
        spread="minmax",
    )

    # the line is the member mean, the band their envelope
    assert list(seen["per_source"]["RESDIFF"][1]) == [2.0] * 4
    lo, hi = seen["bands"]["RESDIFF"][1], seen["bands"]["RESDIFF"][2]
    assert list(lo) == [1.0] * 4 and list(hi) == [3.0] * 4
    # a single-path label carries no band
    assert "RCM" not in seen["bands"]


def test_member_band_follows_the_anomaly_baseline(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a baseline the band is shifted by the same offset as the line."""
    years = np.arange(1981, 1985)
    monkeypatch.setattr(
        ps,
        "read_series",
        lambda d, index: (years, np.arange(4, dtype=float) + float(d[-1])),
    )
    monkeypatch.setattr(ps.os.path, "isdir", lambda p: True)
    monkeypatch.setattr(ps, "metrics_dir", lambda base, var: str(base))

    seen = {}
    monkeypatch.setattr(
        ps,
        "render_series",
        lambda per_source, *a, **kw: seen.update(
            per_source=per_source, bands=kw.get("bands")
        )
        or "fig.png",
    )
    ps.plot_var_series(
        {"RESDIFF": ["m0", "m2"]},
        str(tmp_path),
        "pr",
        "rx1day",
        baseline=(1981, 1984),
        spread="minmax",
    )

    line = seen["per_source"]["RESDIFF"][1]
    lo, hi = seen["bands"]["RESDIFF"][1], seen["bands"]["RESDIFF"][2]
    assert line.mean() == pytest.approx(0.0)  # anomalies about its own mean
    assert list(hi - lo) == pytest.approx([2.0] * 4)  # the band width is preserved


def test_read_series_spread_from_the_store(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ensemble-mean signal run stores its own band: no per-member runs."""
    years = np.array([2041, 2042])
    stored = {
        "rx1day_series": np.array([2.0, 3.0]),
        "rx1day_series_min": np.array([1.0, 2.0]),
        "rx1day_series_max": np.array([3.0, 4.0]),
        "rx1day_series_std": np.array([0.5, 0.5]),
    }
    monkeypatch.setattr(
        ps,
        "read_diagnostic",
        lambda d, diag: (years, stored[diag])
        if diag in stored
        else (np.array([], dtype=int), np.array([], dtype=float)),
    )

    y, lo, hi = ps.read_series_spread("d", "rx1day", "minmax")
    assert list(lo) == [1.0, 2.0] and list(hi) == [3.0, 4.0]

    y, lo, hi = ps.read_series_spread("d", "rx1day", "std")
    assert list(lo) == [1.5, 2.5] and list(hi) == [2.5, 3.5]

    # a single-member run stored no spread: empty, not an error
    monkeypatch.setattr(
        ps,
        "read_diagnostic",
        lambda d, diag: (years, stored["rx1day_series"])
        if diag == "rx1day_series"
        else (np.array([], dtype=int), np.array([], dtype=float)),
    )
    y, lo, hi = ps.read_series_spread("d", "rx1day", "minmax")
    assert y.size == lo.size == hi.size == 0


def test_plot_var_series_uses_the_stored_spread(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One path + spread= reads the band from that run, no member list needed."""
    years = np.arange(2041, 2045)
    monkeypatch.setattr(ps, "read_series", lambda d, i: (years, np.ones(4)))
    monkeypatch.setattr(
        ps,
        "read_series_spread",
        lambda d, i, sp: (years, np.zeros(4), np.full(4, 2.0)),
    )
    monkeypatch.setattr(ps.os.path, "isdir", lambda p: True)
    monkeypatch.setattr(ps, "metrics_dir", lambda base, var: str(base))
    seen = {}
    monkeypatch.setattr(
        ps,
        "render_series",
        lambda per_source, *a, **kw: seen.update(bands=kw.get("bands")) or "fig.png",
    )

    ps.plot_var_series(
        {"RESDIFF": "run"}, str(tmp_path), "pr", "rx1day", spread="minmax"
    )
    assert list(seen["bands"]["RESDIFF"][1]) == [0.0] * 4
    assert list(seen["bands"]["RESDIFF"][2]) == [2.0] * 4


# ------------------------------------------------------------------ variants --


def test_variant_shares_its_base_colour_and_is_dashed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A (perfect, imperfect) pair is one colour, solid then dashed."""
    years = np.arange(2041, 2045)
    drawn = []
    real_plot = ps.plt.Axes.plot

    def spy(self, *args, **kw):
        if "label" in kw:
            drawn.append((kw["label"], kw.get("color"), kw.get("ls")))
        return real_plot(self, *args, **kw)

    monkeypatch.setattr(ps.plt.Axes, "plot", spy)
    per = {
        lab: (years, np.arange(4.0))
        for lab in (
            "RCM",
            "A (perfect)",
            "A (imperfect)",
            "B (perfect)",
            "B (imperfect)",
        )
    }

    ps.render_series(
        per,
        "tasmax",
        "mean",
        str(tmp_path),
        reference="RCM",
        variant_of={"A (imperfect)": "A (perfect)", "B (imperfect)": "B (perfect)"},
    )

    style = {lab: (c, ls) for lab, c, ls in drawn}
    assert style["A (perfect)"][0] == style["A (imperfect)"][0]  # same colour
    assert style["A (perfect)"][1] == "-" and style["A (imperfect)"][1] == "--"
    assert style["B (perfect)"][0] != style["A (perfect)"][0]  # different models differ
    assert style["RCM"][0] == "black"


def test_variants_do_not_consume_the_colour_budget(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pairing lets twice as many lines fit the categorical palette."""
    years = np.arange(2041, 2043)
    n = len(ps.SERIES_COLORS)
    per = {f"M{i} (perfect)": (years, np.ones(2)) for i in range(n)}
    per.update({f"M{i} (imperfect)": (years, np.ones(2)) for i in range(n)})
    variant_of = {f"M{i} (imperfect)": f"M{i} (perfect)" for i in range(n)}

    # 2n lines, n colours: fine when paired ...
    assert ps.render_series(
        per,
        "tasmax",
        "mean",
        str(tmp_path),
        variant_of=variant_of,
        filename="paired.png",
    )
    # ... and rejected when not
    with pytest.raises(ValueError, match="colour-carrying sources exceed"):
        ps.render_series(per, "tasmax", "mean", str(tmp_path), filename="unpaired.png")


def test_variant_without_its_base_gets_its_own_colour(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the base run is missing, the variant is still drawn distinctly."""
    years = np.arange(2041, 2043)
    drawn = []
    real_plot = ps.plt.Axes.plot
    monkeypatch.setattr(
        ps.plt.Axes,
        "plot",
        lambda self, *a, **kw: (
            drawn.append((kw.get("label"), kw.get("color")))
            or real_plot(self, *a, **kw)
        ),
    )
    per = {"A (imperfect)": (years, np.ones(2)), "B (perfect)": (years, np.ones(2))}
    ps.render_series(
        per,
        "tasmax",
        "mean",
        str(tmp_path),
        variant_of={"A (imperfect)": "A (perfect)"},
    )
    colors = {lab: c for lab, c in drawn if lab}
    assert colors["A (imperfect)"] is not None
    assert colors["A (imperfect)"] != colors["B (perfect)"]
