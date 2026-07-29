from __future__ import annotations

import pathlib

import numpy as np
import pytest

import ml_evaluation.viz.plot_series as ps

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

    raw = ps.plot_series(spec, str(tmp_path / "fig"), "tasmax", "mean",
                         reference="RCM truth", driver="GCM", units="K")
    assert pathlib.Path(raw).is_file()
    assert pathlib.Path(raw).name == "series_tasmax_mean_annual.png"

    anom = ps.plot_series(spec, str(tmp_path / "fig"), "tasmax", "mean",
                          reference="RCM truth", driver="GCM",
                          baseline=(1981, 2000), units="K")
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

    assert ps.plot_series(spec, out, "tasmax", "mean", baseline=(1981, 2000)) is None

    captured: dict = {}

    def spy(per_source, var, index, out_dir, **kwargs):
        captured["per_source"] = per_source
        captured["kwargs"] = kwargs
        return "fake.png"

    monkeypatch.setattr(ps, "render_series", spy)
    assert ps.plot_series(spec, out, "tasmax", "mean", baseline=(1981, 2000),
                          baseline_paths={"UNet-single": str(hist_dir)}) == "fake.png"

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
    ps.plot_series(
        {"good": str(good), "absent": str(tmp_path / "absent")},
        str(tmp_path / "fig"), "pr", "rx1day", scale=86400.0,
    )
    assert list(captured) == ["good"]
    assert captured["good"][1][0] == pytest.approx(172800.0)


def test_plot_series_unknown_reference(tmp_path: pathlib.Path) -> None:
    """A reference that is neither a label nor a path is an error."""
    with pytest.raises(KeyError, match="reference 'RCM'"):
        ps.plot_series({"UNet": str(tmp_path)}, str(tmp_path), "tasmax", "mean",
                       reference="RCM")
    with pytest.raises(ValueError, match="reference_path= given without"):
        ps.plot_series({"UNet": str(tmp_path)}, str(tmp_path), "tasmax", "mean",
                       reference_path=str(tmp_path))


def test_render_series_colour_budget(tmp_path: pathlib.Path) -> None:
    """More models than categorical colours is refused rather than cycled."""
    per_source = {f"m{i}": _series(HIST) for i in range(len(ps.SERIES_COLORS) + 1)}
    with pytest.raises(ValueError, match="categorical colours"):
        ps.render_series(per_source, "tasmax", "mean", str(tmp_path / "fig"))
