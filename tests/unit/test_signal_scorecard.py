from __future__ import annotations

import pathlib

import numpy as np
import pytest

import ml_evaluation.viz.plot_signal_scorecard as ss


def test_signal_scorecard_indices() -> None:
    """The index/wedge layout differs for precip vs. other variables."""
    assert ss.signal_scorecard_indices("tasmax") == [
        ("mean", "left"), ("p01", "mid"), ("p99", "right"),
    ]
    assert ss.signal_scorecard_indices("pr") == [
        ("sdii", "left"), ("r01", "mid"), ("rx1day", "right"),
    ]
    # Every wedge position maps to a distinct (perfect, imperfect) wedge pair.
    used = [w for pair in ss._WEDGE_PAIR.values() for w in pair]
    assert sorted(used) == list(range(6))


def test_plot_signal_scorecard_renders(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scorecard renders a PNG from signal-float scalars.

    The zarr reader and directory resolution are monkeypatched with synthetic
    signals so the drawing path (wedges, per-index normalisation, annotations,
    legend hexagons) is exercised without on-disk stores.
    """
    variables = ["tasmax", "pr"]
    models = ["UNet-single", "ResDiff-v3"]
    scale = {"tasmax": 2.4, "pr": 6.0}

    def fake_reader(metrics_dir: str, keys: list[str], kind: str) -> dict[str, float]:
        head, var, drv = metrics_dir.split("::")
        model = head.split("/")[-1]
        base = scale[var]
        is_truth = model == "RCM-truth"
        out = {}
        for j, k in enumerate(keys):
            truth = base * (1.0 + 0.2 * j)
            out[k] = truth if is_truth else truth * (0.9 if drv == "P" else 0.6)
        return out

    monkeypatch.setattr(
        ss, "_metrics_dir",
        lambda root, pat, model, var, case, season, cp:
        f"{root}/{model}::{var}::" + ("I" if "imperfect" in case else "P"),
    )
    monkeypatch.setattr(ss, "_resolve_checkpoint", lambda *a, **k: "")
    monkeypatch.setattr(ss.os.path, "isdir", lambda p: True)
    monkeypatch.setattr(ss, "_read_signal_floats", fake_reader)

    path = ss.plot_signal_scorecard(
        str(tmp_path / "root"),
        str(tmp_path / "fig"),
        variables,
        models,
        case_perfect="EC-Earth3_2041-2060_perfect",
        case_imperfect="EC-Earth3_2041-2060_imperfect",
        units={"tasmax": "K", "pr": "mm/day"},
    )
    assert pathlib.Path(path).is_file()
    assert pathlib.Path(path).stat().st_size > 0


def test_plot_signal_scorecard_perfect_only(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting the imperfect case still renders (imperfect wedges blank)."""
    monkeypatch.setattr(
        ss, "_metrics_dir",
        lambda root, pat, model, var, case, season, cp: f"{root}/{model}::{var}::P",
    )
    monkeypatch.setattr(ss, "_resolve_checkpoint", lambda *a, **k: "")
    monkeypatch.setattr(ss.os.path, "isdir", lambda p: True)
    monkeypatch.setattr(
        ss, "_read_signal_floats",
        lambda d, keys, which: {k: 1.0 for k in keys},
    )
    path = ss.plot_signal_scorecard(
        str(tmp_path / "root"), str(tmp_path / "fig"), ["tasmax"], ["UNet-single"],
        case_perfect="c_perfect",
    )
    assert pathlib.Path(path).is_file()
