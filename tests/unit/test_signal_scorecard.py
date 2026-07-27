from __future__ import annotations

import pathlib

import pytest

import ml_evaluation.viz.plot_signal_scorecard as ss


def test_signal_scorecard_indices() -> None:
    """The index/sector layout differs for precip vs. other variables."""
    assert ss.signal_scorecard_indices("tasmax") == [
        ("mean", "top"),
        ("p01", "left"),
        ("p99", "bottom"),
    ]
    assert ss.signal_scorecard_indices("pr") == [
        ("sdii", "top"),
        ("r01", "left"),
        ("rx1day", "bottom"),
    ]
    # The three sectors tile the hexagon: every wedge is used exactly once.
    used = [w for pair in ss._WEDGE_PAIR.values() for w in pair]
    assert sorted(used) == list(range(6))


def test_plot_signal_scorecard_renders(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scorecard renders a PNG from signal-float scalars.

    The zarr reader is monkeypatched with synthetic signals so the drawing path
    (sectors, per-index normalisation, annotations, legend hexagons) is
    exercised without on-disk stores. Paths are real directories, so the
    ``<path>/<var>`` resolution is exercised for real.
    """
    variables = ["tasmax", "pr"]
    scale = {"tasmax": 2.4, "pr": 6.0}
    rows = {
        "RCM-truth": tmp_path / "truth",
        "UNet-single perfect": tmp_path / "unet_p",
        "UNet-single imperfect": tmp_path / "unet_i",
    }
    for base in rows.values():
        for var in variables:
            (base / var).mkdir(parents=True)

    def fake_reader(metrics_dir: str, keys: list[str], kind: str) -> dict[str, float]:
        d = pathlib.Path(metrics_dir)
        var, model = d.name, d.parent.name
        base = scale[var]
        factor = {"truth": 1.0, "unet_p": 0.9, "unet_i": 0.6}[model]
        return {k: base * (1.0 + 0.2 * j) * factor for j, k in enumerate(keys)}

    monkeypatch.setattr(ss, "_read_signal_floats", fake_reader)

    path = ss.plot_signal_scorecard(
        {k: str(v) for k, v in rows.items()},
        str(tmp_path / "fig"),
        variables,
        units={"tasmax": "K", "pr": "mm/day"},
        case="EC-Earth3_2041-2060",
    )
    assert pathlib.Path(path).is_file()
    assert pathlib.Path(path).stat().st_size > 0


def test_plot_signal_scorecard_missing_rows(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row whose directory does not exist renders blank rather than raising."""
    truth = tmp_path / "truth"
    (truth / "tasmax").mkdir(parents=True)
    monkeypatch.setattr(
        ss,
        "_read_signal_floats",
        lambda d, keys, which: {k: 1.0 for k in keys},
    )
    path = ss.plot_signal_scorecard(
        {"RCM-truth": str(truth), "UNet-single": str(tmp_path / "absent")},
        str(tmp_path / "fig"),
        ["tasmax"],
    )
    assert pathlib.Path(path).is_file()


def test_plot_signal_scorecard_reference_must_be_locatable(
    tmp_path: pathlib.Path,
) -> None:
    """A reference that is neither in paths nor given explicitly is an error."""
    with pytest.raises(KeyError, match="RCM-truth"):
        ss.plot_signal_scorecard(
            {"UNet-single": str(tmp_path)},
            str(tmp_path / "fig"),
            ["tasmax"],
        )
