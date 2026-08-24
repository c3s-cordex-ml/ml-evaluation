from __future__ import annotations

import pathlib

import pytest

import ml_evaluation.viz.plot_multivar_signal_scorecard as ss


def test_signal_indices() -> None:
    """The index/sector layout differs for precip vs. other variables."""
    assert ss.signal_indices("tasmax") == [
        ("mean", "top"),
        ("p01", "left"),
        ("p99", "bottom"),
    ]
    assert ss.signal_indices("pr") == [
        ("sdii", "top"),
        ("r01", "left"),
        ("rx1day", "bottom"),
    ]
    # The three sectors tile the hexagon: every wedge is used exactly once.
    used = [w for pair in ss._WEDGE_PAIR.values() for w in pair]
    assert sorted(used) == list(range(6))


def test_plot_multivar_signal_scorecard_renders(
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

    path = ss.plot_multivar_signal_scorecard(
        {k: str(v) for k, v in rows.items()},
        str(tmp_path / "fig"),
        variables,
        units={"tasmax": "K", "pr": "mm/day"},
    )
    assert pathlib.Path(path).is_file()
    assert pathlib.Path(path).stat().st_size > 0


def test_plot_multivar_signal_scorecard_missing_rows(
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
    path = ss.plot_multivar_signal_scorecard(
        {"RCM-truth": str(truth), "UNet-single": str(tmp_path / "absent")},
        str(tmp_path / "fig"),
        ["tasmax"],
    )
    assert pathlib.Path(path).is_file()


def test_plot_multivar_signal_scorecard_reference_must_be_locatable(
    tmp_path: pathlib.Path,
) -> None:
    """A reference that is neither in paths nor given explicitly is an error."""
    with pytest.raises(KeyError, match="RCM-truth"):
        ss.plot_multivar_signal_scorecard(
            {"UNet-single": str(tmp_path)},
            str(tmp_path / "fig"),
            ["tasmax"],
        )


def test_ring_radii_are_equal_area() -> None:
    """Ring bands split a cell by area, not by radius.

    Equal thickness would give the inner case a quarter of the ink of the outer
    one, making "perfect" read as systematically fainter than "imperfect".
    """
    bands = ss.ring_radii(0.46, 2)
    assert bands[0][0] == 0.0
    assert bands[-1][1] == pytest.approx(0.46)
    areas = [r_out**2 - r_in**2 for r_in, r_out in bands]
    assert areas[0] == pytest.approx(areas[1])


def test_split_case_labels() -> None:
    """``MODEL (case)`` labels regroup into model rows keyed by case."""
    models, by_cell = ss.split_case_labels(
        ["A (perfect)", "A (imperfect)", "B (perfect)"], ["perfect", "imperfect"]
    )
    assert models == ["A", "B"]
    assert by_cell[("A", "imperfect")] == "A (imperfect)"
    assert ("B", "imperfect") not in by_cell  # missing ring renders blank
    with pytest.raises(ValueError, match="do not"):
        ss.split_case_labels(["A (perfect)", "B"], ["perfect", "imperfect"])


def test_plot_multivar_signal_scorecard_rings(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``rings=`` collapses the cases into one cell per model."""
    variables = ["tasmax", "pr"]
    rows = {
        "RCM-truth": tmp_path / "truth",
        "UNet-single (perfect)": tmp_path / "unet_p",
        "UNet-single (imperfect)": tmp_path / "unet_i",
        "ResDiff (perfect)": tmp_path / "rd_p",
        "ResDiff (imperfect)": tmp_path / "rd_i",
    }
    for base in rows.values():
        for var in variables:
            (base / var).mkdir(parents=True)

    def fake_reader(metrics_dir: str, keys: list[str], kind: str) -> dict[str, float]:
        return {k: 2.0 * (1.0 + 0.1 * j) for j, k in enumerate(keys)}

    monkeypatch.setattr(ss, "_read_signal_floats", fake_reader)

    path = ss.plot_multivar_signal_scorecard(
        {k: str(v) for k, v in rows.items()},
        str(tmp_path / "fig"),
        variables,
        units={"tasmax": "K", "pr": "%"},
        rings=["perfect", "imperfect"],
    )
    assert pathlib.Path(path).is_file()
    assert pathlib.Path(path).stat().st_size > 0


def test_plot_multivar_signal_radar(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The radar grid renders one cell per (model, variable).

    Covers the radial normalisation (reference -> unit triangle), a wrong-sign
    signal (clamped to the centre and marked) and a value past ``rlim``
    (clipped + caret).
    """
    variables = ["tasmax", "pr"]
    rows = {
        "RCM-truth": tmp_path / "truth",
        "UNet (perfect)": tmp_path / "unet_p",
        "UNet (imperfect)": tmp_path / "unet_i",
    }
    for base in rows.values():
        for var in variables:
            (base / var).mkdir(parents=True)

    def fake_reader(metrics_dir: str, keys: list[str], kind: str) -> dict[str, float]:
        model = pathlib.Path(metrics_dir).parent.name
        factor = {"truth": 1.0, "unet_p": 0.8, "unet_i": -0.4}[model]
        vals = {k: 2.0 * factor for k in keys}
        if model == "unet_p":
            vals[keys[0]] = 2.0 * 9.0  # past rlim -> clipped, caret drawn
        return vals

    monkeypatch.setattr(ss, "_read_signal_floats", fake_reader)

    path = ss.plot_multivar_signal_radar(
        {k: str(v) for k, v in rows.items()},
        str(tmp_path / "fig"),
        variables,
        cases=["perfect", "imperfect"],
        units={"tasmax": "K", "pr": "%"},
    )
    assert pathlib.Path(path).name == "signal_radar_annual.png"
    assert pathlib.Path(path).stat().st_size > 0


def test_plot_multivar_signal_radar_needs_case_labels(
    tmp_path: pathlib.Path,
) -> None:
    """Labels that do not name a case are rejected, not silently dropped."""
    rows = {"RCM-truth": tmp_path / "truth", "UNet": tmp_path / "unet"}
    for base in rows.values():
        (base / "tasmax").mkdir(parents=True)
    with pytest.raises(ValueError, match="do not"):
        ss.plot_multivar_signal_radar(
            {k: str(v) for k, v in rows.items()}, str(tmp_path / "f"), ["tasmax"]
        )
