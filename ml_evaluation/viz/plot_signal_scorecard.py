"""
Climate-change-signal *scorecard* for ML RCM emulators.

A model x variable grid of six-wedge hexagons summarising the **climate-change
signal** (future minus historical) each emulator produces, from the
``metrics_signal_float.zarr`` bundles the ``ml-eval signal`` command writes.

Unlike the evaluation scorecard in ``plot_hexagon`` (six *metrics* per hexagon,
coloured by error improvement), here each hexagon holds **three indices, each
split into two driving wedges**:

    non-precip : mean, P01, P99      (units: K, %, Pa, ...)
    precip (pr): R01, SDII, RX1day

    wedge layout (see ``_wedge`` numbering 0=top .. counter-clockwise):
        left  index : upper-left(1)=perfect / lower-left(2)=imperfect
        mid   index : top(0)=perfect        / bottom(3)=imperfect
        right index : upper-right(5)=perfect / lower-right(4)=imperfect
    -> the upper half of every hexagon is *perfect* driving, the lower half
       *imperfect*; the three index columns are left / mid / right.

The signal quantity is variable-dependent (see :func:`_signal_kind`): the
absolute ``delta`` for temperature-like variables, the relative ``reldelta`` for
precip -- used for BOTH the colour and the printed number so they always agree.

* colour  = signal / |RCM-truth signal| per (variable, index), clipped to
            [-1, +1] and diverging (red = increase, blue = decrease). The truth
            row therefore saturates at +/-1 and each model reads as its signed
            fraction of the truth signal.
* number  = the change: precip as a percentage vs. the present climate
            (``reldelta``), other variables as the absolute ``delta``.

Every source (the RCM-truth and each model) has its own ``ml-eval signal`` run,
which stores just ``delta``/``reldelta`` (no cross-source bias). The reference
row is read from the RCM-truth's run -- driving-independent, so both wedges of
each index show the same value ("uniform, no driving"); model rows are read from
their own perfect/imperfect runs.

Metrics directories are located exactly like ``plot_hexagon.plot_scorecard``
(``root`` + ``pattern``, defaulting per ``layout``; the d4p ``{checkpoint}``
segment auto-resolves). Two ``case`` tags are supplied for the models (one per
driving) plus ``reference_case`` for the truth's run.

Example
-------
    from ml_evaluation.viz.plot_signal_scorecard import plot_signal_scorecard
    plot_signal_scorecard(
        "/.../EMULATORS",
        "/.../FIGURES/signal",
        ["tas", "tasmax", "pr"],
        ["UNet-single", "UNet-MultiVar-v3", "ResDiff-v3", "DownViT"],
        case_perfect="EC-Earth3_2041-2060_perfect",
        case_imperfect="EC-Earth3_2041-2060_imperfect",
        units={"tas": "K", "tasmax": "K", "pr": "mm d$^{-1}$"},
    )
"""
from __future__ import annotations

import os

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_evaluation.viz.plot_hexagon import (
    _metrics_dir,
    _resolve_checkpoint,
    _resolve_pattern,
    _wedge,
)

EPS = 1e-12
SIGNAL_STORE = "metrics_signal_float.zarr"

# Index order per variable and, for each, which (perfect, imperfect) wedges it
# occupies. Wedge ids follow ``_wedge``: 0 top, 1 upper-left, 2 lower-left,
# 3 bottom, 4 lower-right, 5 upper-right.
_WEDGE_PAIR = {"left": (1, 2), "mid": (0, 3), "right": (5, 4)}
# Where to place each index's label around the legend hexagon (degrees, math
# convention). The pair's wedge ids are non-contiguous in angle, so this is set
# explicitly rather than averaged: left sector -> lower-left, mid -> top,
# right sector -> lower-right (mirrors the target legend layout).
_POS_LABEL_ANGLE = {"left": 205.0, "mid": 90.0, "right": 335.0}
SIGNAL_INDICES_PR = [("sdii", "left"), ("r01", "mid"), ("rx1day", "right")]
SIGNAL_INDICES_TEMP = [("mean", "left"), ("p01", "mid"), ("p99", "right")]
INDEX_LABEL = {
    "mean": "mean", "p01": "P01", "p99": "P99",
    "r01": "R01", "sdii": "SDII", "rx1day": "RX1day",
}


def signal_scorecard_indices(var: str) -> list[tuple[str, str]]:
    """The ``(index_key, wedge_position)`` list for ``var``."""
    return SIGNAL_INDICES_PR if var == "pr" else SIGNAL_INDICES_TEMP


def _signal_kind(var: str) -> str:
    """Which stored signal is the physically-meaningful one for ``var``.

    Precip uses the relative (%) signal (``reldelta``); everything else the
    absolute signal (``delta``). This selects both the colour quantity and the
    printed number so the two always agree.
    """
    return "reldelta" if var == "pr" else "delta"


def _is_pct(var: str) -> bool:
    """True when the printed number is a percentage (precip relative signal)."""
    return var == "pr"


def _read_signal_floats(metrics_dir: str, keys: list[str], kind: str) -> dict[str, float]:
    """Read ``{kind}_{key}_float`` scalars from a signal-float zarr bundle.

    Parameters
    ----------
    metrics_dir : str
        Directory holding ``metrics_signal_float.zarr`` (one source's signal).
    keys : list of str
        Index keys to read (e.g. ``["mean", "p01", "p99"]``).
    kind : {"delta", "reldelta"}
        Which signal to read: the absolute or the relative delta.

    Returns
    -------
    dict[str, float]
        ``{key: scalar}``; missing keys/stores come back as NaN.
    """
    path = os.path.join(metrics_dir, SIGNAL_STORE)
    if not os.path.isdir(path):
        return {k: np.nan for k in keys}
    out: dict[str, float] = {}
    try:
        with xr.open_zarr(path) as ds:
            for k in keys:
                try:
                    out[k] = float(ds["value"].sel(diagnostic=f"{kind}_{k}_float").values)
                except (KeyError, ValueError):
                    out[k] = np.nan
    except (FileNotFoundError, OSError):
        return {k: np.nan for k in keys}
    return out


def _text_color(rgba: tuple[float, float, float, float]) -> str:
    """Black or white, whichever contrasts with the wedge fill."""
    r, g, b = rgba[:3]
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.6 else "white"


def _fmt(v: float, pct: bool = False) -> str:
    """Signed, magnitude-aware short label for the change (``pct`` -> percent)."""
    if pct:
        return f"{v * 100:+.0f}%"
    a = abs(v)
    if a >= 100:
        return f"{v:+.0f}"
    if a >= 10:
        return f"{v:+.1f}"
    return f"{v:+.2f}"


def _wedge_text_xy(cx: float, cy: float, i: int, radius: float) -> tuple[float, float]:
    """Centroid-ish anchor for wedge ``i``'s number label."""
    ang = np.radians(90 + i * 60)
    return cx + 0.55 * radius * np.cos(ang), cy + 0.55 * radius * np.sin(ang)


def _draw_legend_hex(ax, cx: float, cy: float, radius: float, var: str) -> None:
    """Small template hexagon showing the index/driving wedge layout."""
    for key, pos in signal_scorecard_indices(var):
        wp, wi = _WEDGE_PAIR[pos]
        for wid, drv in ((wp, "P"), (wi, "I")):
            _wedge(ax, cx, cy, wid, radius, "0.92", edgecolor="0.4", lw=0.7)
            tx, ty = _wedge_text_xy(cx, cy, wid, radius)
            ax.text(tx, ty, drv, ha="center", va="center", fontsize=7, color="0.3")
        # index label just outside the sector, at an explicit per-position angle
        lab_ang = np.radians(_POS_LABEL_ANGLE[pos])
        ax.text(cx + 1.55 * radius * np.cos(lab_ang),
                cy + 1.55 * radius * np.sin(lab_ang),
                INDEX_LABEL[key], ha="center", va="center", fontsize=7.5)


def plot_signal_scorecard(
    root: str,
    out: str,
    variables: list[str],
    models: list[str],
    *,
    case_perfect: str,
    case_imperfect: str | None = None,
    reference: str = "RCM-truth",
    reference_case: str | None = None,
    season: str = "annual",
    layout: str = "d4p",
    pattern: str | None = None,
    checkpoint=None,
    units: dict[str, str] | None = None,
    title: str | None = None,
    filename: str | None = None,
    cmap: str = "RdBu_r",
    annotate: bool = True,
) -> str:
    """Render the climate-change-signal scorecard.

    Rows are the RCM-truth reference followed by ``models``; columns are
    ``variables``. Each cell is a six-wedge hexagon (three indices x two driving
    wedges) coloured by the per-(variable, index) normalised signal and, when
    ``annotate``, annotated with the actual change.

    Parameters
    ----------
    root : str
        Base directory the ``pattern`` is relative to.
    out : str
        Output directory for the PNG.
    variables : list of str
        Variables (columns).
    models : list of str
        Emulator names (rows, below the reference row).
    case_perfect : str
        Case tag for perfect driving (``{case}`` in the pattern).
    case_imperfect : str, optional
        Case tag for imperfect driving. If None, only the perfect wedge of each
        index is filled and the imperfect wedge is left blank.
    reference : str, default "RCM-truth"
        Name of the reference row -- a source with its own ``ml-eval signal`` run
        (the RCM-truth). Its signal is read from that run, so it is driving-
        independent and drawn as a uniform row.
    reference_case : str, optional
        Case tag for the reference's signal run. Defaults to ``case_perfect``
        (the truth signal does not depend on the emulator driving).
    season : str, default "annual"
        Season tag (``{season}`` in the pattern).
    layout, pattern, checkpoint
        Directory resolution, identical to ``plot_hexagon.plot_scorecard``.
    units : dict, optional
        ``{var: unit_label}`` shown under each column header.
    title : str, optional
        Figure title (default derived from the cases/season).
    filename : str, optional
        Output filename (default ``signal_scorecard_<season>.png``).
    cmap : str, default "RdBu_r"
        Diverging colormap; red = increase, blue = decrease.
    annotate : bool, default True
        Whether to print the actual change in each wedge.

    Returns
    -------
    str
        Path of the written PNG.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    pattern = _resolve_pattern(layout, pattern)
    if not models:
        raise ValueError("models must be a non-empty list of emulator names to plot.")
    units = units or {}
    drivings = [("P", case_perfect)]
    if case_imperfect is not None:
        drivings.append(("I", case_imperfect))

    model_rows = [m for m in dict.fromkeys(models) if m != reference]
    ref_case = reference_case or case_perfect
    cp_by_model = {
        m: _resolve_checkpoint(root, pattern, m, checkpoint)
        for m in model_rows + [reference]
    }

    # Signal quantity per variable: absolute delta (temperature) or relative
    # reldelta (precip). Each source's run stores just {kind}_{key}_float.
    # signal[(model, var, driving)] = {key: model signal}; truth[var] from the
    # reference's own (driving-independent) signal run.
    signal: dict[tuple[str, str, str], dict[str, float]] = {}
    truth: dict[str, dict[str, float]] = {}
    for var in variables:
        keys = [k for k, _ in signal_scorecard_indices(var)]
        kind = _signal_kind(var)
        for m in model_rows:
            for drv, case in drivings:
                d = _metrics_dir(root, pattern, m, var, case, season, cp_by_model[m])
                if d and os.path.isdir(d):
                    signal[(m, var, drv)] = _read_signal_floats(d, keys, kind)
        rd = _metrics_dir(root, pattern, reference, var, ref_case, season,
                          cp_by_model[reference])
        if rd and os.path.isdir(rd):
            t = _read_signal_floats(rd, keys, kind)
            if any(np.isfinite(v) for v in t.values()):
                truth[var] = t

    # Per-(var, index) normalisation constant = |RCM-truth signal|, so the truth
    # row saturates at +/-1 and each model reads as its signed fraction of truth.
    truth_norm: dict[tuple[str, str], float] = {}
    for var in variables:
        for key, _ in signal_scorecard_indices(var):
            tv = truth.get(var, {}).get(key, np.nan)
            truth_norm[(var, key)] = abs(tv) if np.isfinite(tv) else np.nan

    rows = [reference] + model_rows
    ncols, nrows = len(variables), len(rows)
    norm = Normalize(-1.0, 1.0)
    cmap_o = plt.get_cmap(cmap)
    nan_color = "0.9"
    R = 0.46

    fig, ax = plt.subplots(figsize=(1.7 * ncols + 3.0, 1.35 * nrows + 3.2))
    for r, row in enumerate(rows):
        is_ref = row == reference
        for c, var in enumerate(variables):
            cx, cy = c, -r
            pct = _is_pct(var)
            for key, pos in signal_scorecard_indices(var):
                wp, wi = _WEDGE_PAIR[pos]
                m = truth_norm.get((var, key), np.nan)
                for wid, drv in ((wp, "P"), (wi, "I")):
                    if is_ref:
                        val = truth.get(var, {}).get(key, np.nan)
                    elif drv == "I" and case_imperfect is None:
                        val = np.nan
                    else:
                        val = signal.get((row, var, drv), {}).get(key, np.nan)
                    if np.isfinite(val) and np.isfinite(m) and m > EPS:
                        fc = cmap_o(norm(np.clip(val / m, -1.0, 1.0)))
                    else:
                        fc = nan_color
                    _wedge(ax, cx, cy, wid, R, fc)
                    if annotate and np.isfinite(val):
                        tx, ty = _wedge_text_xy(cx, cy, wid, R)
                        ax.text(tx, ty, _fmt(val, pct), ha="center", va="center",
                                fontsize=6.0,
                                color=_text_color(fc) if fc != nan_color else "0.4")

    ax.set_xlim(-0.7, ncols - 0.3)
    ax.set_ylim(-(nrows - 1) - 1.9, 0.9)
    ax.set_aspect("equal")
    ax.set_xticks(range(ncols))
    ax.set_xticklabels([
        f"{v.upper()}\n{units[v]}" if v in units else v.upper() for v in variables
    ])
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([-r for r in range(nrows)])
    ax.set_yticklabels(rows)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # legend template hexagons (non-precip + precip) below the grid
    y_leg = -(nrows - 1) - 1.25
    _draw_legend_hex(ax, 0.35, y_leg, 0.34, "tas")
    ax.text(0.35, y_leg - 0.7, "non-precip variables", ha="center", fontsize=7.5)
    _draw_legend_hex(ax, 2.1, y_leg, 0.34, "pr")
    ax.text(2.1, y_leg - 0.7, "precipitation (pr)", ha="center", fontsize=7.5)
    ax.text(min(ncols - 1, 4.0), y_leg,
            "each index wedge is split\nP = perfect | I = imperfect predictors\n"
            f"({reference} row uniform, no driving)",
            ha="left", va="center", fontsize=7.5)

    sm = ScalarMappable(norm=norm, cmap=cmap_o)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label(f"CC signal / |{reference} signal|  (clipped ±1)")

    head = title or f"CC-signal scorecard — {case_perfect} — {season}"
    ax.set_title(
        f"{head}\n"
        f"colour = signal / |{reference} signal| per index (red = increase, blue = decrease)  ·  "
        "number = change (pr: % vs present; others: absolute)\n"
        "pr wedges = R01/SDII/RX1day, others = P01/mean/P99",
        fontsize=10, pad=16,
    )

    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, filename or f"signal_scorecard_{season}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[signal-scorecard] {path}  (rows: {', '.join(rows)})")
    return path
