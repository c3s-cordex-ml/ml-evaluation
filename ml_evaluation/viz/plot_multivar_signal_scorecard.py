"""
Climate-change-signal *scorecard* for ML RCM emulators.

A model x variable grid of six-wedge hexagons summarising the **climate-change
signal** (future minus historical) each emulator produces, from the
``metrics_signal_float.zarr`` bundles the ``ml-eval signal`` command writes.

Unlike the evaluation scorecard in ``plot_hexagon`` (six *metrics* per hexagon,
coloured by error improvement), here each hexagon holds **three indices**, one
per sector (a sector is a pair of wedges, so the hexagon still reads as six):

    non-precip : mean, P01, P99      (units: K, %, Pa, ...)
    precip (pr): R01, SDII, RX1day

    sector layout (wedge ids per ``_wedge``: 0=top, counter-clockwise) -- each
    sector is a contiguous 120 degree third of the hexagon:
        1st index : "top"    = wedges 5 (upper-right) + 0 (top)
        2nd index : "left"   = wedges 1 (upper-left)  + 2 (lower-left)
        3rd index : "bottom" = wedges 3 (bottom)      + 4 (lower-right)

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
which stores just ``delta``/``reldelta`` (no cross-source bias).

Locating the data
-----------------
As in ``plot_hexagon``, rows come from an explicit ``{label: path}`` spec and one
variable's stores are read from ``<path>/<var>/``. One entry = one row, so a
model evaluated under several drivings is simply listed once per driving, under
whatever labels you want on the rows.

Example
-------
    from ml_evaluation.viz.plot_multivar_signal_scorecard import plot_multivar_signal_scorecard
    plot_multivar_signal_scorecard(
        {
            "RCM-truth":             "/.../RCM-TRUTH_EC-Earth3_2041-2060/annual",
            "UNet-single perfect":   "/.../SINGV-RCM_..._perfect/annual",
            "UNet-single imperfect": "/.../SINGV-RCM_..._imperfect/annual",
            "ResDiff-v3 perfect":    "/.../RESDIFF-RCM_..._perfect/annual",
        },
        "/.../FIGURES/signal",
        ["tas", "tasmax", "pr"],
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

from ml_evaluation.viz.plot_hexagon import metrics_dir, normalize_paths

EPS = 1e-12
SIGNAL_STORE = "metrics_signal_float.zarr"

# Index order per variable and, for each, the two *adjacent* wedges its sector
# occupies. Wedge ids follow ``_wedge``: 0 top, 1 upper-left, 2 lower-left,
# 3 bottom, 4 lower-right, 5 upper-right -- so each sector is a contiguous 120
# degree third of the hexagon, and its bisector is the angle below.
_WEDGE_PAIR = {"top": (5, 0), "left": (1, 2), "bottom": (3, 4)}
_POS_LABEL_ANGLE = {"top": 60.0, "left": 180.0, "bottom": 300.0}
SIGNAL_INDICES_PR = [("sdii", "top"), ("r01", "left"), ("rx1day", "bottom")]
SIGNAL_INDICES_TEMP = [("mean", "top"), ("p01", "left"), ("p99", "bottom")]
INDEX_LABEL = {
    "mean": "mean",
    "p01": "P01",
    "p99": "P99",
    "r01": "R01",
    "sdii": "SDII",
    "rx1day": "RX1day",
}


def signal_indices(var: str) -> list[tuple[str, str]]:
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


def _read_signal_floats(
    metrics_dir: str, keys: list[str], kind: str
) -> dict[str, float]:
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
                    out[k] = float(
                        ds["value"].sel(diagnostic=f"{kind}_{k}_float").values
                    )
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


def _sector(ax, cx, cy, pos, radius, facecolor, edgecolor="white", lw=0.6):
    """Draw one 120-degree sector (the two adjacent wedges of ``pos``).

    Drawn as a single polygon rather than two ``_wedge`` calls so no divider
    line runs through the middle of the sector -- that line would sit exactly
    where the sector's number label goes.
    """
    from matplotlib.patches import Polygon

    i0, i1 = _WEDGE_PAIR[pos]
    angles = [90 + i0 * 60 - 30, 90 + i0 * 60 + 30, 90 + i1 * 60 + 30]
    pts = [(cx, cy)] + [
        (cx + radius * np.cos(np.radians(a)), cy + radius * np.sin(np.radians(a)))
        for a in angles
    ]
    ax.add_patch(
        Polygon(pts, closed=True, facecolor=facecolor, edgecolor=edgecolor, lw=lw)
    )


def _sector_text_xy(
    cx: float, cy: float, pos: str, radius: float
) -> tuple[float, float]:
    """Anchor for a sector's number label: midway out along its bisector."""
    ang = np.radians(_POS_LABEL_ANGLE[pos])
    return cx + 0.55 * radius * np.cos(ang), cy + 0.55 * radius * np.sin(ang)


def _draw_legend_hex(ax, cx: float, cy: float, radius: float, var: str) -> None:
    """Small template hexagon showing which sector holds which index."""
    for key, pos in signal_indices(var):
        _sector(ax, cx, cy, pos, radius, "0.92", edgecolor="0.4", lw=0.7)
        # index label just outside the sector, along its bisector
        lab_ang = np.radians(_POS_LABEL_ANGLE[pos])
        ax.text(
            cx + 1.45 * radius * np.cos(lab_ang),
            cy + 1.45 * radius * np.sin(lab_ang),
            INDEX_LABEL[key],
            ha="center",
            va="center",
            fontsize=7.5,
        )


def plot_multivar_signal_scorecard(
    paths,
    out: str,
    variables: list[str],
    *,
    reference: str = "RCM-truth",
    reference_path: str | None = None,
    season: str = "annual",
    units: dict[str, str] | None = None,
    title: str | None = None,
    filename: str | None = None,
    cmap: str = "RdBu_r",
    annotate: bool = True,
) -> str:
    """Render the climate-change-signal scorecard.

    Rows are the reference followed by the other labels of ``paths``; columns are
    ``variables``. Each cell is a hexagon of three index sectors coloured by the
    per-(variable, index) normalised signal and, when ``annotate``, annotated
    with the actual change.

    Parameters
    ----------
    paths : mapping or sequence
        ``{label: path}`` of the signal runs to plot (see
        ``plot_hexagon.normalize_paths``); one variable's stores are read from
        ``<path>/<var>/``. One entry = one row, in the given order -- list a
        model once per driving if you want both. If it contains the
        ``reference`` label, that entry supplies the reference row.
    out : str
        Output directory for the PNG.
    variables : list of str
        Variables (columns).
    reference : str, default "RCM-truth"
        Label of the reference row -- the source whose signal normalises the
        colours (the RCM-truth).
    reference_path : str, optional
        Base directory of the reference's signal run. Required only when
        ``reference`` is not a label of ``paths``; otherwise it overrides it.
    season : str, default "annual"
        Label only -- title and output file name.
    units : dict, optional
        ``{var: unit_label}`` shown under each column header.
    title : str, optional
        Figure title (default derived from the season).
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

    by_label = normalize_paths(paths)
    units = units or {}

    model_rows = [m for m in by_label if m != reference]
    ref_base = reference_path or by_label.get(reference)
    if ref_base is None:
        raise KeyError(
            f"reference '{reference}' is neither a label of paths "
            f"({', '.join(by_label)}) nor given via reference_path=."
        )

    # Signal quantity per variable: absolute delta (temperature) or relative
    # reldelta (precip). Each source's run stores just {kind}_{key}_float.
    # signal[(row, var)] = {key: that row's signal}; truth[var] from the
    # reference's own signal run.
    signal: dict[tuple[str, str], dict[str, float]] = {}
    truth: dict[str, dict[str, float]] = {}
    for var in variables:
        keys = [k for k, _ in signal_indices(var)]
        kind = _signal_kind(var)
        for m in model_rows:
            d = metrics_dir(by_label[m], var)
            if os.path.isdir(d):
                signal[(m, var)] = _read_signal_floats(d, keys, kind)
        rd = metrics_dir(ref_base, var)
        if os.path.isdir(rd):
            t = _read_signal_floats(rd, keys, kind)
            if any(np.isfinite(v) for v in t.values()):
                truth[var] = t

    # Per-(var, index) normalisation constant = |RCM-truth signal|, so the truth
    # row saturates at +/-1 and each model reads as its signed fraction of truth.
    truth_norm: dict[tuple[str, str], float] = {}
    for var in variables:
        for key, _ in signal_indices(var):
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
            for key, pos in signal_indices(var):
                m = truth_norm.get((var, key), np.nan)
                if is_ref:
                    val = truth.get(var, {}).get(key, np.nan)
                else:
                    val = signal.get((row, var), {}).get(key, np.nan)
                if np.isfinite(val) and np.isfinite(m) and m > EPS:
                    fc = cmap_o(norm(np.clip(val / m, -1.0, 1.0)))
                else:
                    fc = nan_color
                _sector(ax, cx, cy, pos, R, fc)  # one index = one sector
                if annotate and np.isfinite(val):
                    tx, ty = _sector_text_xy(cx, cy, pos, R)
                    ax.text(
                        tx,
                        ty,
                        _fmt(val, pct),
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=_text_color(fc) if fc != nan_color else "0.4",
                    )

    ax.set_xlim(-0.7, ncols - 0.3)
    ax.set_ylim(-(nrows - 1) - 2.7, 0.9)  # bottom band holds the legend
    ax.set_aspect("equal")
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(
        [f"{v.upper()}\n{units[v]}" if v in units else v.upper() for v in variables]
    )
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([-r for r in range(nrows)])
    ax.set_yticklabels(rows)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # legend template hexagons (non-precip + precip) below the grid, placed
    # inside the axes whatever the number of columns
    y_leg = -(nrows - 1) - 1.55
    x_lo, x_hi = -0.7, ncols - 0.3
    xs = [x_lo + (x_hi - x_lo) * f for f in (0.3, 0.7)]
    _draw_legend_hex(ax, xs[0], y_leg, 0.28, "tas")
    ax.text(xs[0], y_leg - 0.60, "non-precip", ha="center", fontsize=7.5)
    _draw_legend_hex(ax, xs[1], y_leg, 0.28, "pr")
    ax.text(xs[1], y_leg - 0.60, "pr", ha="center", fontsize=7.5)
    ax.text(
        0.5 * (x_lo + x_hi),
        y_leg - 1.0,
        "one sector per index  ·  " f"colours normalised by the {reference} row",
        ha="center",
        va="center",
        fontsize=7.5,
    )

    sm = ScalarMappable(norm=norm, cmap=cmap_o)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label(f"CC signal / |{reference} signal|  (clipped ±1)")

    head = title or f"CC-signal scorecard — {season}"
    ax.set_title(
        f"{head}\n"
        f"colour = signal / |{reference} signal| per index (red = increase, blue = decrease)  ·  "
        "number = change (pr: % vs present; others: absolute)\n"
        "pr sectors = R01/SDII/RX1day, others = P01/mean/P99",
        fontsize=10,
        pad=16,
    )

    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, filename or f"signal_scorecard_{season}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[signal-scorecard] {path}  (rows: {', '.join(rows)})")
    return path
