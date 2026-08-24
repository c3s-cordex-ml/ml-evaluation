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

Rows vs rings
-------------
By default one source = one row, so a model evaluated under two boundary cases
occupies two rows and the RCM-truth a third. Passing ``rings=["perfect",
"imperfect"]`` instead stacks the cases as concentric *equal-area* bands inside
a single cell: rows become models, the plate loses a factor two in height, and
the ring contrast of a cell is the perfect-to-imperfect degradation that the
row layout makes the eye reconstruct. The truth is not given a ring -- it is
already the colour normalisation, so it would repeat one saturated value in
every cell of a column; it moves to the column header instead.

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


def _collect_signals(by_label, labels, ref_base, variables):
    """Read every source's signal once: ``({(label, var): {key: v}}, {var: ...})``.

    Shared by both entry points -- the scorecard and the radar plot the same
    numbers, and reading them twice would let the two drift apart.
    """
    signal: dict[tuple[str, str], dict[str, float]] = {}
    truth: dict[str, dict[str, float]] = {}
    for var in variables:
        keys = [k for k, _ in signal_indices(var)]
        kind = _signal_kind(var)
        for m in labels:
            d = metrics_dir(by_label[m], var)
            if os.path.isdir(d):
                signal[(m, var)] = _read_signal_floats(d, keys, kind)
        rd = metrics_dir(ref_base, var)
        if os.path.isdir(rd):
            t = _read_signal_floats(rd, keys, kind)
            if any(np.isfinite(v) for v in t.values()):
                truth[var] = t
    return signal, truth


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


def _sector(ax, cx, cy, pos, radius, facecolor, edgecolor="white", lw=0.6, r_in=0.0):
    """Draw one 120-degree sector (the two adjacent wedges of ``pos``).

    Drawn as a single polygon rather than two ``_wedge`` calls so no divider
    line runs through the middle of the sector -- that line would sit exactly
    where the sector's number label goes.

    With ``r_in > 0`` the sector becomes a ring segment between ``r_in`` and
    ``radius``: the same 120 degrees, hollowed out so several of them stack
    concentrically inside one cell (one per boundary case, see ``rings=``).
    """
    from matplotlib.patches import Polygon

    i0, i1 = _WEDGE_PAIR[pos]
    angles = [90 + i0 * 60 - 30, 90 + i0 * 60 + 30, 90 + i1 * 60 + 30]

    def rim(r):
        return [
            (cx + r * np.cos(np.radians(a)), cy + r * np.sin(np.radians(a)))
            for a in angles
        ]

    pts = rim(radius) + (list(reversed(rim(r_in))) if r_in > 0 else [(cx, cy)])
    ax.add_patch(
        Polygon(pts, closed=True, facecolor=facecolor, edgecolor=edgecolor, lw=lw)
    )


def ring_radii(radius: float, n: int) -> list[tuple[float, float]]:
    """``n`` concentric ``(r_in, r_out)`` bands of *equal area* out to ``radius``.

    Equal area, not equal thickness: splitting a radius in half gives the inner
    band a quarter of the ink of the outer one, which would make the first case
    read as systematically fainter than the second when the two carry the same
    kind of number.
    """
    edges = [radius * np.sqrt(k / n) for k in range(n + 1)]
    return list(zip(edges[:-1], edges[1:]))


def split_case_labels(
    labels: list[str], cases: list[str]
) -> tuple[list[str], dict[tuple[str, str], str]]:
    """Regroup ``"MODEL (case)"`` row labels into models x cases.

    Returns the model order (first appearance) and ``{(model, case): label}``,
    so the caller can look each cell's ring up in data already keyed by label.

    Raises
    ------
    ValueError
        If a label does not end in one of ``cases`` in parentheses.
    """
    models: list[str] = []
    by_cell: dict[tuple[str, str], str] = {}
    bad: list[str] = []
    for label in labels:
        for case in cases:
            suffix = f" ({case})"
            if label.endswith(suffix):
                model = label[: -len(suffix)]
                if model not in models:
                    models.append(model)
                by_cell[(model, case)] = label
                break
        else:
            bad.append(label)
    if bad:
        raise ValueError(
            f"rings={cases!r} needs every non-reference label to end in one of "
            f"{', '.join(f'({c})' for c in cases)}; these do not: {', '.join(bad)}."
        )
    return models, by_cell


def _sector_text_xy(
    cx: float, cy: float, pos: str, radius: float
) -> tuple[float, float]:
    """Anchor for a sector's number label: midway out along its bisector."""
    ang = np.radians(_POS_LABEL_ANGLE[pos])
    return cx + 0.55 * radius * np.cos(ang), cy + 0.55 * radius * np.sin(ang)


def _draw_legend_hex(
    ax, cx: float, cy: float, radius: float, var: str, cases: list[str] | None = None
) -> None:
    """Small template hexagon showing which sector holds which index.

    With ``cases`` the sectors are drawn as rings too, labelled inner-to-outer,
    so the reader can tell which band is which boundary case.
    """
    bands = ring_radii(radius, len(cases)) if cases else [(0.0, radius)]
    for key, pos in signal_indices(var):
        for r_in, r_out in bands:
            _sector(ax, cx, cy, pos, r_out, "0.92", edgecolor="0.4", lw=0.7, r_in=r_in)
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
    if cases:
        # case names below the template, clear of the bottom sector's own label
        ax.text(
            cx,
            cy - 2.05 * radius,
            "  ·  ".join(f"{i + 1}. {c}" for i, c in enumerate(cases)),
            ha="center",
            va="center",
            fontsize=7,
            color="0.3",
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
    rings: list[str] | None = None,
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
        Whether to print the actual change in each wedge. Forced off when
        ``rings`` is given -- one number per ring per sector does not fit.
    rings : list of str, optional
        Boundary cases to stack as concentric rings inside one cell instead of
        as separate rows, e.g. ``["perfect", "imperfect"]``. Every non-reference
        label must then read ``"MODEL (case)"``; rows become models, so the
        plate loses a factor ``len(rings)`` in height and the ring contrast of a
        cell *is* the perfect-to-imperfect degradation. The reference row is
        dropped -- it is already the colour normalisation, so drawing it would
        repeat one saturated value in every cell of a column; it appears in the
        legend template and in the column header instead.

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
    signal, truth = _collect_signals(by_label, model_rows, ref_base, variables)

    # Per-(var, index) normalisation constant = |RCM-truth signal|, so the truth
    # row saturates at +/-1 and each model reads as its signed fraction of truth.
    truth_norm: dict[tuple[str, str], float] = {}
    for var in variables:
        for key, _ in signal_indices(var):
            tv = truth.get(var, {}).get(key, np.nan)
            truth_norm[(var, key)] = abs(tv) if np.isfinite(tv) else np.nan

    # Rows are the sources, one cell per (row, variable). With ``rings`` the
    # cases collapse into concentric bands of one cell, so rows become models
    # and the reference -- already the colour normalisation -- is not drawn.
    if rings:
        rows, ring_label = split_case_labels(model_rows, rings)
        annotate = False
    else:
        rows, ring_label = [reference] + model_rows, {}
    ncols, nrows = len(variables), len(rows)
    norm = Normalize(-1.0, 1.0)
    cmap_o = plt.get_cmap(cmap)
    nan_color = "0.9"
    R = 0.46
    bands = ring_radii(R, len(rings)) if rings else [(0.0, R)]

    fig, ax = plt.subplots(figsize=(1.7 * ncols + 3.0, 1.35 * nrows + 3.2))
    for r, row in enumerate(rows):
        # the label(s) whose signal fills this row, inner ring first
        srcs = [ring_label.get((row, case)) for case in rings] if rings else [row]
        for c, var in enumerate(variables):
            cx, cy = c, -r
            pct = _is_pct(var)
            for key, pos in signal_indices(var):
                m = truth_norm.get((var, key), np.nan)
                for (r_in, r_out), src in zip(bands, srcs):
                    if src == reference:
                        val = truth.get(var, {}).get(key, np.nan)
                    else:
                        val = signal.get((src, var), {}).get(key, np.nan)
                    if np.isfinite(val) and np.isfinite(m) and m > EPS:
                        fc = cmap_o(norm(np.clip(val / m, -1.0, 1.0)))
                    else:
                        fc = nan_color
                    # one index = one sector, one case = one ring of it
                    _sector(ax, cx, cy, pos, r_out, fc, r_in=r_in)
                    if annotate and np.isfinite(val):
                        tx, ty = _sector_text_xy(cx, cy, pos, r_out)
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
    ax.set_ylim(-(nrows - 1) - (3.1 if rings else 2.9), 0.9)  # bottom band holds the legend
    ax.set_aspect("equal")
    ax.set_xticks(range(ncols))

    def _header(v: str) -> str:
        head = f"{v.upper()}\n{units[v]}" if v in units else v.upper()
        if not rings:
            return head
        # the reference is no longer a row, so its signal goes in the header,
        # in sector order, as the value every cell of the column is scaled by
        # one index per line: "mean +2.06" is narrow enough for the column,
        # where the three joined on one line overflow into the neighbour
        lines = [
            f"{INDEX_LABEL[k]} {_fmt(x, _is_pct(v))}"
            for (k, _), x in zip(
                signal_indices(v),
                (truth.get(v, {}).get(k, np.nan) for k, _ in signal_indices(v)),
            )
            if np.isfinite(x)
        ]
        return f"{head}\n{reference}\n" + "\n".join(lines) if lines else head

    ax.set_xticklabels([_header(v) for v in variables], fontsize=7.5)
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
    y_name = y_leg - (0.90 if rings else 0.60)
    _draw_legend_hex(ax, xs[0], y_leg, 0.28, "tas", rings)
    ax.text(xs[0], y_name, "non-precip", ha="center", fontsize=7.5)
    _draw_legend_hex(ax, xs[1], y_leg, 0.28, "pr", rings)
    ax.text(xs[1], y_name, "pr", ha="center", fontsize=7.5)
    ax.text(
        0.5 * (x_lo + x_hi),
        y_leg - (1.30 if rings else 1.15),
        (
            "one sector per index, one ring per case (inner to outer)  ·  "
            f"colours normalised by the {reference} signal in the header"
            if rings
            else "one sector per index  ·  "
            f"colours normalised by the {reference} row"
        ),
        ha="center",
        va="center",
        fontsize=7.5,
    )

    sm = ScalarMappable(norm=norm, cmap=cmap_o)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label(f"CC signal / |{reference} signal|  (clipped ±1)")

    head = title or f"CC-signal scorecard — {season}"
    second = (
        f"rings = {' (inner) → '.join(rings)} (outer); a cell that changes between "
        "rings is a signal that does not survive the switch of boundaries"
        if rings
        else "number = change (pr: % vs present; others: absolute)"
    )
    ax.set_title(
        f"{head}\n"
        f"colour = signal / |{reference} signal| per index (red = increase, blue = decrease)  ·  "
        f"{second}\n"
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


# Trace colours per boundary case, reused across every cell so the reader learns
# them once. Extra cases fall back to tab10.
SIGNAL_CASE_COLOR = {"perfect": "#1f77b4", "imperfect": "#d62728"}

# Spoke angle per sector position, degrees clockwise from the top (the radar
# sets theta_offset=pi/2, theta_direction=-1). The sector *names* come from the
# scorecard's hexagon geometry, where "left"/"bottom" mean something different;
# here they are placed so the low index reads bottom-left and the high index
# bottom-right, matching how the two tails are usually written.
_RADAR_ANGLE = {"top": 0.0, "left": 240.0, "bottom": 120.0}


def plot_multivar_signal_radar(
    paths,
    out: str,
    variables: list[str],
    *,
    reference: str = "RCM-truth",
    reference_path: str | None = None,
    cases: list[str] | None = None,
    season: str = "annual",
    units: dict[str, str] | None = None,
    title: str | None = None,
    filename: str | None = None,
    rlim: tuple[float, float] = (0.0, 2.0),
    colors: dict[str, str] | None = None,
) -> str:
    """Model x variable grid of small *radar* triangles of the CC signal.

    The radar counterpart of :func:`plot_multivar_signal_scorecard`, in the
    polyline style of ``plot_hexagon.plot_hexagon`` rather than the filled-cell
    style: one small polar axes per (model, variable), three spokes -- the three
    indices -- and one closed polyline per boundary case.

    The radial coordinate is ``signal / reference signal``, **signed** -- unlike
    the scorecard, which divides by ``|reference|`` because its colour has to
    keep saying which way the change goes. Dividing by the signed reference puts
    it at +1 on every spoke whatever the sign of its own change, so it is drawn
    once as a dashed unit triangle and the reading is direct:

    * centre               -- no signal at all
    * inner ring, ``r=0.5``-- half the reference's change
    * dashed triangle      -- **reproduces the reference exactly**
    * rim, ``r=2``         -- twice the reference's change

    A **wrong-sign** signal is clamped to the centre and marked, so the spoke
    collapses and the polyline reads as a triangle with one vertex pulled to the
    middle. Values beyond ``rlim`` are clipped and marked with a caret, so a cell
    never silently mis-states a magnitude.

    Parameters
    ----------
    paths : mapping or sequence
        ``{label: path}`` as in :func:`plot_multivar_signal_scorecard`. Every
        non-reference label must read ``"MODEL (case)"``.
    out : str
        Output directory for the PNG.
    variables : list of str
        Variables (columns).
    reference : str, default "RCM-truth"
        Label of the reference source -- the unit triangle of every cell.
    reference_path : str, optional
        Base directory of the reference's signal run, when ``reference`` is not
        a label of ``paths``.
    cases : list of str, optional
        Boundary cases, one polyline each (default ``["perfect", "imperfect"]``).
    season, units, title, filename
        As in :func:`plot_multivar_signal_scorecard`.
    rlim : (float, float), default (0.0, 2.0)
        Radial limits as a fraction of the reference signal, identical on every
        cell so the grid is comparable. The default runs from "no signal" at the
        centre to twice the reference at the rim, putting the reference itself
        halfway out.
    colors : dict, optional
        ``{case: colour}`` override of ``SIGNAL_CASE_COLOR``.

    Returns
    -------
    str
        Path of the written PNG.
    """
    cases = list(cases or ["perfect", "imperfect"])
    units = units or {}
    colors = {**SIGNAL_CASE_COLOR, **(colors or {})}
    tab10 = plt.get_cmap("tab10")

    by_label = normalize_paths(paths)
    model_rows = [m for m in by_label if m != reference]
    ref_base = reference_path or by_label.get(reference)
    if ref_base is None:
        raise KeyError(
            f"reference '{reference}' is neither a label of paths "
            f"({', '.join(by_label)}) nor given via reference_path=."
        )

    models, by_cell = split_case_labels(model_rows, cases)
    signal, truth = _collect_signals(by_label, model_rows, ref_base, variables)

    rmin, rmax = rlim
    nrows, ncols = len(models), len(variables)
    fig, axarr = plt.subplots(
        nrows,
        ncols,
        figsize=(1.75 * ncols + 2.2, 1.75 * nrows + 2.6),
        subplot_kw={"polar": True},
        squeeze=False,
    )

    for r, model in enumerate(models):
        for c, var in enumerate(variables):
            ax = axarr[r][c]
            idx = signal_indices(var)
            keys = [k for k, _ in idx]
            # Spokes are placed by sector position, not by list order: top =
            # mean/SDII, bottom-left = P01/R01, bottom-right = P99/RX1day.
            angles = np.radians([_RADAR_ANGLE[pos] for _, pos in idx])
            closed = np.concatenate([angles, angles[:1]])
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_ylim(rmin, rmax)
            ax.set_rgrids([0.5, 1.0], labels=["", ""])
            ax.grid(color="0.85", lw=0.6)
            ax.set_xticks(angles)
            # spoke names on the top row only -- they are a property of the
            # column (pr has its own indices), not of the cell
            ax.set_xticklabels(
                [INDEX_LABEL[k] for k in keys] if r == 0 else [""] * len(keys),
                fontsize=6.5,
            )
            ax.tick_params(pad=1)
            for spine in ax.spines.values():
                spine.set_edgecolor("0.75")
                spine.set_linewidth(0.7)

            # Divide by the SIGNED reference signal, not by its magnitude: that
            # is what puts the reference at +1 on every spoke whichever way its
            # own change goes. Dividing by |truth| would leave the reference at
            # -1 wherever it falls (uas mean/P99, vas P99 ...), and a model
            # agreeing with it would plot at -0.9 against a +1 outline -- every
            # good wind model would read as catastrophically wrong.
            norm = [truth.get(var, {}).get(k, np.nan) for k in keys]
            if not any(np.isfinite(m) and abs(m) > EPS for m in norm):
                ax.set_facecolor("0.95")  # no reference signal -> nothing to scale by
                continue

            # The reference is the unit triangle; the centre is "no signal" and
            # the rim twice the reference, both read off the polar grid rings.
            ax.plot(
                closed,
                np.ones_like(closed),
                color="0.25",
                lw=1.6,
                ls="--",
                zorder=5,
                label=f"{reference} (ref)" if (r == 0 and c == 0) else None,
            )

            for j, case in enumerate(cases):
                src = by_cell.get((model, case))
                vals = np.array(
                    [
                        (signal.get((src, var), {}).get(k, np.nan) / m)
                        if (src and np.isfinite(m) and abs(m) > EPS)
                        else np.nan
                        for k, m in zip(keys, norm)
                    ],
                    dtype=float,
                )
                if not np.any(np.isfinite(vals)):
                    continue
                col = colors.get(case, tab10(j % 10))
                # A signal opposing the reference has no place on a ratio axis
                # that starts at zero: clamp it to the centre and mark it, so
                # the spoke visibly collapses instead of plotting as a small
                # positive change.
                wrong = np.isfinite(vals) & (vals < 0.0)
                clipped = np.clip(vals, rmin, rmax)
                v = np.concatenate([clipped, clipped[:1]])
                ax.plot(
                    closed,
                    v,
                    color=col,
                    lw=1.6,
                    zorder=4,
                    label=case if (r == 0 and c == 0) else None,
                )
                ax.fill(closed, v, color=col, alpha=0.12, zorder=3)
                # carets where the magnitude was clipped at the rim
                off = np.isfinite(vals) & (vals > rmax)
                if np.any(off):
                    ax.plot(
                        angles[off],
                        clipped[off],
                        marker="^",
                        ls="none",
                        ms=4,
                        color=col,
                        zorder=6,
                    )
                # sign flips, at the centre (all spokes coincide there, but the
                # collapsed vertex of the polyline says which index it was)
                if np.any(wrong):
                    ax.plot(
                        angles[wrong],
                        clipped[wrong],
                        marker="x",
                        ls="none",
                        ms=5,
                        mew=1.4,
                        color=col,
                        zorder=7,
                    )

            if r == 0:
                head = f"{var.upper()}\n{units[var]}" if var in units else var.upper()
                ax.set_title(head, fontsize=9, pad=14)
            if c == 0:
                ax.text(
                    -0.38,
                    0.5,
                    model,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    handles, labels_ = axarr[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels_,
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, 0.012),
        )

    head = title or f"CC-signal radar — {season}"
    fig.suptitle(
        f"{head}\n"
        f"radius = signal / {reference} signal  ·  dashed triangle = {reference}  ·  "
        f"centre = no signal, rim = {rmax:g}x  ·  "
        "x = wrong sign (clamped to centre)  ·  caret = clipped at the rim\n"
        "spokes: top = mean (pr: SDII)  ·  "
        "bottom-left = P01 (R01)  ·  bottom-right = P99 (RX1day)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.02, 0.04, 1.0, 0.94))

    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, filename or f"signal_radar_{season}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[signal-radar] {path}  (rows: {', '.join(models)})")
    return path
