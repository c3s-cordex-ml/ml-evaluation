# SPDX-FileCopyrightText: 2026 c3s-cordex-ml
# SPDX-License-Identifier: MIT

"""
Per-variable *climate-change-signal scorecard* heatmap for ML RCM emulators.

One figure per variable: rows are the sources (the RCM truth and each emulator),
columns are the signal quantities, read from the ``metrics_signal_float.zarr``
bundles an ``ml-eval signal`` call writes. It is the signal counterpart of
``plot_var_skill_scorecard`` (same models-in-rows layout) and the per-variable
counterpart of ``plot_multivar_signal_scorecard`` (which packs every variable
into one grid of hexagons).

Each index contributes up to two columns, the two quantities ``ml-eval signal``
stores for it:

    raw : ``delta_<index>_float``    -- the absolute change, future - historical
    %   : ``reldelta_<index>_float`` -- the same change relative to the
                                        historical climate

The index set defaults to what ``ml-eval signal`` computes for the variable --
mean/P01/P99 for non-precip, R01/SDII/RX1day for precip -- and is overridden with
``indices=``.

* colour = signal / |reference signal| per column, clipped to [-1, +1] and
           diverging (red = increase, blue = decrease). The reference row
           therefore saturates and every model reads as its signed fraction of
           the reference signal. With ``reference=None`` each column is instead
           scaled by its own largest |value|, which needs no reference row.
* number = the signal itself: the absolute change in the "raw" columns, a
           percentage in the "%" columns.

No error/bias column is computed here: an ``ml-eval signal`` run stores one
source's own change, so a model-minus-truth signal error is a *difference of
rows*, which the reader can take directly off the raw numbers.

Locating the data
-----------------
As in ``plot_hexagon``, rows come from an explicit ``{label: path}`` spec and one
variable's stores are read from ``<path>/<var>/``; ``season`` is a label for the
title and the file name only.

Public API
----------
var_signal_columns(var, indices=None, kinds=("delta", "reldelta"))
    The ``(index_key, kind, column_label)`` list for one variable.
read_var_signal_metrics(path, var, indices=None, kinds=...)
    ``{(index_key, kind): value}`` for one ``ml-eval signal --output`` directory.
    Depends only on ml-eval's output format -- layout-agnostic.
render_var_signal_scorecard(per_model, var, out_dir, *, models, indices, ...)
    Render one variable's heatmap from precomputed {label: {(index, kind): v}}.
plot_var_signal_scorecard(paths, out, variables, *, indices, reference, season)
    One PNG per variable, one row per entry of ``paths``.

Example
-------
    from ml_evaluation.viz.plot_var_signal_scorecard import plot_var_signal_scorecard
    plot_var_signal_scorecard(
        {
            "RCM":          "/.../RCM-TRUTH_EC-Earth3_2041-2060/annual",
            "DeepESD":      "/.../DEEPESD_EC-Earth3_2041-2060/annual",
            "UNet":         "/.../UNET_EC-Earth3_2041-2060/annual",
            "UNet-MultiVar": "/.../UNET-MULTI_EC-Earth3_2041-2060/annual",
        },
        "/.../FIGURES/signal",
        ["pr", "tasmax"],
        reference="RCM",
    )
"""

from __future__ import annotations

import copy
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_evaluation.viz.plot_hexagon import metrics_dir, normalize_paths
from ml_evaluation.viz.plot_multivar_signal_scorecard import (
    INDEX_LABEL,
    _read_signal_floats,
    signal_indices,
)

EPS = 1e-12

# The two signal quantities ml-eval stores per index, and how each is labelled
# and printed. ``pct`` marks the ones printed as a percentage (the stored
# reldelta is a fraction).
KIND_SUFFIX = {"delta": "raw", "reldelta": "%"}
KIND_IS_PCT = {"delta": False, "reldelta": True}
DEFAULT_KINDS = ("delta", "reldelta")


def default_indices(var: str) -> list[str]:
    """The index keys ``ml-eval signal`` computes for ``var``, in plotting order.

    mean/P01/P99 for non-precip variables, R01/SDII/RX1day for precip -- taken
    from the same table the multivariate signal scorecard uses, so the two
    figures always cover the same indices.
    """
    return [k for k, _ in signal_indices(var)]


def var_signal_columns(
    var: str, indices=None, kinds=DEFAULT_KINDS
) -> list[tuple[str, str, str]]:
    """The ``(index_key, kind, column_label)`` list for ``var``.

    Parameters
    ----------
    var : str
        Variable name; selects the default index set when ``indices`` is None.
    indices : iterable, optional
        Index keys to plot, in order -- any subset of what ``ml-eval signal``
        writes (``mean``/``p01``/``p99`` or ``r01``/``sdii``/``rx1day``). An
        item may also be an explicit ``(index_key, label)`` pair to override the
        header. Default: :func:`default_indices` for the variable.
    kinds : tuple, default ("delta", "reldelta")
        Which stored quantities to show per index: the absolute change
        (``"delta"``, labelled "raw"), the relative one (``"reldelta"``,
        labelled "%"), or just one of them.

    Raises
    ------
    ValueError
        If ``kinds`` names something ml-eval does not store, or the selection is
        empty.
    """
    if indices is None:
        indices = default_indices(var)
    kinds = tuple(kinds)
    unknown = [k for k in kinds if k not in KIND_SUFFIX]
    if unknown:
        raise ValueError(
            f"unknown signal kind(s) {unknown}; available: {sorted(KIND_SUFFIX)}"
        )
    columns = []
    for item in indices:
        key, label = item if isinstance(item, (tuple, list)) else (item, None)
        label = label or INDEX_LABEL.get(key, key)
        for kind in kinds:
            columns.append((key, kind, f"{label} ({KIND_SUFFIX[kind]})"))
    if not columns:
        raise ValueError("indices/kinds must select at least one column.")
    return columns


def read_var_signal_metrics(
    path: str, var: str, indices=None, kinds=DEFAULT_KINDS
) -> dict[tuple[str, str], float]:
    """Signal scalars for ``var`` from one ``ml-eval signal --output`` directory.

    ``path`` holds the ``metrics_signal_float.zarr`` store written by a single
    ``ml-eval signal`` call. Returns ``{(index_key, kind): value}`` for the
    selected columns; a missing store or diagnostic comes back as NaN. Depends
    only on ml-eval's output format -- no directory-layout assumptions.
    """
    columns = var_signal_columns(var, indices, kinds)
    out: dict[tuple[str, str], float] = {}
    for kind in dict.fromkeys(k for _, k, _ in columns):
        keys = [i for i, k, _ in columns if k == kind]
        for key, val in _read_signal_floats(path, keys, kind).items():
            out[(key, kind)] = val
    return out


def _colour_scale(values: np.ndarray, ref_index) -> np.ndarray:
    """Column values -> [-1, 1] colour positions.

    Normalised by |reference value| when ``ref_index`` points at a row, else by
    the column's largest |value|. Non-finite entries stay NaN (blank cells).
    """
    v = np.asarray(values, dtype=float)
    if ref_index is not None and np.isfinite(v[ref_index]):
        denom = abs(float(v[ref_index]))
    else:
        finite = v[np.isfinite(v)]
        denom = float(np.max(np.abs(finite))) if finite.size else np.nan
    if not np.isfinite(denom) or denom < EPS:
        return np.full(v.shape, np.nan)
    return np.clip(v / denom, -1.0, 1.0)


def _fmt(v: float, pct: bool) -> str:
    """Signed, magnitude-aware label; ``pct`` prints the fraction as a percent."""
    if not np.isfinite(v):
        return ""
    if pct:
        return f"{v * 100:+.0f}%"
    a = abs(v)
    if a >= 100:
        return f"{v:+.0f}"
    if a >= 10:
        return f"{v:+.1f}"
    return f"{v:+.2f}"


def _text_color(rgba) -> str:
    """Black or white, whichever contrasts with the cell fill."""
    r, g, b = rgba[:3]
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.6 else "white"


def render_var_signal_scorecard(
    per_model,
    var,
    out_dir,
    *,
    models=None,
    indices=None,
    kinds=DEFAULT_KINDS,
    reference=None,
    season="annual",
    title=None,
    filename=None,
    cmap="RdBu_r",
    annotate=True,
):
    """Render one variable's signal-scorecard PNG from precomputed signals.

    Parameters
    ----------
    per_model : dict
        ``{label: {(index_key, kind): value}}`` (e.g. from
        :func:`read_var_signal_metrics`).
    var : str
        Variable name (selects the default index set).
    out_dir : str
        Directory for the PNG (created if needed).
    models : list of str, optional
        Subset/order of the rows (default: every key of ``per_model``, in order).
    indices, kinds
        Column selection, see :func:`var_signal_columns`.
    reference : str, optional
        Row whose |signal| normalises each column's colour; it is moved to the
        top and saturates the colormap. None -> scale each column by its own
        largest |value|.
    season : str
        Used for the title and the output file name.
    title, filename, cmap, annotate
        Figure title, output name, diverging colormap, whether to print values.

    Returns
    -------
    str or None
        Path of the written PNG, or None if there was nothing to plot.
    """
    if models is None:
        models = list(per_model)
    else:
        models = [m for m in models if m in per_model]
    if reference is not None and reference in per_model:
        models = [reference] + [m for m in models if m != reference]
    if not models:
        return None

    columns = var_signal_columns(var, indices, kinds)
    labels = [lbl for _, _, lbl in columns]
    cells = [(i, k) for i, k, _ in columns]

    values = np.array(
        [[float(per_model[m].get(c, np.nan)) for c in cells] for m in models]
    )
    ref_index = models.index(reference) if reference in models else None
    colours = np.column_stack(
        [_colour_scale(values[:, c], ref_index) for c in range(len(cells))]
    )

    nrows, ncols = len(models), len(cells)
    fig, ax = plt.subplots(figsize=(1.15 * ncols + 4.0, 0.72 * nrows + 3.2))
    cmap_o = copy.copy(plt.get_cmap(cmap))  # copy: never mutate the global cmap
    cmap_o.set_bad("0.85")
    im = ax.imshow(
        np.ma.masked_invalid(colours), cmap=cmap_o, vmin=-1.0, vmax=1.0, aspect="auto"
    )

    if annotate:
        for r in range(nrows):
            for c, (_, kind) in enumerate(cells):
                txt = _fmt(values[r, c], KIND_IS_PCT[kind])
                if not txt:
                    continue
                pos = colours[r, c]
                colour = (
                    "0.3"
                    if not np.isfinite(pos)
                    else _text_color(cmap_o((pos + 1.0) / 2.0))
                )
                ax.text(c, r, txt, ha="center", va="center", fontsize=8, color=colour)

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(models)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    # a heavier rule between index groups, so the raw/% pair of one index reads
    # as one block
    per_index = max(1, len(dict.fromkeys(k for _, k, _ in columns)))
    for c in range(per_index, ncols, per_index):
        ax.axvline(c - 0.5, color="0.25", lw=1.4)
    for s in ax.spines.values():
        s.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(
        f"CC signal / |{reference} signal|  (clipped ±1)"
        if ref_index is not None
        else "CC signal / column max |signal|"
    )

    head = title or f"Climate-change-signal scorecard — {var} — {season}"
    ax.set_title(head, fontsize=13, pad=12)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename or f"var_signal_scorecard_{var}_{season}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[var-signal-scorecard] {path}  (rows: {', '.join(models)})")
    return path


def plot_var_signal_scorecard(
    paths,
    out,
    variables,
    *,
    indices=None,
    kinds=DEFAULT_KINDS,
    reference=None,
    season="annual",
    **kwargs,
):
    """Plot one signal scorecard per variable for an explicit set of paths.

    Convenience wrapper over :func:`read_var_signal_metrics` +
    :func:`render_var_signal_scorecard`.

    Parameters
    ----------
    paths : mapping or sequence
        ``{label: path}`` (see ``plot_hexagon.normalize_paths``); ``path`` is the
        base directory of that source's ``ml-eval signal`` run and one variable's
        store is read from ``<path>/<var>/``. One entry = one row, in the given
        order.
    out : str
        Output directory for the PNGs.
    variables : iterable of str
        Variables to plot (one scorecard each).
    indices, kinds
        Column selection, see :func:`var_signal_columns`. Default: the index set
        ``ml-eval signal`` writes for the variable, each as raw + %.
    reference : str, optional
        Label whose |signal| normalises the colours (e.g. the RCM truth); it is
        drawn as the top row. None -> per-column max-abs scaling.
    season : str, default "annual"
        Label only -- title and output file name.
    **kwargs
        Forwarded to :func:`render_var_signal_scorecard` (``title``,
        ``filename``, ``cmap``, ``annotate``).

    Returns
    -------
    list of str
        Paths of the PNGs written (one per variable that had data).
    """
    by_label = normalize_paths(paths)
    if reference is not None and reference not in by_label:
        raise KeyError(
            f"reference '{reference}' is not one of the labels in "
            f"paths ({', '.join(by_label)})."
        )

    written = []
    for var in variables:
        per_model = {}
        for label, base in by_label.items():
            d = metrics_dir(base, var)
            if os.path.isdir(d):
                per_model[label] = read_var_signal_metrics(d, var, indices, kinds)
        avail = [m for m in by_label if m in per_model]
        path = render_var_signal_scorecard(
            per_model,
            var,
            out,
            models=avail,
            indices=indices,
            kinds=kinds,
            reference=reference,
            season=season,
            **kwargs,
        )
        if path is None:
            print(
                f"[var-signal-scorecard] {var}: no source directory found "
                "— skipping."
            )
        else:
            written.append(path)
    return written
