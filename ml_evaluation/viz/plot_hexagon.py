"""
Evaluation *hexagon* radar plots for ML RCM emulators.

Renders the 6-metric performance hexagon per variable from the zarr bundles a
reference ``ml-eval fast`` call writes (metrics_fast_field / metrics_fast_float /
metrics_pr_field). Each store holds one ``value`` array stacked along a
``diagnostic`` dimension, selected with ``ds["value"].sel(diagnostic=<name>)``.

The hexagon axes differ by variable:
    non-precip : RMSE, bias mean, bias P01, bias P99, bias lag-1 corr, RALSD
    precip (pr): RMSE, bias R01, bias Rx1day, bias SDII, bias lag-1 corr, RALSD

Every metric is an *error* (lower is better); bias fields are reduced to a domain
scalar as mean(|field|), RMSE/RALSD are already scalars. Two radial-normalisation
modes:

  * default          -- each axis min-max normalised across the plotted models,
                        0 = best-in-set (centre), 1 = worst-in-set (rim).
  * reference=MODEL  -- each axis is the % improvement over the reference,
                        100 * (err(MODEL) - err) / err(MODEL); the reference is
                        the zero hexagon (dashed), outward = better.

Public API
----------
read_hexagon_metrics(metrics_dir, var)
    The hexagon-axis scalars for one ``ml-eval fast --output`` directory. Depends
    only on ml-eval's output format -- layout-agnostic, reusable by any project.
plot_hexagon(per_model, var, out_dir, *, models=None, reference=None, season, case)
    Render one variable's hexagon from precomputed {label: {metric: value}}.
plot_hexagons(root, out, variables, models, *, case, season, reference, layout, pattern)
    Plot one hexagon per variable for an explicit list of ``models``. Each model's
    metrics dir is ``root`` joined with a relative `pattern` filled from
    {model}/{checkpoint}/{var}/{case}/{season} (`case` used verbatim; `checkpoint`
    auto-resolved or pinned via `checkpoint=`). `pattern` defaults per `layout` --
    "d4p": root/{model}/outputs/evaluation/{checkpoint}/{var}/{case}/{season};
    "flat": root/{var}/{case}/{season}. Any concatenation (e.g. a stem dir named
    <model>_<case>) lives in the caller's pattern. Projects with a layout neither
    captures can build the per-model dict themselves (read_hexagon_metrics) and
    call plot_hexagon.

Example
-------
    from ml_evaluation.viz.plot_hexagon import plot_hexagons
    plot_hexagons(
        "/.../EMULATORS",
        "/.../FIGURES/hexagon/figures",
        ["pr", "tasmax"],
        ["UNET-MULTI", "UNET-SINGLE", "RESDIFF_V3"],
        case="EC-Earth3_2041-2060_perfect",
        reference="UNET-MULTI",
    )
"""

from __future__ import annotations

import os

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12

# metric key -> (zarr store, diagnostic name, use_abs). All are errors (low=best).
METRIC_SRC = {
    "rmse": ("metrics_fast_float.zarr", "rmse_float", False),
    "ralsd": ("metrics_fast_float.zarr", "ralsd_float", False),
    "bias_acorr": ("metrics_fast_field.zarr", "bias_acorr_field", True),
    "bias_mean": ("metrics_fast_field.zarr", "bias_mean_field", True),
    "bias_p01": ("metrics_fast_field.zarr", "bias_p1_field", True),
    "bias_p99": ("metrics_fast_field.zarr", "bias_p99_field", True),
    "bias_rx1day": ("metrics_fast_field.zarr", "bias_max1day_field", True),
    "bias_r01": ("metrics_pr_field.zarr", "bias_r1_field", True),
    "bias_sdii": ("metrics_pr_field.zarr", "bias_sdii_field", True),
}

# hexagon axes (metric key, axis label) per variable type
HEX_TEMP = [
    ("rmse", "RMSE"),
    ("bias_mean", "Bias mean"),
    ("bias_p01", "Bias P01"),
    ("bias_p99", "Bias P99"),
    ("bias_acorr", "Bias lag-1 corr"),
    ("ralsd", "RALSD"),
]
HEX_PR = [
    ("rmse", "RMSE"),
    ("bias_r01", "Bias R01"),
    ("bias_rx1day", "Bias Rx1day"),
    ("bias_sdii", "Bias SDII"),
    ("bias_acorr", "Bias lag-1 corr"),
    ("ralsd", "RALSD"),
]


def hexagon_axes(var):
    """The (metric_key, axis_label) list for `var` (pr uses precip-specific axes)."""
    return HEX_PR if var == "pr" else HEX_TEMP


# Scorecard wedge order, indexed 0..5 as drawn by ``_wedge``:
#   0 top, 1 upper-left, 2 lower-left, 3 bottom, 4 lower-right, 5 upper-right.
# Upper 3 (top, upper-left, upper-right): RMSE, Bias lag-1 corr, RALSD.
# Lower 3 (lower-left, bottom, lower-right): Bias P01, Bias mean, Bias P99
# (Bias R01, Bias SDII, Bias Rx1day for pr). Columns pair upper/lower:
# RMSE↕P01, lag-1 corr↕mean, RALSD↕P99.
SCORECARD_TEMP = [
    ("bias_acorr", "Bias lag-1 corr"),  # 0 top
    ("rmse", "RMSE"),  # 1 upper-left
    ("bias_p01", "Bias P01"),  # 2 lower-left
    ("bias_mean", "Bias mean"),  # 3 bottom
    ("bias_p99", "Bias P99"),  # 4 lower-right
    ("ralsd", "RALSD"),  # 5 upper-right
]
SCORECARD_PR = [
    ("bias_acorr", "Bias lag-1 corr"),  # 0 top
    ("rmse", "RMSE"),  # 1 upper-left
    ("bias_r01", "Bias R01"),  # 2 lower-left
    ("bias_sdii", "Bias SDII"),  # 3 bottom
    ("bias_rx1day", "Bias Rx1day"),  # 4 lower-right
    ("ralsd", "RALSD"),  # 5 upper-right
]


def scorecard_axes(var):
    """The (metric_key, wedge_label) list for `var` in the scorecard hexagon."""
    return SCORECARD_PR if var == "pr" else SCORECARD_TEMP


def _read_metric(metrics_dir: str, metric_key: str) -> float:
    """Domain scalar for one metric: select the diagnostic from its zarr bundle
    and reduce over any remaining dims (mean of |field| for biases)."""
    store, diag, use_abs = METRIC_SRC[metric_key]
    path = os.path.join(metrics_dir, store)
    if not os.path.isdir(path):
        return np.nan
    try:
        with xr.open_zarr(path) as ds:
            da = ds["value"].sel(diagnostic=diag)
    except (KeyError, FileNotFoundError, OSError):
        return np.nan
    if use_abs:
        da = np.abs(da)
    return float(da.mean().values)  # mean over lat/lon (fields) or no-op (floats)


def read_hexagon_metrics(metrics_dir: str, var: str) -> dict[str, float]:
    """Hexagon-axis scalars for `var` from one ``ml-eval fast --output`` dir.

    `metrics_dir` holds the metrics_fast_*.zarr stores written by a single
    ``ml-eval fast`` call. Returns ``{metric_key: scalar}`` for the axes of
    ``var``; a metric whose store/diagnostic is missing comes back as NaN. This
    depends only on ml-eval's output format -- no directory-layout assumptions."""
    return {k: _read_metric(metrics_dir, k) for k, _ in hexagon_axes(var)}


# Relative path (from ``root``) of a model's metrics directory, per named layout.
# Override with the ``pattern`` argument. Placeholders: {model} {checkpoint}
# {var} {case} {season}. The metrics dir is the one holding the
# metrics_fast_*.zarr stores. The d4p layout carries a {checkpoint} segment (a
# model may hold several checkpoints, each with its own metrics subtree); when
# the caller does not pin one it is auto-resolved to the sole checkpoint dir.
LAYOUT_PATTERNS = {
    "d4p": "{model}/outputs/evaluation/{checkpoint}/{var}/{case}/{season}",
    "flat": "{var}/{case}/{season}",
}


def _resolve_checkpoint(root, pattern, model, checkpoint):
    """Checkpoint tag to use for `model`, or None if it can't be resolved.

    Only meaningful when `pattern` contains a ``{checkpoint}`` segment (the d4p
    layout does: a model may hold several checkpoints, each with its own metrics
    subtree ``.../outputs/evaluation/<checkpoint>/<var>/<case>/<season>``).
    `checkpoint` may be a plain str (the same tag for every model), a
    ``{model: tag}`` dict, or None. When it does not pin a tag for `model`, the
    tag is auto-resolved to the *sole* checkpoint directory present; if zero or
    several exist the model is skipped (None) -- pass an explicit tag to choose."""
    if "{checkpoint}" not in pattern:
        return ""
    if isinstance(checkpoint, str) and checkpoint:
        return checkpoint
    if isinstance(checkpoint, dict) and checkpoint.get(model):
        return checkpoint[model]
    prefix = pattern.split("{checkpoint}")[0].format(
        model=model, var="", case="", season="", checkpoint=""
    )
    base = os.path.join(root, prefix)
    if not os.path.isdir(base):
        return None
    subs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    if len(subs) == 1:
        return subs[0]
    if len(subs) > 1:
        print(
            f"[hexagon] {model}: {len(subs)} checkpoints {subs} under "
            f"{prefix!r} -- pass checkpoint={{{model!r}: <tag>}} to pick one."
        )
    return None


def _metrics_dir(root, pattern, model, var, case, season, cp):
    """Absolute metrics dir for one cell, or None when the checkpoint is unresolved."""
    if "{checkpoint}" in pattern and not cp:
        return None
    rel = pattern.format(
        model=model, var=var, case=case, season=season, checkpoint=cp or ""
    )
    return os.path.join(root, rel)


def _normalise(values: np.ndarray) -> np.ndarray:
    """Error metrics -> [0,1] with 0 = best (min). NaNs pushed to the rim."""
    v = np.array(values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.ones_like(v)
    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    norm = np.zeros_like(v) if vmax - vmin < EPS else (v - vmin) / (vmax - vmin)
    norm[~np.isfinite(v)] = 1.0
    return norm


def _pct_improvement(model_vals, ref_vals) -> np.ndarray:
    """Per-axis 100 * (score_ref - score_model) / score_ref.

    Score is an error, so positive = the model has a *lower* error than the
    reference (an improvement); 0 = same as the reference. Axes where the
    reference score is 0 or non-finite are returned as NaN (undefined)."""
    model = np.asarray(model_vals, dtype=float)
    ref = np.asarray(ref_vals, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = 100.0 * (ref - model) / ref
    pct[~np.isfinite(pct)] = np.nan
    return pct


def plot_hexagon(
    per_model, var, out_dir, *, models=None, reference=None, season="annual", case=""
):
    """Render one variable's hexagon PNG from precomputed per-model metrics.

    Parameters
    ----------
    per_model : dict
        ``{label: {metric_key: scalar}}`` (e.g. from ``read_hexagon_metrics``).
    var : str
        Variable name (selects the axis set via ``hexagon_axes``).
    out_dir : str
        Directory for the PNG (created if needed).
    models : list of str, optional
        Subset/order of labels to draw (default: every key of ``per_model``).
        The ``reference`` is always drawn.
    reference : str, optional
        Label used as the baseline -> % improvement mode. None -> min-max mode.
    season, case : str
        Only used for the title and the output file name.

    Returns
    -------
    str or None
        Path of the written PNG, or None if there was nothing to plot.
    """
    if models is None:
        models = list(per_model)
    else:
        models = [m for m in models if m in per_model]
    if reference is not None:
        if reference not in per_model:
            raise KeyError(f"reference model '{reference}' not in per_model")
        if reference not in models:
            models = models + [reference]
    if not models:
        return None

    axes = hexagon_axes(var)
    keys = [k for k, _ in axes]
    labels = [lbl for _, lbl in axes]
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False)
    closed = np.concatenate([angles, angles[:1]])

    if reference:
        # Radial value = % improvement over the reference (outward = better).
        ref_vals = [per_model[reference][k] for k in keys]
        vals_by_model = {
            m: _pct_improvement([per_model[m][k] for k in keys], ref_vals)
            for m in models
        }
        finite = np.concatenate(
            [v[np.isfinite(v)] for v in vals_by_model.values()] + [np.array([0.0])]
        )
        lo, hi = float(finite.min()), float(finite.max())
        pad = 0.10 * max(hi - lo, 1.0)
        ymin, ymax = min(0.0, lo) - pad, max(0.0, hi) + pad
        for v in vals_by_model.values():
            v[~np.isfinite(v)] = ymin  # missing -> worst (inward)
        from matplotlib.ticker import MaxNLocator

        yticks = [
            t for t in MaxNLocator(nbins=5).tick_values(ymin, ymax) if ymin <= t <= ymax
        ]
        if 0.0 not in yticks:
            yticks = sorted(yticks + [0.0])
        ylabels = ["ref" if t == 0 else f"{t:+.0f}%" for t in yticks]
        mode = f"% improvement vs {reference}   (0 = ref dashed, outward = better)"
    else:
        # Radial value = per-axis min-max across the plotted models (0 = best).
        vals_by_model = {m: np.zeros(len(axes)) for m in models}
        for i, k in enumerate(keys):
            col = np.array([per_model[m][k] for m in models], dtype=float)
            for m, val in zip(models, _normalise(col)):
                vals_by_model[m][i] = val
        ymin, ymax = 0.0, 1.0
        yticks = [0.25, 0.5, 0.75, 1.0]
        ylabels = ["", "", "", ""]
        mode = "0 = best (centre)"

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_rlabel_position(0)

    cmap = plt.get_cmap("tab10")
    ci = 0
    for m in models:
        vals = np.concatenate([vals_by_model[m], vals_by_model[m][:1]])
        if reference and m == reference:
            ax.plot(
                closed,
                vals,
                color="0.25",
                lw=2.5,
                ls="--",
                label=f"{m} (ref)",
                zorder=5,
            )
        else:
            ax.plot(closed, vals, color=cmap(ci % 10), lw=2, label=m)
            ax.fill(closed, vals, color=cmap(ci % 10), alpha=0.10)
            ci += 1

    ax.set_title(
        f"{var} — evaluation hexagon\n({season}, {case})   {mode}",
        fontsize=12,
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=9)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"hexagon_{var}_{season}_{case}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[hexagon] {out}  (models: {', '.join(models)})")
    return out


def plot_hexagons(
    root,
    out,
    variables,
    models,
    *,
    case,
    season="annual",
    reference=None,
    layout="d4p",
    pattern=None,
    checkpoint=None,
):
    """Plot one hexagon per variable for an explicit list of models.

    Convenience wrapper over ``read_hexagon_metrics`` + ``plot_hexagon``. The
    emulators to plot are given explicitly in ``models`` (no directory is
    searched to discover them). Each model's metrics directory (holding the
    metrics_fast_*.zarr stores) is ``root`` joined with a relative ``pattern``
    formatted with the ``{model}``/``{var}``/``{case}``/``{season}`` placeholders
    -- ``case`` is used verbatim, nothing is concatenated here. If ``pattern`` is
    None it defaults per ``layout`` (see ``LAYOUT_PATTERNS``):

    * ``layout="d4p"`` (deep4production, default) --
        ``root/{model}/outputs/evaluation/{checkpoint}/{var}/{case}/{season}``
    * ``layout="flat"`` (centralized) --
        ``root/{var}/{case}/{season}``

    The d4p ``{checkpoint}`` segment (a model may hold several checkpoints) is
    filled from ``checkpoint``: a str (same tag for all models), a
    ``{model: tag}`` dict, or None -- auto-resolved to the model's sole
    checkpoint dir, skipping any model with zero or several unless pinned.

    Any concatenation (e.g. a d4p stem dir named ``<model>_<case>``) lives in the
    caller's ``pattern``, e.g. ``"{model}/outputs/evaluation/{var}/{model}_{case}/{season}"``.
    One PNG per variable is written under ``out``.

    Parameters
    ----------
    root : str
        Base directory the ``pattern`` is relative to.
    out : str
        Output directory for the PNGs.
    variables : iterable of str
        Variables to plot (one hexagon each).
    models : list of str
        Emulator names to plot (required). The ``reference`` is read and drawn
        even if not listed here.
    case : str
        Case tag, substituted for ``{case}`` in the pattern (verbatim).
    season : str
        Season name, substituted for ``{season}`` (default "annual").
    reference : str, optional
        Model used as the baseline (% improvement mode). None -> min-max mode.
    layout : {"d4p", "flat"}
        Selects the default ``pattern`` when ``pattern`` is None.
    pattern : str, optional
        Relative path from ``root`` with ``{model}``/``{checkpoint}``/``{var}``/
        ``{case}``/``{season}`` placeholders. Overrides ``layout``'s default.
    checkpoint : str or dict, optional
        Which checkpoint to read when the pattern has a ``{checkpoint}`` segment:
        a str (same tag for every model), a ``{model: tag}`` dict, or None
        (default) to auto-resolve each model's sole checkpoint dir.
    """
    if pattern is None:
        if layout not in LAYOUT_PATTERNS:
            raise ValueError(
                f"unknown layout {layout!r}; use one of "
                f"{sorted(LAYOUT_PATTERNS)} or pass pattern=."
            )
        pattern = LAYOUT_PATTERNS[layout]
    if not models:
        raise ValueError("models must be a non-empty list of emulator names to plot.")

    plot_names = list(dict.fromkeys(models))  # de-dup, keep order
    # Models whose metrics we must read: the ones to plot plus the reference.
    need = list(dict.fromkeys(plot_names + ([reference] if reference else [])))
    cp_by_model = {m: _resolve_checkpoint(root, pattern, m, checkpoint) for m in need}

    for var in variables:
        per_model = {}
        for model in need:
            sdir = _metrics_dir(
                root, pattern, model, var, case, season, cp_by_model[model]
            )
            if sdir and os.path.isdir(sdir):
                per_model[model] = read_hexagon_metrics(sdir, var)
        if reference and reference not in per_model:
            print(
                f"[hexagon] {var}: reference '{reference}' has no data for "
                f"case '{case}' / season '{season}' — skipping variable."
            )
            continue
        avail = [m for m in plot_names if m in per_model]
        if (
            plot_hexagon(
                per_model,
                var,
                out,
                models=avail,
                reference=reference,
                season=season,
                case=case,
            )
            is None
        ):
            print(
                f"[hexagon] {var}: no evaluated models for case '{case}' / "
                f"season '{season}' — skipping."
            )


def _resolve_pattern(layout, pattern):
    if pattern is not None:
        return pattern
    if layout not in LAYOUT_PATTERNS:
        raise ValueError(
            f"unknown layout {layout!r}; use one of "
            f"{sorted(LAYOUT_PATTERNS)} or pass pattern=."
        )
    return LAYOUT_PATTERNS[layout]


def _wedge(ax, cx, cy, i, radius, facecolor, edgecolor="white", lw=0.6):
    """Draw wedge ``i`` (0=top, counter-clockwise) of a 6-sided hexagon cell."""
    from matplotlib.patches import Polygon

    a0 = np.radians(90 + i * 60 - 30)
    a1 = np.radians(90 + i * 60 + 30)
    tri = [
        (cx, cy),
        (cx + radius * np.cos(a0), cy + radius * np.sin(a0)),
        (cx + radius * np.cos(a1), cy + radius * np.sin(a1)),
    ]
    ax.add_patch(
        Polygon(tri, closed=True, facecolor=facecolor, edgecolor=edgecolor, lw=lw)
    )


def plot_scorecard(
    root,
    out,
    variables,
    models,
    *,
    case,
    reference,
    season="annual",
    layout="d4p",
    pattern=None,
    checkpoint=None,
    title=None,
    clip=100.0,
    cmap="RdBu",
    filename=None,
):
    """Model x variable scorecard of six-wedge hexagons.

    One figure: rows are ``models`` (the ``reference`` first), columns are
    ``variables``. Each cell is a hexagon whose six wedges (from the top,
    counter-clockwise; see ``hexagon_axes``) are coloured by the % improvement
    of that model over the reference on each metric,
    ``100 * (err_ref - err_model) / err_ref`` -- blue = better (lower error),
    red = worse, white = equal to the reference (so the reference row is blank).
    Values are clipped to +/- ``clip`` for colour.

    Metrics dirs are located exactly like :func:`plot_hexagons` (``root`` +
    ``pattern``, defaulting per ``layout``; the d4p ``{checkpoint}`` segment is
    filled from ``checkpoint`` -- str, ``{model: tag}`` dict, or None to auto-
    resolve the sole checkpoint dir). Writes one PNG to ``out`` and returns its
    path.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    pattern = _resolve_pattern(layout, pattern)
    if not models:
        raise ValueError("models must be a non-empty list of emulator names to plot.")

    rows = list(dict.fromkeys(models))
    if reference not in rows:
        rows = [reference] + rows
    cp_by_model = {m: _resolve_checkpoint(root, pattern, m, checkpoint) for m in rows}

    # metrics[model][var] = {metric_key: scalar}
    metrics = {}
    for m in rows:
        metrics[m] = {}
        for var in variables:
            d = _metrics_dir(root, pattern, m, var, case, season, cp_by_model[m])
            if d and os.path.isdir(d):
                keys = [k for k, _ in scorecard_axes(var)]
                metrics[m][var] = {k: _read_metric(d, k) for k in keys}

    ncols, nrows = len(variables), len(rows)
    norm = Normalize(-clip, clip)
    cmap_o = plt.get_cmap(cmap)
    nan_color = "0.9"
    R = 0.46

    fig, ax = plt.subplots(figsize=(1.55 * ncols + 3.0, 1.25 * nrows + 3.0))
    for r, m in enumerate(rows):
        for c, var in enumerate(variables):
            cx, cy = c, -r
            axes6 = scorecard_axes(var)
            mv = metrics.get(m, {}).get(var)
            rv = metrics.get(reference, {}).get(var)
            for i, (key, _) in enumerate(axes6):
                val = np.nan
                if mv is not None and rv is not None:
                    em, er = mv.get(key, np.nan), rv.get(key, np.nan)
                    if np.isfinite(em) and np.isfinite(er) and er != 0:
                        val = 100.0 * (er - em) / er
                fc = (
                    cmap_o(norm(max(-clip, min(clip, val))))
                    if np.isfinite(val)
                    else nan_color
                )
                _wedge(ax, cx, cy, i, R, fc)

    ax.set_xlim(-0.7, ncols - 0.3)
    ax.set_ylim(-(nrows - 1) - 0.7, 0.7)
    ax.set_aspect("equal")
    ax.set_xticks(range(ncols))
    ax.set_xticklabels([v.upper() for v in variables])
    ax.set_yticks([-r for r in range(nrows)])
    ax.set_yticklabels(rows)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # colour bar
    sm = ScalarMappable(norm=norm, cmap=cmap_o)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, extend="min")
    cb.set_label(f"% error improvement vs {reference}  (+ = better, − = worse)")

    # title + wedge-order key
    head = title or f"Scorecard — {case} — {season}"
    ax.set_title(
        f"{head}\nreference: {reference}\n"
        "wedges — upper: RMSE, Bias lag-1 corr, RALSD   "
        "lower: Bias P01, Bias mean, Bias P99   (PR lower: Bias R01, Bias SDII, Bias Rx1day)\n"
        r"value = 100 × (E$_{ref}$ − E$_{model}$) / E$_{ref}$   "
        "[% change in error vs ref;  +100% = 100% lower error,  −100% = double error]",
        fontsize=10,
    )

    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, filename or f"scorecard_{case}_{season}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[scorecard] {path}  (models: {', '.join(rows)})")
    return path
