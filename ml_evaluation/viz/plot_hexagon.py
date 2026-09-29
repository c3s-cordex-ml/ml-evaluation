"""
Evaluation *hexagon* radar plots for ML RCM emulators.

Renders the 6-metric performance hexagon per variable from the zarr bundles a
reference ``ml-eval fast`` call writes (metrics_fast_field / metrics_fast_float /
metrics_fast_pr_field). Each store holds one ``value`` array stacked along a
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

Locating the data
-----------------
Every entry point takes an explicit ``paths`` spec -- ``{label: path}``, where
``label`` is the name shown in the figure and ``path`` is that model's base
directory. One variable's stores are read from ``<path>/<var>/``. Nothing is
searched, and no directory convention is assumed; see :func:`normalize_paths`
for the accepted spellings. ``season`` is a label for the title and the file
name only.

Public API
----------
normalize_paths(paths)
    Normalise a model->directory spec into an ordered ``{label: path}`` dict.
metrics_dir(path, var)
    The directory holding one variable's stores: ``<path>/<var>``.
read_hexagon_metrics(metrics_dir, var, *, member_dim, member_num)
    The hexagon-axis scalars for one ``ml-eval fast --output`` directory. Depends
    only on ml-eval's output format -- layout-agnostic, reusable by any project.
select_member(da, member_dim, member_num)
    Pick one realisation (or the mean over them) from a stored metric. Shared by
    every reader in this package.
plot_hexagon(per_model, var, out_dir, *, models=None, reference=None, season)
    Render one variable's hexagon from precomputed {label: {metric: value}}.
plot_hexagons(paths, out, variables, *, reference, season)
    Plot one hexagon per variable, one series per entry of ``paths``.
daily_sigma(metrics_dir, *, member_dim, member_num)
    Observed daily standard deviation of the truth -- the denominator the
    scorecard uses to keep physically negligible differences pale.
variability_scale(metrics_dir, metric_key, *, member_dim, member_num)
    That sigma for the metrics carrying the variable's units, NaN for the ones
    that carry none.
plot_multivar_skill_scorecard(paths, out, variables, *, reference, season, ...)
    Model x variable grid of six-wedge hexagons, coloured by how much lower each
    model's error is than ``reference``'s, as a % of the observed daily
    variability (``scale="relative"`` for the older % change in error).

Example
-------
    from ml_evaluation.viz.plot_hexagon import plot_hexagons
    plot_hexagons(
        {
            "UNET-MULTI":  "/.../MULTIV-RCM_EC-Earth3_2041-2060/annual",
            "UNET-SINGLE": "/.../SINGV-RCM_EC-Earth3_2041-2060/annual",
            "RESDIFF_V3":  "/.../RESDIFF-RCM_EC-Earth3_2041-2060/annual",
        },
        "/.../FIGURES/hexagon/figures",
        ["pr", "tasmax"],
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
    "bias_r01": ("metrics_fast_pr_field.zarr", "bias_r1_field", True),
    "bias_sdii": ("metrics_fast_pr_field.zarr", "bias_sdii_field", True),
}

# Denominator of the scorecard wedges in the default ``scale="variability"``
# mode: the *observed* variability of the truth, taken from the reference run's
# ``targ_*`` fields and identical for every model, so a wedge answers "how big is
# this error difference compared with how much the observed quantity itself
# varies?" instead of "how big is it compared with the reference model's own
# error?" -- a question whose answer cannot tell a 0.10 K vs 0.23 K P01 bias
# (both negligible) from a 2.3 K vs 5.0 K one.
#
# Every metric carrying the variable's own units is divided by the SAME number,
# the observed daily standard deviation (see :func:`daily_sigma`). One scale per
# variable keeps the wedges of a cell comparable with each other.
SCALE_IS_SIGMA = {
    "rmse",
    "bias_mean",
    "bias_p01",
    "bias_p99",
    "bias_rx1day",
    "bias_sdii",
}

# The remaining metrics have no sigma-compatible scale -- lag-1 correlation is
# dimensionless, wet-day frequency is a day count, RALSD is a ratio of spectra
# spanning orders of magnitude between variables. They keep the reference model's
# own error as denominator, i.e. the relative formula, remapped by the caller so
# that a +/-100% relative change fills the bar.
SCALE_IS_REFERENCE = {"bias_acorr", "bias_r01", "ralsd"}

# Default colour saturation per scale mode. In "variability" units the useful
# range is a fraction of the observed spread, so +/-100% would wash the plate out.
DEFAULT_CLIP = {"variability": 25.0, "relative": 100.0}

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


def select_member(da, member_dim="member", member_num=0):
    """Reduce the ensemble dimension of a stored metric to one field.

    ml-eval scores every realisation separately, so a generative model's metrics
    carry a ``member`` dimension (length 1 for a deterministic one). Averaging it
    away is *not* the same quantity as the metric of one member, so every reader
    in ``ml_evaluation.viz`` selects explicitly instead of collapsing silently.

    Parameters
    ----------
    da : xr.DataArray
        A stored diagnostic, possibly carrying ``member_dim``.
    member_dim : str or None
        Name of the ensemble dimension. None (or a dimension the data does not
        have) leaves ``da`` untouched.
    member_num : int
        ``>= 0`` selects that member; ``< 0`` averages over the members (the
        mean *of the metric*, which is still not the metric of the ensemble
        mean field -- for that, score the ensemble mean with
        ``ml-eval ... --member-num -1``).

    Returns
    -------
    xr.DataArray
        ``da`` reduced along ``member_dim``, or unchanged.
    """
    if member_dim is None or member_dim not in getattr(da, "dims", ()):
        return da
    if member_num < 0:
        return da.mean(dim=member_dim)
    if member_num in np.asarray(da[member_dim].values):
        return da.sel({member_dim: member_num}, drop=True)
    print(f"Warning: member_num {member_num} doesn't exist so use first member.")
    return da.isel({member_dim: 0}, drop=True)


def _read_metric(
    metrics_dir: str, metric_key: str, *, member_dim="member", member_num=0
) -> float:
    """Domain scalar for one metric: select the diagnostic from its zarr bundle,
    pick the member, and reduce over any remaining dims (mean of |field| for
    biases)."""
    store, diag, use_abs = METRIC_SRC[metric_key]
    path = os.path.join(metrics_dir, store)
    if not os.path.isdir(path):
        return np.nan
    try:
        with xr.open_zarr(path) as ds:
            da = select_member(ds["value"].sel(diagnostic=diag), member_dim, member_num)
    except (KeyError, FileNotFoundError, OSError):
        return np.nan
    if use_abs:
        da = np.abs(da)
    return float(da.mean().values)  # mean over lat/lon (fields) or no-op (floats)


def daily_sigma(metrics_dir: str, *, member_dim="member", member_num=0) -> float:
    """Observed daily variability of the variable, in its own units.

    The domain mean of the truth's local temporal standard deviation,
    ``targ_std_field``, which ``ml-eval fast`` writes into
    ``metrics_fast_field.zarr``. It is the denominator every unit-carrying
    scorecard wedge is divided by.

    Returns
    -------
    float
        The scale, or NaN when the store predates ``targ_std_field`` -- there is
        deliberately no approximation from other stored diagnostics, so a stale
        store shows as a blank wedge rather than as a plausible wrong colour.
    """
    std = _read_target(
        metrics_dir,
        "metrics_fast_field.zarr",
        "targ_std_field",
        member_dim=member_dim,
        member_num=member_num,
    )
    return np.nan if std is None else float(std.mean().values)


def _read_target(
    metrics_dir: str, store: str, diag: str, *, member_dim="member", member_num=0
):
    """The stored truth field ``diag`` of ``store``, or None if unavailable."""
    path = os.path.join(metrics_dir, store)
    if not os.path.isdir(path):
        return None
    try:
        with xr.open_zarr(path) as ds:
            return select_member(
                ds["value"].sel(diagnostic=diag), member_dim, member_num
            ).load()
    except (KeyError, FileNotFoundError, OSError):
        return None


def variability_scale(
    metrics_dir: str, metric_key: str, *, member_dim="member", member_num=0
) -> float:
    """Observed variability to divide ``metric_key`` by, in the metric's units.

    Read from the truth fields of one run -- the reference model's, so every row
    of a scorecard is divided by the same number.

    * a metric in the variable's own units (RMSE and the biases of the mean, of a
      percentile, of Rx1day, of SDII) -- the observed daily standard deviation,
      :func:`daily_sigma`.
    * anything else (``bias_acorr``, ``bias_r01``, ``ralsd``) -- NaN: no
      sigma-compatible scale exists, and the caller falls back to the reference
      model's own error.

    Returns
    -------
    float
        The scale, or NaN when it is unavailable or degenerate.
    """
    if metric_key not in SCALE_IS_SIGMA:
        return np.nan
    scale = daily_sigma(metrics_dir, member_dim=member_dim, member_num=member_num)
    return scale if np.isfinite(scale) and scale > EPS else np.nan


def read_hexagon_metrics(
    metrics_dir: str, var: str, *, member_dim="member", member_num=0
) -> dict[str, float]:
    """Hexagon-axis scalars for `var` from one ``ml-eval fast --output`` dir.

    `metrics_dir` holds the metrics_fast_*.zarr stores written by a single
    ``ml-eval fast`` call. Returns ``{metric_key: scalar}`` for the axes of
    ``var``; a metric whose store/diagnostic is missing comes back as NaN.
    ``member_num`` picks the realisation to score (see :func:`select_member`).
    This depends only on ml-eval's output format -- no directory-layout
    assumptions."""
    return {
        k: _read_metric(metrics_dir, k, member_dim=member_dim, member_num=member_num)
        for k, _ in hexagon_axes(var)
    }


def normalize_paths(paths):
    """Normalise a model->directory spec into an ordered ``{label: path}`` dict.

    Every plotting entry point locates a model's metrics through an explicit
    path, not by searching a root for a conventional directory layout. Each
    ``path`` is the model's *base* directory; the metrics stores for one variable
    live in ``<path>/<var>/`` (see :func:`metrics_dir`).

    Accepted spellings (all preserve the given order, which becomes the row /
    series order in the figures):

    * a mapping -- ``{"UNET-MULTI": "/abs/.../annual", ...}``
    * a sequence of ``(label, path)`` pairs
    * a sequence of dicts -- ``[{"label": "UNET-MULTI", "path": "/abs/..."}, ...]``

    Raises
    ------
    ValueError
        If ``paths`` is empty, an entry is malformed, or a label is duplicated.
    """
    if not paths:
        raise ValueError(
            "paths must be a non-empty {label: path} mapping "
            "(or a sequence of (label, path) / {'label','path'} entries)."
        )
    if isinstance(paths, dict):
        items = list(paths.items())
    else:
        items = []
        for entry in paths:
            if isinstance(entry, dict):
                try:
                    items.append((entry["label"], entry["path"]))
                except KeyError:
                    raise ValueError(
                        f"path entry {entry!r} needs both 'label' and 'path' keys."
                    ) from None
            elif isinstance(entry, (tuple, list)) and len(entry) == 2:
                items.append((entry[0], entry[1]))
            else:
                raise ValueError(
                    f"cannot read path entry {entry!r}: expected a (label, path) "
                    "pair or a {'label': ..., 'path': ...} dict."
                )
    out = {}
    for label, path in items:
        if label in out:
            raise ValueError(f"duplicate label {label!r} in paths.")
        out[str(label)] = str(path)
    return out


def metrics_dir(path, var):
    """The directory holding one variable's zarr stores: ``<path>/<var>``."""
    return os.path.join(path, var)


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
    per_model, var, out_dir, *, models=None, reference=None, season="annual"
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
    season : str
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
        f"{var} — evaluation hexagon\n({season})   {mode}",
        fontsize=12,
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=9)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"hexagon_{var}_{season}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[hexagon] {out}  (models: {', '.join(models)})")
    return out


def plot_hexagons(
    paths,
    out,
    variables,
    *,
    reference=None,
    season="annual",
    member_dim="member",
    member_num=0,
):
    """Plot one hexagon per variable for an explicit set of model paths.

    Convenience wrapper over ``read_hexagon_metrics`` + ``plot_hexagon``. The
    models to plot, their labels and their locations all come from ``paths``; no
    directory is searched and no layout is assumed. One PNG per variable is
    written under ``out``.

    Parameters
    ----------
    paths : mapping or sequence
        ``{label: path}`` (or ``(label, path)`` pairs / ``{'label','path'}``
        dicts -- see :func:`normalize_paths`). ``path`` is the model's base
        directory; one variable's stores are read from ``<path>/<var>/``. The
        given order is the plotting order. The ``reference``, when set, must be
        one of these labels.
    out : str
        Output directory for the PNGs.
    variables : iterable of str
        Variables to plot (one hexagon each).
    reference : str, optional
        Label used as the baseline (% improvement mode). None -> min-max mode.
    season : str
        Label only -- used for the title and the output file name. It plays no
        part in locating the data.

    Returns
    -------
    list of str
        Paths of the PNGs written (one per variable that had data).
    """
    by_label = normalize_paths(paths)
    if reference is not None and reference not in by_label:
        raise KeyError(
            f"reference model '{reference}' is not one of the "
            f"labels in paths ({', '.join(by_label)})."
        )

    written = []
    for var in variables:
        per_model = {}
        for label, base in by_label.items():
            sdir = metrics_dir(base, var)
            if os.path.isdir(sdir):
                per_model[label] = read_hexagon_metrics(
                    sdir, var, member_dim=member_dim, member_num=member_num
                )
        if reference and reference not in per_model:
            print(
                f"[hexagon] {var}: reference '{reference}' has no data under "
                f"{metrics_dir(by_label[reference], var)!r} — skipping variable."
            )
            continue
        avail = [m for m in by_label if m in per_model]
        path = plot_hexagon(
            per_model, var, out, models=avail, reference=reference, season=season
        )
        if path is None:
            print(f"[hexagon] {var}: no model directory found — skipping.")
        else:
            written.append(path)
    return written


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


def plot_multivar_skill_scorecard(
    paths,
    out,
    variables,
    *,
    reference,
    season="annual",
    title=None,
    scale="variability",
    clip=None,
    cmap="RdBu",
    filename=None,
    member_dim="member",
    member_num=0,
):
    """Model x variable scorecard of six-wedge hexagons.

    One figure: rows are the labels of ``paths`` (the ``reference`` first),
    columns are ``variables``. Each cell is a hexagon whose six wedges (from the
    top, counter-clockwise; see ``scorecard_axes``) are coloured by how much lower
    that model's error is than the reference's -- blue = better, red = worse,
    white = equal to the reference (so the reference row is blank). Values are
    clipped to +/- ``clip`` for colour.

    Two ways to scale the difference, chosen with ``scale``:

    * ``"variability"`` (default) -- ``100 * (err_ref - err_model) / sigma``,
      where ``sigma`` is the observed daily standard deviation of the truth (see
      :func:`daily_sigma`). A wedge reads as "% of the observed daily
      variability", so a difference that is negligible against the physical
      spread of the variable stays pale however many times larger than the
      reference error it is. Metrics not in the variable's units keep the
      relative form (below).
    * ``"relative"`` -- ``100 * (err_ref - err_model) / err_ref`` for every
      wedge. Scale-free, which is exactly its weakness: a 0.10 K vs 0.23 K P01
      bias and a 2.3 K vs 5.0 K one both saturate the colour, though only the
      second is a real difference in performance.

    Lag-1 correlation bias, wet-day frequency bias and RALSD have no
    sigma-compatible scale (see ``SCALE_IS_REFERENCE``), so in both modes they
    use ``err_ref`` as denominator, remapped so that a +/-100% relative change
    fills the bar rather than saturating it in the first percent.

    ``scale="variability"`` needs ``targ_std_field``, written by ``ml-eval fast``
    from 2026-08 on. Variables whose stores predate it draw their unit-carrying
    wedges blank, and the run prints which ones need re-evaluating.

    Always plots the six "fast" hexagon metrics; there is no metric selection.

    Parameters
    ----------
    paths : mapping or sequence
        ``{label: path}`` as in :func:`plot_hexagons`; one variable's stores are
        read from ``<path>/<var>/``. Row order follows the given order, with the
        ``reference`` moved to the top.
    out : str
        Output directory for the PNG.
    variables : iterable of str
        Variables (columns).
    reference : str
        Label of the baseline row (required). Must be one of the ``paths``
        labels.
    season : str
        Label only -- title and output file name.
    scale : {"variability", "relative"}
        Denominator of the wedge value (see above). Default "variability".
    title, clip, cmap, filename
        Figure title, colour saturation (%), diverging colormap, output name.
        ``clip`` defaults per mode (``DEFAULT_CLIP``): 25% of the observed daily
        variability, or 100% in "relative" mode.
    member_dim : str or None
        Name of the ensemble dimension in the stored metrics.
    member_num : int
        Realisation to score: ``>= 0`` that member, ``< 0`` the mean over
        members. Default 0 -- the only member of a deterministic model.

    Returns
    -------
    str
        Path of the written PNG.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if scale not in DEFAULT_CLIP:
        raise ValueError(
            f"scale must be one of {', '.join(sorted(DEFAULT_CLIP))}, got {scale!r}."
        )
    if clip is None:
        clip = DEFAULT_CLIP[scale]

    by_label = normalize_paths(paths)
    if reference not in by_label:
        raise KeyError(
            f"reference model '{reference}' is not one of the "
            f"labels in paths ({', '.join(by_label)})."
        )

    rows = [reference] + [m for m in by_label if m != reference]

    # metrics[model][var] = {metric_key: scalar}
    metrics = {}
    for m in rows:
        metrics[m] = {}
        for var in variables:
            d = metrics_dir(by_label[m], var)
            if os.path.isdir(d):
                keys = [k for k, _ in scorecard_axes(var)]
                metrics[m][var] = {
                    k: _read_metric(d, k, member_dim=member_dim, member_num=member_num)
                    for k in keys
                }

    # scales[var][metric]: the wedge denominator. NaN means "use the reference
    # error instead" -- either the metric has no physical scale, or the reference
    # run predates targ_std_field, which is reported rather than papered over.
    scales, stale = {}, []
    for var in variables:
        d = metrics_dir(by_label[reference], var)
        scales[var] = {
            k: (
                variability_scale(d, k, member_dim=member_dim, member_num=member_num)
                if scale == "variability"
                else np.nan
            )
            for k, _ in scorecard_axes(var)
        }
        if scale == "variability" and not any(
            np.isfinite(v) for v in scales[var].values()
        ):
            stale.append(var)
    if stale:
        print(
            f"[scorecard] no targ_std_field for {', '.join(stale)} -- their "
            "RMSE and bias wedges are left blank. Re-run `ml-eval fast` for "
            f"{reference} on those variables to colour them."
        )

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
                    denom, full = scales[var].get(key, np.nan), 100.0
                    if not np.isfinite(denom) and (
                        scale == "relative" or key in SCALE_IS_REFERENCE
                    ):
                        # no physical scale for this metric: the reference error
                        # stands in, scaled so that halving or doubling it -- a
                        # +/-100% relative change -- fills the bar rather than
                        # saturating it at the first percent
                        denom, full = er, clip
                    if np.isfinite(em) and np.isfinite(denom) and denom != 0:
                        val = full * (er - em) / denom
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
    variability = scale == "variability"
    cb = fig.colorbar(
        sm, ax=ax, fraction=0.025, pad=0.02, extend="both" if variability else "min"
    )
    cb.set_label(
        (
            f"error vs {reference}, % of observed daily variability"
            if variability
            else f"% error improvement vs {reference}"
        )
        + "  (+ = better, − = worse)"
    )

    # title + wedge-order key
    head = title or f"Scorecard — {season}"
    formula = (
        r"value = 100 × (E$_{ref}$ − E$_{model}$) / σ   "
        "[σ = observed daily std of the truth;  Bias lag-1 corr, Bias R01 and "
        r"RALSD have no σ: relative change, ±100% fills the bar]"
        if variability
        else r"value = 100 × (E$_{ref}$ − E$_{model}$) / E$_{ref}$   "
        "[% change in error vs ref;  +100% = 100% lower error,  −100% = double error]"
    )
    ax.set_title(
        f"{head}\nreference: {reference}\n"
        "wedges — upper: RMSE, Bias lag-1 corr, RALSD   "
        "lower: Bias P01, Bias mean, Bias P99   (PR lower: Bias R01, Bias SDII, Bias Rx1day)\n"
        + formula
        + f"\ncolour saturates at ±{clip:g}",
        fontsize=10,
    )

    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, filename or f"scorecard_{season}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[scorecard] {path}  (models: {', '.join(rows)})")
    return path
