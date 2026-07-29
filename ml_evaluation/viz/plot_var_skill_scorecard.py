"""
Per-variable *performance scorecard* heatmap for ML RCM emulators.

One figure per variable: rows are emulators, columns are the evaluation metrics,
read from the zarr bundles ``ml-eval fast`` writes (metrics_fast_float /
metrics_fast_field / metrics_fast_pr_field) and, for the distribution-distance
and variability columns, the ones ``ml-eval slow`` writes (metrics_slow_field)
into the same directory. Unlike the six-wedge hexagon scorecard in
``plot_hexagon.plot_multivar_skill_scorecard`` (one cell per model x variable, coloured
against a
reference model), here every metric gets its own column and the comparison is
made *within* the column:

  * colour  = per-column min-max normalised **performance score**, 1 = best
              (lowest error) of the plotted models, 0 = worst. Every metric is an
              error, so the score is ``(worst - value) / (worst - best)``. This
              makes columns with wildly different units/ranges (RMSE in K,
              RALSD dimensionless, biases in mm/day) directly comparable.
  * number  = the raw metric value in its own units, with the spatial
              coefficient of variation ``CV`` of the underlying field below it.

The ``CV`` line answers "is this domain scalar representative?": a bias whose
domain mean is small but whose CV is large is not a uniformly good bias, it is a
field of compensating errors. It is ``std(field) / mean(|field|)`` over the
domain, and is omitted for metrics that ml-eval only stores as a float scalar
(RALSD).

Which metrics become columns is up to the caller: pass ``metrics=`` a list of
metric keys (see ``METRIC_LABELS`` for the full menu) in the order wanted. The
default, when ``metrics`` is None, is variable-dependent:

    non-precip : RMSE, mean bias, P01 bias, P99 bias, RALSD, lag-1 corr bias
    precip (pr): RMSE, mean bias, wet-day freq bias, SDII bias, P98wet bias,
                 RALSD, lag-1 corr bias

Locating the data
-----------------
As in ``plot_hexagon``, models come from an explicit ``{label: path}`` spec and
one variable's stores are read from ``<path>/<var>/``; ``season`` is a label for
the title and the file name only.

Public API
----------
var_skill_columns(var, metrics=None)
    The (metric_key, column_label) list for one variable.
read_var_skill_metrics(metrics_dir, var, metrics=None)
    ``{metric_key: (value, cv)}`` for one ``ml-eval fast --output`` directory.
    Depends only on ml-eval's output format -- layout-agnostic.
render_var_skill_scorecard(per_model, var, out_dir, *, models, metrics, season)
    Render one variable's heatmap from precomputed {label: {metric: (value, cv)}}.
plot_var_skill_scorecard(paths, out, variables, *, metrics, season)
    One PNG per variable, one row per entry of ``paths``.

Example
-------
    from ml_evaluation.viz.plot_var_skill_scorecard import plot_var_skill_scorecard
    plot_var_skill_scorecard(
        {
            "UNet-single": "/.../SINGV-RCM_EC-Earth3_2041-2060/annual",
            "ResDiff-v3":  "/.../RESDIFF-RCM_EC-Earth3_2041-2060/annual",
            "DownViT":     "/.../DOWNVIT-RCM_EC-Earth3_2041-2060/annual",
        },
        "/.../FIGURES/scorecard",
        ["pr", "tasmax"],
        metrics=["rmse", "bias_mean", "bias_r01", "ralsd"],
    )
"""

from __future__ import annotations

import copy
import os

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_evaluation.viz.plot_hexagon import metrics_dir, normalize_paths

EPS = 1e-12

# metric key -> (zarr store, diagnostic, use_abs) for the plotted *value*. All are
# errors (lower is better); bias fields are reduced to a scalar as mean(|field|).
VALUE_SRC = {
    "rmse": ("metrics_fast_float.zarr", "rmse_float", False),
    "ralsd": ("metrics_fast_float.zarr", "ralsd_float", False),
    "bias_mean": ("metrics_fast_field.zarr", "bias_mean_field", True),
    "bias_p01": ("metrics_fast_field.zarr", "bias_p1_field", True),
    "bias_p99": ("metrics_fast_field.zarr", "bias_p99_field", True),
    "bias_acorr": ("metrics_fast_field.zarr", "bias_acorr_field", True),
    "bias_rx1day": ("metrics_fast_field.zarr", "bias_max1day_field", True),
    "bias_r01": ("metrics_fast_pr_field.zarr", "bias_r1_field", True),
    "bias_sdii": ("metrics_fast_pr_field.zarr", "bias_sdii_field", True),
    "bias_p98wet": ("metrics_fast_pr_field.zarr", "bias_p98wet_field", True),
    # --- slow track (metrics_slow_*.zarr, written by `ml-eval slow`) ---------
    # Only the error-type diagnostics are exposed: pss / pcorr / scorr are
    # higher-is-better and would score inverted under `_scores`, which treats
    # every column as an error.
    "wd": ("metrics_slow_field.zarr", "wd_field", False),
    "wdh": ("metrics_slow_field.zarr", "wdh_field", False),
    "fcss": ("metrics_slow_field.zarr", "fcss_field", False),
    "lhd": ("metrics_slow_field.zarr", "lhd_field", False),
    "bias_annstd": ("metrics_slow_field.zarr", "bias_annstd_field", True),
    "bias_p90": ("metrics_slow_field.zarr", "bias_p90_field", True),
}

# metric key -> (store, diagnostic) of the *field* the CV is computed from. The
# float-only metrics map to their field counterpart where ml-eval writes one
# (RMSE); RALSD has none, so its cells carry no CV line.
CV_SRC = {
    "rmse": ("metrics_fast_field.zarr", "rmse_field"),
    "bias_mean": ("metrics_fast_field.zarr", "bias_mean_field"),
    "bias_p01": ("metrics_fast_field.zarr", "bias_p1_field"),
    "bias_p99": ("metrics_fast_field.zarr", "bias_p99_field"),
    "bias_acorr": ("metrics_fast_field.zarr", "bias_acorr_field"),
    "bias_rx1day": ("metrics_fast_field.zarr", "bias_max1day_field"),
    "bias_r01": ("metrics_fast_pr_field.zarr", "bias_r1_field"),
    "bias_sdii": ("metrics_fast_pr_field.zarr", "bias_sdii_field"),
    "bias_p98wet": ("metrics_fast_pr_field.zarr", "bias_p98wet_field"),
    "wd": ("metrics_slow_field.zarr", "wd_field"),
    "wdh": ("metrics_slow_field.zarr", "wdh_field"),
    "fcss": ("metrics_slow_field.zarr", "fcss_field"),
    "lhd": ("metrics_slow_field.zarr", "lhd_field"),
    "bias_annstd": ("metrics_slow_field.zarr", "bias_annstd_field"),
    "bias_p90": ("metrics_slow_field.zarr", "bias_p90_field"),
}

# Column header for every selectable metric key.
METRIC_LABELS = {
    "rmse": "RMSE",
    "ralsd": "RALSD",
    "bias_mean": "Mean bias",
    "bias_p01": "P01 bias",
    "bias_p99": "P99 bias",
    "bias_acorr": "Lag1 corr bias",
    "bias_rx1day": "Rx1day bias",
    "bias_r01": "Wet-day freq bias",
    "bias_sdii": "SDII bias",
    "bias_p98wet": "P98Wet bias",
    "wd": "Wasserstein",
    "wdh": "Wasserstein (PDF)",
    "fcss": "FCSS",
    "lhd": "LHD",
    "bias_annstd": "Ann. std bias",
    "bias_p90": "P90 bias",
}

# Default column set (metric keys, in order) when the caller passes no ``metrics``.
DEFAULT_METRICS_TEMP = [
    "rmse",
    "bias_mean",
    "bias_p01",
    "bias_p99",
    "ralsd",
    "bias_acorr",
]
DEFAULT_METRICS_PR = [
    "rmse",
    "bias_mean",
    "bias_r01",
    "bias_sdii",
    "bias_p98wet",
    "ralsd",
    "bias_acorr",
]


def var_skill_columns(var: str, metrics=None) -> list[tuple[str, str]]:
    """The ``(metric_key, column_label)`` list for ``var``.

    Parameters
    ----------
    var : str
        Variable name; selects the default column set when ``metrics`` is None
        (``pr`` gets the precip columns, everything else the temperature ones).
    metrics : iterable, optional
        Explicit column selection, in the order to plot. Each item is either a
        metric key from ``METRIC_LABELS`` -- labelled from that mapping -- or an
        explicit ``(metric_key, column_label)`` pair to override the header.

    Raises
    ------
    KeyError
        If a requested metric key is not one ml-eval writes (see ``VALUE_SRC``).
    """
    if metrics is None:
        metrics = DEFAULT_METRICS_PR if var == "pr" else DEFAULT_METRICS_TEMP
    columns = []
    for m in metrics:
        key, label = m if isinstance(m, (tuple, list)) else (m, None)
        if key not in VALUE_SRC:
            raise KeyError(
                f"unknown metric {key!r}; available: {', '.join(sorted(VALUE_SRC))}"
            )
        columns.append((key, label or METRIC_LABELS.get(key, key)))
    if not columns:
        raise ValueError("metrics must select at least one column.")
    return columns


def _open_diag(metrics_dir: str, store: str, diag: str):
    """The DataArray for one diagnostic, loaded into memory, or None if absent."""
    path = os.path.join(metrics_dir, store)
    if not os.path.isdir(path):
        return None
    try:
        with xr.open_zarr(path) as ds:
            return ds["value"].sel(diagnostic=diag).load()
    except (KeyError, ValueError, FileNotFoundError, OSError):
        return None


def _read_value(metrics_dir: str, key: str) -> float:
    """Domain scalar for one metric (mean of |field| for biases), NaN if missing."""
    store, diag, use_abs = VALUE_SRC[key]
    da = _open_diag(metrics_dir, store, diag)
    if da is None:
        return np.nan
    if use_abs:
        da = np.abs(da)
    return float(da.mean().values)


def _read_cv(metrics_dir: str, key: str) -> float:
    """Spatial coefficient of variation ``std(field) / mean(|field|)`` of a metric.

    NaN when the metric has no field counterpart (RALSD), the store is missing,
    or the field is uniformly zero. Uses ``mean(|field|)`` as the denominator so
    that a bias field straddling zero does not blow the ratio up.
    """
    src = CV_SRC.get(key)
    if src is None:
        return np.nan
    da = _open_diag(metrics_dir, *src)
    if da is None or da.size <= 1:
        return np.nan
    v = np.asarray(da.values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size <= 1:
        return np.nan
    denom = float(np.mean(np.abs(v)))
    if denom < EPS:
        return np.nan
    return float(np.std(v)) / denom


def read_var_skill_metrics(
    path: str, var: str, metrics=None
) -> dict[str, tuple[float, float]]:
    """Scorecard-column scalars for ``var`` from one ``ml-eval fast --output`` dir.

    ``path`` holds the metrics_fast_*.zarr stores written by a single
    ``ml-eval fast`` call. Returns ``{metric_key: (value, cv)}`` for the columns
    selected by ``metrics`` (default: the ``var`` column set); a metric whose
    store/diagnostic is missing comes back as ``(nan, nan)``. Depends only on
    ml-eval's output format -- no directory-layout assumptions.
    """
    return {
        k: (_read_value(path, k), _read_cv(path, k))
        for k, _ in var_skill_columns(var, metrics)
    }


def _scores(values: np.ndarray) -> np.ndarray:
    """Errors -> [0,1] performance scores with 1 = best (min error).

    All-equal (or single-model) columns score 1 everywhere -- no model is worse
    than any other. Non-finite values stay NaN so their cells render as blank.
    """
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    finite = np.isfinite(v)
    if not finite.any():
        return out
    best, worst = float(np.min(v[finite])), float(np.max(v[finite]))
    if worst - best < EPS:
        out[finite] = 1.0
        return out
    out[finite] = (worst - v[finite]) / (worst - best)
    return out


def _fmt_value(v: float) -> str:
    """Three significant figures, switching to exponent notation when needed."""
    return "" if not np.isfinite(v) else f"{v:.3g}"


def render_var_skill_scorecard(
    per_model,
    var,
    out_dir,
    *,
    models=None,
    metrics=None,
    season="annual",
    title=None,
    filename=None,
    cmap="Blues",
    annotate=True,
    text_color="gold",
):
    """Render one variable's performance-scorecard PNG from precomputed metrics.

    Parameters
    ----------
    per_model : dict
        ``{label: {metric_key: (value, cv)}}`` (e.g. from
        ``read_var_skill_metrics``). A plain ``{metric_key: value}`` mapping
        is accepted too -- the CV line is then omitted.
    var : str
        Variable name (selects the columns via ``var_skill_columns``).
    out_dir : str
        Directory for the PNG (created if needed).
    models : list of str, optional
        Subset/order of the rows (default: every key of ``per_model``, in order).
    metrics : iterable, optional
        Columns to plot, in order -- metric keys from ``METRIC_LABELS`` and/or
        ``(metric_key, column_label)`` pairs. Default: the ``var`` column set.
        A metric absent from ``per_model`` renders as a blank column.
    season : str
        Used for the title and the output file name.
    title : str, optional
        Figure title (default derived from ``var``/``season``).
    filename : str, optional
        Output filename (default ``var_skill_scorecard_<var>_<season>.png``).
    cmap : str, default "Blues"
        Sequential colormap for the normalised score (dark = best).
    annotate : bool, default True
        Print the raw value (and CV) inside each cell.
    text_color : str, default "gold"
        Colour of the in-cell annotations.

    Returns
    -------
    str or None
        Path of the written PNG, or None if there was nothing to plot.
    """
    if models is None:
        models = list(per_model)
    else:
        models = [m for m in models if m in per_model]
    if not models:
        return None

    columns = var_skill_columns(var, metrics)
    keys = [k for k, _ in columns]
    labels = [lbl for _, lbl in columns]

    def _cell(model, key):
        entry = per_model[model].get(key, (np.nan, np.nan))
        if isinstance(entry, tuple):
            return float(entry[0]), float(entry[1])
        return float(entry), np.nan  # plain {metric: value} mapping

    values = np.array([[_cell(m, k)[0] for k in keys] for m in models])
    cvs = np.array([[_cell(m, k)[1] for k in keys] for m in models])

    # Per-column normalisation: the comparison between models happens here.
    scores = np.column_stack([_scores(values[:, c]) for c in range(len(keys))])

    nrows, ncols = len(models), len(keys)
    fig, ax = plt.subplots(figsize=(1.15 * ncols + 4.0, 0.72 * nrows + 3.2))
    masked = np.ma.masked_invalid(scores)
    cmap_o = copy.copy(plt.get_cmap(cmap))  # copy: never mutate the global cmap
    cmap_o.set_bad("0.9")
    im = ax.imshow(masked, cmap=cmap_o, vmin=0.0, vmax=1.0, aspect="auto")

    if annotate:
        for r in range(nrows):
            for c in range(ncols):
                if not np.isfinite(values[r, c]):
                    continue
                txt = _fmt_value(values[r, c])
                if np.isfinite(cvs[r, c]):
                    txt += f"\nCV {cvs[r, c]:.2f}"
                ax.text(
                    c,
                    r,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    color=text_color,
                )

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(models)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Normalized performance score (1 = best)")

    head = title or f"Performance Scorecard — {var} — {season}"
    ax.set_title(head, fontsize=13, pad=12)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename or f"var_skill_scorecard_{var}_{season}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[var-skill-scorecard] {path}  (models: {', '.join(models)})")
    return path


def plot_var_skill_scorecard(
    paths,
    out,
    variables,
    *,
    metrics=None,
    season="annual",
    **kwargs,
):
    """Plot one performance scorecard per variable for an explicit set of paths.

    Convenience wrapper over ``read_var_skill_metrics`` +
    ``render_var_skill_scorecard``.

    Parameters
    ----------
    paths : mapping or sequence
        ``{label: path}`` (see ``plot_hexagon.normalize_paths``); ``path`` is the
        model's base directory and one variable's stores are read from
        ``<path>/<var>/``. One entry = one row, in the given order.
    out : str
        Output directory for the PNGs.
    variables : iterable of str
        Variables to plot (one scorecard each).
    metrics : iterable, optional
        Columns to plot, in order -- metric keys from ``METRIC_LABELS`` and/or
        ``(metric_key, column_label)`` pairs. Default: the per-variable set.
    season : str
        Label only -- title and output file name.
    **kwargs
        Forwarded to :func:`render_var_skill_scorecard` (``title``, ``filename``,
        ``cmap``, ``annotate``, ``text_color``).

    Returns
    -------
    list of str
        Paths of the PNGs written (one per variable that had data).
    """
    by_label = normalize_paths(paths)

    written = []
    for var in variables:
        per_model = {}
        for label, base in by_label.items():
            d = metrics_dir(base, var)
            if os.path.isdir(d):
                per_model[label] = read_var_skill_metrics(d, var, metrics)
        avail = [m for m in by_label if m in per_model]
        path = render_var_skill_scorecard(
            per_model, var, out, models=avail, metrics=metrics, season=season, **kwargs
        )
        if path is None:
            print(f"[var-skill-scorecard] {var}: no model directory found — skipping.")
        else:
            written.append(path)
    return written
