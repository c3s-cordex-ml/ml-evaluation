"""
Probabilistic-skill *triangle* scorecard for ensemble ML RCM emulators.

One figure: rows are the emulators, columns the predicted variables, and each
cell is a triangle whose three wedges carry the diagnostics ``ml-eval prob``
writes to ``metrics_prob_float.zarr`` -- read from the same explicit
``{label: path}`` spec as the rest of ``ml_evaluation.viz``.

    wedge 0 (top)         CRPS   Continuous Ranked Probability Score
    wedge 1 (lower left)  BS     Brier Score for the "> P50" event
    wedge 2 (lower right) SERAT  spread-error ratio

Why a triangle and not a hexagon: the three diagnostics do not share a scale or
even a sense.

  * CRPS carries the units of the predicted variable, BS and SERAT do not, so
    the raw numbers of one column say nothing about another;
  * CRPS and BS are errors (lower is better) while SERAT is best at **1** --
    below 1 the ensemble is under-dispersive, above it over-dispersive.

So the cell text is the **raw** value of each diagnostic, and only the *colour*
is comparable: SERAT is first folded to ``|log10(SERAT)|`` (0 = perfectly
dispersed, larger = worse either way), then every wedge is divided by the
largest value of that (metric, variable) across the plotted models. Lower is
better everywhere after that, so a dark cell is a good one.

CRPS and BS have deterministic counterparts -- on a single-member prediction
CRPS reduces to the mean absolute error and BS to the 0/1 indicator loss -- so
deterministic and generative emulators sit in the same figure. SERAT has none:
it needs at least two members, and ``ml-eval prob`` simply omits it, which shows
up here as a blank wedge.

Locating the data
-----------------
As in ``plot_hexagon``: ``{label: path}`` where ``path`` is one run's base
directory, and one variable's store is read from ``<path>/<var>/``. Nothing is
searched. ``season`` is a label for the title and the file name only.

One store can hold several scored configurations, because ``ml-eval prob
--n-members "2,4,..."`` stacks an ensemble-size sweep along an ``n_members``
dimension rather than splitting it across directories. Rows then differ by
``row_sel`` -- ``{label: {"n_members": N}}`` -- instead of by path, which is how
"the same emulator at 2, 4, 6, 8, 10 members" becomes five rows of one figure.

Public API
----------
prob_columns(metrics=None)
    The ``(metric_key, wedge_label)`` list, in wedge order.
read_prob_metrics(metrics_dir, metrics=None, *, sel=None)
    ``{metric_key: value}`` for one ``ml-eval prob --output`` directory, ``sel``
    picking one configuration out of a swept store. Depends only on ml-eval's
    output format -- layout-agnostic.
render_multivar_probskill_scorecard(per_model, out_dir, *, variables, models, season)
    Render from precomputed ``{label: {var: {metric: value}}}``.
plot_multivar_probskill_scorecard(paths, out, variables, *, season, row_sel=None)
    Read + render: one figure, models x variables.

Example
-------
    from ml_evaluation.viz.plot_multivar_probskill_scorecard import (
        plot_multivar_probskill_scorecard,
    )
    plot_multivar_probskill_scorecard(
        {
            "RESDIFF":       "/.../RESDIFF600_..._2041-2060/annual",
            "FLOW-MATCHING": "/.../FLOWMATCHING_..._2041-2060/annual",
        },
        "/.../FIGURES/scorecard_triangle_prob",
        ["tas", "tasmax", "pr"],
    )
"""

from __future__ import annotations

import os

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_evaluation.viz.plot_hexagon import metrics_dir, normalize_paths, select_member

EPS = 1e-12
PROB_STORE = "metrics_prob_float.zarr"

# metric key -> (store, diagnostic, wedge label). Every one is a domain scalar.
PROB_SRC = {
    "crps": (PROB_STORE, "crps_mean_float", "CRPS"),
    "bs": (PROB_STORE, "bs50p_float", "BS"),
    "serat": (PROB_STORE, "se_ssrat_float", "SERAT"),
}

# Wedge order, as drawn by ``_wedge``: 0 top, then counter-clockwise.
TRIANGLE_WEDGES = ["crps", "bs", "serat"]

# SERAT is not an error: it is best at 1, so it is folded before normalising.
BEST_AT_ONE = {"serat"}

# Dimensions a store may carry because ONE ``ml-eval prob`` call scored several
# configurations into it (``--n-members "2,4,..."``). A reader must select along
# these, never average: see :func:`_select_sweep`.
SWEEP_DIMS = ("n_members",)


class SweepSelectionError(Exception):
    """A requested sweep coordinate (e.g. ``n_members=7``) is not in the store."""


def prob_columns(metrics=None) -> list[tuple[str, str]]:
    """The ``(metric_key, wedge_label)`` list, in wedge order.

    Parameters
    ----------
    metrics : iterable, optional
        Explicit selection, in wedge order. Each item is a metric key from
        :data:`PROB_SRC` or an explicit ``(metric_key, label)`` pair. Default:
        CRPS, BS, SERAT.

    Raises
    ------
    ValueError
        If the selection is not exactly three wedges, or names a metric
        ``ml-eval prob`` does not write.
    """
    metrics = list(metrics) if metrics is not None else list(TRIANGLE_WEDGES)
    columns = []
    for m in metrics:
        key, label = m if isinstance(m, (tuple, list)) else (m, None)
        if key not in PROB_SRC:
            raise ValueError(
                f"unknown metric {key!r}; available: {', '.join(sorted(PROB_SRC))}"
            )
        columns.append((key, label or PROB_SRC[key][2]))
    if len(columns) != 3:
        raise ValueError(f"a triangle needs exactly 3 wedges, got {len(columns)}.")
    return columns


def _select_sweep(da, sel, label=""):
    """Reduce the sweep dimensions of a stored diagnostic to a single value.

    ``ml-eval prob --n-members "2,4,..."`` stacks one bundle per sub-ensemble
    size along an ``n_members`` dimension, so a store can hold several scored
    configurations at the same path. ``sel`` picks one of them -- ``{"n_members":
    4}`` reads the 4-member score.

    Selecting is not optional when such a dimension is present: averaging over
    it would silently return the mean of a diagnostic ACROSS ensemble sizes,
    which is not a diagnostic of anything. So a leftover sweep dimension with no
    ``sel`` falls back to its LARGEST coordinate -- the full ensemble, which is
    what a figure means by the model's score -- and says so.

    Parameters
    ----------
    da : xr.DataArray
        A stored diagnostic, already reduced to one ``diagnostic`` value.
    sel : dict | None
        Coordinate selection, e.g. ``{"n_members": 4}``. Entries naming a
        dimension the store does not have are ignored.
    label : str
        Caller's label, used only in the fallback message.

    Returns
    -------
    xr.DataArray
        The diagnostic with every sweep dimension resolved.

    Raises
    ------
    SweepSelectionError
        If a requested coordinate value is not in the store. Deliberately NOT a
        KeyError: :func:`read_prob_metrics` turns those into NaN (a missing
        metric is normal), which would render a typo'd size as an innocent blank
        wedge instead of an error.
    """
    if sel:
        wanted = {k: v for k, v in sel.items() if k in da.dims}
        for dim, value in wanted.items():
            if value not in da[dim].values:
                raise SweepSelectionError(
                    f"{label or 'store'}: {dim}={value} not scored; available: "
                    f"{list(da[dim].values)}"
                )
        if wanted:
            da = da.sel(wanted)
    for dim in SWEEP_DIMS:
        if dim in da.dims:
            largest = da[dim].max().item()
            print(
                f"[probskill] {label}: {dim} not selected; using the largest "
                f"({dim}={largest})"
            )
            da = da.sel({dim: largest})
    return da


def read_prob_metrics(
    metrics_dir: str, metrics=None, *, member_dim: str = "member", sel=None, label=""
) -> dict[str, float]:
    """Probabilistic scalars for one ``ml-eval prob --output`` directory.

    ``metrics_dir`` holds the ``metrics_prob_float.zarr`` store written by a
    single ``ml-eval prob`` call. Returns ``{metric_key: value}`` in raw units; a
    metric the store does not carry comes back as NaN -- which is the normal case
    for ``serat`` on a single-member prediction, since ``ml-eval prob`` omits it
    rather than writing a meaningless number. Depends only on ml-eval's output
    format -- no directory-layout assumptions.

    Parameters
    ----------
    metrics_dir : str
        Directory holding the store.
    metrics : iterable, optional
        Wedge selection (see :func:`prob_columns`).
    member_dim : str
        Ensemble dimension name in the stored metrics, if any.
    sel : dict, optional
        Selection along a sweep dimension of the store, e.g. ``{"n_members":
        4}`` to read the 4-member score of an ensemble-size sweep. See
        :func:`_select_sweep` for what happens without it.
    label : str
        Caller's label, used only in messages.
    """
    out = {}
    for key, _ in prob_columns(metrics):
        store, diag, _ = PROB_SRC[key]
        path = os.path.join(metrics_dir, store)
        if not os.path.isdir(path):
            out[key] = np.nan
            continue
        try:
            with xr.open_zarr(path) as ds:
                da = select_member(ds["value"].sel(diagnostic=diag), member_dim, 0)
                da = _select_sweep(da, sel, label=f"{label}/{key}".strip("/"))
        except (KeyError, ValueError, FileNotFoundError, OSError):
            out[key] = np.nan
            continue
        out[key] = float(da.mean().values)
    return out


def _comparable(values: np.ndarray, key: str) -> np.ndarray:
    """One column's values on a lower-is-better scale.

    Errors pass through; SERAT is folded to ``|log10(SERAT)|`` so that 1 (a
    perfectly dispersed ensemble) becomes 0 and both under- and over-dispersion
    grow away from it.
    """
    v = np.asarray(values, dtype=float)
    if key not in BEST_AT_ONE:
        return v
    with np.errstate(divide="ignore", invalid="ignore"):
        folded = np.abs(np.log10(np.where(v > 0, v, np.nan)))
    return folded


def _normalise(values: np.ndarray, key: str) -> np.ndarray:
    """Column values -> [0, 1] by dividing by the column's largest magnitude.

    Keeps the *relative* standing of the models while making columns whose units
    and magnitudes differ wildly (CRPS in K or mm/day, BS a probability, SERAT a
    ratio) share one colour bar. All-NaN or all-zero columns come back as NaN.
    """
    v = _comparable(values, key)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.full(v.shape, np.nan)
    top = float(np.max(np.abs(finite)))
    if top < EPS:
        return np.where(np.isfinite(v), 0.0, np.nan)
    return v / top


def _fmt(v: float) -> str:
    """Compact label for a raw diagnostic value."""
    if not np.isfinite(v):
        return ""
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 0.005 or a == 0:
        return f"{v:.2f}"
    return f"{v:.1e}"   # a genuinely tiny, non-zero value


def _wedge(ax, cx, cy, i, radius, facecolor, edgecolor="white", lw=0.6):
    """Draw wedge ``i`` (0 = top, then counter-clockwise) of a triangular cell.

    The cell is a down-pointing triangle: wedge 0 spans its flat top edge,
    wedge 1 its lower-left side and wedge 2 its lower-right side. Returns the
    wedge centroid, where the value is written.
    """
    from matplotlib.patches import Polygon

    # vertices of the down-pointing triangle, counter-clockwise from top-left
    verts = [
        (cx - radius, cy + radius * 0.6),  # top-left
        (cx + radius, cy + radius * 0.6),  # top-right
        (cx, cy - radius),  # bottom apex
    ]
    # wedge i is (centre, verts[i], verts[i-1]): 0 = top edge, 1 = left, 2 = right
    pairs = [(verts[0], verts[1]), (verts[2], verts[0]), (verts[1], verts[2])]
    a, b = pairs[i]
    ax.add_patch(
        Polygon(
            [(cx, cy), a, b],
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            lw=lw,
        )
    )
    return ((cx + a[0] + b[0]) / 3.0, (cy + a[1] + b[1]) / 3.0)


def render_multivar_probskill_scorecard(
    per_model,
    out_dir,
    *,
    variables,
    models=None,
    metrics=None,
    season="annual",
    title=None,
    filename=None,
    cmap="viridis",
    annotate=True,
):
    """Render the triangle scorecard from precomputed values.

    Parameters
    ----------
    per_model : mapping
        ``{label: {var: {metric_key: raw value}}}``.
    out_dir : str
        Output directory (created if missing).
    variables : iterable of str
        Columns, in order.
    models : iterable of str, optional
        Rows; defaults to the order of ``per_model``.
    metrics : iterable, optional
        Wedge selection (see :func:`prob_columns`).
    season : str
        Label only -- title and file name.
    title, filename, cmap, annotate
        Figure title, output name, colormap, and whether to print the raw value
        in each wedge.

    Returns
    -------
    str
        Path of the PNG written.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    columns = prob_columns(metrics)
    variables = list(variables)
    models = [m for m in (models if models is not None else per_model) if m in per_model]
    if not models or not variables:
        raise ValueError("nothing to plot: need at least one model and one variable.")

    # normalise per (variable, metric) column, across the plotted models
    scores = {}
    for var in variables:
        for key, _ in columns:
            raw = np.array(
                [per_model[m].get(var, {}).get(key, np.nan) for m in models], dtype=float
            )
            scores[(var, key)] = _normalise(raw, key)

    allv = np.concatenate([v for v in scores.values()])
    finite = allv[np.isfinite(allv)]
    norm = Normalize(float(np.min(finite)) if finite.size else 0.0, 1.0)
    cmap_o = plt.get_cmap(cmap)
    nan_color = "0.9"

    ncols, nrows = len(variables), len(models)
    R = 0.46
    fig, ax = plt.subplots(figsize=(1.55 * ncols + 3.0, 1.25 * nrows + 3.0))
    for r, m in enumerate(models):
        for c, var in enumerate(variables):
            cx, cy = c, -r
            for i, (key, _) in enumerate(columns):
                value = per_model[m].get(var, {}).get(key, np.nan)
                score = scores[(var, key)][r]
                fc = cmap_o(norm(score)) if np.isfinite(score) else nan_color
                tx, ty = _wedge(ax, cx, cy, i, R, fc)
                if annotate and np.isfinite(value):
                    ax.text(
                        tx,
                        ty,
                        _fmt(value),
                        ha="center",
                        va="center",
                        fontsize=6,
                        # viridis is dark at the good end: light text there only
                        color="white" if np.isfinite(score) and score < 0.5 else "black",
                    )

    ax.set_xlim(-0.7, ncols - 0.3)
    ax.set_ylim(-(nrows - 1) - 0.8, 0.8)
    ax.set_aspect("equal")
    ax.set_xticks(range(ncols))
    ax.set_xticklabels([v.upper() for v in variables])
    ax.set_yticks([-r for r in range(nrows)])
    ax.set_yticklabels(models)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=cmap_o)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("normalised metric wrt max  (lower = better)")

    order = ", ".join(f"{i}:{lab}" for i, (_, lab) in enumerate(columns))
    ax.set_title(
        f"{title or f'Probabilistic scorecard — {season}'}\n"
        f"wedges (from top, counter-clockwise) — {order}\n"
        "numbers are raw values; colour is |log10| for SERAT, then / column max",
        fontsize=10,
    )

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename or f"probskill_scorecard_{season}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[probskill] {path}  (models: {', '.join(models)})")
    return path


def plot_multivar_probskill_scorecard(
    paths,
    out,
    variables,
    *,
    metrics=None,
    season="annual",
    member_dim="member",
    row_sel=None,
    **kwargs,
):
    """Plot the probabilistic triangle scorecard for an explicit set of paths.

    Convenience wrapper over :func:`read_prob_metrics` +
    :func:`render_multivar_probskill_scorecard`.

    Parameters
    ----------
    paths : mapping or sequence
        ``{label: path}`` (see ``plot_hexagon.normalize_paths``); ``path`` is the
        model's base directory and one variable's store is read from
        ``<path>/<var>/``. One entry = one row, in the given order.
    out : str
        Output directory for the PNG.
    variables : iterable of str
        Variables to score (one column each).
    metrics : iterable, optional
        Wedge selection (see :func:`prob_columns`).
    season : str
        Label only -- title and output file name.
    member_dim : str
        Ensemble dimension name in the stored metrics, if any. The probabilistic
        diagnostics already reduce the ensemble, so this is only a safety net.
    row_sel : mapping, optional
        ``{label: {"n_members": N}}`` -- per-row selection along a sweep
        dimension of the store (see :func:`_select_sweep`). This is what lets
        several rows share ONE path and differ only by the configuration read
        out of it, e.g. one row per ensemble size of the same emulator::

            NS = [2, 4, 6, 8, 10]
            paths   = {f"RESDIFF (n={n})": run_dir for n in NS}
            row_sel = {f"RESDIFF (n={n})": {"n_members": n} for n in NS}

        Labels absent from the mapping are read with no selection.
    **kwargs
        Forwarded to :func:`render_multivar_probskill_scorecard` (``title``,
        ``filename``, ``cmap``, ``annotate``).

    Returns
    -------
    str or None
        Path of the PNG written, or None when no source had data.
    """
    by_label = normalize_paths(paths)

    per_model = {}
    for label, base in by_label.items():
        per_var = {}
        for var in variables:
            d = metrics_dir(base, var)
            if not os.path.isdir(d):
                print(f"[probskill] skip {label} / {var}: {d} not found")
                continue
            values = read_prob_metrics(
                d,
                metrics,
                member_dim=member_dim,
                sel=(row_sel or {}).get(label),
                label=f"{label} / {var}",
            )
            if all(not np.isfinite(v) for v in values.values()):
                print(f"[probskill] skip {label} / {var}: no prob store in {d}")
                continue
            per_var[var] = values
        if per_var:
            per_model[label] = per_var

    if not per_model:
        print("[probskill] no metrics found -- run `ml-eval prob` first.")
        return None

    return render_multivar_probskill_scorecard(
        per_model,
        out,
        variables=list(variables),
        models=[m for m in by_label if m in per_model],
        metrics=metrics,
        season=season,
        **kwargs,
    )
