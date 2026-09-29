"""
Interannual *time-series* panel for ML RCM emulators.

One figure, one panel: a single variable and a single index, one line per source,
each year a single value -- read from the ``metrics_signal_series.zarr`` bundles
an ``ml-eval signal`` call writes. It is the time-resolved companion of the
scorecards: those collapse a run to scalars, this one keeps the ``year`` axis
``ml-eval signal`` already stores.

Each ``ml-eval signal`` run writes, per index, the raw annual spatial-mean
series over its two periods concatenated on one ``year`` axis
(``<index>_series``, see ``ml_evaluation.evaluation.signal._annual_series``), so
the historical and future stretches of one run come out of a single store. Years
that are not contiguous are drawn as separate line segments rather than joined
across the gap.

Two quantities, chosen with ``baseline``:

    values          (``baseline=None``) -- the index itself, in data units
    signal/anomaly  (``baseline=(y0, y1)``) -- each source minus *its own* mean
                    over the baseline years, i.e. the climate-change signal with
                    each source's own bias removed

When the baseline period lives in a *different* run directory than the one being
plotted (a separate historical evaluation, say), pass ``baseline_paths`` -- those
series are merged onto the same year axis before the baseline mean is taken, so
the baseline years are both subtracted *and* drawn.

Locating the data
-----------------
As in ``plot_hexagon``, sources come from an explicit ``{label: path}`` spec and
one variable's stores are read from ``<path>/<var>/``. Nothing is searched.
``season`` is a label for the title and the file name only.

Public API
----------
series_indices(var)
    The index keys ``ml-eval signal`` writes a series for, in plotting order.
read_series(metrics_dir, index)
    ``(years, values)`` for one ``ml-eval signal --output`` directory. Depends
    only on ml-eval's output format -- layout-agnostic.
render_series(per_source, var, index, out_dir, *, models, reference, driver, ...)
    Render the panel from precomputed ``{label: (years, values)}``.
plot_var_series(paths, out, var, index, *, reference, driver, baseline, ...)
    Read + render: one PNG for one (variable, index).

Example
-------
    from ml_evaluation.viz.plot_var_series import plot_var_series
    plot_var_series(
        {
            "RCM truth":   "/.../rcm/CNRM-ALADIN63_CNRM-CM5_2041-2060/annual",
            "UNet-single": "/.../UNET-SINGLE_..._2041-2060/annual",
            "ResDiff":     "/.../RESDIFF600_..._2041-2060/annual",
        },
        "/.../FIGURES/series",
        "tasmax",
        "mean",
        reference="RCM truth",
        baseline=(1981, 2000),
        units="K",
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
from ml_evaluation.viz.plot_multivar_signal_scorecard import (
    INDEX_LABEL,
    signal_indices,
)

SERIES_STORE = "metrics_signal_series.zarr"

# Categorical line colours, assigned in this fixed order and never cycled: a
# source keeps its colour whatever else is on the panel. The reference (RCM
# truth) and the driver (GCM) are drawn in ink, not in a categorical hue -- they
# are not "one more model".
SERIES_COLORS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
REFERENCE_STYLE = {"color": "black", "lw": 2.4, "ls": "-", "zorder": 5}
DRIVER_STYLE = {"color": "0.45", "lw": 1.8, "ls": "--", "zorder": 4}
MODEL_STYLE = {"lw": 1.6, "ls": "-", "zorder": 3}
# A *variant* of a plotted source -- the same emulator under the other boundary
# framework, say -- reuses its base's colour and is dashed, so the pair reads as
# one model in two experiments rather than as two unrelated models.
VARIANT_STYLE = {"ls": "--", "lw": 1.4}


def series_indices(var: str) -> list[str]:
    """The index keys ``ml-eval signal`` writes a series for, in plotting order.

    mean/P01/P99 for non-precip variables, R01/SDII/RX1day for precip -- the same
    catalog the signal scorecards use, so a series can be drawn for any index
    those figures score.
    """
    return [k for k, _ in signal_indices(var)]


def _check_index(var: str, index: str | None) -> str:
    """Validate ``index`` against what ml-eval stores for ``var`` (None -> first)."""
    available = series_indices(var)
    if index is None:
        return available[0]
    if index not in available:
        raise ValueError(
            f"index '{index}' is not stored for '{var}'; ml-eval signal writes "
            f"{available} for this variable."
        )
    return index


def read_series(metrics_dir: str, index: str) -> tuple[np.ndarray, np.ndarray]:
    """Annual series of one index from an ``ml-eval signal --output`` directory.

    ``metrics_dir`` holds the ``metrics_signal_series.zarr`` store written by a
    single ``ml-eval signal`` call, whose ``value`` array is stacked along
    ``diagnostic`` and indexed by ``year``.

    Returns ``(years, values)`` sorted by year, both empty when the store or the
    diagnostic is missing. Depends only on ml-eval's output format -- no
    directory-layout assumptions.
    """
    return read_diagnostic(metrics_dir, f"{index}_series")


def read_diagnostic(metrics_dir: str, diagnostic: str):
    """One diagnostic of ``metrics_signal_series.zarr``, by its exact name.

    ``(years, values)`` sorted by year, both empty when the store or the
    diagnostic is missing. :func:`read_series` is the by-index wrapper; the
    ensemble-spread diagnostics are read through this one.
    """
    empty = (np.array([], dtype=int), np.array([], dtype=float))
    path = os.path.join(metrics_dir, SERIES_STORE)
    if not os.path.isdir(path):
        return empty
    try:
        with xr.open_zarr(path) as ds:
            da = ds["value"].sel(diagnostic=diagnostic)
            years = np.asarray(ds["year"].values, dtype=int)
            values = np.asarray(da.values, dtype=float)
    except (KeyError, ValueError, FileNotFoundError, OSError):
        return empty
    order = np.argsort(years)
    return years[order], values[order]


SPREADS = ("minmax", "std")


def member_spread(series, spread="minmax"):
    """Ensemble mean and spread band from several members' series.

    ``ml-eval signal`` collapses the ensemble when it computes a signal, so a
    per-member view needs one run per member (``ml-eval signal --member-num k``,
    each with its own ``--output``). Give this function those runs' series and it
    returns the line to draw and the band around it, aligned on the union of
    their year axes.

    Parameters
    ----------
    series : sequence of (years, values)
        One entry per member, as returned by :func:`read_series`.
    spread : {"minmax", "std"}
        ``"minmax"`` bands the full envelope of the members, ``"std"`` the mean
        +/- one standard deviation across them.

    Returns
    -------
    (years, mean, lo, hi)
        Empty arrays when nothing was readable. With a single member the band
        collapses onto the line.
    """
    if spread not in SPREADS:
        raise ValueError(f"spread must be one of {SPREADS}, got {spread!r}.")
    series = [(y, v) for y, v in series if y.size]
    if not series:
        empty = np.array([], dtype=int)
        return empty, empty.astype(float), empty.astype(float), empty.astype(float)

    years = np.unique(np.concatenate([y for y, _ in series]))
    stack = np.full((len(series), years.size), np.nan)
    for i, (y, v) in enumerate(series):
        stack[i, np.searchsorted(years, y)] = v

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
        if spread == "minmax":
            lo, hi = np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)
        else:
            sd = np.nanstd(stack, axis=0)
            lo, hi = mean - sd, mean + sd
    return years, mean, lo, hi


def read_series_spread(metrics_dir, index, spread="minmax"):
    """The stored ensemble spread of one index, or empty when there is none.

    ``ml-eval signal`` records ``<index>_series_{min,max,std}`` alongside the
    mean series whenever the source had more than one member and no single
    member was selected -- so a spread band costs no extra runs and no extra
    stores. Returns ``(years, lo, hi)``, empty when the run was single-member
    (or predates that output).

    Parameters
    ----------
    metrics_dir : str
        Directory holding ``metrics_signal_series.zarr``.
    index : str
        Index key, as in :func:`read_series`.
    spread : {"minmax", "std"}
        ``"minmax"`` reads the stored envelope; ``"std"`` builds the band as the
        mean +/- one stored standard deviation.
    """
    if spread not in SPREADS:
        raise ValueError(f"spread must be one of {SPREADS}, got {spread!r}.")
    empty = (
        np.array([], dtype=int),
        np.array([], dtype=float),
        np.array([], dtype=float),
    )

    if spread == "minmax":
        ylo, lo = read_diagnostic(metrics_dir, f"{index}_series_min")
        yhi, hi = read_diagnostic(metrics_dir, f"{index}_series_max")
        if not ylo.size or not np.array_equal(ylo, yhi):
            return empty
        return ylo, lo, hi

    ym, mean = read_diagnostic(metrics_dir, f"{index}_series")
    ys, sd = read_diagnostic(metrics_dir, f"{index}_series_std")
    if not ys.size or not np.array_equal(ym, ys):
        return empty
    return ym, mean - sd, mean + sd


def _merge_series(primary, extra):
    """Union of two ``(years, values)`` series on one year axis; primary wins.

    Used to graft a baseline-period run onto the run being plotted, so the
    baseline years are both subtracted and drawn.
    """
    py, pv = primary
    ey, ev = extra
    keep = ~np.isin(ey, py)
    years = np.concatenate([py, ey[keep]])
    values = np.concatenate([pv, ev[keep]])
    order = np.argsort(years)
    return years[order], values[order]


def _baseline_mean(years, values, baseline) -> float:
    """Mean of ``values`` over the closed year window ``baseline``."""
    y0, y1 = baseline
    sel = (years >= y0) & (years <= y1) & np.isfinite(values)
    return float(np.mean(values[sel])) if sel.any() else np.nan


def _select_years(years, values, window):
    """Restrict to a closed ``(y0, y1)`` window or an explicit set of years.

    A *tuple* is always the closed window; any other iterable (list, range,
    array) is always the explicit set -- so ``[1981, 2060]`` means those two
    years, not everything between them.
    """
    if window is None:
        return years, values
    if isinstance(window, tuple):
        if len(window) != 2:
            raise ValueError(
                f"a tuple year window must be (y0, y1); got {window!r}. Pass a "
                "list for an explicit set of years."
            )
        sel = (years >= window[0]) & (years <= window[1])
    else:
        sel = np.isin(years, np.asarray(list(window), dtype=int))
    return years[sel], values[sel]


def _break_gaps(years, values):
    """Insert NaNs where the year axis jumps, so the line breaks at the gap.

    A signal run holds two disjoint periods on one axis (e.g. 1981-2000 and
    2041-2060); without this the plot would draw a straight line across the
    forty-year hole.
    """
    if years.size == 0:
        return years.astype(float), values
    out_y: list[float] = [float(years[0])]
    out_v: list[float] = [float(values[0])]
    for y, v in zip(years[1:], values[1:]):
        if y - out_y[-1] > 1:
            out_y.append(float(out_y[-1]) + 0.5)
            out_v.append(np.nan)
        out_y.append(float(y))
        out_v.append(float(v))
    return np.asarray(out_y), np.asarray(out_v)


def render_series(
    per_source,
    var,
    index,
    out_dir,
    *,
    models=None,
    bands=None,
    variant_of=None,
    reference=None,
    driver=None,
    years=None,
    units=None,
    is_anomaly=False,
    baseline=None,
    season="annual",
    title=None,
    filename=None,
    colors=SERIES_COLORS,
    figsize=None,
    legend_ncol=None,
):
    """Render the interannual panel from precomputed series.

    Parameters
    ----------
    per_source : dict
        ``{label: (years, values)}`` (e.g. from :func:`read_series`), already in
        the units to plot and already anomalised if that is what is wanted.
    var, index : str
        Variable and index key -- labels for the title, the axis and the file
        name.
    out_dir : str
        Directory for the PNG (created if needed).
    models : list of str, optional
        Subset/order of the lines (default: every key of ``per_source``).
    bands : mapping, optional
        ``{label: (years, lo, hi)}`` -- an ensemble spread drawn as a shaded band
        under that label's line, in the line's own colour. Labels without an
        entry are drawn as a plain line.
    variant_of : mapping, optional
        ``{variant_label: base_label}``. A variant is drawn in its base's colour
        and dashed (``VARIANT_STYLE``), and consumes no categorical colour of its
        own -- the way to overlay the same emulators under a second experiment
        (e.g. imperfect boundaries) without doubling the palette. A variant whose
        base is not plotted falls back to a colour of its own.
    reference : str, optional
        Label drawn as the truth line (thick black) and listed first in the
        legend -- typically the RCM.
    driver : str, optional
        Label drawn as the driving line (dashed grey), typically the GCM.
    years : tuple or iterable, optional
        Closed ``(y0, y1)`` window (a tuple) or the explicit years to display
        (any other iterable).
    spread : {"minmax", "std"}, optional
        Shade the ensemble spread of the labels given as several paths:
        ``"minmax"`` the full envelope, ``"std"`` the mean +/- one standard
        deviation. None draws the mean line alone.
    units : str, optional
        Unit label for the y axis.
    is_anomaly : bool, default False
        Whether the values are anomalies (changes the axis and title wording).
    baseline : tuple, optional
        The ``(y0, y1)`` the anomalies are relative to -- quoted in the title.
    season : str, default "annual"
        Label only -- title and output file name.
    title, filename, colors, figsize, legend_ncol
        Figure title, output name, categorical colour order (assigned in order,
        never cycled), figure size, legend columns.

    Returns
    -------
    str or None
        Path of the written PNG, or None if there was nothing to plot.
    """
    if models is None:
        models = list(per_source)
    else:
        models = [m for m in models if m in per_source]
    # reference first, driver second: the two context lines lead the legend.
    for special in (driver, reference):
        if special is not None and special in models:
            models = [special] + [m for m in models if m != special]
    models = [m for m in models if per_source[m][0].size]
    if not models:
        return None
    n_colored = len(
        [
            m
            for m in models
            if m not in (reference, driver) and m not in (variant_of or {})
        ]
    )
    if n_colored > len(colors):
        raise ValueError(
            f"{n_colored} colour-carrying sources exceed the {len(colors)} "
            "categorical colours; "
            "plot fewer models, pair them with variant_of=, or pass a longer "
            "colors= sequence."
        )

    fig, ax = plt.subplots(figsize=figsize or (9.0, 4.6))
    # Colours are handed to base labels only; a variant reuses its base's.
    variant_of = dict(variant_of or {})
    color_of, slot = {}, 0
    for label in models:
        if label in (reference, driver) or label in variant_of:
            continue
        color_of[label] = colors[slot % len(colors)]
        slot += 1
    for label in models:
        base = variant_of.get(label)
        if base is None:
            continue
        if base in color_of:
            color_of[label] = color_of[base]
        else:  # the base was not plotted: the variant stands on its own
            color_of[label] = colors[slot % len(colors)]
            slot += 1

    for label in models:
        y, v = _select_years(*per_source[label], years)
        if not y.size:
            continue
        if label == reference:
            style = dict(REFERENCE_STYLE)
        elif label == driver:
            style = dict(DRIVER_STYLE)
        else:
            style = dict(MODEL_STYLE, color=color_of[label])
            if label in variant_of:
                style.update(VARIANT_STYLE)
        py, pv = _break_gaps(y, v)
        band = (bands or {}).get(label)
        if band is not None:
            by, blo = _break_gaps(*_select_years(band[0], band[1], years))
            _, bhi = _break_gaps(*_select_years(band[0], band[2], years))
            ax.fill_between(
                by,
                blo,
                bhi,
                color=style.get("color", "0.4"),
                alpha=0.20,
                lw=0,
                zorder=1,
            )
        ax.plot(py, pv, label=label, **style)

    if is_anomaly:
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=1)

    name = INDEX_LABEL.get(index, index)
    unit_txt = f"  [{units}]" if units else ""
    ax.set_ylabel(
        f"{var} {name} anomaly{unit_txt}" if is_anomaly else f"{var} {name}{unit_txt}"
    )
    ax.set_xlabel("year")
    ax.grid(color="0.9", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.6")
    ax.tick_params(colors="0.3", length=3)

    if title is None:
        head = (
            f"Climate-change signal — {var} {name} anomaly from the "
            f"{baseline[0]}-{baseline[1]} mean"
            if is_anomaly and baseline
            else f"{var} — {name} — {season}"
        )
    else:
        head = title
    ax.set_title(head, fontsize=12, pad=10)

    # A legend is always present for >=2 lines; identity is never colour-alone.
    if len(models) > 1:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=legend_ncol or min(len(models), 5),
            frameon=False,
            fontsize=9,
        )

    os.makedirs(out_dir, exist_ok=True)
    suffix = "_anom" if is_anomaly else ""
    path = os.path.join(
        out_dir, filename or f"series_{var}_{index}{suffix}_{season}.png"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[series] {path}  (lines: {', '.join(models)})")
    return path


def plot_var_series(
    paths,
    out,
    var,
    index=None,
    *,
    reference=None,
    reference_path=None,
    driver=None,
    driver_path=None,
    baseline=None,
    baseline_paths=None,
    years=None,
    scale=1.0,
    units=None,
    spread=None,
    season="annual",
    **kwargs,
):
    """Plot one interannual panel for one variable and one index.

    Convenience wrapper over :func:`read_series` + :func:`render_series`.

    Parameters
    ----------
    paths : mapping or sequence
        ``{label: path}`` (see ``plot_hexagon.normalize_paths``); ``path`` is the
        base directory of that source's ``ml-eval signal`` run and the variable's
        store is read from ``<path>/<var>/``. One entry = one line, in the given
        order.

        A label's value may instead be a **sequence of paths**, one per ensemble
        member: the line is then their mean and, with ``spread`` set, a shaded
        band shows their spread. ``ml-eval signal`` collapses the ensemble, so
        those per-member runs come from ``ml-eval signal --member-num k``, each
        written to its own ``--output``.
    out : str
        Output directory for the PNG.
    var : str
        Variable to plot (one panel).
    index : str, optional
        Index key to plot -- one of :func:`series_indices` for the variable
        (``mean``/``p01``/``p99``, or ``r01``/``sdii``/``rx1day`` for precip).
        Default: the variable's first index.
    reference : str, optional
        Label of the truth line (the RCM), drawn thick black and first in the
        legend. Optional: with None no line is singled out.
    reference_path : str, optional
        Base directory of the reference's run. Required only when ``reference``
        is not already a label of ``paths``; otherwise it overrides it.
    driver : str, optional
        Label of the driving line (the GCM), drawn dashed grey.
    driver_path : str, optional
        Base directory of the driver's run, as ``reference_path``.
    baseline : tuple, optional
        ``(y0, y1)`` closed year window. Given, the panel shows the *signal*:
        every source minus its own mean over those years, so each source's bias
        drops out and only its change remains. None -> plot the values
        themselves.
    baseline_paths : mapping or sequence, optional
        ``{label: path}`` of runs holding the baseline period, for when it is not
        in the same run as the years being plotted (a separate historical
        evaluation). Merged onto the same year axis before the mean is taken, so
        those years are subtracted *and* drawn. Labels absent here fall back to
        their ``paths`` entry. Ignored without ``baseline``.
    years : tuple or iterable, optional
        Closed ``(y0, y1)`` window (a tuple) or the explicit years to display
        (any other iterable). Applied after the baseline is subtracted, so a
        baseline outside the window still works.
    scale : float, default 1.0
        Multiplies the values -- e.g. 86400 to show precip stored in
        kg m-2 s-1 as mm/day. Applied before the baseline is removed.
    spread : {"minmax", "std"}, optional
        Shade the ensemble spread of the labels given as several paths:
        ``"minmax"`` the full envelope, ``"std"`` the mean +/- one standard
        deviation. None draws the mean line alone.
    units : str, optional
        Unit label for the y axis.
    season : str, default "annual"
        Label only -- title and output file name.
    **kwargs
        Forwarded to :func:`render_series` (``title``, ``filename``, ``colors``,
        ``figsize``, ``legend_ncol``, ``variant_of``).

    Returns
    -------
    str or None
        Path of the PNG written, or None if no source had data.
    """
    # A label may carry several member runs. normalize_paths stringifies its
    # values, so those are pulled out first and put back after normalisation.
    entries = paths.items() if isinstance(paths, dict) else None
    members = {}
    if entries is not None:
        flat = {}
        for label, spec in entries:
            if isinstance(spec, (list, tuple, set)):
                members[str(label)] = [str(p) for p in spec]
                flat[label] = next(iter(spec))  # a representative, for validation
            else:
                flat[label] = spec
        paths = flat

    by_label = normalize_paths(paths)
    by_label.update(members)
    index = _check_index(var, index)

    for name, label, override in (
        ("reference", reference, reference_path),
        ("driver", driver, driver_path),
    ):
        if override is not None:
            if label is None:
                raise ValueError(f"{name}_path= given without {name}=.")
            by_label[label] = override
        elif label is not None and label not in by_label:
            raise KeyError(
                f"{name} '{label}' is neither a label of paths "
                f"({', '.join(by_label)}) nor given via {name}_path=."
            )

    base_by_label = normalize_paths(baseline_paths) if baseline_paths else {}
    if base_by_label and baseline is None:
        print("[series] baseline_paths given without baseline= — ignored.")

    def _one(base):
        """One run's series, or empty when it is not on disk."""
        d = metrics_dir(base, var)
        if not os.path.isdir(d):
            return np.array([], dtype=int), np.array([], dtype=float)
        return read_series(d, index)

    per_source: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    bands: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, base in by_label.items():
        if isinstance(base, (list, tuple, set)):
            # several member runs: the line is their mean, the band their spread
            members = [_one(b) for b in base]
            years_v, values, lo, hi = member_spread(members, spread or "minmax")
            if spread is not None and years_v.size:
                bands[label] = (years_v, lo, hi)
            d = ", ".join(str(b) for b in base)
        else:
            d = metrics_dir(base, var)
            years_v, values = _one(base)
            if spread is not None and os.path.isdir(d):
                # an ensemble-mean signal run stores its own spread
                by, blo, bhi = read_series_spread(d, index, spread)
                if by.size:
                    bands[label] = (by, blo, bhi)
        if baseline is not None and label in base_by_label:
            bd = metrics_dir(base_by_label[label], var)
            if os.path.isdir(bd):
                years_v, values = _merge_series(
                    (years_v, values), read_series(bd, index)
                )
        if not years_v.size:
            print(f"[series] skip {label}: no {index} series under {d}")
            continue
        values = values * scale
        if label in bands:
            by, blo, bhi = bands[label]
            bands[label] = (by, blo * scale, bhi * scale)
        if baseline is not None:
            base_mean = _baseline_mean(years_v, values, baseline)
            if not np.isfinite(base_mean):
                print(
                    f"[series] skip {label}: no data in the baseline window "
                    f"{baseline[0]}-{baseline[1]}"
                )
                continue
            values = values - base_mean
            if label in bands:
                by, blo, bhi = bands[label]
                bands[label] = (by, blo - base_mean, bhi - base_mean)
        per_source[label] = (years_v, values)

    path = render_series(
        per_source,
        var,
        index,
        out,
        models=[m for m in by_label if m in per_source],
        bands=bands or None,
        reference=reference,
        driver=driver,
        years=years,
        units=units,
        is_anomaly=baseline is not None,
        baseline=baseline,
        season=season,
        **kwargs,
    )
    if path is None:
        print(f"[series] {var}/{index}: no source had a series — skipping.")
    return path
