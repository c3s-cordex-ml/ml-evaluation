# SPDX-FileCopyrightText: 2026 c3s-cordex-ml
# SPDX-License-Identifier: MIT

"""
Climate-change-signal computation for ML RCM emulators.

A *signal* is the change of a climatological index between two periods
(future minus historical) of a *single* source, computed per grid point. The
command is **source-agnostic**: it describes one source's own signal (the
RCM-truth or one model) and never needs the ground truth. Signal *fidelity* --
how a model's signal compares to the RCM-truth's -- is a downstream operation
over two such bundles (e.g. in the signal scorecard), not computed here.

The per-index climatology used for the delta is the SAME catalog ``fast`` uses
(:func:`build_index_specs`), so signal and evaluation stay definitionally
consistent. The index set shown per variable (:func:`signal_index_keys`) is:

    non-precip : mean, P01, P99      (absolute delta is the meaningful signal)
    precip (pr): R01, SDII, RX1day   (relative/% delta is the meaningful signal)

Four bundles are produced, mirroring ``fast``'s field/series/float layout plus a
monthly annual-cycle store, each stacked along a ``diagnostic`` dimension and
holding, per index, the absolute ``delta`` and the relative ``reldelta``:

    field  (diagnostic, lat, lon) : delta / reldelta maps per index
    series (diagnostic, year)     : raw annual spatial-mean of each index,
                                     plus ``<index>_series_{min,max,std}`` across
                                     the ensemble when the source has >1 member
                                     and no single member was selected,
                                     hist+fut on one year axis
    month  (diagnostic, month)    : the seasonally-resolved (12-month) signal --
                                     delta / reldelta per index, each a
                                     spatial-mean monthly cycle (all 12 months)
    float  (diagnostic,)          : signed domain-mean delta / reldelta per index

Notes
-----
* Relative fields divide by the historical climatology and are masked where
  ``|hist| < EPS`` (dry-pixel guard for R01/SDII/RX1day).
* The annual ``series`` uses simple calendar-year grouping. For the ``cmean``
  indices (mean, R01, SDII, RX1day) the window means reproduce the stored delta;
  for P01/P99 they do not, because the period quantile pools all days whereas the
  series takes a per-year quantile then averages. The series is a variability /
  trend diagnostic, not an exact decomposition of the delta for quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import xarray as xr

from ml_evaluation.evaluation import indices
from ml_evaluation.evaluation.utils import filter_by_monthseason
from ml_evaluation.transformations import (
    SPATIAL_DIM_CANDIDATES,
    spatial_dims_present,
)

# Relative-delta guard: mask where the historical baseline is below this (in
# canonical units: mm/day for pr, K for temperature).
EPS = 1e-6

MONTHS = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
]

_FLUX_UNITS = {"kg m-2 s-1", "kg/m2/s", "kg m^-2 s^-1"}
_DEGC_UNITS = {"degc", "c", "celsius", "degrees_c", "degree_c"}


@dataclass(frozen=True)
class IndexSpec:
    """Specification of a single climatological index.

    Parameters
    ----------
    key : str
        Short index key (e.g. ``"mean"``, ``"rx1day"``).
    label : str
        Human-readable label for plots/scorecards.
    index_fn : Callable
        Function computing the *period* climatology as a ``(lat, lon)`` field.
    kwargs : dict
        Keyword arguments passed to ``index_fn`` (excluding ``season``).
    relative : bool
        True when the relative (%) signal is the physically-meaningful one
        (precip intensities/frequencies); False for temperature.
    """

    key: str
    label: str
    index_fn: Callable[..., xr.Dataset]
    kwargs: dict[str, Any]
    relative: bool


def build_index_specs(scale: float = 1.0, shift: float = 0.0) -> dict[str, IndexSpec]:
    """Period-climatology index catalog shared by ``fast`` and ``signal``.

    Thresholds are built from ``scale``/``shift`` so a hardcoded 1 mm/day wet
    threshold is expressed in the data's units (see
    :func:`ml_evaluation.transformations.convert_threshold_units`).

    Parameters
    ----------
    scale : float, default 1.0
        Multiplicative unit conversion for thresholds.
    shift : float, default 0.0
        Additive unit conversion for thresholds.

    Returns
    -------
    dict[str, IndexSpec]
        Mapping of index key to its specification.
    """
    wet = 1.0 * scale + shift
    return {
        "mean": IndexSpec("mean", "Mean", indices.mean, {}, False),
        "p01": IndexSpec("p01", "P01", indices.season_quantile, {"q": 0.01}, False),
        "p99": IndexSpec("p99", "P99", indices.season_quantile, {"q": 0.99}, False),
        "rx1day": IndexSpec(
            "rx1day",
            "RX1day",
            indices.ann_extrema,
            {"maxmin": "max", "cmean": True},
            True,
        ),
        "r01": IndexSpec(
            "r01",
            "R01",
            indices.ann_freq_wrt_array,
            {"threshold": wet, "relation": "ge", "cmean": True},
            True,
        ),
        "sdii": IndexSpec(
            "sdii",
            "SDII",
            indices.ann_sdii,
            {"wet_threshold": wet, "cmean": True},
            True,
        ),
    }


_SIGNAL_INDICES = {"pr": ["r01", "sdii", "rx1day"]}
_SIGNAL_INDICES_DEFAULT = ["mean", "p01", "p99"]


def signal_index_keys(var: str) -> list[str]:
    """Index keys whose signal is reported for ``var``.

    Parameters
    ----------
    var : str
        Variable name.

    Returns
    -------
    list[str]
        ``["r01", "sdii", "rx1day"]`` for precip, else ``["mean", "p01", "p99"]``.
    """
    return _SIGNAL_INDICES.get(var, _SIGNAL_INDICES_DEFAULT)


def canonicalize_units(ds: xr.Dataset, var: str) -> xr.Dataset:
    """Convert ``var`` to canonical units so two sources are comparable.

    Precipitation flux (``kg m-2 s-1``) -> mm/day; temperature in degC -> K.
    Data already in mm/day or K are returned unchanged. A magnitude fallback
    treats an unlabelled, very small-valued ``pr`` as a flux.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing ``var``.
    var : str
        Variable name.

    Returns
    -------
    xr.Dataset
        Dataset with ``var`` in canonical units and its ``units`` attribute set.
    """
    units = str(ds[var].attrs.get("units", "")).strip().lower()
    if units in _FLUX_UNITS or (
        var == "pr" and units in ("", "none") and float(ds[var].mean()) < 0.01
    ):
        ds = ds.copy()
        ds[var] = ds[var] * 86400.0
        ds[var].attrs["units"] = "mm/day"
    elif units in _DEGC_UNITS:
        ds = ds.copy()
        ds[var] = ds[var] + 273.15
        ds[var].attrs["units"] = "K"
    return ds


def snap_grid(ref: xr.Dataset, other: xr.Dataset) -> xr.Dataset:
    """Snap ``other``'s lat/lon coords to ``ref``'s when they are close.

    Lets field arithmetic (delta_pred - delta_ref) proceed when the two sources
    share a grid up to floating-point noise.

    Parameters
    ----------
    ref : xr.Dataset
        Reference dataset providing the target coordinates.
    other : xr.Dataset
        Dataset whose coordinates are snapped to ``ref``.

    Returns
    -------
    xr.Dataset
        ``other`` with lat/lon aligned to ``ref`` where within tolerance.
    """
    for c in sorted({n for pair in SPATIAL_DIM_CANDIDATES for n in pair}):
        if c in ref.coords and c in other.coords:
            rv, ov = ref[c].values, other[c].values
            if (
                rv.shape == ov.shape
                and not np.array_equal(rv, ov)
                and np.allclose(rv, ov)
            ):
                other = other.assign_coords({c: ref[c]})
    return other


def _drop_extra(da: xr.DataArray, keep: tuple[str, ...]) -> xr.DataArray:
    """Mean-reduce any dim not in ``keep`` (e.g. an ensemble ``member`` axis)."""
    extra = [d for d in da.dims if d not in keep]
    return da.mean(extra) if extra else da


def _spatial_mean(da: xr.DataArray) -> xr.DataArray:
    dims = list(spatial_dims_present(da))
    return da.mean(dims) if dims else da


def _rel(delta: xr.DataArray, base: xr.DataArray) -> xr.DataArray:
    """Relative delta ``delta / base``, masked where ``|base| < EPS``."""
    return xr.where(np.abs(base) < EPS, np.nan, delta / base)


def _period_field(
    ds: xr.Dataset, var: str, spec: IndexSpec, season: str | None
) -> xr.DataArray:
    """Period climatology of one index as a ``(lat, lon)`` field."""
    out = spec.index_fn(ds, var=var, season=season, **spec.kwargs)
    da = out[var] if isinstance(out, xr.Dataset) else out
    return _drop_extra(da, keep=spatial_dims_present(da))


def _annual_reduce(
    ds: xr.Dataset, var: str, spec: IndexSpec, season: str | None, wet: float
) -> xr.DataArray:
    """Per-year field of one index (simple calendar-year grouping)."""
    da = filter_by_monthseason(ds, season)[var]
    g = da.groupby("time.year")
    if spec.key == "mean":
        out = g.mean("time")
    elif spec.key in ("p01", "p99"):
        out = g.quantile(spec.kwargs["q"], dim="time")
        if "quantile" in out.coords:
            out = out.drop_vars("quantile")
    elif spec.key == "rx1day":
        out = g.max("time")
    elif spec.key == "r01":
        out = (da >= wet).groupby("time.year").sum("time")
    elif spec.key == "sdii":
        out = da.where(da >= wet).groupby("time.year").mean("time")
    else:  # pragma: no cover - catalog is closed
        raise ValueError(f"no annual reducer for index '{spec.key}'")
    return out


def member_dims(da: xr.DataArray) -> tuple[str, ...]:
    """Dimensions of an annual field that are neither the year nor spatial.

    In practice that is the ensemble axis, whatever it is called.
    """
    known = ("year", *spatial_dims_present(da))
    return tuple(d for d in da.dims if d not in known)


def _series_keep(da: xr.DataArray) -> tuple[str, ...]:
    """Dims ``_annual_series`` preserves: the year, the grid, and the members."""
    return ("year", *spatial_dims_present(da), *member_dims(da))


def _annual_series(
    dh: xr.Dataset,
    df: xr.Dataset,
    var: str,
    spec: IndexSpec,
    season: str | None,
    wet: float,
) -> xr.DataArray:
    """Raw annual spatial-mean series over hist+fut on one ``year`` axis.

    Any ensemble axis is *kept*: the caller reduces it, so that an ensemble-mean
    run can also store the spread across members (see :func:`compute_signal`).
    """
    ah = _annual_reduce(dh, var, spec, season, wet)
    af = _annual_reduce(df, var, spec, season, wet)
    sh = _spatial_mean(_drop_extra(ah, keep=_series_keep(ah)))
    sf = _spatial_mean(_drop_extra(af, keep=_series_keep(af)))
    return xr.concat([sh, sf], dim="year")


def _monthly_cycle(ds: xr.Dataset, var: str, spec: IndexSpec) -> xr.DataArray:
    """Raw monthly annual-cycle: index per calendar month, spatial-mean -> (12,)."""
    vals = []
    for m in MONTHS:
        sub = filter_by_monthseason(ds, m)
        out = spec.index_fn(sub, var=var, **spec.kwargs)
        da = out[var] if isinstance(out, xr.Dataset) else out
        vals.append(
            float(_spatial_mean(_drop_extra(da, keep=spatial_dims_present(da))))
        )
    return xr.DataArray(
        np.asarray(vals, dtype="float64"),
        dims=["month"],
        coords={"month": np.arange(1, 13)},
    )


def compute_signal(
    hist: xr.Dataset,
    fut: xr.Dataset,
    var: str,
    scale: float,
    shift: float,
    season: str | None = None,
) -> tuple[dict, dict, dict, dict]:
    """Compute the climate-change-signal bundles for one source and variable.

    ``hist`` and ``fut`` are the two periods of a *single* source (the RCM-truth
    or one model -- the command is source-agnostic, so it never needs the ground
    truth to describe a model's own signal). Both are assumed to be in canonical
    units (see :func:`canonicalize_units`) and on the same grid
    (see :func:`snap_grid`). Comparing a model's signal to the RCM-truth's (the
    signal *bias*) is a downstream operation over two of these bundles.

    Parameters
    ----------
    hist, fut : xr.Dataset
        Historical and future datasets for the same source.
    var : str
        Variable name.
    scale, shift : float
        Threshold unit conversions for the (canonical) data.
    season : str | None, optional
        Season/month to filter for the field/series/float metrics (the ``month``
        store always spans all 12 months).

    Returns
    -------
    tuple[dict, dict, dict, dict]
        ``(field, series, month, float)`` diagnostic dictionaries, each holding
        ``delta`` and ``reldelta`` per index.
    """
    specs = build_index_specs(scale, shift)
    wet = 1.0 * scale + shift
    field: dict[str, xr.DataArray] = {}
    series: dict[str, xr.DataArray] = {}
    month: dict[str, xr.DataArray] = {}
    flt: dict[str, xr.DataArray] = {}

    for key in signal_index_keys(var):
        spec = specs[key]

        hist_field = _period_field(hist, var, spec, season)
        fut_field = _period_field(fut, var, spec, season)
        delta = fut_field - hist_field
        reldelta = _rel(delta, hist_field)

        field[f"delta_{key}_field"] = delta
        field[f"reldelta_{key}_field"] = reldelta

        flt[f"delta_{key}_float"] = _spatial_mean(delta)
        flt[f"reldelta_{key}_float"] = _spatial_mean(reldelta)

        raw = _annual_series(hist, fut, var, spec, season, wet)
        ens = [d for d in raw.dims if d != "year"]
        series[f"{key}_series"] = raw.mean(ens) if ens else raw
        # An ensemble-mean run also records how far the members spread, so a
        # spread band needs no extra runs: one signal call per member would
        # otherwise mean one zarr store per member.
        if ens and int(np.prod([raw.sizes[d] for d in ens])) > 1:
            series[f"{key}_series_min"] = raw.min(ens)
            series[f"{key}_series_max"] = raw.max(ens)
            series[f"{key}_series_std"] = raw.std(ens)

        # Seasonally-resolved signal: difference the monthly climatologies so the
        # month store mirrors the field store (delta/reldelta) on a 12-month axis
        # rather than holding raw per-period cycles.
        mh = _monthly_cycle(hist, var, spec)
        mf = _monthly_cycle(fut, var, spec)
        m_delta = mf - mh
        month[f"delta_{key}_month"] = m_delta
        month[f"reldelta_{key}_month"] = _rel(m_delta, mh)

    return field, series, month, flt
