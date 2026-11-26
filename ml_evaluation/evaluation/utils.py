# SPDX-FileCopyrightText: 2026 c3s-cordex-ml
# SPDX-License-Identifier: MIT

# NOTE: This module contains third-party code distributed under the MIT License.
# Each third-party function is marked with an SPDX header immediately above its
# definition. Full copyright notices and license texts are in
# THIRD_PARTY_NOTICES.md (the project's own license is in LICENSE).

from __future__ import annotations


import numpy as np
import xarray as xr


# SPDX-FileCopyrightText: 2026 CORDEX
# SPDX-License-Identifier: MIT
# Third-party code (modified); see THIRD_PARTY_NOTICES.md, section 1.
def filter_by_monthseason(x: xr.Dataset, season: str | None) -> xr.Dataset:
    """
    Filter the dataset by season or month label.

    Retrieves data corresponding to standard season names ('DJF', 'MAM',
    'JJA', 'SON') or standard month names ('JAN', 'FEB', ..., 'DEC'). If the
    given season is None or invalid, returns the original dataset.

    Parameters
    ----------
    x : xr.Dataset
        Dataset to filter.
    season : str | None
        Season or Month standard name ('DJF', 'JUN',...) or None to skip filtering.

    Returns
    -------
    xr.Dataset
        Filtered dataset.
    """
    MON_LIST = [
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
    SEAS_LIST = ["DJF", "MAM", "JJA", "SON"]

    if season not in MON_LIST and season not in SEAS_LIST:
        season = None
    if season is None:
        return x

    if season in SEAS_LIST:
        return x.where(x["time.season"] == season, drop=True)

    if season in MON_LIST:
        month_num = MON_LIST.index(season) + 1
        return x.where(x["time.month"] == month_num, drop=True)

    return x


# SPDX-FileCopyrightText: 2026 CORDEX
# SPDX-License-Identifier: MIT
# Third-party code (modified); see THIRD_PARTY_NOTICES.md, section 1.
def radial_average(array_2d: np.ndarray) -> np.ndarray:
    """
    Compute the radial average of a two-dimensional field.

    Averages the values of a 2D array over concentric rings starting from the center.

    Parameters
    ----------
    array_2d : np.ndarray
        Two-dimensional array to average.

    Returns
    -------
    np.ndarray
        Radially averaged profile.
    """
    y, x = np.indices(array_2d.shape)
    center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1]).astype(np.int32)
    tbin = np.bincount(r.ravel(), array_2d.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


# SPDX-FileCopyrightText: 2026 CORDEX
# SPDX-License-Identifier: MIT
# Third-party code (modified); see THIRD_PARTY_NOTICES.md, section 1.
def max_spell_length(condition: np.ndarray) -> float:
    """
    Compute the maximum consecutive spell length in a boolean sequence.

    Parameters
    ----------
    condition : np.ndarray
        Boolean sequence describing the spell condition.

    Returns
    -------
    float
        Maximum consecutive spell length.
    """
    max_run = 0
    run = 0
    for flag in condition:
        if flag:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return float(max_run)


# SPDX-FileCopyrightText: 2026 CORDEX
# SPDX-License-Identifier: MIT
# Third-party code (modified); see THIRD_PARTY_NOTICES.md, section 1.
def compute_max_spell_per_year(arr: xr.DataArray) -> xr.DataArray:
    """
    Compute max spell length per year.

    Parameters
    ----------
    arr : xr.DataArray
        Boolean array describing the spell condition over time.

    Returns
    -------
    xr.DataArray
        Resampled maximum spell length.
    """
    tdim = "time"
    output: xr.DataArray = xr.apply_ufunc(
        max_spell_length,
        arr.chunk({tdim: -1}),
        input_core_dims=[[tdim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return output


# SPDX-FileCopyrightText: 2026 CORDEX
# SPDX-License-Identifier: MIT
# Third-party code (modified); see THIRD_PARTY_NOTICES.md, section 1.
def ann_max_spell(condition: xr.DataArray, freq: str = "1YE") -> xr.DataArray:
    """
    Aggregate maximum spell length over the resampled period.

    Parameters
    ----------
    condition : xr.DataArray
        Boolean array describing the spell condition over time.
    freq : str, default '1YE'
        Resampling frequency following pandas offset aliases.

    Returns
    -------
    xr.DataArray
        Resampled maximum spell length.
    """
    output: xr.DataArray = condition.resample(time=freq).map(compute_max_spell_per_year)
    return output


def spell_days_length(condition: np.ndarray, min_length: int) -> float:
    """
    Count days belonging to spells of at least ``min_length`` consecutive days.

    Parameters
    ----------
    condition : np.ndarray
        Boolean sequence describing the spell condition.
    min_length : int
        Minimum spell length (days) for its days to be counted.

    Returns
    -------
    float
        Total number of days in spells of length >= ``min_length``.
    """
    total = 0
    run = 0
    for flag in condition:
        if flag:
            run += 1
        else:
            if run >= min_length:
                total += run
            run = 0
    if run >= min_length:
        total += run
    return float(total)


def compute_spell_days_per_year(arr: xr.DataArray, min_length: int) -> xr.DataArray:
    """
    Count days in spells of length >= ``min_length`` for one resample group.

    Parameters
    ----------
    arr : xr.DataArray
        Boolean array describing the spell condition over time.
    min_length : int
        Minimum spell length (days) for its days to be counted.

    Returns
    -------
    xr.DataArray
        Number of days in qualifying spells.
    """
    tdim = "time"
    output: xr.DataArray = xr.apply_ufunc(
        spell_days_length,
        arr.chunk({tdim: -1}),
        input_core_dims=[[tdim]],
        output_core_dims=[[]],
        kwargs={"min_length": min_length},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return output


def ann_spell_days(
    condition: xr.DataArray, min_length: int, freq: str = "1YE"
) -> xr.DataArray:
    """
    Aggregate number of days in spells of length >= ``min_length`` per period.

    Spells are split at the boundaries of the resampling period.

    Parameters
    ----------
    condition : xr.DataArray
        Boolean array describing the spell condition over time.
    min_length : int
        Minimum spell length (days) for its days to be counted.
    freq : str, default '1YE'
        Resampling frequency following pandas offset aliases.

    Returns
    -------
    xr.DataArray
        Resampled number of days in qualifying spells.
    """
    output: xr.DataArray = condition.resample(time=freq).map(
        compute_spell_days_per_year, min_length=min_length
    )
    return output


def _rank(da, dim):
    if da.chunks is not None:  # dask-backed
        other = {d: "auto" for d in da.dims if d != dim}
        da = da.chunk({dim: -1, **other})
    return da.rank(dim=dim)


def _rank_along(da: xr.DataArray, dim: str, chunk_size: str = "32MiB") -> xr.DataArray:
    """
    Rank ``da`` along ``dim``, keeping dask chunks small enough to rank.

    Ranking needs ``dim`` in a single chunk. For dask-backed input, the
    other dims are re-split ("auto", capped at ``chunk_size``) at the same
    time so each chunk stays small once ``dim`` is merged. Non-dask input
    is ranked eagerly with numpy and is not converted to dask.

    Ranks are cast back to the input's float precision (bottleneck always
    returns float64), halving memory for float32 data. float64 is kept
    if ``dim`` is longer than 2**24, beyond which float32 cannot hold
    every rank exactly.

    Parameters
    ----------
    da : xr.DataArray
        Array to rank.
    dim : str
        Dimension to rank along.
    chunk_size : str, default "32MiB"
        Target size of each input chunk after rechunking (dask only).

    Returns
    -------
    xr.DataArray
        Ranks of ``da`` along ``dim``.
    """
    if da.chunks is not None:  # dask-backed
        import dask

        other = {d: "auto" for d in da.dims if d != dim}
        with dask.config.set({"array.chunk-size": chunk_size}):
            da = da.chunk({dim: -1, **other})
    out_dtype = np.result_type(da.dtype, np.float32)
    if da.sizes[dim] > 2**24:
        out_dtype = np.dtype(np.float64)
    return da.rank(dim=dim).astype(out_dtype, copy=False)


# SPDX-FileCopyrightText: 2026 Klima consulting
# SPDX-License-Identifier: MIT
# Third-party code (modified); see THIRD_PARTY_NOTICES.md, section 2.
def xr_spearman_corr(da_a, da_b, dim=None):
    """
    Spearman rank correlation between two DataArrays, mirroring xr.corr's API.

    Computed as Pearson correlation (xr.corr) on rank-transformed data,
    which is the standard definition of Spearman's rho.

    Parameters
    ----------
    da_a, da_b : xr.DataArray
        Arrays to correlate. Must share the dimension(s) given in `dim`.
    dim : str, Sequence[str], or None
        Dimension(s) to reduce over. If None, uses all dims shared by
        da_a and da_b (same default behavior as xr.corr).

    Copyright (c) 2026 Klima consulting
    License: MIT (see THIRD_PARTY_NOTICES.md)

    Returns
    -------
    xr.DataArray
        Spearman correlation coefficient, with `dim` reduced out.
    """
    if dim is None:
        dim = list(set(da_a.dims) & set(da_b.dims))

    dims = [dim] if isinstance(dim, str) else list(dim)

    if len(dims) == 1:
        rank_a = _rank_along(da_a, dims[0])
        rank_b = _rank_along(da_b, dims[0])
    else:
        # DataArray.rank only supports a single dim, so stack the
        # requested dims into one before ranking, then unstack.
        stacked = "__spearman_stack__"
        rank_a = _rank_along(da_a.stack({stacked: dims}), stacked).unstack(stacked)
        rank_b = _rank_along(da_b.stack({stacked: dims}), stacked).unstack(stacked)

    return xr.corr(rank_a, rank_b, dim=dim)
