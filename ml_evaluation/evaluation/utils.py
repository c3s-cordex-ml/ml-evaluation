from __future__ import annotations


import numpy as np
import pandas as pd
import xarray as xr


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


def xr_assign_dayofyear_noleap(x: xr.Dataset) -> xr.Dataset:
    """
    Assign new coordinate "doy_noleap" for day of the year.

    Assumes all years have length 365 for simplicity. Removes 29th Feb in leap
    years.

    Parameters
    ----------
    x : xr.Dataset
        Dataset to assign new coordinate, must contain a 'time' dimension.

    Returns
    -------
    xr.Dataset
        Dataset with new coordinate "doy_noleap" added.
    """
    # Make sure 29th Feb in leap years are removed.
    x_noleap = x.sel(time=~((x.time.dt.month == 2) & (x.time.dt.day == 29)))

    has_doy_noleap = "doy_noleap" in x_noleap.coords
    if has_doy_noleap:
        print("doy_noleap coordinate already exists so don't add")
        return x_noleap

    x_dates = pd.to_datetime(x_noleap["time"].dt.date)
    x_doy_orig = np.array(x_dates.dayofyear)
    x_doy_new = np.array(x_dates.dayofyear)
    # Adjust pandas day of year in leap years to revert to 365 day year
    adj_index = x_dates.is_leap_year * (x_doy_orig > 60)
    x_doy_new[adj_index] = x_doy_orig[adj_index] - 1
    x_noleap = x_noleap.assign_coords(doy_noleap=("time", x_doy_new))

    return x_noleap


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

    Returns
    -------
    xr.DataArray
        Spearman correlation coefficient, with `dim` reduced out.
    """
    if dim is None:
        dim = list(set(da_a.dims) & set(da_b.dims))

    dims = [dim] if isinstance(dim, str) else list(dim)

    if len(dims) == 1:
        rank_a = da_a.rank(dim=dims[0])
        rank_b = da_b.rank(dim=dims[0])
    else:
        # DataArray.rank only supports a single dim, so stack the
        # requested dims into one before ranking, then unstack.
        stacked = "__spearman_stack__"
        rank_a = da_a.stack({stacked: dims}).rank(dim=stacked).unstack(stacked)
        rank_b = da_b.stack({stacked: dims}).rank(dim=stacked).unstack(stacked)

    return xr.corr(rank_a, rank_b, dim=dim)


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
    output: xr.DataArray = xr.apply_ufunc(
        max_spell_length,
        arr,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return output


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


def resample_max_spell(condition: xr.DataArray, freq: str = "1YE") -> xr.DataArray:
    """
    Aggregate maximum spell length over the resampled period.

    Returns the mean annual spell length.

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

    def _compute(arr: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(
            max_spell_length,
            arr,
            input_core_dims=[["time"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

    return condition.resample(time=freq).map(_compute).mean("time")
