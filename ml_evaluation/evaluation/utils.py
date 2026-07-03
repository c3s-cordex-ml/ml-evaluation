from __future__ import annotations


import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import rankdata


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


def calculate_error_nparray(
    arr0: np.ndarray,
    arr1: np.ndarray,
    axis: int | tuple[int, ...] | None = 0,
    typeerror: str = "rmse",
) -> np.ndarray:
    """
    Calculate error between two N-dimensional numpy arrays.

    Computes various types of error metrics (RMSE, bias, absolute bias, or
    ratio) along the specified axis or axes.

    Parameters
    ----------
    arr0 : np.ndarray
        First N-dimensional array (Target).
    arr1 : np.ndarray
        Second N-dimensional array (Prediction, must have same shape as arr0).
    axis : int or tuple of int or None, default 0
        Axis/axes along which to compute error.
    typeerror : str, default 'rmse'
        Type of error to compute. Options are 'rmse', 'bias', 'abias', 'ratio'.

    Returns
    -------
    np.ndarray
        Computed error with the specified axis or axes removed.

    Raises
    ------
    ValueError
        If the shape of arr0 and arr1 do not match.
    """
    if arr0.shape != arr1.shape:
        raise ValueError(
            f"Arrays must have the same shape. Got {arr0.shape} and {arr1.shape}"
        )

    # Compute Error over axis/axes specified
    # - subtract first then take mean so consistent method for bias and rmse
    # - better level of accuracy than taking means then subtract (ie slight difference)
    if typeerror == "bias":
        output_error = (arr1 - arr0).mean(axis=axis)
    elif typeerror == "abias":
        output_error = np.abs(arr1 - arr0).mean(axis=axis)
    elif typeerror == "rmse":
        output_error = np.sqrt(((arr1 - arr0) ** 2).mean(axis=axis))
    elif typeerror == "ratio":
        output_error = arr1.mean(axis=axis) / arr0.mean(axis=axis)
    else:
        raise ValueError(f"Unrecognized typeerror: {typeerror}")

    return output_error


def calculate_pearsoncorr_nparray(
    arr0: np.ndarray,
    arr1: np.ndarray,
    axis: int | tuple[int, ...] | None = 0,
) -> np.ndarray:
    """
    Calculate Pearson correlation between two N-dimensional numpy arrays.

    Computes Pearson correlation coefficients along the specified axis or axes.

    Parameters
    ----------
    arr0 : np.ndarray
        First N-dimensional array (Target).
    arr1 : np.ndarray
        Second N-dimensional array (Prediction, must have same shape as arr0).
    axis : int or tuple of int or None, default 0
        Axis/axes over which to compute correlation.

    Returns
    -------
    np.ndarray
        Pearson correlation coefficients.

    Raises
    ------
    ValueError
        If the shape of arr0 and arr1 do not match.
    """
    if arr0.shape != arr1.shape:
        raise ValueError(
            f"Arrays must have the same shape. Got {arr0.shape} and {arr1.shape}"
        )

    # Center the data over axis/axes specified
    arr0_centered = arr0 - arr0.mean(axis=axis, keepdims=True)
    arr1_centered = arr1 - arr1.mean(axis=axis, keepdims=True)

    # Compute correlation over axis/axes specified
    numerator = (arr0_centered * arr1_centered).sum(axis=axis)
    denominator = np.sqrt(
        (arr0_centered**2).sum(axis=axis) * (arr1_centered**2).sum(axis=axis)
    )

    # Avoid division by zero (set as 0.0 instead of inf or nan)
    correlations = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )

    return correlations


def rank_array(
    arr0: np.ndarray,
    axis: int | tuple[int, ...] | None = 0,
) -> np.ndarray:
    """
    Calculate ranks using scipy.stats.rankdata along given axis.

    Extends ranking for a tuple of axes by first flattening data along these axes.

    Parameters
    ----------
    arr0 : np.ndarray
        N-dimensional array to compute ranks for.
    axis : int or tuple of int or None, default 0
        Axis/axes over which to compute ranks.

    Returns
    -------
    np.ndarray
        Ranks of data.

    Raises
    ------
    ValueError
        If axis is not None, int, or tuple of int.
    """
    if isinstance(axis, type(None)):
        ranked: np.ndarray = rankdata(
            arr0, method="average", axis=axis, nan_policy="propagate"
        )
        return ranked

    axis_tuple: tuple[int, ...]
    if isinstance(axis, int):
        axis_tuple = (axis,)
    elif isinstance(axis, tuple):
        axis_tuple = axis
    else:
        raise ValueError(f"axis must be None, int or tuple of int. Got {axis}")

    # reorder array so flatten axes in axis
    new_axis = list(range(0, len(axis_tuple)))
    arr0_moved = np.moveaxis(arr0, axis_tuple, new_axis)
    arr0_reshaped = arr0_moved.reshape(
        np.prod(arr0_moved.shape[0 : len(axis_tuple)]),
        *arr0_moved.shape[len(axis_tuple) :],
    )
    ranked_reshaped: np.ndarray = rankdata(
        arr0_reshaped, method="average", axis=0, nan_policy="propagate"
    )
    return ranked_reshaped


def calculate_spearmancorr_nparray(
    arr0: np.ndarray,
    arr1: np.ndarray,
    axis: int | tuple[int, ...] | None = 0,
) -> np.ndarray:
    """
    Calculate Spearman's Rank correlation between two N-dimensional numpy arrays.

    Computes Spearman's rank correlation along the specified axis or axes.

    Parameters
    ----------
    arr0 : np.ndarray
        First N-dimensional array (Target).
    arr1 : np.ndarray
        Second N-dimensional array (Prediction, must have same shape as arr0).
    axis : int or tuple of int or None, default 0
        Axis/axes over which to compute correlation.

    Returns
    -------
    np.ndarray
        Spearman correlation coefficients.

    Raises
    ------
    ValueError
        If the shape of arr0 and arr1 do not match.
    """
    if arr0.shape != arr1.shape:
        raise ValueError(
            f"Arrays must have the same shape. Got {arr0.shape} and {arr1.shape}"
        )

    # Convert data to ranks
    # - using sp.stats.rankdata with xarray rank defaults
    arr0_ranked = rank_array(arr0, axis=axis)
    arr1_ranked = rank_array(arr1, axis=axis)

    # Compute Pearson correlation on ranked data
    correlations = calculate_pearsoncorr_nparray(arr0_ranked, arr1_ranked, axis=0)

    return correlations


def calculate_corr_xrnp(
    da0: xr.DataArray,
    da1: xr.DataArray,
    dim: str | tuple[str, ...] | None = None,
    corrtype: str = "Pearson",
) -> xr.DataArray:
    """
    Compute the Correlation between da0 and da1 DataArrays.

    Calls numpy version of correlation which is quicker than xarray for moderate
    sized datasets.

    Parameters
    ----------
    da0 : xr.DataArray
        Ground truth DataArray.
    da1 : xr.DataArray
        Predicted DataArray.
    dim : str or tuple of str or None, optional
        Dimension(s) along which to compute the correlation.
    corrtype : str, default 'Pearson'
        Type of correlation, options are 'Pearson' or 'Spearman'.

    Returns
    -------
    xr.DataArray
        Correlation between da0 and da1.

    Raises
    ------
    ValueError
        If corrtype is not recognized or no matches found for dimensions.
    """
    corr_list = ["Pearson", "Spearman"]
    if corrtype not in corr_list:
        raise ValueError(f"corrtype not recognised: {corrtype}")

    dim_tuple: tuple[str, ...] | None
    if isinstance(dim, str):
        dim_tuple = (dim,)
    else:
        dim_tuple = dim

    axis_num: int | tuple[int, ...] | None
    if isinstance(dim_tuple, tuple):
        dim_list = list(da0.dims)
        matches = [dim1 for dim1 in dim_list if dim1 in dim_tuple]
        if not matches:
            raise ValueError(
                f"Warning: no matches found for {list(dim_tuple)} in {dim_list}"
            )
        axis_num = tuple(dim_list.index(dim1) for dim1 in matches)
    else:
        axis_num = None

    pcor_xarray = da0.mean(dim=dim) * 0.0

    pcor_result: np.ndarray
    if corrtype == "Pearson":
        pcor_result = calculate_pearsoncorr_nparray(
            da0.values, da1.values, axis=axis_num
        )
    elif corrtype == "Spearman":
        pcor_result = calculate_spearmancorr_nparray(
            da0.values, da1.values, axis=axis_num
        )
    else:
        raise ValueError(f"Unrecognized corrtype: {corrtype}")

    pcor_xarray = pcor_xarray + pcor_result

    return pcor_xarray


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
