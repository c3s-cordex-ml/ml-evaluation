from __future__ import annotations

import numpy as np
import xarray as xr

from ml_evaluation.transformations import spatial_dims

from ml_evaluation.evaluation.utils import (
    filter_by_monthseason,
    xr_assign_dayofyear_noleap,
    ann_max_spell,
    resample_max_spell,
)


def mean(x: xr.Dataset, var: str, season: str | None = None) -> xr.Dataset:
    """
    Compute the mean of a variable in a dataset over the time dimension.

    Filters the dataset by season or month if specified, before computing
    the mean.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Variable name to compute the mean for.
    season : str | None, optional
        Season or Month standard name to filter before computing the mean.

    Returns
    -------
    xr.Dataset
        Mean of the input data over time.
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]
    result = da.mean(dim="time", skipna=True)
    return result


def quantile(x: xr.Dataset, var: str, q: float) -> xr.Dataset:
    """
    Compute the specified time quantile of a variable in a dataset.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Variable name to compute the quantile for.
    q : float
        Quantile value in the interval [0, 1].

    Returns
    -------
    xr.Dataset
        Quantile of the input data.
    """
    da = x[[var]]
    result = da.quantile(q, dim="time", skipna=True)
    if "quantile" in result.dims:
        result = result.sel(quantile=q)
    return result


def season_quantile(
    x: xr.Dataset,
    var: str,
    q: float,
    season: str | None = None,
    wet_threshold: float | None = None,
) -> xr.Dataset:
    """
    Compute the specified time quantile of a variable in a dataset.

    Filters the dataset by season or month if specified, before computing
    the quantile.

    Also option for precip to first ignore values below wet_threshold.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Variable name to compute the quantile for.
    q : float
        Quantile value in the interval [0, 1].
    season : str | None, optional
        Season or Month standard name to filter before computing the quantile.
    wet_threshold : float | None
        Precipitation threshold to qualify as a wet day e.g. 1mm/day.

    Returns
    -------
    xr.Dataset
        Quantile of the input data.
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]

    if wet_threshold is not None:
        da = da.where(da >= wet_threshold)

    result = da.quantile(q, dim="time", skipna=True)
    if "quantile" in result.dims:
        result = result.sel(quantile=q)
    return result


def calday_quantile(x: xr.Dataset, var: str, q: float, padding: int = 2) -> xr.Dataset:
    da = x[[var]]
    n_members = padding * 2 + 1
    center = n_members // 2  # = 2
    shifts = [i - center for i in range(n_members)]  # [-2, -1, 0, 1, 2]
    members = [da.roll(time=s, roll_coords=False) for s in shifts]
    da_members = xr.concat(members, dim="member")
    da_members = da_members.assign_coords(member=range(n_members))

    # Group by calendar day-of-year, then take the quantile over
    # both the member (day-window) and time (all years) dimensions
    result = da_members.groupby("time.dayofyear").quantile(
        q, dim=("member", "time"), skipna=True
    )

    if "quantile" in result.dims:
        result = result.sel(quantile=q)
    return result


def day_noleap_quantile(
    x: xr.Dataset, var: str, q: float, padding: int = 2, axis: int = 0
) -> xr.Dataset:
    """
    Compute the specified daily time quantile of a variable.

    Uses a centred 5 day window (assumes 365 day year) and adds coordinate
    'doy_noleap' to output while removing coordinate 'time'.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable, with dimensions {'time', 'lat', 'lon'}.
    var : str
        Variable name to compute the quantile for.
    q : float
        Quantile value in the interval [0, 1].
    padding : int, default 2
        Padding to make window (e.g. padding=2 leads to a window of size 5 days).
    axis : int, default 0
        Axis along which to compute quantile.

    Returns
    -------
    xr.Dataset
        Quantile of the input data with dimensions {'doy_noleap', 'lat', 'lon'}.
    """
    da = x[[var]]

    da = xr_assign_dayofyear_noleap(da)

    doy_vals = da["doy_noleap"].values  # np array of 'day in year' values
    data_vals = da[var].values  # np array of data values

    n_doys = 365

    # list of arrays for each day in year showing position of elements
    # of input data with that day
    day_indices: list[np.ndarray] = [
        np.where(doy_vals == d)[0] for d in range(1, n_doys + 1)
    ]

    # output array
    spatial_shape = data_vals.shape[1:] if data_vals.ndim > 1 else ()
    out = np.empty((n_doys, *spatial_shape), dtype=np.float64)

    if padding > 0:
        for d in range(n_doys):
            # Wrap-around indices for the window
            window_doys = [
                (d + offset) % n_doys for offset in range(-padding, padding + 1)
            ]
            # Concatenate time indices for every DOY in the window
            idx = np.concatenate([day_indices[wd] for wd in window_doys])
            out[d] = np.nanquantile(data_vals[idx], q, axis=axis)
    else:
        # No window: just standard per-DOY quantile
        for d in range(n_doys):
            out[d] = np.nanquantile(data_vals[day_indices[d]], q, axis=axis)

    doy_coord = np.arange(1, n_doys + 1)
    spatial_coords = {
        k: da.coords[k] for k in da.coords if k != "doy_noleap" and k != "time"
    }

    result_da = xr.DataArray(
        out,
        dims=["doy_noleap", *[d for d in da[var].dims[1:]]],
        coords={"doy_noleap": doy_coord, **spatial_coords},
    )

    return xr.Dataset({var: result_da})


def ann_extrema(
    x: xr.Dataset,
    var: str,
    maxmin: str = "max",
    season: str | None = None,
    drop_nonfull: bool = True,
    cmean: bool = False,
) -> xr.Dataset:
    """
    Compute the annual maximum or minimum of target variable per year.

    Has options to output annual values or climate mean.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Name of the daily variable.
    maxmin : str, default 'max'
        Type of extreme, options are 'max' or 'min'.
    season : str | None, optional
        Season or Month standard name to filter.
    drop_nonfull : bool, default True
        Treatment of DJF season: True to only include seasons with full 3 months
        of data.
    cmean : bool, default False
        True to compute climate mean over time, False to leave as annual output.

    Returns
    -------
    xr.Dataset
        Annual extrema values of daily variable (or mean thereof).
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]

    if maxmin not in ["max", "min"]:
        maxmin = "max"
        print(
            f"WARNING: maxmin='{maxmin}' not in list ['max','min'] so default to 'max'"
        )

    if season == "DJF":
        # Compute on consecutive months across year boundary ending at the end of Feb
        if maxmin == "max":
            annual_extrema = da.resample(time="1YE-FEB").max()
        else:
            annual_extrema = da.resample(time="1YE-FEB").min()
        # If data covers n years going from Jan Year1 to Dec YearN, there will be
        # n+1 max values for seasons [JF, DJF, ..., DJF, D] so drop shortest season (D)
        years = np.unique(da["time.year"])
        counts = da.resample(time="1YE-FEB").count()
        if drop_nonfull:
            # Only include full DJF seasons (90 days)
            annual_extrema = annual_extrema.where(counts == 90, drop=True)
        else:
            # Include full DJF seasons (90 days) and JF season (59 days)
            # but not D season (31 days)
            if len(counts[var]) > len(years):
                annual_extrema = annual_extrema.where(counts >= 59, drop=True)
    else:
        if maxmin == "max":
            annual_extrema = da.resample(time="1YE").max()
        else:
            annual_extrema = da.resample(time="1YE").min()

    if cmean:
        output: xr.Dataset = annual_extrema.mean(dim="time")
    else:
        output = annual_extrema

    return output


def ann_freq_wrt_array(
    x: xr.Dataset,
    var: str,
    threshold: xr.Dataset | int | float,
    relation: str = "gt",
    season: str | None = None,
    drop_nonfull: bool = True,
    cmean: bool = False,
) -> xr.Dataset:
    """
    Count number of event days, defined wrt threshold, per year.

    Has options to output annual values or annual mean.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Name of the daily variable.
    threshold : xr.Dataset or int or float
        Array threshold or scalar to define event.
    relation : str, default 'gt'
        Relationship of event to threshold: 'gt', 'ge', 'lt', 'le', 'eq', 'ne'.
    season : str | None, optional
        Season name to filter.
    drop_nonfull : bool, default True
        True to ignore DJF seasons < 90 days.
    cmean : bool, default False
        True to compute climate mean over time, False to leave as annual output.

    Returns
    -------
    xr.Dataset
        Annual count of event days (or mean thereof).

    Raises
    ------
    ValueError
        If shape of threshold does not match input or if threshold dimensions are invalid.
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]

    threshold_new: xr.Dataset | int | float
    if isinstance(threshold, (int, float)):
        threshold_new = threshold
    elif isinstance(threshold, xr.Dataset):
        # Should include check that shape matches
        sdims = spatial_dims(da, var)
        if any(da.sizes[d] != threshold.sizes.get(d) for d in sdims):
            raise ValueError(
                f"Spatial shape mismatch: input "
                f"({', '.join(str(da.sizes[d]) for d in sdims)}) "
                f"!= threshold "
                f"({', '.join(str(threshold.sizes.get(d)) for d in sdims)}) "
                f"on dims {sdims}"
            )

        dims_thresh = set(threshold.dims)
        other_dims_thresh = dims_thresh - set(sdims)

        if not other_dims_thresh:
            threshold_new = threshold

        else:
            # Broadcast the threshold back onto actual time axis via dayofyear
            doy = da["time"].dt.dayofyear
            threshold_new = threshold.sel(dayofyear=doy)

    else:
        threshold_new = threshold

    if relation not in ["gt", "ge", "lt", "le", "eq", "ne"]:
        relation = "gt"
        print(f"WARNING: relation='{relation}' not in list: ")
        print("['gt', 'ge', 'lt', 'le', 'eq', 'ne'] so default to 'gt'")

    # Event Condition
    if relation == "gt":
        econdition = da[var] > threshold_new
    elif relation == "ge":
        econdition = da[var] >= threshold_new
    elif relation == "lt":
        econdition = da[var] < threshold_new
    elif relation == "le":
        econdition = da[var] <= threshold_new
    elif relation == "eq":
        econdition = da[var] == threshold_new
    elif relation == "ne":
        econdition = da[var] != threshold_new
    else:
        econdition = da[var] > threshold_new

    if season == "DJF":
        # Compute on consecutive months across year boundary ending at the end of Feb
        ann_freq = econdition.resample(time="1YE-FEB").sum("time")
        # If data covers n years going from Jan Year1 to Dec YearN, there will be
        # n+1 max values for seasons [JF, DJF, ..., DJF, D] so drop shortest season (D)
        years = np.unique(da["time.year"])
        counts = da.resample(time="1YE-FEB").count()
        if drop_nonfull:
            # Only include full DJF seasons (90 days)
            ann_freq = ann_freq.where(counts == 90, drop=True)
        else:
            # Include full DJF seasons (90 days) and JF season (59 days)
            # but not D season (31 days)
            if len(counts[var]) > len(years):
                ann_freq = ann_freq.where(counts >= 59, drop=True)
    else:
        ann_freq = econdition.resample(time="1YE").sum("time")

    if isinstance(ann_freq, xr.Dataset):
        ann_freq_ds = ann_freq
    else:
        ann_freq_ds = xr.Dataset({var: ann_freq})

    if cmean:
        output: xr.Dataset = ann_freq_ds.mean("time")
    else:
        output = ann_freq_ds

    return output


def ann_maxspell_wrt_array(
    x: xr.Dataset,
    var: str,
    threshold: xr.Dataset | int | float,
    relation: str = "gt",
    season: str | None = None,
    drop_nonfull: bool = True,
    cmean: bool = False,
) -> xr.Dataset:
    """
    Compute max spell length of event days, defined wrt threshold, per year.

    Has options to output annual values or annual mean.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Name of the daily variable.
    threshold : xr.Dataset or int or float
        Array threshold or scalar to define event.
    relation : str, default 'gt'
        Relationship of event to threshold: 'gt', 'ge', 'lt', 'le', 'eq', 'ne'.
    season : str | None, optional
        Season name to filter.
    drop_nonfull : bool, default True
        Treatment of DJF season: True to only include seasons with full 3 months
        of data.
    cmean : bool, default False
        True to compute climate mean over time, False to leave as annual output.

    Returns
    -------
    xr.Dataset
        Annual max spell length of event days (or mean thereof).

    Raises
    ------
    ValueError
        If shape of threshold does not match input or if threshold dimensions are invalid.
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]

    threshold_new: xr.Dataset | int | float
    if isinstance(threshold, (int, float)):
        threshold_new = threshold
    elif isinstance(threshold, xr.Dataset):
        # Should include check that shape matches
        sdims = spatial_dims(da, var)
        if any(da.sizes[d] != threshold.sizes.get(d) for d in sdims):
            raise ValueError(
                f"Spatial shape mismatch: input "
                f"({', '.join(str(da.sizes[d]) for d in sdims)}) "
                f"!= threshold "
                f"({', '.join(str(threshold.sizes.get(d)) for d in sdims)}) "
                f"on dims {sdims}"
            )

        dims_thresh = set(threshold.dims)
        other_dims_thresh = dims_thresh - set(sdims)

        if not other_dims_thresh:
            threshold_new = threshold
        else:
            # Broadcast the threshold back onto actual time axis via dayofyear
            doy = da["time"].dt.dayofyear
            threshold_new = threshold.sel(dayofyear=doy)
    else:
        threshold_new = threshold

    if relation not in ["gt", "ge", "lt", "le", "eq", "ne"]:
        relation = "gt"
        print(f"WARNING: relation='{relation}' not in list: ")
        print("['gt', 'ge', 'lt', 'le', 'eq', 'ne'] so default to 'gt'")

    # Event Condition
    if relation == "gt":
        econdition = da[var] > threshold_new
    elif relation == "ge":
        econdition = da[var] >= threshold_new
    elif relation == "lt":
        econdition = da[var] < threshold_new
    elif relation == "le":
        econdition = da[var] <= threshold_new
    elif relation == "eq":
        econdition = da[var] == threshold_new
    elif relation == "ne":
        econdition = da[var] != threshold_new
    else:
        econdition = da[var] > threshold_new

    if season == "DJF":
        # Compute on consecutive months across year boundary ending at the end of Feb
        ann_spell = ann_max_spell(econdition, freq="1YE-FEB")
        years = np.unique(da["time.year"])
        counts = da.resample(time="1YE-FEB").count()
        # If data covers n years going from Jan Year1 to Dec YearN, there will be
        # n+1 max values for seasons [JF, DJF, ..., DJF, D] so either
        # - drop all non-full seasons i.e. JF and D
        if drop_nonfull:
            # Only include full DJF seasons (90 days)
            ann_spell = ann_spell.where(counts == 90, drop=True)
        # - drop shortest season (D)
        else:
            # Include full DJF seasons (90 days) and JF season (59 days)
            # but not D season (31 days)
            if len(counts[var]) > len(years):
                ann_spell = ann_spell.where(counts >= 59, drop=True)
    else:
        ann_spell = ann_max_spell(econdition, freq="1YE")

    if isinstance(ann_spell, xr.Dataset):
        ann_spell_ds = ann_spell
    else:
        ann_spell_ds = xr.Dataset({var: ann_spell})

    if cmean:
        output: xr.Dataset = ann_spell_ds.mean("time")
    else:
        output = ann_spell_ds

    return output


def ann_sdii(
    x: xr.Dataset,
    var: str,
    season: str | None = None,
    wet_threshold: float = 1.0,
    drop_nonfull: bool = True,
    cmean: bool = False,
) -> xr.Dataset:
    """
    Compute the simple precipitation intensity index.

    Has options to output annual values or annual mean.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the precipitation variable.
    var : str
        Variable name of daily precipitation.
    season : str | None, optional
        Season name to filter before computing the index.
    wet_threshold : float, default 1.0
        Minimum precipitation to consider a wet day.
    drop_nonfull : bool, default True
        Treatment of DJF season: True to only include seasons with full 3 months
        of data.
    cmean : bool, default False
        True to compute climate mean over time, False to leave as annual output.

    Returns
    -------
    xr.Dataset
        Simple precipitation intensity index.
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]
    wet = da.where(da >= wet_threshold)

    if season == "DJF":
        # Compute on consecutive months across year boundary ending at the end of Feb
        output = wet.resample(time="1YE-FEB").mean(dim="time")
        # If data covers n years going from Jan Year1 to Dec YearN, there will be
        # n+1 max values for seasons [JF, DJF, ..., DJF, D] so drop shortest season (D)
        years = np.unique(da["time.year"])
        counts = da.resample(time="1YE-FEB").count()
        if drop_nonfull:
            # Only include full DJF seasons (90 days)
            output = output.where(counts == 90, drop=True)
        else:
            # Include full DJF seasons (90 days) and JF season (59 days)
            # but not D season (31 days)
            if len(counts[var]) > len(years):
                output = output.where(counts >= 59, drop=True)
    else:
        output = wet.resample(time="1YE").mean(dim="time")

    if cmean:
        output = output.mean("time")

    return output


def interannual_var(
    x: xr.Dataset,
    var: str,
    season: str | None = None,
    drop_nonfull: bool = True,
) -> xr.Dataset:
    """
    Compute the interannual variability of the target variable along time.

    Filters dataset and handles seasons including DJF.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season name to filter before computing the index.
    drop_nonfull : bool, default True
        Treatment of DJF season: True to only include seasons with full 3 months
        of data.

    Returns
    -------
    xr.Dataset
        Interannual variability computed along the time dimension.
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]

    if season == "DJF":
        # Compute on consecutive months across year boundary ending at the end of Feb
        output = da.resample(time="1YE-FEB").mean(dim="time")
        # If data covers n years going from Jan Year1 to Dec YearN, there will be
        # n+1 max values for seasons [JF, DJF, ..., DJF, D] so drop shortest season (D)
        years = np.unique(da["time.year"])
        counts = da.resample(time="1YE-FEB").count()
        if drop_nonfull:
            # Only include full DJF seasons (90 days)
            output = output.where(counts == 90, drop=True)
        else:
            # Include full DJF seasons (90 days) and JF season (59 days)
            # but not D season (31 days)
            if len(counts[var]) > len(years):
                output = output.where(counts >= 59, drop=True)
        output_std: xr.Dataset = output.std("time")
    else:
        output_std = da.resample(time="1YE").mean(dim="time").std("time")

    return output_std


def lag1_corr(x: xr.Dataset, var: str, season: str | None = None) -> xr.Dataset:
    """
    Compute the lag-1 autocorrelation of the target variable along time.

    Filters dataset and handles seasonal/monthly first-day masking to avoid
    pairing values with more than a day lag.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season name to filter before computing the index.

    Returns
    -------
    xr.Dataset
        Lag-1 autocorrelation computed along the time dimension.
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[var]
    shifted = da.shift(time=1)

    # For annual data, .shift automatically masks the first day in data (=nan)
    # For seasonal or monthly data, also mask the first day of each season/month
    # to avoid pairing values with more than a day lag
    if season is not None:
        month1 = da.time.dt.month[0]
        if season == "DJF":
            month1 = 12
        da = da.sel(time=~((da.time.dt.month == month1) & (da.time.dt.day == 1)))
        shifted = shifted.sel(
            time=~((shifted.time.dt.month == month1) & (shifted.time.dt.day == 1))
        )

    lag1_corr_da = xr.corr(da, shifted, dim="time")
    return xr.Dataset({var: lag1_corr_da})


def su(
    x: xr.Dataset, var: str, season: str | None = None, threshold: float = 300.0
) -> xr.Dataset:
    """
    Count summer days defined by daily maximum temperature above a threshold.

    First the summer days are counted for each year, then the mean is computed.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Name of the daily maximum temperature variable.
    season : str | None, optional
        Season name to filter before computing the index.
    threshold : float, default 300.0
        Temperature threshold in Kelvin (or Celsius depending on units).

    Returns
    -------
    xr.Dataset
        Annual count of summer days.
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]
    return (da > threshold).groupby("time.year").sum("time").mean("year")


def txx(x: xr.Dataset, var: str, season: str | None = None) -> xr.Dataset:
    """
    Compute the mean annual maximum of daily maximum temperature.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Name of the daily maximum temperature variable.
    season : str | None, optional
        Season name to filter before computing the index.

    Returns
    -------
    xr.Dataset
        Mean annual maximum values of daily maximum temperature.
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]
    annual_max = da.resample(time="1YE").max()
    return annual_max.mean(dim="time")


def txn(x: xr.Dataset, var: str, season: str | None = None) -> xr.Dataset:
    """
    Compute the mean annual minimum of daily maximum temperature.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Name of the daily maximum temperature variable.
    season : str | None, optional
        Season name to filter before computing the index.

    Returns
    -------
    xr.Dataset
        Mean annual minimum values of daily maximum temperature.
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]
    annual_min = da.resample(time="1YE").min()
    return annual_min.mean(dim="time")


def rx1day(x: xr.Dataset, var: str, season: str | None = None) -> xr.Dataset:
    """
    Compute the mean annual maximum 1-day precipitation.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the precipitation variable.
    var : str
        Variable name of daily precipitation.
    season : str | None, optional
        Season name to filter before computing the index.

    Returns
    -------
    xr.Dataset
        Mean annual maximum daily precipitation.
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]
    annual_max = da.resample(time="1YE").max()
    return annual_max.mean(dim="time")


def sdii(
    x: xr.Dataset,
    var: str,
    season: str | None = None,
    wet_threshold: float = 1.0,
) -> xr.Dataset:
    """
    Compute the simple precipitation intensity index.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the precipitation variable.
    var : str
        Variable name of daily precipitation.
    season : str | None, optional
        Season name to filter before computing the index.
    wet_threshold : float, default 1.0
        Minimum precipitation to consider a wet day.

    Returns
    -------
    xr.Dataset
        Simple precipitation intensity index.
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]
    wet = da.where(da >= wet_threshold)
    return wet.mean(dim="time")


def cdd(
    x: xr.Dataset,
    var: str,
    season: str | None = None,
    dry_threshold: float = 1.0,
) -> xr.Dataset:
    """
    Compute the maximum length of dry spells with precipitation below a threshold.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the precipitation variable.
    var : str
        Variable name of daily precipitation.
    season : str | None, optional
        Season name to filter before computing the index.
    dry_threshold : float, default 1.0
        Precipitation threshold to qualify as a dry day.

    Returns
    -------
    xr.Dataset
        Annual maximum dry spell length.
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]
    condition = da < dry_threshold
    # extract DataArray from Dataset for resample_max_spell
    cond_da = condition[var]
    resampled = resample_max_spell(cond_da)
    return xr.Dataset({var: resampled})


def cwd(
    x: xr.Dataset,
    var: str,
    season: str | None = None,
    wet_threshold: float = 1.0,
) -> xr.Dataset:
    """
    Compute the maximum length of wet spells with precipitation above a threshold.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the precipitation variable.
    var : str
        Variable name of daily precipitation.
    season : str | None, optional
        Season name to filter before computing the index.
    wet_threshold : float, default 1.0
        Precipitation threshold to qualify as a wet day.

    Returns
    -------
    xr.Dataset
        Annual maximum wet spell length.
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]
    condition = da >= wet_threshold
    # extract DataArray from Dataset for resample_max_spell
    cond_da = condition[var]
    resampled = resample_max_spell(cond_da)
    return xr.Dataset({var: resampled})
