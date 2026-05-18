import numpy as np
import xarray as xr
import pandas as pd
import indices

TEST_SPEED=False # Option to test speed of certain functions
if TEST_SPEED: import datetime # only needed when testing speed of different versions of functions

# -------------------------------------------------------------------------------------
def _filter_by_monthseason(x: xr.Dataset, season: str | None) -> xr.Dataset:
    """
    Filter the dataset by season or month label.
    Added by R. Eade.
    Adapted from ml-benchmark indices.py _filter_by_season()
    - changed season filter so more generalised

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
    
    MON_LIST=["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    SEAS_LIST=["DJF","MAM","JJA","SON"]
    
    if not season in MON_LIST and not season in SEAS_LIST: 
        season=None
    if season is None:
        return x
        
    if season in SEAS_LIST:
        return x.where(x["time.season"] == season, drop=True)
        
    if season in MON_LIST:
        month_num = MON_LIST.index(season) + 1 
        return x.where(x["time.month"] == month_num, drop=True)

    return x

# -------------------------------------------------------------------------------------
def xr_assign_dayofyear_noleap(x: xr.Dataset):
    """
    Assign new coordinate "doy_noleap" for day of the year (Dataset needs time dim).
    Assumes all years have length 365 for simplicity.
    Need to make sure 29th Feb in leap years are removed.
    Added by R. Eade.

    Parameters
    ----------
    x : xr.Dataset
        Dataset to assign new coordinate.

    Returns
    -------
    xr.Dataset
        Dataset with new coordinate "doy_noleap" added.
    
    """

    # Make sure 29th Feb in leap years are removed.
    x = x.sel(time=~((x.time.dt.month == 2) & (x.time.dt.day == 29))) 

    has_doy_noleap = "doy_noleap" in x.coords
    if has_doy_noleap is True:
        print("doy_noleap coordinate already exists so don't add")
        return x

    x_dates = pd.to_datetime(x['time'].dt.date)
    x_doy_orig = np.array(x_dates.dayofyear)
    x_doy_new = np.array(x_dates.dayofyear)
    # Adjust pandas day of year in leap years to revert to 365 day year 
    adj_index = (x_dates.is_leap_year == True) * (x_doy_orig > 60)
    x_doy_new[adj_index] = x_doy_orig[adj_index]-1
    x = x.assign_coords(doy_noleap=("time", x_doy_new))

    return x

# -------------------------------------------------------------------------------------
def season_quantile(x: xr.Dataset, var: str, q: float, season: str | None = None) -> xr.Dataset:
    """
    Compute the specified time quantile of a variable in a dataset.
    Option to restrict to a specified season or month.
    Added by R. Eade.
    Adapted from ml-benchmark indices.py quantile()
    - changed season filter

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Variable name to compute the quantile for.
    q : float
        Quantile value in the interval [0, 1].
    season : str | None, optional
        Season or Month standard name to filter before computing the index:
        ('DJF', 'JUN',...) or None to skip filtering.

    Returns
    -------
    xr.Dataset
        Quantile of the input data.
    """
    
    x = _filter_by_monthseason(x, season)
    da = x[[var]]
    
    result = da.quantile(q, dim="time", skipna=True)
    if "quantile" in result.dims:
        result = result.sel(quantile=q)
    return result

# -------------------------------------------------------------------------------------
def day_noleap_quantile(x: xr.Dataset, var: str, q: float, padding: int = 2, axis: int = 0) -> xr.Dataset:
    """
    Compute the specified daily time quantile of a variable in a dataset
    using a centred 5 day window (assumes 365 day year). 
    Adds coordinate doy_noleap to output (if not already there) and removes coordinate time.
        
    Added by Rosie, May 2026
    * Check: only 1 year of data
    * Check: less than 1 year of data or missing days (careful with window wrapping)

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable, with dimensions {'time', 'lat', 'lon'}
    var : str
        Variable name to compute the quantile for.
    q : float
        Quantile value in the interval [0, 1].
    padding : int
        Padding to make window e.g. padding=2 leads to a centred window of size 5 days.
    axis : int, default=0
        Axis/axes along which to compute error - integer or tuple

    Returns
    -------
    xr.Dataset
        Quantile of the input data with dimensions {'doy_noleap', 'lat', 'lon'}
    """
    if TEST_SPEED: 
        print("day_noleap_quantile: start time")
        print(datetime.datetime.now().time())

    da = x[[var]]
    
    da = xr_assign_dayofyear_noleap(da)
    
    doy_vals = da["doy_noleap"].values # np array of 'day in year' values x0doy
    data_vals = da[var].values # np array of data values
    
    n_doys=365
    window=2*padding+1
    
    # list of arrays for each day in year showing position of elements of input data with that day    
    day_indices: list[np.ndarray] = [np.where(doy_vals == d)[0] for d in range(1, n_doys + 1)] 
    
    # output array
    spatial_shape = data_vals.shape[1:] if data_vals.ndim > 1 else ()
    out = np.empty((n_doys, *spatial_shape), dtype=np.float64)

    if padding > 0:
        for d in range(n_doys):
            # Wrap-around indices for the window
            window_doys = [(d + offset) % n_doys for offset in range(-padding, padding + 1)]
            # Concatenate time indices for every DOY in the window
            idx = np.concatenate([day_indices[wd] for wd in window_doys])
            out[d] = np.nanquantile(data_vals[idx], q, axis=0)
    else:
        # No window: just standard per-DOY quantile
        for d in range(n_doys):
            out[d] = np.nanquantile(data_vals[day_indices[d]], q, axis=0)

    doy_coord = np.arange(1, n_doys + 1)
    spatial_coords = {k: da.coords[k] for k in da.coords if k != "doy_noleap" and k != "time"}

    result_da = xr.DataArray(out,dims=["doy_noleap", *[d for d in da[var].dims[1:]]],coords={"doy_noleap": doy_coord, **spatial_coords})

    if TEST_SPEED: 
        print("day_noleap_quantile: end time")
        print(datetime.datetime.now().time())

    return xr.Dataset({var: result_da})
# -------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------
def ann_extrema(x: xr.Dataset, var: str, maxmin: str = 'max', season: str | None = None, drop_nonfull=True, cmean=False) -> xr.Dataset:
    """
    Compute the annual maximum or minimum of target variable per year (season or month).
    Option to output annual values or annual mean.
    Added by R. Eade.
    Adapted from ml-benchmark indices.py txx()
    - changed season filter, generalised (max or min options), 
    - special case for DJF,
    - outputs annual data not just climate mean

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Name of the daily variable.
    maxmin : str
        Type of extreme, options ('max','min').
    season : str | None, optional
        Season or Month standard name to filter before computing the index:
        ('DJF', 'JUN',...) or None to skip filtering.

    Returns
    -------
    xr.Dataset
        (Mean of ) Annual extrema values of daily variable.

    """
    x = _filter_by_monthseason(x, season)
    da = x[[var]]

    if maxmin not in ['max','min']:
        maxmin = 'max'
        print(f"WARNING: max='{max}' not in list ['max','min'] so default to 'max'")

    if season=='DJF':
        # Compute on consecutive months across year boundary ending at the end of Feb
        if maxmin=='max': annual_extrema = da.resample(time="1YE-FEB").max()
        if maxmin=='min': annual_extrema = da.resample(time="1YE-FEB").min()
        # If data covers n years going from Jan Year1 to Dec YearN, there will be
        # n+1 max values for seasons [JF, DJF, ..., DJF, D] so drop shortest season (D)
        years=np.unique(da["time.year"])
        counts = da.resample(time="1YE-FEB").count()
        if drop_nonfull:
            # Only include full DJF seasons (90 days) 
            annual_extrema = annual_extrema.where(counts == 90, drop=True)
        if not drop_nonfull:
            # Include full DJF seasons (90 days) and JF season (59 days) but not D season (31 days)  
            if len(counts[var]) > len(years): annual_extrema = annual_extrema.where(counts >= 59, drop=True)
    else:
        if maxmin=='max': annual_extrema = da.resample(time="1YE").max()
        if maxmin=='min': annual_extrema = da.resample(time="1YE").min()
        
    if cmean is True: output = annual_extrema.mean(dim="time")
    if cmean is False: output = annual_extrema
    
    return output

# -------------------------------------------------------------------------------------
def ann_freq_wrt_array(x: xr.Dataset, var: str, threshold: xr.Dataset | int | float, relation: str = 'gt', season: str | None = None, drop_nonfull=True, cmean=False) -> xr.Dataset:
    """
    Count number of event days, defined wrt threshold, per year (season or month).
    Option to output annual values or annual mean.
    Added by R. Eade.
    Adapted from ml-benchmark indices.py su()
    - changed season filter, generalised (gt, le etc), 
    - special case for DJF,
    - outputs annual data not just climate mean

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Name of the daily variable.
    threshold : xr.DataArray
        Array threshold to define event in same units as x dataset (e.g. 90 percentile field).
    relation : str, default 'gt'
        Relationship of event to threshold, options ('gt', 'ge', 'lt', 'le') e.g. 'gt' for events where x > threshold.
    season : str | None, optional
        Season name to filter before computing the index.
    drop_nonfull : boolean
        True: Ignore DJF seasons that are < 90 days long (JF start year, Dec end year)
        False: Ignore DJF seasons that are < 59 days long (Dec end year)
    cmean : boolean
        True: Output annual mean values; False output annual values

    Returns
    -------
    xr.Dataset
        (Mean of ) Annual count of event days.
        
    Function code adapted and generalised by Rosie Eade from original function su()
    """
    x = _filter_by_monthseason(x, season)
    da = x[[var]]
    
    if isinstance(threshold, int): threshold_new=threshold
    if isinstance(threshold, float): threshold_new=threshold

    if isinstance(threshold, xr.Dataset):
        # Should include check that shape matches
        if da.sizes["lat"] != threshold.sizes["lat"] or da.sizes["lon"] != threshold.sizes["lon"]:
            raise ValueError(
                f"Spatial shape mismatch: input ({da.sizes['lat']}, {da.sizes['lon']}) "
                f"!= threhsold ({threhsold.sizes['lat']}, {threhsold.sizes['lon']})"
            )

        dims_input = set(da.dims)
        spatial_dims_input = dims_input - {"time"}
        other_dims_input = dims_input - {"lat","lon"}
        
        dims_thresh = set(threshold.dims)
        spatial_dims_thresh = dims_thresh - {"doy_noleap"}
        other_dims_thresh = dims_thresh - {"lat","lon"}

        if not other_dims_thresh:
            threshold_new=threshold
        if other_dims_thresh:
            if other_dims_thresh=={"doy_noleap"}:
                # Should include check that shape matches
                threshold_new=xr_assign_dayofyear_noleap(da) 
                # This will become a matched array containing percentile based thresholds (matched to doy)
                input_doy=threshold_new["doy_noleap"].values # doy for input array
                thresh_doy=threshold["doy_noleap"].values # doy for daily percentile threshold array
                # Check of all doy needed for input x are present in threshold
                missing = set(np.unique(input_doy)) - set(thresh_doy)
                if missing:
                    raise ValueError(
                        f"Threshold doy doesn't cover all input doy, missing: {sorted(int(x) for x in missing)}"
                    )                
                if not missing:
                    for dcount in range(len(input_doy)): threshold_new[var].values[dcount]=threshold[var].values[np.where(thresh_doy == input_doy[dcount])[0]]
            if other_dims_thresh!={"doy_noleap"}:
                raise ValueError(
                    f"threshold dimensions {set(threshold.dims)} do not match "
                    f"x input full dims {dims_input}: Expected doy_noleap in threshold."
                )

    
    if relation not in ['gt', 'ge', 'lt', 'le', 'eq', 'ne']:
        relation = 'gt'
        print(f"WARNING: relation='{relation}' not in list ['gt', 'ge', 'lt', 'le', 'eq', 'ne'] so default to 'gt'")
    
    # Event Condition
    if relation=='gt': econdition = (da > threshold_new)
    if relation=='ge': econdition = (da >= threshold_new)
    if relation=='lt': econdition = (da < threshold_new)
    if relation=='le': econdition = (da <= threshold_new)
    if relation=='eq': econdition = (da == threshold_new)
    if relation=='ne': econdition = (da != threshold_new)

    if season=='DJF':
        # Compute on consecutive months across year boundary ending at the end of Feb
        ann_freq = econdition.resample(time="1YE-FEB").sum('time') 
        # If data covers n years going from Jan Year1 to Dec YearN, there will be
        # n+1 max values for seasons [JF, DJF, ..., DJF, D] so drop shortest season (D)
        years=np.unique(da["time.year"])
        counts = da.resample(time="1YE-FEB").count()
        if drop_nonfull:
            # Only include full DJF seasons (90 days) 
            ann_freq = ann_freq.where(counts == 90, drop=True)
        if not drop_nonfull:
            # Include full DJF seasons (90 days) and JF season (59 days) but not D season (31 days)  
            if len(counts[var]) > len(years): ann_freq = ann_freq.where(counts >= 59, drop=True)
    else:
        ann_freq = econdition.resample(time="1YE").sum('time')

    if cmean is True: output = ann_freq.mean("time")
    if cmean is False: output = ann_freq
    
    return output

# -------------------------------------------------------------------------------------
def _compute_max_spell_per_year(arr: xr.DataArray) -> xr.DataArray: 
    """
    Compute maxspell length per year.
    Added by R. Eade.
    Adapted from ml-benchmark indices.py _compute in _resample_max_spell()

    Parameters
    ----------
    arr : xr.DataArray
        Boolean array describing the spell condition over time.

    Returns
    -------
    xr.DataArray
        Resampled maximum spell length.
    """

    output=xr.apply_ufunc(indices._max_spell_length, arr, input_core_dims=[["time"]], 
        output_core_dims=[[]], vectorize=True, dask="parallelized", output_dtypes=[float] )

    return output

# -------------------------------------------------------------------------------------
def _ann_max_spell(condition: xr.DataArray, freq: str = "1YE") -> xr.DataArray:
    """
    Aggregate maximum spell length over the resampled period and 
    return annual max spell lengths.
    Added by R. Eade.
    Adapted from ml-benchmark indices.py _resample_max_spell()

    Parameters
    ----------
    condition : xr.DataArray
        Boolean array describing the spell condition over time.
    freq : str, default "1YE"
        Resampling frequency following pandas offset aliases.

    Returns
    -------
    xr.DataArray
        Resampled maximum spell length.
    """

    output=condition.resample(time=freq).map(_compute_max_spell_per_year)

    return output

# -------------------------------------------------------------------------------------
def ann_maxspell_wrt_array(x: xr.Dataset, var: str, threshold: xr.Dataset | int | float, relation: str = 'gt', season: str | None = None, drop_nonfull=True, cmean=False) -> xr.Dataset:
    """
    Compute max spell length of event days, defined wrt threshold, per year (season or month).
    Option to output annual values or annual mean.
    Added by R. Eade
    Adapted from ml-benchmark indices.py cdd()
    - changed season filter, generalised (gt, le etc), 
    - special case for DJF,
    - outputs annual data not just climate mean

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Name of the daily variable.
    threshold : xr.DataArray
        Array threshold to define event in same units as x dataset (e.g. 90 percentile field).
    relation : str, default 'gt'
        Relationship of event to threshold, options ('gt', 'ge', 'lt', 'le') e.g. 'gt' for events where x > threshold.
    season : str | None, optional
        Season name to filter before computing the index.
    drop_nonfull : boolean
        True: Ignore DJF seasons that are < 90 days long (JF start year, Dec end year)
        False: Ignore DJF seasons that are < 59 days long (Dec end year)
    cmean : boolean
        True: Output annual mean values; False output annual values

    Returns
    -------
    xr.Dataset
        (Mean of ) Annual max spell length of event days.
        
    Function code adapted and generalised by Rosie Eade from original function su()
    """

    x = _filter_by_monthseason(x, season)
    da = x[[var]]
    
    if isinstance(threshold, int): threshold_new=threshold
    if isinstance(threshold, float): threshold_new=threshold

    if isinstance(threshold, xr.Dataset):
        # Should include check that shape matches
        if da.sizes["lat"] != threshold.sizes["lat"] or da.sizes["lon"] != threshold.sizes["lon"]:
            raise ValueError(
                f"Spatial shape mismatch: input ({da.sizes['lat']}, {da.sizes['lon']}) "
                f"!= threhsold ({threhsold.sizes['lat']}, {threhsold.sizes['lon']})"
            )

        dims_input = set(da.dims)
        spatial_dims_input = dims_input - {"time"}
        other_dims_input = dims_input - {"lat","lon"}
        
        dims_thresh = set(threshold.dims)
        spatial_dims_thresh = dims_thresh - {"doy_noleap"}
        other_dims_thresh = dims_thresh - {"lat","lon"}

        if not other_dims_thresh:
            threshold_new=threshold
        if other_dims_thresh:
            if other_dims_thresh=={"doy_noleap"}:
                # Should include check that shape matches
                threshold_new=xr_assign_dayofyear_noleap(da) 
                # This will become a matched array containing percentile based thresholds (matched to doy)
                input_doy=threshold_new["doy_noleap"].values # doy for input array
                thresh_doy=threshold["doy_noleap"].values # doy for daily percentile threshold array
                # Check of all doy needed for input x are present in threshold
                missing = set(np.unique(input_doy)) - set(thresh_doy)
                if missing:
                    raise ValueError(
                        f"Threshold doy doesn't cover all input doy, missing: {sorted(int(x) for x in missing)}"
                    )                
                if not missing:
                    for dcount in range(len(input_doy)): threshold_new[var].values[dcount]=threshold[var].values[np.where(thresh_doy == input_doy[dcount])[0]]
            if other_dims_thresh!={"doy_noleap"}:
                raise ValueError(
                    f"threshold dimensions {set(threshold.dims)} do not match "
                    f"x input full dims {dims_input}: Expected doy_noleap in threshold."
                )

    
    if relation not in ['gt', 'ge', 'lt', 'le', 'eq', 'ne']:
        relation = 'gt'
        print(f"WARNING: relation='{relation}' not in list ['gt', 'ge', 'lt', 'le', 'eq', 'ne'] so default to 'gt'")

    # Event Condition
    if relation=='gt': econdition = (da > threshold_new)
    if relation=='ge': econdition = (da >= threshold_new)
    if relation=='lt': econdition = (da < threshold_new)
    if relation=='le': econdition = (da <= threshold_new)
    if relation=='eq': econdition = (da == threshold_new)
    if relation=='ne': econdition = (da != threshold_new)

    if season=='DJF':
        # Compute on consecutive months across year boundary ending at the end of Feb
        ann_spell = _ann_max_spell(econdition, freq="1YE-FEB")
        years=np.unique(da["time.year"])
        counts = da.resample(time="1YE-FEB").count()
        # If data covers n years going from Jan Year1 to Dec YearN, there will be
        # n+1 max values for seasons [JF, DJF, ..., DJF, D] so either
        # - drop all non-full seasons i.e. JF and D
        if drop_nonfull:
            # Only include full DJF seasons (90 days) 
            ann_spell = ann_spell.where(counts == 90, drop=True)
        # - drop shortest season (D)
        if not drop_nonfull:
            # Include full DJF seasons (90 days) and JF season (59 days) but not D season (31 days)
            if len(counts[var]) > len(years): ann_spell = ann_spell.where(counts >= 59, drop=True)
    else:
        ann_spell = _ann_max_spell(econdition, freq="1YE")

    if cmean is True: output = ann_spell.mean("time")
    if cmean is False: output = ann_spell
    
    return output


# -------------------------------------------------------------------------------------
def ann_sdii(
    x: xr.Dataset,
    var: str,
    season: str | None = None,
    wet_threshold: float = 1.0,
    drop_nonfull=True, 
    cmean=False,
) -> xr.Dataset:
    """
    Compute the simple precipitation intensity index.
    Added by R. Eade
    Adapted from ml-benchmark indices.py sdii()
    - changed season filter 
    - special case for DJF
    - outputs annual data not just climate mean

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
    drop_nonfull : boolean
        True: Ignore DJF seasons that are < 90 days long (JF start year, Dec end year)
        False: Ignore DJF seasons that are < 59 days long (Dec end year)
    cmean : boolean
        True: Output annual mean values; False output annual values

    Returns
    -------
    xr.Dataset
        Simple precipitation intensity index.
    """
    x = _filter_by_monthseason(x, season)
    da = x[[var]]
    wet = da.where(da >= wet_threshold)

    if season=='DJF':
        # Compute on consecutive months across year boundary ending at the end of Feb
        output = wet.resample(time="1YE-FEB").mean(dim='time')
        # If data covers n years going from Jan Year1 to Dec YearN, there will be
        # n+1 max values for seasons [JF, DJF, ..., DJF, D] so drop shortest season (D)
        years=np.unique(da["time.year"])
        counts = da.resample(time="1YE-FEB").count()
        if drop_nonfull:
            # Only include full DJF seasons (90 days) 
            output = output.where(counts == 90, drop=True)
        if not drop_nonfull:
            # Include full DJF seasons (90 days) and JF season (59 days) but not D season (31 days)  
            if len(counts[var]) > len(years): output = output.where(counts >= 59, drop=True)
    else:
        output = wet.resample(time="1YE").mean(dim='time')

    if cmean is True: output = output.mean("time")
    if cmean is False: output = output

    return output

# -------------------------------------------------------------------------------------
def interannual_var(x: xr.Dataset, var: str, season: str | None = None, drop_nonfull=True) -> xr.Dataset:
    """
    Compute the interannual variability of the target variable along time.
    Added by R. Eade
    Adapted from ml-benchmark indices.py interannual_var()
    - changed season filter 
    - special case for DJF

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season name to filter before computing the index.
    drop_nonfull : boolean
        True: Ignore DJF seasons that are < 90 days long (JF start year, Dec end year)
        False: Ignore DJF seasons that are < 59 days long (Dec end year)

    Returns
    -------
    xr.Dataset
        Interannual variability computed along the time dimension.
    """

    x = _filter_by_monthseason(x, season)
    da = x[[var]]

    if season=='DJF':
        # Compute on consecutive months across year boundary ending at the end of Feb
        output = da.resample(time="1YE-FEB").mean(dim='time').std('time')
        # If data covers n years going from Jan Year1 to Dec YearN, there will be
        # n+1 max values for seasons [JF, DJF, ..., DJF, D] so drop shortest season (D)
        years=np.unique(da["time.year"])
        counts = da.resample(time="1YE-FEB").count()
        if drop_nonfull:
            # Only include full DJF seasons (90 days) 
            output = output.where(counts == 90, drop=True)
        if not drop_nonfull:
            # Include full DJF seasons (90 days) and JF season (59 days) but not D season (31 days)  
            if len(counts[var]) > len(years): output = output.where(counts >= 59, drop=True)
    else:
        output = da.resample(time="1YE").mean(dim='time').std('time')

    return output


# -------------------------------------------------------------------------------------
# Code directly from ml-benchmark indices.py with just the season filter updated
# -------------------------------------------------------------------------------------

def lag1_corr(x: xr.Dataset, var: str, season: str | None = None) -> xr.Dataset:
    """
    Compute the lag-1 autocorrelation of the target variable along time.

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

    x = _filter_by_monthseason(x, season)
    da = x[var]
    shifted = da.shift(time=1)
    lag1_corr = xr.corr(da, shifted, dim="time")
    return xr.Dataset({var: lag1_corr})

