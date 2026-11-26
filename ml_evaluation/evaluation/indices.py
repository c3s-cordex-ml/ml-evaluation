# SPDX-FileCopyrightText: 2026 c3s-cordex-ml
# SPDX-License-Identifier: MIT

# NOTE: This module contains third-party code distributed under the MIT License.
# Each third-party function is marked with an SPDX header immediately above its
# definition. Full copyright notices and license texts are in
# THIRD_PARTY_NOTICES.md (the project's own license is in LICENSE).

from __future__ import annotations

from datetime import timedelta

import numpy as np
import xarray as xr

from ml_evaluation.transformations import spatial_dims

from ml_evaluation.evaluation.utils import (
    filter_by_monthseason,
    ann_max_spell,
    ann_spell_days,
)


# SPDX-FileCopyrightText: 2026 CORDEX
# SPDX-License-Identifier: MIT
# Third-party code (modified); see THIRD_PARTY_NOTICES.md, section 1.
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


def season_quantile(
    x: xr.Dataset,
    var: str,
    q: float,
    season: str | None = None,
    wet_threshold: float | None = None,
    pool_dim: str | None = None,
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
    pool_dim : str | None, optional
        Extra dimension (e.g. ensemble ``"member"``) to pool into the quantile,
        giving one quantile shared by all members. If None (default), or if
        the dimension is absent, any such dimension is kept and the quantile
        is computed separately along it.

    Returns
    -------
    xr.Dataset
        Quantile of the input data.
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]

    if wet_threshold is not None:
        da = da.where(da >= wet_threshold)

    quantile_dims = ["time"]
    if pool_dim is not None:
        if pool_dim in da.dims:
            quantile_dims.append(pool_dim)
        else:
            print(
                f"WARNING: pool_dim='{pool_dim}' is not a dimension of the "
                "dataset, so nothing is pooled."
            )

    result = da.quantile(q, dim=tuple(quantile_dims), skipna=True)
    if "quantile" in result.dims:
        result = result.sel(quantile=q)
    return result


def season_std(x: xr.Dataset, var: str, season: str | None = None) -> xr.Dataset:
    """
    Compute the standard deviation of a variable along time, per grid point.

    Filters the dataset by season or month if specified, then takes the spread of
    the values themselves -- the *daily* variability, unlike
    :func:`interannual_var`, which averages each year away first.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Variable name to compute the standard deviation for.
    season : str | None, optional
        Season or Month standard name to filter before computing the spread.

    Returns
    -------
    xr.Dataset
        Standard deviation along the time dimension.
    """
    x_filtered = filter_by_monthseason(x, season)
    return x_filtered[[var]].std("time", skipna=True)


def calday_quantile(
    x: xr.Dataset,
    var: str,
    q: float,
    padding: int = 2,
    pool_dim: str | None = None,
) -> xr.Dataset:
    """
    Compute a calendar-day quantile using a moving day-window.

    For each day of year, the quantile is taken over all years and over a
    window of +/- ``padding`` days around that day.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the daily variable.
    var : str
        Name of the daily variable.
    q : float
        Quantile to compute (0-1).
    padding : int, default 2
        Half-width of the day-window (window length is ``2 * padding + 1``).
    pool_dim : str | None, optional
        Extra dimension (e.g. ensemble ``"member"``) to pool into the quantile,
        giving one threshold shared by all members. If None (default), or if
        the dimension is absent, any such dimension is kept and the threshold
        is computed separately along it.

    Returns
    -------
    xr.Dataset
        Quantile field with a ``dayofyear`` dimension.
    """
    da = x[[var]]
    # Private name for the day-window dimension so it cannot clash with an
    # existing ensemble "member" dimension (which is kept, giving per-member
    # thresholds).
    window_dim = "_calday_window"
    n_window = padding * 2 + 1
    center = n_window // 2  # = 2
    shifts = [i - center for i in range(n_window)]  # [-2, -1, 0, 1, 2]
    windows = [da.roll(time=s, roll_coords=False) for s in shifts]
    da_windows = xr.concat(windows, dim=window_dim)

    quantile_dims = [window_dim, "time"]
    if pool_dim is not None:
        if pool_dim in da_windows.dims:
            quantile_dims.append(pool_dim)
        else:
            print(
                f"WARNING: pool_dim='{pool_dim}' is not a dimension of the "
                "dataset, so nothing is pooled."
            )

    # Group by calendar day-of-year, then take the quantile over the
    # day-window, time (all years) and any pooled dimension
    result = da_windows.groupby("time.dayofyear").quantile(
        q, dim=tuple(quantile_dims), skipna=True
    )

    if "quantile" in result.dims:
        result = result.sel(quantile=q)
    return result


# SPDX-FileCopyrightText: 2026 CORDEX
# SPDX-License-Identifier: MIT
# Third-party code (modified); see THIRD_PARTY_NOTICES.md, section 1.
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
        print(
            f"WARNING: maxmin='{maxmin}' not in list ['max','min'] so default to 'max'"
        )
        maxmin = "max"

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


def ann_max_nday(
    x: xr.Dataset,
    var: str,
    ndays: int = 5,
    stat: str = "sum",
    season: str | None = None,
    drop_nonfull: bool = True,
    cmean: bool = False,
) -> xr.Dataset:
    """
    Compute the annual maximum of the n-day running total (or mean) per year.

    With the defaults this is the Rx5day index (maximum consecutive 5-day
    precipitation total). Has options to output annual values or climate mean.

    Each n-day window is labelled by its last day and assigned to the year (or
    DJF season) containing that day. Only windows of ``ndays`` consecutive
    calendar days are used: windows spanning a gap in the time axis (e.g. the
    jump between seasons after filtering with ``season``, or missing time
    steps) are discarded, as are windows containing any NaN value.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Name of the daily variable.
    ndays : int, default 5
        Length of the running window in days.
    stat : str, default 'sum'
        Statistic over each window, options are 'sum' (n-day total) or 'mean'
        (n-day mean).
    season : str | None, optional
        Season or Month standard name to filter. Windows must lie entirely
        within the season.
    drop_nonfull : bool, default True
        Treatment of DJF season: True to only include seasons with full 3 months
        of data.
    cmean : bool, default False
        True to compute climate mean over time, False to leave as annual output.

    Returns
    -------
    xr.Dataset
        Annual maximum of the n-day running total or mean of the daily variable
        (or mean thereof).

    Raises
    ------
    ValueError
        If ``ndays`` is less than 1 or ``stat`` is not 'sum' or 'mean'.
    """
    if ndays < 1:
        raise ValueError(f"ndays must be >= 1, got {ndays}")
    if stat not in ["sum", "mean"]:
        raise ValueError(f"stat='{stat}' not in list ['sum', 'mean']")

    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]

    # Running n-day statistic, labelled at the last day of each window. The
    # default min_periods (= ndays) gives NaN for any window containing a NaN.
    rolling = da.rolling(time=ndays)
    running = rolling.sum() if stat == "sum" else rolling.mean()

    # Discard windows that are not ndays consecutive calendar days, e.g. those
    # spanning the gap between one season and the next after filtering.
    if ndays > 1:
        tindex = da.indexes["time"]
        span = tindex[ndays - 1 :] - tindex[: -(ndays - 1)]
        contiguous = np.zeros(da.sizes["time"], dtype=bool)
        contiguous[ndays - 1 :] = span == timedelta(days=ndays - 1)
        running = running.where(
            xr.DataArray(contiguous, dims="time", coords={"time": da["time"]})
        )

    if season == "DJF":
        # Compute on consecutive months across year boundary ending at the end of Feb
        annual_max = running.resample(time="1YE-FEB").max()
        # If data covers n years going from Jan Year1 to Dec YearN, there will be
        # n+1 max values for seasons [JF, DJF, ..., DJF, D] so drop shortest season (D)
        years = np.unique(da["time.year"])
        counts = da.resample(time="1YE-FEB").count()
        if drop_nonfull:
            # Only include full DJF seasons (90 days)
            annual_max = annual_max.where(counts == 90, drop=True)
        else:
            # Include full DJF seasons (90 days) and JF season (59 days)
            # but not D season (31 days)
            if len(counts[var]) > len(years):
                annual_max = annual_max.where(counts >= 59, drop=True)
    else:
        annual_max = running.resample(time="1YE").max()

    if cmean:
        output: xr.Dataset = annual_max.mean(dim="time")
    else:
        output = annual_max

    return output


# SPDX-FileCopyrightText: 2026 CORDEX
# SPDX-License-Identifier: MIT
# Third-party code (modified); see THIRD_PARTY_NOTICES.md, section 1.
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
        print(f"WARNING: relation='{relation}' not in list: ")
        print("['gt', 'ge', 'lt', 'le', 'eq', 'ne'] so default to 'gt'")
        relation = "gt"

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


# SPDX-FileCopyrightText: 2026 CORDEX
# SPDX-License-Identifier: MIT
# Third-party code (modified); see THIRD_PARTY_NOTICES.md, section 1.
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
        print(f"WARNING: relation='{relation}' not in list: ")
        print("['gt', 'ge', 'lt', 'le', 'eq', 'ne'] so default to 'gt'")
        relation = "gt"

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


def ann_spellfreq_wrt_array(
    x: xr.Dataset,
    var: str,
    threshold: xr.Dataset | int | float,
    min_length: int = 6,
    relation: str = "gt",
    season: str | None = None,
    drop_nonfull: bool = True,
    cmean: bool = False,
) -> xr.Dataset:
    """
    Count days in spells of at least ``min_length`` event days per year.

    Event days are defined wrt threshold. Only days belonging to runs of
    >= ``min_length`` consecutive event days are counted (e.g. WSDI/CSDI with
    ``min_length=6``). Spells are split at year (or DJF-season) boundaries.
    Has options to output annual values or annual mean.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the target variable.
    var : str
        Name of the daily variable.
    threshold : xr.Dataset or int or float
        Array threshold or scalar to define event.
    min_length : int, default 6
        Minimum number of consecutive event days for a spell to be counted.
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
        Annual count of days in spells of length >= ``min_length``
        (or mean thereof).

    Raises
    ------
    ValueError
        If ``min_length`` < 1 or if shape of threshold does not match input.
    """
    if min_length < 1:
        raise ValueError(f"min_length must be >= 1, got {min_length}")

    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[[var]]

    threshold_new: xr.Dataset | int | float
    if isinstance(threshold, (int, float)):
        threshold_new = threshold
    elif isinstance(threshold, xr.Dataset):
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
        print(f"WARNING: relation='{relation}' not in list: ")
        print("['gt', 'ge', 'lt', 'le', 'eq', 'ne'] so default to 'gt'")
        relation = "gt"

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
        ann_sfreq = ann_spell_days(econdition, min_length, freq="1YE-FEB")
        # If data covers n years going from Jan Year1 to Dec YearN, there will be
        # n+1 values for seasons [JF, DJF, ..., DJF, D] so drop shortest season (D)
        years = np.unique(da["time.year"])
        counts = da.resample(time="1YE-FEB").count()
        if drop_nonfull:
            # Only include full DJF seasons (90 days)
            ann_sfreq = ann_sfreq.where(counts == 90, drop=True)
        else:
            # Include full DJF seasons (90 days) and JF season (59 days)
            # but not D season (31 days)
            if len(counts[var]) > len(years):
                ann_sfreq = ann_sfreq.where(counts >= 59, drop=True)
    else:
        ann_sfreq = ann_spell_days(econdition, min_length, freq="1YE")

    if isinstance(ann_sfreq, xr.Dataset):
        ann_sfreq_ds = ann_sfreq
    else:
        ann_sfreq_ds = xr.Dataset({var: ann_sfreq})

    if cmean:
        output: xr.Dataset = ann_sfreq_ds.mean("time")
    else:
        output = ann_sfreq_ds

    return output


# SPDX-FileCopyrightText: 2026 CORDEX
# SPDX-License-Identifier: MIT
# Third-party code (modified); see THIRD_PARTY_NOTICES.md, section 1.
def ann_sdii(
    x: xr.Dataset,
    var: str,
    season: str | None = None,
    wet_threshold: float = 1.0,
    drop_nonfull: bool = True,
    cmean: bool = False,
) -> xr.Dataset:
    """
    Compute the simple precipitation intensity index (average rainfall on wet days).

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


def ann_sum_wrt_array(
    x: xr.Dataset,
    var: str,
    threshold: xr.Dataset | xr.DataArray | int | float,
    season: str | None = None,
    drop_nonfull: bool = True,
    cmean: bool = False,
) -> xr.Dataset:
    """
    Compute the total precipitation from days exceeding a threshold, per year.

    E.g. R95pTOT-style totals when ``threshold`` is a 95th percentile field, or
    PRCPTOT-style wet-day totals when ``threshold`` is a scalar such as 1.0.
    Has options to output annual values or annual mean.

    Parameters
    ----------
    x : xr.Dataset
        Dataset containing the precipitation variable.
    var : str
        Variable name of daily precipitation.
    threshold : xr.Dataset or xr.DataArray or int or float
        Scalar threshold, or threshold field. A field must match the spatial
        shape of ``x``; it may optionally carry a ``dayofyear`` dimension (e.g.
        output of ``calday_quantile``), which is mapped onto the time axis. If a
        Dataset, ``threshold[var]`` is used (or its only variable).
    season : str | None, optional
        Season name to filter before computing the index.
    drop_nonfull : bool, default True
        Treatment of DJF season: True to only include seasons with full 3 months
        of data.
    cmean : bool, default False
        True to compute climate mean over time, False to leave as annual output.

    Returns
    -------
    xr.Dataset
        Annual total of precipitation on days with values > threshold (or mean
        thereof). Years with no exceedances give 0; points that are entirely
        missing remain NaN.

    Raises
    ------
    ValueError
        If threshold spatial shape does not match input, or a threshold Dataset
        has no unambiguous variable to use.
    """
    x_filtered = filter_by_monthseason(x, season)
    da = x_filtered[var]

    threshold_new: xr.DataArray | int | float
    if isinstance(threshold, (int, float)):
        threshold_new = threshold
    else:
        if isinstance(threshold, xr.Dataset):
            if var in threshold.data_vars:
                threshold = threshold[var]
            elif len(threshold.data_vars) == 1:
                threshold = threshold[next(iter(threshold.data_vars))]
            else:
                raise ValueError(
                    f"Threshold Dataset has no variable '{var}' and more than one "
                    f"data variable: {list(threshold.data_vars)}"
                )

        sdims = spatial_dims(x_filtered, var)
        if any(da.sizes[d] != threshold.sizes.get(d) for d in sdims):
            raise ValueError(
                f"Spatial shape mismatch: input "
                f"({', '.join(str(da.sizes[d]) for d in sdims)}) "
                f"!= threshold "
                f"({', '.join(str(threshold.sizes.get(d)) for d in sdims)}) "
                f"on dims {sdims}"
            )

        if "dayofyear" in threshold.dims:
            # Broadcast the threshold back onto actual time axis via dayofyear
            threshold_new = threshold.sel(dayofyear=da["time"].dt.dayofyear)
        else:
            threshold_new = threshold

    # Keep values above threshold, 0 elsewhere, NaN where input is missing
    exceed = xr.where(da > threshold_new, da, 0.0).where(da.notnull())
    # Drop dayofyear coord picked up from a day-of-year threshold
    exceed = exceed.drop_vars("dayofyear", errors="ignore")

    if season == "DJF":
        # Compute on consecutive months across year boundary ending at the end of Feb
        output = exceed.resample(time="1YE-FEB").sum(dim="time", min_count=1)
        # If data covers n years going from Jan Year1 to Dec YearN, there will be
        # n+1 values for seasons [JF, DJF, ..., DJF, D] so drop shortest season (D)
        years = np.unique(da["time.year"])
        counts = da.resample(time="1YE-FEB").count()
        if drop_nonfull:
            # Only include full DJF seasons (90 days)
            output = output.where(counts == 90, drop=True)
        else:
            # Include full DJF seasons (90 days) and JF season (59 days)
            # but not D season (31 days)
            if counts.sizes["time"] > len(years):
                output = output.where(counts >= 59, drop=True)
    else:
        output = exceed.resample(time="1YE").sum(dim="time", min_count=1)

    if cmean:
        output = output.mean("time")

    return output.to_dataset(name=var)


# SPDX-FileCopyrightText: 2026 CORDEX
# SPDX-License-Identifier: MIT
# Third-party code (modified); see THIRD_PARTY_NOTICES.md, section 1.
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


# SPDX-FileCopyrightText: 2026 CORDEX
# SPDX-License-Identifier: MIT
# Third-party code (modified); see THIRD_PARTY_NOTICES.md, section 1.
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
