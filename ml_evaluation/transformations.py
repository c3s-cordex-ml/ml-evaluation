from __future__ import annotations

import xarray as xr
import pandas as pd
import cftime


def align_calendar(dataset: xr.Dataset, align_leap_days: bool = False) -> xr.Dataset:
    """
    Align the calendar of an xarray Dataset to a standard calendar.

    Converts the dataset to a standard calendar, ensuring that February 29th
    is present and filled (forward-filled from February 28th) in leap years.

    Parameters
    ----------
    dataset : xr.Dataset
        The input xarray Dataset to be aligned.
    align_leap_days : bool, optional
        Whether to align the calendar by filling missing leap days (default is False).

    Returns
    -------
    xr.Dataset
        The aligned xarray Dataset.
    """
    if align_leap_days:
        t = dataset.time.to_index()

        if isinstance(t[0], cftime.datetime):
            # cftime-based (e.g. noleap calendar) -> build a standard-calendar cftime range
            full_time = xr.date_range(
                start=t[0], end=t[-1], freq="D", calendar="standard", use_cftime=True
            )
        else:
            # plain numpy/pandas datetime64
            full_time = pd.date_range(start=t[0], end=t[-1], freq="D")

        dataset = dataset.reindex(time=full_time)
        dataset = dataset.ffill(dim="time")

    print("TEST OUTPUT")
    print(dataset.time.values)
    print(dataset["temp"].values)
    print(len(dataset.time))

    #    # Original code failed for 365 day calendars with Gregorian timestamps?
    #    # as found conflicting time frequency (1 day, but 2 days for 28 Feb to 1 mar)
    #    if align_leap_days:
    #        try:
    #            dataset = dataset.convert_calendar("standard", missing=np.nan)
    #        except ValueError as e:
    #            print("convert_calendar failed:", e)
    #            dataset = dataset.convert_calendar("standard")
    #        dataset = dataset.ffill(dim="time")

    return dataset
