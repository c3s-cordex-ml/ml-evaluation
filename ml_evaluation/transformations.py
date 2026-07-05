from __future__ import annotations

import numpy as np
import xarray as xr
import pandas as pd
import cftime


def align_calendar(
    dataset: xr.Dataset, align_leap_days: str | None = None
) -> xr.Dataset:
    """
    Align the calendar of an xarray Dataset to a standard calendar.

    Converts the dataset to a standard calendar, ensuring that either:
    February 29th is present and filled (forward-filled from February 28th) in leap years.
    February 29th is removed from datasets where present

    Parameters
    ----------
    dataset : xr.Dataset
        The input xarray Dataset to be aligned.
    align_leap_days : str | None
        Whether to align the calendar by filling missing leap days (default is None).
        "fill" : fill using February 28th
        "remove" : remove February 29th

    Returns
    -------
    xr.Dataset
        The aligned xarray Dataset.
    """
    if align_leap_days == "fill":
        t = dataset.time.to_index()

        if isinstance(t[0], cftime.datetime):
            # Original code (cftime based)
            try:
                dataset = dataset.convert_calendar("standard", missing=np.nan)
            except ValueError as e:
                print("convert_calendar failed:", e)
                dataset = dataset.convert_calendar("standard")
            dataset = dataset.ffill(dim="time")
        else:
            # Alt numpy/pandas datetime64 for 365 day calendars with Gregorian timestamps?
            # as found conflicting time frequency (1 day, but 2 days for 28 Feb to 1 mar)
            full_time = pd.date_range(start=t[0], end=t[-1], freq="D")

            dataset = dataset.reindex(time=full_time)
            dataset = dataset.ffill(dim="time")

    elif align_leap_days == "remove":
        dataset = dataset.sel(
            time=~((dataset.time.dt.month == 2) & (dataset.time.dt.day == 29))
        )

    elif align_leap_days is not None:
        raise ValueError(
            f"Invalid align_leap_days={align_leap_days!r}; "
            'expected "fill", "remove", or None.'
        )

    return dataset
