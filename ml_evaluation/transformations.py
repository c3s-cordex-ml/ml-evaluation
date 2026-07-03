from __future__ import annotations

import numpy as np
import xarray as xr


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
        try:
            dataset = dataset.convert_calendar("standard", missing=np.nan)
        except ValueError:
            dataset = dataset.convert_calendar("standard")
        dataset = dataset.ffill(dim="time")
    return dataset
