from __future__ import annotations

import numpy as np
import xarray as xr
import pandas as pd
import cftime


# Horizontal dimension names, most specific convention first. A regular grid
# dimensioned by lat/lon (TROPICAL) matches the first pair; a rotated or
# otherwise curvilinear grid dimensioned by y/x with 2-D lat(y,x)/lon(y,x)
# coordinates (CORDEX-BENCH-ALPS) matches the second.
SPATIAL_DIM_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("lat", "lon"),
    ("latitude", "longitude"),
    ("y", "x"),
    ("rlat", "rlon"),
)


def spatial_dims(
    obj: xr.DataArray | xr.Dataset, var: str | None = None
) -> tuple[str, str]:
    """
    The two horizontal dimensions of a variable, whatever convention it uses.

    Resolution is done on the DIMENSIONS OF THE VARIABLE, never on coordinate
    names and never on ``Dataset.dims``. That distinction is what makes the
    common curvilinear layout unambiguous: a CORDEX file carries lat, lon, y
    and x all four as coordinates, but its data variables are dimensioned by
    (y, x) alone, with lat and lon as 2-D auxiliary coordinates over them. Only
    (y, x) can be reduced over, and only (y, x) is returned here.

    Passing a Dataset without ``var`` resolves against the dataset's dimensions
    and is therefore only safe when every variable shares one horizontal grid;
    prefer passing the DataArray, or the Dataset plus the variable name.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        The object whose horizontal dims are wanted.
    var : str, optional
        Variable to resolve against when ``obj`` is a Dataset.

    Returns
    -------
    tuple of str
        The (vertical, horizontal) dimension names, e.g. ("lat", "lon") or
        ("y", "x").

    Raises
    ------
    ValueError
        If no candidate pair is present, or if several are and the trailing
        two dimensions do not settle which one carries the data.
    """
    if var is not None:
        obj = obj[var]
    dims = tuple(str(d) for d in obj.dims)
    dim_set = set(dims)

    matches = [pair for pair in SPATIAL_DIM_CANDIDATES if set(pair) <= dim_set]
    if not matches:
        raise ValueError(
            f"No horizontal dimensions found in {dims}; expected one of "
            f"{[list(p) for p in SPATIAL_DIM_CANDIDATES]}."
        )
    if len(matches) > 1:
        # Both conventions are dimensions of the same variable (rare). The
        # horizontal axes are the trailing two by convention -- if that picks
        # one of the candidates, trust it; otherwise refuse to guess.
        trailing = set(dims[-2:])
        trailing_matches = [pair for pair in matches if set(pair) == trailing]
        if len(trailing_matches) != 1:
            raise ValueError(
                f"Ambiguous horizontal dimensions in {dims}: "
                f"{[list(p) for p in matches]} all present."
            )
        return trailing_matches[0]
    return matches[0]


def spatial_dims_present(
    obj: xr.DataArray | xr.Dataset, var: str | None = None
) -> tuple[str, ...]:
    """
    Like :func:`spatial_dims`, but returns ``()`` instead of raising.

    For code that merely wants to reduce over the horizontal dims if there are
    any (spatial means over already-reduced metrics, chunk sizing, ...).
    """
    try:
        return spatial_dims(obj, var)
    except ValueError:
        return ()


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


def convert_threshold_units(x0_ds, var):
    """
    Hardcoded index functions thresholds assume units:
    * Temperature in degC
    * Precipitation in mm/day
    Read in units of var in x0_ds and output conversions
    needed to apply to thresholds in index functions.

    Parameters
    ----------
    dataset : xr.Dataset
        The input xarray Dataset to be aligned.
    var : str
        Variable name

    Returns
    -------
    (float, float)

    """

    scale = 1.0
    shift = 0.0

    ds_units = x0_ds[var].attrs.get("units")

    if ds_units in ["K", "k", "degk", "kelvin", "degrees_k"]:
        shift = +273.15  # deg C threshold to K

    if ds_units in ["kg m-2 s-1", "kg/m2/s", "kg m^-2 s^-1"]:
        scale = 1 / 86400.0  # mm/day threshold to kg m-2 s-1

    if isinstance(ds_units, type(None)):
        print(f"Warning: Units not detected for variable {var}")

    return scale, shift


def align_dims(x0_ds, x1_ds, member=None):
    """
    Align x0_ds and x1_ds along lat/lon and time, and detect a member dimension.

    - Verifies lat/lon coordinates match (exactly or within tolerance), and
      snaps x1_ds's coords to x0_ds's if they're only "close".
    - Subsets both datasets to their common time steps, so the returned
      datasets are time-aligned.
    - Detects an extra "member"-like dimension in x1_ds relative to x0_ds
      and standardizes its coordinate/name to "member".
    - Optionally subsets x1_ds to a single member (by position), or
      collapses to the ensemble mean, while keeping "member" as a
      (length-1) dimension.

    Parameters
    ----------
    x0_ds : xr.Dataset
    x1_ds : xr.Dataset
    member : int, optional
        If provided, select from x1_ds's standardized "member" dimension
        after member detection, retaining "member" as a size-1 dimension
        rather than squeezing it out:
        - member == -1 : compute the ensemble mean over all members.
        - member >= 0  : select that single member by integer position.
        Ignored (with a warning) if x1_ds has no member dimension.

    Returns
    -------
    x0_ds : xr.Dataset
        Possibly subset to the common time steps.
    x1_ds : xr.Dataset
        Possibly subset to the common time steps, with coords aligned to x0_ds,
        member dimension standardized if present, and optionally reduced to
        a single member or ensemble mean (dimension retained).
    member_dim : str or None
        Name of the standardized member dimension in x1_ds, if any.
    """

    # Check that lat and lon coords exactly match
    CheckEqual = np.array_equal(x0_ds.lat.values, x1_ds.lat.values)
    CheckClose = np.allclose(x0_ds.lat.values, x1_ds.lat.values)
    if not CheckEqual and not CheckClose:
        raise ValueError("lat and/or lon coordinates don't match")
    if not CheckEqual and CheckClose:
        print("Warning: lat and/or lon coordinates close but needed aligning")
        x1_ds = x1_ds.assign_coords(lat=x0_ds.lat, lon=x0_ds.lon)

    # Align time steps: subset both datasets to their common times
    if "time" in x0_ds.dims and "time" in x1_ds.dims:
        common_times = np.intersect1d(x0_ds.time.values, x1_ds.time.values)
        if common_times.size == 0:
            raise ValueError("x0_ds and x1_ds have no overlapping time steps")
        if not np.array_equal(x0_ds.time.values, common_times):
            x0_ds = x0_ds.sel(time=common_times)
        if not np.array_equal(x1_ds.time.values, common_times):
            x1_ds = x1_ds.sel(time=common_times)

    # Spot member dimension name if exists
    member_dim = None
    x0_dims = set(x0_ds.dims)
    x1_dims = set(x1_ds.dims)
    extra_dim = x1_dims - x0_dims
    # if there is more than one extra dim, we cannot handle this case
    if len(extra_dim) > 1:
        raise ValueError(
            f"Error: can't handle x1 dims {x1_dims} with more than "
            f"one extra dim than x0 dims {x0_dims}"
        )
    if len(extra_dim) == 1:
        member_dim = list(extra_dim)[0]
        x1_ds = ensure_member_coords(x1_ds, dim=member_dim, new_name="member")
        member_dim = "member"

        # Optionally select a single member, keeping "member" as a dimension
        if member is not None:
            if member == -1:
                # Ensemble mean, keeping "member" as a size-1 dimension
                x1_ds = x1_ds.mean(dim="member", keep_attrs=True)
                x1_ds = x1_ds.expand_dims(member=["mean"])
            else:
                n_members = x1_ds.sizes["member"]
                if member < 0 or member >= n_members:
                    raise IndexError(
                        f"member index {member} out of bounds for dimension "
                        f"of size {n_members}"
                    )
                x1_ds = x1_ds.isel(member=[member])
    elif member is not None:
        print(
            "Warning: 'member' argument was given but x1_ds has no "
            "member dimension to select from"
        )

    return x0_ds, x1_ds, member_dim


def ensure_member_coords(ds, dim="ensemble", labels=None, new_name="member"):
    """
    Check if `dim` has real coordinates on ds; assign default ones if not.
    Then rename the dimension (and its coordinate) to `new_name`.

    Parameters
    ----------
    ds : xarray.Dataset or DataArray
    dim : str
        Name of the dimension to check.
    labels : list-like, optional
        Coordinate values to assign if missing. Defaults to
        [0, 1, 2, ...] based on the dimension size.
    new_name : str, optional
        New name to give the dimension after assigning coordinates.
        Set to None to skip renaming.
    """
    if dim not in ds.dims:
        print(
            f"WARNING: '{dim}' is not a dimension of this dataset. Return dataset unchanged."
        )
        return ds

    # A dimension has "real" coordinates only if it appears in ds.coords
    if dim in ds.coords:
        print(f"'{dim}' already has coordinates: {ds.coords[dim].values}")
    else:
        size = ds.sizes[dim]
        if labels is None:
            labels = list(range(size))
        else:
            if len(labels) != size:
                raise ValueError(
                    f"Length of labels ({len(labels)}) does not match "
                    f"size of '{dim}' ({size})"
                )

        print(f"'{dim}' has no coordinates - assigning: {labels}")
        ds = ds.assign_coords({dim: labels})

    # Rename the dimension (and its coordinate, since they share the name)
    if new_name is not None and dim in ds.dims and dim != new_name:
        print(f"Renaming dimension '{dim}' -> '{new_name}'")
        ds = ds.rename({dim: new_name})

    return ds
