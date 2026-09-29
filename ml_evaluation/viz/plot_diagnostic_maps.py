"""
Code to plot a matrices of lat/lon gridded fields from the validation package.

There are two different methods for obtaining data:

1. Pre-validation package NetCDF files -- single-date "event" figures

  ml-eval only stores time-reduced diagnostics, so a single day is never in a
  zarr store: these read the prediction and target NetCDFs directly. This is the
  eyeball check for wrong units, spatial artefacts or over-smoothing, done
  before any metric is trusted.

  `paths` is the same {label: path} mapping as below, except `path` is that
  source's NetCDF file.

  plot_multivar_event_maps()
  - Multiple variables (columns) x multiple models (rows), one figure, one date
  - Same layout and `kind` / `show_truth` behaviour as plot_multivar_skill_maps()

  plot_var_event_maps()
  - A single variable, one model per row; one figure per variable, one date
  - Same layout and `kind` / `show_truth` behaviour as plot_var_skill_maps()

  - Colorbar ranges come from EVENT_VMINMAX (per variable, standard units)

2. Post-validation package Zarr files

  Sources are an ordered ``{label: path}`` mapping, exactly as in the rest of
  ml_evaluation.viz (plot_hexagon, plot_var_skill_scorecard, plot_var_series): the
  label is what appears in the figure, ``path`` is that run's base directory,
  and one variable's stores are read from ``<path>/<var>/``. Nothing is searched
  and no directory layout is assumed.

  There are 2 main types of figures included:

  Both plot ONE index (e.g. "mean") at a time; `kind` says what the model
  panels show -- "raw" (the index itself), "bias", or "relbias" (the bias as a
  percentage of the truth).

  plot_multivar_skill_maps()
  - Multiple variables (columns) x multiple models (rows), one figure
  - Grid is (1 + N models) x len(variables); the truth takes the top row
    (show_truth=False drops it)
  - kind="bias"/"relbias" REPLACES the field in the model rows
  - The map counterpart of plot_multivar_skill_scorecard()
  - Colorbar range and colormap defined by get_example_color_scales()

  plot_var_skill_maps()
  - A single variable, one model per row; one figure per variable listed
  - Truth at position (1,1) with column 1 blank below it (show_truth=False
    drops the column), the model field in column 2
  - kind="bias"/"relbias" ADDS that error as column 3
  - The map counterpart of plot_var_skill_scorecard()
  - Colorbar range and colormap defined by get_example_color_scales()

  Loading is split from drawing -- read_skill_map_metrics() reads one directory's
  fields, render_multivar_maps() / render_var_maps() draw pre-read fields, and the
  plot_* functions are the convenience wrappers over both. Every entry point
  returns the path(s) of the PNG(s) written.

  The name of the zarr file is inferred from the choice of diagnostics.
  Available diagnostics are listed in METRIC_INFO; skill_map_indices() returns the
  ones valid for a given variable.

------------------------------------------------------------------------------
Further details of code

The main plotting function is plot_map_matrix_pcolormesh() from
https://github.com/RosieEade/plotting_code

There are also a selection of functions related:

- Data loading
  get_variable_info()
  - This uses a dictionary to provide data information for specific metrics
    including the name of the required Zarr file and related text needed for
    plots.
  - There is a flag <variable_specific> that will raise an error if a metric
    is demanded in relation to a predicted variable that is unconnected.

  reduce_mem_dim()
  - If the prediction contains more than 1 ensemble member, the plotting
    functions have the option to plot a single member or the ensemble mean
    using the arguments
        member_dim="member",
        member_num=0,
    which are passed to reduce_mem_dim() to filter the data correctly.
  - This is necessary for the target and single members too as they are given
    a member dimension of length 1 to maintain consistency across all ML model
    submissions.

- Data alignment and color scales:

  get_units_scale()
  - if the metrics are in the same units as the original prediction data
    (e.g. for mean bias etc), this makes sure the units are standardised
    to a preference for mm/day for pr and hPa for psl as this makes the
    color bar look nicer.
  - it also decides the format of the units e.g. mm/day not mm day-1

  get_example_color_scales()
  - This has some pre-populated choices for colorscale ranges and for colormaps.
  - It is not exhaustive; add your own (metric, variable) entries as needed.

  resolve_styles()
  - Decides the range and colormap of every panel, in this order:
      1. the caller's vminmax / vminmax_bias / cmap / cmap_bias (one value, or
         a {var: value} mapping for a figure spanning several variables),
      2. get_example_color_scales(),
      3. a range derived from the data and SHARED by every panel of that
         variable in the figure (symmetric about zero for the error panels).
  - Rule 3 means an unlisted (metric, variable) still gives comparable panels,
    instead of matplotlib stretching each map to its own min/max.


------------------------------------------------------------------------------
Example calls

import plot_diagnostic_maps as pf

# --------------------
# plot_multivar_event_maps() / plot_var_event_maps()   (paths point at NetCDF files)

event_paths = {
    "RCM":    "/home/target.nc",       # the truth, named by reference= below
    "MODEL1": "/home/prediction1.nc",
    "MODEL2": "/home/prediction2.nc",
}
out_dir = "/home/PLOTS/"
variable_list = ["tas", "hurs", "tasmax", "psl", "tasmin", "uas", "pr", "vas"]

pf.plot_multivar_event_maps(event_paths, out_dir, variable_list, "1983-08-15",
                       reference="RCM", experiment_info="perfect predictors")

pf.plot_var_event_maps(event_paths, out_dir, ["pr"], "1983-08-15",
                  kind="bias", reference="RCM")

# --------------------
# plot_multivar_skill_maps() / plot_var_skill_maps()
#   One variable's stores are read from <path>/<var>, e.g.
#   "/home/historical_perfect/MODEL1/ANN/pr/metrics_fast_field.zarr"

paths = {
    "MODEL1": "/home/historical_perfect/MODEL1/ANN",
    "MODEL2": "/home/historical_perfect/MODEL2/ANN",
}
out_dir = "/home/historical_perfect/PLOTS/"
variable_list = ["tas", "hurs", "tasmax", "psl", "tasmin", "uas", "pr", "vas"]

# one figure: variables across, models down, truth on top
pf.plot_multivar_skill_maps(paths, out_dir, variable_list, metric="mean", season="annual",
                      experiment_info="perfect predictors")

# the same grid, but the model rows showing the bias instead of the field
pf.plot_multivar_skill_maps(paths, out_dir, variable_list, metric="mean", kind="bias")

# one figure per variable: truth at (1,1), models down column 2
pf.plot_var_skill_maps(paths, out_dir, ["pr"], metric="mean", reference="MODEL1")

# ... plus a relative-bias column 3
pf.plot_var_skill_maps(paths, out_dir, ["pr"], metric="mean", kind="relbias")

"""

import os
import numpy as np
import xarray as xr

import matplotlib

matplotlib.use(
    "Agg"
)  # Improves plotting efficiency. Remove if you want to pause code and view plots within a run.
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature

from ml_evaluation.viz.plot_hexagon import (
    metrics_dir,
    normalize_paths,
    select_member,
)


# ----------------------------------------------------------------------------
def plot_map_matrix_pcolormesh(
    lon_list,
    lat_list,
    data_list,
    ncols=1,
    nrows=1,
    vminmax=None,
    filepath="filepath.png",
    title_list=None,
    cbartitle_list=None,
    color_map_list=None,
    FIGWIDTH=None,
    FIGHEIGHT=None,
    panel_size=4.0,
    title_fs=None,
    figtitle=None,
    incl_corner_mean=True,
):
    """
    Plot a matrix of lat/lon gridded fields using pcolormesh on a
    Cartopy PlateCarree projection.

    Each element of the input lists corresponds to one panel in the
    matrix, filled in order from left to right, then top to bottom
    (row-major order), starting at the top-left axis. To skip a
    panel (e.g. so rows can have different numbers of plots), set
    the corresponding entry in `data_list` (and/or `lon_list`/
    `lat_list`) to None - that grid cell's axis will be turned off
    rather than plotted. Lists shorter than nrows*ncols are treated
    as implicitly padded with None (i.e. trailing panels are skipped).

    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade

    Parameters
    ----------
    lon_list : list of array-like or None
        Longitude arrays, one per panel (same length as data_list).
        Use None for a panel to skip.
    lat_list : list of array-like or None
        Latitude arrays, one per panel (same length as data_list).
        Use None for a panel to skip.
    data_list : list of array-like or None
        2D data arrays to plot, one per panel. Use None (or omit
        by leaving the list shorter than nrows*ncols) to skip a
        panel and turn off its axis.
    ncols : int, optional
        Number of columns in the subplot grid (default 1).
    nrows : int, optional
        Number of rows in the subplot grid (default 1).
    vminmax : list of (float, float) or None, optional
        Per-panel (vmin, vmax) colour scale limits. Must have at
        least len(data_list) entries, otherwise it is ignored and
        matplotlib's automatic scaling is used for every panel.
        Entries for skipped panels are ignored.
    filepath : str, optional
        Path (including filename) the figure is saved to
        (default "filepath.png").
    title_list : list of str or None, optional
        Per-panel axis titles. Must have at least len(data_list)
        entries, otherwise ignored. Entries for skipped panels are
        ignored.
    cbartitle_list : list of str or None, optional
        Per-panel colourbar labels. Must have at least
        len(data_list) entries, otherwise ignored. Entries for
        skipped panels are ignored.
    color_map_list : list of matplotlib colormaps or None, optional
        Per-panel colormap. Must have at least len(data_list)
        entries, otherwise every panel defaults to 'RdBu_r'.
    FIGWIDTH : float, optional
        Figure width in inches. If None (default), it is computed
        automatically from ncols, panel_size, and the mean lon/lat
        aspect ratio across all (non-skipped) panels.
    FIGHEIGHT : float, optional
        Figure height in inches. If None (default), it is computed
        automatically from nrows, panel_size, and the mean lon/lat
        aspect ratio across all (non-skipped) panels.
    panel_size : float, optional
        Baseline size in inches used to scale each panel when
        FIGWIDTH/FIGHEIGHT are auto-computed (default 4.0). Larger
        values produce a larger overall figure for the same grid
        shape and aspect ratio.
    title_fs : int or float, optional
        Font size for panel titles (default None, uses matplotlib's
        default).
    figtitle : str, optional
        Overall figure suptitle (default None).
    incl_corner_mean : bool, optional
        If True, annotate each (non-skipped) panel with its data's
        spatial mean (nanmean) in the top-right corner (default True).

    Returns
    -------
    None
        The figure is saved to `filepath` and closed; nothing is
        returned.

    Raises
    ------
    ValueError
        If lon_list, lat_list, and data_list are not all the same
        length, or if that length exceeds ncols*nrows.
    """

    len_data_list = len(data_list)
    if len(lon_list) != len_data_list or len(lat_list) != len_data_list:
        raise ValueError("Input lists must have the same length.")
    if len_data_list > ncols * nrows:
        raise ValueError("Input lists must have length <= ncols*nrows.")

    # A panel is skipped if its data entry (or lon/lat) is None
    def _is_skipped(i):
        if i >= len_data_list:
            return True
        return data_list[i] is None or lon_list[i] is None or lat_list[i] is None

    # Auto-compute a "nice" figure size if not explicitly given, based on
    # the grid shape (nrows x ncols) and the average geographic aspect
    # ratio (lon span / lat span) of the non-skipped panels, so panels
    # aren't stretched or squashed relative to their real-world proportions.
    if FIGWIDTH is None or FIGHEIGHT is None:
        aspect_ratios = []
        for i in range(len_data_list):
            if _is_skipped(i):
                continue
            lon, lat = lon_list[i], lat_list[i]
            lon_span = np.nanmax(lon) - np.nanmin(lon)
            lat_span = np.nanmax(lat) - np.nanmin(lat)
            if lat_span > 0:
                aspect_ratios.append(lon_span / lat_span)
        mean_aspect = np.mean(aspect_ratios) if aspect_ratios else 1.0
        # Clip to avoid pathologically thin/wide panels dominating the layout
        mean_aspect = np.clip(mean_aspect, 0.3, 3.0)

        if FIGWIDTH is None:
            FIGWIDTH = ncols * panel_size * mean_aspect
        if FIGHEIGHT is None:
            # Extra headroom per row for title + horizontal colorbar
            FIGHEIGHT = nrows * panel_size * 1.15

    # Set up figure
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIGWIDTH, FIGHEIGHT),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
        squeeze=False,
    )

    # Only use optional per-panel lists if they cover every panel
    if vminmax and len(vminmax) < len_data_list:
        vminmax = None
    if title_list and len(title_list) < len_data_list:
        title_list = None
    if cbartitle_list and len(cbartitle_list) < len_data_list:
        cbartitle_list = None
    if color_map_list and len(color_map_list) < len_data_list:
        color_map_list = [plt.get_cmap("hot_r")] * len_data_list

    icount = 0
    total_panels = nrows * ncols
    for iiR in range(nrows):
        for iiC in range(ncols):
            ax = axes[iiR, iiC]

            if _is_skipped(icount):
                ax.axis("off")
                icount += 1
                continue

            ivmin, ivmax = None, None
            if vminmax:
                if len(vminmax[icount]) == 2:
                    ivmin, ivmax = vminmax[icount]
                else:
                    ivmin, ivmax = None, None

            color_map = color_map_list[icount]

            im_ax = ax.pcolormesh(
                lon_list[icount],
                lat_list[icount],
                data_list[icount],
                vmin=ivmin,
                vmax=ivmax,
                cmap=color_map,
                transform=ccrs.PlateCarree(),
                shading="auto",
            )

            if title_list:
                ax.set_title(title_list[icount], fontsize=title_fs)

            if incl_corner_mean:
                mean_val = np.nanmean(data_list[icount])
                ax.text(
                    0.98,
                    0.98,
                    f"{mean_val:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            # Geographic features
            ax.coastlines(resolution="50m")  # '10m' for more detail
            ax.add_feature(cartopy.feature.BORDERS, linewidth=0.5)

            cbar = plt.colorbar(im_ax, ax=ax, orientation="horizontal", pad=0.01)
            if cbartitle_list:
                cbar.set_label(cbartitle_list[icount])

            icount += 1

        if icount >= total_panels:
            break

    fig.suptitle(figtitle)
    print(filepath)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    return filepath


# ----------------------------------------------------------------------------
def check_latlon_field(ds, latlon_dims):
    """
    Check that a DataArray/Dataset has exactly the expected lat/lon
    dimensions and nothing else (e.g. no leftover member or time
    dimension), raising an informative error if not.

    Parameters
    ----------
    ds : xarray.DataArray or xarray.Dataset
        Object whose `.dims` should equal `latlon_dims` exactly.
    latlon_dims : list of str
        The two expected dimension names, e.g. ["lat", "lon"]
        (adjust if your coordinates are named e.g. "latitude"/
        "longitude").

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `ds.dims` is not exactly equal to `set(latlon_dims)`.
        The message lists any unexpected extra dimensions (with a
        hint to check `member_dim`/`member_num`) and/or any missing
        expected dimensions.
    """
    expected_dims = set(latlon_dims)
    actual_dims_ds = set(ds.dims)
    if actual_dims_ds != expected_dims:
        extra = actual_dims_ds - expected_dims
        missing = expected_dims - actual_dims_ds
        msg = f"Expected only {expected_dims} dims, got {actual_dims_ds}."
        if extra:
            sizes = {d: ds.sizes[d] for d in extra}
            msg += f" Unexpected extra dims: {extra} (sizes: {sizes})."
            msg += " Check if need to specify a single member with member_dim and member_num)."
        if missing:
            msg += f" Missing expected dims: {missing}."
        raise ValueError(msg)


# ----------------------------------------------------------------------------
def get_units_scale(ds, var, is_zarr=False):
    """
    Work out a multiplicative scale factor to convert a variable to
    a standard "nice for plotting" unit (e.g. mm/day, m/s, hPa),
    based on the variable name and its declared units attribute.

    Handles two dataset layouts:
    - NetCDF-style (`is_zarr=False`, default): the field of interest
      is `ds[var]` and its "units" attribute lives there.
    - Validation-package Zarr diagnostics (`is_zarr=True`): the
      field of interest is held in a generic "value" DataArray
      (`ds['value']`), with `var` only naming what it represents
      (not an actual key in `ds`).

    If units are missing or unrecognised, a sensible default is
    assumed and a warning is printed.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the field to inspect - either directly
        under `var` (`is_zarr=False`) or under `'value'`
        (`is_zarr=True`).
    var : str
        Variable name the data represents. Recognised names: "pr"
        (-> mm/day), "uas"/"vas" (-> m/s), "psl" (-> hPa). Any
        other name is passed through unscaled (scale=1.0).
    is_zarr : bool, optional
        Set True if `ds` is a validation-package Zarr diagnostics
        dataset (field stored under `ds['value']`) rather than a
        regular NetCDF-style dataset (field stored under `ds[var]`)
        (default False).

    Returns
    -------
    scale : float
        Multiplicative factor to apply to the raw data to convert
        it to the standard unit.
    units : str or None
        The units string read from the field's "units" attribute
        (may be None if not present).
    """
    data = ds["value"] if is_zarr else ds[var]

    scale = 1.0
    units = data.attrs.get("units")
    if var == "pr":
        # Convert to mm/day where necessary.
        # Units may differ between datasets and may not be specified.
        if units in [
            "kg m-2 s-1",
            "kg/m2/s",
            "kg m^-2 s^-1",
            "kg.m-2.s-1",
            "kg m**-2 s**-1",
        ]:
            scale = 86400.0
        elif units in ["mm/day", "mm day-1", "mm d-1", "mm.day-1"]:
            scale = 1.0
        elif data.max() > 1:
            scale = 1.0
            print(
                f"WARNING: {var} has units = {units}, i.e. unknown or missing: assume mm/day"
            )
        else:
            scale = 86400.0
            print(
                f"WARNING: {var} has units = {units}, i.e. unknown or missing: assume kg/m2/s so convert to mm/day"
            )
        units = "mm/day"

    if var == "uas" or var == "vas":
        units = "m/s"
        if units not in ["m/s", "m s-1", "m.s-1"]:
            print(
                f"WARNING: {var} has units = {units}, i.e. unknown or missing: assume m/s"
            )

    if var == "psl":
        # Convert to hPa where necessary
        if units in ["Pa"]:
            scale = 1 / 100.0
        elif units in ["hPa"]:
            scale = 1.0
        elif data.max() < 5000:
            scale = 1.0
            print(
                f"WARNING: {var} has units = {units}, i.e. unknown or missing: assume hPa"
            )
        else:
            scale = 1 / 100.0
            print(
                f"WARNING: {var} has units = {units}, i.e. unknown or missing: assume Pa so convert to hPa"
            )
        units = "hPa"

    return scale, units


# ----------------------------------------------------------------------------
def get_example_color_scales(variable_name, metric_name, units=None, period=None):
    """
    Look up example colour bar scales and colormaps for a given
    variable/metric combination.

    This only contains a handful of pre-populated examples of how
    scales may be stored - add entries and adjust ranges for your
    own plot cases.

    Parameters
    ----------
    variable_name : str
        Name of the variable being plotted, e.g. "pr", "tas",
        "psl". Used with `metric_name` to look up a colour range.
    metric_name : str
        Which diagnostic the scale is for, e.g. "mean", "acorr",
        "mv_pcorr", "mv_scorr". Used with `variable_name` to look
        up a colour range.
    units : str or None, optional
        Not currently used. Reserved for a possible future need to
        vary the scale by units, e.g. hPa vs Pa (default None).
    period : str or None, optional
        Not currently used. Reserved for a possible future need to
        have different scales for e.g. historical vs future periods
        (default None).

    Returns
    -------
    vminmax : list of (float or None)
        [vmin, vmax] colour scale limits for the main (truth/
        prediction) field. [None, None] if no specific range is
        defined for this `variable_name`/`metric_name` combination.
    cmap_name : str
        Name of the matplotlib colormap for the main field, e.g.
        'hot_r', 'PiYG_r' (default 'hot_r').
    vminmax_bias : list of (float or None)
        [vmin, vmax] colour scale limits for the bias field.
        [None, None] if no specific range is defined.
    cmap_name_bias : str
        Name of the matplotlib colormap for the bias field
        (default 'hot_r').

    Notes
    -----
    Assumes standard units: precipitation in mm/day, temperature in
    K, winds in m/s, pressure in hPa, relative humidity in %. The
    set of populated (metric_name, variable_name) combinations is
    illustrative only, not exhaustive.
    """

    manually_defined = False
    vminmax = [None, None]
    vminmax_bias = [None, None]
    cmap_name = "hot_r"
    # A bias/relbias field straddles zero, so its default colormap is diverging
    # with white in the middle -- brown-blue-green for precipitation (the CORDEX
    # convention: brown = too dry), blue-red for everything else. Paired with the
    # symmetric range resolve_styles derives, white sits exactly on zero.
    cmap_name_bias = "BrBG" if variable_name == "pr" else "RdBu_r"

    if metric_name in ["mean"]:
        cmap_name = "hot_r"
        if variable_name in ["pr"]:
            vminmax = [0, 40]
            manually_defined = True
        if variable_name in ["tas", "tasmax", "tasmin"]:
            vminmax = [290, 305]
            manually_defined = True
        if variable_name in ["uas", "vas"]:
            vminmax = [-2, 2]
            manually_defined = True
        if variable_name in ["psl"]:
            vminmax = [1008, 1012]
            manually_defined = True
        if variable_name in ["hurs"]:
            vminmax = [70, 90]
            manually_defined = True

    if metric_name in ["acorr"]:
        if variable_name in ["pr"]:
            vminmax = [0, 0.6]
            vminmax_bias = [-0.3, 0.3]
            manually_defined = True
            cmap_name = "hot_r"
            cmap_name_bias = "PiYG_r"

    if metric_name in ["mv_pcorr", "mv_scorr"]:
        if variable_name in ["pr"]:
            vminmax = [-0.9, 0.9]
            vminmax_bias = [-0.06, 0.06]
            manually_defined = True
            cmap_name = "PiYG_r"
            cmap_name_bias = "PiYG_r"

    if not manually_defined:
        print(
            "WARNING: color scales not set by get_example_color_scales() so may not match."
        )

    return vminmax, cmap_name, vminmax_bias, cmap_name_bias


# ----------------------------------------------------------------------------
# Plotting metadata per diagnostic metric, keyed by metric name:
#   (title_name, units, VariableUnits, zarr_name, var_specific)
# Module level so the catalogue can be introspected -- see `skill_map_indices`.
METRIC_INFO = {
    # fast (basic) track metrics
    "mean": ("Climatology mean", "", True, "metrics_fast_field.zarr", None),
    "p99": ("99% Quantile", "", True, "metrics_fast_field.zarr", None),
    "p1": ("1% Quantile", "", True, "metrics_fast_field.zarr", None),
    "max1day": ("Mean of max1day", "", True, "metrics_fast_field.zarr", None),
    "min1day": ("Mean of min1day", "", True, "metrics_fast_field.zarr", None),
    "acorr": (
        "Auto-correlation",
        "correlation",
        False,
        "metrics_fast_field.zarr",
        None,
    ),
    # pr: fast (basic) track metrics
    "p98wet": (
        "98% Quantile (wet days)",
        "",
        True,
        "metrics_fast_pr_field.zarr",
        "pr",
    ),
    "sdii": (
        "Simple Daily Intensity Index",
        "",
        True,
        "metrics_fast_pr_field.zarr",
        "pr",
    ),
    "r1": ("Mean of r1", "no. days", False, "metrics_fast_pr_field.zarr", "pr"),
    # slow (extended) track metrics
    "annstd": (
        "Annual standard deviation",
        "",
        True,
        "metrics_slow_field.zarr",
        None,
    ),
    "p90": ("90% Quantile", "", True, "metrics_slow_field.zarr", None),
    # pr: slow (extended) track metrics
    "p90wet": (
        "90% Quantile (wet days)",
        "",
        True,
        "metrics_slow_pr_field.zarr",
        "pr",
    ),
    "r10": ("Mean of r10", "no. days", False, "metrics_slow_pr_field.zarr", "pr"),
    "cdd": ("Mean of cdd", "no. days", False, "metrics_slow_pr_field.zarr", "pr"),
    "cwd": ("Mean of cwd", "no. days", False, "metrics_slow_pr_field.zarr", "pr"),
    # tasmax: slow (extended) track metrics
    "su": (
        "Mean no. Summer days",
        "no. days",
        False,
        "metrics_slow_tasmax_field.zarr",
        "tasmax",
    ),
    "hot": (
        "Mean no. Hot days",
        "no. days",
        False,
        "metrics_slow_tasmax_field.zarr",
        "tasmax",
    ),
    "ice": (
        "Mean no. Ice days",
        "no. days",
        False,
        "metrics_slow_tasmax_field.zarr",
        "tasmax",
    ),
    # tasmin: slow (extended) track metrics
    "tropnight": (
        "Mean no. Tropical nights",
        "no. days",
        False,
        "metrics_slow_tasmin_field.zarr",
        "tasmin",
    ),
    "frost": (
        "Mean no. Frost days",
        "no. days",
        False,
        "metrics_slow_tasmin_field.zarr",
        "tasmin",
    ),
    # multivariate metrics
    "mv_pcorr": (
        "MV Pearson correlation",
        "correlation",
        False,
        "metrics_mv_field.zarr",
        None,
    ),
    "mv_scorr": (
        "MV Spearman correlation",
        "correlation",
        False,
        "metrics_mv_field.zarr",
        None,
    ),
}


# ----------------------------------------------------------------------------
def get_variable_info(metric_name):
    """
    Look up default plotting metadata for a given diagnostic metric.

    Provides the title text, units, and the name of the Zarr file
    the metric is stored in, along with flags describing how the
    metric relates to variables.

    Parameters
    ----------
    metric_name : str
        Name of the diagnostic metric, e.g. "mean", "p99", "acorr",
        "p98wet", "mv_pcorr". See the `METRIC_INFO` dict in this
        function for the full list of recognised names.

    Returns
    -------
    title_name : str
        Default title text describing the metric, e.g.
        "Climatology mean".
    units : str
        Default units string for the metric. Empty string if the
        units instead depend on the underlying variable (see
        `VariableUnits`).
    VariableUnits : bool
        True if the metric's units depend on the prediction
        variable, in which case `units` should be filled in via
        `get_units_scale` instead. False if `units` is definitive.
    zarr_name : str
        Filename of the Zarr diagnostics file this metric is stored
        in, e.g. "metrics_fast_field.zarr".
    var_specific : str or None
        If not None, the single variable name this metric applies
        to (e.g. "pr" for "p98wet"); the metric should not be used
        with any other variable. None if the metric applies to any
        variable.

    Raises
    ------
    ValueError
        If `metric_name` is not a recognised key in `METRIC_INFO`.
    """

    try:
        title_name, units, VariableUnits, zarr_name, var_specific = METRIC_INFO[
            metric_name
        ]
    except KeyError:
        raise ValueError(f"Unknown metric_name: {metric_name!r}")

    return title_name, units, VariableUnits, zarr_name, var_specific


# ----------------------------------------------------------------------------
def reduce_mem_dim(ds, member_dim, member_num):
    """
    Reduce an ensemble-member dimension down to a single member (or
    the ensemble mean), so the resulting field is ready for 2D
    lat/lon plotting.

    Parameters
    ----------
    ds : xarray.DataArray or xarray.Dataset
        Field that may contain a member dimension named
        `member_dim`.
    member_dim : str or None
        Name of the ensemble-member dimension to look for. If None,
        or if `member_dim` is not present in `ds.dims`, a warning is
        printed and `ds` is returned unchanged (see Returns).
    member_num : int
        Which member to select:
        - If < 0: average over the member dimension (ensemble mean).
        - If it matches a value in `ds[member_dim]`: select that
          member.
        - Otherwise: fall back to the first member (index 0) and
          print a warning.

    Returns
    -------
    ds : xarray.DataArray or xarray.Dataset
        The field with the member dimension removed (via selection
        or averaging), or unchanged if `member_dim` was None or not
        present in `ds.dims`.
    member_num : int
        The member number actually used: unchanged if averaging
        (negative) or if the requested member existed; 0 if it had
        to fall back to the first member of an existing
        `member_dim`; unchanged (echoing back the input) if
        `member_dim` itself was None or absent, since `ds` is left
        untouched in that case.
    """
    reduced = select_member(ds, member_dim, member_num)
    if member_dim is None or member_dim not in ds.dims:
        return ds, member_num
    if member_num >= 0 and member_num not in np.asarray(ds[member_dim].values):
        return reduced, 0
    return reduced, member_num


# ============================================================================
# Aligned API: `{label: path}` locating, read_* / render_* split
# ============================================================================
# The entry points below follow the same conventions as the rest of
# ``ml_evaluation.viz`` (plot_hexagon, plot_var_skill_scorecard, plot_var_series):
#
#   * sources are an explicit ``{label: path}`` mapping -- ``path`` is one run's
#     base directory and one variable's stores live in ``<path>/<var>/``, exactly
#     as ``metrics_dir`` resolves it. Nothing is searched and no directory layout
#     is assumed;
#   * loading is split from drawing: ``read_skill_map_metrics`` depends only on
#     ml-eval's output format, ``render_*`` take pre-read fields, and ``plot_*``
#     are the convenience wrappers over both;
#   * ``season`` is a label for the title and the file name only;
#   * each entry point returns the path(s) of the PNG(s) written.


def skill_map_indices(var=None):
    """Metric keys that can be mapped, optionally filtered by variable.

    Mirrors ``plot_var_series.series_indices`` / ``plot_var_skill_scorecard.
    var_skill_columns``: the catalogue of what this module can plot, so callers
    can discover it instead of guessing.

    Parameters
    ----------
    var : str, optional
        Variable name. When given, variable-specific metrics belonging to other
        variables are dropped (e.g. ``p98wet`` only survives for ``pr``).

    Returns
    -------
    list of str
        Metric keys, in catalogue order.
    """
    keys = []
    for key, (_title, _units, _var_units, _zarr, var_specific) in METRIC_INFO.items():
        if var is None or var_specific is None or var_specific == var:
            keys.append(key)
    return keys


def _member_tag(member_num):
    """Filename suffix identifying the member plotted (empty for member 0)."""
    if member_num is None or member_num == 0:
        return ""
    return "_em" if member_num < 0 else f"_m{int(member_num)}"


def _season_suffix(season):
    """Filename/title fragment for a season label (empty when season is None)."""
    return "" if season is None else f"_{season}"


# Dimensions that are never horizontal: anything left over after selecting a
# diagnostic and a member means the field was not reduced to a map.
NON_SPATIAL_DIMS = frozenset(
    {"time", "member", "diagnostic", "year", "month", "wavenumber", "bins", "quantile"}
)


def horizontal_dims(da, latlon_dims=None):
    """The two horizontal dimension names of a reduced field.

    ``latlon_dims=None`` (the default) reads them off the data, so a regular
    grid (``lat``/``lon``) and a curvilinear one (``y``/``x`` with 2-D lat/lon
    coordinates, as CORDEX rotated/Lambert grids are) both work without the
    caller having to say which. Pass an explicit pair to override.

    Raises
    ------
    ValueError
        If the field does not reduce to exactly two horizontal dimensions.
    """
    if latlon_dims is not None:
        return tuple(latlon_dims)
    dims = tuple(da.dims)
    leftover = [d for d in dims if d in NON_SPATIAL_DIMS]
    if leftover or len(dims) != 2:
        raise ValueError(
            f"expected a 2-D field, got dims {dims}"
            + (f" including {leftover}" if leftover else "")
            + ". Check member_dim / member_num, or name the horizontal dims "
            "explicitly with latlon_dims."
        )
    return dims


def plot_coords(ds, da, dims):
    """The (lat, lon) arrays to draw ``da`` with.

    Prefers the ``lat``/``lon`` coordinates when the file carries them -- on a
    curvilinear grid those are the 2-D fields pcolormesh needs, while the dims
    are ``y``/``x`` -- and falls back to the dimension coordinates otherwise.
    """
    if "lat" in ds.coords and "lon" in ds.coords:
        return ds["lat"].values, ds["lon"].values
    return da[dims[0]].values, da[dims[1]].values


# What a model panel can show: the index itself, its error against the truth, or
# that error as a percentage of the truth.
PANEL_KINDS = ("raw", "bias", "relbias")


def read_skill_map_metrics(
    metrics_dir,
    var,
    metric="mean",
    *,
    kinds=("targ", "raw", "bias"),
    member_dim="member",
    member_num=0,
    latlon_dims=None,
):
    """Map fields of one diagnostic for one variable, from one metrics directory.

    ``metrics_dir`` holds the ``metrics_*.zarr`` stores written by a single
    ``ml-eval`` call; the store a metric lives in is looked up in
    :data:`METRIC_INFO`. Depends only on ml-eval's output format -- no
    directory-layout assumptions, so it is reusable from any project's own
    plotting code (as ``read_var_skill_metrics`` and ``read_series`` are).

    The truth field is always read at member 0; ``member_num`` applies to the
    model fields only, matching how the target is stored with a length-1 member
    dimension for consistency across submissions.

    Parameters
    ----------
    metrics_dir : str
        Directory holding the zarr stores (e.g. ``<path>/<var>``).
    var : str
        Variable the metrics describe -- used for the units, the colour scales
        and the variable-specific check.
    metric : str
        Metric key; see :func:`skill_map_indices` for the catalogue.
    kinds : tuple of str
        Which fields to read: ``"targ"`` (the truth) and any of
        :data:`PANEL_KINDS`. ``"relbias"`` is derived as
        ``100 * (raw - targ) / targ``, masked where the truth is ~0, so it is
        available for every metric rather than only the few ml-eval stores it
        for.
    member_dim : str or None
        Name of the ensemble dimension, or None if there is none.
    member_num : int
        Member to plot: ``>= 0`` selects that member, ``< 0`` the mean over
        members.
    latlon_dims : tuple of str, optional
        The two horizontal dimension names. None (default) reads them off the
        data, so ``(lat, lon)`` and a curvilinear ``(y, x)`` both work.

    Returns
    -------
    dict or None
        ``None`` when the store is missing. Otherwise a dict holding the scaled
        2-D fields under ``"fields"`` (``{kind: ndarray}``, a kind absent from
        the store maps to None), the ``"lat"``/``"lon"`` arrays, and the display
        metadata ``"units"``, ``"title"``, ``"vminmax"``, ``"cmap"``,
        ``"vminmax_bias"``, ``"cmap_bias"``.

    Raises
    ------
    ValueError
        If ``metric`` is unknown, if it belongs to a different variable, or if a
        field does not reduce to the expected lat/lon dims.
    """
    title_name, units, variable_units, zarr_name, var_specific = get_variable_info(
        metric
    )
    if var_specific is not None and var != var_specific:
        raise ValueError(
            f"Metric {metric} only suitable for variable {var_specific}, "
            f"but variable loaded is {var}"
        )
    unknown = [k for k in kinds if k not in ("targ",) + PANEL_KINDS]
    if unknown:
        raise ValueError(
            f"unknown kinds {unknown}; expected 'targ' or one of {PANEL_KINDS}."
        )

    store = os.path.join(metrics_dir, zarr_name)
    if not os.path.isdir(store):
        return None
    ds = xr.open_zarr(store)

    scale = 1
    if variable_units:
        scale, units = get_units_scale(ds, var, is_zarr=True)

    vminmax, cmap_name, vminmax_bias, cmap_name_bias = get_example_color_scales(
        var, metric
    )

    # relbias needs the truth and the prediction, whether or not they are drawn.
    wanted = set(kinds)
    if "relbias" in wanted:
        wanted |= {"targ", "raw"}

    # kind -> the diagnostic ml-eval stores it under ("relbias" is derived below)
    diag_of = {"targ": "targ", "raw": "pred", "bias": "bias"}

    fields = {}
    dims, ref_da = None, None
    for kind in ("targ", "raw", "bias"):
        if kind not in wanted:
            continue
        try:
            da = ds["value"].sel(diagnostic=f"{diag_of[kind]}_{metric}_field").load()
        except KeyError:
            fields[kind] = None
            continue
        # The target carries a length-1 member dim for consistency: always m0.
        da = select_member(da, member_dim, 0 if kind == "targ" else member_num)
        if dims is None:
            dims, ref_da = horizontal_dims(da, latlon_dims), da
        check_latlon_field(da, list(dims))
        fields[kind] = da.values * scale
    if dims is None:
        return None

    if "relbias" in kinds:
        targ, raw = fields.get("targ"), fields.get("raw")
        if targ is None or raw is None:
            fields["relbias"] = None
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = 100.0 * (raw - targ) / targ
            fields["relbias"] = np.where(np.abs(targ) < 1e-12, np.nan, rel)

    # drop the helpers the caller did not ask to draw
    fields = {k: v for k, v in fields.items() if k in kinds}
    lat, lon = plot_coords(ds, ref_da, dims)

    return {
        "fields": fields,
        "lat": lat,
        "lon": lon,
        "units": units,
        "title": title_name,
        "vminmax": vminmax,
        "cmap": cmap_name,
        "vminmax_bias": vminmax_bias,
        "cmap_bias": cmap_name_bias,
    }


def _is_set(vminmax):
    """True when a catalogue range actually pins both ends."""
    return bool(vminmax) and vminmax[0] is not None and vminmax[1] is not None


def _shared_range(fields, symmetric=False):
    """A single range covering every panel of one variable in the figure.

    Used only when neither the caller nor :func:`get_example_color_scales`
    pinned one, so panels of the same variable stay comparable instead of each
    being stretched to its own min/max. ``symmetric`` centres the range on zero,
    which is what an error field needs.
    """
    vals = [np.asarray(f, dtype=float).ravel() for f in fields if f is not None]
    vals = np.concatenate(vals) if vals else np.array([])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [None, None]
    if symmetric:
        m = float(np.max(np.abs(vals)))
        return [-m, m] if m > 0 else [None, None]
    return [float(vals.min()), float(vals.max())]


def _pick(override, var):
    """One variable's slice of an override that may be per-variable."""
    if isinstance(override, dict):
        return override.get(var)
    return override


def resolve_styles(
    bundles,
    var,
    kind,
    *,
    vminmax=None,
    vminmax_bias=None,
    cmap=None,
    cmap_bias=None,
    raw_symmetric=False,
):
    """Colour scales for one variable's panels, by explicit precedence.

    For each of the two panel families -- the field itself and, when ``kind`` is
    not ``"raw"``, the error -- the range is taken from the first of:

    1. the caller's ``vminmax`` / ``vminmax_bias`` (a ``[vmin, vmax]`` pair, or a
       ``{var: pair}`` mapping when a figure spans several variables). The string
       ``"auto"`` means "skip the catalogue and go straight to step 3" -- useful
       when a catalogue entry is pinned for a different domain than yours;
    2. :func:`get_example_color_scales`, whichever entries it pins;
    3. a range derived from the plotted data, shared by every panel of this
       variable (symmetric about zero for the error panels).

    Rule 3 is what keeps two models comparable when the catalogue has no entry:
    without it matplotlib autoscales each panel separately and equal-looking
    maps can differ by a factor.

    Parameters
    ----------
    bundles : iterable of dict
        The :func:`read_skill_map_metrics` bundles drawn for this variable (the truth
        and every model).
    var : str
        Variable being styled -- selects the per-variable slice of an override.
    kind : {"raw", "bias", "relbias"}
        Which error family, if any, the model panels show.
    vminmax, vminmax_bias : list, "auto", or mapping, optional
        Explicit ranges for the field and the error panels; ``"auto"`` forces
        the data-derived range even when the catalogue has an entry.
    cmap, cmap_bias : str or mapping, optional
        Explicit colormaps for the same two families.
    raw_symmetric : bool
        Centre a derived field range on zero. The skill maps plot a field
        (False); the signal maps plot a *change*, which straddles zero exactly
        as an error does (True).

    Returns
    -------
    dict
        ``{"raw": (cmap, vminmax, units), <kind>: (cmap, vminmax, units)}``,
        ready for :func:`_panel`. ``"targ"`` is an alias of ``"raw"``.
    """
    bundles = [b for b in bundles if b is not None]
    if not bundles:
        return {}
    first = bundles[0]

    # --- the field itself (truth and, unless kind replaces them, the models) --
    rng = _pick(vminmax, var)
    if rng == "auto":
        rng = None  # skip the catalogue, fall through to the data
    elif not _is_set(rng):
        rng = first["vminmax"]
    if not _is_set(rng):
        rng = _shared_range(
            [b["fields"].get(k) for b in bundles for k in ("targ", "raw")],
            symmetric=raw_symmetric,
        )
    raw_style = (
        plt.get_cmap(_pick(cmap, var) or first["cmap"]),
        rng,
        first["units"],
    )
    styles = {"raw": raw_style, "targ": raw_style}

    # --- the error panels ----------------------------------------------------
    if kind != "raw":
        rng = _pick(vminmax_bias, var)
        if rng == "auto":
            rng = None  # skip the catalogue, fall through to the data
        elif not _is_set(rng):
            # a percentage has nothing to do with the stored (unit-ful) range
            rng = [None, None] if kind == "relbias" else first["vminmax_bias"]
        if not _is_set(rng):
            rng = _shared_range(
                [b["fields"].get(kind) for b in bundles], symmetric=True
            )
        styles[kind] = (
            plt.get_cmap(_pick(cmap_bias, var) or first["cmap_bias"]),
            rng,
            "%" if kind == "relbias" else first["units"],
        )
    return styles


def _panel(info, kind, title, styles, cbar=None):
    """One panel tuple for :func:`_draw`, or a blank when the field is missing."""
    if info is None or info["fields"].get(kind) is None or kind not in styles:
        return BLANK
    cmap, vminmax, units = styles[kind]
    return (
        info["fields"][kind],
        info["lat"],
        info["lon"],
        title,
        units if cbar else None,
        cmap,
        vminmax,
    )


BLANK = (None, None, None, None, None, None, [None, None])


def _draw(panels, out_dir, filename, figtitle, ncols, nrows, panel_size=2.6):
    """Send an assembled panel list to the matrix renderer.

    ``panels`` is a list of ``(field, lat, lon, title, cbar_title, cmap,
    vminmax)`` tuples in row-major order; a panel whose field is None is drawn
    as an empty cell. ``nrows`` is a minimum -- it grows if the panels need more
    rows, so a caller's default can never silently drop a model or a variable.
    ``panel_size`` is the height of one map in inches: every panel carries its
    own colourbar, so too small a value makes the next row's title collide with
    it.
    """
    needed = -(-len(panels) // ncols)  # ceil
    nrows = max(nrows or 0, needed)

    os.makedirs(out_dir, exist_ok=True)
    return plot_map_matrix_pcolormesh(
        [p[2] for p in panels],
        [p[1] for p in panels],
        [p[0] for p in panels],
        ncols=ncols,
        nrows=nrows,
        vminmax=[p[6] for p in panels],
        filepath=os.path.join(out_dir, filename + ".png"),
        title_list=[p[3] for p in panels],
        cbartitle_list=[p[4] for p in panels],
        color_map_list=[p[5] for p in panels],
        FIGWIDTH=None,
        FIGHEIGHT=None,
        panel_size=panel_size,
        title_fs=None,
        figtitle=figtitle,
    )


def _figure_title(title, name, metric_title, kind, season, experiment_info):
    """Default figure title: '<metric> [bias] <season> - <name> <experiment>'."""
    if title is not None:
        return title
    what = metric_title if kind == "raw" else f"{metric_title} {kind}"
    if season is not None:
        what = f"{what} {season}"
    title = f"{what} - {name}"
    if experiment_info is not None:
        title = f"{title} {experiment_info}"
    return title


def render_multivar_maps(
    per_model,
    truth,
    out_dir,
    *,
    variables,
    models=None,
    metric="mean",
    kind="raw",
    season="annual",
    member_num=0,
    experiment_info=None,
    title=None,
    filename=None,
    ncols=None,
    nrows=None,
    panel_size=2.6,
    vminmax=None,
    vminmax_bias=None,
    cmap=None,
    cmap_bias=None,
    raw_symmetric=False,
):
    """Render one index across variables (columns) and models (rows).

    The grid is ``(1 + N models) x len(variables)``: the truth occupies the top
    row when ``truth`` is given, and each model row shows ``kind`` -- the index
    itself (``"raw"``), its bias, or its relative bias.

    Parameters
    ----------
    per_model : mapping
        ``{label: {var: read_skill_map_metrics(...) dict}}``.
    truth : mapping or None
        ``{var: read_skill_map_metrics(...) dict}`` for the top row; None omits it.
    out_dir : str
        Output directory (created if missing).
    variables : iterable of str
        Column order.
    models : iterable of str, optional
        Row order; defaults to the order of ``per_model``.
    metric : str
        Metric key -- title and file name only, the fields were already selected.
    kind : {"raw", "bias", "relbias"}
        What the model rows show.
    season, member_num, experiment_info, title, filename
        Season label, member label, extra title text, full-title override and
        file-name override (without the ``.png``).
    ncols, nrows
        Panel grid; defaults to one column per variable and one row per model
        (plus the truth row). ``nrows`` is a minimum and grows to fit.
    panel_size : float
        Height of one map in inches. Each panel carries its own colourbar, so
        raise it if a row's titles crowd the bar above them.
    vminmax, vminmax_bias, cmap, cmap_bias
        Colour-scale overrides, each either one value for the whole figure or a
        ``{var: value}`` mapping. They win over
        :func:`get_example_color_scales`, which in turn wins over the range
        derived from the data; see :func:`resolve_styles`. Since the columns are
        different variables, each column keeps its own scale and its own
        colourbar either way.

    Returns
    -------
    str
        Path of the PNG written.
    """
    if kind not in PANEL_KINDS:
        raise ValueError(f"kind must be one of {PANEL_KINDS}, got {kind!r}.")
    variables = list(variables)
    models = [
        m for m in (models if models is not None else per_model) if m in per_model
    ]
    if not variables or not models:
        raise ValueError("nothing to plot: need at least one variable and one model.")

    any_info = next(
        (
            per_model[m][v]
            for m in models
            for v in variables
            if per_model[m].get(v) is not None
        ),
        None,
    )
    metric_title = (truth or {}).get(variables[0], any_info or {}).get("title", metric)

    # one style per variable (i.e. per column), shared by every row
    styles = {
        var: resolve_styles(
            [(truth or {}).get(var)] + [per_model[m].get(var) for m in models],
            var,
            kind,
            vminmax=vminmax,
            vminmax_bias=vminmax_bias,
            cmap=cmap,
            cmap_bias=cmap_bias,
            raw_symmetric=raw_symmetric,
        )
        for var in variables
    }

    panels = []
    if truth is not None:
        for var in variables:
            panels.append(_panel(truth.get(var), "targ", f"Truth: {var}", styles[var]))
    for i, model in enumerate(models):
        # only the bottom row's colourbar is labelled: with a bar under every
        # panel, a label anywhere else collides with the title below it
        last = i == len(models) - 1
        for var in variables:
            info = per_model[model].get(var)
            label = model if kind == "raw" else f"{kind}: {model}"
            panels.append(_panel(info, kind, f"{label}: {var}", styles[var], cbar=last))

    name = f"{len(models)} models" if len(models) > 1 else models[0]
    figtitle = _figure_title(title, name, metric_title, kind, season, experiment_info)
    if filename is None:
        filename = (
            f"multivar_maps_{metric}_{kind}"
            f"{_member_tag(member_num)}{_season_suffix(season)}"
        )
    return _draw(
        panels, out_dir, filename, figtitle, ncols or len(variables), nrows, panel_size
    )


def render_var_maps(
    per_model,
    truth,
    var,
    out_dir,
    *,
    models=None,
    metric="mean",
    kind="raw",
    season="annual",
    member_num=0,
    experiment_info=None,
    title=None,
    filename=None,
    ncols=None,
    nrows=None,
    panel_size=2.6,
    vminmax=None,
    vminmax_bias=None,
    cmap=None,
    cmap_bias=None,
    raw_symmetric=False,
):
    """Render one index for one variable, one model per row.

    Column 1 holds the truth at ``(1, 1)`` (blank below it) when ``truth`` is
    given, column 2 the model's field, and -- when ``kind`` is ``"bias"`` or
    ``"relbias"`` -- column 3 that model's error. With ``kind="raw"`` there is
    no third column.

    Parameters
    ----------
    per_model : mapping
        ``{label: read_skill_map_metrics(...) dict}``.
    truth : dict or None
        The bundle supplying the ``targ`` field for ``(1, 1)``; None omits the
        truth column entirely.
    var : str
        Variable being plotted (one figure).
    out_dir : str
        Output directory (created if missing).
    models : iterable of str, optional
        Row order; defaults to the order of ``per_model``.
    metric : str
        Metric key -- title and file name only.
    kind : {"raw", "bias", "relbias"}
        ``"raw"`` plots the index alone; the others add the error column.
    season, member_num, experiment_info, title, filename, ncols, nrows
        As in :func:`render_multivar_maps`.
    vminmax, vminmax_bias, cmap, cmap_bias
        Colour-scale overrides; see :func:`resolve_styles`. Columns 1-2 (the
        truth and the models) always share one scale so they are comparable;
        column 3 (the error) carries its own.

    Returns
    -------
    str
        Path of the PNG written.
    """
    if kind not in PANEL_KINDS:
        raise ValueError(f"kind must be one of {PANEL_KINDS}, got {kind!r}.")
    models = [
        m for m in (models if models is not None else per_model) if m in per_model
    ]
    if not models:
        raise ValueError("per_model holds no models to plot.")

    metric_title = (truth or per_model[models[0]] or {}).get("title", metric)

    styles = resolve_styles(
        [truth] + [per_model[m] for m in models],
        var,
        kind,
        vminmax=vminmax,
        vminmax_bias=vminmax_bias,
        cmap=cmap,
        cmap_bias=cmap_bias,
        raw_symmetric=raw_symmetric,
    )

    panels = []
    for i, model in enumerate(models):
        # only the bottom row's colourbars are labelled (see render_multivar_maps)
        last = i == len(models) - 1
        if truth is not None:
            panels.append(
                _panel(truth, "targ", "Truth", styles, cbar=(len(models) == 1))
                if i == 0
                else BLANK
            )
        panels.append(_panel(per_model[model], "raw", model, styles, cbar=last))
        if kind != "raw":
            panels.append(
                _panel(per_model[model], kind, f"{kind}: {model}", styles, cbar=last)
            )

    if ncols is None:
        ncols = (1 if truth is not None else 0) + (1 if kind == "raw" else 2)
    figtitle = _figure_title(title, var, metric_title, kind, season, experiment_info)
    if filename is None:
        filename = (
            f"var_maps_{var}_{metric}_{kind}"
            f"{_member_tag(member_num)}{_season_suffix(season)}"
        )
    return _draw(panels, out_dir, filename, figtitle, ncols, nrows, panel_size)


def _read_sources(
    by_label, variables, metric, kinds, member_dim, member_num, latlon_dims
):
    """``{label: {var: bundle}}`` for every source that has metrics on disk."""
    per_model = {}
    for label, base in by_label.items():
        per_var = {}
        for var in variables:
            d = metrics_dir(base, var)
            if not os.path.isdir(d):
                print(f"[maps] skip {label} / {var}: {d} not found")
                continue
            info = read_skill_map_metrics(
                d,
                var,
                metric,
                kinds=kinds,
                member_dim=member_dim,
                member_num=member_num,
                latlon_dims=latlon_dims,
            )
            if info is None:
                print(f"[maps] skip {label} / {var}: no {metric} store in {d}")
                continue
            per_var[var] = info
        if per_var:
            per_model[label] = per_var
    return per_model


def _truth_source(by_label, reference, reference_path):
    """(label, base directory) the truth fields are read from."""
    if reference is not None and reference not in by_label and reference_path is None:
        raise KeyError(
            f"reference model '{reference}' is not one of the labels in paths "
            f"({', '.join(by_label)}); pass reference_path to read the truth "
            "from elsewhere."
        )
    label = reference or next(iter(by_label))
    return label, (reference_path or by_label[label])


def plot_multivar_skill_maps(
    paths,
    out,
    variables,
    *,
    metric="mean",
    kind="raw",
    show_truth=True,
    reference=None,
    reference_path=None,
    season="annual",
    member_dim="member",
    member_num=0,
    latlon_dims=None,
    **kwargs,
):
    """Plot one index across several variables and models, as a single figure.

    Convenience wrapper over :func:`read_skill_map_metrics` + :func:`render_multivar_maps`.
    Columns are the variables, rows are the models, and the truth takes the top
    row -- the map counterpart of ``plot_multivar_skill_scorecard``.

    Parameters
    ----------
    paths : mapping or sequence
        ``{label: path}`` (see ``plot_hexagon.normalize_paths``); ``path`` is the
        model's base directory and one variable's stores are read from
        ``<path>/<var>/``. One entry = one row, in the given order.
    out : str
        Output directory for the PNG.
    variables : iterable of str
        Variables to plot, one column each.
    metric : str
        Metric key; see :func:`skill_map_indices`.
    kind : {"raw", "bias", "relbias"}
        What the model rows show: the index itself, its bias against the truth,
        or that bias as a percentage of the truth.
    show_truth : bool
        Draw the truth as the top row. The truth is stored in every model's own
        bundle, so this costs no extra run.
    reference : str, optional
        Label whose store supplies the truth row; default the first entry.
    reference_path : str, optional
        Base directory to read the truth from, when it is not one of ``paths``.
    season : str
        Label only -- title and output file name.
    member_dim, member_num, latlon_dims
        Passed to :func:`read_skill_map_metrics`.
    **kwargs
        Forwarded to :func:`render_multivar_maps` (``title``, ``filename``,
        ``experiment_info``, ``ncols``, ``nrows``, and the colour-scale
        overrides ``vminmax`` / ``vminmax_bias`` / ``cmap`` / ``cmap_bias`` --
        each one value or a ``{var: value}`` mapping).

    Returns
    -------
    str or None
        Path of the PNG written, or None when no source had data.
    """
    by_label = normalize_paths(paths)
    variables = list(variables)
    per_model = _read_sources(
        by_label,
        variables,
        metric,
        ("raw",) if kind == "raw" else ("targ", "raw", kind),
        member_dim,
        member_num,
        latlon_dims,
    )
    if not per_model:
        print(f"[maps] {metric}: no metrics found -- nothing plotted.")
        return None

    truth = None
    if show_truth:
        _, truth_base = _truth_source(by_label, reference, reference_path)
        truth = {}
        for var in variables:
            d = metrics_dir(truth_base, var)
            if os.path.isdir(d):
                truth[var] = read_skill_map_metrics(
                    d,
                    var,
                    metric,
                    kinds=("targ",),
                    member_dim=member_dim,
                    latlon_dims=latlon_dims,
                )

    return render_multivar_maps(
        per_model,
        truth,
        out,
        variables=variables,
        models=[m for m in by_label if m in per_model],
        metric=metric,
        kind=kind,
        season=season,
        member_num=member_num,
        **kwargs,
    )


def plot_var_skill_maps(
    paths,
    out,
    variables,
    *,
    metric="mean",
    kind="raw",
    show_truth=True,
    reference=None,
    reference_path=None,
    season="annual",
    member_dim="member",
    member_num=0,
    latlon_dims=None,
    **kwargs,
):
    """Plot one index for one variable at a time, one model per row.

    Convenience wrapper over :func:`read_skill_map_metrics` + :func:`render_var_maps`.
    The truth sits at ``(1, 1)``, the models fill column 2, and their error --
    when ``kind`` is ``"bias"`` or ``"relbias"`` -- column 3. The map
    counterpart of ``plot_var_skill_scorecard``.

    Parameters
    ----------
    paths : mapping or sequence
        ``{label: path}`` as in :func:`plot_multivar_skill_maps`. One entry = one row.
    out : str
        Output directory for the PNGs.
    variables : iterable of str
        Variables to plot (one figure each).
    metric, kind, show_truth, reference, reference_path, season
        As in :func:`plot_multivar_skill_maps`.
    member_dim, member_num, latlon_dims
        Passed to :func:`read_skill_map_metrics`. The truth is always read at member 0.
    **kwargs
        Forwarded to :func:`render_var_maps` (``title``, ``filename``,
        ``experiment_info``, ``ncols``, ``nrows``, and the colour-scale
        overrides ``vminmax`` / ``vminmax_bias`` / ``cmap`` / ``cmap_bias``).

    Returns
    -------
    list of str
        Paths of the PNGs written (one per variable that had data).
    """
    by_label = normalize_paths(paths)
    variables = list(variables)
    per_model = _read_sources(
        by_label,
        variables,
        metric,
        ("raw",) if kind == "raw" else ("targ", "raw", kind),
        member_dim,
        member_num,
        latlon_dims,
    )
    if not per_model:
        print(f"[maps] {metric}: no metrics found -- nothing plotted.")
        return []

    truth_base = None
    if show_truth:
        _, truth_base = _truth_source(by_label, reference, reference_path)

    written = []
    for var in variables:
        for_var = {m: v[var] for m, v in per_model.items() if var in v}
        if not for_var:
            continue
        truth = None
        if truth_base is not None:
            d = metrics_dir(truth_base, var)
            if os.path.isdir(d):
                truth = read_skill_map_metrics(
                    d,
                    var,
                    metric,
                    kinds=("targ",),
                    member_dim=member_dim,
                    latlon_dims=latlon_dims,
                )
        written.append(
            render_var_maps(
                for_var,
                truth,
                var,
                out,
                models=[m for m in by_label if m in for_var],
                metric=metric,
                kind=kind,
                season=season,
                member_num=member_num,
                **kwargs,
            )
        )
    return written


# ============================================================================
# Single-date event figures (read from NetCDF, not from the metric stores)
# ============================================================================
# ml-eval only writes time-reduced diagnostics, so a single day never appears in
# a metrics_*.zarr store. These two entry points therefore read the prediction
# and target NetCDFs directly -- the pre-evaluation eyeball check that catches
# wrong units, spatial artefacts or over-smoothing before any metric is trusted.
# Everything else (the `{label: path}` spec, `kind`, `show_truth`, the layouts,
# the read/render split) is identical to the metric maps above; only `path` now
# points at one source's NetCDF file rather than at a run directory.

# Display ranges for a single day, in the standard units get_units_scale gives.
EVENT_VMINMAX = {
    "pr": [0, 250],  # mm/day
    "tas": [290, 305],  # K
    "tasmax": [290, 305],
    "tasmin": [290, 305],
    "uas": [-10, 10],  # m/s
    "vas": [-10, 10],
    "psl": [1000, 1015],  # hPa
    "hurs": [65, 100],  # %
}
EVENT_CMAP = "hot_r"
EVENT_CMAP_BIAS = "RdBu_r"


def read_event_fields(
    path,
    var,
    date,
    *,
    key="raw",
    member_dim="member",
    member_num=0,
    latlon_dims=None,
):
    """One source's field for one variable on one date, read from NetCDF.

    The event counterpart of :func:`read_skill_map_metrics`: same returned bundle, so
    the same renderers draw it. Depends only on the file's own layout -- no
    directory convention, no metric stores.

    Parameters
    ----------
    path : str
        NetCDF file holding ``var`` on a time axis (a prediction, or the target).
    var : str
        Variable to read.
    date : str
        Date to select, e.g. ``"1983-08-15"``. A partial-string match that
        resolves to a single timestep is accepted (so a file stamped at 12:00
        works), anything ambiguous is an error.
    key : str
        Which slot of the bundle the field lands in -- ``"targ"`` for the truth,
        ``"raw"`` for a model.
    member_dim, member_num, latlon_dims
        As in :func:`read_skill_map_metrics`.

    Returns
    -------
    dict or None
        A bundle shaped like :func:`read_skill_map_metrics`' output, or None when the
        file or the variable is missing.

    Raises
    ------
    ValueError
        If ``date`` does not resolve to exactly one timestep, or the field does
        not reduce to the expected lat/lon dims.
    """
    if not os.path.exists(path):
        return None
    ds = xr.open_dataset(path)
    if var not in ds:
        return None

    da = ds[var].sel(time=date)
    if "time" in da.dims:
        if da.sizes["time"] != 1:
            raise ValueError(
                f"date {date!r} selects {da.sizes['time']} timesteps in {path}; "
                "give a date that resolves to exactly one."
            )
        da = da.isel(time=0, drop=True)

    da = select_member(da, member_dim, 0 if key == "targ" else member_num)
    dims = horizontal_dims(da, latlon_dims)
    check_latlon_field(da, list(dims))

    scale, units = get_units_scale(ds, var)
    vminmax = EVENT_VMINMAX.get(var)
    if vminmax is None:
        print(f"WARNING: no display range for {var}; colour scales may not match.")
        vminmax = [None, None]

    lat, lon = plot_coords(ds, da, dims)

    return {
        "fields": {key: da.values * scale},
        "lat": lat,
        "lon": lon,
        "units": units,
        "title": str(date),
        "vminmax": vminmax,
        "cmap": EVENT_CMAP,
        "vminmax_bias": [None, None],
        "cmap_bias": EVENT_CMAP_BIAS,
    }


def _add_event_error(info, truth, kind):
    """Fill ``info["fields"][kind]`` with the model's error against the truth."""
    if kind == "raw" or info is None:
        return info
    raw = info["fields"].get("raw")
    targ = None if truth is None else truth["fields"].get("targ")
    if raw is None or targ is None:
        info["fields"][kind] = None
        return info
    diff = raw - targ
    if kind == "relbias":
        with np.errstate(divide="ignore", invalid="ignore"):
            diff = 100.0 * diff / targ
        diff = np.where(np.abs(targ) < 1e-12, np.nan, diff)
    info["fields"][kind] = diff
    return info


def _read_event_sources(
    by_label, variables, date, truth_by_var, kind, member_dim, member_num, latlon_dims
):
    """``{label: {var: bundle}}`` for every source whose file has the variable."""
    per_model = {}
    for label, path in by_label.items():
        per_var = {}
        for var in variables:
            info = read_event_fields(
                path,
                var,
                date,
                member_dim=member_dim,
                member_num=member_num,
                latlon_dims=latlon_dims,
            )
            if info is None:
                print(f"[event] skip {label} / {var}: not in {path}")
                continue
            per_var[var] = _add_event_error(info, (truth_by_var or {}).get(var), kind)
        if per_var:
            per_model[label] = per_var
    return per_model


def _read_event_truth(
    by_label, variables, date, reference, reference_path, member_dim, latlon_dims
):
    """``{var: bundle}`` holding the ``targ`` field of each variable."""
    _, base = _truth_source(by_label, reference, reference_path)
    truth = {}
    for var in variables:
        info = read_event_fields(
            base,
            var,
            date,
            key="targ",
            member_dim=member_dim,
            latlon_dims=latlon_dims,
        )
        if info is not None:
            truth[var] = info
    return truth


def plot_multivar_event_maps(
    paths,
    out,
    variables,
    date,
    *,
    kind="raw",
    show_truth=True,
    reference=None,
    reference_path=None,
    member_dim="member",
    member_num=0,
    latlon_dims=None,
    **kwargs,
):
    """Plot one date across several variables and models, as a single figure.

    The event twin of :func:`plot_multivar_skill_maps`, with the same
    ``(1 + N models) x len(variables)`` grid: the truth on top, one model per
    row, ``kind`` replacing the field in the model rows with the error.

    Parameters
    ----------
    paths : mapping or sequence
        ``{label: path}`` where ``path`` is that source's **NetCDF file** (see
        ``plot_hexagon.normalize_paths`` for the accepted spellings).
    out : str
        Output directory for the PNG.
    variables : iterable of str
        Variables to plot, one column each.
    date : str
        The day to plot, e.g. ``"1983-08-15"``.
    kind : {"raw", "bias", "relbias"}
        What the model rows show, as in :func:`plot_multivar_skill_maps`.
    show_truth : bool
        Draw the truth as the top row. Unlike the metric maps, the truth is a
        separate file here, so name it with ``reference``/``reference_path``.
    reference : str, optional
        Label of the truth source; default the first entry of ``paths``.
    reference_path : str, optional
        NetCDF file to read the truth from, when it is not one of ``paths``.
    member_dim, member_num, latlon_dims
        As in :func:`read_event_fields`.
    **kwargs
        Forwarded to :func:`render_multivar_maps`.

    Returns
    -------
    str or None
        Path of the PNG written, or None when no source had data.
    """
    by_label = normalize_paths(paths)
    variables = list(variables)
    truth = (
        _read_event_truth(
            by_label,
            variables,
            date,
            reference,
            reference_path,
            member_dim,
            latlon_dims,
        )
        if show_truth or kind != "raw"
        else None
    )
    per_model = _read_event_sources(
        by_label, variables, date, truth, kind, member_dim, member_num, latlon_dims
    )
    if not per_model:
        print(f"[event] {date}: nothing to plot.")
        return None

    kwargs.setdefault(
        "filename", f"multivar_event_{date}_{kind}{_member_tag(member_num)}"
    )
    return render_multivar_maps(
        per_model,
        truth if show_truth else None,
        out,
        variables=variables,
        models=[m for m in by_label if m in per_model],
        metric=str(date),
        kind=kind,
        season=None,
        member_num=member_num,
        **kwargs,
    )


def plot_var_event_maps(
    paths,
    out,
    variables,
    date,
    *,
    kind="raw",
    show_truth=True,
    reference=None,
    reference_path=None,
    member_dim="member",
    member_num=0,
    latlon_dims=None,
    **kwargs,
):
    """Plot one date for one variable at a time, one model per row.

    The event twin of :func:`plot_var_skill_maps`: truth at ``(1, 1)``, models down
    column 2, and their error in column 3 when ``kind`` is ``"bias"`` or
    ``"relbias"``. One figure per variable.

    Parameters
    ----------
    paths, out, variables, date, kind, show_truth, reference, reference_path
        As in :func:`plot_multivar_event_maps`.
    member_dim, member_num, latlon_dims
        As in :func:`read_event_fields`.
    **kwargs
        Forwarded to :func:`render_var_maps`.

    Returns
    -------
    list of str
        Paths of the PNGs written (one per variable that had data).
    """
    by_label = normalize_paths(paths)
    variables = list(variables)
    truth = (
        _read_event_truth(
            by_label,
            variables,
            date,
            reference,
            reference_path,
            member_dim,
            latlon_dims,
        )
        if show_truth or kind != "raw"
        else None
    )
    per_model = _read_event_sources(
        by_label, variables, date, truth, kind, member_dim, member_num, latlon_dims
    )
    if not per_model:
        print(f"[event] {date}: nothing to plot.")
        return []

    # one figure per variable, so an explicit filename can only apply to one
    override = kwargs.pop("filename", None)
    if override is not None and len(variables) > 1:
        print("[event] filename ignored: one figure per variable is written.")
        override = None

    written = []
    for var in variables:
        for_var = {m: v[var] for m, v in per_model.items() if var in v}
        if not for_var:
            continue
        written.append(
            render_var_maps(
                for_var,
                (truth or {}).get(var) if show_truth else None,
                var,
                out,
                models=[m for m in by_label if m in for_var],
                metric=str(date),
                kind=kind,
                season=None,
                member_num=member_num,
                filename=override
                or f"var_event_{var}_{date}_{kind}{_member_tag(member_num)}",
                **kwargs,
            )
        )
    return written


# ============================================================================
# Climate-change-signal maps (read from the `ml-eval signal` stores)
# ============================================================================
# The signal twins of plot_var_skill_maps / plot_multivar_skill_maps. Same
# layouts, same renderers, same {label: path} spec -- but a different store and
# two contract changes that make them separate entry points rather than a flag:
#
#   * the truth is the RCM's OWN signal, which lives in its own run directory,
#     so `reference_path` is required (as it is for the signal scorecards);
#     `ml-eval signal` writes no targ_/pred_ pair inside a model's bundle;
#   * the error panel is DERIVED here (model delta - RCM delta), where the skill
#     maps read a bias_ field ml-eval already computed.
#
# `ml-eval signal` canonicalises units before computing (the same reason
# plot_var_series applies no scaling), so these readers convert nothing: pass
# `units=` for the colourbar label. ``reldelta`` is stored as a fraction and is
# multiplied by 100 for display, matching the signal scorecards.

SIGNAL_STORE = "metrics_signal_field.zarr"
SIGNAL_QUANTITIES = ("delta", "reldelta")

# index -> (label, variable it belongs to or None), mirroring METRIC_INFO.
SIGNAL_INDEX_INFO = {
    "mean": ("Mean", None),
    "p01": ("P01", None),
    "p99": ("P99", None),
    "r01": ("Wet-day frequency", "pr"),
    "sdii": ("SDII", "pr"),
    "rx1day": ("RX1day", "pr"),
}
# what `ml-eval signal` actually stores per variable
_SIGNAL_INDICES = {"pr": ["r01", "sdii", "rx1day"]}
_SIGNAL_INDICES_DEFAULT = ["mean", "p01", "p99"]


def signal_map_indices(var=None):
    """Index keys `ml-eval signal` stores, optionally for one variable.

    The signal twin of :func:`skill_map_indices`. Note the spellings differ from
    the fast track: the annual maximum is ``rx1day`` here and ``max1day`` there,
    and the low percentile is ``p01`` rather than ``p1``.
    """
    if var is None:
        return list(SIGNAL_INDEX_INFO)
    return list(_SIGNAL_INDICES.get(var, _SIGNAL_INDICES_DEFAULT))


def read_signal_map_metrics(
    metrics_dir,
    var,
    index="mean",
    *,
    quantity="delta",
    key="raw",
    units=None,
    latlon_dims=None,
):
    """One source's climate-change-signal map for one variable and index.

    Reads ``<quantity>_<index>_field`` from ``metrics_signal_field.zarr`` -- the
    change between the future and historical periods that ``ml-eval signal``
    computed for this source against its *own* baseline. Depends only on
    ml-eval's output format; no directory-layout assumptions.

    Parameters
    ----------
    metrics_dir : str
        Directory holding the store (e.g. ``<path>/<var>``).
    var : str
        Variable the signal describes -- used for the index check and the
        diverging colormap.
    index : str
        Index key; see :func:`signal_map_indices`.
    quantity : {"delta", "reldelta"}
        Absolute change (data units) or relative change. ``reldelta`` is stored
        as a fraction and returned as a percentage.
    key : str
        Bundle slot: ``"targ"`` for the RCM truth, ``"raw"`` for a model.
    units : str, optional
        Colourbar label for ``delta``; ignored for ``reldelta`` (always ``%``).
    latlon_dims : tuple of str, optional
        Horizontal dims; None auto-detects (see :func:`horizontal_dims`).

    Returns
    -------
    dict or None
        A bundle shaped like :func:`read_skill_map_metrics`' output, or None
        when the store or the diagnostic is missing.

    Raises
    ------
    ValueError
        If ``quantity`` is unknown, or ``index`` belongs to another variable.
    """
    if quantity not in SIGNAL_QUANTITIES:
        raise ValueError(
            f"quantity must be one of {SIGNAL_QUANTITIES}, got {quantity!r}."
        )
    try:
        label, var_specific = SIGNAL_INDEX_INFO[index]
    except KeyError:
        raise ValueError(
            f"unknown index {index!r}; available: {', '.join(SIGNAL_INDEX_INFO)}"
        ) from None
    if var_specific is not None and var != var_specific:
        raise ValueError(
            f"Index {index} only suitable for variable {var_specific}, "
            f"but variable loaded is {var}"
        )

    store = os.path.join(metrics_dir, SIGNAL_STORE)
    if not os.path.isdir(store):
        return None
    ds = xr.open_zarr(store)
    try:
        da = ds["value"].sel(diagnostic=f"{quantity}_{index}_field").load()
    except KeyError:
        return None

    dims = horizontal_dims(da, latlon_dims)
    check_latlon_field(da, list(dims))
    lat, lon = plot_coords(ds, da, dims)

    # a fraction on disk, a percentage on the colourbar (as in the scorecards)
    scale = 100.0 if quantity == "reldelta" else 1.0
    shown = "%" if quantity == "reldelta" else (units or "")

    # A change straddles zero, so both the signal and its error are diverging.
    cmap = "BrBG" if var == "pr" else "RdBu_r"
    return {
        "fields": {key: da.values * scale},
        "lat": lat,
        "lon": lon,
        "units": shown,
        "title": f"{label} {quantity}",
        "vminmax": [None, None],
        "cmap": cmap,
        "vminmax_bias": [None, None],
        "cmap_bias": cmap,
    }


def _read_signal_sources(
    by_label, variables, index_of, quantity, kind, truth_by_var, units, latlon_dims
):
    """``{label: {var: bundle}}``, each with the signal and (if asked) its error."""
    per_model = {}
    for label, base in by_label.items():
        per_var = {}
        for var in variables:
            d = metrics_dir(base, var)
            if not os.path.isdir(d):
                print(f"[signal-maps] skip {label} / {var}: {d} not found")
                continue
            info = read_signal_map_metrics(
                d,
                var,
                index_of[var],
                quantity=quantity,
                units=(units or {}).get(var),
                latlon_dims=latlon_dims,
            )
            if info is None:
                print(f"[signal-maps] skip {label} / {var}: no signal store in {d}")
                continue
            # the error against the RCM's own signal, derived exactly as the
            # event maps derive theirs
            per_var[var] = _add_event_error(info, (truth_by_var or {}).get(var), kind)
        if per_var:
            per_model[label] = per_var
    return per_model


def _read_signal_truth(
    by_label,
    variables,
    index_of,
    quantity,
    reference,
    reference_path,
    units,
    latlon_dims,
):
    """``{var: bundle}`` holding the RCM's own signal."""
    _, base = _truth_source(by_label, reference, reference_path)
    truth = {}
    for var in variables:
        d = metrics_dir(base, var)
        if not os.path.isdir(d):
            continue
        info = read_signal_map_metrics(
            d,
            var,
            index_of[var],
            quantity=quantity,
            key="targ",
            units=(units or {}).get(var),
            latlon_dims=latlon_dims,
        )
        if info is not None:
            truth[var] = info
    return truth


def _signal_indices_for(variables, index):
    """``{var: index}``: one index for all, or a per-variable mapping."""
    out = {}
    for var in variables:
        chosen = index.get(var) if isinstance(index, dict) else index
        out[var] = chosen or signal_map_indices(var)[0]
    return out


def plot_var_signal_maps(
    paths,
    out,
    variables,
    *,
    index=None,
    quantity="delta",
    kind="raw",
    show_truth=True,
    reference=None,
    reference_path=None,
    season="annual",
    units=None,
    latlon_dims=None,
    **kwargs,
):
    """Map one index's climate-change signal for one variable at a time.

    The signal twin of :func:`plot_var_skill_maps`, and the map counterpart of
    ``plot_var_signal_scorecard``: the RCM's own signal at (1, 1), each model's
    signal down column 2, and -- with ``kind="bias"`` -- its signal *error*
    (model minus RCM) in column 3.

    Parameters
    ----------
    paths : mapping or sequence
        ``{label: path}``; one variable's store is read from ``<path>/<var>/``.
    out : str
        Output directory for the PNGs.
    variables : iterable of str
        Variables to plot (one figure each).
    index : str or mapping, optional
        Index to map -- one value, or ``{var: index}``. Default: the variable's
        first index (see :func:`signal_map_indices`).
    quantity : {"delta", "reldelta"}
        Absolute or relative change.
    kind : {"raw", "bias"}
        The signal itself, or its error against the RCM's signal.
    show_truth : bool
        Draw the RCM's own signal at (1, 1).
    reference, reference_path : str, optional
        Where the RCM signal is read from. ``ml-eval signal`` writes no truth
        into a model's bundle, so unless the RCM is one of ``paths`` you must
        pass ``reference_path`` (its own run directory).
    season : str
        Label only -- title and output file name.
    units : mapping, optional
        ``{var: label}`` for the colourbar when ``quantity="delta"``.
    latlon_dims : tuple of str, optional
        Passed to :func:`read_signal_map_metrics`.
    **kwargs
        Forwarded to :func:`render_var_maps`.

    Returns
    -------
    list of str
        Paths of the PNGs written (one per variable that had data).
    """
    if kind not in ("raw", "bias"):
        raise ValueError(f"kind must be 'raw' or 'bias' for signal maps, got {kind!r}.")
    by_label = normalize_paths(paths)
    variables = list(variables)
    index_of = _signal_indices_for(variables, index)

    truth = (
        _read_signal_truth(
            by_label,
            variables,
            index_of,
            quantity,
            reference,
            reference_path,
            units,
            latlon_dims,
        )
        if show_truth or kind != "raw"
        else None
    )
    per_model = _read_signal_sources(
        by_label, variables, index_of, quantity, kind, truth, units, latlon_dims
    )
    if not per_model:
        print(f"[signal-maps] {quantity}: no metrics found -- nothing plotted.")
        return []

    written = []
    for var in variables:
        for_var = {m: v[var] for m, v in per_model.items() if var in v}
        if not for_var:
            continue
        written.append(
            render_var_maps(
                for_var,
                (truth or {}).get(var) if show_truth else None,
                var,
                out,
                models=[m for m in by_label if m in for_var],
                metric=f"{index_of[var]}_{quantity}",
                kind=kind,
                season=season,
                raw_symmetric=True,  # a change straddles zero
                filename=(
                    f"var_signal_maps_{var}_{index_of[var]}_{quantity}_{kind}"
                    f"{_season_suffix(season)}"
                ),
                **kwargs,
            )
        )
    return written


def plot_multivar_signal_maps(
    paths,
    out,
    variables,
    *,
    index=None,
    quantity="delta",
    kind="raw",
    show_truth=True,
    reference=None,
    reference_path=None,
    season="annual",
    units=None,
    latlon_dims=None,
    **kwargs,
):
    """Map one index's climate-change signal across variables and models.

    The signal twin of :func:`plot_multivar_skill_maps`: columns are the
    variables, rows the sources, the RCM's own signal on top, and ``kind="bias"``
    replaces the model rows with their signal error.

    Parameters
    ----------
    Same as :func:`plot_var_signal_maps`, except that one figure is produced for
    all ``variables`` (columns) rather than one per variable.

    Returns
    -------
    str or None
        Path of the PNG written, or None when no source had data.
    """
    if kind not in ("raw", "bias"):
        raise ValueError(f"kind must be 'raw' or 'bias' for signal maps, got {kind!r}.")
    by_label = normalize_paths(paths)
    variables = list(variables)
    index_of = _signal_indices_for(variables, index)

    truth = (
        _read_signal_truth(
            by_label,
            variables,
            index_of,
            quantity,
            reference,
            reference_path,
            units,
            latlon_dims,
        )
        if show_truth or kind != "raw"
        else None
    )
    per_model = _read_signal_sources(
        by_label, variables, index_of, quantity, kind, truth, units, latlon_dims
    )
    if not per_model:
        print(f"[signal-maps] {quantity}: no metrics found -- nothing plotted.")
        return None

    tag = "-".join(index_of[v] for v in variables)
    return render_multivar_maps(
        per_model,
        truth if show_truth else None,
        out,
        variables=variables,
        models=[m for m in by_label if m in per_model],
        metric=f"{tag}_{quantity}",
        kind=kind,
        season=season,
        raw_symmetric=True,
        filename=f"multivar_signal_maps_{tag}_{quantity}_{kind}{_season_suffix(season)}",
        **kwargs,
    )
