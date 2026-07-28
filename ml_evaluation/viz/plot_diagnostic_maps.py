"""
Code to plot a matrices of lat/lon gridded fields from the validation package.

There are two different methods for obtaining data:

1. Pre-validation package NetCDF files

  Input data paths: simply input entire path for target and for prediction
  e.g.
  truth_file = "/home/target.nc"
  prediction_file = "/home/prediction.nc"

  There is just 1 type of figure included:

  plot_example_day()
  - Designed for displaying 1 model and multiple variables
  - Shows results for a single day (time step)
  - Takes truth and prediction data straight from netcdf files rather
    than first using the validation package.
  - Plots (truth, prediction) pairs in rows for all variables listed
  - Colorbar range and colormap defined within function

2. Post-validation package Zarr files

  Input data paths: these need to be built
  e.g.
  pre_path="/home/historical_perfect/"
  mid_path="/"
  post_path="/ANN/"

  They assume a structure of the form:

  $pre_path$model_dir$mid_path$variable_name$post_path

  where pre_path, mid_path and post_path are arguments of the plotting
  functions and model_dir and variable_name are either a single argument,
  or come from looping over a list.

  The name of the zarr file is inferred from the choice of diagnostics.
  Available diagnostics are listed in the dictionary function:
  get_variable_info()

  There are 2 main types of figures included:

  plot_truth_predict_fields_1model()
  - Designed for displaying 1 model and multiple variables
  - Shows results for a single diagnostic metric from the validation package
  - Takes truth and prediction data from Zarr files (output from the
    validation package).
  - Plots (truth, prediction) pairs in rows for all variables listed
  - Colorbar range and colormap defined by get_example_color_scales()

  plot_truth_predict_bias_1variable_portrait()
  plot_truth_predict_bias_1variable_landscape()
  - Designed for displaying 1 variable and multiple models
  - Shows results for a single diagnostic metric from the validation package
  - Takes truth and prediction data from Zarr files (output from the
    validation package).
  - Plots (truth, prediction, prediction bias) in rows for all models listed
  - Colorbar range and colormap defined by get_example_color_scales()
  - The portrait and landscape versions control where the truth appears, so
    that it is only plotted once
    Portrait: Truth has top row to itself, further rows show
    (prediction, prediction bias) pairs
    Landscape: Truth has left column to itself, further column rows show
    (prediction, prediction bias) pairs

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
  - Warning: this is not exhaustive! It is left to the user to populate this
    with their own preferences related to the data plotted. If your option is
    not included here, it will default to matplotlib derived ranges and related
    plots will not necessarily have the correctly matching color scales.
    

------------------------------------------------------------------------------
Example calls

import plot_diagnostic_maps as pf

# --------------------
# plot_example_day()
truth_file="/home/target.nc"
prediction_file="/home/prediction.nc"
out_dir="/home/PLOTS/"
variable_list=["tas","hurs","tasmax","psl","tasmin","uas","pr","vas"]
date_chosen="1983-08-15"
model_name="MODEL1"
experiment_info="perfect predictors"

pf.plot_example_day(truth_file,prediction_file,out_dir,variable_list,date_chosen,model_name,experiment_info=experiment_info)

# --------------------
# plot_truth_predict_fields_1model()
#   In this example the built filepath is
#   $pre_path$model_dir$mid_path$variable_name$post_path
#   where $variable_name comes from looping over the list variable_list
#   e.g. "/home/historical_perfect/MODEL1/tas/ANN/*.zarr"

pre_path="/home/historical_perfect/"
mid_path="/"
post_path="/ANN/"
out_dir="/home/historical_perfect/PLOTS/"
variable_list=["tas","hurs","tasmax","psl","tasmin","uas","pr","vas"]
model_dir="MODEL1"
model_name="MODEL1"
experiment_info="perfect predictors"
metric_name="mean"

pf.plot_truth_predict_fields_1model(pre_path,mid_path,post_path,out_dir,model_dir,model_name,variable_list,metric_name=metric_name,season=None,experiment_info=experiment_info)


"""

import os
import numpy as np
import xarray as xr

import matplotlib
matplotlib.use('Agg')  # Improves plotting efficiency. Remove if you want to pause code and view plots within a run.
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature


# ----------------------------------------------------------------------------
def plot_map_matrix_pcolormesh(lon_list, lat_list, data_list, ncols=1,
    nrows=1, vminmax=None, filepath="filepath.png", title_list=None,
    cbartitle_list=None, color_map_list=None, FIGWIDTH=None, FIGHEIGHT=None,
    panel_size=4.0, title_fs=None, figtitle=None, incl_corner_mean=True):
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
        nrows, ncols,
        figsize=(FIGWIDTH, FIGHEIGHT),
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True,
        squeeze=False
    )

    # Only use optional per-panel lists if they cover every panel
    if vminmax and len(vminmax) < len_data_list:
        vminmax = None
    if title_list and len(title_list) < len_data_list:
        title_list = None
    if cbartitle_list and len(cbartitle_list) < len_data_list:
        cbartitle_list = None
    if color_map_list and len(color_map_list) < len_data_list:
        color_map_list = [plt.get_cmap('hot_r')] * len_data_list

    icount = 0
    total_panels = nrows * ncols
    for iiR in range(nrows):
        for iiC in range(ncols):
            ax = axes[iiR, iiC]

            if _is_skipped(icount):
                ax.axis('off')
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
                lon_list[icount], lat_list[icount], data_list[icount],
                vmin=ivmin, vmax=ivmax,
                cmap=color_map,
                transform=ccrs.PlateCarree(),
                shading='auto',
            )

            if title_list:
                ax.set_title(title_list[icount], fontsize=title_fs)

            if incl_corner_mean:
                mean_val = np.nanmean(data_list[icount])
                ax.text(
                    0.98, 0.98, f'{mean_val:.2f}',
                    transform=ax.transAxes,
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )

            # Geographic features
            ax.coastlines(resolution='50m')  # '10m' for more detail
            ax.add_feature(cartopy.feature.BORDERS, linewidth=0.5)

            cbar = plt.colorbar(im_ax, ax=ax, orientation='horizontal', pad=0.01)
            if cbartitle_list:
                cbar.set_label(cbartitle_list[icount])

            icount += 1

        if icount >= total_panels:
            break

    fig.suptitle(figtitle)
    print(filepath)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
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
            sizes = {d: ds.dims[d] for d in extra}
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
    data = ds['value'] if is_zarr else ds[var]

    scale = 1.0
    units = data.attrs.get("units")
    if var == "pr":
        # Convert to mm/day where necessary.
        # Units may differ between datasets and may not be specified.
        if units in ["kg m-2 s-1", "kg/m2/s", "kg m^-2 s^-1", "kg.m-2.s-1", "kg m**-2 s**-1"]:
            scale = 86400.0
        elif units in ["mm/day", "mm day-1", "mm d-1", "mm.day-1"]:
            scale = 1.0
        elif data.max() > 1:
            scale = 1.0
            print(f"WARNING: {var} has units = {units}, i.e. unknown or missing: assume mm/day")
        else:
            scale = 86400.0
            print(f"WARNING: {var} has units = {units}, i.e. unknown or missing: assume kg/m2/s so convert to mm/day")
        units="mm/day"

    if var == "uas" or var == "vas":
        units = "m/s"
        if units not in ["m/s", "m s-1", "m.s-1"]:
            print(f"WARNING: {var} has units = {units}, i.e. unknown or missing: assume m/s")

    if var == "psl":
        # Convert to hPa where necessary
        if units in ["Pa"]:
            scale = 1 / 100.0
        elif units in ["hPa"]:
            scale = 1.0
        elif data.max() < 5000:
            scale = 1.0
            print(f"WARNING: {var} has units = {units}, i.e. unknown or missing: assume hPa")
        else:
            scale = 1 / 100.0
            print(f"WARNING: {var} has units = {units}, i.e. unknown or missing: assume Pa so convert to hPa")
        units="hPa"

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
    cmap_name = 'hot_r' # default?
    cmap_name_bias = 'hot_r' # default?

    if metric_name in ["mean"]:
        cmap_name = 'hot_r'
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
            vminmax = [0,0.6]
            vminmax_bias = [-0.3,0.3]
            manually_defined = True
            cmap_name = 'hot_r'
            cmap_name_bias = 'PiYG_r'

    if metric_name in ["mv_pcorr", "mv_scorr"]:
        if variable_name in ["pr"]:
            vminmax = [-0.9,0.9]
            vminmax_bias = [-0.06,0.06]
            manually_defined = True
            cmap_name = 'PiYG_r'
            cmap_name_bias = 'PiYG_r'

    if not manually_defined:
        print(
            f"WARNING: color scales not set by get_example_color_scales() so may not match."
        )

    return vminmax, cmap_name, vminmax_bias, cmap_name_bias

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

    METRIC_INFO = {
        # fast (basic) track metrics
        "mean":     ("Climatology mean",      "", True,  "metrics_fast_field.zarr", None),
        "p99":      ("99% Quantile",       "", True,  "metrics_fast_field.zarr", None),
        "p1":       ("1% Quantile",        "", True,  "metrics_fast_field.zarr", None),
        "max1day":  ("Mean of max1day",     "", True,  "metrics_fast_field.zarr", None),
        "min1day":  ("Mean of min1day",     "", True,  "metrics_fast_field.zarr", None),
        "acorr":    ("Auto-correlation", "correlation", False, "metrics_fast_field.zarr", None),

        # pr: fast (basic) track metrics
        "p98wet":   ("98% Quantile (wet days)", "", True, "metrics_fast_pr_field.zarr", "pr"),
        "sdii":     ("Simple Daily Intensity Index", "", True, "metrics_fast_pr_field.zarr", "pr"),
        "r1":       ("Mean of r1", "no. days", False, "metrics_fast_pr_field.zarr", "pr"),

        # slow (extended) track metrics
        "annstd":   ("Annual standard deviation", "", True, "metrics_slow_field.zarr", None),
        "p90":      ("90% Quantile",           "", True, "metrics_slow_field.zarr", None),

        # pr: slow (extended) track metrics
        "p90wet":   ("90% Quantile (wet days)", "", True, "metrics_slow_pr_field.zarr", "pr"),
        "r10":      ("Mean of r10", "no. days", False, "metrics_slow_pr_field.zarr", "pr"),
        "cdd":      ("Mean of cdd",       "no. days", False, "metrics_slow_pr_field.zarr", "pr"),
        "cwd":      ("Mean of cwd",       "no. days", False, "metrics_slow_pr_field.zarr", "pr"),

        # tasmax: slow (extended) track metrics
        "su":       ("Mean no. Summer days",  "no. days", False, "metrics_slow_tasmax_field.zarr", "tasmax"),
        "hot":      ("Mean no. Hot days",     "no. days", False, "metrics_slow_tasmax_field.zarr", "tasmax"),
        "ice":      ("Mean no. Ice days",     "no. days", False, "metrics_slow_tasmax_field.zarr", "tasmax"),

        # tasmin: slow (extended) track metrics
        "tropnight":("Mean no. Tropical nights", "no. days", False, "metrics_slow_tasmin_field.zarr", "tasmin"),
        "frost":    ("Mean no. Frost days",      "no. days", False, "metrics_slow_tasmin_field.zarr", "tasmin"),

        # multivariate metrics
        "mv_pcorr": ("MV Pearson correlation",  "correlation", False, "metrics_mv_field.zarr", None),
        "mv_scorr": ("MV Spearman correlation", "correlation", False, "metrics_mv_field.zarr", None),
    }

    try:
        title_name, units, VariableUnits, zarr_name, var_specific = METRIC_INFO[metric_name]
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
    mem_num_new = member_num
    if member_dim is not None and member_dim in ds.dims:
        nmems = ds[member_dim].size
        if member_num < 0:
            # Average over the member dimension only
            ds = ds.mean(dim=member_dim)
            mem_num_new = member_num
        elif member_num in ds[member_dim].values:
            ds = ds.sel({member_dim: member_num}, drop=True)
            mem_num_new = member_num
        else:
            print(f"Warning: member_num {member_num} doesn't exist "
                  f"so use first member in list.")
            ds = ds.isel({member_dim: 0}, drop=True)
            mem_num_new = 0
    else:
        raise ValueError(
            f"Given member_dim {member_dim} but not found in ds.dims {ds.dims}"
            f" check if using correct dim name."
        )
    return ds, mem_num_new


# ----------------------------------------------------------------------------
def plot_example_day(
    truth_file,
    prediction_file,
    out_dir,
    variable_list,
    date_chosen,
    model_name,
    experiment_info="",
    member_dim="member",
    member_num=0,
    filename=None,
    latlon_dims=["lat", "lon"],
    ncols=4,
    nrows=4,
    ):
    """
    Plot a truth-vs-prediction matrix of maps for a single chosen
    date, one row pair (truth, prediction) per variable in
    `variable_list`.

    Parameters
    ----------
    truth_file : str
        Path to the truth NetCDF-style dataset (opened with
        `xr.open_dataset`).
    prediction_file : str
        Path to the prediction NetCDF-style dataset (opened with
        `xr.open_dataset`).
    out_dir : str
        Directory the output figure is saved into (created if it
        doesn't already exist).
    variable_list : list of str
        Variable names to plot, e.g. ["pr", "tas", "psl"]. Each
        variable contributes a "Truth" and "Pred" panel.
    date_chosen : str
        Date/time label used to select the timestep via
        `.sel(time=date_chosen)`, and included in the default
        filename and figure title.
    model_name : str
        Name of the prediction model, used in the default filename
        and figure title.
    experiment_info : str, optional
        Extra text appended to the figure title (default "").
    member_dim : str or None, optional
        Name of the ensemble-member dimension in the prediction
        data (default "member"). Passed to `reduce_mem_dim`.
    member_num : int, optional
        Which ensemble member to plot; < 0 means ensemble mean
        (default 0).
    filename : str or None, optional
        Output filename (without extension). If None (default), a
        name is auto-generated from `date_chosen`, the number of
        variables, `model_name`, and the member info.
    latlon_dims : list of str, optional
        Names of the lat/lon dimensions in the datasets (default
        ["lat", "lon"]).
    ncols : int, optional
        Number of columns in the plot matrix (default 4).
        Should be even to align (truth, prediction) columns
    nrows : int, optional
        Number of rows in the plot matrix (default 4).

    Returns
    -------
    out_filepath : str
        Full path to the saved PNG figure.

    Raises
    ------
    ValueError
        If lon_list, lat_list, and data_list are not all the same
        length, or if that length exceeds ncols*nrows.
    """

    # ncols should be even to align (truth, prediction) columns
    if ncols % 2 != 0:
        print(f"Warning: ncols should be even to align (truth, prediction) columns, but got {ncols}")

    x0_ds_tstep = xr.open_dataset(truth_file).sel(time=date_chosen).load()
    x1_ds_tstep = xr.open_dataset(prediction_file).sel(time=date_chosen).load()

    # Reduce ensemble data to single member or ensemble mean
    x1_ds_tstep, member_num = reduce_mem_dim(x1_ds_tstep, member_dim, member_num)

    mem_info = "_m0"
    if member_dim is not None:
        if member_num < 0:
            # average over the member dimension
            mem_info = "_em"
        else:
            mem_info = "_m" + str(int(member_num))

    # Final check: dataset should now be a 2D lat/lon field (no member/time/etc left)
    check_latlon_field(x0_ds_tstep, latlon_dims)
    check_latlon_field(x1_ds_tstep, latlon_dims)

    os.makedirs(out_dir, exist_ok=True)

    nvariables = len(variable_list)
    if filename is None:
        filename = "maps_exampleday_" + date_chosen + "_" + str(int(nvariables)) + "var_" + model_name + mem_info

    # Set up list of data fields and lat/lon info for plotting to matrix
    # (row-major order)
    field_list = []
    title_list = []
    vminmax_list = []
    ctitle_list = []

    for nn in range(nvariables):
        var = variable_list[nn]

        ds0 = x0_ds_tstep[var]
        ds1 = x1_ds_tstep[var]

        title_list.append("Truth: " + var)
        title_list.append("Pred: " + var)

        # Plot Maps of field metrics
        # Get units from truth field (check wrt prediction if possible?)
        scale0, units = get_units_scale(x0_ds_tstep, var)
        scale1, units1 = get_units_scale(x1_ds_tstep, var)

        # Set up colour bar scales (variable-specific, standard-unit ranges)
        manually_defined = False
        if var == "pr":
            # assumes units="mm/day"
            vminmax_list.append([0, 250])
            vminmax_list.append([0, 250])
            manually_defined = True

        if var == "tas" or var == "tasmax" or var == "tasmin":
            # assumes units="K"
            vminmax_list.append([290, 305])
            vminmax_list.append([290, 305])
            manually_defined = True

        if var == "uas" or var == "vas":
            # assumes units="m/s"
            vminmax_list.append([-10, 10])
            vminmax_list.append([-10, 10])
            manually_defined = True

        if var == "psl":
            # assumes units="hPa"
            vminmax_list.append([1000, 1015])
            vminmax_list.append([1000, 1015])
            manually_defined = True

        if var == "hurs":
            # assumes units="%"
            vminmax_list.append([65, 100])
            vminmax_list.append([65, 100])
            manually_defined = True

        if not manually_defined:
            print(
                f"WARNING: color scales not set so may not match."
            )

        field_list.append(ds0.values * scale0)
        field_list.append(ds1.values * scale1)
        ctitle_list.append(units)
        ctitle_list.append(units)

    lat_list = [ds0[latlon_dims[0]].values] * nvariables * 2
    lon_list = [ds0[latlon_dims[1]].values] * nvariables * 2
    cmap_list = [plt.get_cmap('hot_r')] * nvariables * 2  # 'gist_ncar_r'

    out_filepath = plot_map_matrix_pcolormesh(
        lon_list, lat_list, field_list, ncols=ncols, nrows=nrows,
        vminmax=vminmax_list, filepath=out_dir + filename + ".png",
        title_list=title_list, cbartitle_list=ctitle_list,
        color_map_list=cmap_list, FIGWIDTH=None, FIGHEIGHT=None,
        panel_size=2.0, title_fs=None,
        figtitle=date_chosen + " - " + model_name + " " + experiment_info)

    return out_filepath


# ----------------------------------------------------------------------------
def plot_truth_predict_fields_1model(
    pre_path,
    mid_path,
    post_path,
    out_dir,
    model_dir,
    model_name,
    variable_list,
    inv_mod_var=False,
    metric_name="mean",
    season=None,
    full_title=None,
    experiment_info=None,
    member_dim="member",
    member_num=0,
    filename=None,
    latlon_dims=["lat", "lon"],
    ncols=4,
    nrows=4,
    ):
    """
    Plot a truth-vs-prediction matrix of climatological mean maps
    for a single model, one row pair (truth, prediction) per
    variable in `variable_list`, reading data from the validation
    package's Zarr diagnostics files.

    Parameters
    ----------
    pre_path, mid_path, post_path : str
        Path fragments used to build the diagnostics file path as
        `pre_path + model_dir + mid_path + variable_name + post_path
        + "metrics_fast_field.zarr"` (or with `model_dir` and
        `variable_name` swapped if `inv_mod_var` is True).
    out_dir : str
        Directory the output figure is saved into (created if it
        doesn't already exist).
    model_dir : str
        Model directory name/segment used to build the diagnostics
        file path.
    model_name : str
        Name of the model, used in the default filename and figure
        title.
    variable_list : list of str
        Variable names to plot. Each variable contributes a "Truth"
        and "Pred" panel.
    inv_mod_var : bool, optional
        If True, swap the order of `model_dir` and `variable_name`
        when building the diagnostics file path (default False).
    metric_name : str, optional
        Which diagnostic to plot, default: "mean" (climate mean,
        `metrics_fast_field.zarr`).
        See function get_variable_info() to check what is available.
    season : str or None, optional
        If given, appended to the default output filename and full_title
        (default None). Not used to filter input files.
    full_title : str or None, optional
        Title for full matrix figure. If None (default), a
        title is auto-generated from the `metric_name`,
        `model_name`, member info, and `season`.
    experiment_info : str or None, optional
        Optional extra text appended to the figure title (default None).
    member_dim : str or None, optional
        Name of the ensemble-member dimension (default "member").
        Passed to `reduce_mem_dim`.
    member_num : int, optional
        Which ensemble member to plot; < 0 means ensemble mean
        (default 0).
    filename : str or None, optional
        Output filename (without extension). If None (default), a
        name is auto-generated from the number of variables,
        `model_name`, member info, and `season`.
    latlon_dims : list of str, optional
        Names of the lat/lon dimensions (default ["lat", "lon"]).
    ncols : int, optional
        Number of columns in the plot matrix (default 4).
        Should be even to align (truth, prediction) columns
    nrows : int, optional
        Number of rows in the plot matrix (default 4).

    Returns
    -------
    out_filepath : str
        Full path to the saved PNG figure.

    Raises
    ------
    ValueError
        If try to plot an unsupported combination of metric_name and variable,
        e.g. p98wet for tasmax;
        If fields loaded for plotting are not correctly reduced to lat/lon dims.
        If lon_list, lat_list, and data_list are not all the same
        length, or if that length exceeds ncols*nrows.

    """

    # ncols should be even to align (truth, prediction) columns
    if ncols % 2 != 0:
        print(f"Warning: ncols should be even to align (truth, prediction) columns, but got {ncols}")

    # Reduce ensemble data to single member or ensemble mean
    mem_info = "_m0"
    if member_dim is not None:
        if member_num < 0:
            # average over the member dimension
            mem_info = "_em"
        else:
            mem_info = "_m" + str(int(member_num))

    # Make sure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    nvariables = len(variable_list)

    # Output filename if not given
    if filename is None:
        filename = "maps_clim_" + metric_name + "_" + str(int(nvariables)) + "var_" + model_name + mem_info
        if season is not None:
            filename = filename + "_" + season

    # Default values for
    # - scaling field if necessary (e.g. psl Pa to hPa)
    # - defining units of data
    scale = 1
    units = ""

    VariableUnits = False # used to flag if metric units depend on original variable

    # Get data information for specific metric_name
    # default text for title, data units and the name of the linked zarr file
    # VariableUnits : bool, flags that units depended on prediction variable
    # variable_specific : str, None, flags if metric specific to just one variable
    title_name, units, VariableUnits, zarr_name, var_specific = get_variable_info(metric_name)

    # Overall title for matrix of plots if not given
    if full_title is None:
        full_title=title_name + " - " + model_name
        if season is not None:
            full_title=title_name + " " +season + " - " + model_name
        if experiment_info is not None:
            full_title = full_title + " " + experiment_info

    # Set up list of data fields and lat/lon info for plotting to matrix
    # (row-major order)
    field_list = [] # list of fields to be plotted
    title_list = [] # list of titles for each single plot
    vminmax_list = [] # list of color ranges where available else use default
    ctitle_list = [] # list of labels for colorbar (generally the units of the variable)
    cmap_list = []

    # Loop over all variables in variable_list
    for nn in range(nvariables):
        variable_name = variable_list[nn]

        if var_specific is not None and variable_name != var_specific:
            raise ValueError(
                f"Metric {metric_name} only suitable for "
                f"variable {var_specific}, but variable "
                f"loaded is {variable_name}"
            )

        title_list.append("Truth: " + variable_name)
        title_list.append("Pred: " + variable_name)

        # Build filepath for target and prediction fields
        diagnostics_file = pre_path + model_dir + mid_path + variable_name + post_path + zarr_name
        if inv_mod_var:
            # If variable directory comes before model directory, inverse order
            diagnostics_file = pre_path + variable_name + mid_path + model_dir + post_path + zarr_name
        ds = xr.open_zarr(diagnostics_file)

        # Load target and prediction fields
        targ_ds = ds['value'].sel(diagnostic="targ_" + metric_name + "_field").load()
        pred_ds = ds['value'].sel(diagnostic="pred_" + metric_name + "_field").load()

        # Check if member dimension, if so pick correct member
        # Reduce ensemble data to single member or ensemble mean
        targ_ds, _t_member_num = reduce_mem_dim(targ_ds, member_dim, 0)
        pred_ds, member_num = reduce_mem_dim(pred_ds, member_dim, member_num)

        # Final check: dataset should now be a 2D lat/lon field (no member/time/etc left)
        # - Raises error if not
        check_latlon_field(targ_ds, latlon_dims)
        check_latlon_field(pred_ds, latlon_dims)

        # Plot Maps of field metrics
        # Get units from zarr file or guess if need be
        if VariableUnits:
            scale, units = get_units_scale(ds, variable_name, is_zarr=True)

        # Load defined colorscales if available
        vminmax1, cmap_name1, vminmax_bias1, cmap_name_bias1 = get_example_color_scales(variable_name, metric_name)
        vminmax_list.append(vminmax1) # For truth
        vminmax_list.append(vminmax1) # Same for prediction
        cmap_list.append(cmap_name1)
        cmap_list.append(cmap_name1)

        # Append data to list of fields to be plotted
        field_list.append(targ_ds.values * scale)
        field_list.append(pred_ds.values * scale)

        # Append info for colorbar label
        ctitle_list.append(units)
        ctitle_list.append(units)

    lat_list = [ds[latlon_dims[0]].values] * len(field_list)
    lon_list = [ds[latlon_dims[1]].values] * len(field_list)

    out_filepath = plot_map_matrix_pcolormesh(
        lon_list, lat_list, field_list, ncols=ncols, nrows=nrows,
        vminmax=vminmax_list, filepath=out_dir + filename + ".png",
        title_list=title_list, cbartitle_list=ctitle_list,
        color_map_list=cmap_list, FIGWIDTH=None, FIGHEIGHT=None,
        panel_size=2.0, title_fs=None,
        figtitle=full_title)

    return out_filepath


# ----------------------------------------------------------------------------
def plot_truth_predict_bias_1variable_portrait(
    pre_path,
    mid_path,
    post_path,
    out_dir,
    model_dir_list,
    model_name_list,
    variable_name,
    inv_mod_var=False,
    metric_name="acorr",
    season=None,
    full_title=None,
    experiment_info=None,
    member_dim="member",
    member_num=0,
    filename=None,
    latlon_dims=["lat", "lon"],
    ncols=2,
    nrows=2,
    ):
    """
    Plot a prediction metric/bias map matrix for one variable across
    multiple models, in "portrait" panel order: for each model the
    pair (prediction metric, bias) is added as a new row, with the truth
    correlation map inserted as an extra row-pair at the very start
    (i.e. truth occupies the first two panel slots of row 1,
    alongside the first model's correlation/bias underneath).

    Parameters
    ----------
    pre_path, mid_path, post_path : str
        Path fragments used to build each model's diagnostics file
        path as `pre_path + model_dir + mid_path + variable_name +
        post_path + zarr_name` (or with `model_dir` and
        `variable_name` swapped if `inv_mod_var` is True), where
        `zarr_name` depends on `metric_name`.
    out_dir : str
        Directory the output figure is saved into (created if it
        doesn't already exist).
    model_dir_list : list of str
        Model directory names/segments, one per model.
    model_name_list : list of str
        Display names for each model, one per model (same length
        and order as `model_dir_list`).
    variable_name : str
        Name of the variable being diagnosed, used to build file
        paths and titles.
    inv_mod_var : bool, optional
        If True, swap the order of `model_dir` and `variable_name`
        when building each diagnostics file path (default False).
    metric_name : str, optional
        Which diagnostic to plot: "acorr" (auto-correlation,
        `metrics_fast_field.zarr`), "mv_pcorr" (multivariate
        Pearson correlation), or "mv_scorr" (multivariate Spearman
        correlation) - the latter two both read
        `metrics_mv_field.zarr` (default "acorr").
    season : str or None, optional
        If given, appended to the default output filename and full_title
        (default None). Not used to filter input files.
    full_title : str or None, optional
        Title for full matrix figure. If None (default), a
        title is auto-generated from the `metric_name`,
        `model_name`, member info, and `season`.
    experiment_info : str or None, optional
        Optional extra text appended to the figure title (default None).
    member_dim : str or None, optional
        Name of the ensemble-member dimension (default "member").
        Passed to `reduce_mem_dim`.
    member_num : int, optional
        Which ensemble member to plot; < 0 means ensemble mean
        (default 0).
    filename : str or None, optional
        Output filename (without extension). If None (default), a
        name is auto-generated from `metric_name`, `variable_name`,
        the number of models, member info, and `season`.
    latlon_dims : list of str, optional
        Names of the lat/lon dimensions (default ["lat", "lon"]).
    ncols : int, optional
        Number of columns in the plot matrix (default 2).
        Should be even to align (prediction, bias) columns
    nrows : int, optional
        Number of rows in the plot matrix (default 2). Should be
        large enough to fit `nmodels + 1` row-pairs given `ncols`.

    Returns
    -------
    out_filepath : str
        Full path to the saved PNG figure.

    Raises
    ------
    ValueError
        If try to plot an unsupported combination of metric_name and variable,
        e.g. p98wet for tasmax;
        If fields loaded for plotting are not correctly reduced to lat/lon dims.
        If lon_list, lat_list, and data_list are not all the same
        length, or if that length exceeds ncols*nrows.
    """

    # ncols should be even to align (prediction metric, bias) columns
    if ncols % 2 != 0:
        print(f"Warning: ncols should be even to align (prediction metric, bias) columns, but got {ncols}")


    # Reduce ensemble data to single member or ensemble mean
    mem_info = "_m0"
    if member_dim is not None:
        if member_num < 0:
            # average over the member dimension
            mem_info = "_em"
        else:
            mem_info = "_m" + str(int(member_num))

    # Make sure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    nmodels = len(model_dir_list)

    # Output filename if not given
    if filename is None:
        filename = "mapsP_" + metric_name + "_" + variable_name + "_" + str(int(nmodels)) + "models" + mem_info
        if season is not None:
            filename = filename + "_" + season

    # Default values for
    # - scaling field if necessary (e.g. psl Pa to hPa)
    # - defining units of data
    scale = 1
    units = ""

    VariableUnits = False # used to flag if metric units depend on original variable

    # Get data information for specific metric_name
    # default text for title, data units and the name of the linked zarr file
    # VariableUnits : bool, flags that units depended on prediction variable
    # variable_specific : str, None, flags if metric specific to just one variable
    title_name, units, VariableUnits, zarr_name, var_specific = get_variable_info(metric_name)

    if var_specific is not None and variable_name != var_specific:
        raise ValueError(
            f"Metric {metric_name} only suitable for "
            f"variable {var_specific}, but variable "
            f"loaded is {variable_name}"
        )

    # Overall title for matrix of plots if not given
    if full_title is None:
        full_title=title_name + " - " + variable_name
        if season is not None:
            full_title=title_name + " " +season + " - " + variable_name
        if experiment_info is not None:
            full_title = full_title + " " + experiment_info

    # Set up list of data fields and lat/lon info for plotting to matrix
    # (row-major order)
    field_list = [] # list of fields to be plotted
    title_list = [] # list of titles for each single plot
    vminmax_list = [] # list of color ranges where available else use default
    ctitle_list = [] # list of labels for colorbar (generally the units of the variable)
    cmap_list = [] # list of color maps (so can have different colors for bias if need be)

    for nn in range(nmodels):
        model_dir = model_dir_list[nn]
        model_name = model_name_list[nn]

        # Build filepath for target and prediction fields
        diagnostics_file = pre_path + model_dir + mid_path + variable_name + post_path + zarr_name
        if inv_mod_var:
            # If variable directory comes before model directory, inverse order
            diagnostics_file = pre_path + variable_name + mid_path + model_dir + post_path + zarr_name

        ds = xr.open_zarr(diagnostics_file)

        if nn == 0:
            # Only load target field once
            targ_ds = ds['value'].sel(diagnostic="targ_" + metric_name + "_field").load()

        # Load prediction and bias fields
        pred_ds = ds['value'].sel(diagnostic="pred_" + metric_name + "_field").load()
        bias_ds = ds['value'].sel(diagnostic="bias_" + metric_name + "_field").load()

        # Check if member dimension, if so pick correct member
        # Reduce ensemble data to single member or ensemble mean
        if nn == 0:
            targ_ds, _t_member_num = reduce_mem_dim(targ_ds, member_dim, 0)
        pred_ds, member_num = reduce_mem_dim(pred_ds, member_dim, member_num)
        bias_ds, member_num = reduce_mem_dim(bias_ds, member_dim, member_num)

        # Final check: dataset should now be a 2D lat/lon field (no member/time/etc left)
        # - Raises error if not
        if nn == 0:
            check_latlon_field(targ_ds, latlon_dims)
        check_latlon_field(pred_ds, latlon_dims)
        check_latlon_field(bias_ds, latlon_dims)

        # Plot Maps of field metrics
        # Get units from zarr file or guess if need be
        if VariableUnits:
            scale, units = get_units_scale(ds, variable_name, is_zarr=True)

        # Load defined colorscales if available
        vminmax1, cmap_name1, vminmax_bias1, cmap_name_bias1 = get_example_color_scales(variable_name, metric_name)

        # - Append text for single plot title
        # - Append info for colorbar label
        # - Load defined colorscales if available

        if nn == 0:
            # Truth correlation map gets its own row-pair slot up front
            title_list.append("Truth") # plot titles
            title_list.append(None)
            cmap_list.append(plt.get_cmap(cmap_name1)) # plot color maps
            cmap_list.append(None)
            vminmax_list.append(vminmax1) # plot color scale
            vminmax_list.append([None, None])
            ctitle_list.append(units)
            ctitle_list.append(None)

        title_list.append(model_name) # plot titles
        title_list.append("bias: " + model_name)
        cmap_list.append(plt.get_cmap(cmap_name1)) # plot color maps
        cmap_list.append(plt.get_cmap(cmap_name_bias1))
        vminmax_list.append(vminmax1) # plot color scale
        vminmax_list.append(vminmax_bias1)
        ctitle_list.append(None)
        ctitle_list.append(None)

        if nn == 0:
            field_list.append(targ_ds.values * scale)
            field_list.append(None)

        field_list.append(pred_ds.values * scale)
        field_list.append(bias_ds.values * scale)

    lat_list = [ds[latlon_dims[0]].values] * len(field_list)
    lon_list = [ds[latlon_dims[1]].values] * len(field_list)

    out_filepath = plot_map_matrix_pcolormesh(
        lon_list, lat_list, field_list, ncols=ncols, nrows=nrows,
        vminmax=vminmax_list, filepath=out_dir + filename + ".png",
        title_list=title_list, cbartitle_list=ctitle_list,
        color_map_list=cmap_list, FIGWIDTH=None, FIGHEIGHT=None,
        panel_size=2.0, title_fs=None,
        figtitle=full_title)

    return out_filepath


# ----------------------------------------------------------------------------
def plot_truth_predict_bias_1variable_landscape(
    pre_path,
    mid_path,
    post_path,
    out_dir,
    model_dir_list,
    model_name_list,
    variable_name,
    inv_mod_var=False,
    metric_name="acorr",
    season=None,
    full_title=None,
    experiment_info=None,
    member_dim="member",
    member_num=0,
    filename=None,
    latlon_dims=["lat", "lon"],
    ncols=3,
    nrows=1,
    ):
    """
    Plot a correlation/bias map matrix for one variable across
    multiple models, in "landscape" panel order: the truth
    correlation map is added as its own leading panel, then each
    model contributes its (correlation, bias) pair as the next two
    panels in the same row-major sequence (rather than starting a
    new row per model, as in `plot_corr_bias_1variable_portrait`).

    Parameters
    ----------
    pre_path, mid_path, post_path : str
        Path fragments used to build each model's diagnostics file
        path as `pre_path + model_dir + mid_path + variable_name +
        post_path + zarr_name` (or with `model_dir` and
        `variable_name` swapped if `inv_mod_var` is True), where
        `zarr_name` depends on `metric_name`.
    out_dir : str
        Directory the output figure is saved into (created if it
        doesn't already exist).
    model_dir_list : list of str
        Model directory names/segments, one per model.
    model_name_list : list of str
        Display names for each model, one per model (same length
        and order as `model_dir_list`).
    variable_name : str
        Name of the variable being diagnosed, used to build file
        paths and titles.
    inv_mod_var : bool, optional
        If True, swap the order of `model_dir` and `variable_name`
        when building each diagnostics file path (default False).
    metric_name : str, optional
        Which diagnostic to plot: "acorr" (auto-correlation,
        `metrics_fast_field.zarr`), "mv_pcorr" (multivariate
        Pearson correlation), or "mv_scorr" (multivariate Spearman
        correlation) - the latter two both read
        `metrics_mv_field.zarr` (default "acorr").
    season : str or None, optional
        If given, appended to the default filename (default None).
    full_title : str or None, optional
        Title for full matrix figure. If None (default), a
        title is auto-generated from the `metric_name`,
        `model_name`, member info, and `season`.
    experiment_info : str or None, optional
        Optional extra text appended to the figure title (default None).
    member_dim : str or None, optional
        Name of the ensemble-member dimension (default "member").
        Passed to `reduce_mem_dim`.
    member_num : int, optional
        Which ensemble member to plot; < 0 means ensemble mean
        (default 0).
    filename : str or None, optional
        Output filename (without extension). If None (default), a
        name is auto-generated from `metric_name`, `variable_name`,
        the number of models, member info, and `season`.
    latlon_dims : list of str, optional
        Names of the lat/lon dimensions (default ["lat", "lon"]).
    ncols : int, optional
        Number of columns in the plot matrix (default 3).
        Should be odd to align (target, prediction metric, bias) columns
    nrows : int, optional
        Number of rows in the plot matrix (default 1). Should be
        large enough to fit `2*nmodels + 1` panels given `ncols`.

    Returns
    -------
    out_filepath : str
        Full path to the saved PNG figure.

    Raises
    ------
    ValueError
        If try to plot an unsupported combination of metric_name and variable,
        e.g. p98wet for tasmax;
        If fields loaded for plotting are not correctly reduced to lat/lon dims.
        If lon_list, lat_list, and data_list are not all the same
        length, or if that length exceeds ncols*nrows.
    """

    # ncols should be odd to align (target, prediction metric, bias) columns
    if ncols % 2 != 1:
        print(f"Warning: ncols should be odd to align (target, prediction metric, bias) columns, but got {ncols}")

    # Reduce ensemble data to single member or ensemble mean
    mem_info = "_m0"
    if member_dim is not None:
        if member_num < 0:
            # average over the member dimension
            mem_info = "_em"
        else:
            mem_info = "_m" + str(int(member_num))

    # Make sure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    nmodels = len(model_dir_list)

    # Output filename if not given
    if filename is None:
        filename = "mapsL_" + metric_name + "_" + variable_name + "_" + str(int(nmodels)) + "models" + mem_info
        if season is not None:
            filename = filename + "_" + season

    # Default values for
    # - scaling field if necessary (e.g. psl Pa to hPa)
    # - defining units of data
    scale = 1
    units = ""

    VariableUnits = False # used to flag if metric units depend on original variable

    # Get data information for specific metric_name
    # default text for title, data units and the name of the linked zarr file
    # VariableUnits : bool, flags that units depended on prediction variable
    # variable_specific : str, None, flags if metric specific to just one variable
    title_name, units, VariableUnits, zarr_name, var_specific = get_variable_info(metric_name)

    if var_specific is not None and variable_name != var_specific:
        raise ValueError(
            f"Metric {metric_name} only suitable for "
            f"variable {var_specific}, but variable "
            f"loaded is {variable_name}"
        )

    # Overall title for matrix of plots if not given
    if full_title is None:
        full_title=title_name + " - " + variable_name
        if season is not None:
            full_title=title_name + " " +season + " - " + variable_name
        if experiment_info is not None:
            full_title = full_title + " " + experiment_info

    # Set up list of data fields and lat/lon info for plotting to matrix
    # (row-major order)
    field_list = [] # list of fields to be plotted
    title_list = [] # list of titles for each single plot
    vminmax_list = [] # list of color ranges where available else use default
    ctitle_list = [] # list of labels for colorbar (generally the units of the variable)
    cmap_list = [] # list of color maps (so can have different colors for bias if need be)

    for nn in range(nmodels):
        model_dir = model_dir_list[nn]
        model_name = model_name_list[nn]

        # Build filepath for target and prediction fields
        diagnostics_file = pre_path + model_dir + mid_path + variable_name + post_path + zarr_name
        if inv_mod_var:
            # If variable directory comes before model directory, inverse order
            diagnostics_file = pre_path + variable_name + mid_path + model_dir + post_path + zarr_name

        # Load data
        ds = xr.open_zarr(diagnostics_file)

        if nn == 0:
            # Only load target field once
            targ_ds = ds['value'].sel(diagnostic="targ_" + metric_name + "_field").load()

        # Load prediction and bias fields
        pred_ds = ds['value'].sel(diagnostic="pred_" + metric_name + "_field").load()
        bias_ds = ds['value'].sel(diagnostic="bias_" + metric_name + "_field").load()

        # Check if member dimension, if so pick correct member
        # Reduce ensemble data to single member or ensemble mean
        if nn == 0:
            targ_ds, _t_member_num = reduce_mem_dim(targ_ds, member_dim, 0)
        pred_ds, member_num = reduce_mem_dim(pred_ds, member_dim, member_num)
        bias_ds, member_num = reduce_mem_dim(bias_ds, member_dim, member_num)

        # Final check: dataset should now be a 2D lat/lon field (no member/time/etc left)
        # - Raises error if not
        if nn == 0:
            check_latlon_field(targ_ds, latlon_dims)
        check_latlon_field(pred_ds, latlon_dims)
        check_latlon_field(bias_ds, latlon_dims)

        # Plot Maps of field metrics
        # Get units from zarr file or guess if need be
        if VariableUnits:
            scale, units = get_units_scale(ds, variable_name, is_zarr=True)

        # Load defined colorscales if available
        vminmax1, cmap_name1, vminmax_bias1, cmap_name_bias1 = get_example_color_scales(variable_name, metric_name)

        # - Append text for single plot title
        # - Append info for colorbar label
        # - Load defined colorscales if available

        if nn == 0:
            # Truth map gets its own column slot up front
            title_list.append("Truth")
            cmap_list.append(plt.get_cmap(cmap_name1))
            vminmax_list.append(vminmax1)
            ctitle_list.append(units)

        if nn > 0:
            title_list.append(None)
            cmap_list.append(None)
            vminmax_list.append([None, None])
            ctitle_list.append(None)

        title_list.append(model_name)
        title_list.append("bias: " + model_name)
        cmap_list.append(plt.get_cmap(cmap_name1))
        cmap_list.append(plt.get_cmap(cmap_name_bias1))
        vminmax_list.append(vminmax1)
        vminmax_list.append(vminmax_bias1)
        ctitle_list.append(None) # None or repeat units for every single plot?
        ctitle_list.append(None)

        # Plot Maps of field metrics
        if nn == 0:
            field_list.append(targ_ds.values * scale) # Truth just once
        if nn > 0:
            field_list.append(None)

        field_list.append(pred_ds.values * scale) # Prediction
        field_list.append(bias_ds.values * scale) # Prediction bias

    lat_list = [ds[latlon_dims[0]].values] * len(field_list)
    lon_list = [ds[latlon_dims[1]].values] * len(field_list)

    out_filepath = plot_map_matrix_pcolormesh(
        lon_list, lat_list, field_list, ncols=ncols, nrows=nrows,
        vminmax=vminmax_list, filepath=out_dir + filename + ".png",
        title_list=title_list, cbartitle_list=ctitle_list,
        color_map_list=cmap_list, FIGWIDTH=None, FIGHEIGHT=None,
        panel_size=2.0, title_fs=None,
        figtitle=full_title)

    return out_filepath

# ----------------------------------------------------------------------------