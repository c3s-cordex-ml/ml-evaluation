import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg') # Improves plotting efficiency. Remove if want to pause code and make plots within run.
import matplotlib.pyplot as plt

from ml_evaluation.viz.plot_hexagon import metrics_dir

# Histogram store (written by ``ml-eval slow``) inside each ``<path>/<var>/`` dir.
HIST_STORE = "metrics_slow_hist.zarr"

# -----------------------------------------------------------------------------
def plot_pdf_multi_model_Mvar(
    input_dir_list,
    input_target_pred,
    input_member,
    legend_list,
    color_list,
    variable_list,
    out_dir=None,
    out_fname=None,
    season="ANN",
    incl_diag=None,
    ymin=None,
    log_scale=True):

    """
    Plot histograms (pdfs) straight from zarr files.
    As in ``plot_var_skill_scorecard`` / ``plot_hexagon``, each entry of
    input_dir_list is a model's base directory and one variable's histogram
    store is read from ``<path>/<var>/metrics_slow_hist.zarr``.

    One subplot is drawn per variable (2 columns, as many rows as needed;
    a single panel if only one variable). Within each subplot, one histogram
    line is drawn per entry in input_dir_list. The figure is saved as a PNG
    to os.path.join(out_dir, out_fname + ".png").

    Parameters
    ----------
    input_dir_list : list of str
        Model base directories, one entry per line to plot. For each variable
        the histogram store is read from
        ``os.path.join(path, variable, "metrics_slow_hist.zarr")`` (see
        ``plot_hexagon.metrics_dir``). A trailing "/" is optional.
    input_target_pred : list of str
        One entry per input_dir_list entry, either 'TARG' (plot the target /
        ground truth histogram, diagnostic 'targ_hist_series') or 'PRED' (plot
        the model prediction histogram, diagnostic 'pred_hist_series').
    input_member : list
        One entry per input_dir_list entry giving the ensemble member to
        select from each zarr file (passed to ds.sel(member=...)).
    legend_list : list of str
        One entry per input_dir_list entry giving the legend label for each
        line.
    color_list : list of str
        One entry per input_dir_list entry giving the matplotlib colour for
        each line.
    variable_list : list of str
        Variables to plot, one subplot each, e.g. ['pr', 'tasmax'].
        Recognised variables ('pr', 'tas', 'tasmax', 'tasmin', 'uas', 'vas',
        'psl', 'hurs') get unit conversion and x-axis labels; 'pr' is
        converted from kg m-2 s-1 to mm/day and 'psl' from Pa to hPa.
        Each variable must have a ``<path>/<variable>/`` subdirectory under
        every entry of input_dir_list.
    out_dir : str, optional
        Directory in which to save the figure; created (including any parent
        directories) if it does not exist. A trailing "/" is optional. If None
        or not a string (default None), the figure is saved in the current
        working directory.
    out_fname : str, optional
        Output filename without the ".png" extension. If None (default), it
        is built as "pdf_hist_<season>_<N>var", with "_<variable>" appended
        when only one variable is plotted. In both cases "_<incl_diag>" is
        appended when incl_diag is set.
    season : str, optional
        Season label used only in the default output filename (no seasonal
        selection is applied to the data). Default "ANN".
    incl_diag : {'wdh', 'fcss', 'pss', 'lhd'} or None, optional
        Summary diagnostic to append to the legend label of each 'PRED' line
        (read from '<incl_diag>_float' in the zarr file). Any other value is
        treated as None. 'wdh' is scaled into the plotted units. Default None.
    ymin : float, optional
        Lower limit of the y-axis (density). Default None (matplotlib
        autoscale).
    log_scale : bool, optional
        If True (default), use a logarithmic y-axis.

    Returns
    -------
    None
        The figure is written to disk and closed.

    Raises
    ------
    ValueError
        If any constructed zarr store path does not exist (the message names
        the missing path).

    # Example input parameters
    input_target_pred=['TARG','PRED','PRED']
    input_member=[0,0,0]    
    legend_list=['Ground truth', 'UNET-MULTI', 'RESDIFF']
    color_list=['black','blue','red']

    out_dir='/home/historical_perfect/'
    input_dir_list=['/home/historical_perfect/UNET-MULTI',
                    '/home/historical_perfect/UNET-MULTI',
                    '/home/historical_perfect/RESDIFF']

    variable_list=['pr','tasmax']

    """
    
    num_variables=len(variable_list)

    # Default to the current working directory if no valid out_dir given
    if not isinstance(out_dir,str):
        out_dir=""

    print(out_dir)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not isinstance(out_fname,str):
        out_fname=None

    if isinstance(out_fname,type(None)):
        out_fname="pdf_hist_"+season+"_"+str(int(num_variables))+"var"
        if num_variables==1: out_fname=out_fname+"_"+variable_list[0]

    if not incl_diag in {'wdh','fcss','pss','lhd'}:
        incl_diag=None

    if not isinstance(incl_diag,type(None)):
        out_fname=out_fname+"_"+incl_diag

    nrows=int(np.ceil(num_variables/2))
    ncols=2
    figure_size=(8,10)
    
    if num_variables==1:
        nrows=1
        ncols=1
        figure_size=(5,4)

    fig, axes = plt.subplots(nrows, ncols, figsize=figure_size,
        layout='constrained', squeeze=False)

    for jj in range(num_variables):
        variable=variable_list[jj]

        shift=0.0
        scale=1.0
        units=""

        if variable=="pr":
            scale=86400.0
            units="mm/day"

        if variable=="tas" or variable=="tasmax" or variable=="tasmin":
            units="K"

        if variable=="uas" or variable=="vas":
            units="m/s"

        if variable=="psl":
            scale=1/100.
            units="hPa"

        if variable=="hurs":
           units="%"

        ax=axes.flatten()[jj]
        num_hist=len(input_dir_list)
        for ii in range(num_hist):
            filename1=os.path.join(metrics_dir(input_dir_list[ii], variable), HIST_STORE)
            print(filename1)
            if not os.path.isdir(filename1):
                raise ValueError(f"zarr store not found: {filename1}")
            ds1 = xr.open_zarr(filename1)
            ds1 = ds1.sel(member=input_member[ii])
            ds1_legend=f"{legend_list[ii]}"

            if input_target_pred[ii]=='TARG':
                ds1_hist=ds1['value'].sel(diagnostic='targ_hist_series').load().values[0:-1]
	
            if input_target_pred[ii]=='PRED':
                ds1_hist=ds1['value'].sel(diagnostic='pred_hist_series').load().values[0:-1]
                if incl_diag=='wdh': ds1_legend=f"{ds1_legend}, WDh={ds1['value'].sel(diagnostic='wdh_float').load().values[0]*scale:.3f}"
                if incl_diag=='fcss': ds1_legend=f"{ds1_legend}, FCSS={ds1['value'].sel(diagnostic='fcss_float').load().values[0]:.3f}"
                if incl_diag=='pss': ds1_legend=f"{ds1_legend}, PSS={ds1['value'].sel(diagnostic='pss_float').load().values[0]:.3f}"
                if incl_diag=='lhd': ds1_legend=f"{ds1_legend}, LHD={ds1['value'].sel(diagnostic='lhd_float').load().values[0]:.3f}"

            ds1_binedges=ds1['value'].sel(diagnostic='bins_hist_series').load().values*scale

            ax.stairs(ds1_hist, ds1_binedges, label=ds1_legend, linewidth=1, color=color_list[ii])

        if log_scale:
            ax.set_yscale('log')
        ax.set_ylim([ymin,None])
        ax.set_title(variable)
        ax.set_xlabel(units)
        ax.set_ylabel('Density')
        ax.legend()

    plt.savefig(os.path.join(out_dir, out_fname+".png"))
    plt.close('all')
