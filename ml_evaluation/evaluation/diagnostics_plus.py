from __future__ import annotations

from typing import Any, Callable

import numpy as np
import xarray as xr
# from tqdm import tqdm # taken out so don't need to install tqdm

from numpy import fft
from scipy.stats import wasserstein_distance_nd
from scipy.stats import wasserstein_distance
from scipy.stats import rankdata
from scipy.spatial.distance import jensenshannon


# -------------------------------------------------------------------------------------
def _filter_by_monthseason(x: xr.Dataset, season: str | None) -> xr.Dataset:
    """
    Filter the dataset by season or month label.
    Added by R. Eade.
    Adapted from ml-benchmark diagnostics.py _filter_by_season()
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

    MON_LIST = [
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    ]
    SEAS_LIST = ["DJF", "MAM", "JJA", "SON"]

    if season not in MON_LIST and season not in SEAS_LIST:
        season = None
    if season is None:
        return x

    if season in SEAS_LIST:
        return x.where(x["time.season"] == season, drop=True)

    if season in MON_LIST:
        month_num = MON_LIST.index(season) + 1
        return x.where(x["time.month"] == month_num, drop=True)

    return x


# -------------------------------------------------------------------------------------
def calculate_error_nparray(arr0, arr1, axis=0, typeerror="rmse"):
    """
    Calculate (rms) error between 2 N-dimensional numpy arrays
    Added by R. Eade.

    Parameters:
    -----------
    arr0 : numpy.ndarray
        First N-dimensional array (Target)
    arr1 : numpy.ndarray
        Second N-dimensional array (must have same shape as arr0) (Prediction)
    axis : int, default=0
        Axis/axes along which to compute error
    typeerror : str, default='rmse'
        Type of error to compute, options ('rmse', 'bias', 'abias', 'ratio')

    Returns:
    --------
    numpy.ndarray
        Error. Output has N - len(axis) dimensions
        (input shape with the specified axis/axes removed).

    """

    if arr0.shape != arr1.shape:
        raise ValueError(
            f"Arrays must have the same shape. Got {arr0.shape} and {arr1.shape}"
        )

    # Compute Error over axis/axes specified
    # - subtract first then take mean so consistent method for bias and rmse
    # - better level of accuracy than taking means then subtract (ie slight difference)
    if typeerror == "bias":
        output_error = (arr1 - arr0).mean(axis=axis)
    if typeerror == "abias":
        output_error = np.abs(arr1 - arr0).mean(axis=axis)
    if typeerror == "rmse":
        output_error = np.sqrt(((arr1 - arr0) ** 2).mean(axis=axis))
    if typeerror == "ratio":
        output_error = arr1.mean(axis=axis) / arr0.mean(axis=axis)

    return output_error


# -------------------------------------------------------------------------------------
def calculate_pearsoncorr_nparray(arr0, arr1, axis=0):
    """
    Calculate Pearson correlation between 2 N-dimensional numpy arrays
    Added by R. Eade.

    Parameters:
    -----------
    arr0 : numpy.ndarray
        First N-dimensional array (Target)
    arr1 : numpy.ndarray
        Second N-dimensional array (must have same shape as arr0) (Prediction)
    axis : int or tuple of int, default=0
        Axis or tuple of axes over which to compute correlation
        None : if want to compute over all axes

    Returns:
    --------
    numpy.ndarray
        Pearson correlation coefficients. Output has N - len(axis) dimensions
        (input shape with the specified axis/axes removed).
        Or single value if axis=None.

    """

    if arr0.shape != arr1.shape:
        raise ValueError(
            f"Arrays must have the same shape. Got {arr0.shape} and {arr1.shape}"
        )

    # Center the data over axis/axes specified
    arr0_centered = arr0 - arr0.mean(axis=axis, keepdims=True)
    arr1_centered = arr1 - arr1.mean(axis=axis, keepdims=True)

    # Compute correlation over axis/axes specified
    numerator = (arr0_centered * arr1_centered).sum(axis=axis)
    denominator = np.sqrt(
        (arr0_centered**2).sum(axis=axis) * (arr1_centered**2).sum(axis=axis)
    )

    # Avoid division by zero (set as 0.0 instead of inf or nan)
    correlations = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )

    return correlations


# -------------------------------------------------------------------------------------
def rank_array(arr0, axis=0):
    """
    Calculate ranks using scipy.stats.rankdata along given axis.
    Extended for tuple of axes by first flattening data along these axes.
    Added by R. Eade.

    Parameters:
    -----------
    arr0 : numpy.ndarray
        N-dimensional array
    axis : int or tuple of int, default=0
        Axis or tuple of axes over which to compute correlation
        None : if want to compute over all axes

    Returns:
    --------
    numpy.ndarray
        Ranks of data. Output has N - len(axis) + 1 dimensions
        For axis=None, Output has 1 dimension.

    """

    if isinstance(axis, type(None)):
        arr0 = rankdata(arr0, method="average", axis=axis, nan_policy="propagate")
        return arr0

    if isinstance(axis, int):
        axis = (axis,)

    if isinstance(axis, tuple):
        # reorder array so flatten axes in axis
        new_axis = list(range(0, len(axis)))
        arr0 = np.moveaxis(arr0, axis, new_axis)
        arr0 = arr0.reshape(
            np.prod(arr0.shape[0 : len(axis)]), *arr0.shape[len(axis) :]
        )
        arr0 = rankdata(arr0, method="average", axis=0, nan_policy="propagate")
        return arr0

    raise ValueError(f"axis must be None, int or tuple of int. Got {axis}")


# -------------------------------------------------------------------------------------
def calculate_spearmancorr_nparray(arr0, arr1, axis=0):
    """
    Calculate Spearman's Rank correlation between 2 N-dimensional numpy arrays
    Added by R. Eade.

    Parameters:
    -----------
    arr0 : numpy.ndarray
        First N-dimensional array (Target)
    arr1 : numpy.ndarray
        Second N-dimensional array (must have same shape as arr0) (Prediction)
    axis : int or tuple of int, default=0
        Axis or tuple of axes over which to compute correlation
        None : if want to compute over all axes

    Returns:
    --------
    numpy.ndarray
        Spearman correlation coefficients. Output has N - len(axis) dimensions
        (input shape with the specified axis/axes removed).
        Or single value if axis=None.

    """

    if arr0.shape != arr1.shape:
        raise ValueError(
            f"Arrays must have the same shape. Got {arr0.shape} and {arr1.shape}"
        )

    # Convert data to ranks
    # - using sp.stats.rankdata with xarray rank defaults
    arr0 = rank_array(arr0, axis=axis)
    arr1 = rank_array(arr1, axis=axis)

    # Compute Pearson correlation on ranked data
    correlations = calculate_pearsoncorr_nparray(arr0, arr1, axis=0)

    return correlations


# -------------------------------------------------------------------------------------
def calculate_corr_xrnp(da0, da1, dim=None, corrtype="Pearson"):
    """
    Compute the Correlation between da0 and da1 DataArrays.
    Default is Pearson's corr. Alternative option for Spearman's rank corr.
    Calls numpy version of correlation.
    Quicker than xarray for moderate sized datasets.

    Parameters
    ----------
    da0 : xr.DataArray
        Ground truth DataArray.
    da1 : xr.DataArray
        Predicted DataArray.
    dim : str | None, optional
        Dimension(s) along which to compute the RMSE. If None, computes over all dimensions.
    corrtype : str
        Define typr of correlation: 'Pearson' or 'Spearman'

    Returns
    -------
    xr.DataArray
        Pearson Correlation between da0 and da1.
        Output has N - len(dim) dimensions
        (input shape with the specified dimension(s) removed).
        Or single value if dim=None

    """

    corr_list = ["Pearson", "Spearman"]
    if corrtype not in corr_list:
        raise ValueError(f"corrtype not recognised: {corrtype}")

    if isinstance(dim, str):
        # Convert string to tuple
        dim = (dim,)

    if isinstance(dim, tuple):
        dim_list = list(da0.dims)
        matches = [dim1 for dim1 in dim_list if dim1 in dim]
        if not matches:
            raise ValueError(f"Warning: no matches found for {list(dim)} in {dim_list}")
        axis_num = tuple(dim_list.index(dim1) for dim1 in matches)

    if isinstance(dim, type(None)):
        axis_num = None

    pcor_xarray = da0.mean(dim=dim) * 0.0

    if corrtype == "Pearson":
        pcor_result = calculate_pearsoncorr_nparray(
            da0.values, da1.values, axis=axis_num
        )
    if corrtype == "Spearman":
        pcor_result = calculate_spearmancorr_nparray(
            da0.values, da1.values, axis=axis_num
        )

    pcor_xarray = pcor_xarray + pcor_result

    return pcor_xarray


# -------------------------------------------------------------------------------------
def bias_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    index_fn: Callable[..., xr.DataArray],
    season: str | None = None,
    relative: bool = False,
    **index_kwargs: Any,
) -> xr.Dataset:
    """
    Compute the bias between x0 and x1 for a supplied index function.
    From ml-benchmark diagnostics.py.
    - uses updated season filter _filter_by_monthseason via index_fn

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    index_fn : Callable[..., xr.DataArray]
        Function computing the desired index.
    season : str | None, optional
        Season to filter before computing the index.
    relative : bool, default False
        If True, return the relative bias.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.
        e.g. var="pr" where index function needs choice of climate variable
        e.g. cmean=True (compute climate mean over time) is needed for index
        functions which provide annual output e.g. indices_plus.ann_extrema()
        See indices_plus.py for more details on specific functions.

    Returns
    -------
    xr.Dataset
        Absolute or relative bias of the selected index.
    """

    call_kwargs = dict(index_kwargs)
    if season is not None and "season" not in call_kwargs:
        call_kwargs["season"] = season

    x0_index = index_fn(x0, **call_kwargs)
    x1_index = index_fn(x1, **call_kwargs)
    bias = x1_index - x0_index

    if relative:
        return bias / x0_index
    else:
        return bias


# -------------------------------------------------------------------------------------
def ratio_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    index_fn: Callable[..., xr.DataArray],
    season: str | None = None,
    **index_kwargs: Any,
) -> xr.Dataset:
    """
    Compute the ratio between x1 and x0 for a supplied index function.
    From ml-benchmark diagnostics.py.
    - uses updated season filter _filter_by_monthseason via index_fn

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    index_fn : Callable[..., xr.DataArray]
        Function computing the desired index.
    season : str | None, optional
        Season to filter before computing the index.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.
        e.g. var="pr" where index function needs choice of climate variable
        e.g. cmean=True (compute climate mean over time) is needed for index
        functions which provide annual output e.g. indices_plus.ann_extrema()
        See indices_plus.py for more details on specific functions.

    Returns
    -------
    xr.Dataset
        Ratio of the selected index (x1 / x0).

    """

    call_kwargs = dict(index_kwargs)
    if season is not None and "season" not in call_kwargs:
        call_kwargs["season"] = season

    x0_index = index_fn(x0, **call_kwargs)
    x1_index = index_fn(x1, **call_kwargs)
    ratio = x1_index / x0_index
    return ratio


# -------------------------------------------------------------------------------------
def rmse_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    dim: str | None = None,
    index_fn: Callable[..., xr.DataArray] | None = None,
    season: str | None = None,
    mae: bool = False,
    mse: bool = False,
    **index_kwargs: Any,
) -> xr.Dataset:
    """
    Compute the RMSE between x0 and x1 for a supplied index function.
    Alternative options for MAE or MSE.
    Added by R. Eade.
    Adapted from ml-benchmark diagnostics.py rmse() and bias_index()
    - changed season filter
    - added options for alternative error types
    - works on annual index functions such as ann_extrema()

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    dim : str | None, optional
        Dimension(s) along which to compute the RMSE.
        If None, computes over all dimensions.
    index_fn : Callable[..., xr.DataArray]
        Function computing the desired index. (None = Raw data)
    season : str | None, optional
        Season to filter before computing the index.
    mae : bool, default False
        If True, return the mean absolute error.
    mse : bool, default False
        If True, return the mean squared error.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.
        e.g. cmean=False (retain annual output) is needed for index
        functions which provide annual output e.g. indices_plus.ann_extrema()
        See indices_plus.py for more details on specific functions.

    Returns
    -------
    xr.Dataset
        RMSE for the selected index (or MAE or MSE).

    """

    rmse = True
    if mae or mse:
        rmse = False

    if mae and mse:
        raise ValueError("Warning: mae and mse can't both be True")

    if index_fn is None:
        x0_index = _filter_by_monthseason(x0, season)[[var]]
        x1_index = _filter_by_monthseason(x1, season)[[var]]

    else:
        call_kwargs = dict(index_kwargs)
        if season is not None and "season" not in call_kwargs:
            call_kwargs["season"] = season
        if var is not None and "var" not in call_kwargs:
            call_kwargs["var"] = var

        x0_index = index_fn(x0, **call_kwargs)
        x1_index = index_fn(x1, **call_kwargs)

    diff = x1_index - x0_index

    if dim is None:
        if rmse or mse:
            error_result = (diff**2).mean()
        if mae:
            error_result = abs(diff).mean()
    else:
        if rmse or mse:
            error_result = (diff**2).mean(dim=dim)
        if mae:
            error_result = abs(diff).mean(dim=dim)

    if rmse:
        error_result = np.sqrt(error_result)

    return error_result


# -------------------------------------------------------------------------------------
def corr_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    dim: str | None = None,
    index_fn: Callable[..., xr.DataArray] | None = None,
    season: str | None = None,
    srank: bool = False,
    npcorr: bool = False,
    **index_kwargs: Any,
) -> xr.Dataset:
    """
    Compute the Correlation between x0 and x1 for a supplied index function.
    Default is Pearson's corr. Alternative option for Spearman's rank corr.
    Added by R. Eade.
    Adapted from ml-benchmark diagnostics.py rmse() and bias_index()
    - changed season filter
    - options for alternative correlation types (srank)
    - option for numpy version of correlation instead of xarray (npcorr)
    - works on annual index functions such as ann_extrema()

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    dim : str | None, optional
        Dimension(s) along which to compute the correlation.
        If None, computes over all dimensions.
    index_fn : Callable[..., xr.DataArray]
        Function computing the desired index. (None = Raw data)
    season : str | None, optional
        Season to filter before computing the index.
    srank : bool, default False
        If True, return the Spearman's rank corr instead.
    npcorr : bool, default False
        If True, use numpy version correlation instead of xarray.
        NB for Spearman, currently only have numpy option
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.
        e.g. cmean=False (retain annual output) is needed for index
        functions which provide annual output e.g. indices_plus.ann_extrema()
        See indices_plus.py for more details on specific functions.

    Returns
    -------
    xr.Dataset
        RMSE for the selected index (or MAE or MSE).

    """

    pcorr = True
    if srank:
        pcorr = False
    if srank:
        npcorr = True

    if index_fn is None:
        x0_index = _filter_by_monthseason(x0, season)[[var]]
        x1_index = _filter_by_monthseason(x1, season)[[var]]

    else:
        call_kwargs = dict(index_kwargs)
        if season is not None and "season" not in call_kwargs:
            call_kwargs["season"] = season
        if var is not None and "var" not in call_kwargs:
            call_kwargs["var"] = var

        x0_index = index_fn(x0, **call_kwargs)
        x1_index = index_fn(x1, **call_kwargs)

    corr_result = x0_index.mean(dim=dim) * 0.0

    # xr.corr requires DataArray
    if pcorr and npcorr:
        corr_result = corr_result + calculate_corr_xrnp(
            x1_index[var], x0_index[var], dim=dim, corrtype="Pearson"
        )
    if pcorr and not (npcorr):
        corr_result = corr_result + xr.corr(x1_index[var], x0_index[var], dim=dim)
    if srank and npcorr:
        corr_result = corr_result + calculate_corr_xrnp(
            x1_index[var], x0_index[var], dim=dim, corrtype="Spearman"
        )

    return corr_result


# -------------------------------------------------------------------------------------
def bias_multivariable_correlation(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var_x: str,
    var_y: str,
    season: str | None = None,
    srank: bool = False,
    npcorr: bool = False,
) -> xr.Dataset:
    """
    Compute correlations between two variables for x0 and x1 datasets
    and return the correlation bias.
    Added by R. Eade.
    Adapted from ml-benchmark diagnostics.py bias_multivariable_correlation()
    - changed season filter
    - options for alternative correlation types (srank)
    - option for numpy version of correlation instead of xarray (npcorr)

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset containing ``var_x`` and ``var_y``.
    x1 : xr.Dataset
        Predicted dataset containing ``var_x`` and ``var_y``.
    var_x : str
        Name of the first variable used in the correlation computation.
    var_y : str
        Name of the second variable used in the correlation computation.
    season : str | None, optional
        Season to filter before computing the correlations.
    srank : bool, default False
        If True, return the Spearman's rank corr instead.
    npcorr : bool, default False
        If True, use numpy version correlation instead of xarray.
        NB for Spearman, currently only have numpy option

    Returns
    -------
    xr.Dataset
        Dataset containing the x0 correlation, x1 correlation, signed
        correlation difference, and the requested correlation bias.

    """

    pcorr = True
    if srank:
        pcorr = False
    if srank:
        npcorr = True

    x0_filtered = _filter_by_monthseason(x0, season)
    x1_filtered = _filter_by_monthseason(x1, season)

    dim = ("time",)

    dim_list = list(x0_filtered.dims)
    matches = [dim1 for dim1 in dim_list if dim1 in dim]
    if not matches:
        raise ValueError(f"Warning: no matches found for {list(dim)} in {dim_list}")

    x0_corr_xarray = x0_filtered[var_x].mean(dim=dim) * 0.0
    x1_corr_xarray = x1_filtered[var_x].mean(dim=dim) * 0.0

    # xr.corr requires DataArray
    if pcorr and npcorr:
        x0_corr_xarray = x0_corr_xarray + calculate_corr_xrnp(
            x0_filtered[var_x], x0_filtered[var_y], dim=dim, corrtype="Pearson"
        )
        x1_corr_xarray = x1_corr_xarray + calculate_corr_xrnp(
            x1_filtered[var_x], x1_filtered[var_y], dim=dim, corrtype="Pearson"
        )

    if pcorr and not (npcorr):
        x0_corr_xarray = x0_corr_xarray + xr.corr(
            x0_filtered[var_x], x0_filtered[var_y], dim=dim
        )
        x1_corr_xarray = x1_corr_xarray + xr.corr(
            x1_filtered[var_x], x1_filtered[var_y], dim=dim
        )

    if srank and npcorr:
        x0_corr_xarray = x0_corr_xarray + calculate_corr_xrnp(
            x0_filtered[var_x], x0_filtered[var_y], dim=dim, corrtype="Spearman"
        )
        x1_corr_xarray = x1_corr_xarray + calculate_corr_xrnp(
            x1_filtered[var_x], x1_filtered[var_y], dim=dim, corrtype="Spearman"
        )

    correlation_bias = x1_corr_xarray - x0_corr_xarray

    return xr.Dataset(
        {
            "correlation_x0": x0_corr_xarray,
            "correlation_x1": x1_corr_xarray,
            "correlation_bias": correlation_bias,
        }
    )


# -------------------------------------------------------------------------------------
def calc_pdf_metric_nparray(
    arr0,
    arr1,
    axis=0,
    bins_arr=None,
    nbins=100,
    bin_target_only=False,
    typedistance="pss",
    printBins=False,
    min_freq=10,
):
    """
    Calculate distribution comparison metrics between 2 N-dimensional numpy arrays
    - over all dimensions (axes) or
    - over 0th dimension (axis) (may need to reshape input arrays first)
    Computes bin probabilities, then uses typedistance string:
    - pss: perkins skill score (1 best; similarity: O low to 1 high)
    - jsd: scipy.spatial.distance.jensenshannon (zero best)
    - fcss: fractional contribution skill score (0 best to 2 worst)
      only works for non-negative data e.g. precipitation
    - lhd: log histogram distance (zero best)
    - wdh: scipy.stats.wasserstein_distance on histogram (zero best)
      good for very large data
    - wd: scipy.stats.wasserstein_distance on data cdf method
      (not histogram) slow for very large data
    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade

    Parameters:
    -----------
    arr0 : numpy.ndarray
        First N-dimensional array (Target)
    arr1 : numpy.ndarray
        Second N-dimensional array (must have same shape as arr0) (Prediction)
    axis : 0 | None
        None : compute over all axes
        0 : compute over 0th axis so output of form N-1 dimensions
    bins_arr : numpy.ndarray | None
        Explicitly define bins for histogram (default method, else use nbins)
        e.g. np.arange(0,1200,20)
        None => Use default of hist function with nbins
    nbins : int
        Define bins using default hist method with nbins=no. bins based on target arr0
        If nbins input as a float, this will be rounded down to int(nbins)
    bin_target_only : Boolean
        True: for nbins option, base on target truth range only
        False: for nbins option, base on truth and prediction range
    typedistance : str
        Define type of distance or score to compute
    printBins : Boolean
        True: Option to print info regarding number bins and edge values
    min_freq: 10
        Option for FCSS: filter out bins that have frequency <= min_freq threshold

    Returns:
    --------
    numpy.ndarray | float
        Array of distribution comparison metrics for sets of input target (arr0) and
        prediction (arr1) fields.
        if axis==None: Output is a float
        if axis==0: Output has N - 1 dimensions (input shape with the 0th axis removed).

    See Also
    --------

    Perkins Skill Score
    Perkins et al, 2007
    https://journals.ametsoc.org/view/journals/clim/20/17/jcli4253.1.xml

    Jensen Shannon distance
    Using scipy.spatial.distance.jensenshannon
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    <<Compute the Jensen-Shannon distance (metric) between two probability
    arrays. This is the square root of the Jensen-Shannon divergence.>>

    Fractional Contribution Skill Score
    Berthou et al., 2020
    https://link.springer.com/article/10.1007/s00382-018-4114-6
    NB this score was developed for precipitation and only works for non-negative data
    They apply to precip with uneven histogram bins based on Klingaman et al. (2017)
    so can assess the contribution from different physical intensities of precip.

    Logarithmic Histogram Distance
    Rampal et al., 2024
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004668
    <<the LHD (units of decibels) measures the logarithmic distance between the
    predicted histogram and ground truth histogram>>
    They apply to precip for just rain days with linearly spaced histogram bins:
    <<53 evenly spaced bins (1- 1,050 mm/day) with a spacing of 20 mm>>

    Wasserstein Distance
    Using scipy.stats.wasserstein_distance
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
    wdh: compute distance on histogram data (faster for very large data)
    wd: use cdf method and compute directly on data without need for histogram

    """

    distance_list = ["pss", "jsd", "fcss", "lhd", "alhd", "wdh", "wd"]
    if typedistance not in distance_list:
        raise ValueError(f"typedistance not recognised: {typedistance}")

    axis_list = [None, 0]
    if axis not in axis_list:
        raise ValueError(f"axis not recognised as None or 0: {axis}")

    dist_out = np.nan

    if not isinstance(bins_arr, np.ndarray):
        bins_arr = None

    if isinstance(bins_arr, type(None)):
        if not isinstance(
            nbins, (int, float, np.int32, np.int64, np.float32, np.float64)
        ):
            raise ValueError(f"nbins not recognised as integer: {nbins}")
        if nbins < 1:
            raise ValueError(f"nbins not a positive integer: {nbins}")
        nbins = int(nbins)
        arr_min = np.array([arr0.min(), arr1.min()]).min()
        arr_max = np.array([arr0.max(), arr1.max()]).max()
        if bin_target_only:
            # Fit bins to target truth only so same for all prediction model comparisons
            bins_arr = np.linspace(arr_min, arr_max, nbins + 1)
        if not bin_target_only:
            # Fit bins to target truth and prediction so entire distributions are captured
            bins_arr = np.linspace(arr0.min(), arr0.max(), nbins + 1)

    if printBins:
        print(f"edges: {bins_arr[0]}, {bins_arr[1]},... {bins_arr[-1]}")
        print(f"nbins: {len(bins_arr)-1}")

    if axis == 0 and len(arr0.shape) < 2:
        axis = None

    if axis is None:
        if typedistance == "pss":
            phist, pedge = np.histogram(arr0, bins=bins_arr, density=True)
            qhist, qedge = np.histogram(arr1, bins=bins_arr, density=True)

            # Normalise so sums to 1
            phist = phist / phist.sum()
            qhist = qhist / qhist.sum()

            dist_out = float(np.sum(np.minimum(phist, qhist)))

        if typedistance == "jsd":
            phist, pedge = np.histogram(arr0, bins=bins_arr, density=True)
            qhist, qedge = np.histogram(arr1, bins=bins_arr, density=True)

            # Normalise so sums to 1
            phist = phist / phist.sum()
            qhist = qhist / qhist.sum()

            dist_out = float(jensenshannon(phist, qhist))

        if typedistance == "fcss":
            phist, pedge = np.histogram(arr0, bins=bins_arr, density=True)
            qhist, qedge = np.histogram(arr1, bins=bins_arr, density=True)

            # Normalise so sums to 1
            phist = phist / phist.sum(axis=0)
            qhist = qhist / qhist.sum(axis=0)

            # Use bin midpoints as representative values
            bin_centres = (bins_arr[:-1] + bins_arr[1:]) / 2

            # Contribution to mean
            p_contrib = phist * bin_centres
            q_contrib = qhist * bin_centres

            # Fractional contribution to mean
            p_fcontrib = p_contrib / np.sum(p_contrib)
            q_fcontrib = q_contrib / np.sum(q_contrib)

            dist_out = float(np.sum(np.abs(q_fcontrib - p_fcontrib)))

        if typedistance == "lhd":
            phist, pedge = np.histogram(arr0, bins=bins_arr, density=True)
            qhist, qedge = np.histogram(arr1, bins=bins_arr, density=True)

            # Normalise so sums to 1
            phist = phist / phist.sum()
            qhist = qhist / qhist.sum()

            phist_freq, pedge = np.histogram(arr0, bins=bins_arr, density=False)
            qhist_freq, qedge = np.histogram(arr1, bins=bins_arr, density=False)

            # Add small epsilon to avoid log(0)
            # and filter out bins that contain too few values
            eps = 1e-10
            valid_bins = (phist_freq > min_freq) & (qhist_freq > min_freq)
            phist[~valid_bins] = np.nan
            qhist[~valid_bins] = np.nan
            phist = np.clip(phist, eps, None)
            qhist = np.clip(qhist, eps, None)

            # Logarithmic histogram distance, ignoring sparse bins
            log_diff = (10.0 * (np.log10(phist) - np.log10(qhist))) ** 2

            dist_out = np.sqrt(np.nanmean(log_diff)).astype(float)

        if typedistance == "alhd":
            # LHD using abs(diff) method instead of rmse method
            phist, pedge = np.histogram(arr0, bins=bins_arr, density=True)
            qhist, qedge = np.histogram(arr1, bins=bins_arr, density=True)

            # Normalise so sums to 1
            phist = phist / phist.sum()
            qhist = qhist / qhist.sum()

            phist_freq, pedge = np.histogram(arr0, bins=bins_arr, density=False)
            qhist_freq, qedge = np.histogram(arr1, bins=bins_arr, density=False)

            # Add small epsilon to avoid log(0)
            # and filter out bins that contain too few values
            eps = 1e-10
            valid_bins = (phist_freq > min_freq) & (qhist_freq > min_freq)
            phist[~valid_bins] = np.nan
            qhist[~valid_bins] = np.nan
            phist = np.clip(phist, eps, None)
            qhist = np.clip(qhist, eps, None)

            # Logarithmic histogram distance, ignoring sparse bins
            log_diff = 10.0 * np.abs(np.log10(phist) - np.log10(qhist))

            dist_out = np.nanmean(log_diff).astype(float)

        if typedistance == "wdh":
            phist, pedge = np.histogram(arr0, bins=bins_arr, density=True)
            qhist, qedge = np.histogram(arr1, bins=bins_arr, density=True)

            # Normalise so sums to 1
            phist = phist / phist.sum(axis=0)
            qhist = qhist / qhist.sum(axis=0)

            # Use bin midpoints as representative values
            bin_centres = (bins_arr[:-1] + bins_arr[1:]) / 2

            # scipy's wasserstein_distance accepts weighted samples:
            # pass bin centers as "samples" and normalized counts as weights
            dist_out = wasserstein_distance(
                bin_centres, bin_centres, u_weights=phist, v_weights=qhist
            )

        if typedistance == "wd":
            arr0_flat = arr0.reshape(np.prod(arr0.shape))
            arr1_flat = arr1.reshape(np.prod(arr1.shape))
            dist_out = wasserstein_distance(arr0_flat, arr1_flat)

    if axis == 0:
        if typedistance == "pss":
            phist = np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr0
            )
            qhist = np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr1
            )

            # Normalise so sums to 1
            phist = phist / phist.sum(axis=0)
            qhist = qhist / qhist.sum(axis=0)

            dist_out = np.sum(np.minimum(phist, qhist), axis=0).astype(float)

        if typedistance == "jsd":
            phist = np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr0
            )
            qhist = np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr1
            )

            # Normalise so sums to 1 at each grid point?
            phist = phist / phist.sum(axis=0)
            qhist = qhist / qhist.sum(axis=0)

            dist_out = jensenshannon(phist, qhist)
            dist_out = dist_out.astype(float)

        if typedistance == "fcss":
            phist = np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr0
            )
            qhist = np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr1
            )

            # Normalise so sums to 1
            phist = phist / phist.sum(axis=0)
            qhist = qhist / qhist.sum(axis=0)

            # Use bin midpoints as representative values
            bin_centres = (bins_arr[:-1] + bins_arr[1:]) / 2

            # Contribution to mean
            extra_dims = phist.ndim - 1
            bclen = bin_centres.shape[0]
            p_contrib = bin_centres.reshape(bclen, *([1] * extra_dims)) * phist
            q_contrib = bin_centres.reshape(bclen, *([1] * extra_dims)) * qhist

            # Fractional contribution to mean
            p_fcontrib = p_contrib / np.sum(p_contrib, axis=0)
            q_fcontrib = q_contrib / np.sum(q_contrib, axis=0)

            dist_vec = np.abs(q_fcontrib - p_fcontrib)
            dist_out = np.sum(dist_vec, axis=0).astype(float)

        if typedistance == "lhd":
            phist = np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr0
            )
            qhist = np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr1
            )

            # Normalise so sums to 1
            phist = phist / phist.sum(axis=0)
            qhist = qhist / qhist.sum(axis=0)

            phist_freq = np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins_arr, density=False)[0], 0, arr0
            )
            qhist_freq = np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins_arr, density=False)[0], 0, arr1
            )

            # Add small epsilon to avoid log(0)
            # and filter out bins that contain too few values
            eps = 1e-10
            valid_bins = (phist_freq > min_freq) & (qhist_freq > min_freq)
            phist[~valid_bins] = np.nan
            qhist[~valid_bins] = np.nan
            phist = np.clip(phist, eps, None)
            qhist = np.clip(qhist, eps, None)

            # Logarithmic histogram distance, ignoring sparse bins
            log_diff = (10.0 * (np.log10(phist) - np.log10(qhist))) ** 2
            print(log_diff.shape)
            print(np.nanmean(log_diff, axis=0).shape)

            dist_out = np.sqrt(np.nanmean(log_diff, axis=0)).astype(float)

        if typedistance == "alhd":
            phist = np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr0
            )
            qhist = np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr1
            )

            # Normalise so sums to 1
            phist = phist / phist.sum(axis=0)
            qhist = qhist / qhist.sum(axis=0)

            phist_freq = np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins_arr, density=False)[0], 0, arr0
            )
            qhist_freq = np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins_arr, density=False)[0], 0, arr1
            )

            # Add small epsilon to avoid log(0)
            # and filter out bins that contain too few values
            eps = 1e-10
            valid_bins = (phist_freq > min_freq) & (qhist_freq > min_freq)
            phist[~valid_bins] = np.nan
            qhist[~valid_bins] = np.nan
            phist = np.clip(phist, eps, None)
            qhist = np.clip(qhist, eps, None)

            # Logarithmic histogram distance, ignoring sparse bins
            log_diff = 10.0 * np.abs(np.log10(phist) - np.log10(qhist))
            print(log_diff.shape)
            print(np.nanmean(log_diff, axis=0).shape)

            dist_out = np.nanmean(log_diff, axis=0).astype(float)

        if typedistance == "wdh":
            phist = np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr0
            )
            qhist = np.apply_along_axis(
                lambda a: np.histogram(a, bins=bins_arr, density=True)[0], 0, arr1
            )

            # Normalise so sums to 1
            phist = phist / phist.sum(axis=0)
            qhist = qhist / qhist.sum(axis=0)

            # Use bin midpoints as representative values
            bin_centres = (bins_arr[:-1] + bins_arr[1:]) / 2

            # scipy's wasserstein_distance accepts weighted samples:
            # pass bin centers as "samples" and normalized counts as weights
            dist_out = np.zeros(np.prod(arr0.shape[1:]))
            phist_flat = phist.reshape([phist.shape[0], np.prod(phist.shape[1:])])
            qhist_flat = qhist.reshape([qhist.shape[0], np.prod(qhist.shape[1:])])

            for ii in range(np.prod(arr0.shape[1:])):
                dist_out[ii] = wasserstein_distance(
                    bin_centres,
                    bin_centres,
                    u_weights=phist_flat[:, ii],
                    v_weights=qhist_flat[:, ii],
                )

            dist_out = dist_out.reshape(arr0.shape[1:])

        if typedistance == "wd":
            dist_out = np.zeros(np.prod(arr0.shape[1:]))
            arr0_flat = arr0.reshape(arr0.shape[0], np.prod(arr0.shape[1:]))
            arr1_flat = arr1.reshape(arr1.shape[0], np.prod(arr1.shape[1:]))
            for ii in range(np.prod(arr0.shape[1:])):
                dist_out[ii] = wasserstein_distance(arr0_flat[:, ii], arr1_flat[:, ii])

            dist_out = dist_out.reshape(arr0.shape[1:])

    return dist_out


# -------------------------------------------------------------------------------------
def calc_pdf_metric_xr(
    x0: xr.Dataset,  # target_xr,
    x1: xr.Dataset,  # prediction_xr,
    var: str,
    dim: str | None = None,
    index_fn: Callable[..., xr.DataArray] | None = None,
    season: str | None = None,
    member_name=None,
    emean=False,
    bins_arr=None,
    nbins=100,
    bin_target_only=False,
    printBins=False,
    typedistance="pss",
    min_freq=10,
    **index_kwargs: Any,
) -> xr.Dataset:
    """
    Compute distribution comparison metrics between 2 N-dimensional numpy arrays
    - over all dimensions (axes) or
    - over 0th dimension (axis) (may need to reshape input arrays first)
    Computes bin probabilities, then uses typedistance string:
    - pss: perkins skill score (1 best; similarity: O low to 1 high)
    - jsd: scipy.spatial.distance.jensenshannon (zero best)
    - fcss: fractional contribution skill score (0 best to 2 worst)
      only works for non-negative data e.g. precipitation
    - lhd: log histogram distance (zero best)
    - wdh: scipy.stats.wasserstein_distance on histogram (zero best)
      good for very large data
    - wd: scipy.stats.wasserstein_distance on data cdf method
      (not histogram) slow for very large data

    **Assumes data arrays have the same size, though prediction_xr may have
      an additional ensemble dimension: member_name must be specified to
    match the name of the member dimensions
    **If prediction_xr is an ensemble and emean is True, the array
    mean will be taken over this member dimension before computing the
    distance metric. If emean is False, the distance metric will be
    computed for each member separately.
    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade

    Parameters:
    -----------
    x0 : xr.Dataset
        Ground truth dataset, same shape as prediction_xr but without
        ensemble dimension.
    x1 : xr.Dataset
        Predicted dataset. Must contain 'lat' and 'lon' dimensions,
        plus any number of leading dimensions (e.g. [time, lat, lon] or
        [time, member, lat, lon])
    var : str
        Variable name to analyse.
    dim : tuple of str | None
        Define dimensions to compute distance over e.g.
        ('time,) or ('member', 'time')
        None : compute distance over all data flattened
    member_name : str | None
        Define name of member dimension (ensemble prediction) e.g.
        'member' or 'realization'
    emean : booleon
        True : Option to compute ensemble mean of prediction over
        member_name dim if available
    bins_arr : numpy.ndarray | None
        Define the bins used for the histogram
        e.g. np.arange(0,1200,20)
        None => Use default of hist function based on arr0
    nbins : int
        Define bins using default hist method with nbins=no. bins based on target arr0
        If nbins input as a float, this will be rounded down to int(nbins)
    bin_target_only : Boolean
        True: for nbins option, base on target truth range only
        False: for nbins option, base on truth and prediction range
    printBins : Boolean
        True: Option to print info regarding number bins and edge values
    typedistance : str
        Define type of distance or score to compute
    min_freq: 10
        Option for FCSS: filter out bins that have frequency <= min_freq threshold

    Returns:
    --------
    xr.Dataset
        Array of distribution comparison metrics for sets of input target and
        prediction fields.
        if dim_over==None: Output is a float
        Else Output has N - 1 dimensions (input shape with the 0th axis removed).
        e.g. [lat, lon] or [member, lat, lon]

    See Also
    --------

    Perkins Skill Score
    Perkins et al, 2007
    https://journals.ametsoc.org/view/journals/clim/20/17/jcli4253.1.xml

    Jensen Shannon distance
    Using scipy.spatial.distance.jensenshannon
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    <<Compute the Jensen-Shannon distance (metric) between two probability
    arrays. This is the square root of the Jensen-Shannon divergence.>>

    Fractional Contribution Skill Score
    Berthou et al., 2020
    https://link.springer.com/article/10.1007/s00382-018-4114-6
    NB this score was developed for precipitation and only works for non-negative data
    They apply to precip with uneven histogram bins based on Klingaman et al. (2017)
    so can assess the contribution from different physical intensities of precip.

    Logarithmic Histogram Distance
    Rampal et al., 2024
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004668
    <<the LHD (units of decibels) measures the logarithmic distance between the
    predicted histogram and ground truth histogram>>
    They apply to precip for just rain days with linearly spaced histogram bins:
    <<53 evenly spaced bins (1- 1,050 mm/day) with a spacing of 20 mm>>

    Wasserstein Distance
    Using scipy.stats.wasserstein_distance
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
    wdh: compute distance on histogram data (faster for very large data)
    wd: use cdf method and compute directly on data without need for histogram

    """

    distance_list = ["pss", "jsd", "fcss", "lhd", "alhd", "wdh", "wd"]
    if typedistance not in distance_list:
        raise ValueError(f"typedistance not recognised: {typedistance}")

    if index_fn is None:
        target_xr = _filter_by_monthseason(x0, season)[var]
        prediction_xr = _filter_by_monthseason(x1, season)[var]

    else:
        call_kwargs = dict(index_kwargs)
        if season is not None and "season" not in call_kwargs:
            call_kwargs["season"] = season
        if var is not None and "var" not in call_kwargs:
            call_kwargs["var"] = var

        target_xr = index_fn(x0, **call_kwargs)[var]
        prediction_xr = index_fn(x1, **call_kwargs)[var]

    if dim is None:
        dim = prediction_xr.dims

    # Detect if dim dimensions present in prediction
    pred_dims = list(prediction_xr.dims)
    for dname in dim:
        if dname not in pred_dims:
            raise ValueError(
                f"dim dimension '{dname}' not found in dims {pred_dims}. "
                f"Expected something like 'time', 'member', etc."
            )

    # Detect if ensemble dim present in prediction but not target, and expand
    # target to match using expand_dims + broadcast
    extra_member_dim = None
    if member_name in prediction_xr.dims and member_name not in target_xr.dims:
        extra_member_dim = member_name
        if emean:
            # Compute ensemble mean over member_name dim
            prediction_xr = prediction_xr.mean(dim=member_name)
            extra_member_dim = None
            pred_dims = list(prediction_xr.dims)

    if extra_member_dim is not None:
        target_xr = target_xr.expand_dims(
            {extra_member_dim: prediction_xr[extra_member_dim]}
        )
        target_xr = target_xr.broadcast_like(prediction_xr)

    if target_xr.values.shape != prediction_xr.values.shape:
        raise ValueError(
            f"Shape mismatch: Target ({target_xr.values.shape}), "
            f"Prediction ({prediction_xr.values.shape})"
            f"Specify member_name if ensemble prediction)"
        )

    # Identify output dimensions i.e. those not in dim dimensions
    output_dims = [d for d in pred_dims if d not in dim]

    # --- Special case: output_dims empty, ---
    #  e.g. if prediction dims just [member, time]
    #  => compute metric over all data
    if not output_dims:
        dist_out = calc_pdf_metric_nparray(
            target_xr.values,
            prediction_xr.values,
            axis=None,
            bins_arr=bins_arr,
            nbins=nbins,
            bin_target_only=bin_target_only,
            typedistance=typedistance,
            printBins=printBins,
            min_freq=min_freq,
        )

        return xr.Dataset({var: dist_out})

    # Transpose to ensure dim dims come first, output_dims last e.g. spatial dims:
    # e.g. [member, time, lat, lon]
    ordered_dims = list(dim) + output_dims
    target_xr = target_xr.transpose(*ordered_dims)
    prediction_xr = prediction_xr.transpose(*ordered_dims)

    # Flatten all dim dims into a single axis so can compute over 0th dim
    dimover_shape = target_xr.values.shape[0 : len(dim)]  # e.g. [member, time]
    output_shape = target_xr.values.shape[len(dim) :]  # e.g. [lat, lon]
    n_total = int(np.prod(dimover_shape))

    target_flat = target_xr.values.reshape(n_total, *output_shape)
    prediction_flat = prediction_xr.values.reshape(n_total, *output_shape)

    dist_out = calc_pdf_metric_nparray(
        target_flat,
        prediction_flat,
        axis=0,
        bins_arr=bins_arr,
        nbins=nbins,
        bin_target_only=bin_target_only,
        typedistance=typedistance,
        printBins=printBins,
        min_freq=min_freq,
    )

    dist_out_xr = xr.Dataset({var: (output_dims, dist_out)})

    # Restore coordinates for all output dimensions
    for dim in output_dims:
        if dim in prediction_xr.coords:
            dist_out_xr[dim] = prediction_xr[dim]

    return dist_out_xr


# -------------------------------------------------------------------------------------
# Code directly from ml-benchmark diagnostics.py with just the season filter updated
# -------------------------------------------------------------------------------------
def _radial_average(array_2d: np.ndarray) -> np.ndarray:
    """
    Compute the radial average of a two-dimensional field.
    From ml-benchmark diagnostics.py.

    Parameters
    ----------
    array_2d : np.ndarray
        Two-dimensional array to average.

    Returns
    -------
    np.ndarray
        Radially averaged profile.
    """
    y, x = np.indices(array_2d.shape)
    center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - center[0], y - center[1]).astype(np.int32)
    tbin = np.bincount(r.ravel(), array_2d.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


# -------------------------------------------------------------------------------------
def psd(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    season: str | None = None,
):
    """
    Compute the power spectral density for x0 and x1 datasets.
    From ml-benchmark diagnostics.py.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season to filter before computing the PSD.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Power spectral densities for x0 and x1.
    """
    x0 = _filter_by_monthseason(x0, season)
    x1 = _filter_by_monthseason(x1, season)

    x0_da = x0[var]
    x1_da = x1[var]

    x0_np = np.nan_to_num(x0_da.values, nan=0.0)
    x1_np = np.nan_to_num(x1_da.values, nan=0.0)

    fft_x0 = fft.fftshift(fft.fft2(x0_np, axes=(-2, -1)), axes=(-2, -1))
    fft_x1 = fft.fftshift(fft.fft2(x1_np, axes=(-2, -1)), axes=(-2, -1))

    power_x0 = np.abs(fft_x0) ** 2
    power_x1 = np.abs(fft_x1) ** 2

    psd_x0_list = [_radial_average(p) for p in power_x0]
    psd_x1_list = [_radial_average(p) for p in power_x1]

    avg_psd_x0 = np.mean(psd_x0_list, axis=0)
    avg_psd_x1 = np.mean(psd_x1_list, axis=0)

    psd_x0_da = xr.DataArray(avg_psd_x0, dims=["wavenumber"], name="PSD_x0")
    psd_x1_da = xr.DataArray(avg_psd_x1, dims=["wavenumber"], name="PSD_x1")

    return psd_x0_da, psd_x1_da


# -------------------------------------------------------------------------------------
def ralsd(psd_x0: xr.DataArray, psd_x1: xr.DataArray) -> float:
    """
    Compute the Radially Averaged Log Spectral Distance (RALSD).

    RALSD measures the distance between two radially averaged power spectra
    in log space (Harris et al., 2022). Lower values indicate better agreement
    between the spectra.

    From ml-benchmark diagnostics.py.

    Parameters
    ----------
    psd_x0 : xr.DataArray
        Power spectral density of the ground truth (as returned by psd function).
    psd_x1 : xr.DataArray
        Power spectral density of the prediction (as returned by psd function).

    Returns
    -------
    float
        Scalar RALSD value.
    """
    ratio = psd_x0.values / np.maximum(psd_x1.values, 1e-30)
    log_ratio = 10.0 * np.log10(np.maximum(ratio, 1e-30))
    return float(np.mean(log_ratio**2))


# -------------------------------------------------------------------------------------
def calc_wasserstein_distance(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    season: str | None = None,
    spatial: bool = False,
) -> float | xr.Dataset:
    """
    Compute the Wasserstein distance between x0 and x1 datasets.
    Adapted from  ml-benchmark diagnostics.py.
    - changed function name as wasserstein_distance() exists in sp.stats
    - changed season filter

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season to filter before computing the Wasserstein distance.
    spatial : bool, default False
        If True, compute distance at each grid point across time dimension.
        If False, compute a single scalar distance by flattening both datasets.

    Returns
    -------
    float or xr.Dataset
        If spatial=False: single Wasserstein distance value.
        If spatial=True: Wasserstein distance field with same spatial dimensions as input.
    """
    x0 = _filter_by_monthseason(x0, season)
    x1 = _filter_by_monthseason(x1, season)

    x0_da = x0[var]
    x1_da = x1[var]

    if not spatial:
        x0_flat = np.nan_to_num(x0_da.values, nan=0.0).flatten()
        x1_flat = np.nan_to_num(x1_da.values, nan=0.0).flatten()

        print("Computing Wasserstein distance...")
        distance = wasserstein_distance_nd(u_values=x0_flat, v_values=x1_flat)
        return distance

    else:
        x0_np = np.nan_to_num(x0_da.values, nan=0.0)
        x1_np = np.nan_to_num(x1_da.values, nan=0.0)

        time_dim = x0_np.shape[0]
        spatial_shape = x0_np.shape[1:]
        x0_reshaped = x0_np.reshape(time_dim, -1)
        x1_reshaped = x1_np.reshape(time_dim, -1)

        n_spatial = x0_reshaped.shape[1]
        distances = np.zeros(n_spatial)

        # Loop with progress bar # taken out so don't need to install tqdm
        ##for i in tqdm(range(n_spatial), desc="Computing Wasserstein distances"):

        # Standard loop
        for i in range(n_spatial):
            x0_temporal = x0_reshaped[:, i].flatten()
            x1_temporal = x1_reshaped[:, i].flatten()
            distances[i] = wasserstein_distance_nd(x0_temporal, x1_temporal)

        distance_field = distances.reshape(spatial_shape)

        wd_da = x0.isel(time=0).copy(deep=True)
        wd_da.attrs = {}
        wd_da[var].values = distance_field

        return wd_da


# -------------------------------------------------------------------------------------
# OBSELETE
# -------------------------------------------------------------------------------------
def mean_bias_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    typeerror: str = "bias",
    index_fn: Callable[..., xr.DataArray] | None = None,
    season: str | None = None,
    **index_kwargs: Any,
) -> xr.Dataset:
    """
    Compute the time mean bias between (index function of) x0 and x1 datasets.
    Added by R. Eade.
    Adapted from ml-benchmark diagnostics.py rmse() and bias_index()
    - changed season filter and added options for error type (typeerror)

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    typeerror : str, default='rmse'
        Type of error to compute, options ('bias', 'ratio')
    index_fn : Callable[..., xr.DataArray]
        Function computing the desired index.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season to filter before computing the index.
    relative : bool, default False
        If True, return the relative bias.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.

    Returns
    -------
    xr.Dataset
        Error between x0 and x1 (x1-x0).
        Output has N - len(dim) dimensions
        (input shape with the specified dimension(s) removed).
    """

    if index_fn is None:
        x0_index = _filter_by_monthseason(x0, season)[[var]]
        x1_index = _filter_by_monthseason(x1, season)[[var]]

    else:
        call_kwargs = dict(index_kwargs)
        if season is not None and "season" not in call_kwargs:
            call_kwargs["season"] = season
        if var is not None and "var" not in call_kwargs:
            call_kwargs["var"] = var

        x0_index = index_fn(x0, **call_kwargs)
        x1_index = index_fn(x1, **call_kwargs)

    dim = ("time",)

    dim_list = list(x0_index.dims)
    matches = [dim1 for dim1 in dim_list if dim1 in dim]

    if matches:
        x0_index = x0_index.mean("time")
        x1_index = x1_index.mean("time")

    bias = x1_index - x0_index

    if typeerror == "ratio":
        bias = bias / x0_index

    return bias


# -------------------------------------------------------------------------------------
def error_ann_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    typeerror: str = "rmse",
    dim: str | None = None,
    index_fn: Callable[..., xr.DataArray] | None = None,
    season: str | None = None,
    **index_kwargs: Any,
) -> xr.Dataset:
    """
    Compute the (rms) error between (index function of) x0 and x1 datasets.
    Added by R. Eade.
    Adapted from ml-benchmark diagnostics.py rmse() and bias_index()
    - changed season filter and added options for error type (typeerror)

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    typeerror : str, default='rmse'
        Type of error to compute, options ('rmse', 'bias', 'abias')
    dim : str | None, optional
        Dimension(s) along which to compute the RMSE. If None, computes over all dimensions.
    index_fn : Callable[..., xr.DataArray]
        Function computing the desired index.
    season : str | None, optional
        Season to filter before computing the index.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.

    Returns
    -------
    xr.Dataset
        Error between x0 and x1 (x1-x0).
        Output has N - len(dim) dimensions
        (input shape with the specified dimension(s) removed).
    """

    if index_fn is None:
        x0_index = _filter_by_monthseason(x0, season)[[var]]
        x1_index = _filter_by_monthseason(x1, season)[[var]]

    else:
        call_kwargs = dict(index_kwargs)
        if season is not None and "season" not in call_kwargs:
            call_kwargs["season"] = season
        if var is not None and "var" not in call_kwargs:
            call_kwargs["var"] = var

        x0_index = index_fn(x0, **call_kwargs)
        x1_index = index_fn(x1, **call_kwargs)

    if isinstance(dim, str):
        # Convert string to tuple
        dim = (dim,)

    if isinstance(dim, tuple):
        dim_list = list(x0_index.dims)
        matches = [dim1 for dim1 in dim_list if dim1 in dim]
        axis_num = tuple(dim_list.index(dim1) for dim1 in matches)
        if not matches:
            print(
                f"Warning: no matches found for {list(dim)} in {dim_list} so set dim=None"
            )
            dim = None

    if dim is None:
        error_xarray = (x1_index - x0_index).mean(dim=dim) * 0.0
        error_result = calculate_error_nparray(
            x0_index[var].values, x1_index[var].values, axis=None, typeerror=typeerror
        )
        error_xarray = error_xarray + error_result
    else:
        error_xarray = (x1_index - x0_index).mean(dim=dim) * 0.0
        error_result = calculate_error_nparray(
            x0_index[var].values,
            x1_index[var].values,
            axis=axis_num,
            typeerror=typeerror,
        )
        error_xarray = error_xarray + error_result

    return error_xarray


# -------------------------------------------------------------------------------------
def pearsoncorr_ann_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    dim: str | None = None,
    index_fn: Callable[..., xr.DataArray] | None = None,
    season: str | None = None,
    **index_kwargs: Any,
) -> xr.Dataset:
    """
    Compute the Pearson Correlation between (index function of) x0 and x1 datasets.
    Added by R. Eade.
    Adapted from ml-benchmark diagnostics.py rmse() and bias_index()
    - changed season filter and added correlation function

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    dim : str | None, optional
        Dimension(s) along which to compute the RMSE. If None, computes over all dimensions.
    index_fn : Callable[..., xr.DataArray]
        Function computing the desired index.
    season : str | None, optional
        Season to filter before computing the index.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.

    Returns
    -------
    xr.Dataset
        Pearson Correlation between x0 and x1.
        Output has N - len(dim) dimensions
        (input shape with the specified dimension(s) removed).
    """

    if index_fn is None:
        x0_index = _filter_by_monthseason(x0, season)[[var]]
        x1_index = _filter_by_monthseason(x1, season)[[var]]

    else:
        call_kwargs = dict(index_kwargs)
        if season is not None and "season" not in call_kwargs:
            call_kwargs["season"] = season
        if var is not None and "var" not in call_kwargs:
            call_kwargs["var"] = var

        x0_index = index_fn(x0, **call_kwargs)
        x1_index = index_fn(x1, **call_kwargs)

    if isinstance(dim, str):
        # Convert string to tuple
        dim = (dim,)

    if isinstance(dim, tuple):
        dim_list = list(x0_index.dims)
        matches = [dim1 for dim1 in dim_list if dim1 in dim]
        axis_num = tuple(dim_list.index(dim1) for dim1 in matches)
        if not matches:
            print(
                f"Warning: no matches found for {list(dim)} in {dim_list} so set dim=None"
            )
            dim = None

    if dim is None:
        pcor_xarray = x0_index.mean(dim=dim) * 0.0
        pcor_result = calculate_pearsoncorr_nparray(
            x0_index[var].values, x1_index[var].values, axis=None
        )
        pcor_xarray = pcor_xarray + pcor_result
    else:
        pcor_xarray = x0_index.mean(dim=dim) * 0.0
        pcor_result = calculate_pearsoncorr_nparray(
            x0_index[var].values, x1_index[var].values, axis=axis_num
        )
        pcor_xarray = pcor_xarray + pcor_result

    return pcor_xarray
