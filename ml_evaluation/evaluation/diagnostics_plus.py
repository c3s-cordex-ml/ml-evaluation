from __future__ import annotations

from typing import Any, Callable

import numpy as np
import xarray as xr
# from tqdm import tqdm # taken out so don't need to install tqdm

from numpy import fft
from scipy.stats import wasserstein_distance_nd
from scipy.stats import rankdata


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
def wasserstein_distance(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    season: str | None = None,
    spatial: bool = False,
) -> float | xr.Dataset:
    """
    Compute the Wasserstein distance between x0 and x1 datasets.
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
