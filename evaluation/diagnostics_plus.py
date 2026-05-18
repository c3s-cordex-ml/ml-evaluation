from __future__ import annotations

from typing import Any, Callable

import numpy as np
import xarray as xr
#from tqdm import tqdm # taken out so don't need to install tqdm

from numpy import fft
from scipy.stats import wasserstein_distance_nd

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
    
    MON_LIST=["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    SEAS_LIST=["DJF","MAM","JJA","SON"]
    
    if not season in MON_LIST and not season in SEAS_LIST: 
        season=None
    if season is None:
        return x
        
    if season in SEAS_LIST:
        return x.where(x["time.season"] == season, drop=True)
        
    if season in MON_LIST:
        month_num = MON_LIST.index(season) + 1 
        return x.where(x["time.month"] == month_num, drop=True)

    return x

# -------------------------------------------------------------------------------------
def calculate_error_nparray(arr0, arr1, axis=0, typeerror='rmse'):
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
        raise ValueError(f"Arrays must have the same shape. Got {arr0.shape} and {arr1.shape}")
    
    # Compute Error over axis/axes specified
    # - subtract first then take mean so consistent method for bias and rmse
    # - better level of accuracy than taking means then subtract (ie slight difference)
    if typeerror=='bias': output_error = (arr1-arr0).mean(axis=axis)
    if typeerror=='abias': output_error = np.abs(arr1-arr0).mean(axis=axis)
    if typeerror=='rmse': output_error = np.sqrt(((arr1-arr0)**2).mean(axis=axis))
    if typeerror=='ratio': output_error = arr1.mean(axis=axis)/arr0.mean(axis=axis)
    
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
    axis : int or type of int, default=0
        Axis or tuple of axes over which to compute correlation
        None : if want to compute over all axes
    
    Returns:
    --------
    numpy.ndarray
        Pearson correlation coefficients. Output has N - len(axis) dimensions
        (input shape with the specified axis/axes removed).
    
    """

    if arr0.shape != arr1.shape:
        raise ValueError(f"Arrays must have the same shape. Got {arr0.shape} and {arr1.shape}")
    
    # Center the data over axis/axes specified
    arr0_centered = arr0 - arr0.mean(axis=axis, keepdims=True)
    arr1_centered = arr1 - arr1.mean(axis=axis, keepdims=True)
    
    # Compute correlation over axis/axes specified
    numerator = (arr0_centered * arr1_centered).sum(axis=axis)
    denominator = np.sqrt((arr0_centered**2).sum(axis=axis) * (arr1_centered**2).sum(axis=axis))

    # Avoid division by zero (set as 0.0 instead of inf or nan)
    correlations = np.divide(numerator, denominator, 
                            out=np.zeros_like(numerator), 
                            where=denominator!=0)
    
    return correlations

# -------------------------------------------------------------------------------------
def mean_bias_index( 
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    typeerror: str = 'bias',
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

    dim_list=list(x0_index.dims)
    matches = [dim1 for dim1 in dim_list if dim1 in dim]
    
    if matches:
        x0_index=x0_index.mean("time")
        x1_index=x1_index.mean("time")

    bias = x1_index - x0_index

    if typeerror=="ratio":
        bias = bias / x0_index
    
    return bias

# -------------------------------------------------------------------------------------
def error_ann_index( 
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    typeerror: str = 'rmse',
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
        dim_list=list(x0_index.dims)
        matches = [dim1 for dim1 in dim_list if dim1 in dim]
        axis_num = tuple(dim_list.index(dim1) for dim1 in matches)
        if not matches:
            print(f"Warning: no matches found for {list(dim)} in {dim_list} so set dim=None")
            dim=None

    if dim is None:
        error_xarray=(x1_index-x0_index).mean(dim=dim)*0.0
        error_result=calculate_error_nparray(x0_index[var].values, x1_index[var].values, axis=None, typeerror=typeerror)
        error_xarray=error_xarray+error_result
    else:
        error_xarray=(x1_index-x0_index).mean(dim=dim)*0.0
        error_result=calculate_error_nparray(x0_index[var].values, x1_index[var].values, axis=axis_num, typeerror=typeerror)
        error_xarray=error_xarray+error_result
    
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
        dim_list=list(x0_index.dims)
        matches = [dim1 for dim1 in dim_list if dim1 in dim]
        axis_num = tuple(dim_list.index(dim1) for dim1 in matches)
        if not matches:
            print(f"Warning: no matches found for {list(dim)} in {dim_list} so set dim=None")
            dim=None

    if dim is None:
        pcor_xarray=x0_index.mean(dim=dim)*0.0
        pcor_result=calculate_pearsoncorr_nparray(x0_index[var].values, x1_index[var].values, axis=None)
        pcor_xarray=pcor_xarray+pcor_result
    else:
        pcor_xarray=x0_index.mean(dim=dim)*0.0
        pcor_result=calculate_pearsoncorr_nparray(x0_index[var].values, x1_index[var].values, axis=axis_num)
        pcor_xarray=pcor_xarray+pcor_result
    
    return pcor_xarray


# -------------------------------------------------------------------------------------
def bias_multivariable_correlation(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var_x: str,
    var_y: str,
    season: str | None = None,
) -> xr.Dataset:
    """
    Compute correlations between two variables for x0 and x1 datasets
    and return the correlation bias.
    Added by R. Eade.
    Adapted from ml-benchmark diagnostics.py bias_multivariable_correlation()
    - changed season filter and correlation function (as quicker?)

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

    Returns
    -------
    xr.Dataset
        Dataset containing the x0 correlation, x1 correlation, signed
        correlation difference, and the requested correlation bias.
    """

    x0_filtered = _filter_by_monthseason(x0, season)
    x1_filtered = _filter_by_monthseason(x1, season)
    
    dim=('time',)

    dim_list=list(x0_filtered.dims)
    matches = [dim1 for dim1 in dim_list if dim1 in dim]
    axis_num = tuple(dim_list.index(dim1) for dim1 in matches)
    if not matches:
        raise ValueError(f"Warning: no matches found for {list(dim)} in {dim_list}")

    x0_corr_xarray=x0_filtered[var_x].mean(dim=dim)*0.0
    x0_corr_result=calculate_pearsoncorr_nparray(x0_filtered[var_x].values, x0_filtered[var_y].values, axis=axis_num)
    x0_corr_xarray=x0_corr_xarray+x0_corr_result

    x1_corr_xarray=x1_filtered[var_x].mean(dim=dim)*0.0
    x1_corr_result=calculate_pearsoncorr_nparray(x1_filtered[var_x].values, x1_filtered[var_y].values, axis=axis_num)
    x1_corr_xarray=x1_corr_xarray+x1_corr_result

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
    return float(np.mean(log_ratio ** 2))

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

        print('Computing Wasserstein distance...')
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