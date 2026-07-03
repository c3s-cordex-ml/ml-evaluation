from __future__ import annotations

from typing import Any, Callable

import numpy as np
import xarray as xr
from numpy import fft
from scipy.stats import (
    wasserstein_distance_nd,
    wasserstein_distance as scipy_wasserstein_distance,
)
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from ml_evaluation.evaluation.utils import (
    filter_by_monthseason,
    radial_average,
    calculate_error_nparray,
    calculate_pearsoncorr_nparray,
    calculate_corr_xrnp,
)


def bias_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    index_fn: Callable[..., xr.Dataset | xr.DataArray],
    season: str | None = None,
    relative: bool = False,
    **index_kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """
    Compute the bias between x0 and x1 for a supplied index function.

    Filters datasets using the season filter before computing index values.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    index_fn : Callable[..., xr.Dataset | xr.DataArray]
        Function computing the desired index.
    season : str | None, optional
        Season to filter before computing the index.
    relative : bool, default False
        If True, return the relative bias.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.

    Returns
    -------
    xr.Dataset or xr.DataArray
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


def ratio_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    index_fn: Callable[..., xr.Dataset | xr.DataArray],
    season: str | None = None,
    **index_kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """
    Compute the ratio between x1 and x0 for a supplied index function.

    Filters datasets using the season filter before computing index values.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    index_fn : Callable[..., xr.Dataset | xr.DataArray]
        Function computing the desired index.
    season : str | None, optional
        Season to filter before computing the index.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Ratio of the selected index (x1 / x0).
    """
    call_kwargs = dict(index_kwargs)
    if season is not None and "season" not in call_kwargs:
        call_kwargs["season"] = season

    x0_index = index_fn(x0, **call_kwargs)
    x1_index = index_fn(x1, **call_kwargs)
    ratio = x1_index / x0_index
    return ratio


def rmse_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    dim: str | tuple[str, ...] | None = None,
    index_fn: Callable[..., xr.Dataset | xr.DataArray] | None = None,
    season: str | None = None,
    mae: bool = False,
    mse: bool = False,
    **index_kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """
    Compute the RMSE between x0 and x1 for a supplied index function.

    Provides options for alternative error types like MAE or MSE.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    dim : str or tuple of str or None, optional
        Dimension(s) along which to compute the RMSE.
    index_fn : Callable[..., xr.Dataset | xr.DataArray] or None, optional
        Function computing the desired index (None for raw data).
    season : str | None, optional
        Season to filter before computing the index.
    mae : bool, default False
        If True, return the mean absolute error.
    mse : bool, default False
        If True, return the mean squared error.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.

    Returns
    -------
    xr.Dataset or xr.DataArray
        RMSE, MAE, or MSE for the selected index.

    Raises
    ------
    ValueError
        If mae and mse can't both be True.
    """
    rmse_flag = True
    if mae or mse:
        rmse_flag = False

    if mae and mse:
        raise ValueError("Warning: mae and mse can't both be True")

    x0_index: xr.Dataset | xr.DataArray
    x1_index: xr.Dataset | xr.DataArray

    if index_fn is None:
        x0_index = filter_by_monthseason(x0, season)[[var]]
        x1_index = filter_by_monthseason(x1, season)[[var]]
    else:
        call_kwargs = dict(index_kwargs)
        if season is not None and "season" not in call_kwargs:
            call_kwargs["season"] = season
        if var is not None and "var" not in call_kwargs:
            call_kwargs["var"] = var

        x0_index = index_fn(x0, **call_kwargs)
        x1_index = index_fn(x1, **call_kwargs)

    diff = x1_index - x0_index

    error_result: xr.Dataset | xr.DataArray
    if dim is None:
        if rmse_flag or mse:
            error_result = (diff**2).mean()
        elif mae:
            error_result = abs(diff).mean()
    else:
        if rmse_flag or mse:
            error_result = (diff**2).mean(dim=dim)
        elif mae:
            error_result = abs(diff).mean(dim=dim)

    if rmse_flag:
        error_result = np.sqrt(error_result)  # type: ignore[assignment]

    return error_result


def corr_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    dim: str | tuple[str, ...] | None = None,
    index_fn: Callable[..., xr.Dataset | xr.DataArray] | None = None,
    season: str | None = None,
    srank: bool = False,
    npcorr: bool = False,
    **index_kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """
    Compute the Correlation between x0 and x1 for a supplied index function.

    Default is Pearson's correlation, with options for Spearman's rank correlation.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    dim : str or tuple of str or None, optional
        Dimension(s) along which to compute the correlation.
    index_fn : Callable[..., xr.Dataset | xr.DataArray] or None, optional
        Function computing the desired index (None for raw data).
    season : str | None, optional
        Season to filter before computing the index.
    srank : bool, default False
        If True, return the Spearman's rank correlation instead.
    npcorr : bool, default False
        If True, use numpy version of correlation instead of xarray.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Correlation result for the selected index.
    """
    pcorr = True
    if srank:
        pcorr = False
        npcorr = True

    x0_index: xr.Dataset | xr.DataArray
    x1_index: xr.Dataset | xr.DataArray

    if index_fn is None:
        x0_index = filter_by_monthseason(x0, season)[[var]]
        x1_index = filter_by_monthseason(x1, season)[[var]]
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
    elif pcorr and not npcorr:
        corr_result = corr_result + xr.corr(x1_index[var], x0_index[var], dim=dim)
    elif srank and npcorr:
        corr_result = corr_result + calculate_corr_xrnp(
            x1_index[var], x0_index[var], dim=dim, corrtype="Spearman"
        )

    return corr_result


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
    Compute correlations between two variables for x0 and x1 datasets.

    Returns the correlation bias between the reference and predicted datasets.

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
        If True, return the Spearman's rank correlation instead.
    npcorr : bool, default False
        If True, use numpy version of correlation instead of xarray.

    Returns
    -------
    xr.Dataset
        Dataset containing the correlation bias and individual correlations.

    Raises
    ------
    ValueError
        If no matches found for dimensions.
    """
    pcorr = True
    if srank:
        pcorr = False
        npcorr = True

    x0_filtered = filter_by_monthseason(x0, season)
    x1_filtered = filter_by_monthseason(x1, season)

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
    elif pcorr and not npcorr:
        x0_corr_xarray = x0_corr_xarray + xr.corr(
            x0_filtered[var_x], x0_filtered[var_y], dim=dim
        )
        x1_corr_xarray = x1_corr_xarray + xr.corr(
            x1_filtered[var_x], x1_filtered[var_y], dim=dim
        )
    elif srank and npcorr:
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


def psd(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    season: str | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute the power spectral density for x0 and x1 datasets.

    Performs a 2D FFT and radially averages to get the power spectral densities.

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
    tuple of xr.DataArray
        Power spectral densities for x0 and x1.
    """
    x0_filtered = filter_by_monthseason(x0, season)
    x1_filtered = filter_by_monthseason(x1, season)

    x0_da = x0_filtered[var]
    x1_da = x1_filtered[var]

    x0_np = np.nan_to_num(x0_da.values, nan=0.0)
    x1_np = np.nan_to_num(x1_da.values, nan=0.0)

    fft_x0 = fft.fftshift(fft.fft2(x0_np, axes=(-2, -1)), axes=(-2, -1))
    fft_x1 = fft.fftshift(fft.fft2(x1_np, axes=(-2, -1)), axes=(-2, -1))

    power_x0 = np.abs(fft_x0) ** 2
    power_x1 = np.abs(fft_x1) ** 2

    psd_x0_list = [radial_average(p) for p in power_x0]
    psd_x1_list = [radial_average(p) for p in power_x1]

    avg_psd_x0 = np.mean(psd_x0_list, axis=0)
    avg_psd_x1 = np.mean(psd_x1_list, axis=0)

    psd_x0_da = xr.DataArray(avg_psd_x0, dims=["wavenumber"], name="PSD_x0")
    psd_x1_da = xr.DataArray(avg_psd_x1, dims=["wavenumber"], name="PSD_x1")

    return psd_x0_da, psd_x1_da


def ralsd(psd_x0: xr.DataArray, psd_x1: xr.DataArray) -> float:
    """
    Compute the Radially Averaged Log Spectral Distance (RALSD).

    RALSD measures the distance between two radially averaged power spectra
    in log space. Lower values indicate better agreement.

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


def ralsd_ds(psd_x0: xr.DataArray, psd_x1: xr.DataArray, var: str) -> float:
    """
    Compute the Radially Averaged Log Spectral Distance (RALSD).

    RALSD measures the distance between two radially averaged power spectra
    in log space. Lower values indicate better agreement.

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
    res = xr.Dataset({var: ((), float(np.mean(log_ratio**2)))})
    return res


def wasserstein_distance(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    season: str | None = None,
    spatial: bool = False,
) -> float | xr.Dataset:
    """
    Compute the Wasserstein distance between x0 and x1 datasets.

    Can compute either a spatial field of distances or a single flattened scalar distance.

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

    Returns
    -------
    float or xr.Dataset
        If spatial=False: single Wasserstein distance value.
        If spatial=True: Wasserstein distance field dataset.
    """
    x0_filtered = filter_by_monthseason(x0, season)
    x1_filtered = filter_by_monthseason(x1, season)

    x0_da = x0_filtered[var]
    x1_da = x1_filtered[var]

    if not spatial:
        x0_flat = np.nan_to_num(x0_da.values, nan=0.0).flatten()
        x1_flat = np.nan_to_num(x1_da.values, nan=0.0).flatten()

        print("Computing Wasserstein distance...")
        distance: float = wasserstein_distance_nd(u_values=x0_flat, v_values=x1_flat)
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

        for i in tqdm(range(n_spatial), desc="Computing Wasserstein distances"):
            x0_temporal = x0_reshaped[:, i].flatten()
            x1_temporal = x1_reshaped[:, i].flatten()
            distances[i] = wasserstein_distance_nd(x0_temporal, x1_temporal)

        distance_field = distances.reshape(spatial_shape)

        wd_da = x0_filtered.isel(time=0).copy(deep=True)
        wd_da.attrs = {}
        wd_da[var].values = distance_field

        return wd_da


def mean_bias_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    typeerror: str = "bias",
    index_fn: Callable[..., xr.Dataset | xr.DataArray] | None = None,
    season: str | None = None,
    **index_kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """
    Compute the time mean bias between (index function of) x0 and x1 datasets.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    typeerror : str, default 'bias'
        Type of error to compute, options are 'bias' or 'ratio'.
    index_fn : Callable[..., xr.Dataset | xr.DataArray] or None, optional
        Function computing the desired index (None for raw data).
    season : str | None, optional
        Season to filter before computing the index.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Mean bias or ratio between predictions and reference.
    """
    x0_index: xr.Dataset | xr.DataArray
    x1_index: xr.Dataset | xr.DataArray

    if index_fn is None:
        x0_index = filter_by_monthseason(x0, season)[[var]]
        x1_index = filter_by_monthseason(x1, season)[[var]]
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


def error_ann_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    typeerror: str = "rmse",
    dim: str | tuple[str, ...] | None = None,
    index_fn: Callable[..., xr.Dataset | xr.DataArray] | None = None,
    season: str | None = None,
    **index_kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """
    Compute the (rms) error between (index function of) x0 and x1 datasets.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    typeerror : str, default 'rmse'
        Type of error to compute, options are 'rmse', 'bias', 'abias'.
    dim : str or tuple of str or None, optional
        Dimension(s) along which to compute the error.
    index_fn : Callable[..., xr.Dataset | xr.DataArray] or None, optional
        Function computing the desired index (None for raw data).
    season : str | None, optional
        Season to filter before computing the index.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Error between x0 and x1.
    """
    x0_index: xr.Dataset | xr.DataArray
    x1_index: xr.Dataset | xr.DataArray

    if index_fn is None:
        x0_index = filter_by_monthseason(x0, season)[[var]]
        x1_index = filter_by_monthseason(x1, season)[[var]]
    else:
        call_kwargs = dict(index_kwargs)
        if season is not None and "season" not in call_kwargs:
            call_kwargs["season"] = season
        if var is not None and "var" not in call_kwargs:
            call_kwargs["var"] = var

        x0_index = index_fn(x0, **call_kwargs)
        x1_index = index_fn(x1, **call_kwargs)

    dim_tuple: tuple[str, ...] | None
    if isinstance(dim, str):
        dim_tuple = (dim,)
    else:
        dim_tuple = dim

    axis_num: int | tuple[int, ...] | None
    if isinstance(dim_tuple, tuple):
        dim_list = list(x0_index.dims)
        matches = [dim1 for dim1 in dim_list if dim1 in dim_tuple]
        axis_num = tuple(dim_list.index(dim1) for dim1 in matches)
        if not matches:
            print(
                f"Warning: no matches found for {list(dim_tuple)} in {dim_list} so set dim=None"
            )
            axis_num = None
    else:
        axis_num = None

    if axis_num is None:
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


def pearsoncorr_ann_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    dim: str | tuple[str, ...] | None = None,
    index_fn: Callable[..., xr.Dataset | xr.DataArray] | None = None,
    season: str | None = None,
    **index_kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """
    Compute the Pearson Correlation between (index function of) x0 and x1 datasets.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    dim : str or tuple of str or None, optional
        Dimension(s) along which to compute the correlation.
    index_fn : Callable[..., xr.Dataset | xr.DataArray] or None, optional
        Function computing the desired index (None for raw data).
    season : str | None, optional
        Season to filter before computing the index.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Pearson correlation between x0 and x1.
    """
    x0_index: xr.Dataset | xr.DataArray
    x1_index: xr.Dataset | xr.DataArray

    if index_fn is None:
        x0_index = filter_by_monthseason(x0, season)[[var]]
        x1_index = filter_by_monthseason(x1, season)[[var]]
    else:
        call_kwargs = dict(index_kwargs)
        if season is not None and "season" not in call_kwargs:
            call_kwargs["season"] = season
        if var is not None and "var" not in call_kwargs:
            call_kwargs["var"] = var

        x0_index = index_fn(x0, **call_kwargs)
        x1_index = index_fn(x1, **call_kwargs)

    dim_tuple: tuple[str, ...] | None
    if isinstance(dim, str):
        dim_tuple = (dim,)
    else:
        dim_tuple = dim

    axis_num: int | tuple[int, ...] | None
    if isinstance(dim_tuple, tuple):
        dim_list = list(x0_index.dims)
        matches = [dim1 for dim1 in dim_list if dim1 in dim_tuple]
        axis_num = tuple(dim_list.index(dim1) for dim1 in matches)
        if not matches:
            print(
                f"Warning: no matches found for {list(dim_tuple)} in {dim_list} so set dim=None"
            )
            axis_num = None
    else:
        axis_num = None

    if axis_num is None:
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


def rmse(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    season: str | None = None,
    dim: str | tuple[str, ...] | None = None,
) -> xr.Dataset:
    """
    Compute the root mean square error between x0 and x1 datasets.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    season : str | None, optional
        Season to filter before computing the RMSE.
    dim : str or tuple of str or None, optional
        Dimension(s) along which to compute the RMSE.

    Returns
    -------
    xr.Dataset
        Root mean square error dataset.
    """
    x0_filtered = filter_by_monthseason(x0, season)
    x1_filtered = filter_by_monthseason(x1, season)

    x0_da = x0_filtered[[var]]
    x1_da = x1_filtered[[var]]

    squared_diff = (x1_da - x0_da) ** 2

    if dim is None:
        mse = squared_diff.mean()
    else:
        mse = squared_diff.mean(dim=dim)

    rmse_result: xr.Dataset = np.sqrt(mse)  # type: ignore[assignment]

    return rmse_result


def calc_pdf_metric_nparray(
    arr0: np.ndarray,
    arr1: np.ndarray,
    axis: int | None = 0,
    bins_arr: np.ndarray | None = None,
    nbins: int = 100,
    bin_target_only: bool = False,
    typedistance: str = "pss",
    printBins: bool = False,
    min_freq: int = 10,
) -> np.ndarray | float:
    """
    Calculate distribution comparison metrics between two N-dimensional numpy arrays.

    Computes bin probabilities, then calculates the requested distance or similarity
    metric using the specified method.

    Parameters
    ----------
    arr0 : np.ndarray
        First N-dimensional array (Target).
    arr1 : np.ndarray
        Second N-dimensional array (must have same shape as arr0) (Prediction).
    axis : int or None, default 0
        None : compute over all axes.
        0 : compute over 0th axis so output has N-1 dimensions.
    bins_arr : np.ndarray or None, optional
        Explicitly define bins for histogram (default method, else use nbins).
    nbins : int, default 100
        Define bins using default hist method with nbins based on target arr0.
    bin_target_only : bool, default False
        True: for nbins option, base on target truth range only.
        False: for nbins option, base on truth and prediction range.
    typedistance : str, default 'pss'
        Define type of distance or score to compute ('pss', 'jsd', 'fcss', 'lhd', 'alhd', 'wdh', 'wd').
    printBins : bool, default False
        True: Option to print info regarding number bins and edge values.
    min_freq : int, default 10
        Option for FCSS/LHD: filter out bins/frequencies <= min_freq.

    Returns
    -------
    np.ndarray or float
        Array of distribution comparison metrics or a single float.
    """
    distance_list = ["pss", "jsd", "fcss", "lhd", "alhd", "wdh", "wd"]
    if typedistance not in distance_list:
        raise ValueError(f"typedistance not recognised: {typedistance}")

    axis_list = [None, 0]
    if axis not in axis_list:
        raise ValueError(f"axis not recognised as None or 0: {axis}")

    dist_out: np.ndarray | float = np.nan

    if not isinstance(bins_arr, np.ndarray):
        bins_arr = None

    if bins_arr is None:
        if nbins < 1:
            raise ValueError(f"nbins not a positive integer: {nbins}")
        nbins = int(nbins)
        arr_min = np.array([arr0.min(), arr1.min()]).min()
        arr_max = np.array([arr0.max(), arr1.max()]).max()
        if bin_target_only:
            # Fit bins to target truth only so same for all prediction model comparisons
            bins_arr = np.linspace(arr_min, arr_max, nbins + 1)
        else:
            # Fit bins to target truth and prediction so entire distributions are captured
            bins_arr = np.linspace(arr0.min(), arr0.max(), nbins + 1)

    if printBins and bins_arr is not None:
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

        elif typedistance == "jsd":
            phist, pedge = np.histogram(arr0, bins=bins_arr, density=True)
            qhist, qedge = np.histogram(arr1, bins=bins_arr, density=True)

            # Normalise so sums to 1
            phist = phist / phist.sum()
            qhist = qhist / qhist.sum()

            dist_out = float(jensenshannon(phist, qhist))

        elif typedistance == "fcss":
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

        elif typedistance == "lhd":
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

            dist_out = float(np.sqrt(np.nanmean(log_diff)))

        elif typedistance == "alhd":
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

            dist_out = float(np.nanmean(log_diff))

        elif typedistance == "wdh":
            phist, pedge = np.histogram(arr0, bins=bins_arr, density=True)
            qhist, qedge = np.histogram(arr1, bins=bins_arr, density=True)

            # Normalise so sums to 1
            phist = phist / phist.sum(axis=0)
            qhist = qhist / qhist.sum(axis=0)

            # Use bin midpoints as representative values
            bin_centres = (bins_arr[:-1] + bins_arr[1:]) / 2

            # scipy's wasserstein_distance accepts weighted samples:
            # pass bin centers as "samples" and normalized counts as weights
            dist_out = float(
                scipy_wasserstein_distance(
                    bin_centres, bin_centres, u_weights=phist, v_weights=qhist
                )
            )

        elif typedistance == "wd":
            arr0_flat = arr0.reshape(np.prod(arr0.shape))
            arr1_flat = arr1.reshape(np.prod(arr1.shape))
            dist_out = float(scipy_wasserstein_distance(arr0_flat, arr1_flat))

    elif axis == 0:
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

        elif typedistance == "jsd":
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

        elif typedistance == "fcss":
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

        elif typedistance == "lhd":
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

            dist_out = np.sqrt(np.nanmean(log_diff, axis=0)).astype(float)

        elif typedistance == "alhd":
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

            dist_out = np.nanmean(log_diff, axis=0).astype(float)

        elif typedistance == "wdh":
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
                dist_out[ii] = scipy_wasserstein_distance(
                    bin_centres,
                    bin_centres,
                    u_weights=phist_flat[:, ii],
                    v_weights=qhist_flat[:, ii],
                )

            dist_out = dist_out.reshape(arr0.shape[1:])

        elif typedistance == "wd":
            dist_out = np.zeros(np.prod(arr0.shape[1:]))
            arr0_flat = arr0.reshape(arr0.shape[0], np.prod(arr0.shape[1:]))
            arr1_flat = arr1.reshape(arr1.shape[0], np.prod(arr1.shape[1:]))
            for ii in range(np.prod(arr0.shape[1:])):
                dist_out[ii] = scipy_wasserstein_distance(
                    arr0_flat[:, ii], arr1_flat[:, ii]
                )

            dist_out = dist_out.reshape(arr0.shape[1:])

    return dist_out


def calc_pdf_metric_xr(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    dim: str | tuple[str, ...] | None = None,
    index_fn: Callable[..., xr.Dataset | xr.DataArray] | None = None,
    season: str | None = None,
    member_name: str | None = None,
    emean: bool = False,
    bins_arr: np.ndarray | None = None,
    nbins: int = 100,
    bin_target_only: bool = False,
    printBins: bool = False,
    typedistance: str = "pss",
    min_freq: int = 10,
    **index_kwargs: Any,
) -> xr.Dataset:
    """
    Compute distribution comparison metrics between two xarray Datasets.

    Filters by season, applies any index functions if specified, detects and handles
    ensemble member dimensions, and flattens along the specified dimensions before
    computing distribution comparison metrics.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    dim : str or tuple of str or None, optional
        Dimensions to compute distance over (e.g. ('time',)).
        None: compute distance over all data flattened.
    index_fn : Callable[..., xr.Dataset | xr.DataArray] or None, optional
        Optional function computing a desired index before metric calculations.
    season : str | None, optional
        Optional season or month standard name filter.
    member_name : str | None, optional
        Name of member dimension (ensemble prediction).
    emean : bool, default False
        Option to compute ensemble mean of prediction over member_name dimension.
    bins_arr : np.ndarray or None, optional
        Explicitly defined bins for histogram.
    nbins : int, default 100
        Number of bins for histogram.
    bin_target_only : bool, default False
        True: base nbins range on target truth range only.
        False: base range on truth and prediction range combined.
    printBins : bool, default False
        Option to print details regarding bins and edges.
    typedistance : str, default 'pss'
        Type of distance or score to compute ('pss', 'jsd', 'fcss', 'lhd', 'alhd', 'wdh', 'wd').
    min_freq : int, default 10
        Minimum frequency threshold to filter out sparse bins in LHD.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.

    Returns
    -------
    xr.Dataset
        Dataset of distribution comparison metrics.
    """
    distance_list = ["pss", "jsd", "fcss", "lhd", "alhd", "wdh", "wd"]
    if typedistance not in distance_list:
        raise ValueError(f"typedistance not recognised: {typedistance}")

    target_xr: xr.DataArray
    prediction_xr: xr.DataArray

    if index_fn is None:
        target_xr = filter_by_monthseason(x0, season)[var]
        prediction_xr = filter_by_monthseason(x1, season)[var]
    else:
        call_kwargs = dict(index_kwargs)
        if season is not None and "season" not in call_kwargs:
            call_kwargs["season"] = season
        if var is not None and "var" not in call_kwargs:
            call_kwargs["var"] = var

        x0_idx = index_fn(x0, **call_kwargs)
        x1_idx = index_fn(x1, **call_kwargs)

        if isinstance(x0_idx, xr.Dataset):
            target_xr = x0_idx[var]
        else:
            target_xr = x0_idx

        if isinstance(x1_idx, xr.Dataset):
            prediction_xr = x1_idx[var]
        else:
            prediction_xr = x1_idx

    dim_tuple: tuple[str, ...]
    if dim is None:
        dim_tuple = tuple(prediction_xr.dims)
    elif isinstance(dim, str):
        dim_tuple = (dim,)
    else:
        dim_tuple = dim

    # Detect if dim dimensions present in prediction
    pred_dims = list(prediction_xr.dims)
    for dname in dim_tuple:
        if dname not in pred_dims:
            raise ValueError(
                f"dim dimension '{dname}' not found in dims {pred_dims}. "
                f"Expected something like 'time', 'member', etc."
            )

    # Detect if ensemble dim present in prediction but not target, and expand
    # target to match using expand_dims + broadcast
    extra_member_dim: str | None = None
    if (
        member_name is not None
        and member_name in prediction_xr.dims
        and member_name not in target_xr.dims
    ):
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
            f"Prediction ({prediction_xr.values.shape}). "
            f"Specify member_name if ensemble prediction."
        )

    # Identify output dimensions i.e. those not in dim dimensions
    output_dims = [d for d in pred_dims if d not in dim_tuple]

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
    ordered_dims = list(dim_tuple) + output_dims
    target_xr = target_xr.transpose(*ordered_dims)
    prediction_xr = prediction_xr.transpose(*ordered_dims)

    # Flatten all dim dims into a single axis so can compute over 0th dim
    dimover_shape = target_xr.values.shape[0 : len(dim_tuple)]  # e.g. [member, time]
    output_shape = target_xr.values.shape[len(dim_tuple) :]  # e.g. [lat, lon]
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
    for d in output_dims:
        if d in prediction_xr.coords:
            dist_out_xr[d] = prediction_xr[d]

    return dist_out_xr
