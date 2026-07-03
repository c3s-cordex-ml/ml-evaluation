from __future__ import annotations

from typing import Any, Callable

import numpy as np
import xarray as xr
from numpy import fft
from scipy.stats import wasserstein_distance_nd
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
