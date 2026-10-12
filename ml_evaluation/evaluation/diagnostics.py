from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np
import scipy as sp
import xarray as xr
from numpy import fft
from scipy.stats import (
    wasserstein_distance as scipy_wasserstein_distance,
)
from scipy.spatial.distance import jensenshannon

from ml_evaluation.evaluation.utils import (
    filter_by_monthseason,
    radial_average,
    xr_spearman_corr,
)
from ml_evaluation.transformations import spatial_dims


def bias_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    index_fn: Callable[..., xr.Dataset | xr.DataArray],
    season: str | None = None,
    relative: bool = False,
    output_index=False,
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
    output_index : bool, default False
        If True, output indices too: bias, x0_index, x1_index
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
        bias = bias / x0_index

    if output_index:
        return bias, x0_index, x1_index

    else:
        return bias


def ratio_index(
    x0: xr.Dataset,
    x1: xr.Dataset,
    index_fn: Callable[..., xr.Dataset | xr.DataArray],
    season: str | None = None,
    output_index=False,
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
    output_index : bool, default False
        If True, output indices too: bias, x0_index, x1_index
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

    if output_index:
        return ratio, x0_index, x1_index

    else:
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
        raise ValueError("Error: mae and mse can't both be True")

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
    **index_kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """
    Compute the Correlation between x0 and x1 for a supplied index function.

    Default is Pearson's correlation, with option for Spearman's rank correlation.

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
    if pcorr:
        corr_result = corr_result + xr.corr(x1_index[var], x0_index[var], dim=dim)
    elif srank:
        corr_result = corr_result + xr_spearman_corr(
            x1_index[var], x0_index[var], dim=dim
        )

    return corr_result


def bias_multivariable_correlation(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var_x: str,
    var_y: str,
    season: str | None = None,
    srank: bool = False,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Compute correlations between two variables for x0 and x1 datasets.

    Default is Pearson's correlation, with option for Spearman's rank correlation.

    Returns the correlations and bias between the reference and predicted datasets.

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

    Returns
    -------
    tuple of xr.DataArray
        Dataset containing the correlation bias and individual correlations.

    Raises
    ------
    ValueError
        If no matches found for dimensions.
    """
    pcorr = True
    if srank:
        pcorr = False

    x0_filtered = filter_by_monthseason(x0, season)
    x1_filtered = filter_by_monthseason(x1, season)

    dim = ("time",)

    dim_list = list(x0_filtered.dims)
    matches = [dim1 for dim1 in dim_list if dim1 in dim]
    if not matches:
        raise ValueError(f"Error: no matches found for {list(dim)} in {dim_list}")

    x0_corr_xarray = x0_filtered[var_x].mean(dim=dim) * 0.0
    x1_corr_xarray = x1_filtered[var_x].mean(dim=dim) * 0.0

    # xr.corr requires DataArray
    if pcorr:
        x0_corr_xarray = x0_corr_xarray + xr.corr(
            x0_filtered[var_x], x0_filtered[var_y], dim=dim
        )
        x1_corr_xarray = x1_corr_xarray + xr.corr(
            x1_filtered[var_x], x1_filtered[var_y], dim=dim
        )
    elif srank:
        x0_corr_xarray = x0_corr_xarray + xr_spearman_corr(
            x0_filtered[var_x], x0_filtered[var_y], dim=dim
        )
        x1_corr_xarray = x1_corr_xarray + xr_spearman_corr(
            x1_filtered[var_x], x1_filtered[var_y], dim=dim
        )

    correlation_bias = x1_corr_xarray - x0_corr_xarray

    output_ds = (
        x0_corr_xarray,
        x1_corr_xarray,
        correlation_bias,
    )

    return output_ds


def psd(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    season: str | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
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
    tuple of xr.Dataset
        Power spectral densities for x0 and x1.
    """
    # Spot member dimension name if exists
    member_dim = None
    x0_dims = set(x0.dims)
    x1_dims = set(x1.dims)
    extra_dim = x1_dims - x0_dims
    if len(extra_dim) > 1:
        raise ValueError("Error: can't handle x1 with more than one extra dim than x0")
    if len(extra_dim) == 1:
        member_dim = list(extra_dim)[0]

    x0_filtered = filter_by_monthseason(x0, season)
    x1_filtered = filter_by_monthseason(x1, season)

    x0_da = x0_filtered[var]
    x1_da = x1_filtered[var]

    # Construct dims for psd output
    x0_da_dims = list(x0_da.dims)
    x1_da_dims = list(x1_da.dims)
    x0_new_dims = x0_da_dims[:-3] + ["wavenumber"]
    x1_new_dims = x1_da_dims[:-3] + ["wavenumber"]

    x0_np = np.nan_to_num(x0_da.values, nan=0.0)  # 0 only sensible for precip
    x1_np = np.nan_to_num(x1_da.values, nan=0.0)

    fft_x0 = fft.fftshift(fft.fft2(x0_np, axes=(-2, -1)), axes=(-2, -1))
    fft_x1 = fft.fftshift(fft.fft2(x1_np, axes=(-2, -1)), axes=(-2, -1))

    power_x0 = np.abs(fft_x0) ** 2
    power_x1 = np.abs(fft_x1) ** 2
    
    psd_x0_list = [radial_average(p) for p in power_x0]
    # psd_x1_list = [radial_average(p) for p in power_x1]
    # ensemble -> too many values to unpack as expects p shape [lat, lon]
    # flatten power_x1 then put back together
    flat_power_x1 = power_x1.reshape(
        (np.prod(power_x1.shape[:-2]), *power_x1.shape[-2:])
    )
    psd_x1_list = [radial_average(p) for p in flat_power_x1]
    psd_x1_np = np.array(psd_x1_list)
    psd_x1_np = psd_x1_np.reshape(*power_x1.shape[:-2], psd_x1_np.shape[-1])

    avg_psd_x0 = np.mean(psd_x0_list, axis=-2)
    avg_psd_x1 = np.mean(psd_x1_np, axis=-2)

    psd_x0_da = xr.DataArray(avg_psd_x0, dims=x0_new_dims, name="PSD_x0")
    if member_dim is None:
        psd_x1_da = xr.DataArray(avg_psd_x1, dims=x1_new_dims, name="PSD_x1")
    else:
        psd_x1_da = xr.DataArray(
            avg_psd_x1,
            dims=x1_new_dims,
            coords={"member": x1.coords["member"]},
            name="PSD_x1",
        )

    psd_x0_ds = xr.Dataset({var: psd_x0_da})
    psd_x1_ds = xr.Dataset({var: psd_x1_da})

    return psd_x0_ds, psd_x1_ds


def ralsd(psd_x0: xr.Dataset, psd_x1: xr.Dataset, var: str) -> xr.Dataset:
    """
    Compute the Radially Averaged Log Spectral Distance (RALSD).

    RALSD measures the distance between two radially averaged power spectra
    in log space. Lower values indicate better agreement.

    Parameters
    ----------
    psd_x0 : xr.Dataset
        Power spectral density of the ground truth (as returned by psd function).
    psd_x1 : xr.Dataset
        Power spectral density of the prediction (as returned by psd function).
    var : str
        Variable name to analyse.

    Returns
    -------
    xr.Dataset
        Scalar RALSD value.
    """
    # ratio = psd_x0[var].values / np.maximum(psd_x1[var].values, 1e-30)
    # log_ratio = 10.0 * np.log10(np.maximum(ratio, 1e-30))
    # res = xr.Dataset({var: float(np.mean(log_ratio**2))})

    # check update for enxemble also copes with single member
    ratio = psd_x0[var] / psd_x1[var].clip(min=1e-30)
    log_ratio = 10.0 * np.log10(ratio.clip(min=1e-30))
    log_ratio_da = cast(xr.DataArray, log_ratio)
    # res = xr.Dataset({var: (log_ratio_da**2).mean(dim="wavenumber")}) # ERROR Should be sqrt!
    res = xr.Dataset({var: np.sqrt((log_ratio_da**2).mean(dim="wavenumber"))})

    return res


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

    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade

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
        raise ValueError(f"Error: typedistance not recognised: {typedistance}")

    axis_list = [None, 0]
    if axis not in axis_list:
        raise ValueError(f"Error: axis not recognised as None or 0: {axis}")

    dist_out: np.ndarray | float = np.nan

    if not isinstance(bins_arr, np.ndarray):
        bins_arr = None

    if bins_arr is None:
        if nbins < 1:
            raise ValueError(f"Error: nbins not a positive integer: {nbins}")
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
            # dist_out = dist_out.astype(float)

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
            arr0_flat_2d = arr0.reshape(arr0.shape[0], np.prod(arr0.shape[1:]))
            arr1_flat_2d = arr1.reshape(arr1.shape[0], np.prod(arr1.shape[1:]))
            for ii in range(np.prod(arr0.shape[1:])):
                dist_out[ii] = scipy_wasserstein_distance(
                    arr0_flat_2d[:, ii], arr1_flat_2d[:, ii]
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

    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade

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
        raise ValueError(f"Error: typedistance not recognised: {typedistance}")

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
        dim_tuple = tuple(str(d) for d in prediction_xr.dims)
    elif isinstance(dim, str):
        dim_tuple = (dim,)
    else:
        dim_tuple = dim

    # Detect if dim dimensions present in prediction
    pred_dims = list(prediction_xr.dims)
    for dname in dim_tuple:
        if dname not in pred_dims:
            raise ValueError(
                f"Error; dim dimension '{dname}' not found in dims {pred_dims}. "
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
            f"Error: Shape mismatch: Target ({target_xr.values.shape}), "
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


def calc_histdata_xr(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    index_fn: Callable[..., xr.Dataset | xr.DataArray] | None = None,
    season: str | None = None,
    bins_arr: np.ndarray | None = None,
    nbins: int = 100,
    bin_target_only: bool = False,
    **index_kwargs: Any,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Compute histogram for two xarray Datasets.

    Filters by season, applies any index functions if specified, and
    flattens across time, lat, lon.

    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    index_fn : Callable[..., xr.Dataset | xr.DataArray] or None, optional
        Optional function computing a desired index before metric calculations.
    season : str | None, optional
        Optional season or month standard name filter.
    bins_arr : np.ndarray or None, optional
        Explicitly defined bins for histogram.
    nbins : int, default 100
        Number of bins for histogram.
    bin_target_only : bool, default False
        True: base nbins range on target truth range only.
        False: base range on truth and prediction range combined.
    **index_kwargs : Any
        Additional keyword arguments passed to the index function.

    Returns
    -------
    xr.Dataset, xr.Dataset, xr.Dataset
        Datasets of histogram output.
    """

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

    arr0 = target_xr.values.flatten()
    arr1 = prediction_xr.values.flatten()

    if not isinstance(bins_arr, np.ndarray):
        bins_arr = None

    if bins_arr is None:
        if nbins < 1:
            raise ValueError(f"Error: nbins not a positive integer: {nbins}")
        nbins = int(nbins)
        arr_min = np.array([arr0.min(), arr1.min()]).min()
        arr_max = np.array([arr0.max(), arr1.max()]).max()
        if bin_target_only:
            # Fit bins to target truth only so same for all prediction model comparisons
            bins_arr = np.linspace(arr_min, arr_max, nbins + 1)
        else:
            # Fit bins to target truth and prediction so entire distributions are captured
            bins_arr = np.linspace(arr0.min(), arr0.max(), nbins + 1)

    phist, pedge = np.histogram(arr0, bins=bins_arr, density=True)
    qhist, qedge = np.histogram(arr1, bins=bins_arr, density=True)

    phist_ds = xr.Dataset({var: ("bins", np.append(phist, np.nan))})
    qhist_ds = xr.Dataset({var: ("bins", np.append(qhist, np.nan))})
    edge_ds = xr.Dataset({var: ("bins", pedge)})

    return phist_ds, qhist_ds, edge_ds


def crps_xr(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    dim: str | tuple[str, ...] | None = None,
    season: str | None = None,
    fair_est: bool = True,
) -> xr.Dataset:
    """
    Compute the crps for two xarray Datasets,
    using the form:
    CRPS(X, y) = E|X - y| - 1/2 * E|X - X'|


    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    dim : str or tuple of str or None, optional
        Dimension(s) along which to aggregate over.
        None -> output has same dimensions as x0
        Note, subsequent averages should define dim order for
        floating point accuracy wrt crps_xr with dim ne None,
        e.g. crps.mean(dim=("time", "lat", "lon"))
    season : str | None, optional
        Season to filter before computing the PSD.
    fair_est : bool
        True includes option to adjust to fair spread estimate,
        i.e. ignores pairs i=j on diagonal.

    Returns
    -------
    xr.Dataset
        crps.

    Raises
    ------
    ValueError
        If can't determine member dimension
    """
    # Spot member dimension name if exists
    member_dim = None
    x0_dims = set(x0.dims)
    x1_dims = set(x1.dims)
    extra_dim = x1_dims - x0_dims
    if len(extra_dim) > 1:
        raise ValueError("Error: can't handle x1 with more than one extra dim than x0")
    if len(extra_dim) == 1:
        member_dim = list(extra_dim)[0]

    x0_filtered = filter_by_monthseason(x0, season)
    x1_filtered = filter_by_monthseason(x1, season)

    x0_da = x0_filtered[var]
    x1_da = x1_filtered[var]

    nmembers = x1_da[member_dim].size
    if nmembers < 2:
        fair_est = False

    # Accuracy
    accuracy = np.abs(x1_da - x0_da).mean(member_dim)
    # Spread
    pred_i = x1_da
    pred_j = x1_da.rename({member_dim: member_dim + "_2"})
    spread = np.abs(pred_i - pred_j).mean([member_dim, member_dim + "_2"])
    if fair_est:
        spread = spread * (nmembers / (nmembers - 1))

    crps_da = accuracy - 0.5 * spread

    if dim is not None:
        crps_da = crps_da.mean(dim=dim)

    crps_ds = xr.Dataset({var: crps_da})

    return crps_ds


def data_to_binary(
    x: xr.Dataset,
    var: str,
    threshold: xr.Dataset | xr.DataArray | int | float,
    relation: str = "gt",
) -> xr.Dataset:
    ops = {
        "ge": lambda a, t: a >= t,
        "gt": lambda a, t: a > t,
        "le": lambda a, t: a <= t,
        "lt": lambda a, t: a < t,
    }
    if relation not in ops:
        raise ValueError(f"relation must be one of {list(ops)}, got {relation!r}")

    threshold_new: xr.DataArray | int | float
    if isinstance(threshold, (int, float)):
        threshold_new = threshold
    elif isinstance(threshold, (xr.Dataset, xr.DataArray)):
        # Pull out the matching variable var so work with a DataArray
        thresh_da = threshold[var] if isinstance(threshold, xr.Dataset) else threshold

        sdims = spatial_dims(x, var)
        if any(x.sizes[d] != thresh_da.sizes.get(d) for d in sdims):
            raise ValueError(
                f"Spatial shape mismatch: input "
                f"({', '.join(str(x.sizes[d]) for d in sdims)}) "
                f"!= threshold "
                f"({', '.join(str(thresh_da.sizes.get(d)) for d in sdims)}) "
                f"on dims {sdims}"
            )

        dims_thresh = set(thresh_da.dims)
        other_dims_thresh = dims_thresh - set(sdims)
        if not other_dims_thresh:
            threshold_new = thresh_da
        else:
            # Broadcast the threshold back onto actual time axis via dayofyear
            doy = x["time"].dt.dayofyear
            threshold_new = thresh_da.sel(dayofyear=doy)
    else:
        threshold_new = threshold

    x_binary = xr.Dataset({var: ops[relation](x[var], threshold_new).astype(float)})
    return x_binary


def bs_xr(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    threshold: xr.Dataset | int | float,
    relation: str = "gt",
    dim: str | tuple[str, ...] | None = None,
    season: str | None = None,
    fair_est: bool = True,
) -> xr.Dataset:
    """
    Compute the Brier Score (BS) for two xarray Datasets by first
    converting truth and prediction datasets to a binary event fields
    (relative to threshold) and then calling `crps_xr` on these
    binary fields, as CRPS == Brier Score for probabilistic
    forecasts (p in [0, 1]) of a binary event (y in {0, 1}):

        E|X - y| - 1/2*E|X - X'| == (p - y)^2

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    threshold : xr.Dataset | int | float
        Threshold used to define the binary event.
    relation : str, default 'gt'
        Relationship of event to threshold: 'gt', 'ge', 'lt', 'le'.
    dim : str or tuple of str or None, optional
        Dimension(s) along which to aggregate over.
        None -> output has same dimensions as x0 (as crps_xr).
    season : str | None, optional
        Season to filter before computing the BS.
    fair_est : bool
        True adjusts to the fair (unbiased for finite ensemble size)
        estimator, ignoring pairs i=j on the diagonal. Passed through
        to crps_xr.

    Returns
    -------
    xr.Dataset
        Brier Score (bs).

    Raises
    ------
    ValueError
        If comparison is invalid, or crps_xr can't determine the
        member dimension.
    """

    x0_filtered = filter_by_monthseason(x0, season)
    x1_filtered = filter_by_monthseason(x1, season)

    # Convert both fields to binary (0/1) event indicators, preserving
    # all dims/coords (including the member dim on x1).
    x0_binary = data_to_binary(x0_filtered, var, threshold, relation)
    x1_binary = data_to_binary(x1_filtered, var, threshold, relation)

    # CRPS of binary fields == (fair) Brier Score.
    bs_ds = crps_xr(
        x0_binary, x1_binary, var, dim=dim, season=season, fair_est=fair_est
    )

    return bs_ds


def mean_spread_skill_ratio(
    x0: xr.Dataset,
    x1: xr.Dataset,
    var: str,
    dim: str | tuple[str, ...] | None = None,
    season: str | None = None,
) -> xr.Dataset | xr.DataArray:
    """
    Compute the spread skill ratio for an ensemble prediction.
    For ensembles with at least 2 members.

    Parameters
    ----------
    x0 : xr.Dataset
        Ground truth dataset.
    x1 : xr.Dataset
        Predicted dataset.
    var : str
        Variable name to analyse.
    dim : str or tuple of str or None, optional
        Dimension(s) along which to aggregate over.
    season : str | None, optional
        Season to filter before computing the index.

    Returns
    -------
    xr.Dataset or xr.DataArray
        spread skill ratio
        e.g. dim=['time'] -> output dims ['lat','lon']
        ~1 = well calibrated
        >1 => over-dispersive
        <1 => under-dispersive

    Raises
    ------
    ValueError
        If can't determine member dimension
    """
    # Spot member dimension name if exists
    member_dim = None
    x0_dims = set(x0.dims)
    x1_dims = set(x1.dims)
    extra_dim = x1_dims - x0_dims
    if len(extra_dim) > 1:
        raise ValueError("Error: can't handle x1 with more than one extra dim than x0")
    if len(extra_dim) == 1:
        member_dim = list(extra_dim)[0]

    if member_dim is None:
        raise ValueError("Error: can't compute ensemble spread if no member dimension")

    nmem = x1.sizes[member_dim]
    if nmem < 2:
        raise ValueError("Error: can't compute ensemble spread if < 2 members")

    x0_filtered = filter_by_monthseason(x0, season)
    x1_filtered = filter_by_monthseason(x1, season)

    x0_da = x0_filtered[var]
    x1_da = x1_filtered[var]

    # Ensemble mean and spread with correction for small samples
    ens_mean = x1_da.mean(member_dim)
    ens_var = x1_da.var(member_dim, ddof=0) * (nmem + 1) / (nmem - 1)

    # Error of ensemble mean vs target
    error = ens_mean - x0_da
    sq_error = error**2

    rmserror = np.sqrt(sq_error.mean(dim=dim))
    mean_spread = np.sqrt(ens_var.mean(dim=dim))

    spread_skill_ratio = mean_spread / rmserror

    spread_skill_ratio_ds = xr.Dataset({var: spread_skill_ratio})

    return spread_skill_ratio_ds


def compute_sal_xr(
    prediction_xr,
    target_xr,
    var,
    eThreshFix=None,
    eThreshPrFix=None,
    thr_quantile=None,
    thr_factor=None,
    minFac=None,
    minsize=0,
    structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    lat_name="lat",
    lon_name="lon",
    member_name=None,
):
    """
    Compute SAL score wrt given threhold(s)
    Based on Wernli et al., 2008 and 2009
    https://journals.ametsoc.org/view/journals/mwre/136/11/2008mwr2415.1.xml
    https://journals.ametsoc.org/view/journals/wefo/24/6/2009waf2222271_1.xml

    *Assumes input fields are non-negative like precipitation. May behave
    unexpectedly if negative values exist.

    Extended to allow the use of:
    - fixed thresholds independent of input data
    - different minimum object size thresholds
    - different structure for neighbour definitions
    **If prediction_xr is an ensemble, member_name must be specified to
    match the name of the member dimensions and lat_name and lon_name
    must also be correctly specified

    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade
    License: MIT (see LICENSE)

    Parameters:
    -----------
    prediction_xr : xr.DataArray
        Prediction field data. Must contain 'lat' and 'lon' dimensions,
        plus any number of leading dimensions (e.g. [time, lat, lon] or
        [time, member, lat, lon]).
    target_xr : xr.DataArray
        Target field data, same shape as prediction_xr but without
        ensemble dimension.
    var : str
        Variable name to analyse.
    eThreshFix : float | None
        Fixed threshold to be used to define event (same units as target)
        If eThreshFix value given, this overrides quantile based options
    eThreshPrFix : float | None
        Fixed threshold to be used to define event (same units as target)
        If None, uses eThreshFix.
    thr_quantile : float | None
        Quantile value in [0.0, 1.0] used to compute threshold to define
        event, as Wernli et al. 2009 (they use thr_quantile=0.95)
    thr_factor : float | None
        Factor to reduce the quantile by, as Wernli et al. 2008 & 2009
        (they use thr_factor=1/15)
    minFac : float | str | None
        Option to mask data less than threshold=minFac before computing
        quantile, as Wernli et al. 2009 (they use 0.1 mm for precip)
        Special case: minFac='min' implies use min value of field,
        chosen to align with option in pysteps.
    minsize : int = 0
        Option to ignore options with size (no. grid points) < minsize
    structure : numpy.ndarray, dtype=int, shape [3, 3]
        This array defines what are classed as neighbouring grid points.
        2 Options:
        np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]], dtype=int) # Orthogonal-
         only (default)
        np.ones((3, 3), dtype=int) # Orthogonal and diagonal
    lat_name : str
        Define name of latitude dimension e.g. 'lat' or 'y'
    lon_name : str
        Define name of longitude dimension e.g. 'lon' or 'x'
    member_name : str | None
        Define name of member dimension (ensemble prediction) e.g.
        'member' or 'realization'

    Returns:
    --------
    Tuple of xr.Dataset
        Arrays of SAL score elements for input target and prediction field pairs
        (np.nan if no objects found). Has the same leading dimensions as
        the input arrays, e.g. [time] or [time, member].
        (structure, amplitude, location, location_1, location_2,
        number of objects found in target field,
        number of objects found in prediction field)

    See Also
    --------

    scipy.ndimage.label :
    Identify objects using sp.ndimage.label
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html

    Method checked against pysteps
    https://pysteps.readthedocs.io/en/stable/
    Using default pysteps parameters, compute_sal() has identical results
    except:
    - pysteps rejects objects that are within 10 pixels of another object
      (distance between location of peaks),
      i.e. doesn't appear to merge such objects as expected?
    - compute_sal keeps objects as separate, regardless of proximity

    """

    # Validate spatial dim names exist in the input
    for name, xr_obj in [
        (lat_name, target_xr),
        (lon_name, target_xr),
        (lat_name, prediction_xr),
        (lon_name, prediction_xr),
    ]:
        if name not in xr_obj.dims:
            raise ValueError(
                f"Spatial dimension '{name}' not found in dims {list(xr_obj.dims)}. "
                f"Use lat_name/lon_name to specify the correct dimension names."
            )

    spatial_dims = {lat_name, lon_name}

    # Detect if ensemble dim present in prediction but not target, and expand
    # target to match using expand_dims + broadcast
    extra_member_dim = None
    if member_name in prediction_xr.dims and member_name not in target_xr.dims:
        extra_member_dim = member_name

    if extra_member_dim is not None:
        target_xr = target_xr.expand_dims(
            {extra_member_dim: prediction_xr[extra_member_dim]}
        )
        target_xr = target_xr.broadcast_like(prediction_xr)

    if target_xr.values.shape != prediction_xr.values.shape:
        raise ValueError(
            f"Shape mismatch: Target ({target_xr.values.shape}), "
            f"Prediction ({prediction_xr.values.shape})"
        )

    # Identify spatial vs leading dimensions
    all_dims = list(prediction_xr.dims)
    leading_dims = [d for d in all_dims if d not in spatial_dims]

    # --- Special case: no leading dimensions, single [lat, lon] field ---
    if not leading_dims:
        sal_tmp, _, _ = compute_sal(
            prediction_xr.values,
            target_xr.values,
            eThreshFix=eThreshFix,
            eThreshPrFix=eThreshPrFix,
            thr_factor=thr_factor,
            thr_quantile=thr_quantile,
            minFac=minFac,
            minsize=minsize,
            structure=structure,
        )
        sal_s_xr = xr.Dataset({var: sal_tmp.sal_s})
        sal_a_xr = xr.Dataset({var: sal_tmp.sal_a})
        sal_l_xr = xr.Dataset({var: sal_tmp.sal_l})
        sal_l1_xr = xr.Dataset({var: sal_tmp.sal_l1})
        sal_l2_xr = xr.Dataset({var: sal_tmp.sal_l2})
        sal_targ_num_xr = xr.Dataset({var: sal_tmp.sal_targ_num})
        sal_pred_num_xr = xr.Dataset({var: sal_tmp.sal_pred_num})

        sal_all = (
            sal_s_xr,
            sal_a_xr,
            sal_l_xr,
            sal_l1_xr,
            sal_l2_xr,
            sal_targ_num_xr,
            sal_pred_num_xr,
        )

        return sal_all

    # Transpose to ensure leading dims come first, spatial dims last:
    # [...leading, lat_name, lon_name]
    ordered_dims = leading_dims + [lat_name, lon_name]
    target_xr = target_xr.transpose(*ordered_dims)
    prediction_xr = prediction_xr.transpose(*ordered_dims)

    # Flatten all leading dims into a single axis for iteration
    leading_shape = target_xr.values.shape[
        :-2
    ]  # e.g. (ntime,) or (ntime, nrealization)
    spatial_shape = target_xr.values.shape[-2:]  # (nlat, nlon)
    n_total = int(np.prod(leading_shape))

    target_flat = target_xr.values.reshape(n_total, *spatial_shape)
    prediction_flat = prediction_xr.values.reshape(n_total, *spatial_shape)

    # Check for negative values
    targ_min = target_flat.min()
    pred_min = prediction_flat.min()
    if targ_min < 0 or pred_min < 0:
        print("WARNING: input fields contain negative values. SAL score function ")
        print("assumes non-negative values, so may behave unexpectedly.")

    sal_s = []
    sal_a = []
    sal_l = []
    sal_l1 = []
    sal_l2 = []
    sal_targ_n = []
    sal_pred_n = []

    for tcount in range(n_total):
        sal_tmp, targ_obj, pred_obj = compute_sal(
            prediction_flat[tcount],
            target_flat[tcount],
            eThreshFix=eThreshFix,
            eThreshPrFix=eThreshPrFix,
            thr_factor=thr_factor,
            thr_quantile=thr_quantile,
            minFac=minFac,
            minsize=minsize,
            structure=structure,
        )

        sal_s.append(sal_tmp.sal_s)
        sal_a.append(sal_tmp.sal_a)
        sal_l.append(sal_tmp.sal_l)
        sal_l1.append(sal_tmp.sal_l1)
        sal_l2.append(sal_tmp.sal_l2)
        sal_targ_n.append(sal_tmp.sal_targ_num)
        sal_pred_n.append(sal_tmp.sal_pred_num)

    # Reshape results back to the leading dimensions shape
    def to_array(lst):
        return np.array(lst).reshape(leading_shape)

    sal_s_xr = xr.Dataset({var: (leading_dims, to_array(sal_s))})
    sal_a_xr = xr.Dataset({var: (leading_dims, to_array(sal_a))})
    sal_l_xr = xr.Dataset({var: (leading_dims, to_array(sal_l))})
    sal_l1_xr = xr.Dataset({var: (leading_dims, to_array(sal_l1))})
    sal_l2_xr = xr.Dataset({var: (leading_dims, to_array(sal_l2))})
    sal_targ_num_xr = xr.Dataset({var: (leading_dims, to_array(sal_targ_n))})
    sal_pred_num_xr = xr.Dataset({var: (leading_dims, to_array(sal_pred_n))})

    # Restore coordinates for all leading dimensions
    for dim in leading_dims:
        if dim in prediction_xr.coords:
            sal_s_xr[dim] = prediction_xr[dim]
            sal_a_xr[dim] = prediction_xr[dim]
            sal_l_xr[dim] = prediction_xr[dim]
            sal_l1_xr[dim] = prediction_xr[dim]
            sal_l2_xr[dim] = prediction_xr[dim]
            sal_targ_num_xr[dim] = prediction_xr[dim]
            sal_pred_num_xr[dim] = prediction_xr[dim]

    sal_all = (
        sal_s_xr,
        sal_a_xr,
        sal_l_xr,
        sal_l1_xr,
        sal_l2_xr,
        sal_targ_num_xr,
        sal_pred_num_xr,
    )

    return sal_all


def compute_sal(
    prediction,
    target,
    eThreshFix=None,
    eThreshPrFix=None,
    thr_quantile=None,
    thr_factor=None,
    minFac=None,
    minsize=0,
    structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    printThresh=False,
):
    """
    Compute SAL score wrt given threhold(s)
    Based on Wernli et al., 2008 and 2009
    https://journals.ametsoc.org/view/journals/mwre/136/11/2008mwr2415.1.xml
    https://journals.ametsoc.org/view/journals/wefo/24/6/2009waf2222271_1.xml

    *Assumes input fields are non-negative like precipitation. May behave
    unexpectedly if negative values exist.

    Extended to allow the use of:
    - fixed thresholds independent of input data
    - different minimum object size thresholds
    - different structure for neighbour definitions

    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade
    License: MIT (see LICENSE)

    Parameters:
    -----------
    prediction : numpy.ndarray
        Prediction field data as 2d [lat, lon] array
    target : numpy.ndarray
        Target field data, same shape as prediction
    eThreshFix : float | None
        Fixed threshold to be used to define event (same units as target)
        If eThreshFix value given, this overrides quantile based options
    eThreshPrFix : float | None
        Fixed threshold to be used to define event (same units as target)
        If None, uses eThreshFix.
    thr_quantile : float | None
        Quantile value in [0.0, 1.0] used to compute threshold to define
        event, as Wernli et al. 2009 (they use thr_quantile=0.95)
    thr_factor : float | None
        Factor to reduce the quantile by, as Wernli et al. 2008 & 2009
        (they use thr_factor=1/15)
    minFac : float | str | None
        Option to mask data less than threshold=minFac before computing
        quantile, as Wernli et al. 2009 (they use 0.1 mm for precip)
        Special case: minFac='min' implies use min value of field,
        chosen to align with option in pysteps.
    minsize : int = 0
        Option to ignore options with size (no. grid points) < minsize
    structure : numpy.ndarray, dtype=int, shape [3, 3]
        This array defines what are classed as neighbouring grid points.
        2 Options:
        np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]], dtype=int) # Orthogonal-
         only (default)
        np.ones((3, 3), dtype=int) # Orthogonal and diagonal
    printThresh : boolean
        If True: print thresholds used

    Returns:
    --------
    xr.Dataset
        SAL scores for single input target and prediction field pair
        (np.nan if no objects found)
        .sal_s = structure
        .sal_a = amplitude
        .sal_l = location
        .sal_l1 = location_1
        .sal_l2 = location_2
        .sal_targ_num = number of objects found in target field
        .sal_pred_num = number of objects found in prediction field
    xr.Dataset | None
        Object information for target (None if no objects found)
        .obj_com_lat : object centre of mass lat. (units: no. grid boxes)
        .obj_com_lon : object centre of mass lon. (units: no. grid boxes)
        .obj_max : object max value
        .obj_size : object size (no grid points)
        .obj_labeled_array : object field, integer labels (shape==target)
        Sub-select single objects:
            pred_object_xarray.sel(object_num=6).obj_max
    xr.Dataset | None
        Object information for prediction (as for target)

    See Also
    --------

    scipy.ndimage.label :
    Identify objects using sp.ndimage.label
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html

    Method checked against pysteps
    https://pysteps.readthedocs.io/en/stable/
    Using default pysteps parameters, compute_sal() has identical results
    except:
    - pysteps rejects objects that are within 10 pixels of another object
      (distance between location of peaks),
      i.e. doesn't appear to merge such objects as expected?
    - compute_sal keeps objects as separate, regardless of proximity

    Examples
    --------

    Example to sub-select single objects:
    >>> sal, targ, pred = compute_sal(target, prediction)
    >>> pred.sel(object_num=1).obj_max

    """

    if target.shape != prediction.shape:
        raise ValueError(
            f"Shape mismatch: target ({target.shape}, prediction {prediction.shape})"
        )

    # Code assumes that minsize is an integer so check (and convert)
    if minsize is None:
        minsize = 0
    if isinstance(minsize, float):
        minsize = int(minsize)

    # If fixed target threshold defined, default to using this
    if eThreshFix is not None:
        # Use fixed thresholds
        eThresh = eThreshFix
        if eThreshPrFix is None:
            eThreshPr = eThresh
        if eThreshPrFix is not None:
            eThreshPr = eThreshPrFix

    # If no fixed target threshold defined, use quantile based thresholds
    # Set default values if not supplied
    if eThreshFix is None:
        if thr_quantile is None:
            thr_quantile = 0.95
        if thr_factor is None:
            thr_factor = 1 / 15.0

        # Compute quantile-based thresholds from input data,
        # with option to first mask out very small values
        if isinstance(minFac, (int, float)):
            eThresh = thr_factor * np.nanquantile(target[target > minFac], thr_quantile)
            eThreshPr = thr_factor * np.nanquantile(
                prediction[prediction > minFac], thr_quantile
            )
        if isinstance(minFac, str):
            if minFac == "min":
                eThresh = thr_factor * np.nanquantile(
                    target[target > np.nanmin(target)], thr_quantile
                )
                eThreshPr = thr_factor * np.nanquantile(
                    prediction[prediction > np.nanmin(prediction)], thr_quantile
                )
            else:
                minFac = None
        if minFac is None:
            eThresh = thr_factor * np.nanquantile(target, thr_quantile)
            eThreshPr = thr_factor * np.nanquantile(prediction, thr_quantile)

    if printThresh is True:
        print(f"eThresh={eThresh:.3f}")
        print(f"eThreshPr={eThreshPr:.3f}")

    # Setup empty xarray dataset for the case where no objects are found
    no_object_xr = xr.Dataset(
        {
            "sal_s": np.nan,
            "sal_a": np.nan,
            "sal_l": np.nan,
            "sal_l1": np.nan,
            "sal_l2": np.nan,
            "sal_targ_num": 0,
            "sal_pred_num": 0,
        }
    )

    # If all data below event thresholds, then no objects can be found
    targ_max = target.max()
    pred_max = prediction.max()
    if targ_max <= eThresh or pred_max <= eThreshPr:
        print("No Objects Found: Event thresholds too large")
        return no_object_xr, None, None

    targ_masked: np.ndarray = (target > eThresh) & np.isfinite(target)
    targ_labeled_array, targ_num_features = sp.ndimage.label(
        targ_masked, structure=structure
    )

    pred_masked: np.ndarray = (prediction > eThreshPr) & np.isfinite(prediction)
    pred_labeled_array, pred_num_features = sp.ndimage.label(
        pred_masked, structure=structure
    )

    # Compute size of objects
    targSize = np.zeros(targ_num_features)
    for icount in range(targ_num_features):
        targSize[icount] = targ_labeled_array[targ_labeled_array == icount + 1].size
    predSize = np.zeros(pred_num_features)
    for icount in range(pred_num_features):
        predSize[icount] = pred_labeled_array[pred_labeled_array == icount + 1].size

    # Option to discard objects if too small and then renumber
    # - assumes minsize is an integer
    ReCompSize = False
    targSizemin = targSize.min()
    targSizemax = targSize.max()
    predSizemin = predSize.min()
    predSizemax = predSize.max()
    # If all objects below size thresholds, then no objects can be found
    if targSizemax < minsize or predSizemax < minsize:
        print("No Objects Found: Objects too small")
        return no_object_xr, None, None

    if minsize > 1 and targSizemin < minsize:
        targ_labeled_tmp = targ_labeled_array.copy()
        for ocount in range(targ_num_features):
            if targSize[ocount] < minsize:
                targ_labeled_tmp[targ_labeled_tmp == (ocount + 1)] = 0
        unq_lab_tmp = np.unique(targ_labeled_tmp)
        targ_num_features_tmp = len(unq_lab_tmp) - 1
        for ocount in range(targ_num_features_tmp):
            targ_labeled_tmp[targ_labeled_tmp == unq_lab_tmp[ocount + 1]] = ocount + 1
        targ_labeled_array = targ_labeled_tmp.copy()
        targ_num_features = targ_num_features_tmp
        ReCompSize = True

    if minsize > 1 and predSizemin < minsize:
        pred_labeled_tmp = pred_labeled_array.copy()
        for ocount in range(pred_num_features):
            if predSize[ocount] < minsize:
                pred_labeled_tmp[pred_labeled_tmp == (ocount + 1)] = 0
        unq_lab_tmp = np.unique(pred_labeled_tmp)
        pred_num_features_tmp = len(unq_lab_tmp) - 1
        for ocount in range(pred_num_features_tmp):
            pred_labeled_tmp[pred_labeled_tmp == unq_lab_tmp[ocount + 1]] = ocount + 1
        pred_labeled_array = pred_labeled_tmp.copy()
        pred_num_features = pred_num_features_tmp
        ReCompSize = True

    # Re-Compute size of objects
    if ReCompSize:
        targSize = np.zeros(targ_num_features)
        for icount in range(targ_num_features):
            targSize[icount] = targ_labeled_array[targ_labeled_array == icount + 1].size
        predSize = np.zeros(pred_num_features)
        for icount in range(pred_num_features):
            predSize[icount] = pred_labeled_array[pred_labeled_array == icount + 1].size

    # Compute SAL score (as Wernli et al, 2008)

    # ----------------------------
    # - Amplitude
    #   Measure of total over whole domain
    sal_amplitude = 2.0 * (prediction.mean() - target.mean())
    sal_amplitude = sal_amplitude / (prediction.mean() + target.mean())

    # ----------------------------
    # - Location
    #   Measure of location of objects wrt whole domain
    # - based on centre of mass of fields and objects
    # scipy.ndimage.center_of_mass
    # docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.center_of_mass.html

    # --- location 1 based on centre of mass of total fields
    # taken from pysteps
    # Normalised difference of target and prediction centres of mass
    # - distances computed in number of grid points
    # - assumes small region so actual distances roughly equal?
    # centre_of_mass() assumes data non-negative and non-missing
    max_d = np.sqrt(target.shape[0] ** 2 + target.shape[1] ** 2)
    targ_shifted = target - np.nanmin(target)
    targCoM = sp.ndimage.center_of_mass(np.nan_to_num(targ_shifted, nan=0))
    pred_shifted = prediction - np.nanmin(prediction)
    predCoM = sp.ndimage.center_of_mass(np.nan_to_num(pred_shifted, nan=0))
    diffCoM = np.sqrt((predCoM[1] - targCoM[1]) ** 2 + (predCoM[0] - targCoM[0]) ** 2)
    Loc1 = np.abs(diffCoM) / max_d

    # --- location 2 based on centre of mass of individual objects
    # - Compute total sum of values per object (sum of all grid points) [Rn]
    targTotal = np.zeros(targ_num_features)
    for icount in range(targ_num_features):
        targTotal[icount] = target[targ_labeled_array == icount + 1].sum()
    predTotal = np.zeros(pred_num_features)
    for icount in range(pred_num_features):
        predTotal[icount] = prediction[pred_labeled_array == icount + 1].sum()

    # Compute centre of mass of each object and distance from centre of its total field
    targ_distCoM = np.zeros(targ_num_features)
    targ_obj_CoM = []
    for icount in range(targ_num_features):
        tmp_shifted = target.copy() - np.nanmin(target)
        tmp_shifted[targ_labeled_array != icount + 1] = np.nan
        CoMtmp = sp.ndimage.center_of_mass(np.nan_to_num(tmp_shifted, nan=0))
        dist_tmp = np.sqrt(
            (CoMtmp[1] - targCoM[1]) ** 2 + (CoMtmp[0] - targCoM[0]) ** 2
        )
        targ_distCoM[icount] = dist_tmp
        targ_obj_CoM.append(CoMtmp)

    pred_distCoM = np.zeros(pred_num_features)
    pred_obj_CoM = []
    for icount in range(pred_num_features):
        tmp_shifted = prediction.copy() - np.nanmin(prediction)
        tmp_shifted[pred_labeled_array != icount + 1] = np.nan
        CoMtmp = sp.ndimage.center_of_mass(np.nan_to_num(tmp_shifted, nan=0))
        dist_tmp = np.sqrt(
            (CoMtmp[1] - predCoM[1]) ** 2 + (CoMtmp[0] - predCoM[0]) ** 2
        )
        pred_distCoM[icount] = dist_tmp
        pred_obj_CoM.append(CoMtmp)

    # Compute weighted average distance of each object from centre of its total field [r]
    targ_wadistCoM = (targTotal * np.abs(targ_distCoM)).sum() / targTotal.sum()
    pred_wadistCoM = (predTotal * np.abs(pred_distCoM)).sum() / predTotal.sum()

    Loc2 = 2 * np.abs(pred_wadistCoM - targ_wadistCoM) / max_d

    sal_location = Loc1 + Loc2

    # ----------------------------
    # - Structure
    #   Measure of relative 'volume' in objects

    # - Compute max value in each object
    targMax = np.zeros(targ_num_features)
    for icount in range(targ_num_features):
        targMax[icount] = target[targ_labeled_array == icount + 1].max()
    predMax = np.zeros(pred_num_features)
    for icount in range(pred_num_features):
        predMax[icount] = prediction[pred_labeled_array == icount + 1].max()

    targVOL = targTotal / targMax
    predVOL = predTotal / predMax

    targ_waVOL = (targTotal * targVOL).sum() / targTotal.sum()
    pred_waVOL = (predTotal * predVOL).sum() / predTotal.sum()

    sal_structure = 2 * (pred_waVOL - targ_waVOL) / (pred_waVOL + targ_waVOL)

    # ----------------------------

    sal_xarray = xr.Dataset(
        {
            "sal_s": sal_structure,
            "sal_a": sal_amplitude,
            "sal_l": sal_location,
            "sal_l1": Loc1,
            "sal_l2": Loc2,
            "sal_targ_num": targ_num_features,
            "sal_pred_num": pred_num_features,
        }
    )

    targ_com_array = np.array(targ_obj_CoM)
    pred_com_array = np.array(pred_obj_CoM)

    targ_object_xarray = xr.Dataset(
        {
            "obj_com_lat": ("object_num", targ_com_array[:, 0]),
            "obj_com_lon": ("object_num", targ_com_array[:, 1]),
            "obj_max": ("object_num", targMax),
            "obj_size": ("object_num", targSize),
            "obj_labeled_array": (["lat", "lon"], targ_labeled_array),
        }
    )

    pred_object_xarray = xr.Dataset(
        {
            "obj_com_lat": ("object_num", pred_com_array[:, 0]),
            "obj_com_lon": ("object_num", pred_com_array[:, 1]),
            "obj_max": ("object_num", predMax),
            "obj_size": ("object_num", predSize),
            "obj_labeled_array": (["lat", "lon"], pred_labeled_array),
        }
    )

    return sal_xarray, targ_object_xarray, pred_object_xarray


def make_mask(input, eThresh, relation="gt"):
    """
    Convert input array into a boolean mask based on threshold eThresh and relation.

    Parameters
    ----------
    input : numpy.ndarray | xr.DataArray
        Array of floats
    eThresh : float | numpy.ndarray | xr.DataArray
        Event threshold to define binary event
        If an array, this should match shape and type of input,
        or follow numpy broadcasting rules.
    relation : str
        Relationship of event to threshold: 'gt', 'ge', 'lt', 'le', 'eq', 'ne'

    Return
    -----------
    numpy.ndarray | xr.DataArray
        Boolean version of input, same format.

    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade
    License: MIT (see LICENSE)

    """

    comparison_map = {
        "gt": lambda x, t: x > t,
        "lt": lambda x, t: x < t,
        "ge": lambda x, t: x >= t,
        "le": lambda x, t: x <= t,
        "eq": lambda x, t: x == t,
        "ne": lambda x, t: x != t,
    }

    if relation not in comparison_map:
        raise ValueError(
            f"relation must be one of {list(comparison_map.keys())}, got '{relation}'"
        )

    return comparison_map[relation](input, eThresh) & np.isfinite(input)


def compute_sfss2D_xr(
    prediction,
    target,
    var,
    eThresh,
    relation="gt",
    window=1,
    lat_name="lat",
    lon_name="lon",
    member_name=None,
):
    """
    Spatial Fractional Skill Score
    Threshold based binary fields [0, 1] = [none-event, event]
    Compute for each 2d image separately e.g. for each timestep
    ***Assumes SFSS computed on 2d fields defined by lat_name and lon_name dimensions***
    e.g. [ time, lat, lon ] => compute for lat/lon fields at each timestep.

    Roberts, N. M., & Lean, H. W. (2008). Scale-Selective Verification of
    Rainfall Accumulations from High-Resolution Forecasts of Convective
    Events, https://doi.org/10.1175/2007MWR2123.1

    Copyright (c) 2026 Klima consulting
    Author: Rosie Eade
    License: MIT (see LICENSE)

    Parameters
    ----------
    prediction : xr.DataArray
        N-dimensional array Prediction (same shape as target)
        ***Assumes spatial dimensions are final 2 in array***
        e.g. [time, member, lat, lon]
    target : xr.DataArray
        N-dimensional array Target (same shape as prediction)
    var : str
        Variable name to analyse.
    eThresh : float
        Event Threshold to define binary event
    relation : str, default 'gt'
        Relationship of event to threshold: 'gt', 'ge', 'lt', 'le', 'eq', 'ne'
    window : int
        Size of square window for allowed neighbours, centred on point;
        window=1 reduces to a pointwise comparison.
    lat_name : str
        Define name of latitude dimension e.g. 'lat' or 'y'
    lon_name : str
        Define name of longitude dimension e.g. 'lon' or 'x'
    member_name : str | None
        Define name of member dimension (ensemble prediction) e.g.
        'member' or 'realization'

    Return
    ------
    xr.Dataset
        Array of SFSS, same format as input target

    """

    # Check input array types
    if not isinstance(target, xr.DataArray) and not isinstance(
        prediction, xr.DataArray
    ):
        raise ValueError(
            f"input type error, expected xr.DataArray but got: T {type(target)}, P {type(prediction)}"
        )

    # Validate spatial dim names exist in the input
    for name, xr_obj in [
        (lat_name, target),
        (lon_name, target),
        (lat_name, prediction),
        (lon_name, prediction),
    ]:
        if name not in xr_obj.dims:
            raise ValueError(
                f"Spatial dimension '{name}' not found in dims {list(xr_obj.dims)}. "
                f"Use lat_name/lon_name to specify the correct dimension names."
            )

    spatial_dims = {lat_name, lon_name}

    # Detect if ensemble dim present in prediction but not target, and expand
    # target to match using expand_dims + broadcast
    extra_member_dim = None
    if member_name in prediction.dims and member_name not in target.dims:
        extra_member_dim = member_name

    if extra_member_dim is not None:
        target = target.expand_dims({extra_member_dim: prediction[extra_member_dim]})
        target = target.broadcast_like(prediction)

    if target.values.shape != prediction.values.shape:
        raise ValueError(
            f"Shape mismatch: Target ({target.values.shape}), "
            f"Prediction ({prediction.values.shape})"
        )

    # Identify spatial vs leading dimensions
    all_dims = list(prediction.dims)
    leading_dims = [d for d in all_dims if d not in spatial_dims]

    # Create the transpose order so lat and lon are last
    ordered_dims = leading_dims + [lat_name, lon_name]

    # Transpose both arrays
    prediction = prediction.transpose(*ordered_dims)
    target = target.transpose(*ordered_dims)

    # False/True fields
    prediction_masked = make_mask(prediction, eThresh, relation=relation)
    target_masked = make_mask(target, eThresh, relation=relation)

    prediction_bin = prediction.copy()
    target_bin = target.copy()
    prediction_bin.values[prediction_masked] = 1.0
    prediction_bin.values[~prediction_masked] = 0.0
    target_bin.values[target_masked] = 1.0
    target_bin.values[~target_masked] = 0.0

    # Define filter_axes, used for assigning neighbours within window
    if len(prediction.shape) <= 2:
        filter_axes = None
    if len(prediction.shape) > 2:
        filter_axes = (-1, -2)

    # Generate fractions: for each point, compute fraction of surrounding points
    # within given length scale (window) that have event (value = 1)
    # e.g. use smoothing fn so Xij becomes the average of all values in surrounding square window
    # For values at edge, this fills empty part of window with zeros
    # - outputs np.ndarray regardless if input type

    prediction_frac = sp.ndimage.uniform_filter(
        prediction_bin.astype(float),
        size=window,
        axes=filter_axes,
        mode="constant",
        cval=0.0,
    )
    target_frac = sp.ndimage.uniform_filter(
        target_bin.astype(float),
        size=window,
        axes=filter_axes,
        mode="constant",
        cval=0.0,
    )

    tmp = prediction.copy()
    tmp.values = prediction_frac
    prediction_frac = tmp.copy()
    tmp = target.copy()
    tmp.values = target_frac
    target_frac = tmp.copy()
    tmp = 0.0

    # Compute MSE between 2 fields
    mse = ((prediction_frac - target_frac) ** 2).mean(axis=filter_axes)

    # Compute MSE reference <= worst (largest) possible MSE (assumes completely unaligned fields)
    mse_ref = (prediction_frac**2).mean(axis=filter_axes) + (target_frac**2).mean(
        axis=filter_axes
    )
    # Replace 0s with nan
    mse_ref = mse_ref.where(mse_ref != 0)

    # Compute SFFS
    mse_ratio = mse / mse_ref
    sfss_out = 1.0 - mse_ratio

    sfss_xr = xr.Dataset({var: sfss_out})

    return sfss_xr
