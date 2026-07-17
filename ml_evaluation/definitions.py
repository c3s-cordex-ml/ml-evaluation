from __future__ import annotations

import os
from typing import Any, Callable
import yaml
import click
import xarray as xr

from ml_evaluation.evaluation.diagnostics import (
    bias_index,
    rmse_index,
    corr_index,
    psd,
    ralsd,
    calc_pdf_metric_xr,
    calc_histdata_xr,
    bias_multivariable_correlation,
)
import ml_evaluation.evaluation.indices as indices

# Helper mapping index_fn names to actual functions in indices module
INDEX_FN_MAP: dict[str, Callable[..., Any]] = {
    "mean": indices.mean,
    "season_quantile": indices.season_quantile,
    "lag1_corr": indices.lag1_corr,
    "ann_extrema": indices.ann_extrema,
    "ann_freq_wrt_array": indices.ann_freq_wrt_array,
    "ann_sdii": indices.ann_sdii,
    "ann_maxspell_wrt_array": indices.ann_maxspell_wrt_array,
    "interannual_var": indices.interannual_var,
}


def load_definitions() -> dict[str, Any]:
    """
    Load the metric definitions from definitions.yaml.

    Returns
    -------
    dict[str, Any]
        Loaded YAML definitions.
    """
    yaml_path = os.path.join(os.path.dirname(__file__), "definitions.yaml")
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


_DEFINITIONS: dict[str, Any] = load_definitions()


def resolve_threshold(value: Any, scale: float, shift: float) -> Any:
    """
    Resolve numeric expressions involving scale and shift.

    Parameters
    ----------
    value : Any
        The value to resolve. Can be a string expression or numeric/array.
    scale : float
        The scale factor.
    shift : float
        The shift factor.

    Returns
    -------
    Any
        The resolved numeric value or the original value if not a string.
    """
    if isinstance(value, str):
        cleaned = value.replace("scale", str(scale)).replace("shift", str(shift))
        if "+" in cleaned:
            left, right = cleaned.split("+", 1)
            return resolve_threshold(left.strip(), scale, shift) + resolve_threshold(
                right.strip(), scale, shift
            )
        if "*" in cleaned:
            parts = cleaned.split("*")
            prod = 1.0
            for p in parts:
                prod *= float(p.strip())
            return prod
        return float(cleaned.strip())
    return value


def compute_fast_metrics(
    x0_ds: xr.Dataset,
    x1_ds: xr.Dataset,
    var: str,
    season: str | None,
    scale: float,
    shift: float,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    """
    Run fast-track evaluation metrics for a specific variable and season.

    Parameters
    ----------
    x0_ds : xr.Dataset
        Reference dataset.
    x1_ds : xr.Dataset
        Prediction dataset.
    var : str
        Variable name.
    season : str | None
        The season name.
    scale : float
        Threshold scale factor.
    shift : float
        Threshold shift factor.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
        Five dictionaries containing results for field, series, float, pr_field, and psd metrics.
    """
    metrics_field: dict[str, Any] = {}
    metrics_series: dict[str, Any] = {}
    metrics_float: dict[str, Any] = {}
    metrics_pr_field: dict[str, Any] = {}
    metrics_psd: dict[str, Any] = {}

    category_maps = {
        "field": metrics_field,
        "series": metrics_series,
        "float": metrics_float,
        "pr_field": metrics_pr_field,
        "psd": metrics_psd,
    }

    # Fetch track definitions
    track_def = _DEFINITIONS.get("fast", {})
    metrics_list = list(track_def.get("general", []))

    # Add variable-specific definitions if present
    var_defs = track_def.get("variables", {})
    if var in var_defs:
        metrics_list.extend(var_defs[var])

    results: dict[str, Any] = {}

    for metric in metrics_list:
        name = metric["name"]
        mtype = metric["type"]
        description = metric.get("description", f"Computing {name}...")
        category = metric["category"]

        click.echo(description)

        if mtype == "bias_index":
            fn_name = metric["index_fn"]
            resolved_fn = INDEX_FN_MAP[fn_name]

            # Build call kwargs
            kwargs: dict[str, Any] = {}
            if "q" in metric:
                kwargs["q"] = metric["q"]
            if "maxmin" in metric:
                kwargs["maxmin"] = metric["maxmin"]
            if "cmean" in metric:
                kwargs["cmean"] = metric["cmean"]
            if "relative" in metric:
                kwargs["relative"] = metric["relative"]
            if "relation" in metric:
                kwargs["relation"] = metric["relation"]
            if "threshold" in metric:
                kwargs["threshold"] = resolve_threshold(
                    metric["threshold"], scale, shift
                )
            if "wet_threshold" in metric:
                kwargs["wet_threshold"] = resolve_threshold(
                    metric["wet_threshold"], scale, shift
                )

            res = bias_index(
                x0_ds,
                x1_ds,
                var=var,
                season=season,
                index_fn=resolved_fn,
                **kwargs,
            )
            category_maps[category][name] = res
            results[name] = res

        elif mtype == "rmse_index":
            dim = tuple(metric["dim"])
            res = rmse_index(
                x0_ds,
                x1_ds,
                var=var,
                season=season,
                dim=dim,
            )
            category_maps[category][name] = res
            results[name] = res

        elif mtype == "corr_index":
            dim = tuple(metric["dim"])
            srank = metric.get("srank", False)
            res = corr_index(
                x0_ds,
                x1_ds,
                var=var,
                season=season,
                dim=dim,
                srank=srank,
            )
            category_maps[category][name] = res
            results[name] = res

        elif mtype == "lag1_corr":
            ds_to_use = x0_ds if metric.get("dataset") == "x0" else x1_ds
            res = indices.lag1_corr(
                ds_to_use,
                var=var,
                season=season,
            )
            category_maps[category][name] = res
            results[name] = res

        elif mtype == "psd":
            psd_x0, psd_x1 = psd(x0_ds, x1_ds, var=var, season=season)
            category_maps[category]["psd_x0"] = psd_x0
            category_maps[category]["psd_x1"] = psd_x1
            results["psd_x0"] = psd_x0
            results["psd_x1"] = psd_x1

        elif mtype == "ralsd":
            res = ralsd(results["psd_x0"], results["psd_x1"], var)
            category_maps[category][name] = res
            results[name] = res

    return metrics_field, metrics_series, metrics_float, metrics_pr_field, metrics_psd


def compute_slow_metrics(
    x0_ds: xr.Dataset,
    x1_ds: xr.Dataset,
    var: str,
    season: str | None,
    scale: float,
    shift: float,
    member_dim: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Run slow-track evaluation metrics for a specific variable and season.

    Parameters
    ----------
    x0_ds : xr.Dataset
        Reference dataset.
    x1_ds : xr.Dataset
        Prediction dataset.
    var : str
        Variable name.
    season : str | None
        The season name.
    scale : float
        Threshold scale factor.
    shift : float
        Threshold shift factor.
    member_dim : str | None
        Name of the member dimension.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
        Four dictionaries containing results for slow, hist, tasmax, and tasmin metrics.
    """
    metrics_slow: dict[str, Any] = {}
    metrics_hist: dict[str, Any] = {}
    metrics_tasmax: dict[str, Any] = {}
    metrics_tasmin: dict[str, Any] = {}

    category_maps = {
        "slow": metrics_slow,
        "hist": metrics_hist,
        "tasmax": metrics_tasmax,
        "tasmin": metrics_tasmin,
    }

    # Fetch track definitions
    track_def = _DEFINITIONS.get("slow", {})
    metrics_list = list(track_def.get("general", []))

    # Add variable-specific definitions if present
    var_defs = track_def.get("variables", {})
    if var in var_defs:
        metrics_list.extend(var_defs[var])

    for metric in metrics_list:
        name = metric["name"]
        mtype = metric["type"]
        description = metric.get("description", f"Computing {name}...")
        category = metric["category"]

        click.echo(description)

        if mtype == "bias_index":
            fn_name = metric["index_fn"]
            resolved_fn = INDEX_FN_MAP[fn_name]

            kwargs: dict[str, Any] = {}
            if "q" in metric:
                kwargs["q"] = metric["q"]
            if "cmean" in metric:
                kwargs["cmean"] = metric["cmean"]
            if "relation" in metric:
                kwargs["relation"] = metric["relation"]
            if "threshold" in metric:
                kwargs["threshold"] = resolve_threshold(
                    metric["threshold"], scale, shift
                )

            res = bias_index(
                x0_ds,
                x1_ds,
                var=var,
                season=season,
                index_fn=resolved_fn,
                **kwargs,
            )
            category_maps[category][name] = res

        elif mtype == "corr_index":
            dim = tuple(metric["dim"])
            srank = metric.get("srank", False)
            res = corr_index(
                x0_ds,
                x1_ds,
                var=var,
                season=season,
                dim=dim,
                srank=srank,
            )
            category_maps[category][name] = res

        elif mtype == "calc_pdf_metric_xr":
            dim = tuple(metric["dim"])
            typedistance = metric["typedistance"]
            bin_target_only = metric.get("bin_target_only", False)

            kwargs = {}
            if "nbins" in metric:
                kwargs["nbins"] = metric["nbins"]

            res = calc_pdf_metric_xr(
                x0_ds,
                x1_ds,
                var=var,
                season=season,
                index_fn=None,
                dim=dim,
                bin_target_only=bin_target_only,
                typedistance=typedistance,
                member_name=member_dim,
                **kwargs,
            )
            category_maps[category][name] = res

        elif mtype == "calc_histdata_xr":
            kwargs = {}
            if "nbins" in metric:
                kwargs["nbins"] = metric["nbins"]
            if "bin_target_only" in metric:
                kwargs["bin_target_only"] = metric["bin_target_only"]

            targ_hist_ds, pred_hist_ds, bins_hist_ds = calc_histdata_xr(
                x0_ds,
                x1_ds,
                var=var,
                season=season,
                index_fn=None,
                **kwargs,
            )
            category_maps[category]["targ_hist_series"] = targ_hist_ds
            category_maps[category]["pred_hist_series"] = pred_hist_ds
            category_maps[category]["bins_hist_series"] = bins_hist_ds

    return metrics_slow, metrics_hist, metrics_tasmax, metrics_tasmin


def compute_multivar_metrics(
    x0_ds: xr.Dataset,
    x1_ds: xr.Dataset,
    var: str,
    var2: str,
    season: str | None,
) -> dict[str, Any]:
    """
    Run multi-variate correlation metrics.

    Parameters
    ----------
    x0_ds : xr.Dataset
        Reference dataset.
    x1_ds : xr.Dataset
        Prediction dataset.
    var : str
        First variable.
    var2 : str
        Second variable.
    season : str | None
        The season name.

    Returns
    -------
    dict[str, Any]
        Dictionary of computed multi-variate metrics.
    """
    results: dict[str, Any] = {}
    track_def = _DEFINITIONS.get("multivar", {})
    metrics_list = track_def.get("general", [])

    for metric in metrics_list:
        click.echo(metric["description"])
        srank = metric.get("srank", False)
        prefix = "scorr" if srank else "pcorr"

        targ_ds, pred_ds, bias_ds = bias_multivariable_correlation(
            x0_ds,
            x1_ds,
            var_x=var,
            var_y=var2,
            season=season,
            srank=srank,
        )
        results[f"targ_mv_{prefix}_field"] = targ_ds
        results[f"pred_mv_{prefix}_field"] = pred_ds
        results[f"bias_mv_{prefix}_field"] = bias_ds

    return results


def compute_cdextremes_metrics(
    x0_ds: xr.Dataset,
    x1_ds: xr.Dataset,
    var: str,
    season: str | None,
    targ_var90p_quantile: xr.Dataset | None,
    targ_var10p_quantile: xr.Dataset | None,
) -> dict[str, Any]:
    """
    Run calendar day extremes metrics relative to 90th and 10th percentile thresholds.

    Parameters
    ----------
    x0_ds : xr.Dataset
        Reference dataset.
    x1_ds : xr.Dataset
        Prediction dataset.
    var : str
        Variable name.
    season : str | None
        The season name.
    targ_var90p_quantile : xr.Dataset | None
        90th percentile calendar day thresholds.
    targ_var10p_quantile : xr.Dataset | None
        10th percentile calendar day thresholds.

    Returns
    -------
    dict[str, Any]
        Dictionary of computed extremes metrics.
    """
    results: dict[str, Any] = {}
    track_def = _DEFINITIONS.get("cdextremes", {})
    metrics_list = track_def.get("general", [])

    for metric in metrics_list:
        click.echo(metric["description"])
        q_threshold = metric["q_threshold"]
        relation = metric["relation"]

        thresh = targ_var90p_quantile if q_threshold == "90p" else targ_var10p_quantile

        if thresh is not None:
            # Frequency bias
            freq_name = f"bias_freq{q_threshold}_field"
            results[freq_name] = bias_index(
                x0_ds,
                x1_ds,
                var=var,
                season=season,
                index_fn=indices.ann_freq_wrt_array,
                threshold=thresh,
                relation=relation,
                cmean=True,
            )

            # Spell bias
            spell_name = f"bias_spell{q_threshold}_field"
            results[spell_name] = bias_index(
                x0_ds,
                x1_ds,
                var=var,
                season=season,
                index_fn=indices.ann_maxspell_wrt_array,
                threshold=thresh,
                relation=relation,
                cmean=True,
            )

    return results
