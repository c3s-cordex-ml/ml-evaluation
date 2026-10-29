# SPDX-FileCopyrightText: 2026 c3s-cordex-ml
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import xarray as xr

from typing import Any, Literal

from ml_evaluation.transformations import align_calendar

dask_default_params: dict[str, Any] = dict(
    n_workers=4, threads_per_worker=1, memory_limit="4GiB"
)


def get_output_path(base: str, suffix: str) -> str:
    """
    Construct the output file path based on a base path and a suffix.

    If the base path is an existing directory, the suffix is appended as a filename.
    Otherwise, the base path is treated as a file prefix, and the suffix is appended.

    Parameters
    ----------
    base : str
        The base output directory or file prefix.
    suffix : str
        The suffix to append (e.g. 'mean_bias.nc').

    Returns
    -------
    str
        The constructed file path.
    """
    if os.path.isdir(base):
        return os.path.join(base, suffix)
    return f"{base}_{suffix}"


def load_dataset(
    path: str,
    align_leap_days: str | None = None,
    chunks: str | dict[str, int] | None = None,
) -> xr.Dataset:
    """
    Load a NetCDF file into an xarray Dataset.

    Reads a NetCDF dataset from the specified file path and optionally
    aligns the dataset to a standard calendar by ensuring February 29th
    is present and filled in leap years.

    Parameters
    ----------
    path : str
        The absolute or relative path to the NetCDF file.
    align_leap_days : str | None
        Whether to align the calendar by filling missing leap days (default is None).
        "fill" : fill using February 28th
        "remove" : remove February 29th
    chunks : str | dict[str, int] | None, optional
        Chunk sizes to use for Dask chunking (default is None).

    Returns
    -------
    xr.Dataset
        The loaded NetCDF dataset.
    """
    dataset = xr.open_dataset(path, chunks=chunks)
    dataset = align_calendar(dataset, align_leap_days=align_leap_days)
    return dataset


def prepare_diagnostics(
    diagnostics: dict[str, Any],
    var: str,
) -> xr.Dataset:
    """
    Consolidate a dictionary of diagnostics into a single Dataset.

    Combines multiple diagnostic/index outputs (Datasets, DataArrays,
    or scalars) along a new 'diagnostic' dimension. The resulting
    Dataset contains a single variable named 'value'.

    Parameters
    ----------
    diagnostics : dict[str, Any]
        Dictionary mapping diagnostic names to their computed values.
    var : str
        The variable name to extract from Dataset elements.

    Returns
    -------
    xr.Dataset
        The consolidated dataset with a single 'value' variable.
    """
    da_list: list[xr.DataArray] = []
    names: list[str] = []

    for name, item in diagnostics.items():
        names.append(name)
        if isinstance(item, xr.Dataset):
            da_list.append(item[var])
        elif isinstance(item, xr.DataArray):
            da_list.append(item)
        else:
            da_list.append(xr.DataArray(float(item)))

    da_concat = xr.concat(
        da_list, dim="diagnostic", coords="minimal", compat="override"
    )
    da_concat = da_concat.assign_coords(diagnostic=names)
    return xr.Dataset({"value": da_concat})


def write_netcdf(
    dataset: xr.Dataset | xr.DataArray | dict[str, Any],
    path: str,
    var: str | None = None,
) -> None:
    """
    Save an xarray Dataset or DataArray to a NetCDF file.

    Writes the provided dataset (or dictionary of diagnostics) to the
    designated file path in NetCDF format, with optimized chunks
    and compression.

    Parameters
    ----------
    dataset : xr.Dataset | xr.DataArray | dict[str, Any]
        The dataset, dataarray, or diagnostic dictionary to write.
    path : str
        The output file path.
    var : str | None, optional
        The variable name to extract if `dataset` is a dictionary (default is None).

    Returns
    -------
    None
    """
    from ml_evaluation.chunking import get_optimized_chunks

    if isinstance(dataset, dict):
        if var is None:
            raise ValueError(
                "var must be specified when dataset is a dictionary of diagnostics."
            )
        ds = prepare_diagnostics(dataset, var)
    elif isinstance(dataset, xr.DataArray):
        ds = xr.Dataset({"value": dataset})
    else:
        ds = dataset

    encoding = {}
    if "value" in ds:
        chunks = get_optimized_chunks(ds, "value")
        encoding["value"] = {"zlib": True, "complevel": 5, "chunksizes": chunks}

    ds.load().to_netcdf(path, encoding=encoding)


def write_zarr(
    dataset: xr.Dataset | xr.DataArray | dict[str, Any],
    path: str,
    mode: Literal["w", "w-", "a", "a-", "r+", "r"] = "w",
    var: str | None = None,
) -> None:
    """
    Save an xarray Dataset or DataArray to a Zarr store.

    Writes the provided dataset (or dictionary of diagnostics) to the
    designated file path in Zarr format, with optimized chunks
    and compression.

    Parameters
    ----------
    dataset : xr.Dataset | xr.DataArray | dict[str, Any]
        The dataset, dataarray, or diagnostic dictionary to write.
    path : str
        The output Zarr store path.
    mode : str, default 'w'
        The persistence mode to use when writing the store.
    var : str | None, optional
        The variable name to extract if `dataset` is a dictionary (default is None).

    Returns
    -------
    None
    """
    from ml_evaluation.chunking import get_optimized_chunks

    if isinstance(dataset, dict):
        if var is None:
            raise ValueError(
                "var must be specified when dataset is a dictionary of diagnostics."
            )
        ds = prepare_diagnostics(dataset, var)
    elif isinstance(dataset, xr.DataArray):
        ds = xr.Dataset({"value": dataset})
    else:
        ds = dataset

    encoding = {}
    if "value" in ds:
        chunks = get_optimized_chunks(ds, "value")
        encoding["value"] = {
            "chunks": chunks,
            "compressors": [{"name": "zstd", "configuration": {"level": 5}}],
        }

    ds.load().to_zarr(path, encoding=encoding, mode=mode)


def write_text(content: str, path: str) -> None:
    """
    Write a string content to a text file.

    Writes the specified text content into a file at the given path.
    If the file exists, it will be overwritten.

    Parameters
    ----------
    content : str
        The text content to be written.
    path : str
        The output file path.

    Returns
    -------
    None
    """
    with open(path, "w") as f:
        f.write(content)
