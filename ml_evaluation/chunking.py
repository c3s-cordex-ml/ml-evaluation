# SPDX-FileCopyrightText: 2026 c3s-cordex-ml
# SPDX-License-Identifier: MIT

from __future__ import annotations

import xarray as xr

from ml_evaluation.transformations import spatial_dims_present


def get_optimized_chunks(
    dataset: xr.Dataset, var_name: str = "value"
) -> tuple[int, ...]:
    """
    Calculate optimized chunk sizes for a variable in a Dataset.

    Returns the chunk sizes as a tuple matching the dimension order
    of the specified variable. The chunk size for 'diagnostic' is
    always 1, while spatial/wavenumber dimensions use their full
    sizes (or 250 if spatial dimensions exceed 500).

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset containing the variable.
    var_name : str, default 'value'
        The name of the variable to get chunks for.

    Returns
    -------
    tuple[int, ...]
        A tuple of chunk sizes in the dimension order of the variable.
    """
    if var_name not in dataset:
        raise ValueError(f"Variable '{var_name}' not found in dataset.")

    da = dataset[var_name]
    sdims = spatial_dims_present(da)
    chunks: list[int] = []

    for dim in da.dims:
        size = da.sizes[dim]
        if dim == "diagnostic":
            chunks.append(1)
        elif dim in sdims:
            if size > 500:
                chunks.append(250)
            else:
                chunks.append(size)
        else:
            chunks.append(size)

    return tuple(chunks)
