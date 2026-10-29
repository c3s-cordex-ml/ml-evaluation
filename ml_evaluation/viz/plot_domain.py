# SPDX-FileCopyrightText: 2026 c3s-cordex-ml
# SPDX-License-Identifier: MIT

"""
Experiment-domain map: where the predictors and the predictands live.

One figure showing the outline of each grid an emulation experiment uses -- the
coarse predictor domain and the high-resolution predictand domain -- over
coastlines, so a reader can see at a glance how much larger the driving domain
is and which region is actually downscaled.

This is documentation rather than evaluation: it reads the grids straight from
the NetCDF files (or from explicit corners), never from a metrics store, and is
the natural first figure of a benchmark description.

Both grid conventions are handled:

  * a regular grid (1-D ``lat``/``lon``) is drawn as a rectangle;
  * a curvilinear one (2-D ``lat``/``lon`` over ``y``/``x``, as CORDEX rotated
    or Lambert grids are) has its actual perimeter traced, so the box appears
    tilted -- which is the truth, not an artefact.

Public API
----------
domain_outline(path, var=None)
    ``(lon, lat)`` of one grid's perimeter, plus its shape.
plot_domain(domains, out, *, title, ...)
    Render one map from ``{label: path-or-corners}``.

Example
-------
    from ml_evaluation.viz.plot_domain import plot_domain
    plot_domain(
        {
            "Predictors":  ".../predictors/CNRM-CM5_1961-1980.nc",
            "Predictands": ".../target/pr_tasmax_CNRM-CM5_1961-1980.nc",
        },
        "./FIGURES/domain",
        title="CORDEX-BENCH-ALPS domains",
    )
"""

from __future__ import annotations

import os

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature

LON_NAMES = ("lon", "longitude", "x")
LAT_NAMES = ("lat", "latitude", "y")


def _coord(ds, names):
    """The first of ``names`` present as a coordinate or variable."""
    for n in names:
        if n in ds.coords or n in ds.variables:
            return ds[n]
    raise KeyError(f"none of {names} found in the dataset; give explicit corners.")


def domain_outline(path, var=None):
    """The perimeter of one grid, ready to draw.

    Parameters
    ----------
    path : str
        NetCDF file holding the grid.
    var : str, optional
        Variable whose grid to use. Only needed when the file's coordinates are
        ambiguous; the lat/lon coordinates are used directly otherwise.

    Returns
    -------
    (lon, lat, shape) : tuple
        Closed perimeter arrays and the grid shape ``(ny, nx)`` -- the latter is
        what the "16 x 16" style annotation is built from.
    """
    ds = xr.open_dataset(path)
    if var is not None and var in ds:
        ds = ds[[var]]
    lon = np.asarray(_coord(ds, LON_NAMES).values, dtype=float)
    lat = np.asarray(_coord(ds, LAT_NAMES).values, dtype=float)

    if lon.ndim == 1 and lat.ndim == 1:
        # regular grid: the perimeter is the bounding rectangle
        x0, x1 = float(lon.min()), float(lon.max())
        y0, y1 = float(lat.min()), float(lat.max())
        px = np.array([x0, x1, x1, x0, x0])
        py = np.array([y0, y0, y1, y1, y0])
        return px, py, (lat.size, lon.size)

    if lon.ndim != 2 or lat.ndim != 2:
        raise ValueError(
            f"expected 1-D or 2-D lat/lon, got {lat.ndim}-D and {lon.ndim}-D."
        )
    # curvilinear grid: walk the four edges of the 2-D coordinate arrays
    px = np.concatenate([lon[0, :], lon[:, -1], lon[-1, ::-1], lon[::-1, 0]])
    py = np.concatenate([lat[0, :], lat[:, -1], lat[-1, ::-1], lat[::-1, 0]])
    return px, py, lat.shape


def _resolve(spec, var=None):
    """Turn one domain spec into ``(lon, lat, shape)``.

    ``spec`` is a path to a NetCDF file, or explicit corners
    ``(lon_min, lon_max, lat_min, lat_max)``.
    """
    if isinstance(spec, (str, os.PathLike)):
        return domain_outline(spec, var)
    x0, x1, y0, y1 = (float(v) for v in spec)
    px = np.array([x0, x1, x1, x0, x0])
    py = np.array([y0, y0, y1, y1, y0])
    return px, py, None


def plot_domain(
    domains,
    out,
    *,
    title="Experiment domains",
    filename=None,
    colors=("tab:red", "tab:blue"),
    resolutions=None,
    var=None,
    margin=2.0,
    annotate=True,
    figsize=(7.0, 7.0),
):
    """Draw the outline of each experiment grid on one map.

    Parameters
    ----------
    domains : mapping
        ``{label: spec}`` in drawing order, where ``spec`` is a NetCDF path or
        ``(lon_min, lon_max, lat_min, lat_max)``. The first entry is normally the
        predictor domain (the larger one), the second the predictand domain.
    out : str
        Output directory (created if missing).
    title : str
        Figure title.
    filename : str, optional
        Output name; default ``domains.png``.
    colors : sequence of str
        One colour per domain, in the same order.
    resolutions : mapping, optional
        ``{label: "~1.5 deg"}`` -- appended to that domain's legend entry. The
        grid shape is read from the file, so only the resolution needs saying.
    var : str, optional
        Variable whose grid to use, when a file's coordinates are ambiguous.
    margin : float
        Degrees of padding around the outermost domain.
    annotate : bool
        Write the grid shape (e.g. "16 x 16") above each box.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    str
        Path of the PNG written.
    """
    if not domains:
        raise ValueError("domains must be a non-empty {label: spec} mapping.")

    resolved = {}
    for i, (label, spec) in enumerate(domains.items()):
        resolved[label] = _resolve(spec, var)

    all_lon = np.concatenate([r[0] for r in resolved.values()])
    all_lat = np.concatenate([r[1] for r in resolved.values()])
    extent = [
        float(all_lon.min()) - margin,
        float(all_lon.max()) + margin,
        float(all_lat.min()) - margin,
        float(all_lat.max()) + margin,
    ]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", linewidth=0.6)
    ax.add_feature(cartopy.feature.BORDERS, linewidth=0.4, edgecolor="0.5")

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.8", linestyle=":")
    gl.top_labels = gl.right_labels = False

    handles = []
    for (label, (px, py, shape)), color in zip(resolved.items(), colors):
        (line,) = ax.plot(
            px, py, color=color, lw=2.0, transform=ccrs.PlateCarree(), zorder=5
        )
        bits = [label]
        if shape is not None:
            bits.append(f"({shape[0]} x {shape[1]}")
            res = (resolutions or {}).get(label)
            bits[-1] += f", {res})" if res else ")"
        handles.append((line, "  ".join(bits)))

        if annotate and shape is not None:
            ax.text(
                float(np.mean(px)),
                float(np.max(py)) + 0.3 * margin,  # clear of the box edge
                f"{shape[0]} x {shape[1]}",
                color=color,
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="bottom",
                transform=ccrs.PlateCarree(),
                zorder=6,
            )

    ax.legend(
        [h for h, _ in handles],
        [t for _, t in handles],
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")

    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, filename or "domains.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[domain] {path}")
    return path
