from __future__ import annotations

import pathlib
import numpy as np
import xarray as xr
from ml_evaluation.io import load_dataset


def test_load_dataset_no_align(tmp_path: pathlib.Path) -> None:
    """
    Test loading a NetCDF dataset without calendar alignment.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Fixture providing a temporary directory.

    Returns
    -------
    None
    """
    times = xr.date_range("2020-02-27", "2020-03-02", calendar="standard", freq="D")
    ds = xr.Dataset(
        {"temp": (["time"], np.arange(len(times), dtype=float))},
        coords={"time": times},
    )
    file_path = tmp_path / "test.nc"
    ds.to_netcdf(file_path)

    loaded = load_dataset(str(file_path), align_leap_days=None)
    xr.testing.assert_identical(ds, loaded)


def test_load_dataset_align_leap_days(tmp_path: pathlib.Path) -> None:
    """
    Test loading a NetCDF dataset with calendar alignment.

    Converts a no-leap calendar dataset containing leap years to a
    standard calendar by inserting and filling February 29th.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Fixture providing a temporary directory.

    Returns
    -------
    None
    """
    # 2020 is a leap year, noleap calendar does not have Feb 29th
    times = xr.date_range("2020-02-27", "2020-03-02", calendar="noleap", freq="D")
    ds = xr.Dataset(
        {"temp": (["time"], np.arange(len(times), dtype=float))},
        coords={"time": times},
    )
    file_path = tmp_path / "noleap.nc"
    ds.to_netcdf(file_path)

    loaded = load_dataset(str(file_path), align_leap_days="fill")

    # The loaded dataset should now have a standard calendar with 5 timestamps (including Feb 29)
    assert len(loaded.time) == 5
    assert "2020-02-29" in loaded.time.dt.strftime("%Y-%m-%d").values

    # Temp value for Feb 29 should be forward-filled from Feb 28
    # Times index in noleap: 2020-02-27 (idx 0), 2020-02-28 (idx 1), 2020-03-01 (idx 2), 2020-03-02 (idx 3)
    # The temperature on Feb 28 is 1.0. So Feb 29 should also be 1.0.
    temp_values = loaded["temp"].values
    expected = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(temp_values, expected)


def test_write_zarr(tmp_path: pathlib.Path) -> None:
    """
    Test saving a dataset using write_zarr function.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Fixture providing a temporary directory.

    Returns
    -------
    None
    """
    ds = xr.Dataset(
        {"temp": (["time"], [1.0, 2.0])},
        coords={"time": [1, 2]},
    )
    file_path = tmp_path / "test.zarr"
    from ml_evaluation.io import write_zarr

    write_zarr(ds, str(file_path), mode="w")

    loaded = xr.open_zarr(str(file_path))
    xr.testing.assert_identical(ds, loaded)


def test_write_diagnostics_dict(tmp_path: pathlib.Path) -> None:
    """
    Test preparing and saving a diagnostics dictionary using write_zarr and write_netcdf.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Fixture providing a temporary directory.

    Returns
    -------
    None
    """
    lat = [45.0, 46.0]
    lon = [10.0, 11.0]

    # Create dummy datasets/dataarrays/scalars
    ds_item = xr.Dataset(
        {"tasmax": (["lat", "lon"], [[1.0, 2.0], [3.0, 4.0]])},
        coords={"lat": lat, "lon": lon},
    )
    da_item = xr.DataArray(
        [[0.5, 0.6], [0.7, 0.8]],
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"],
    )
    scalar_item = 42.0

    diagnostics = {
        "mean_bias": ds_item,
        "rmsefld": da_item,
        "rmse": scalar_item,
    }

    from ml_evaluation.io import write_zarr, write_netcdf

    # Write to Zarr
    zarr_path = tmp_path / "metrics.zarr"
    write_zarr(diagnostics, str(zarr_path), mode="w", var="tasmax")

    ds_zarr = xr.open_zarr(str(zarr_path))
    assert "value" in ds_zarr
    assert set(ds_zarr["diagnostic"].values) == {"mean_bias", "rmsefld", "rmse"}
    assert float(ds_zarr["value"].sel(diagnostic="rmse", lat=45.0, lon=10.0)) == 42.0

    # Write to NetCDF
    nc_path = tmp_path / "metrics.nc"
    write_netcdf(diagnostics, str(nc_path), var="tasmax")

    ds_nc = xr.open_dataset(str(nc_path))
    assert "value" in ds_nc
    assert set(ds_nc["diagnostic"].values) == {"mean_bias", "rmsefld", "rmse"}
