"""
Regrid files within the CMIP ESGF catalog.
"""
from __future__ import annotations

import gc
import os
import time
from datetime import datetime as dt
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

from load_cmip_esgf import CMIPESGFLoader


CATALOG_PATH = Path("/glade/derecho/scratch/bbuchovecky/cmip_intake_esgf_fetch/manifests/esgf_cache_catalog.csv")
REGRID_ROOT = Path("/glade/campaign/univ/uwas0155/cmip6/regridded")

VARIABLES = ["evspsbl", "tran", "evspsblsoi", "evspsblveg", "lai", "gpp"]
EXPERIMENT_ID = "historical"
SOURCE_IDS = None
MEMBER_IDS = None
TIME_SLICE = slice("1950-01", "2014-12")

TARGET_GRID = xe.util.grid_global(1.0, 1.0, cf=True)


def _coord_name(ds: xr.DataArray, candidates: list[str], standard_names: list[str]) -> str | None:
    for name in candidates:
        if name in ds.coords:
            return name
    for name, var in ds.coords.items():
        if str(var.attrs.get("standard_name", "")) in standard_names:
            return name
    return None


def _approx_resolution(ds: xr.DataArray) -> tuple[float | None, float | None]:
    lat_name = _coord_name(ds, ["lat", "latitude", "nav_lat"], ["latitude"])
    lon_name = _coord_name(ds, ["lon", "longitude", "nav_lon"], ["longitude"])

    def spacing(name: str | None) -> float | None:
        if name is None or name not in ds.coords:
            return None
        arr = np.asarray(ds[name].values)
        if arr.ndim == 0:
            return None
        vals = np.unique(arr[np.isfinite(arr)])
        if vals.size < 2:
            return None
        diffs = np.diff(np.sort(vals))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return None
        return float(np.nanmedian(diffs))

    return spacing(lat_name), spacing(lon_name)


def _format_lat_lon(da: xr.DataArray) -> xr.DataArray:
    """Clean and format lat/lon coords (primarily for EC-Earth)."""
    if ("i" in da.dims) and ("j" in da.dims):
        if ("longitude" in da.coords) and ("latitude" in da.coords):
            da = da.rename(i="lon")
            da = da.rename(j="lat")

            latitude = da["latitude"]
            longitude = da["longitude"]

            da = da.assign_coords(lat=latitude.isel(lon=0), lon=longitude.isel(lat=0))
            return da.drop_vars(["latitude", "longitude"])
    return da


def regrid_to_target(da: xr.DataArray, target_grid: xr.DataArray, verbose: bool = False) -> xr.DataArray:
    """
    Attempt conservative (area-weighted) regridding using available tools.

    Use xESMF conservative regridding. Raises RuntimeError if xesmf is not
    installed or the regridding fails.
    """
    da = _format_lat_lon(da)

    # Build minimal source/target grid structures for xESMF. xESMF accepts
    # xarray.Dataset or dict-like inputs with 1D lon/lat coordinates.
    try:
        # xESMF expects xarray Dataset/DataArray inputs with lon/lat variables
        # Build minimal xarray.Datasets retaining the original dims (1D or 2D).
        lon_dims_src = da["lon"].dims
        lat_dims_src = da["lat"].dims
        src = xr.Dataset(
            {
                "lon": (lon_dims_src, da["lon"].values),
                "lat": (lat_dims_src, da["lat"].values),
            }
        )
        if verbose:
            print(f"Source: lon={len(src.lon)}, lat={len(src.lat)}")

        lon_dims_tgt = target_grid["lon"].dims
        lat_dims_tgt = target_grid["lat"].dims
        tgt = xr.Dataset(
            {
                "lon": (lon_dims_tgt, target_grid["lon"].values),
                "lat": (lat_dims_tgt, target_grid["lat"].values),
            }
        )
        if verbose:
            print(f"Target: lon={len(tgt.lon)}, lat={len(tgt.lat)}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regridder = xe.Regridder(src, tgt, "conservative", reuse_weights=False)
            return regridder(da, keep_attrs=True)

    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(f"xESMF regridding failed - da: {da.dims} {da.shape}") from exc


def to_yyyymm(time) -> str:
    time_raw = time.values
    if isinstance(time_raw, np.datetime64):
        return pd.Timestamp(time_raw).strftime("%Y%m")
    elif isinstance(time_raw, np.ndarray):
        return time.item().strftime("%Y%m")
    raise TypeError(f"Unsupported type {type(time)!r} for time")


def main() -> None:
    run_t0 = time.perf_counter()
    print(f"Target grid: nlon={TARGET_GRID['lon'].shape}, nlat={TARGET_GRID['lat'].shape}")

    loader = CMIPESGFLoader(CATALOG_PATH)
    catalog = loader.catalog

    print("Load data:")
    load_t0 = time.perf_counter()
    data_dict = loader.load_data(
        VARIABLES,
        source_id=SOURCE_IDS,
        member_id=MEMBER_IDS,
        experiment_id=EXPERIMENT_ID,
        time_slice=TIME_SLICE
    )
    load_elapsed = time.perf_counter() - load_t0
    print(f"  loaded in {load_elapsed:.2f}s")

    print(f"Available models:  {list(data_dict.keys())}\n")

    # (source_id, variable) -> {"regrid": s, "save": s}
    timings: dict[tuple[str, str], dict[str, float]] = {}

    for sid, vardict in data_dict.items():
        print(f"==== {sid} ====")

        for var, da in vardict.items():

            # Regrid
            print(f"  {var}: IN {da.dims} {da.shape} Regridding...", end="", flush=True)
            regrid_t0 = time.perf_counter()
            da_regridded = regrid_to_target(da, TARGET_GRID)
            regrid_elapsed = time.perf_counter() - regrid_t0
            print(
                f"done in {regrid_elapsed:.2f}s. OUT "
                f"{da_regridded.dims} {da_regridded.shape} "
                f"{da_regridded.nbytes / 1024 / 1024 / 1024:.2f}GB",
                end=" ",
            )

            # Add attributes
            da_regridded.attrs["src_dims"] = da.dims
            da_regridded.attrs["src_shape"] = da.shape
            da_regridded.attrs["src_lat_name"] = _coord_name(da, ["lat", "alatitude", "nav_lat"], ["latitude"])
            da_regridded.attrs["src_lon_name"] = _coord_name(da, ["lon", "longitude", "nav_lon"], ["longitude"])
            da_regridded.attrs["src_dlat_deg"], da_regridded.attrs["src_dlon_deg"] = _approx_resolution(da)
            da_regridded.attrs["tgt_dlat_deg"], da_regridded.attrs["tgt_dlon_deg"] = _approx_resolution(da_regridded)
            da_regridded.attrs["regrid_script"] = os.path.basename(__file__)
            da_regridded.attrs["regrid_date"] = dt.now().strftime("%Y-%m-%d %H:%M:%S%Z")

            # Get table ID
            table_id = da.attrs.get("mipTable", "none")
            if table_id == "none":
                table_id = catalog.loc[
                    (catalog["experiment_id"] == EXPERIMENT_ID) &
                    (catalog["source_id"] == sid) &
                    (catalog["variable_id"] == var)
                ].iloc[0].table_id

            # Handle output path
            start_str = to_yyyymm(da_regridded.time.isel(time=0))
            stop_str = to_yyyymm(da_regridded.time.isel(time=-1))
            fname = f"{var}_{table_id}_{sid}_{EXPERIMENT_ID}_gr1deg_{start_str}-{stop_str}.nc"
            outpath = REGRID_ROOT / sid / EXPERIMENT_ID / table_id / var
            outpath.mkdir(parents=True, exist_ok=True)

            # Save to NetCDF
            print(f"Saving...", end="", flush=True)
            save_t0 = time.perf_counter()
            xr.Dataset({var: da_regridded}).to_netcdf(outpath / fname)
            save_elapsed = time.perf_counter() - save_t0
            print(f"done in {save_elapsed:.2f}s.  {outpath / fname}")

            del da
            del da_regridded
            gc.collect()

            timings[(sid, var)] = {"regrid": regrid_elapsed, "save": save_elapsed}

    run_elapsed = time.perf_counter() - run_t0

    print("\n==== Timing summary ====")
    print(f"{'source_id':<15}{'variable':<12}{'regrid [s]':>12}{'save [s]':>12}")
    total_regrid = 0.0
    total_save = 0.0
    for (sid, var), t in timings.items():
        print(f"{sid:<15}{var:<12}{t['regrid']:>12.2f}{t['save']:>12.2f}")
        total_regrid += t["regrid"]
        total_save += t["save"]
    print(f"\nLoad time:              {load_elapsed:.2f}s")
    print(f"Total regridding time:  {total_regrid:.2f}s")
    print(f"Total save time:        {total_save:.2f}s")
    print(f"Total run time:         {run_elapsed:.2f}s")


if __name__ == "__main__":
    main()
