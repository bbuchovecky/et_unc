"""
regrid_cmip.py
==============
DEPRECATED: prep_cmip.py regrids the native fields itself,
so no pipeline reads the files written here; only the check-cmip-grids and
load-cmip notebooks do. Kept until it is decided whether those still need them.

Conservatively regrid CMIP6 fields listed in the ESGF cache catalog
(config.ESGF_CACHE_CATALOG) onto the common grid (TARGET_RES) with
`etunc.legacy.load_cmip.regrid_to_target`, one file per (model, variable)
under `etunc.legacy.load_cmip.CMIP_REGRID_ROOT`. (Was regrid_cmip_esgf.py.)
"""
from __future__ import annotations

import gc
import os
import time
from datetime import datetime as dt
import xarray as xr

import etunc.config as config
import etunc.grid as rg
import etunc.legacy.grid as legacy_grid
import etunc.legacy.load_cmip as legacy_cmip
import etunc.legacy.temporal as legacy_temporal
from etunc.load.cmip import CMIPESGFLoader


CATALOG_PATH = config.ESGF_CACHE_CATALOG
REGRID_ROOT = legacy_cmip.CMIP_REGRID_ROOT

VARIABLES = ["evspsbl", "tran", "evspsblsoi", "evspsblveg", "lai", "gpp"]
EXPERIMENT_ID = "historical"
SOURCE_IDS = None
MEMBER_IDS = None
TIME_SLICE = slice("1950-01", "2014-12")

TARGET_RES = 1.0  # spacing [deg] of the common grid, see regrid.target_grid


def main() -> None:
    run_t0 = time.perf_counter()
    grid = rg.target_grid(TARGET_RES)
    print(f"Target grid {rg.grid_tag(TARGET_RES)}: nlat={grid.sizes['lat']}, nlon={grid.sizes['lon']}")

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
            da_regridded = legacy_cmip.regrid_to_target(da, TARGET_RES)
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
            da_regridded.attrs["src_lat_name"] = legacy_grid.coord_name(da, ["lat", "latitude", "nav_lat"], ["latitude"])
            da_regridded.attrs["src_lon_name"] = legacy_grid.coord_name(da, ["lon", "longitude", "nav_lon"], ["longitude"])
            da_regridded.attrs["src_dlat_deg"], da_regridded.attrs["src_dlon_deg"] = legacy_grid.approx_resolution(da)
            da_regridded.attrs["tgt_dlat_deg"], da_regridded.attrs["tgt_dlon_deg"] = legacy_grid.approx_resolution(da_regridded)
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
            start_str = legacy_temporal.to_yyyymm(da_regridded.time.isel(time=0))
            stop_str = legacy_temporal.to_yyyymm(da_regridded.time.isel(time=-1))
            fname = f"{var}_{table_id}_{sid}_{EXPERIMENT_ID}_gr{rg.grid_tag(TARGET_RES)}_{start_str}-{stop_str}.nc"
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
