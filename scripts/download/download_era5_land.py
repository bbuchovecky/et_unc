"""Download ERA5-Land monthly means (0.1 deg, 1950 onward) from the Copernicus Climate Data Store.
Dataset: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means

Local layout:  OUTPUT_ROOT/monthly/<year>/*.nc   (time-varying fields, one CDS request per year)
               OUTPUT_ROOT/invariant/*.nc        (static fields, one request for a single month)
OUTPUT_ROOT is SCRATCH_ROOT/era5-land (config/paths.yml); move it to OBS_ROOT/era5-land, where
etunc.load.era5.load_era5_land reads it, by hand once complete. Each request comes as a zip, extracted
here: the time-varying fields gave one file per year (data_stream-moda.nc), but the CDS may split a
request into one file per step type (accumulated vs instantaneous fields), which the loader merges.
Requests are sent one at a time, since the CDS rejects requests beyond its per-user queue limit. A folder that exists is skipped, so re-running is safe
and retries only the failed requests.

At 0.1 deg one field is ~26 MB uncompressed, so a year of the time-varying fields is up to ~2.5 GB.

Needs a CDS account with the dataset licence accepted on its web page, and ~/.cdsapirc with
    url: https://cds.climate.copernicus.eu/api
    key: <your personal access token>
Run:  pixi run python scripts/download/download_era5_land.py
Long runs: run inside tmux, or nohup ... > download_era5_land.log 2>&1 &
"""

import shutil
import time
import zipfile

import cdsapi

from etunc.config import SCRATCH_ROOT

OUTPUT_ROOT = SCRATCH_ROOT / "era5-land"
DATASET = "reanalysis-era5-land-monthly-means"
# Complete years only: ERA5-Land lags real time by ~2-3 months, and a partial year would be
# dropped by the complete-year rule of the pipeline anyway
YEARS = range(1950, 2026)
MONTHS = [f"{m:02d}" for m in range(1, 13)]
VARIABLES = [
    "surface_latent_heat_flux",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_sensible_heat_flux",
    "total_evaporation",
    "total_precipitation",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
]
# Constant in time in ERA5-Land, so retrieved for one month only
INVARIANT_VARIABLES = [
    "high_vegetation_cover",
    "low_vegetation_cover",
    "glacier_mask",
    "lake_cover",
    "land_sea_mask",
]
INVARIANT_DATE = ("2001", "01")


def retrieve(client, variables, years, months, out_dir):
    """Retrieve one request as a zip and extract it to out_dir (skipped if out_dir exists)."""
    if out_dir.exists():
        return
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": variables,
        "year": years,
        "month": months,
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "zip",
    }
    zip_path = out_dir.with_name(f"{out_dir.name}.zip")
    client.retrieve(DATASET, request, str(zip_path))

    # Extract to a temporary folder and rename it, so out_dir only exists once complete
    tmp_dir = out_dir.with_name(f"{out_dir.name}.part")
    shutil.rmtree(tmp_dir, ignore_errors=True)  # left over from an interrupted run
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)
    tmp_dir.rename(out_dir)
    zip_path.unlink()


def main():
    (OUTPUT_ROOT / "monthly").mkdir(parents=True, exist_ok=True)

    tasks = [("invariant", INVARIANT_VARIABLES, [INVARIANT_DATE[0]], [INVARIANT_DATE[1]],
              OUTPUT_ROOT / "invariant")]
    tasks += [(str(y), VARIABLES, [str(y)], MONTHS, OUTPUT_ROOT / "monthly" / str(y)) for y in YEARS]
    print(f"{time.strftime('%F %T')}  {len(tasks)} requests to {OUTPUT_ROOT}")

    client = cdsapi.Client()
    failed = []
    for name, variables, years, months, out_dir in tasks:
        try:
            retrieve(client, variables, years, months, out_dir)
            print(f"{time.strftime('%F %T')}  done {name}")
        except Exception as err:
            print(f"{time.strftime('%F %T')}  FAILED {name}: {err}")
            failed.append(name)

    n_files = len(list(OUTPUT_ROOT.rglob("*.nc")))
    print(f"{time.strftime('%F %T')}  {n_files} .nc files in {OUTPUT_ROOT}")
    if failed:
        raise SystemExit(f"FAILED: {failed}; re-run to retry.")


if __name__ == "__main__":
    main()
