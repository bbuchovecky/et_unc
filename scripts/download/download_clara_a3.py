"""Download the CLARA-A3 surface radiation budget (EUMETSAT CM SAF, 0.25 deg monthly means) from the
Copernicus Climate Data Store, as in the C3S tutorial
    https://ecmwf-projects.github.io/copernicus-training-c3s/ecv-notebooks/srb-climatology-and-anomaly.html
Dataset: https://cds.climate.copernicus.eu/datasets/satellite-surface-radiation-budget

The tutorial downloads only the downwelling shortwave flux (SIS); this script also gets the
longwave and net fluxes. The thematic climate data record (TCDR) covers 1979-2020; the interim
record (ICDR, CDR_TYPE = "interim_climate_data_record") continues it from 2021.

Local layout:  OUTPUT_ROOT/<variable>/<year>/*.nc
There is one CDS request per variable and year, sent one at a time: the CDS rejects requests
beyond its per-user queue limit for this dataset ("Maximum number queued requests ... reached").
A year whose folder exists is skipped, so re-running is safe and retries only the failed requests.

Needs a CDS account with the dataset licence accepted on its web page, and ~/.cdsapirc with
    url: https://cds.climate.copernicus.eu/api
    key: <your personal access token>
Run:  pixi run python scripts/download/download_clara_a3.py
Long runs: run inside tmux, or nohup ... > download_clara_a3.log 2>&1 &
"""

import shutil
import time
import zipfile
from pathlib import Path

import cdsapi

OUTPUT_ROOT = Path("/glade/derecho/scratch/bbuchovecky/clara-a3")
DATASET = "satellite-surface-radiation-budget"
CDR_TYPE = "thematic_climate_data_record"
YEARS = range(1979, 2021)
MONTHS = [f"{m:02d}" for m in range(1, 13)]
VARIABLES = [
    "surface_downwelling_shortwave_flux",    # SIS
    "surface_downwelling_longwave_flux",     # SDL
    "surface_net_downward_shortwave_flux",
    "surface_net_downward_longwave_flux",
    "surface_net_downward_radiative_flux",   # net radiation Rn
]


def download_year(client, variable, year):
    """Retrieve one variable-year zip and extract it to OUTPUT_ROOT/<variable>/<year>/."""
    out_dir = OUTPUT_ROOT / variable / str(year)
    if out_dir.exists():
        return
    request = {
        "product_family": "clara_a3",
        "origin": "eumetsat",
        "variable": [variable],
        "climate_data_record_type": CDR_TYPE,
        "time_aggregation": "monthly_mean",
        "year": [str(year)],
        "month": MONTHS,
        "format": "zip",
    }
    zip_path = OUTPUT_ROOT / variable / f"{variable}_{year}.zip"
    client.retrieve(DATASET, request, str(zip_path))

    # Extract to a temporary folder and rename it, so out_dir only exists once complete
    tmp_dir = out_dir.with_name(f"{year}.part")
    shutil.rmtree(tmp_dir, ignore_errors=True)  # left over from an interrupted run
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)
    tmp_dir.rename(out_dir)
    zip_path.unlink()


def main():
    for variable in VARIABLES:
        (OUTPUT_ROOT / variable).mkdir(parents=True, exist_ok=True)

    tasks = [(v, y) for v in VARIABLES for y in YEARS]
    print(f"{time.strftime('%F %T')}  {len(tasks)} variable-years to {OUTPUT_ROOT}")

    # Sequential requests: the CDS rejects parallel ones once its queue limit is reached
    client = cdsapi.Client()
    failed = []
    for variable, year in tasks:
        try:
            download_year(client, variable, year)
            print(f"{time.strftime('%F %T')}  done {variable} {year}")
        except Exception as err:
            print(f"{time.strftime('%F %T')}  FAILED {variable} {year}: {err}")
            failed.append((variable, year))

    n_files = len(list(OUTPUT_ROOT.rglob("*.nc")))
    print(f"{time.strftime('%F %T')}  {n_files} .nc files in {OUTPUT_ROOT}")
    if failed:
        raise SystemExit(f"FAILED: {failed}; re-run to retry.")


if __name__ == "__main__":
    main()
