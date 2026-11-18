"""Download FLUXCOM-X-BASE ET (0.5 deg monthly, 2001-2021) from the ICOS Carbon Portal, following
    https://gitlab.gwdg.de/fluxcom/fluxcomxdata/-/blob/main/download_xbase_from_icos.py
(Nelson et al. 2024, Biogeosciences, doi:10.5194/bg-21-5079-2024). The data object URLs below are
those of that script for ET at 050_monthly; each file is one year (~5 MB, ~105 MB in all).

Local layout:  OUTPUT_ROOT/ET_<year>_050_monthly.nc
OUTPUT_ROOT is SCRATCH_ROOT/fluxcom-x/050_monthly/ET (scratch_root in config/paths.yml); move it to
OBS_ROOT by hand once complete.
A file whose size matches the ICOS metadata is skipped, so re-running is safe and retries only the
missing or incomplete files.

Needs an ICOS Carbon Portal account (https://cpauth.icos-cp.eu/) and the icoscp_core and pyyaml packages
(envs/icos.yml). Store the login once per machine, interactively:
    conda env create -f envs/icos.yml
    conda run -n icos python -c "from icoscp_core.icos import auth; auth.init_config_file()"
Run:  conda run -n icos python scripts/download/download_fluxcom_x.py
"""

import os
import time
from pathlib import Path

import yaml
from icoscp_core.icos import data, meta

# This script runs in the icos conda env, without etunc, so it reads scratch_root from the paths file
# itself: config/paths.yml, or the file named by ETUNC_PATHS, as etunc.config does
PATHS_FILE = Path(os.environ.get("ETUNC_PATHS") or Path(__file__).resolve().parents[2] / "config" / "paths.yml")
with open(PATHS_FILE) as f:
    SCRATCH_ROOT = Path(yaml.safe_load(f)["scratch_root"])
OUTPUT_ROOT = SCRATCH_ROOT / "fluxcom-x" / "050_monthly" / "ET"
YEARS = range(2001, 2022)

# (year, ICOS data object) of ET at 050_monthly, from FLUX_MAP of download_xbase_from_icos.py
ET_050_MONTHLY = {
    2001: "https://meta.icos-cp.eu/objects/tWXxHjp2EjPrf0zs3NBVqaK4",
    2002: "https://meta.icos-cp.eu/objects/jF7AwiBrBwEqiY3yQ71zytJe",
    2003: "https://meta.icos-cp.eu/objects/05_kcCnAchqbb5qSxfiDqUZC",
    2004: "https://meta.icos-cp.eu/objects/AkRQz64QEunOblGBykntfrUA",
    2005: "https://meta.icos-cp.eu/objects/FNEagrqaXwLEkjesUMf7-FI1",
    2006: "https://meta.icos-cp.eu/objects/IQqBF81927krn2ZPVAMtdcrt",
    2007: "https://meta.icos-cp.eu/objects/HCSByFXVWuS92o73yc5mV_Yb",
    2008: "https://meta.icos-cp.eu/objects/FoXg13h0Jnn9hycgobFWFj3V",
    2009: "https://meta.icos-cp.eu/objects/IzmRTvri7KF0uPoeQ7GfiYwE",
    2010: "https://meta.icos-cp.eu/objects/7SOLdmdcpCDcdJN_rw0XDNu6",
    2011: "https://meta.icos-cp.eu/objects/TkNilYrx51pBlU-XlgEzTnON",
    2012: "https://meta.icos-cp.eu/objects/Ox4UewNUdQp7wI8f_DpO9i5g",
    2013: "https://meta.icos-cp.eu/objects/ovg_VE_kmoZJUul6bX5Jl3r0",
    2014: "https://meta.icos-cp.eu/objects/s9aEPeENzEpypFWkYcIKFGSY",
    2015: "https://meta.icos-cp.eu/objects/6lXSMMh8XAPlGo3kshQLP9YR",
    2016: "https://meta.icos-cp.eu/objects/vih6JZ6DVSzOXdBxzMg5xRU7",
    2017: "https://meta.icos-cp.eu/objects/dwfpDyDVjk5WCVimY3lBjRxD",
    2018: "https://meta.icos-cp.eu/objects/5B-kX-ay0uEGaM84a-Tic5Q1",
    2019: "https://meta.icos-cp.eu/objects/UcNnInhqhJmId-2Xhv8yXgkb",
    2020: "https://meta.icos-cp.eu/objects/9L6zqwQPYBeETakE8qihkrT2",
    2021: "https://meta.icos-cp.eu/objects/4t19mTKVrL-Vt_plO7TCFQDE",
}


def download_year(year):
    """Save one year's file to OUTPUT_ROOT unless a complete copy is already there."""
    uri = ET_050_MONTHLY[year]
    dobj = meta.get_dobj_meta(uri)  # file name and size, no login needed
    if dobj.nextVersion:
        print(f"  note: {dobj.fileName} has a newer version on ICOS: {dobj.nextVersion}")
    path = OUTPUT_ROOT / dobj.fileName
    if path.exists() and path.stat().st_size == dobj.size:
        return
    data.save_to_folder(uri, str(OUTPUT_ROOT))
    if path.stat().st_size != dobj.size:
        raise OSError(f"{path} is {path.stat().st_size} bytes, ICOS says {dobj.size}")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"{time.strftime('%F %T')}  FLUXCOM-X-BASE ET 050_monthly {YEARS[0]}-{YEARS[-1]} to {OUTPUT_ROOT}")

    failed = []
    for year in YEARS:
        try:
            download_year(year)
            print(f"{time.strftime('%F %T')}  done {year}")
        except Exception as err:
            print(f"{time.strftime('%F %T')}  FAILED {year}: {err}")
            failed.append(year)

    if failed:
        raise SystemExit(f"FAILED: {failed}; re-run to retry.")


if __name__ == "__main__":
    main()
