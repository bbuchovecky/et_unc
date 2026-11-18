"""Download GLDAS evapotranspiration (Evap_tavg, kg m-2 s-1) as monthly means from Google Earth Engine:
    GLDAS-2.0 Noah, 3-hourly    NASA/GLDAS/V20/NOAH/G025/T3H    (1948-2014; Rodell et al. 2004)
    GLDAS-2.1 Noah, 3-hourly    NASA/GLDAS/V021/NOAH/G025/T3H   (2000-present)
    GLDAS-2.2 Catchment, daily, GRACE data assimilation
                                NASA/GLDAS/V022/CLSM/G025/DA1D  (Feb 2003-present; Li et al. 2019)
https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V022_CLSM_G025_DA1D

The native fields (~210 GB per 3-hourly product for 2001-2021) are averaged to monthly means on
Earth Engine: each month is the mean of the images whose time_start falls in that month (3-hourly:
00-21 UTC, 8 per day; daily: one per day). n_steps counts the valid time steps per gridcell, so
incomplete months can be dropped downstream (n_steps_full is a complete month); ocean is 0. A month
without images (GLDAS-2.2 before Feb 2003) is NaN.
Every product is on the same native 0.25 deg grid (1440 x 600, lon -180 to 180, lat -60 to 90),
which is requested exactly, so nothing is resampled. Years are YEARS clipped to each product's record.

Local layout:  OUTPUT_ROOT/<product>/GLDAS_<product>_Evap_tavg_monthly_<year>.nc   (~25 MB per year)
OUTPUT_ROOT is SCRATCH_ROOT/gldas (scratch_root in config/paths.yml); move it to OBS_ROOT by hand once
complete.
A year whose file exists is skipped, so re-running is safe and retries only the failed years.

Needs the openet conda env (envs/openet.yml, earthengine-api) and Earth Engine credentials
(`earthengine authenticate`). The Google Cloud project is EE_PROJECT if set, otherwise the default
project of the credentials.
Run:  conda run -n openet python scripts/download/download_gldas.py
"""

import calendar
import os
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import xarray as xr
import yaml

# This script runs in the openet conda env, without etunc, so it reads scratch_root from the paths file
# itself: config/paths.yml, or the file named by ETUNC_PATHS, as etunc.config does
PATHS_FILE = Path(os.environ.get("ETUNC_PATHS") or Path(__file__).resolve().parents[2] / "config" / "paths.yml")
with open(PATHS_FILE) as f:
    SCRATCH_ROOT = Path(yaml.safe_load(f)["scratch_root"])
OUTPUT_ROOT = SCRATCH_ROOT / "gldas"
YEARS = range(2001, 2022)
BAND = "Evap_tavg"

# product: (collection, first year, last year, time steps per day)
PRODUCTS = {
    "2.0": ("NASA/GLDAS/V20/NOAH/G025/T3H", 1948, 2014, 8),
    "2.1": ("NASA/GLDAS/V021/NOAH/G025/T3H", 2000, 2025, 8),
    "2.2-grace-da": ("NASA/GLDAS/V022/CLSM/G025/DA1D", 2003, 2025, 1),
}

# Native GLDAS 0.25 deg grid of all three collections (EPSG:4326, rows from north to south)
RES = 0.25
NX, NY = 1440, 600
WEST, NORTH = -180.0, 90.0
GRID = {
    "dimensions": {"width": NX, "height": NY},
    "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": WEST,
                        "shearY": 0, "scaleY": -RES, "translateY": NORTH},
    "crsCode": "EPSG:4326",
}
FILL = -9999.0  # stands for masked pixels in the transfer, NaN in the files


def monthly_mean(collection, year, month):
    """Mean Evap_tavg and number of valid time steps per gridcell for one month, as numpy arrays
    (rows north to south). A month without images (before a product starts) is NaN and 0."""
    start = ee.Date.fromYMD(year, month, 1)
    images = ee.ImageCollection(collection).filterDate(start, start.advance(1, "month")).select(BAND)
    if images.size().getInfo() == 0:
        return np.full((NY, NX), np.nan, dtype="float32"), np.zeros((NY, NX), dtype="int16")

    mean = images.mean().unmask(FILL).toFloat().rename("mean")
    n_steps = images.count().unmask(0).toInt16().rename("n_steps")
    pixels = ee.data.computePixels(
        {"expression": mean.addBands(n_steps), "fileFormat": "NUMPY_NDARRAY", "grid": GRID}
    )
    return np.where(pixels["mean"] == FILL, np.nan, pixels["mean"]), pixels["n_steps"]


def download_year(product, year):
    """Write one year of monthly means to OUTPUT_ROOT/<product>/ (skipped if the file exists)."""
    collection, _, _, steps_per_day = PRODUCTS[product]
    path = OUTPUT_ROOT / product / f"GLDAS_{product}_{BAND}_monthly_{year}.nc"
    if path.exists():
        return

    means, counts = [], []
    for month in range(1, 13):
        mean, n_steps = monthly_mean(collection, year, month)
        means.append(mean)
        counts.append(n_steps)

    # Flip the rows so lat is ascending, as in the rest of the pipeline
    lat = np.arange(NORTH - RES / 2, NORTH - NY * RES, -RES)[::-1]
    lon = np.arange(WEST + RES / 2, WEST + NX * RES, RES)
    coords = {"time": pd.date_range(f"{year}-01-01", periods=12, freq="MS"), "lat": lat, "lon": lon}
    days = [calendar.monthrange(year, m)[1] for m in range(1, 13)]
    ds = xr.Dataset(
        {
            BAND: (("time", "lat", "lon"), np.stack(means)[:, ::-1, :].astype("float32"),
                   {"units": "kg m-2 s-1", "long_name": "evapotranspiration, monthly mean"}),
            "n_steps": (("time", "lat", "lon"), np.stack(counts)[:, ::-1, :].astype("int16"),
                        {"long_name": "number of valid time steps in the monthly mean"}),
            "n_steps_full": (("time",), np.array(days, dtype="int16") * steps_per_day,
                             {"long_name": "number of time steps in a complete month"}),
        },
        coords=coords,
        attrs={
            "source": f"Google Earth Engine collection {collection}, band {BAND}",
            "processing": "monthly mean of the images whose time_start is in the month, on the native grid",
            "created": time.strftime("%F %T"),
        },
    )
    ds.lat.attrs.update(units="degrees_north", standard_name="latitude")
    ds.lon.attrs.update(units="degrees_east", standard_name="longitude")

    # Write to a temporary name and rename, so the file only exists once complete
    tmp = path.with_suffix(".nc.part")
    encoding = {v: {"zlib": True, "complevel": 4} for v in (BAND, "n_steps")}
    ds.to_netcdf(tmp, encoding=encoding)
    tmp.rename(path)


def main():
    ee.Initialize(project=os.environ.get("EE_PROJECT"))

    tasks = []
    for product, (_, first, last, _) in PRODUCTS.items():
        (OUTPUT_ROOT / product).mkdir(parents=True, exist_ok=True)
        tasks += [(product, y) for y in YEARS if first <= y <= last]
    print(f"{time.strftime('%F %T')}  {len(tasks)} product-years to {OUTPUT_ROOT}")

    # One request per month, sent one at a time to stay within the Earth Engine request quota
    failed = []
    for product, year in tasks:
        try:
            download_year(product, year)
            print(f"{time.strftime('%F %T')}  done GLDAS-{product} {year}")
        except Exception as err:
            print(f"{time.strftime('%F %T')}  FAILED GLDAS-{product} {year}: {err}")
            failed.append((product, year))

    if failed:
        raise SystemExit(f"FAILED: {failed}; re-run to retry.")


if __name__ == "__main__":
    main()
