"""
etunc.load.trendy
=================
Load TRENDY (GCB2025) DGVM output and the forcing shared by every model,
downloaded by scripts/download/trendy/ to TRENDY_ROOT: model output as
<MODEL>/<experiment dir>/<prefix>_<S2|S3>_<var>.nc, the forcing in crujra3/
(CRU-JRA v3) and Radiation/. The file prefix is not always the model
directory (CLM -> CLM6.0, ELM -> E3SM, TEM -> GDSTEM, ...).
Experiments: "S2" (climate + CO2) and "S3" (climate + CO2 + land use change).

- `models`, `find_file`, `missing_variables`: which models have which variables.
- `load_native`: monthly fields of one model and experiment over a time slice
  on the native grid, with lon in [-180, 180] and lat ascending, in native
  units (step 1 of the binning pipeline, scripts/prep_trendy.py).
- `load_forcing`: monthly means or totals of a 6-hourly forcing variable
  (FORCING: precipitation "pre", downward longwave "dlwrf", diffuse fraction "fd").

The model files are not consistent with each other:
- time: every file is a consecutive monthly series ending in Dec 2024, but 8
  models' time cannot be CF-decoded ("months since", decimal years, or no
  units), and the others stamp the start, middle or end of each month.
  `_monthly_time` rebuilds every axis from its length (see LAST_MONTH).
- coords: latitude/longitude or lat/lon dims, lon in [0, 360) or [-180, 180],
  lat ascending or descending; ELM keeps its coordinates in the non-index
  variables latitude(lat) and longitude(lon).
- units: the labels vary ("kg m-2 s-1", "kg.m-2.s-1", "kg m$^{-2}$ s$^{-1}$",
  ...) and are kept as they are. evapotrans is kg m-2 s-1 in every model.
- ocean: NaN in most models, 0 in some (iMAPLE); the land mask of step 5
  removes it.

Example
-------
>>> import etunc.load.trendy as lt
>>> lt.missing_variables("ISAM", "S3", ["evapotrans", "lai"])
['lai']
>>> raw = lt.load_native("CLM", "S3", slice("1995-01", "2014-12"), ["evapotrans", "lai"])
>>> pr = lt.load_forcing("pre", slice("1995-01", "2014-12"))  # mm/month
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

from etunc.config import TRENDY_ROOT
from etunc.grid import check_same_grid, format_grid


ROOT = TRENDY_ROOT

EXPERIMENTS = ("S2", "S3")
# Experiment directories not named after the experiment: CLASSIC re-uploaded a corrected S2
EXPERIMENT_DIRS = {("CLASSIC", "S2"): "correctedS2"}
# Last month of every GCB2025 run (checked for every downloaded file)
LAST_MONTH = "2024-12"

# Forcing variables: {variable: (directory under ROOT, file of one year, monthly "sum" or "mean")}. Every
# file holds one year of 6-hourly values in a 365-day calendar, each stamped at the start of its 6 hours,
# on one 0.5 deg grid with values on land only
FORCING = {
    "pre": ("crujra3", "crujra.v3.02.5d.pre.{year}.365d.noc.nc", "sum"),     # precipitation [mm/6h]
    "dlwrf": ("crujra3", "crujra.v3.5d.dlwrf.{year}.365d.noc.nc", "mean"),   # downward longwave [W/m^2]
    # Diffuse fraction of the downward shortwave [1]; 0 at night, so its plain monthly mean is not
    # weighted by the shortwave
    "fd": ("Radiation", "fd_gcb2025_{year}.nc", "mean"),
}


# ------------------------------------------------------------------
# Files
# ------------------------------------------------------------------

def models() -> list[str]:
    """Model directories under ROOT: those with an S2 or S3 subdirectory (so not crujra3/, Radiation/)."""
    return sorted(
        (p.name for p in ROOT.iterdir() if p.is_dir() and any((p / e).is_dir() for e in EXPERIMENTS)),
        key=str.lower,
    )


def find_file(model: str, experiment: str, variable: str) -> Path | None:
    """The file of `variable` for `model` and `experiment`, or None if there is none."""
    folder = ROOT / model / EXPERIMENT_DIRS.get((model, experiment), experiment)
    # Top level only: LPJ-GUESS keeps an older version of its S3 lai in S3/v1.0/
    files = sorted(folder.glob(f"*_{experiment}_{variable}.nc"))
    if len(files) > 1:
        raise ValueError(f"{model}/{experiment}: more than one {variable} file: {[f.name for f in files]}")
    return files[0] if files else None


def missing_variables(model: str, experiment: str, variables: Sequence[str]) -> list[str]:
    """The `variables` without a file for `model` and `experiment`."""
    return [v for v in variables if find_file(model, experiment, v) is None]


# ------------------------------------------------------------------
# Monthly fields for binning (scripts/prep_trendy.py)
# ------------------------------------------------------------------

def _monthly_time(time: xr.DataArray, label: str) -> xr.CFTimeIndex:
    """
    First-of-month stamps for the raw (undecoded) `time` of a monthly series
    ending in LAST_MONTH, in the file's calendar ("standard" if it has none).
    Where `time` can be CF-decoded, its year and month must match each stamp,
    so a series that does not end in LAST_MONTH, or that stamps each month at
    the start of the next one, raises ValueError.
    """
    calendar = time.attrs.get("calendar", "standard")
    months = xr.date_range(end=f"{LAST_MONTH}-01", periods=time.size, freq="MS", calendar=calendar, use_cftime=True)

    # Undecodable units ("months since" outside a 360-day calendar, decimal years) raise
    # ValueError; a time without units stays numeric. Both keep the rebuilt axis unchecked
    try:
        decoded = xr.decode_cf(
            xr.Dataset({"t": ("time", time.values, time.attrs)}),
            decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
        )["t"].values
    except ValueError:
        return months
    if decoded.dtype != object:
        return months

    found = np.array([(t.year, t.month) for t in decoded])
    expected = np.array([(t.year, t.month) for t in months])
    bad = np.flatnonzero((found != expected).any(axis=1))
    if bad.size:
        i = bad[0]
        raise ValueError(
            f"{label}: time step {i} is {found[i][0]}-{found[i][1]:02d}, but a monthly series ending in "
            f"{LAST_MONTH} has {expected[i][0]}-{expected[i][1]:02d} there"
        )
    return months


def _format(ds: xr.Dataset, variable: str) -> xr.DataArray:
    """`variable` of `ds` with lat/lon dims and coords, lon in [-180, 180] and lat ascending."""
    # ELM: the lat/lon dims have no index; the coordinates are the variables latitude(lat), longitude(lon)
    for name, dim in (("latitude", "lat"), ("longitude", "lon")):
        if name in ds.variables and ds[name].dims == (dim,):
            ds = ds.assign_coords({dim: ds[name].values}).drop_vars(name)
    ds = ds.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in ds.dims})
    # Only the variable itself: some files also hold others (CARDAMOM quantiles, ISAM `i`)
    return format_grid(ds[variable])


def load_native(
    model: str, experiment: str, time_slice: slice, variables: Sequence[str],
) -> dict[str, xr.DataArray]:
    """
    Monthly fields {variable: (time, lat, lon)} of `model` in `experiment`
    over `time_slice` on the model's native grid, with lon in [-180, 180] and
    lat ascending, in native units, with first-of-month time stamps in the
    file's calendar (step 1 of the binning pipeline). Raises ValueError if a
    variable has no file or the variables are not on one grid.
    """
    out = {}
    for v in variables:
        path = find_file(model, experiment, v)
        if path is None:
            raise ValueError(f"{model}/{experiment}: no {v} file")
        label = f"{model}/{experiment}/{v}"
        ds = xr.open_dataset(path, decode_times=False, chunks={"time": 120})
        ds = ds.assign_coords(time=_monthly_time(ds["time"], label))
        out[v] = _format(ds, v).sel(time=time_slice).reset_coords(drop=True)

    ref = out[variables[0]]
    for v, da in out.items():
        check_same_grid(da, ref, f"{model}/{experiment}/{v}")
    return out


def load_forcing(variable: str, time_slice: slice) -> xr.DataArray:
    """
    Monthly values (time, lat, lon) of the 6-hourly forcing `variable` (a key
    of FORCING) over `time_slice`, on the 0.5 deg forcing grid with lon in
    [-180, 180] and lat ascending, with first-of-month stamps in the 365-day
    calendar. Totals ("pre", mm/6h) are summed into monthly totals [mm/month],
    which `etunc.units.accumulation_to_flux` converts to kg m-2 s-1; fluxes and
    fractions are averaged, in their native units.
    """
    folder, filename, how = FORCING[variable]
    first, last = int(str(time_slice.start)[:4]), int(str(time_slice.stop)[:4])
    files = [ROOT / folder / filename.format(year=year) for year in range(first, last + 1)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"No {variable} forcing in {ROOT / folder}: {missing}")

    da = xr.open_mfdataset(
        files, combine="by_coords", chunks={"time": 4 * 31},
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )[variable]
    # The 6-hourly steps of each month (stamped at their start, so all inside the month); min_count
    # keeps a month without any valid step NaN, as the mean does
    if how == "sum":
        monthly = da.resample(time="MS").sum(min_count=1)
        monthly.attrs = {**da.attrs, "units": "mm/month"}
    else:
        monthly = da.resample(time="MS").mean()
        monthly.attrs = da.attrs
    return format_grid(monthly).sel(time=time_slice)
