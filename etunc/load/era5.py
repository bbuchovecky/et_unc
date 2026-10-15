"""
etunc.load.era5
===============
Load ERA5 monthly means from the NCAR GDEX archive (ds633.0 / ds633.5) and the
ERA5 grid file (LANDFRAC, ...).

- `load_era5`: one monthly-mean parameter on the native 0.25 deg grid.
- `load_native`: several "meanflux" parameters over a time slice, with lon in
  [-180, 180] and lat ascending, in native units (step 1 of the binning
  pipeline, scripts/prep_obs.py).
- `load_land_fraction`: LANDFRAC and the native grid with cell edges.

Example
-------
>>> import etunc.load.era5 as le
>>> grid = le.load_era5_grid()
>>> swnet = le.load_era5("msnswrf", "1995-01", "2014-12", kind="meanflux")
>>> raw = le.load_native(("mer", "mtpr"), slice("1995-01", "2014-12"))
>>> landfrac, src_grid = le.load_land_fraction()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
from typing import Sequence

import xarray as xr

from etunc.grid import bounded_src_grid, format_grid


GDEX_ROOTS = {
    # (years < SPLIT_YEAR, years >= SPLIT_YEAR)
    "sfc": (Path("/gdex/data/d633005/e5p.moda.an.sfc"), Path("/gdex/data/d633001/e5.moda.an.sfc")),
    "meanflux": Path("/gdex/data/d633001/e5.moda.fc.sfc.meanflux"),
    "accum": Path("/gdex/data/d633001/e5.moda.fc.sfc.accumu"),
}
SPLIT_YEAR = 1979

ERA5_GRID_PATH = Path("/glade/work/bbuchovecky/data/era5/e5.invariant.grid.nc")

def _root(kind: str, year: int) -> Path:
    paths = GDEX_ROOTS[kind]
    if isinstance(paths, tuple):
        return paths[0] if year < SPLIT_YEAR else paths[1]
    return paths


def load_era5(
    var_code: str,
    start: str,
    end: str,
    kind: str | None = None,
    chunks: dict | None = None,
) -> xr.DataArray | xr.Dataset:
    """
    Load an ERA5 monthly-mean variable for the years of [start, end].

    Parameters
    ----------
    var_code : str
        Parameter code as it appears at the end of the filename, before the
        grid tag (e.g. "msnswrf", or "128_167_2t" for 2 m temperature).
    start, end : str
        "YYYY-MM" strings. Whole years are loaded; the months are not trimmed.
    kind : str, optional
        Which GDEX_ROOTS stream to read ("sfc", "meanflux" or "accum"). If
        None, the first stream with a matching file is used, in that order.
    chunks : dict, optional
        Dask chunks. Defaults to {"time": 1}, one month per chunk.

    Returns
    -------
    xr.DataArray
        Lazily loaded and concatenated along time, with dims renamed to
        lat/lon. If the files hold several variables (e.g. "MER" and
        "utc_date") and none is named "<VAR_CODE>" or "VAR_<VAR_CODE>", the
        whole Dataset is returned with a warning.

    Raises
    ------
    FileNotFoundError
        If any year has no matching file, since that more often means a wrong
        `var_code` or `kind` than a real gap.
    """
    if chunks is None:
        chunks = {"time": 1}

    files = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        # The first stream with a match wins, so pass `kind` for a code found in more than one stream
        for k in [kind] if kind is not None else GDEX_ROOTS:
            subdir = _root(k, year) / f"{year}"
            matches = sorted(subdir.glob(f"*_{var_code}.*.nc"))
            if matches:
                break
        if not matches:
            raise FileNotFoundError(
                f"No file matching '*_{var_code}.*.nc' under {subdir}. "
                f"Verify var_code against actual filenames there, and "
                f"confirm {year} is assigned to the correct root."
            )
        files.extend(matches)

    if not files:
        raise FileNotFoundError(f"No files found for '{var_code}' in range {start} to {end}.")

    ds = xr.open_mfdataset(files, combine="by_coords", parallel=True)
    ds = ds.chunk(chunks).rename({"latitude": "lat", "longitude": "lon"})

    # Mean fluxes keep ERA5's sign (positive downward), so evaporation ("mer") is negative;
    # `etunc.units.convert_units` flips it
    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        return ds[data_vars[0]]
    # Mean fluxes are named by their code ("MER", next to "utc_date"), analysis fields "VAR_<CODE>"
    for name in (var_code.upper(), "VAR_" + var_code.upper()):
        if name in ds:
            return ds[name]
    sys.stderr.write(
        f"WARNING: {len(data_vars)} data variables found for '{var_code}': "
        f"{data_vars}. Returning the full Dataset; select explicitly.\n"
    )
    return ds


@lru_cache(maxsize=1)
def load_era5_grid() -> xr.Dataset:
    """ERA5 grid dataset (LANDFRAC, ...) on the 0.25 deg lat/lon grid."""
    return xr.open_dataset(ERA5_GRID_PATH)


def load_land_fraction() -> tuple[xr.DataArray, xr.Dataset]:
    """
    ERA5 LANDFRAC [0-1] with lon in [-180, 180] and lat ascending, and its
    grid with cell edges for regridding.
    """
    # The grid file's scalar `time` coord would clash with the time axis of the monthly fields
    landfrac = format_grid(load_era5_grid()["LANDFRAC"]).reset_coords(drop=True)
    return landfrac, bounded_src_grid(landfrac)


def load_native(codes: Sequence[str], time_slice: slice) -> dict[str, xr.DataArray]:
    """
    Monthly means {code: (time, lat, lon)} of the "meanflux" parameters
    `codes` (e.g. "mer", "mtpr", "msnswrf", "msnlwrf") over `time_slice` on
    the native 0.25 deg grid, with lon in [-180, 180] and lat ascending, in
    native units and sign (fluxes positive downward, so evaporation "mer" is
    negative). Step 1 of the binning pipeline (scripts/prep_obs.py).
    """
    # Whole years are read (load_era5 takes years); the months are cut by `.sel(time=time_slice)`
    first, last = str(time_slice.start)[:4], str(time_slice.stop)[:4]
    return {
        code: format_grid(load_era5(code, f"{first}-01", f"{last}-12", kind="meanflux")).sel(time=time_slice)
        for code in codes
    }
