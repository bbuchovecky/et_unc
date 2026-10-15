"""
etunc.load.era5
===============
Load ERA5 monthly means from the NCAR GDEX archive (ds633.0 / ds633.5) and the
ERA5 grid file (LANDFRAC, ...).

- `load_era5`: one monthly-mean parameter on the native 0.25 deg grid.
- Annual means for binning, as in every load module: `load_native` (binning
  variables et, pr and rn of VARIABLES in W/m2, complete years on the native
  grid, masked to land with LANDFRAC > LF_THRESH, no Greenland/Iceland) and
  `load_annual` = `etunc.grid.regrid_annual` of it (conservative).

Example
-------
>>> import etunc.load.era5 as le
>>> grid = le.load_era5_grid()
>>> swnet = le.load_era5("msnswrf", "1995-01", "2014-12", kind="meanflux")
>>> ann = le.load_annual(("et", "pr", "rn"), slice("2001", "2010"), 0.5)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
from typing import Sequence

import xarray as xr

from etunc.config import LF_THRESH, NA_THRESH
from etunc.grid import bounded_src_grid, check_same_grid, format_grid, landfrac_mask
from etunc.units import convert_units, net_radiation_era5
from etunc.legacy.grid import regrid_annual
from etunc.legacy.temporal import annual_means, print_annual


GDEX_ROOTS = {
    # (years < SPLIT_YEAR, years >= SPLIT_YEAR)
    "sfc": (Path("/gdex/data/d633005/e5p.moda.an.sfc"), Path("/gdex/data/d633001/e5.moda.an.sfc")),
    "meanflux": Path("/gdex/data/d633001/e5.moda.fc.sfc.meanflux"),
    "accum": Path("/gdex/data/d633001/e5.moda.fc.sfc.accumu"),
}
SPLIT_YEAR = 1979

ERA5_GRID_PATH = Path("/glade/work/bbuchovecky/data/era5/e5.invariant.grid.nc")

# Mean-rate parameter codes ("meanflux" stream) of each binning variable; rn is the
# sum of the net SW and LW fluxes (`to_wm2`)
VARIABLES = {
    "et": ("mer",),
    "pr": ("mtpr",),
    "rn": ("msnswrf", "msnlwrf"),
}


def _root(kind: str, year: int) -> Path:
    paths = GDEX_ROOTS[kind]
    if isinstance(paths, tuple):
        return paths[0] if year < SPLIT_YEAR else paths[1]
    return paths


def available_years(kind: str = "meanflux") -> list[int]:
    """Sorted years with a directory under the GDEX root that `load_era5` reads for `kind`."""
    roots = GDEX_ROOTS[kind] if isinstance(GDEX_ROOTS[kind], tuple) else (GDEX_ROOTS[kind],)
    years = {int(d.name) for r in roots if r.is_dir() for d in r.iterdir() if d.name.isdigit()}
    # A year counts only under the root that `_root` assigns it to
    return sorted(y for y in years if (_root(kind, y) / str(y)).is_dir())


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


def land_fraction() -> xr.DataArray:
    """ERA5 LANDFRAC with lon in [-180, 180] and lat ascending."""
    # The grid file's scalar `time` coord would clash with the time axis of the monthly fields
    return format_grid(load_era5_grid()["LANDFRAC"]).reset_coords(drop=True)


def to_wm2(variable: str, fields: dict[str, xr.DataArray]) -> xr.DataArray:
    """Binning variable `variable` [W/m2] from its parameter fields {code: DataArray} (VARIABLES)."""
    if variable == "rn":
        return net_radiation_era5(fields["msnswrf"], fields["msnlwrf"])
    (code,) = VARIABLES[variable]
    # mer is positive downward, so convert_units also flips its sign to make ET positive
    return convert_units(code, fields[code])


def load_native(
    variables: Sequence[str], time_slice: slice = slice(None, None),
) -> tuple[dict[str, xr.DataArray], xr.DataArray, xr.Dataset]:
    """
    Annual means {variable: (year, lat, lon)} [W/m2] of binning variables
    `variables` ("et", "pr", "rn"; see VARIABLES) on the native 0.25 deg grid
    over the complete years in `time_slice`, masked to land (LANDFRAC >
    LF_THRESH, no Greenland/Iceland). All 12 months are needed for a valid
    annual mean. An open end of `time_slice` means the first or last year on
    GDEX.

    Returns (annual means, native land mask, native grid), the input of
    `etunc.grid.regrid_annual`.
    """
    # Whole years are read (load_era5 takes years); the months are cut by `.sel(time=time_slice)`
    if time_slice.start is None or time_slice.stop is None:
        years = available_years("meanflux")
    first = int(str(time_slice.start)[:4]) if time_slice.start is not None else min(years)
    last = int(str(time_slice.stop)[:4]) if time_slice.stop is not None else max(years)
    landfrac = land_fraction()
    # Sets the area ERA5 contributes, as sftlf does for CMIP6. Masking on the native grid keeps
    # ocean values out of the area-weighted means of coastal target cells (ERA5 is global)
    native_mask = landfrac_mask(landfrac, LF_THRESH)

    monthly = {}
    for v in variables:
        fields = {
            code: format_grid(load_era5(code, f"{first}-01", f"{last}-12", kind="meanflux")).sel(time=time_slice)
            for code in VARIABLES[v]
        }
        da = to_wm2(v, fields)
        check_same_grid(da, landfrac, f"{v}/ERA5")
        # Exact landfrac coords, so `.where(native_mask)` cannot misalign on round-off
        da = da.assign_coords(lat=landfrac.lat, lon=landfrac.lon)
        with xr.set_options(keep_attrs=True):
            monthly[v] = da.where(native_mask)
    return annual_means(monthly), native_mask, bounded_src_grid(landfrac)


def load_annual(
    variables: Sequence[str],
    time_slice: slice,
    res: float | str,
    *,
    mask: xr.DataArray | None = None,
    na_thresh: float = NA_THRESH,
) -> dict[str, xr.DataArray]:
    """
    Annual means {variable: (year, lat, lon)} [W/m2] of `variables` on
    `target_grid(res)` within LAT_BNDS, masked with `mask` if given:
    `etunc.grid.regrid_annual` of `load_native`.
    """
    ann = regrid_annual(*load_native(variables, time_slice), res, mask=mask, na_thresh=na_thresh, label="ERA5")
    for v, da in ann.items():
        print_annual(da, f"{v:3} {'ERA5':12}")
    return ann
