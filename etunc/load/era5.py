"""
etunc.load.era5
===============
Load ERA5 monthly means from the NCAR GDEX archive (ds633.0 / ds633.5) and the
ERA5 grid file (LANDFRAC, ...).

- `load_era5`: one monthly-mean parameter on the native 0.25 deg grid.
- `load_era5_annual`: annual means of a binning variable (et, pr or rns, see
  VARIABLES) in W/m2 on `etunc.grid.target_grid(res)` within LAT_BNDS. Annual
  means are taken on the native grid over complete years, masked to land
  (LANDFRAC > LF_THRESH) and then conservatively regridded, as for CMIP6.

Example
-------
>>> import etunc.load.era5 as le
>>> grid = le.load_era5_grid()
>>> rns = le.load_era5("msnswrf", "1995-01", "2014-12", kind="meanflux")
>>> et = le.load_era5_annual("et", (2001, 2010), res=0.5)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import xarray as xr

from etunc.config import LF_THRESH, NA_THRESH
from etunc.grid import check_same_grid, conservative_regridder, format_grid, on_grid, regrid_with_na_thres
from etunc.temporal import annual_mean, complete_years, period_str
from etunc.units import convert_units, net_radiation_era5


GDEX_ROOTS = {
    # (years < SPLIT_YEAR, years >= SPLIT_YEAR)
    "sfc": (Path("/gdex/data/d633005/e5p.moda.an.sfc"), Path("/gdex/data/d633001/e5.moda.an.sfc")),
    "meanflux": Path("/gdex/data/d633001/e5.moda.fc.sfc.meanflux"),
    "accum": Path("/gdex/data/d633001/e5.moda.fc.sfc.accumu"),
}
SPLIT_YEAR = 1979

ERA5_GRID_PATH = Path("/glade/work/bbuchovecky/data/era5/e5.invariant.grid.nc")

# Mean-rate parameter codes ("meanflux" stream) of each binning variable; rns is the
# sum of the net SW and LW fluxes (`to_wm2`)
VARIABLES = {
    "et":  ("mer",),
    "pr":  ("mtpr",),
    "rns": ("msnswrf", "msnlwrf"),
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
    return format_grid(load_era5_grid()["LANDFRAC"])


def to_wm2(variable: str, fields: dict[str, xr.DataArray]) -> xr.DataArray:
    """Binning variable `variable` [W/m2] from its parameter fields {code: DataArray} (VARIABLES)."""
    if variable == "rns":
        return net_radiation_era5(fields["msnswrf"], fields["msnlwrf"])
    (code,) = VARIABLES[variable]
    # mer is positive downward, so convert_units also flips its sign to make ET positive
    return convert_units(code, fields[code])


def regrid_annual(
    ann: xr.DataArray, landfrac: xr.DataArray, res: float | str, na_thres: float = NA_THRESH, label: str = "ERA5",
) -> xr.DataArray:
    """
    Native annual means `ann` masked to land (`landfrac` > LF_THRESH) and
    conservatively regridded onto `target_grid(res)`: each target cell is the
    area-weighted mean of its land source cells, and NaN where more than
    `na_thres` of its area is not land.
    """
    check_same_grid(ann, landfrac, label)
    # Exact landfrac coords, so `.where` cannot misalign on round-off
    ann = ann.assign_coords(lat=landfrac.lat, lon=landfrac.lon)
    # Masking before regridding keeps ocean values out of coastal land cells (ERA5 is global)
    with xr.set_options(keep_attrs=True):
        ann = ann.where(landfrac > LF_THRESH)
    return regrid_with_na_thres(ann, conservative_regridder(ann, res), na_thres)


def load_era5_annual(
    variable: str, years: tuple[int, int], res: float | str = 0.5, na_thres: float = NA_THRESH,
) -> xr.DataArray:
    """
    Annual means [W/m2] (year, lat, lon) of binning variable `variable` ("et",
    "pr" or "rns") over the complete years within `years` (first, last), on
    `target_grid(res)` within LAT_BNDS (`regrid_annual`). All 12 months are
    needed for a valid annual mean.
    """
    label = f"{variable}/ERA5"
    fields = {
        code: format_grid(load_era5(code, f"{years[0]}-01", f"{years[1]}-12", kind="meanflux"))
        for code in VARIABLES[variable]
    }
    da = to_wm2(variable, fields)
    # Drop partial first/last years: they would bias annual means toward the months present
    da = da.sel(time=da.time.dt.year.isin(complete_years(da)))
    # Annual means before regridding: 12x fewer regridding steps, and no NaN to spread (ERA5 is gap-free)
    ann = annual_mean(da, require_all_months=True).load()
    ann = regrid_annual(ann, land_fraction(), res, na_thres, label)
    # Exact target coords so that ERA5 aligns with the other products
    ann = on_grid(ann, res, label).rename(variable)

    print(
        f"{variable:3} {'ERA5':12}: {ann.dims} {ann.shape} {period_str(ann.year.values)} "
        f"[{float(ann.min()):0.3g}, {float(ann.max()):0.3g}] {ann.attrs['units']}"
    )
    return ann
