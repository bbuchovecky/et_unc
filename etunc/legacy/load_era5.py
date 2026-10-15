"""
etunc.legacy.load_era5
======================
Unmaintained and untested. The annual loaders of `etunc.load.era5` before the
binning pipeline was restructured (docs/binning-workflow.md): `load_native`
(annual means of et, pr and rn in W/m2 on the native grid, masked with
LANDFRAC > LF_THRESH) and `load_annual` (`regrid_annual` of it).
scripts/prep_obs.py now writes out these steps. Also `available_years`
(years on GDEX), which the former ERA5 year span of bin_obs.py used.
"""

from __future__ import annotations

from typing import Sequence

import xarray as xr

from etunc.config import LF_THRESH, NA_THRESH
from etunc.grid import check_same_grid, format_grid, landfrac_mask
from etunc.legacy.grid import regrid_annual
from etunc.legacy.temporal import annual_means, print_annual
from etunc.load.era5 import GDEX_ROOTS, _root, load_era5, load_land_fraction
from etunc.units import convert_units, net_radiation_era5


# Mean-rate parameter codes ("meanflux" stream) of each binning variable; rn is the
# sum of the net SW and LW fluxes (`to_wm2`)
VARIABLES = {
    "et": ("mer",),
    "pr": ("mtpr",),
    "rn": ("msnswrf", "msnlwrf"),
}


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
    landfrac, src_grid = load_land_fraction()
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
    return annual_means(monthly), native_mask, src_grid


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


def available_years(kind: str = "meanflux") -> list[int]:
    """Sorted years with a directory under the GDEX root that `load_era5` reads for `kind`."""
    roots = GDEX_ROOTS[kind] if isinstance(GDEX_ROOTS[kind], tuple) else (GDEX_ROOTS[kind],)
    years = {int(d.name) for r in roots if r.is_dir() for d in r.iterdir() if d.name.isdigit()}
    # A year counts only under the root that `_root` assigns it to
    return sorted(y for y in years if (_root(kind, y) / str(y)).is_dir())
