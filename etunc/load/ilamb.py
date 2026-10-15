"""
etunc.load.ilamb
================
Load the ILAMB observational products in PRODUCTS (ILAMB_DATA_ROOT), with
lon in [-180, 180] and lat ascending.

- `load_native`: the monthly field of one product on its native grid, in
  native units (step 1 of the binning pipeline, scripts/prep_obs.py).
- `load_ilamb`: monthly fields over a time slice within LAT_BNDS, each month
  conservatively regridded (or native with `res=None`), in native units or
  (`wm2=True`) converted to W/m2 first (make_mask.py, obs_et_availability.py).

Both set undecoded fill values (|value| >= FILL_THRESH) to NaN.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from etunc.config import ILAMB_ROOT, LAT_BNDS, NA_THRESH
from etunc.grid import conservative_regridder, format_grid, on_grid, regrid_with_na_thresh
from etunc.units import flux_to_wm2


ILAMB_DATA_ROOT = ILAMB_ROOT / "ILAMB-Data"

FILL_THRESH = 1e30  # GPCCv2018 stores undecoded ~9.97e36 fill values

# {variable: {product: (file relative to ILAMB_DATA_ROOT, variable name in file)}}
# ET products come from both evspsbl and hfls. Not used: FLUXNET2015 and
# WRMC.BSRN (site data), CARDAMOM (4x5 deg), FLUXCOM le.nc (hfls.nc used).
PRODUCTS = {
    "et": {
        "CLASS":      ("hfls/CLASS/hfls.nc", "hfls"),
        "DOLCE":      ("evspsbl/DOLCE/DOLCE.nc", "hfls"),
        "FLUXCOM":    ("hfls/FLUXCOM/hfls.nc", "hfls"),
        "GLEAMv3.3a": ("evspsbl/GLEAMv3.3a/et.nc", "et"),
        "MOD16A2":    ("evspsbl/MOD16A2/et.nc", "et"),
        "MODIS":      ("evspsbl/MODIS/et_0.5x0.5.nc", "et"),
        "WECANN":     ("hfls/WECANN/hfls.nc", "hfls"),
    },
    "lai": {
        "AVH15C1":     ("lai/AVH15C1/lai.nc", "lai"),
        "AVHRR":       ("lai/AVHRR/lai_0.5x0.5.nc", "lai"),
        "GIMMS_LAI4g": ("lai/GIMMS_LAI4g/cao2023_lai.nc", "lai"),
        "MODIS":       ("lai/MODIS/lai_0.5x0.5.nc", "lai"),
    },
    "pr": {
        "CLASS":     ("pr/CLASS/pr.nc", "pr"),
        "CMAPv1904": ("pr/CMAPv1904/pr.nc", "pr"),
        "GPCCv2018": ("pr/GPCCv2018/pr.nc", "pr"),
        "GPCPv2.3":  ("pr/GPCPv2.3/pr.nc", "pr"),
    },
    "rn": {
        "CERESed4.2": ("rns/CERESed4.2/rns.nc", "rns"),
        "CLASS":      ("rns/CLASS/rns.nc", "rns"),
        "GEWEX.SRB":  ("rns/GEWEX.SRB/rns_0.5x0.5.nc", "rns"),
    },
}


def _open(variable: str, product: str) -> xr.DataArray:
    """The product's field, with lon in [-180, 180] and lat ascending (not loaded)."""
    relpath, name = PRODUCTS[variable][product]
    ds = xr.open_dataset(ILAMB_DATA_ROOT / relpath)
    return format_grid(ds[name])


def load_native(variable: str, product: str, time_slice: slice) -> xr.DataArray:
    """
    Monthly field (time, lat, lon) of one product over `time_slice` on its
    native grid, in its native units (step 1 of docs/binning-workflow.md),
    loaded, with undecoded fill values as NaN. Raises FileNotFoundError if the
    product has no data in `time_slice`.
    """
    da = _open(variable, product).sel(time=time_slice).load()
    if da.sizes["time"] == 0:
        raise FileNotFoundError(f"{variable}/{product}: no data in {time_slice}")
    # Fill values must be NaN before averaging, or a single ~1e37 month swamps the year
    with xr.set_options(keep_attrs=True):
        return da.where(np.abs(da) < FILL_THRESH)


def load_ilamb(
    variable: str,
    product: str,
    time_slice: slice,
    res: float | str | None = 0.5,
    *,
    wm2: bool = False,
    na_thresh: float = NA_THRESH,
) -> xr.DataArray:
    """
    Monthly field (time, lat, lon) of one product over `time_slice` within
    LAT_BNDS, in native units or, with `wm2`, converted to W/m2 (`flux_to_wm2`)
    before regridding. Partial years are kept. With `res`, each month is
    conservatively regridded onto `target_grid(res)` (`regrid_with_na_thresh`:
    NaN where more than `na_thresh` of a target cell is missing); with
    `res=None`, it stays on the native grid. Raises FileNotFoundError if the
    product has no data in `time_slice`.
    """
    label = f"{variable}/{product}"
    da = load_native(variable, product, time_slice)
    if wm2:
        da = flux_to_wm2(da)
    if res is None:
        return da.sel(lat=LAT_BNDS)
    da = regrid_with_na_thresh(da, conservative_regridder(da, res), na_thresh)
    return on_grid(da, res, label).sel(lat=LAT_BNDS)
