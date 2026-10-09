"""
etunc.load.ilamb
================
Load the ILAMB observational products in PRODUCTS (ILAMB_DATA_ROOT) onto a
common target grid (`etunc.grid.target_grid`), lon in [-180, 180] and lat
ascending, within LAT_BNDS.

- `load_ilamb_annual`: annual means over complete years, taken *before*
  products on another grid are bilinearly interpolated (as for binning). LAI
  is m2/m2 and a missing LAI month counts as 0; ET, pr and rns are converted
  to W/m2 and need all 12 months.
- `load_ilamb`: monthly fields over a time slice, each month interpolated,
  in native units or (`wm2=True`) converted to W/m2 first.

Both set undecoded fill values (|value| >= FILL_THRESH) to NaN.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from etunc.config import ILAMB_ROOT
from etunc.grid import bilinear_regridder, format_grid, on_grid, target_grid
from etunc.temporal import annual_mean, complete_years, period_str
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
    "rns": {
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


def _off_grid(da: xr.DataArray, res: float | str) -> bool:
    grid = target_grid(res)
    return da.sizes["lat"] != grid.sizes["lat"] or da.sizes["lon"] != grid.sizes["lon"]


def load_ilamb_annual(variable: str, product: str, res: float | str = 0.5) -> xr.DataArray:
    """Annual means (year, lat, lon) of one product on `target_grid(res)` within LAT_BNDS."""
    da = _open(variable, product)
    da = da.sel(time=da.time.dt.year.isin(complete_years(da))).load()
    with xr.set_options(keep_attrs=True):
        da = da.where(np.abs(da) < FILL_THRESH)

    if variable == "lai":
        da.attrs["units"] = "m2/m2"
    else:
        da = flux_to_wm2(da)
    ann = annual_mean(da, require_all_months=(variable != "lai"))

    if _off_grid(ann, res):
        print(f"{variable}/{product}: regridding {ann.sizes['lat']}x{ann.sizes['lon']} -> {res} deg")
        # Bilinear; target points outside the source grid are NaN
        ann = bilinear_regridder(ann, res)(ann, keep_attrs=True)
    # Exact coordinate values so that fields from different products align
    ann = on_grid(ann, res, f"{variable}/{product}").rename(variable)

    print(
        f"{variable:3} {product:12}: {ann.dims} {ann.shape} {period_str(ann.year.values)} "
        f"[{float(ann.min()):0.3g}, {float(ann.max()):0.3g}] {ann.attrs['units']}"
    )
    return ann


def load_ilamb(
    variable: str,
    product: str,
    time_slice: slice,
    res: float | str = 0.5,
    *,
    wm2: bool = False,
) -> xr.DataArray:
    """
    Monthly field (time, lat, lon) of one product over `time_slice` on
    `target_grid(res)` within LAT_BNDS, in native units or, with `wm2`,
    converted to W/m2 (`flux_to_wm2`) before regridding. Partial years are
    kept. Products on another grid are bilinearly interpolated month by month,
    so a target cell next to a NaN source cell is NaN. Raises
    FileNotFoundError if the product has no data in `time_slice`.
    """
    label = f"{variable}/{product}"
    da = _open(variable, product).sel(time=time_slice).load()
    if da.sizes["time"] == 0:
        raise FileNotFoundError(f"{label}: no data in {time_slice}")
    # Undecoded fill values (e.g. GPCCv2018) count as missing
    with xr.set_options(keep_attrs=True):
        da = da.where(np.abs(da) < FILL_THRESH)
    if wm2:
        da = flux_to_wm2(da)
    if _off_grid(da, res):
        print(f"{label}: regridding {da.sizes['lat']}x{da.sizes['lon']} -> {res}")
        da = bilinear_regridder(da, res)(da, keep_attrs=True)
    return on_grid(da, res, label)
