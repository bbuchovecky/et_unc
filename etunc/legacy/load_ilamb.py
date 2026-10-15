"""
etunc.legacy.load_ilamb
=======================
Unmaintained and untested. The annual loaders of `etunc.load.ilamb` before
the binning pipeline was restructured (docs/binning-workflow.md):
`load_native` (annual means on the native grid, masked to Natural Earth land)
and `load_annual` (`regrid_annual` of it). scripts/prep_obs.py now writes out
these steps.
"""

from __future__ import annotations

import xarray as xr

from etunc.config import NA_THRESH
from etunc.grid import bounded_src_grid, landne_mask
from etunc.legacy.grid import regrid_annual
from etunc.legacy.temporal import annual_means, print_annual
from etunc.load.ilamb import load_native as load_monthly
from etunc.units import flux_to_wm2


def load_native(
    variable: str, product: str, time_slice: slice = slice(None, None),
) -> tuple[dict[str, xr.DataArray], xr.DataArray, xr.Dataset]:
    """
    Annual means {variable: (year, lat, lon)} of one product on its native
    grid over the complete years in `time_slice`, masked to land
    (`etunc.grid.landne_mask` of the native grid). ET, pr and rn are in W/m2
    and need all 12 months; LAI is m2/m2 and a missing month counts as 0.

    Returns (annual means, native land mask, native grid), the input of
    `etunc.grid.regrid_annual`.
    """
    da = load_monthly(variable, product, time_slice)
    if variable == "lai":
        da.attrs["units"] = "m2/m2"
    else:
        da = flux_to_wm2(da)
    # Natural Earth land, as the target-grid mask of the binning. Removes the ocean of the
    # global pr and rn products (GPCP, CMAP, CERES, GEWEX) before it can enter coastal cells
    native_mask = landne_mask(da)
    with xr.set_options(keep_attrs=True):
        da = da.where(native_mask)
    return annual_means({variable: da}), native_mask, bounded_src_grid(da)


def load_annual(
    variable: str,
    product: str,
    time_slice: slice,
    res: float | str,
    *,
    mask: xr.DataArray | None = None,
    na_thresh: float = NA_THRESH,
) -> dict[str, xr.DataArray]:
    """
    Annual means {variable: (year, lat, lon)} of one product on
    `target_grid(res)` within LAT_BNDS, masked with `mask` if given:
    `etunc.grid.regrid_annual` of `load_native`.
    """
    label = f"{variable}/{product}"
    ann = regrid_annual(*load_native(variable, product, time_slice), res, mask=mask, na_thresh=na_thresh, label=label)
    print_annual(ann[variable], f"{variable:3} {product:12}")
    return ann


