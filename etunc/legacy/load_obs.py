"""
etunc.legacy.load_obs
=====================
Unmaintained and untested. The annual loaders of `etunc.load.obs` before the
binning pipeline was restructured (docs/binning-workflow.md): `load_native`
(annual means in W/m2 on the native grid, masked to Natural Earth land, read
one year at a time) and `load_annual` (`regrid_annual` of it).
scripts/prep_obs.py now writes out these steps.
"""

from __future__ import annotations

import gc

import xarray as xr

from etunc.config import NA_THRESH
from etunc.grid import bounded_src_grid, landne_mask
from etunc.legacy.grid import regrid_annual
from etunc.legacy.temporal import annual_means, print_annual
from etunc.load.obs import ObsDataset, _has_year, _resolve, _year_bounds, list_years, load_obs
from etunc.temporal import complete_years, yearly_to_annual
from etunc.units import accumulation_to_flux, latent_heat_to_wm2


def load_native(
    dataset: str | ObsDataset,
    var: str,
    time_slice: slice = slice(None, None),
    *,
    version: str | None = None,
    freq: str = "monthly",
) -> tuple[dict[str, xr.DataArray], xr.DataArray, xr.Dataset]:
    """
    Annual means {var: (year, lat, lon)} [W/m2] of a water flux (mm per day,
    month or year, e.g. ET or E) on the native grid over the complete years
    in `time_slice`, masked to land (`etunc.grid.landne_mask` of the native
    grid). Monthly products need all 12 months for a valid annual mean; for
    yearly products the yearly total is the annual mean. Read one year at a
    time, so only one native year is in memory.

    Returns (annual means, native land mask, native grid), the input of
    `etunc.grid.regrid_annual`.
    """
    spec, version = _resolve(dataset, version, freq)
    if freq not in ("monthly", "yearly"):
        raise ValueError(f"{spec.name}: annual means need monthly or yearly files, not {freq!r}")
    if _has_year(spec):
        y0, y1 = _year_bounds(time_slice)
        years = [
            y for y in list_years(spec, var, version, freq) if (y0 is None or y >= y0) and (y1 is None or y <= y1)
        ]
        reads = [slice(str(y), str(y)) for y in years]
    else:
        reads = [time_slice]  # one file holds every year
    if not reads:
        raise FileNotFoundError(f"No {spec.name} {version} {freq} files for {var!r} in {time_slice}")

    pieces, native_mask = [], None
    for read in reads:
        da = load_obs(spec, var, read, version=version, freq=freq).sel(time=time_slice)
        if da.sizes["time"] == 0 or (freq == "monthly" and not complete_years(da)):
            continue  # nothing, or a partial year, in `time_slice`
        # Totals (mm/month, mm/year) -> W/m2, using each time step's month or year length
        da = latent_heat_to_wm2(accumulation_to_flux(da.load()))
        ann = annual_means({var: da})[var] if freq == "monthly" else yearly_to_annual(da)
        if native_mask is None:
            # Natural Earth land, as the target-grid mask of the binning
            native_mask = landne_mask(ann)
        with xr.set_options(keep_attrs=True):
            pieces.append(ann.where(native_mask))
        del da
        gc.collect()
    if not pieces:
        raise ValueError(f"{spec.name} {version} {freq} {var}: no complete year in {time_slice}")
    ann = xr.concat(pieces, dim="year").rename(var)
    return {var: ann}, native_mask, bounded_src_grid(ann)


def load_annual(
    dataset: str | ObsDataset,
    var: str,
    time_slice: slice,
    res: float | str,
    *,
    version: str | None = None,
    freq: str = "monthly",
    mask: xr.DataArray | None = None,
    na_thresh: float = NA_THRESH,
) -> dict[str, xr.DataArray]:
    """
    Annual means {var: (year, lat, lon)} [W/m2] of a water flux on
    `target_grid(res)` within LAT_BNDS, masked with `mask` if given:
    `etunc.grid.regrid_annual` of `load_native`.
    """
    spec, version = _resolve(dataset, version, freq)
    label = f"{spec.name} {version}"
    native = load_native(spec, var, time_slice, version=version, freq=freq)
    ann = regrid_annual(*native, res, mask=mask, na_thresh=na_thresh, label=label)
    print_annual(ann[var], f"{var:3} {label}")
    return ann


