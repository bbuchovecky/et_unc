"""
etunc.legacy.grid
=================
Unmaintained and untested. Moved from `etunc.grid` when the binning pipeline
was restructured (docs/binning-workflow.md): `regrid_annual`, the second stage
of the former `load_annual` of every load module. The prep scripts now write
out its steps (land mask, conservative regridding, `on_grid`, LAT_BNDS).
Also `approx_resolution` and `coord_name`, used by the deprecated writers of
regridded files (scripts/legacy_regrid/).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from etunc.config import LAT_BNDS, NA_THRESH
from etunc.grid import bounded_conservative_regridder, check_same_grid, on_grid, regrid_with_na_thresh


def regrid_annual(
    ann: dict[str, xr.DataArray],
    native_mask: xr.DataArray,
    src_grid: xr.Dataset | xr.DataArray,
    res: float | str,
    *,
    mask: xr.DataArray | None = None,
    na_thresh: float = NA_THRESH,
    label: str = "",
) -> dict[str, xr.DataArray]:
    """
    Second stage of every `load_annual`: native annual means `ann` ({key:
    ([member,] year, lat, lon)}, from a `load_native`) masked with the native
    land mask `native_mask`, conservatively regridded onto `target_grid(res)`
    within LAT_BNDS, and masked with the target `mask` if given. Each target
    cell is the area-weighted mean of its valid (land) source cells, and NaN
    where more than `na_thresh` of its area is missing. A `member_id` coord on
    `member` is kept.
    """
    regridder = bounded_conservative_regridder(src_grid, res)
    out = {}
    for k, a in ann.items():
        name = f"{label}/{k}" if label else k
        check_same_grid(a, native_mask, name)
        with xr.set_options(keep_attrs=True):
            # Mask snapped to the coords of `a`, so `.where` cannot misalign on round-off. A no-op
            # where `load_native` masked already; here it guarantees that only land is averaged
            a = a.where(native_mask.assign_coords(lat=a.lat, lon=a.lon))
        member_id = a["member_id"].values if "member_id" in a.coords else None
        a = regrid_with_na_thresh(a.reset_coords(drop=True), regridder, na_thresh)
        # Exact target coords, so datasets line up under the exact-join alignment of the binning
        a = on_grid(a, res, name).sel(lat=LAT_BNDS)
        if member_id is not None:
            a = a.assign_coords(member_id=("member", member_id))
        if mask is not None:
            with xr.set_options(keep_attrs=True):
                a = a.where(mask)
        out[k] = a.rename(k)
    return out


def coord_name(ds: xr.DataArray | xr.Dataset, candidates: list[str], standard_names: list[str]) -> str | None:
    """First of `candidates` among the coords of `ds`, else the first coord with one of `standard_names`."""
    for name in candidates:
        if name in ds.coords:
            return name
    for name, var in ds.coords.items():
        if str(var.attrs.get("standard_name", "")) in standard_names:
            return name
    return None


def approx_resolution(ds: xr.DataArray | xr.Dataset) -> tuple[float | None, float | None]:
    """Median (lat, lon) spacing [deg] of the unique coordinate values; None if it cannot be found."""
    lat_name = coord_name(ds, ["lat", "latitude", "nav_lat"], ["latitude"])
    lon_name = coord_name(ds, ["lon", "longitude", "nav_lon"], ["longitude"])

    def spacing(name: str | None) -> float | None:
        if name is None or name not in ds.coords:
            return None
        arr = np.asarray(ds[name].values)
        if arr.ndim == 0:
            return None
        vals = np.unique(arr[np.isfinite(arr)])
        if vals.size < 2:
            return None
        diffs = np.diff(np.sort(vals))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return None
        return float(np.nanmedian(diffs))

    return spacing(lat_name), spacing(lon_name)
