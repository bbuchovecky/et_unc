"""
etunc.legacy.load_cmip
======================
Unmaintained and untested. The annual loaders of `etunc.load.cmip` before the
binning pipeline was restructured (docs/binning-workflow.md): `load_native`
(annual means on the native grid, masked with sftlf > LF_THRESH) and
`load_annual` (`regrid_annual` of it). scripts/prep_cmip.py now writes out
these steps.
"""

from __future__ import annotations

from typing import Sequence

import xarray as xr

from etunc.config import LF_THRESH, NA_THRESH
from etunc.grid import check_same_grid, landfrac_mask
from etunc.legacy.grid import regrid_annual
from etunc.legacy.temporal import annual_means, print_annual
from etunc.load.cmip import CMIPESGFLoader, load_land_fraction
from etunc.load.cmip import load_native as load_monthly
from etunc.units import convert_units, net_radiation_cmip


def load_native(
    loader: CMIPESGFLoader,
    sid: str,
    members: list[str],
    sftlf_path: str,
    time_slice: slice,
    *,
    variables: Sequence[str],
    experiment_id: str,
) -> tuple[dict[str, xr.DataArray], xr.DataArray, xr.Dataset]:
    """
    Annual means (member, year, lat, lon) of ET, LAI, pr and Rn [W/m2, m2/m2]
    for one model on its native grid, masked with its land fraction (sftlf >
    LF_THRESH, no Greenland/Iceland), with `member_id` on `member`.
    `variables` must include evspsbl, lai, pr, rsds, rsus, rlds and rlus.
    Complete years only; a missing month counts as LAI = 0, while ET, pr and
    Rn need all 12 months (`annual_means`).

    Returns (annual means, native land mask, native grid with cell edges), the
    input of `etunc.grid.regrid_annual`.
    """
    lf, src_grid = load_land_fraction(sftlf_path)
    # Sets the area each model contributes: native cells with sftlf > LF_THRESH, minus
    # Greenland/Iceland. Masking on the native grid keeps ocean and ice-sheet values
    # out of the area-weighted means of the regridding
    native_mask = landfrac_mask(lf, LF_THRESH)

    data = load_monthly(loader, sid, members, time_slice, variables=variables, experiment_id=experiment_id)
    fields = {}
    for v in variables:
        da = data[v]
        # sftlf comes from another experiment, so check that it is on the same grid
        check_same_grid(da, lf, f"{sid}/{v}")
        # Exact sftlf coords, so `.where(native_mask)` cannot misalign on round-off
        fields[v] = da.assign_coords(lat=lf.lat, lon=lf.lon)

    monthly = {
        # ET is evspsbl (evaporation including sublimation and transpiration), x L to W/m2
        "et": convert_units("evspsbl", fields["evspsbl"]),
        "lai": fields["lai"].assign_attrs(units="m2/m2"),
        "pr": convert_units("pr", fields["pr"]),
        "rn": net_radiation_cmip(fields["rsds"], fields["rsus"], fields["rlds"], fields["rlus"]),
    }
    with xr.set_options(keep_attrs=True):
        monthly = {k: da.where(native_mask) for k, da in monthly.items()}
    return annual_means(monthly), native_mask, src_grid


def load_annual(
    loader: CMIPESGFLoader,
    sid: str,
    members: list[str],
    sftlf_path: str,
    time_slice: slice,
    res: float | str,
    *,
    variables: Sequence[str],
    experiment_id: str,
    mask: xr.DataArray | None = None,
    na_thresh: float = NA_THRESH,
) -> dict[str, xr.DataArray]:
    """
    Annual means (member, year, lat, lon) of ET, LAI, pr and Rn [W/m2, m2/m2]
    for one model on `target_grid(res)` within LAT_BNDS, masked with `mask` if
    given: `etunc.grid.regrid_annual` of `load_native`.
    """
    ann = regrid_annual(
        *load_native(loader, sid, members, sftlf_path, time_slice, variables=variables, experiment_id=experiment_id),
        res, mask=mask, na_thresh=na_thresh, label=sid,
    )
    for k, da in ann.items():
        print_annual(da, f"{k:3} {sid:16}")
    return ann


