"""
etunc.legacy.load_cesm
======================
Unmaintained and untested. The annual loaders of `etunc.load.cesm` before the
binning pipeline was restructured (docs/binning-workflow.md): `load_native`
(LENS2 or FHIST PPE annual means of ET, LAI, pr and Rn on the native grid,
masked with LANDFRAC > LF_THRESH) and `load_annual` (`regrid_annual` of it).
They take `variables` as {key: (CESM variable, gcomp)}. scripts/prep_lens.py
and prep_fppe.py now write out these steps.
"""

from __future__ import annotations

from typing import Literal, Mapping, Sequence

import xarray as xr

from etunc.config import LF_THRESH, NA_THRESH
from etunc.grid import check_same_grid, landfrac_mask
from etunc.legacy.grid import regrid_annual
from etunc.legacy.temporal import annual_means, print_annual
from etunc.load.cesm import load_land_fraction
from etunc.load.cesm import load_native as load_monthly
from etunc.units import convert_units, net_radiation_cesm


def load_native(
    source: Literal["lens", "fppe"],
    members: Sequence[str] | Sequence[int] | None,
    time_slice: slice,
    *,
    variables: Mapping[str, tuple[str, str]],
    frequency: str = "month_1",
    stream: str = "h0",
    bb: str = "cmip6",
    drop_outliers: Sequence[int] = (13, 28),
) -> tuple[dict[str, xr.DataArray], xr.DataArray, xr.Dataset]:
    """
    Annual means (member, year, lat, lon) of ET, LAI, pr and Rn [W/m2, m2/m2]
    of a CESM ensemble on its native grid, masked with the grid file's land
    fraction (LANDFRAC > LF_THRESH, no Greenland/Iceland), with `member_id`
    on `member`. Complete years only; a missing month counts as LAI = 0, while
    ET, pr and Rn need all 12 months (`annual_means`).

    source : "lens" (LENS2; `members` are ens_names, `bb` its biomass burning
        forcing) or "fppe" (FHIST PPE; `members` are integer ids, None = all
        but `drop_outliers`).
    variables : {key: (CESM variable, gcomp)} for the keys "et" (EFLX_LH_TOT,
        in W/m2), "lai", "pr" (PRECT or PRECT_calculated, m/s), "fsns" and "flns".

    Returns (annual means, native land mask, native grid with cell edges), the
    input of `etunc.grid.regrid_annual`.
    """
    landfrac, src_grid = load_land_fraction(source)
    # Sets the area the ensemble contributes: native cells with LANDFRAC > LF_THRESH, minus
    # Greenland/Iceland. Masking on the native grid keeps ocean and ice-sheet values
    # out of the area-weighted means of the regridding
    native_mask = landfrac_mask(landfrac, LF_THRESH)

    raw = load_monthly(
        source, members, time_slice, variables={v: gcomp for v, gcomp in variables.values()},
        frequency=frequency, stream=stream, bb=bb, drop_outliers=drop_outliers,
    )
    fields = {}
    for k, (v, _) in variables.items():
        check_same_grid(raw[v], landfrac, f"{source}/{k}")
        # Exact grid-file coords, so `.where(native_mask)` cannot misalign on round-off
        fields[k] = raw[v].assign_coords(lat=landfrac.lat, lon=landfrac.lon)

    monthly = {
        # EFLX_LH_TOT is the total latent heat flux, already in W/m2
        "et": fields["et"].assign_attrs(units="W/m2"),
        "lai": fields["lai"].assign_attrs(units="m2/m2"),
        # Keyed on the variable name: PRECT(_calculated) m/s -> W/m2
        "pr": convert_units(variables["pr"][0], fields["pr"]),
        "rn": net_radiation_cesm(fields["fsns"], fields["flns"]),
    }
    with xr.set_options(keep_attrs=True):
        monthly = {k: da.where(native_mask) for k, da in monthly.items()}
    return annual_means(monthly), native_mask, src_grid


def load_annual(
    source: Literal["lens", "fppe"],
    members: Sequence[str] | Sequence[int] | None,
    time_slice: slice,
    res: float | str,
    *,
    variables: Mapping[str, tuple[str, str]],
    frequency: str = "month_1",
    stream: str = "h0",
    bb: str = "cmip6",
    drop_outliers: Sequence[int] = (13, 28),
    mask: xr.DataArray | None = None,
    na_thresh: float = NA_THRESH,
) -> dict[str, xr.DataArray]:
    """
    Annual means (member, year, lat, lon) of ET, LAI, pr and Rn [W/m2, m2/m2]
    of a CESM ensemble on `target_grid(res)` within LAT_BNDS, masked with
    `mask` if given: `etunc.grid.regrid_annual` of `load_native`.
    """
    native = load_native(
        source, members, time_slice, variables=variables, frequency=frequency, stream=stream, bb=bb,
        drop_outliers=drop_outliers,
    )
    ann = regrid_annual(*native, res, mask=mask, na_thresh=na_thresh, label=source)
    for k, da in ann.items():
        print_annual(da, f"{k:3} {source}")
    return ann
