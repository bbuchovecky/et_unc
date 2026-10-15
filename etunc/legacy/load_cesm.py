"""
etunc.legacy.load_cesm
======================
Unmaintained and untested. The annual loaders of `etunc.load.cesm` before the
binning pipeline was restructured (docs/binning-workflow.md): `load_native`
(LENS2 or FHIST PPE annual means of ET, LAI, pr and Rn on the native grid,
masked with LANDFRAC > LF_THRESH) and `load_annual` (`regrid_annual` of it).
They take `variables` as {key: (CESM variable, gcomp)}. scripts/prep_lens.py
and prep_fppe.py now write out these steps. Also the GOGA2 loader
(`load_goga2`) and `load_cesm_grid`, used only by notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping, Sequence

import xarray as xr

from etunc.config import LAT_BNDS, LF_THRESH, NA_THRESH
from etunc.grid import check_same_grid, landfrac_mask
from etunc.legacy.grid import regrid_annual
from etunc.legacy.temporal import annual_means, print_annual
from etunc.load.cesm import (
    CESM2_COMPONENT_MAP, VARIABLES_TO_DROP, _as_list, _drop_cosp, load_grid, load_land_fraction, shift_time,
)
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


GOGA2_ROOT = Path("/glade/campaign/collections/rda/data/d651010/global/CESM2.1_GOGA_ERSSTv5")


def load_goga2(
    variable: str | Sequence[str],
    gcomp: str,
    frequency: str,
    stream: str,
    experiment: Literal["historical", "ssp370"] = "historical",
    member: str = "",
) -> xr.Dataset:
    """
    Load variables from the GOGA2 10-member ensemble.

    Parameters
    ----------
    variable : str or list of str
        CESM variable name(s).
    gcomp : str
        Model component the variable belongs to: "atm" or "lnd".
    frequency : str
        Output frequency, e.g. "month_1".
    stream : str
        History tape stream, e.g. "h0".
    experiment : str, optional
        "historical" or "ssp370".
    member : str, optional
        Only load a single member (e.g. "01"); the result then has no
        ``member`` dimension.

    Returns
    -------
    xr.Dataset
        Dataset with a ``member`` dimension (1-10), without COSP variables.
    """
    variables = _as_list(variable)
    scomp = CESM2_COMPONENT_MAP[gcomp]
    exp_tag = {"historical": "FHIST_BGC", "ssp370": "FSSP370_BGC"}[experiment]

    def _open_member(m: int) -> list[xr.Dataset]:
        datasets = []
        for var_name in variables:
            tsdir = GOGA2_ROOT / f"{gcomp}/proc/tseries/{frequency}/{var_name}"
            pattern = (
                f"f.e21.{exp_tag}.f09_f09.{experiment}.ersstv5.goga.ens{m:02d}."
                f"{scomp}.{stream}.{var_name}.*.nc"
            )
            files = sorted(tsdir.glob(pattern))
            if files:
                datasets.append(
                    xr.open_mfdataset(
                        files,
                        decode_timedelta=False,
                        drop_variables=VARIABLES_TO_DROP[gcomp],
                        coords="minimal",
                    )
                )
        return datasets

    with xr.set_options(use_new_combine_kwarg_defaults=True):
        if isinstance(member, str) and (len(member) > 0):
            datasets = _open_member(int(member))
            if not datasets:
                return xr.Dataset()
            combined_ds = xr.merge(datasets)
        else:
            member_datasets = []
            for m in range(1, 11):
                datasets = _open_member(m)
                if datasets:
                    member_datasets.append(xr.merge(datasets))
            if not member_datasets:
                return xr.Dataset()
            combined_ds = xr.concat(member_datasets, dim="member", coords="minimal")
            # Numbered 1..n by position: a missing member shifts the labels of the later ones
            combined_ds = combined_ds.assign_coords(member=range(1, len(member_datasets) + 1))

    return shift_time(_drop_cosp(combined_ds))


def load_cesm_grid(source: Literal["fppe", "goga", "lens"], lat_bnds: slice = LAT_BNDS) -> xr.Dataset:
    """Grid dataset (LANDFRAC, LANDAREA, ...) of a CESM ensemble."""
    return load_grid(source).sel(lat=lat_bnds)
