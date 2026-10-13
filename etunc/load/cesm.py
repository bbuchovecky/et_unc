"""
etunc.load.cesm
===============
Load CESM2 ensemble output from the CESM time-series (tseries) archives on glade:
the FHIST perturbed-parameter ensemble ("fppe"), the GOGA2 10-member ensemble
("goga") and the CESM2 Large Ensemble ("lens"). Also holds each ensemble's grid
file (LANDFRAC, LANDAREA, ...) and the FHIST PPE member names.

Every ensemble loader (`load_fhist_ppe`, `load_goga2`, `load_cesm2le`)
returns a lazy (dask) Dataset of native monthly fields with a ``member``
dimension. Its time axis is shifted from CESM's end-of-interval stamps to the
month the data averages (see ``shift_time``).

Annual means for binning, as in every load module: `load_native` (LENS2 or
FHIST PPE annual means of ET, LAI, pr and Rn on the native grid, LANDFRAC
mask) and `load_annual` = `etunc.grid.regrid_annual` of it.

Example
-------
>>> import etunc.load.cesm as lc
>>> grid = lc.load_grid("fppe")
>>> ds = lc.load_fhist_ppe("EFLX_LH_TOT", "lnd", "month_1")
>>> ds = lc.load_goga2("EFLX_LH_TOT", "lnd", "month_1", "h0")
>>> ds = lc.load_cesm2le("EFLX_LH_TOT", "lnd", "month_1", "h0", bb="cmip6")
>>> lc.ppe_member_name(2)   # "2.dleaf.max"
>>> variables = {"et": ("EFLX_LH_TOT", "lnd"), "lai": ("TLAI", "lnd"), "pr": ("PRECT", "atm"),
...              "fsns": ("FSNS", "atm"), "flns": ("FLNS", "atm")}
>>> ann = lc.load_annual("lens", None, slice("1995-01", "2014-12"), 1, variables=variables)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import xarray as xr

from etunc.config import LAT_BNDS, LF_THRESH, NA_THRESH
from etunc.grid import bounded_source_grid, check_same_grid, mask_greenland, regrid_annual
from etunc.temporal import annual_means, print_annual
from etunc.units import convert_units, net_radiation_cesm


CESM2_COMPONENT_MAP = {
    "atm": "cam",
    "lnd": "clm2",
}

VARIABLES_TO_DROP = {
    "atm": ["gw", "hyam", "hybm", "P0", "ilev"],
    "lnd": [
        "nbedrock",
        "ZSOI",
        "DZSOI",
        "WATSAT",
        "SUCSAT",
        "BSW",
        "HKSAT",
        "ZLAKE",
        "DZLAKE",
        "time_written",
        "date_written",
    ],
}

FHIST_PPE_ROOT = Path("/glade/campaign/univ/uwas0155/ppe/historical/coupled_simulations")
FHIST_PPE_BASENAME = "f.e21.FHIST_BGC.f19_f19_mg17.historical.coupPPE"
GOGA2_ROOT = Path("/glade/campaign/collections/rda/data/d651010/global/CESM2.1_GOGA_ERSSTv5")
LENS2_ROOT = Path("/glade/campaign/collections/gdex/data/d651056/CESM2-LE")

LENS_BRANCH_YEARS = [
    "1001", "1021", "1041", "1061", "1081", "1101", "1121",
    "1141", "1161", "1181", "1231", "1251", "1281", "1301",
]

GRID_PATHS = {
    "fppe": Path("/glade/campaign/univ/uwas0155/ppe/f.e21.FHIST_BGC.f19_f19_mg17.historical.AREA_GRID.nc"),
    "goga": Path("/glade/campaign/univ/uwas0155/ppe/f.e21.FHIST_BGC.f09_f09.GOGA2.AREA_GRID.nc"),
    "lens": Path("/glade/campaign/univ/uwas0155/ppe/b.e21.BHISTsmbb.f09_g17.LE2.AREA_GRID.nc"),
}

# FHIST PPE member id of the minimum and maximum value of each perturbed parameter
FHIST_PPE_MEMBERS = {
    "default": {"default": 0},
    "dleaf": {"max": 2, "min": 1},
    "d_max": {"max": 4, "min": 3},
    "max_leaf_wet_frac": {"max": 6, "min": 5},
    "fff": {"max": 8, "min": 7},
    "medlynslope": {"max": 10, "min": 9},
    "medlynintercept": {"max": 12, "min": 11},
    "Jmaxb0": {"max": 14, "min": 13},
    "kmax": {"max": 16, "min": 15},
    "psi50": {"max": 18, "min": 17},
    "FUN_fracfixers": {"max": 20, "min": 19},
    "leafcn": {"max": 22, "min": 21},
    "lmrha": {"max": 24, "min": 23},
    "KCN": {"max": 26, "min": 25},
    "ACCLIM_SF": {"max": 28, "min": 27},
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def shift_time(ds: xr.Dataset) -> xr.Dataset:
    """Shifts time coordinate from [startyear-02, endyear-01] to [startyear-01, (endyear-1)-12]"""
    assert "time" in ds.dims
    # CESM stamps each monthly mean at the end of its interval (January at Feb 1). Unshifted,
    # time.year would group Feb-Jan and every annual mean would be off by one month.
    # Any other first/last month is returned unshifted, without a warning
    if (ds.time[0].dt.month.item() == 2) and (ds.time[-1].dt.month.item() == 1):
        new_time = xr.date_range(
            start=str(ds.time[0].dt.year.item()) + "-01",
            end=str(ds.time[-1].dt.year.item() - 1) + "-12",
            freq="MS",
            calendar="noleap",
            use_cftime=True,
        )
        return ds.assign_coords(time=new_time)
    return ds


def _as_list(variable: str | Sequence[str]) -> list[str]:
    return [variable] if isinstance(variable, str) else list(variable)


def _drop_cosp(ds: xr.Dataset) -> xr.Dataset:
    """Drop all COSP-related variables and coordinates."""
    cosp_vars = [var for var in ds.variables if "cosp" in var]
    return ds.drop_vars(cosp_vars) if cosp_vars else ds


# ------------------------------------------------------------------
# Grids
# ------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_grid(source: Literal["fppe", "goga", "lens"]) -> xr.Dataset:
    """Grid dataset (LANDFRAC, LANDAREA, ...) of a CESM ensemble."""
    return xr.open_dataset(GRID_PATHS[source])


# ------------------------------------------------------------------
# FHIST perturbed-parameter ensemble
# ------------------------------------------------------------------

def _glob_member_files(
    mcase: str,
    gcomp: str,
    scomp: str,
    frequency: str,
    stream: str,
    variables: list[str],
) -> dict[str, list[Path]]:
    """Return {variable: [Path, ...]} for all requested variables for one member."""
    result: dict[str, list[Path]] = {}
    for v in variables:
        files = sorted(
            FHIST_PPE_ROOT.glob(
                f"{mcase}/{gcomp}/proc/tseries/{frequency}/"
                f"{mcase}.{scomp}.{stream}.{v}.*.nc"
            )
        )
        if files:
            result[v] = files
    return result


def _load_member_netcdf(
    m: int,
    gcomp: str,
    scomp: str,
    frequency: str,
    stream: str,
    variables: list[str],
) -> tuple[int, xr.Dataset | None]:
    """
    Load a single PPE member from NetCDF tseries files.

    Opens all per-variable file lists in a single ``open_mfdataset`` call.
    Each tseries file holds one variable, so xarray separates them by name.

    Returns
    -------
    (member_id, dataset); dataset is None if no files were found.
    """
    mcase = f"{FHIST_PPE_BASENAME}.{str(m).zfill(3)}"
    var_files = _glob_member_files(mcase, gcomp, scomp, frequency, stream, variables)
    if not var_files:
        return m, None

    all_files = sorted({f for flist in var_files.values() for f in flist})
    ds = xr.open_mfdataset(
        all_files,
        decode_timedelta=False,
        drop_variables=VARIABLES_TO_DROP[gcomp],
        coords="minimal",
        combine="by_coords",
        join="outer",
    )
    return m, shift_time(ds)


def load_fhist_ppe(
    variable: str | Sequence[str],
    gcomp: str,
    frequency: str,
    stream: str = "h0",
    members: int | Sequence[int] | None = None,
    drop_outliers: Sequence[int] = (13, 28),
    max_workers: int = 8,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Load variables from the FHIST perturbed-parameter ensemble.

    Parameters
    ----------
    variable : str or list of str
        CESM variable name(s), e.g. "EFLX_LH_TOT" or "PRECT_calculated".
    gcomp : str
        Model component the variable belongs to: "atm" or "lnd".
    frequency : str
        Output frequency, e.g. "month_1" or "day_1".
    stream : str, optional
        History tape stream, e.g. "h0" (monthly) or "h2" (daily).
    members : int or sequence of int, optional
        Explicit member subset (0-28). Defaults to all members not in `drop_outliers`.
    drop_outliers : sequence of int, optional
        Members excluded when `members` is None. Defaults to (13, 28), which
        have unreasonable LAI and ET.
    max_workers : int, optional
        Maximum threads used to load members in parallel.
    verbose : bool, optional
        Print the members that were found.

    Returns
    -------
    xr.Dataset
        Dataset with a ``member`` dimension whose coordinate values are the
        integer member ids (squeezed out if only one member is found).
    """
    variables = _as_list(variable)
    scomp = CESM2_COMPONENT_MAP[gcomp]

    if isinstance(members, int):
        members = [members]
    drop_set = set(drop_outliers)
    iter_members = (
        [m for m in range(29) if m not in drop_set] if members is None else list(members)
    )
    if not iter_members:
        return xr.Dataset()

    with xr.set_options(use_new_combine_kwarg_defaults=True):
        member_ids: list[int] = []
        member_datasets: list[xr.Dataset] = []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(iter_members))) as pool:
            futures = [
                pool.submit(_load_member_netcdf, m, gcomp, scomp, frequency, stream, variables)
                for m in iter_members
            ]
            for fut in as_completed(futures):
                m_id, ds = fut.result()
                if ds is not None:
                    member_ids.append(m_id)
                    member_datasets.append(ds)

        if not member_datasets:
            return xr.Dataset()

        # Sort by member id to restore deterministic ordering
        order = np.argsort(member_ids)
        member_ids = [member_ids[i] for i in order]
        member_datasets = [member_datasets[i] for i in order]
        if verbose:
            print(member_ids)
            print("n =", len(member_datasets))

        combined_ds = xr.concat(member_datasets, dim="member").assign_coords(
            member=np.array(member_ids)
        )

    if len(combined_ds.member) == 1:
        combined_ds = combined_ds.squeeze(dim="member")
    return combined_ds


def _member_info(member_id) -> tuple | list[tuple]:
    """(member_id, parameter, "min"|"max") for one member id, or a list of them."""
    inverted = {
        mem_id: (mem_id, param, minmax)
        for param, minmax_dict in FHIST_PPE_MEMBERS.items()
        for minmax, mem_id in minmax_dict.items()
    }

    if isinstance(member_id, (int, float, str, np.floating, np.integer)):
        member_id = [member_id]
    elif isinstance(member_id, xr.DataArray):
        member_id = member_id.values.flatten()
    elif isinstance(member_id, np.ndarray):
        member_id = member_id.flatten()

    info = [inverted[int(m)] for m in member_id if int(m) in inverted]
    return info[0] if len(info) == 1 else info


def ppe_member_name(
    member_id: int | float | str | Sequence[int | float | str] | np.ndarray | xr.DataArray,
    no_id: bool = False,
    delimiter: str = ".",
) -> str | list[str]:
    """
    Name of an FHIST PPE member, e.g. "2.dleaf.max" ("dleaf.max" with `no_id`).

    Accepts one id or a list/array of ids. Returns a list for more than one
    id; unknown ids are skipped.
    """
    info = _member_info(member_id)

    def _name(i: tuple) -> str:
        return delimiter.join(str(x) for x in (i[1:] if no_id else i))

    if isinstance(info, list):
        return [_name(i) for i in info]
    return _name(info)


# ------------------------------------------------------------------
# GOGA2 and the CESM2 Large Ensemble
# ------------------------------------------------------------------

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


def load_cesm2le(
    variable: str | Sequence[str],
    gcomp: str,
    frequency: str,
    stream: str,
    experiment: Literal["historical", "ssp370", "both"] = "historical",
    bb: Literal["cmip6", "smbb"] = "cmip6",
    member: str = "",
    verbose: bool = True,
) -> xr.Dataset:
    """
    Load variables from the CESM2 Large Ensemble.

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
        "historical", "ssp370" or "both" (concatenated in time).
    bb : str, optional
        Biomass burning forcing: "cmip6" (default, matches the FHIST PPE) or
        smoothed ("smbb"). Each forcing covers 50 of the 100 members.
    member : str, optional
        Only load a single member (e.g. "1091.005"); the result then has no
        ``member`` dimension.
    verbose : bool, optional
        Print loading progress.

    Returns
    -------
    xr.Dataset
        Dataset with a ``member`` dimension (1-50) and an ``ens_name``
        coordinate ("<branch year>-<member>"), without COSP variables.
    """
    variables = _as_list(variable)
    scomp = CESM2_COMPONENT_MAP[gcomp]
    log = print if verbose else (lambda *a, **k: None)

    def _load_member(v: str, yr: str, mem_str: str) -> xr.Dataset:
        tsdir = LENS2_ROOT / f"{gcomp}/proc/tseries/{frequency}/{v}"
        hist = f"{tsdir}/b.e21.BHIST{bb}.f09_g17.LE2-{yr}.{mem_str}.{scomp}.{stream}.{v}.*.nc"
        ssp = f"{tsdir}/b.e21.BSSP370{bb}.f09_g17.LE2-{yr}.{mem_str}.{scomp}.{stream}.{v}.*.nc"
        if experiment == "both":
            files = sorted(glob(hist) + glob(ssp))
        elif experiment == "historical":
            files = sorted(glob(hist))
        elif experiment == "ssp370":
            files = sorted(glob(ssp))
        else:
            raise ValueError("experiment must be one of historical, ssp370, or both.")

        ds = xr.open_mfdataset(
            files,
            parallel=True,
            combine="nested",
            concat_dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            decode_timedelta=False,
            drop_variables=VARIABLES_TO_DROP[gcomp],
        )
        return shift_time(ds)

    with xr.set_options(use_new_combine_kwarg_defaults=True):
        if isinstance(member, str) and (len(member) > 0):
            yr, im = member.split(".")[:2]
            combined_ds = xr.merge([_load_member(v, yr, im) for v in variables])
        else:
            member_datasets = []
            for var_name in variables:
                members = []
                tags = []

                def _append(yr: str, mem_str: str) -> None:
                    try:
                        members.append(_load_member(var_name, yr, mem_str))
                        tags.append(yr + "-" + mem_str)
                    except OSError:  # open_mfdataset raises OSError when no file matches
                        log("no files")

                # Branch years before 1200 have one member each (macro
                # perturbations), later ones have 10 (micro perturbations).
                # LENS_BRANCH_YEARS and these member numbers are the cmip6 members;
                # the smbb members use other names (1011.001, ..., 1231.011-020)
                for im, yr in enumerate(LENS_BRANCH_YEARS):
                    log(yr)
                    if int(yr) < 1200:
                        _append(yr, str(im + 1).zfill(3))
                    else:
                        for i in range(1, 11):
                            log(i, end=" ")
                            _append(yr, str(i).zfill(3))
                        log()

                # 10 macro + 4 x 10 micro = 50 members; assumes none was skipped above
                member_datasets.append(
                    xr.concat(members, dim="member")
                    .assign_coords(member=np.arange(1, 51), ens_name=("member", tags))
                )
            combined_ds = xr.merge(member_datasets)

    return shift_time(_drop_cosp(combined_ds))


# ------------------------------------------------------------------
# Grid file
# ------------------------------------------------------------------

def load_cesm_grid(source: Literal["fppe", "goga", "lens"], lat_bnds: slice = LAT_BNDS) -> xr.Dataset:
    """Grid dataset (LANDFRAC, LANDAREA, ...) of a CESM ensemble."""
    return load_grid(source).sel(lat=lat_bnds)


# ------------------------------------------------------------------
# Annual means for binning (scripts/bin_lens.py, scripts/bin_fppe.py)
# ------------------------------------------------------------------

def _lens_fields(
    variables: Mapping[str, tuple[str, str]],
    members: Sequence[str] | None,
    time_slice: slice,
    frequency: str,
    stream: str,
    bb: str,
) -> dict[str, xr.DataArray]:
    """{key: (member, time, lat, lon)} of LENS2 `members` (ens_names; None = all), with member_id = ens_name."""
    fields, ens = {}, None
    for k, (v, gcomp) in variables.items():
        print(f"{v} ({gcomp})")
        da = load_cesm2le(v, gcomp, frequency, stream, bb=bb, verbose=False)[v].sel(time=time_slice)
        names = da["ens_name"].values.tolist()
        if ens is None:
            ens = names if members is None else list(members)
            missing = [m for m in ens if m not in names]
            if missing:
                raise ValueError(f"Members {missing} not in LENS2 ({bb}): {names}")
        # Only member_id is kept of the non-index coords
        da = da.isel(member=[names.index(m) for m in ens]).reset_coords(drop=True)
        fields[k] = da.assign_coords(member_id=("member", ens))
    return fields


def _fppe_fields(
    variables: Mapping[str, tuple[str, str]],
    members: Sequence[int] | None,
    time_slice: slice,
    frequency: str,
    stream: str,
    drop_outliers: Sequence[int],
) -> dict[str, xr.DataArray]:
    """
    {key: (member, time, lat, lon)} of the FHIST PPE `members` (None = all but
    `drop_outliers`) that have every variable, with member_id = parameter name.
    """
    # One `load_fhist_ppe` call per component
    by_gcomp: dict[str, list[str]] = {}
    for v, gcomp in variables.values():
        by_gcomp.setdefault(gcomp, []).append(v)
    ds = {}
    for gcomp, vs in by_gcomp.items():
        d = load_fhist_ppe(vs, gcomp, frequency, stream, members=members, drop_outliers=drop_outliers)
        if not d.data_vars:
            raise RuntimeError(f"No FHIST PPE files found for {vs} ({gcomp})")
        missing = [v for v in vs if v not in d]
        if missing:
            raise RuntimeError(f"Missing FHIST PPE variables {missing} ({gcomp})")
        # load_fhist_ppe squeezes out `member` when only one member is found
        ds[gcomp] = d if "member" in d.dims else d.expand_dims("member")

    # A member counts only if it has every variable of every component
    ids = sorted(set.intersection(*(set(d.member.values.tolist()) for d in ds.values())))
    for gcomp, d in ds.items():
        dropped = sorted(set(d.member.values.tolist()) - set(ids))
        if dropped:
            print(f"{gcomp}: members {dropped} lack another component's variables, skipping them")
    if not ids:
        raise RuntimeError("No FHIST PPE member has every variable.")
    names = [ppe_member_name(m) for m in ids]
    return {
        k: ds[gcomp][v].sel(member=ids, time=time_slice).reset_coords(drop=True).assign_coords(member_id=("member", names))
        for k, (v, gcomp) in variables.items()
    }


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
    grid = load_grid(source)
    # Sets the area the ensemble contributes: native cells with LANDFRAC > LF_THRESH, minus
    # Greenland/Iceland. Masking on the native grid keeps ocean and ice-sheet values
    # out of the area-weighted means of the regridding
    native_mask = mask_greenland(grid["LANDFRAC"], LF_THRESH)

    if source == "lens":
        fields = _lens_fields(variables, members, time_slice, frequency, stream, bb)
    elif source == "fppe":
        fields = _fppe_fields(variables, members, time_slice, frequency, stream, drop_outliers)
    else:
        raise ValueError(f"Unsupported CESM source {source!r}; use 'lens' or 'fppe'")
    for k, da in fields.items():
        check_same_grid(da, grid, f"{source}/{k}")
        # Exact grid-file coords, so `.where(native_mask)` cannot misalign on round-off
        fields[k] = da.assign_coords(lat=grid.lat, lon=grid.lon)

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
    return annual_means(monthly), native_mask, bounded_source_grid(grid)


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
    na_thres: float = NA_THRESH,
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
    ann = regrid_annual(*native, res, mask=mask, na_thres=na_thres, label=source)
    for k, da in ann.items():
        print_annual(da, f"{k:3} {source}")
    return ann
