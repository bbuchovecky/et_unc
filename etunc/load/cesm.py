"""
etunc.load.cesm
===============
Load CESM2 ensemble output from the CESM time-series (tseries) archives on glade:
the coupled FHIST perturbed-parameter ensemble ("fppe"), its offline CLM
counterpart ("ippe", same parameter perturbations and member ids), the GOGA2
10-member ensemble ("goga") and the CESM2 Large Ensemble ("lens"). Also holds
each ensemble's grid file (LANDFRAC, LANDAREA, ...) and the PPE member names.

Every ensemble loader (`load_ppe`, `load_cesm2le`; `load_goga2` is in
`etunc.legacy.load_cesm`)
returns a lazy (dask) Dataset of native monthly fields with a ``member``
dimension. Its time axis is shifted from CESM's end-of-interval stamps to the
month the data averages (see ``shift_time``).

For the binning pipeline (scripts/prep_lens.py, prep_fppe.py, prep_ippe.py),
`load_native` returns the LENS2, FHIST PPE or IHIST PPE monthly fields on the
native grid in native units (step 1), and `load_land_fraction` the grid file's
LANDFRAC.

Example
-------
>>> import etunc.load.cesm as lc
>>> grid = lc.load_grid("fppe")
>>> ds = lc.load_ppe("fppe", "EFLX_LH_TOT", "lnd", "month_1")
>>> ds = lc.load_ppe("ippe", "EFLX_LH_TOT", "lnd", "month_1")   # offline: "lnd" only
>>> ds = lc.load_cesm2le("EFLX_LH_TOT", "lnd", "month_1", "h0", bb="cmip6")
>>> lc.ppe_member_name(2)   # "2.dleaf.max"
>>> variables = {"EFLX_LH_TOT": "lnd", "TLAI": "lnd", "PRECT": "atm", "FSNS": "atm", "FLNS": "atm"}
>>> raw = lc.load_native("lens", None, slice("1995-01", "2014-12"), variables=variables)
>>> landfrac, src_grid = lc.load_land_fraction("lens")
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import xarray as xr

from etunc.grid import bounded_src_grid


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
# Offline CLM runs (lnd output only), forced by a coupler history (CPLHIST) shared by all members
IHIST_PPE_ROOT = Path("/glade/campaign/univ/uwas0155/ppe/historical/offline_simulations")
IHIST_PPE_BASENAME = "i.e21.CPLHIST_BGC.f19_f19_mg17.historical.coupPPE"
# {ppe: (root, case basename)}; both PPEs share the member ids of FHIST_PPE_MEMBERS
PPE_CASES = {
    "fppe": (FHIST_PPE_ROOT, FHIST_PPE_BASENAME),
    "ippe": (IHIST_PPE_ROOT, IHIST_PPE_BASENAME),
}
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
# The offline PPE runs on the same f19 grid, with the same LANDFRAC
GRID_PATHS["ippe"] = GRID_PATHS["fppe"]

# PPE (FHIST and IHIST) member id of the minimum and maximum value of each perturbed parameter
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
def load_grid(source: Literal["fppe", "ippe", "goga", "lens"]) -> xr.Dataset:
    """Grid dataset (LANDFRAC, LANDAREA, ...) of a CESM ensemble."""
    return xr.open_dataset(GRID_PATHS[source])


# ------------------------------------------------------------------
# Perturbed-parameter ensembles: coupled FHIST ("fppe") and offline IHIST ("ippe")
# ------------------------------------------------------------------

def _glob_member_files(
    root: Path,
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
            root.glob(
                f"{mcase}/{gcomp}/proc/tseries/{frequency}/"
                f"{mcase}.{scomp}.{stream}.{v}.*.nc"
            )
        )
        if files:
            result[v] = files
    return result


def _load_member_netcdf(
    ppe: str,
    m: int,
    gcomp: str,
    scomp: str,
    frequency: str,
    stream: str,
    variables: list[str],
) -> tuple[int, xr.Dataset | None]:
    """
    Load a single member of PPE `ppe` ("fppe" or "ippe") from NetCDF tseries files.

    Opens all per-variable file lists in a single ``open_mfdataset`` call.
    Each tseries file holds one variable, so xarray separates them by name.

    Returns
    -------
    (member_id, dataset); dataset is None if no files were found.
    """
    root, basename = PPE_CASES[ppe]
    mcase = f"{basename}.{str(m).zfill(3)}"
    var_files = _glob_member_files(root, mcase, gcomp, scomp, frequency, stream, variables)
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


def load_ppe(
    ppe: Literal["fppe", "ippe"],
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
    Load variables from a perturbed-parameter ensemble.

    Parameters
    ----------
    ppe : str
        "fppe" (coupled FHIST PPE) or "ippe" (offline IHIST PPE, "lnd" output only).
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
                pool.submit(_load_member_netcdf, ppe, m, gcomp, scomp, frequency, stream, variables)
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
    Name of a PPE (FHIST or IHIST) member, e.g. "2.dleaf.max" ("dleaf.max" with `no_id`).

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

# ------------------------------------------------------------------
# Monthly fields and land fraction for binning (scripts/prep_lens.py, prep_fppe.py, prep_ippe.py)
# ------------------------------------------------------------------

def _lens_fields(
    variables: Mapping[str, str],
    members: Sequence[str] | None,
    time_slice: slice,
    frequency: str,
    stream: str,
    bb: str,
) -> dict[str, xr.DataArray]:
    """{variable: (member, time, lat, lon)} of LENS2 `members` (ens_names; None = all), with member_id = ens_name."""
    fields, ens = {}, None
    for v, gcomp in variables.items():
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
        fields[v] = da.assign_coords(member_id=("member", ens))
    return fields


def _fppe_fields(
    variables: Mapping[str, str],
    members: Sequence[int] | None,
    time_slice: slice,
    frequency: str,
    stream: str,
    drop_outliers: Sequence[int],
) -> dict[str, xr.DataArray]:
    """
    {variable: (member, time, lat, lon)} of the FHIST PPE `members` (None = all but
    `drop_outliers`) that have every variable, with member_id = parameter name.
    """
    # One `load_ppe` call per component
    by_gcomp: dict[str, list[str]] = {}
    for v, gcomp in variables.items():
        by_gcomp.setdefault(gcomp, []).append(v)
    ds = {}
    for gcomp, vs in by_gcomp.items():
        d = load_ppe("fppe", vs, gcomp, frequency, stream, members=members, drop_outliers=drop_outliers)
        if not d.data_vars:
            raise RuntimeError(f"No FHIST PPE files found for {vs} ({gcomp})")
        missing = [v for v in vs if v not in d]
        if missing:
            raise RuntimeError(f"Missing FHIST PPE variables {missing} ({gcomp})")
        # load_ppe squeezes out `member` when only one member is found
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
        v: ds[gcomp][v].sel(member=ids, time=time_slice).reset_coords(drop=True).assign_coords(member_id=("member", names))
        for v, gcomp in variables.items()
    }


def _ippe_fields(
    variables: Mapping[str, str],
    members: Sequence[int] | None,
    time_slice: slice,
    frequency: str,
    stream: str,
    drop_outliers: Sequence[int],
) -> dict[str, xr.DataArray]:
    """
    {variable: (member, time, lat, lon)} of the offline IHIST PPE `members` (None = all but
    `drop_outliers`), with member_id = parameter name. Offline CLM has "lnd" output only.
    """
    not_lnd = {v: g for v, g in variables.items() if g != "lnd"}
    if not_lnd:
        raise ValueError(f"The offline IHIST PPE has only 'lnd' output, not {not_lnd}")

    # One `load_ppe` call: every variable is in "lnd", so each member has all or none of them
    vs = list(variables)
    ds = load_ppe("ippe", vs, "lnd", frequency, stream, members=members, drop_outliers=drop_outliers)
    if not ds.data_vars:
        raise RuntimeError(f"No IHIST PPE files found for {vs}")
    missing = [v for v in vs if v not in ds]
    if missing:
        raise RuntimeError(f"Missing IHIST PPE variables {missing}")
    # load_ppe squeezes out `member` when only one member is found
    ds = ds if "member" in ds.dims else ds.expand_dims("member")

    ids = ds.member.values.tolist()
    names = [ppe_member_name(m) for m in ids]
    return {
        v: ds[v].sel(time=time_slice).reset_coords(drop=True).assign_coords(member_id=("member", names))
        for v in vs
    }


def load_native(
    source: Literal["lens", "fppe", "ippe"],
    members: Sequence[str] | Sequence[int] | None,
    time_slice: slice,
    *,
    variables: Mapping[str, str],
    frequency: str = "month_1",
    stream: str = "h0",
    bb: str = "cmip6",
    drop_outliers: Sequence[int] = (13, 28),
) -> dict[str, xr.DataArray]:
    """
    Monthly fields {variable: (member, time, lat, lon)} of a CESM ensemble over
    `time_slice` on its native grid, in native units (step 1 of the binning
    pipeline), with `member_id` on `member`.

    source : "lens" (LENS2; `members` are ens_names, None = all, `bb` its
        biomass burning forcing) or "fppe" (FHIST PPE; `members` are integer
        ids, None = all but `drop_outliers`; only members with every variable)
        or "ippe" (offline IHIST PPE; as "fppe", with "lnd" variables only).
    variables : {CESM variable: gcomp ("lnd" or "atm")}.
    """
    if source == "lens":
        return _lens_fields(variables, members, time_slice, frequency, stream, bb)
    if source == "fppe":
        return _fppe_fields(variables, members, time_slice, frequency, stream, drop_outliers)
    if source == "ippe":
        return _ippe_fields(variables, members, time_slice, frequency, stream, drop_outliers)
    raise ValueError(f"Unsupported CESM source {source!r}; use 'lens', 'fppe' or 'ippe'")


def load_land_fraction(source: Literal["fppe", "ippe", "goga", "lens"]) -> tuple[xr.DataArray, xr.Dataset]:
    """LANDFRAC [0-1] of a CESM ensemble's grid file, and that grid with cell edges for regridding."""
    grid = load_grid(source)
    return grid["LANDFRAC"], bounded_src_grid(grid)
