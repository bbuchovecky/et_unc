"""
load_cesm.py
============
Load CESM2 ensemble output from the CESM time-series (tseries) archives on glade:
the FHIST perturbed-parameter ensemble ("fppe"), the GOGA2 10-member ensemble
("goga") and the CESM2 Large Ensemble ("lens"). Also holds each ensemble's grid
file (LANDFRAC, LANDAREA, ...) and the FHIST PPE member names.

Every loader returns a lazy (dask) Dataset with a ``member`` dimension. Its time
axis is shifted from CESM's end-of-interval stamps to the month the data
averages (see ``shift_time``).

Example
-------
>>> import load_cesm as lc
>>> grid = lc.load_grid("fppe")
>>> ds = lc.load_fhist_ppe("EFLX_LH_TOT", "lnd", "month_1")
>>> ds = lc.load_goga2("EFLX_LH_TOT", "lnd", "month_1", "h0")
>>> ds = lc.load_cesm2le("EFLX_LH_TOT", "lnd", "month_1", "h0", bb="cmip6")
>>> lc.ppe_member_name(2)   # "2.dleaf.max"
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import xarray as xr


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
                    except OSError:
                        log("no files")

                # Branch years before 1200 have one member each (macro
                # perturbations), later ones have 10 (micro perturbations)
                for im, yr in enumerate(LENS_BRANCH_YEARS):
                    log(yr)
                    if int(yr) < 1200:
                        _append(yr, str(im + 1).zfill(3))
                    else:
                        for i in range(1, 11):
                            log(i, end=" ")
                            _append(yr, str(i).zfill(3))
                        log()

                member_datasets.append(
                    xr.concat(members, dim="member")
                    .assign_coords(member=np.arange(1, 51), ens_name=("member", tags))
                )
            combined_ds = xr.merge(member_datasets)

    return shift_time(_drop_cosp(combined_ds))
