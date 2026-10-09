"""
etunc.load.cmip
===============
Module to load CMIP ESGF catalog data and selected datasets from a catalog CSV file.
Primarily for data downloaded using `cmip-intake-esgf-fetch`.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from etunc.config import LAT_BNDS, LF_THRESH
from etunc.grid import (
    NA_THRES, bounded_conservative_regridder, bounded_source_grid, check_same_grid, make_regridder, mask_greenland,
    target_grid,
)
from etunc.temporal import annual_mean, complete_years, period_str
from etunc.units import convert_units, net_radiation_cmip

# -----------------------
# Helpers
# -----------------------

def data_dict_nybtes(data_dict):
    total_ngb = 0
    sid_ngb = {}
    for sid, vardict in data_dict.items():
        sid_ngb[sid] = 0
        for var, da in vardict.items():
            sid_ngb[sid] += da.nbytes / 1024 / 1024 / 1024
        total_ngb += sid_ngb[sid]
    
    print(f"total: {total_ngb:0.3f} GB")
    for sid, ngb in sid_ngb.items():
        print(f"{sid:20}: {ngb:0.3f} GB")


# def clean_data_dict(data_dict):
#     """Remove models without any output."""


def check_coords(da: xr.DataArray) -> bool:
    for coord in ("lat", "lon"):
        if coord not in da.coords:
            return False
        if da[coord].ndim == 0:
            continue  # scalar coord counts as present
        if len(da[coord]) == 0:
            return False
    return True


def equal_coords(a: xr.DataArray, b: xr.DataArray, atol: float = 1e-3) -> bool:
    if not check_coords(a) or not check_coords(b):
        return False
    for coord in ("lat", "lon"):
        if a[coord].shape != b[coord].shape:
            return False
        if not np.allclose(a[coord], b[coord], atol=atol):
            return False
    return True


def align_dicts(
        data_dict: dict,
        grid_dict: dict,
        grid_var: str | None = None,
) -> tuple[dict, dict]:
    """Remove models with mismatched or missing coordinates between data and grid."""
    sids_to_remove: set = set()
    for sid, vardict in data_dict.items():
        if sid not in grid_dict:
            sids_to_remove.add(sid)
            continue
        for var, da in vardict.items():
            if grid_var is not None:
                if grid_var not in grid_dict[sid] or not equal_coords(
                    da, grid_dict[sid][grid_var]
                ):
                    sids_to_remove.add(sid)
            else:
                for gda in grid_dict[sid].values():
                    if not equal_coords(da, gda):
                        sids_to_remove.add(sid)

    aligned_data = {k: v for k, v in data_dict.items() if k not in sids_to_remove}
    aligned_grid = {k: v for k, v in grid_dict.items() if k not in sids_to_remove}

    return aligned_data, aligned_grid


class CMIPESGFLoader:
    """
    Load CMIP ESGF catalog data and selected datasets from a catalog CSV file.
    """

    MEMBER_ID_RE = re.compile(r"r(?P<r>\d+)i(?P<i>\d+)p(?P<p>\d+)f(?P<f>\d+)")


    def __init__(self, catalog_csv_path: str | Path):
        """
        Load the ESGF catalog CSV into memory.

        Parameters
        ----------
        catalog_csv_path:
            Path to the CSV file produced by the ESGF cache catalog export.
        """
        self.catalog_csv_path = Path(catalog_csv_path)
        self.catalog = pd.read_csv(self.catalog_csv_path)


    @staticmethod
    def _as_list(value: str | Sequence[str] | None) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        return list(value)


    @staticmethod
    def _check_coords(da: xr.DataArray) -> bool:
        for coord in ("lat", "lon"):
            if coord not in da.coords:
                return False
            if da[coord].ndim == 0:
                continue  # scalar coord counts as present
            if len(da[coord]) == 0:
                return False
        return True


    def _experiment_mask(self, experiment_id: str | None) -> pd.Series:
        if experiment_id is None:
            return pd.Series(True, index=self.catalog.index)
        return self.catalog["experiment_id"] == experiment_id


    def _source_mask(self, source_id: str | Sequence[str] | None) -> pd.Series:
        source_ids = self._as_list(source_id)
        if source_ids is None:
            return pd.Series(True, index=self.catalog.index)
        return self.catalog["source_id"].isin(source_ids)


    def _member_mask(self, member_id: str | Sequence[str] | None) -> pd.Series:
        member_ids = self._as_list(member_id)
        if member_ids is None:
            return pd.Series(True, index=self.catalog.index)
        return self.catalog["member_id"].isin(member_ids)


    @classmethod
    def _member_sort_key(cls, member_id: str) -> tuple[int, int, int, int]:
        match = cls.MEMBER_ID_RE.fullmatch(member_id)
        if match is None:
            raise ValueError(f"Invalid member_id format: {member_id!r}")
        r = int(match.group("r"))
        i = int(match.group("i"))
        p = int(match.group("p"))
        f = int(match.group("f"))
        return (r, i, p, f)


    @classmethod
    def sort_member_ids(cls, member_ids: Iterable[str]) -> list[str]:
        """Return member IDs sorted by the CMIP r/i/p/f components."""
        return sorted(member_ids, key=cls._member_sort_key)


    @classmethod
    def group_member_ids_by_ipf(cls, member_ids: Iterable[str]) -> dict[tuple[int, int, int], list[str]]:
        """Group member IDs by their i/p/f ensemble key, preserving sorted r order."""
        grouped: dict[tuple[int, int, int], list[str]] = {}

        for member_id in cls.sort_member_ids(member_ids):
            match = cls.MEMBER_ID_RE.fullmatch(member_id)
            if match is None:
                continue

            i = int(match.group("i"))
            p = int(match.group("p"))
            f = int(match.group("f"))
            key = (i, p, f)
            grouped.setdefault(key, []).append(member_id)

        return grouped


    def source_ids(self, experiment_id: str | None = None) -> list[str]:
        """Return all source IDs in the catalog for the selected experiment."""
        subset = self.catalog.loc[self._experiment_mask(experiment_id)]
        return sorted(subset["source_id"].astype(str).unique().tolist())


    def source_with_area(self, experiment_id: str | None = None) -> list[str]:
        """Return source IDs that provide areacella for the selected experiment."""
        subset = self.catalog.loc[
            (self.catalog["variable_id"] == "areacella") & self._experiment_mask(experiment_id)
        ]
        return sorted(subset["source_id"].astype(str).unique().tolist())


    def source_with_sftlf(self, experiment_id: str | None = None) -> list[str]:
        """Return source IDs that provide sftlf for the selected experiment."""
        subset = self.catalog.loc[
            (self.catalog["variable_id"] == "sftlf") & self._experiment_mask(experiment_id)
        ]
        return sorted(subset["source_id"].astype(str).unique().tolist())


    def source_with_landgrid(self, experiment_id: str | None = None) -> list[str]:
        """Return source IDs that provide both areacella and sftlf."""
        source_with_area = set(self.source_with_area(experiment_id=experiment_id))
        source_with_sftlf = set(self.source_with_sftlf(experiment_id=experiment_id))
        source_with_landgrid = sorted(source_with_area & source_with_sftlf)
        print(f"number of models with areacella and sftlf: {len(source_with_landgrid)}")
        return source_with_landgrid


    def get_source_with_area(self, experiment_id: str | None = None) -> list[str]:
        """Compatibility wrapper for source_with_area()."""
        return self.source_with_area(experiment_id=experiment_id)


    def get_source_with_sftlf(self, experiment_id: str | None = None) -> list[str]:
        """Compatibility wrapper for source_with_sftlf()."""
        return self.source_with_sftlf(experiment_id=experiment_id)


    def get_source_with_landgrid(self, experiment_id: str | None = None) -> list[str]:
        """Compatibility wrapper for source_with_landgrid()."""
        return self.source_with_landgrid(experiment_id=experiment_id)


    def get_availability(
        self,
        variables: str | Sequence[str],
        experiment_id: str = "historical",
        source_id: str | Sequence[str] | None = None,
        *,
        source_candidates: Sequence[str] | None = None,
        verbose: bool = True,
    ) -> tuple[dict[str, set[str]], dict[str, list[str]], list[str]]:
        """
        Return available member IDs, variables, and complete source IDs.
        By default, searches every source ID in the catalog for the experiment.

        Parameters
        ----------
        variables:
            One variable name or a list of variable names to test.
        experiment_id, optional:
            Experiment name to filter on, for example ``historical``.
        source_id, optional:
            Source ID or list of source IDs to restrict the search.
        source_candidates, optional:
            Explicit source list to test instead of every source ID in the catalog.
        verbose, optional:
            If True, print the availability summary.
        """
        variables = self._as_list(variables) or []

        if source_id is not None:
            candidate_sources = self._as_list(source_id) or []
        elif source_candidates is not None:
            candidate_sources = list(source_candidates)
        else:
            candidate_sources = self.source_ids(experiment_id=experiment_id)

        avail_member_id: dict[str, set[str]] = {}
        avail_variables: dict[str, list[str]] = {}
        avail_source_id: list[str] = []

        for sid in candidate_sources:
            if verbose:
                print(f"=== {sid} ===")

            avail_variables[sid] = []
            num_avail_vars = 0
            avail_mid = set(self.catalog["member_id"].unique())

            for var in variables:
                query = (
                    (self.catalog["experiment_id"] == experiment_id)
                    & (self.catalog["source_id"] == sid)
                    & (self.catalog["variable_id"] == var)
                )
                subset = self.catalog.loc[query]

                if len(subset) > 0:
                    if verbose:
                        print(f"  ✅ {var:11}")

                    unique_member_id = subset["member_id"].unique()
                    avail_mid &= set(unique_member_id)
                    avail_variables[sid].append(var)
                    num_avail_vars += 1
                elif verbose:
                    print(f"  ❌ {var}")

            avail_member_id[sid] = avail_mid
            if verbose and len(avail_mid) == 0:
                print("  ⚠️ no shared member_id")
            if num_avail_vars == len(variables):
                avail_source_id.append(sid)

        return avail_member_id, avail_variables, avail_source_id


    def get_avail_member_id(
        self,
        variables: str | Sequence[str],
        experiment_id: str = "historical",
        source_id: str | Sequence[str] | None = None,
        *,
        source_candidates: Sequence[str] | None = None,
        verbose: bool = True,
    ) -> dict[str, set[str]]:
        """Return the available member IDs per source."""
        avail_member_id, _, _ = self.get_availability(
            variables,
            experiment_id=experiment_id,
            source_id=source_id,
            source_candidates=source_candidates,
            verbose=verbose,
        )
        return avail_member_id


    def get_avail_variables(
        self,
        variables: str | Sequence[str],
        experiment_id: str = "historical",
        source_id: str | Sequence[str] | None = None,
        *,
        source_candidates: Sequence[str] | None = None,
        verbose: bool = True,
    ) -> dict[str, list[str]]:
        """Return the available variables per source."""
        _, avail_variables, _ = self.get_availability(
            variables,
            experiment_id=experiment_id,
            source_id=source_id,
            source_candidates=source_candidates,
            verbose=verbose,
        )
        return avail_variables


    def get_avail_source_id(
        self,
        variables: str | Sequence[str],
        experiment_id: str = "historical",
        source_id: str | Sequence[str] | None = None,
        *,
        source_candidates: Sequence[str] | None = None,
        verbose: bool = True,
    ) -> list[str]:
        """Return source IDs that have all requested variables available."""
        _, _, avail_source_id = self.get_availability(
            variables,
            experiment_id=experiment_id,
            source_id=source_id,
            source_candidates=source_candidates,
            verbose=verbose,
        )
        return avail_source_id


    def get_member_ids_max_r(
        self,
        variables: str | Sequence[str],
        experiment_id: str = "historical",
        source_id: str | Sequence[str] | None = None,
        *,
        source_candidates: Sequence[str] | None = None,
        verbose: bool = True,
    ) -> dict[str, list[str]]:
        """Return the member IDs from the largest r-group for each source."""
        avail_member_id, avail_variables, avail_source_id = self.get_availability(
            variables,
            experiment_id=experiment_id,
            source_id=source_id,
            source_candidates=source_candidates,
            verbose=verbose,
        )

        member_ids_max_r: dict[str, list[str]] = {}

        if verbose:
            print("n_r is the number of groups of members with constant i,p,f and different r (i.e., number of unique initial condition ensembles)")
            print("max_r is the number of members in the i,p,f group with the greatest number of members - this is one selected in 'top_member_id'")
            print(f"complete with all variables: {variables}")
            print(f"\n{'source_id':17}  {'top_member_id':12}  {'n_member_id':11}  {'n_r':3}  {'max_r':6}  {'complete?'}")
            print("-" * 69)

        for sid, mids in avail_member_id.items():
            if len(mids) == 0:
                continue

            sort_avail_mid = self.sort_member_ids(mids)
            member_ids_by_source = self.group_member_ids_by_ipf(mids)

            top_mid = sort_avail_mid[0]
            nr = len(member_ids_by_source.keys())

            max_r = -np.inf
            max_r_ipf: list[str] = []
            for _, member_group in member_ids_by_source.items():
                if len(member_group) > max_r:
                    max_r = len(member_group)
                    max_r_ipf = member_group
            member_ids_max_r[sid] = max_r_ipf

            if verbose:
                complete_tag = f"❌  {len(avail_variables[sid])}/{len(self._as_list(variables) or [])}   {avail_variables[sid]}"
                if sid in avail_source_id:
                    complete_tag = "✅"
                print(f"{sid:17}  {top_mid:13}  {len(avail_member_id[sid]):<11}  {nr:<3}  {int(max_r):<6}  {complete_tag}")

        return member_ids_max_r


    def get_top_member_id(
        self,
        variables: str | Sequence[str],
        experiment_id: str = "historical",
        source_id: str | Sequence[str] | None = None,
        *,
        source_candidates: Sequence[str] | None = None,
        verbose: bool = True,
    ) -> dict[str, str]:
        """
        Return the top member ID selection per source.

        The top member is the lexicographically smallest available member ID after
        sorting by r/i/p/f components.
        """
        avail_member_id, _, _ = self.get_availability(
            variables,
            experiment_id=experiment_id,
            source_id=source_id,
            source_candidates=source_candidates,
            verbose=verbose,
        )

        top_member_id: dict[str, str] = {}
        for sid, mids in avail_member_id.items():
            if len(mids) == 0:
                continue
            top_member_id[sid] = self.sort_member_ids(mids)[0]

        return top_member_id


    def _resolve_member_ids(
        self,
        sid: str,
        member_id: str | Sequence[str] | None,
        member_ids_max_r: dict[str, list[str]],
        top_member_id: dict[str, str],
    ) -> list[str]:
        if member_id is None:
            return member_ids_max_r.get(sid, [])
        if isinstance(member_id, str) and member_id == "top":
            top_mid = top_member_id.get(sid)
            return [top_mid] if top_mid is not None else []

        resolved = self._as_list(member_id) or []
        if not resolved:
            return []

        return resolved


    def load_data(
        self,
        variables: str | Sequence[str],
        experiment_id: str,
        source_id: str | Sequence[str] | None = None,
        member_id: str | Sequence[str] | None = None,
        time_slice: slice | None = slice(None, None),
        omit_source_id: str | Sequence[str] | None = None,
        *,
        source_candidates: Sequence[str] | None = None,
        parallel: bool = False,
        verbose: bool = True,
    ) -> dict[str, dict[str, xr.DataArray]]:
        """
        Load the requested variables into a nested ``data_dict`` structure.
        By default, searches every source ID in the catalog for the experiment.

        Parameters
        ----------
        variables:
            One variable name or a list of variable names to load.
        experiment_id:
            Experiment name to filter on, for example ``historical``.
        source_id, optional:
            Source ID or list of source IDs to restrict loading.
        member_id, optional:
            A single member ID, a list of member IDs, ``None`` for ``member_ids_max_r``,
            or ``top`` for the top member ID selection.
        time_slice, optional:
            Time slice passed to ``DataArray.sel``.
        source_candidates, optional:
            Explicit source list to load instead of every source ID in the catalog.
        parallel, optional:
            xr.open_mfdataset(parallel=parallel)
        verbose, optional:
            If True, print the loading progress messages.

        Returns
        -------
        dict[str, dict[str, xarray.DataArray]]
            Nested dictionary keyed by source ID and then variable name.
        """
        variables = self._as_list(variables) or []

        if source_id is not None:
            candidate_sources = self._as_list(source_id) or []
        elif source_candidates is not None:
            candidate_sources = list(source_candidates)
        else:
            candidate_sources = self.source_ids(experiment_id=experiment_id)

        if omit_source_id is not None:
            omit_sid = self._as_list(omit_source_id) or []
            candidate_sources = list(set(candidate_sources) - set(omit_sid))

        if member_id == "top":
            print("only selecting top member_id")

        top_member_id = self.get_top_member_id(
            variables,
            experiment_id=experiment_id,
            source_id=source_id,
            source_candidates=candidate_sources,
            verbose=False,
        )
        member_ids_max_r = self.get_member_ids_max_r(
            variables,
            experiment_id=experiment_id,
            source_id=source_id,
            source_candidates=candidate_sources,
            verbose=False,
        )

        data_dict: dict[str, dict[str, xr.DataArray]] = {}

        for sid in sorted(candidate_sources):
            if verbose:
                print(f"=== {sid} ===")

            data_dict[sid] = {}

            mids = self._resolve_member_ids(sid, member_id, member_ids_max_r, top_member_id)
            if len(mids) == 0:
                if verbose:
                    print("  ⚠️ no members to load")
                continue

            for var in variables:
                if verbose:
                    print(f"  {var:11}  n={len(mids):<3}", end="   ")

                das: list[xr.DataArray] = []
                member_file_paths: dict[str, list[str]] = {}

                for mid in mids:
                    query = (
                        (self.catalog["experiment_id"] == experiment_id)
                        & (self.catalog["source_id"] == sid)
                        & (self.catalog["variable_id"] == var)
                        & (self.catalog["member_id"] == mid)
                    )
                    subset = self.catalog.loc[query]
                    subset = subset.sort_values(by="path")

                    # Keep one file per time_range when a file is cataloged twice
                    # (e.g., .../Amon/tas/gn/v20210816/... and .../Amon/tas/gn/files/d20210816/...)
                    nfull = len(subset)
                    subset = subset.drop_duplicates(subset=["time_range"], keep="first")
                    if len(subset) < nfull:
                        print(f"{nfull/len(subset)}x", end="-")

                    if len(subset) == 0:
                        if verbose:
                            print(f"❌{mid}", end=" ")
                        continue

                    # Open exactly the cataloged files: one dataset's files may be split
                    # across directories (e.g., the glade CMIP mirror and the ESGF cache).
                    member_file_paths[mid] = subset["path"].tolist()

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        da = xr.open_mfdataset(member_file_paths[mid], parallel=parallel)[var]

                    # Check that coordinates exist and look ok
                    if self._check_coords(da):
                        if ("time" in da.dims) and (time_slice is not None):
                            da = da.sel(time=time_slice)
                        das.append(da)
                        if verbose:
                            print(f"✅{mid}", end=" ")
                    else:
                        print(f"❌{mid}", end=" ")


                if len(das) > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        concatenated = xr.concat(das, dim="member").assign_coords(
                            member=np.arange(len(mids)),
                            member_id=("member", mids),
                        )

                    concatenated.attrs["source_paths"] = "\n".join(
                        f"{mid}: {', '.join(member_file_paths[mid])}" for mid in mids
                    )
                    concatenated.attrs["catalog_csv_path"] = str(self.catalog_csv_path)
                    data_dict[sid][var] = concatenated

                if verbose:
                    print()

        return data_dict


# ------------------------------------------------------------------
# Model and member selection, land fraction (cmip_binned_et.py)
# ------------------------------------------------------------------

def available_members(
    catalog: pd.DataFrame, variables: Sequence[str], experiment_id: str = "historical",
) -> dict[str, list[str]]:
    """{source_id: sorted members with every one of `variables`} for `experiment_id`."""
    hist = catalog[catalog["experiment_id"] == experiment_id]
    avail = {}
    for sid, g in hist.groupby("source_id"):
        mids = set.intersection(*(set(g.loc[g["variable_id"] == v, "member_id"]) for v in variables))
        if mids:
            avail[sid] = CMIPESGFLoader.sort_member_ids(mids)
    return avail


def sftlf_files(fx_catalog: Path) -> dict[str, str]:
    """{source_id: sftlf file}. The fx catalog holds one file per model (often from piControl)."""
    fx = pd.read_csv(fx_catalog)
    rows = fx[fx["variable_id"] == "sftlf"]
    dup = rows["source_id"][rows["source_id"].duplicated()].tolist()
    if dup:
        raise ValueError(f"{fx_catalog}: more than one sftlf file for {dup}")
    return dict(zip(rows["source_id"], rows["path"]))


def select_members(
    sid: str,
    avail: list[str],
    member_ids: Mapping[str, str | list[str]] | None = None,
    default: str | list[str] = "top",
) -> list[str]:
    """
    Members of model `sid` to use, from its available members `avail` (sorted,
    see `available_members`) and the selection `member_ids.get(sid, default)`:
    a list of member_ids (unavailable ones are dropped), or
      "top"   : the first member
      "max_r" : the largest group of members sharing i/p/f (ties: the first group)
      "all"   : every available member
    """
    spec = (member_ids or {}).get(sid, default)
    if spec == "top":
        return avail[:1]
    if spec == "all":
        return avail
    if spec == "max_r":
        return max(CMIPESGFLoader.group_member_ids_by_ipf(avail).values(), key=len)
    if isinstance(spec, str):
        raise ValueError(f"{sid}: unknown member selection {spec!r}")
    missing = [m for m in spec if m not in avail]
    if missing:
        print(f"{sid}: members {missing} are not available (not all variables), skipping them")
    return [m for m in spec if m in avail]


def member_tag(members: list[str]) -> str:
    """File-name tag: the member_id of a single member, "<n>members" otherwise."""
    return members[0] if len(members) == 1 else f"{len(members)}members"


def load_land_fraction(path: str | Path) -> tuple[xr.DataArray, xr.Dataset]:
    """Land fraction [0-1] on a model's native grid, and that grid with cell edges for regridding."""
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    lf = ds["sftlf"].reset_coords(drop=True)
    # sftlf should be in %, but some files store a fraction labelled "%" (E3SM-1-0), so check the values
    if float(lf.max()) > 1.5:
        lf = lf / 100
    return lf.assign_attrs(units="1"), bounded_source_grid(ds)


# ------------------------------------------------------------------
# Annual means on the common grid (cmip_binned_et.py)
# ------------------------------------------------------------------

def load_native(
    loader: CMIPESGFLoader,
    sid: str,
    members: list[str],
    sftlf_path: str,
    variables: Sequence[str],
    experiment_id: str,
    time_slice: slice,
) -> tuple[dict[str, xr.DataArray], xr.DataArray, xr.Dataset]:
    """
    Annual means (member, year, lat, lon) of ET, LAI, pr and Rn [W/m2, m2/m2]
    for one model on its native grid, masked with its land fraction (sftlf >
    LF_THRESH, no Greenland/Iceland). `variables` must include evspsbl, lai,
    pr, rsds, rsus, rlds and rlus. Complete years only; a missing month
    counts as LAI = 0, while ET, pr and Rn need all 12 months.

    Returns (annual means, native land mask, native grid with cell edges for
    `regrid_annual`).
    """
    lf, src_grid = load_land_fraction(sftlf_path)
    native_mask = mask_greenland(lf, LF_THRESH)

    data = loader.load_data(
        variables, experiment_id, source_id=sid, member_id=members, time_slice=time_slice, verbose=False,
    )[sid]
    missing = set(variables) - set(data)
    if missing:
        raise ValueError(f"{sid}: could not load {sorted(missing)}")

    fields = {}
    for v in variables:
        da = data[v].reset_coords(drop=True)  # member_id is re-added after regridding
        # sftlf comes from another experiment, so check that it is on the same grid
        check_same_grid(da, lf, f"{sid}/{v}")
        da = da.assign_coords(lat=lf.lat, lon=lf.lon)
        fields[v] = da.sel(time=da.time.dt.year.isin(complete_years(da)))

    monthly = {
        "et": convert_units("evspsbl", fields["evspsbl"]),
        "lai": fields["lai"].assign_attrs(units="m2/m2"),
        "pr": convert_units("pr", fields["pr"]),
        "rn": net_radiation_cmip(fields["rsds"], fields["rsus"], fields["rlds"], fields["rlus"]),
    }

    ann = {}
    for k, da in monthly.items():
        with xr.set_options(keep_attrs=True):
            da = da.where(native_mask).load()
        ann[k] = annual_mean(da, require_all_months=(k != "lai")).transpose(..., "lat", "lon")
    return ann, native_mask, src_grid


def regrid_annual(
    ann: dict[str, xr.DataArray],
    src_grid: xr.Dataset,
    res: float | str,
    mask: xr.DataArray,
    na_thres: float = NA_THRES,
    members: list[str] | None = None,
) -> dict[str, xr.DataArray]:
    """
    Conservatively regrid native annual means `ann` (from `load_native`) onto
    `target_grid(res)` within LAT_BNDS, masked with `mask`. Each target cell is
    the area-weighted mean of its valid (land) source cells, and NaN where more
    than `na_thres` of its area is missing. `members` adds the member_id coord.
    """
    regridder = bounded_conservative_regridder(src_grid, res)
    grid = target_grid(res)
    member_coord = {} if members is None else {"member_id": ("member", members)}
    out = {}
    for k, a in ann.items():
        # Area-weighted mean of the native land cells in each target cell (ocean is NaN and skipped)
        a = regridder(a, skipna=True, na_thres=na_thres, keep_attrs=True)
        a = a.assign_coords(lat=grid.lat, lon=grid.lon, **member_coord)
        with xr.set_options(keep_attrs=True):
            out[k] = a.sel(lat=LAT_BNDS).where(mask).rename(k)
    return out


def load_model(
    loader: CMIPESGFLoader,
    sid: str,
    members: list[str],
    sftlf_path: str,
    mask: xr.DataArray,
    *,
    variables: Sequence[str],
    experiment_id: str,
    time_slice: slice,
    res: float | str,
    na_thres: float = NA_THRES,
) -> dict[str, xr.DataArray]:
    """
    Annual means (member, year, lat, lon) of ET, LAI, pr and Rn [W/m2, m2/m2]
    for one model, conservatively regridded onto `target_grid(res)` within
    LAT_BNDS and masked with `mask`: `regrid_annual` of `load_native`.
    """
    ann, native_mask, src_grid = load_native(loader, sid, members, sftlf_path, variables, experiment_id, time_slice)
    ann = regrid_annual(ann, src_grid, res, mask, na_thres, members=members)
    print(
        f"{sid:16}: {len(members)} member(s), native {native_mask.sizes['lat']}x{native_mask.sizes['lon']} "
        f"-> {ann['et'].dims} {ann['et'].shape}, years {period_str(ann['et'].year.values)}"
    )
    return ann


# ------------------------------------------------------------------
# Regridding whole fields (regrid_cmip_esgf.py)
# ------------------------------------------------------------------

def _format_lat_lon(da: xr.DataArray) -> xr.DataArray:
    """Clean and format lat/lon coords (primarily for EC-Earth)."""
    if ("i" in da.dims) and ("j" in da.dims):
        if ("longitude" in da.coords) and ("latitude" in da.coords):
            da = da.rename(i="lon")
            da = da.rename(j="lat")

            latitude = da["latitude"]
            longitude = da["longitude"]

            da = da.assign_coords(lat=latitude.isel(lon=0), lon=longitude.isel(lat=0))
            return da.drop_vars(["latitude", "longitude"])
    return da


def regrid_to_target(da: xr.DataArray, res: float | str = 1.0, verbose: bool = False) -> xr.DataArray:
    """
    Conservative (area-weighted) xESMF regridding onto `etunc.grid.target_grid(res)`, as written
    by regrid_cmip_esgf.py. EC-Earth's (j, i) grids are first given 1-D lat/lon
    (`_format_lat_lon`).

    The source grid keeps its original lat/lon dims (1-D or 2-D); xESMF infers
    its cell edges. Raises RuntimeError if the regridding fails.
    """
    da = _format_lat_lon(da)

    try:
        src = xr.Dataset(
            {
                "lon": (da["lon"].dims, da["lon"].values),
                "lat": (da["lat"].dims, da["lat"].values),
            }
        )
        if verbose:
            print(f"Source: lon={len(src.lon)}, lat={len(src.lat)}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regridder = make_regridder(src, res, "conservative")
            if verbose:
                print(f"Target: lon={regridder.shape_out[1]}, lat={regridder.shape_out[0]}")
            return regridder(da, keep_attrs=True)

    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(f"xESMF regridding failed - da: {da.dims} {da.shape}") from exc
