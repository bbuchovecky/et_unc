"""
etunc.legacy.load_cmip
======================
Unmaintained and untested. The annual loaders of `etunc.load.cmip` before the
binning pipeline was restructured (docs/binning-workflow.md): `load_native`
(annual means on the native grid, masked with sftlf > LF_THRESH) and
`load_annual` (`regrid_annual` of it). scripts/prep_cmip.py now writes out
these steps. Also:

- `regrid_to_target` (with the EC-Earth (j, i) grid fix) and
  `CMIP_REGRID_ROOT`, of the deprecated scripts/legacy_regrid/regrid_cmip.py;
- `member_tag`, the file-name tag of the former bin_*.py scripts;
- `LegacyCMIPESGFLoader`: `CMIPESGFLoader` with its unused convenience
  methods (`get_avail_*`, `get_source_with_*`).
"""

from __future__ import annotations

import warnings
from typing import Sequence

import pandas as pd
import xarray as xr

from etunc.config import CAMPAIGN_ROOT, LF_THRESH, NA_THRESH
from etunc.grid import check_same_grid, landfrac_mask, make_regridder
from etunc.legacy.grid import regrid_annual
from etunc.legacy.temporal import annual_means, print_annual
from etunc.load.cmip import CMIPESGFLoader, load_land_fraction
from etunc.load.cmip import load_native as load_monthly
from etunc.units import convert_units, net_radiation_cmip


# Regridded files written by scripts/legacy_regrid/regrid_cmip.py (read only by notebooks)
CMIP_REGRID_ROOT = CAMPAIGN_ROOT / "cmip6" / "regridded"


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


def member_tag(members: list[str]) -> str:
    """File-name tag: the member_id of a single member, "<n>members" otherwise."""
    return members[0] if len(members) == 1 else f"{len(members)}members"


def _format_lat_lon(da: xr.DataArray) -> xr.DataArray:
    """Clean and format lat/lon coords (primarily for EC-Earth)."""
    if ("i" in da.dims) and ("j" in da.dims):
        if ("longitude" in da.coords) and ("latitude" in da.coords):
            da = da.rename(i="lon")
            da = da.rename(j="lat")

            latitude = da["latitude"]
            longitude = da["longitude"]

            # Assumes a regular grid (lat varies only along j, lon only along i),
            # so one column and one row give the 1-D coords
            da = da.assign_coords(lat=latitude.isel(lon=0), lon=longitude.isel(lat=0))
            return da.drop_vars(["latitude", "longitude"])
    return da


def regrid_to_target(da: xr.DataArray, res: float | str = 1.0, verbose: bool = False) -> xr.DataArray:
    """
    Conservative (area-weighted) xESMF regridding onto `etunc.grid.target_grid(res)`, as written
    by scripts/regrid_cmip.py. EC-Earth's (j, i) grids are first given 1-D lat/lon
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
            # No skipna/na_thresh, unlike the binning pipeline: a target cell that overlaps any
            # NaN source cell (e.g. ocean in land variables) is NaN
            return regridder(da, keep_attrs=True)

    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(f"xESMF regridding failed - da: {da.dims} {da.shape}") from exc


class LegacyCMIPESGFLoader(CMIPESGFLoader):
    """`CMIPESGFLoader` with the convenience methods that the binning pipeline no longer uses."""

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

    def get_source_with_area(self, experiment_id: str | None = None) -> list[str]:
        """Compatibility wrapper for source_with_area()."""
        return self.source_with_area(experiment_id=experiment_id)

    def get_source_with_sftlf(self, experiment_id: str | None = None) -> list[str]:
        """Compatibility wrapper for source_with_sftlf()."""
        return self.source_with_sftlf(experiment_id=experiment_id)

    def get_source_with_landgrid(self, experiment_id: str | None = None) -> list[str]:
        """Compatibility wrapper for source_with_landgrid()."""
        return self.source_with_landgrid(experiment_id=experiment_id)

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
