"""
etunc.legacy
============
Notebook-only code that no script uses, kept because the exploratory
notebooks (binned_stats, cmip-et, check-cmip-*, load-cmip, cmip-regrid-res)
still call it: the original ILAMB and CMIP loaders from the notebook era,
`open_bin_stats` for bin-stats files written by the notebooks, and small
helpers. New code should use `etunc.load` and the drivers' own loaders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import regionmask as regmask
import xarray as xr

from etunc.binning import ensure_bin_coords
from etunc.config import ILAMB_ROOT, LAT_BNDS, LF_THRESH
from etunc.grid import cell_area, equal_coords, mask_greenland
from etunc.temporal import to_yyyymm
from etunc.units import convert_units


def get_one_mid(da: xr.DataArray) -> str:
    """Member id of a single-member CMIP DataArray, for file names."""
    if ("member" in da.dims):
        if ("member_id" in da.coords):
            return str(da.member_id[0].item())
    return "onemember"


def safe_squeeze(da: xr.DataArray, dim: str, drop: bool = True) -> xr.DataArray:
    """Squeeze out `dim` if `da` has it, else return `da` unchanged."""
    if dim in da.dims:
        return da.squeeze(dim=dim, drop=drop)
    return da


def compute_cell_area(ds: xr.Dataset | xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Grid cell area and land grid cell area [m2], using `etunc.grid.cell_area`
    (a port of ILAMB's CellAreas) and the Natural Earth land mask. Uses
    `lat_bounds`/`lon_bounds` if present.
    """
    land = regmask.defined_regions.natural_earth_v5_1_2.land_50

    if "lat_bounds" in ds and "lon_bounds" in ds:
        method = "bounds"
        lat_bounds = ds["lat_bounds"].values
        lon_bounds = ds["lon_bounds"].values
    else:
        method = "coords"
        lat_bounds = None
        lon_bounds = None

    area = cell_area(
        lat=ds["lat"].values,
        lon=ds["lon"].values,
        lat_bnds=lat_bounds,
        lon_bnds=lon_bounds,
    )

    area = xr.DataArray(area, dims=["lat", "lon"], coords={"lat": ds["lat"], "lon": ds["lon"]})
    area.attrs["units"] = "m2"
    area.attrs["long_name"] = "grid cell area"
    area.attrs["method"] = method

    # Unlike `etunc.grid.land_mask`, Greenland and Iceland are kept
    mask = xr.where(np.isnan(land.mask(lon_or_obj=area.lon, lat=area.lat)), 0, 1)
    la = area * mask
    la.attrs["units"] = "m2"
    la.attrs["long_name"] = "land grid cell area"
    la.attrs["method"] = method

    return area, la


def load_ilamb_obs(
    variable: str,
    product: str,
    *,
    root: Path = ILAMB_ROOT,
    time_slice: slice = slice(None, None),
    lat_bnds: slice = LAT_BNDS,
    verbose: bool = True,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Load an ILAMB observational product, converted to W/m2 if a water flux.

    Returns
    -------
    (field, cell area, land cell area)
    """
    ds = xr.open_dataset(root / variable / f"{variable}_{product}.nc")
    if product == "WECANN":
        ds = ds.sortby("lat", ascending=True)  # flip the lat dimension for WECANN

    obs = ds[variable].sel(time=time_slice)
    area, la = compute_cell_area(ds)
    if verbose:
        print(
            f"{product:11}: {obs.dims} {obs.shape} {to_yyyymm(obs.time[0])}-{to_yyyymm(obs.time[-1])} "
            f"(cell area from {area.attrs['method']})"
        )
    # Keyed on the variable name, not the units attr (cf. `units.latent_heat_to_wm2`):
    # only "et" and "pr" are converted, assuming kg m-2 s-1
    return (
        convert_units(variable, obs).sel(lat=lat_bnds),
        area.sel(lat=lat_bnds),
        la.sel(lat=lat_bnds),
    )


def filter_all_variables_available(
    data_dict: dict,
    variables: Sequence[str],
    coords: Iterable[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Keep sources in {source: {var: da}} that have all `variables` on matching `coords`."""
    sid_avail = []
    missing = {}
    for sid, vardict in data_dict.items():
        absent = set(variables) - set(vardict.keys())
        if absent:
            missing[sid] = f"missing {absent}"
            continue
        test_da = vardict[variables[0]]
        if coords is not None:
            bad = [v for v in variables[1:] if not equal_coords(vardict[v], test_da, coords)]
            if bad:
                missing[sid] = f"{bad} mismatched coords {tuple(coords)}"
                continue
        sid_avail.append(sid)

    if verbose:
        print("=== Not available ===")
        for sid, reason in missing.items():
            print(f"{sid:16}: {reason}")
        print(f"\n=== Available: {len(sid_avail)} ===\n{sid_avail}")

    return {sid: data_dict[sid] for sid in sid_avail}


def load_cmip(
    variables: Sequence[str],
    catalog: str | Path,
    *,
    experiment_id: str = "historical",
    time_slice: slice = slice(None, None),
    lat_bnds: slice = LAT_BNDS,
    lf_thresh: float = LF_THRESH,
    member_id: str | Sequence[str] | None = "top",
    source_id: str | Sequence[str] | None = None,
    omit_source_id: str | Sequence[str] | None = None,
    grid_variables: Sequence[str] = ("areacella", "sftlf"),
    verbose: bool = True,
    **loader_kwargs,
) -> dict[str, dict[str, xr.DataArray]]:
    """
    Load CMIP6 variables with CMIPESGFLoader, keeping models with all
    variables and grid variables. Every variable is put on the areacella grid,
    masked with `mask_greenland`, and converted to W/m2 where applicable.

    Returns
    -------
    {source_id: {variable: da, "la": land area, "lf": land fraction, "mask": mask}}
    """
    from etunc.load.cmip import CMIPESGFLoader

    loader = CMIPESGFLoader(catalog)
    cmip = loader.load_data(
        variables=list(variables) + list(grid_variables),
        experiment_id=experiment_id,
        source_id=source_id,
        omit_source_id=omit_source_id,
        member_id=member_id,
        time_slice=time_slice,
        **loader_kwargs,
    )
    cmip = filter_all_variables_available(cmip, list(variables), coords=("lat", "lon", "time"), verbose=verbose)
    cmip = filter_all_variables_available(cmip, list(grid_variables), coords=None, verbose=verbose)

    for sid, vardict in cmip.items():
        area = vardict.pop("areacella").sel(lat=lat_bnds)
        lf = vardict.pop("sftlf").sel(lat=lat_bnds).reindex_like(area, method="nearest", tolerance=1e-3)
        if lf.attrs.get("units") == "%":  # sftlf is in %, lf_thresh is a fraction
            lf = lf / 100
            lf.attrs["units"] = "1"
        mask = mask_greenland(lf, lf_thresh)
        mask = mask.where(mask.notnull(), other=False)
        # Snapped to the areacella grid (round-off only, no regridding) and masked on the native grid
        for v, da in vardict.items():
            vardict[v] = convert_units(
                v,
                da.sel(lat=lat_bnds).reindex_like(mask, method="nearest", tolerance=1e-3).where(mask),
                verbose=verbose,
            )
        vardict["lf"] = lf
        vardict["la"] = (area * lf).compute()  # fractional land area, not area * mask
        vardict["mask"] = mask
        if verbose:
            print(f"{sid:20}: {[(v, vardict[v].shape) for v in variables]}")

    return cmip


def open_bin_stats(path: str | Path, var: str | None = None) -> xr.DataArray:
    """
    Open a saved bin-stats file (from `bin_stats` or the older notebook
    version) and ensure it has the bin coords used by the functions below.
    """
    ds = xr.open_dataset(path)
    if var is None:
        if len(ds.data_vars) != 1:
            raise ValueError(f"{path} has several variables {list(ds.data_vars)}; pass `var`.")
        var = next(iter(ds.data_vars))
    return ensure_bin_coords(ds[var])


def data_dict_nybtes(data_dict):
    """Print the size [GB] of each model's variables in {source_id: {variable: da}} and the total."""
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
                    da, grid_dict[sid][grid_var], ("lat", "lon")
                ):
                    sids_to_remove.add(sid)
            else:
                for gda in grid_dict[sid].values():
                    if not equal_coords(da, gda, ("lat", "lon")):
                        sids_to_remove.add(sid)

    aligned_data = {k: v for k, v in data_dict.items() if k not in sids_to_remove}
    aligned_grid = {k: v for k, v in grid_dict.items() if k not in sids_to_remove}

    return aligned_data, aligned_grid
