"""
etunc.legacy.load_obs
=====================
Unmaintained and untested. The annual loaders of `etunc.load.obs` before the
binning pipeline was restructured (docs/binning-workflow.md): `load_native`
(annual means in W/m2 on the native grid, masked to Natural Earth land, read
one year at a time) and `load_annual` (`regrid_annual` of it).
scripts/prep_obs.py now writes out these steps. Also the per-product
wrappers `load_gleam`, `load_pml` and `load_sith`, and the writer of the
deprecated saved regridded files (`regrid_file`, `output_path`,
`code_provenance`) of scripts/legacy_regrid/regrid_obs.py.
"""

from __future__ import annotations

import functools
import gc
import importlib.metadata
import os
import subprocess
import time
from datetime import datetime as dt, timezone
from pathlib import Path
from typing import Mapping

import xarray as xr

from etunc.config import NA_THRESH, OBS_REGRID_ROOT
from etunc.grid import RESOLUTIONS, bounded_src_grid, landne_mask
from etunc.legacy.grid import approx_resolution, regrid_annual
from etunc.legacy.temporal import annual_means, print_annual
from etunc.load.obs import (
    ObsDataset, _has_year, _regrid_blocks, _resolve, _year_bounds, list_years, load_obs, load_obs_regridded,
)
from etunc.temporal import complete_years, yearly_to_annual
from etunc.units import accumulation_to_flux, latent_heat_to_wm2


def load_native(
    dataset: str | ObsDataset,
    var: str,
    time_slice: slice = slice(None, None),
    *,
    version: str | None = None,
    freq: str = "monthly",
) -> tuple[dict[str, xr.DataArray], xr.DataArray, xr.Dataset]:
    """
    Annual means {var: (year, lat, lon)} [W/m2] of a water flux (mm per day,
    month or year, e.g. ET or E) on the native grid over the complete years
    in `time_slice`, masked to land (`etunc.grid.landne_mask` of the native
    grid). Monthly products need all 12 months for a valid annual mean; for
    yearly products the yearly total is the annual mean. Read one year at a
    time, so only one native year is in memory.

    Returns (annual means, native land mask, native grid), the input of
    `etunc.grid.regrid_annual`.
    """
    spec, version = _resolve(dataset, version, freq)
    if freq not in ("monthly", "yearly"):
        raise ValueError(f"{spec.name}: annual means need monthly or yearly files, not {freq!r}")
    if _has_year(spec):
        y0, y1 = _year_bounds(time_slice)
        years = [
            y for y in list_years(spec, var, version, freq) if (y0 is None or y >= y0) and (y1 is None or y <= y1)
        ]
        reads = [slice(str(y), str(y)) for y in years]
    else:
        reads = [time_slice]  # one file holds every year
    if not reads:
        raise FileNotFoundError(f"No {spec.name} {version} {freq} files for {var!r} in {time_slice}")

    pieces, native_mask = [], None
    for read in reads:
        da = load_obs(spec, var, read, version=version, freq=freq).sel(time=time_slice)
        if da.sizes["time"] == 0 or (freq == "monthly" and not complete_years(da)):
            continue  # nothing, or a partial year, in `time_slice`
        # Totals (mm/month, mm/year) -> W/m2, using each time step's month or year length
        da = latent_heat_to_wm2(accumulation_to_flux(da.load()))
        ann = annual_means({var: da})[var] if freq == "monthly" else yearly_to_annual(da)
        if native_mask is None:
            # Natural Earth land, as the target-grid mask of the binning
            native_mask = landne_mask(ann)
        with xr.set_options(keep_attrs=True):
            pieces.append(ann.where(native_mask))
        del da
        gc.collect()
    if not pieces:
        raise ValueError(f"{spec.name} {version} {freq} {var}: no complete year in {time_slice}")
    ann = xr.concat(pieces, dim="year").rename(var)
    return {var: ann}, native_mask, bounded_src_grid(ann)


def load_annual(
    dataset: str | ObsDataset,
    var: str,
    time_slice: slice,
    res: float | str,
    *,
    version: str | None = None,
    freq: str = "monthly",
    mask: xr.DataArray | None = None,
    na_thresh: float = NA_THRESH,
) -> dict[str, xr.DataArray]:
    """
    Annual means {var: (year, lat, lon)} [W/m2] of a water flux on
    `target_grid(res)` within LAT_BNDS, masked with `mask` if given:
    `etunc.grid.regrid_annual` of `load_native`.
    """
    spec, version = _resolve(dataset, version, freq)
    label = f"{spec.name} {version}"
    native = load_native(spec, var, time_slice, version=version, freq=freq)
    ann = regrid_annual(*native, res, mask=mask, na_thresh=na_thresh, label=label)
    print_annual(ann[var], f"{var:3} {label}")
    return ann


def _load_named(
    name: str, var: str, time_slice: slice, regrid_res: float | str | None, na_thresh: float, kwargs: dict,
) -> xr.DataArray:
    """`load_obs` of a registered dataset, or `load_obs_regridded` if `regrid_res` is given."""
    if regrid_res is None:
        return load_obs(name, var, time_slice, **kwargs)
    if kwargs.pop("res", "native") != "native":
        raise ValueError("Give either regrid_res (regrid in memory) or res (saved regridded files), not both")
    return load_obs_regridded(name, var, time_slice, regrid_res, na_thresh=na_thresh, **kwargs)


def load_gleam(
    var: str, time_slice: slice = slice(None, None), *,
    regrid_res: float | str | None = None, na_thresh: float = NA_THRESH, **kwargs,
) -> xr.DataArray:
    """
    GLEAM v4.3 variable (E, Et, Ec, Es, Ei, Eb, Ew, Ep); see `load_obs` for kwargs.
    With `regrid_res`, regridded in memory by `load_obs_regridded`.
    """
    return _load_named("gleam", var, time_slice, regrid_res, na_thresh, kwargs)


def load_pml(
    var: str, time_slice: slice = slice(None, None), *,
    regrid_res: float | str | None = None, na_thresh: float = NA_THRESH, **kwargs,
) -> xr.DataArray:
    """
    PML-V2.2 variable (ET, Ec, Es, Ei, E, Ew, PET, GPP); see `load_obs` for kwargs.
    With `regrid_res`, regridded in memory by `load_obs_regridded`.
    """
    return _load_named("pml", var, time_slice, regrid_res, na_thresh, kwargs)


def load_sith(
    var: str, time_slice: slice = slice(None, None), *,
    regrid_res: float | str | None = None, na_thresh: float = NA_THRESH, **kwargs,
) -> xr.DataArray:
    """
    SiTHv2 variable (ET, Tr, Es, Ei, En); see `load_obs` for kwargs.
    With `regrid_res`, regridded in memory by `load_obs_regridded`.
    """
    return _load_named("sith", var, time_slice, regrid_res, na_thresh, kwargs)


REPO_ROOT = Path(__file__).resolve().parents[2]  # editable install, so the git checkout


PROVENANCE_PACKAGES = ("etunc", "xesmf", "esmpy", "xarray", "numpy", "netCDF4")


@functools.cache
def code_provenance() -> dict[str, str]:
    """
    Code state that produced a regridded file: the git commit of the repo
    (suffixed "-dirty" if etunc/ or scripts/ have uncommitted changes) and the
    versions of the packages that do the regridding. Unavailable entries are
    "unknown". Cached, so git runs once per process.
    """
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                              check=True, timeout=30).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        if git("status", "--porcelain", "--untracked-files=no", "--", "etunc", "scripts"):
            commit += "-dirty"
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    prov = {"etunc_git_commit": commit}
    for pkg in PROVENANCE_PACKAGES:
        try:
            prov[f"{pkg}_version"] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            prov[f"{pkg}_version"] = "unknown"
    return prov


def output_path(spec: ObsDataset, res_tag: str, src: Path) -> Path:
    """Regridded file of `src` at `res_tag` ("0.5deg", "1deg"): the tree that `at_resolution` reads."""
    return OBS_REGRID_ROOT / spec.root.name / res_tag / src.relative_to(spec.root)


def _save(da: xr.DataArray, var: str, src: Path, outpath: Path, complevel: int, history: str | None = None) -> None:
    """
    Write `da` next to `outpath` as .tmp, then move it into place. The source's
    global attrs are kept, with `history` prepended to its CF "history" attr.
    """
    with xr.open_dataset(src) as ds_src:
        global_attrs = dict(ds_src.attrs)
    if history is not None:
        global_attrs["history"] = "\n".join(filter(None, [history, str(global_attrs.get("history", ""))]))
    da.encoding = {}
    ds = xr.Dataset({var: da}, attrs=global_attrs)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = outpath.with_name(outpath.name + ".tmp")
    ds.to_netcdf(tmp, encoding={var: {"zlib": True, "complevel": complevel}})
    # Atomic rename: a killed job leaves only a .tmp, never a truncated file that overwrite=False would skip
    os.replace(tmp, outpath)


def regrid_file(
    spec: ObsDataset, var: str, src: Path, years: tuple[int, int], version: str, freq: str,
    *,
    resolutions: Mapping[str, float] = RESOLUTIONS,
    na_thresh: float = NA_THRESH,
    overwrite: bool = True,
    complevel: int = 4,
    script: str = "regrid_obs.py",
) -> dict[str, float] | None:
    """
    Conservatively regrid one source file, covering `years` (first, last), to
    every grid in `resolutions` ({tag: spacing}), and write each to
    `output_path` (`regrid_with_na_thresh`). A multi-year file is read in
    blocks of its on-disk time chunks. Existing outputs are skipped unless
    `overwrite`. `script` is recorded in the "regrid_script" attr. Returns
    timings, or None if skipped.

    Provenance in the variable attrs: source file with its mtime and size,
    the load_obs preprocess, regrid settings and date, and `code_provenance`
    (git commit, package versions). The global "history" attr gets one line.
    """
    outpaths = {tag: output_path(spec, tag, src) for tag in resolutions}
    if not overwrite and all(p.exists() for p in outpaths.values()):
        return None

    first, last = years
    label = str(first) if first == last else f"{first}-{last}"
    print(f"    {label}: load+regrid...", end="", flush=True)
    # Per-year file: read whole. Multi-year file: dask chunks = on-disk chunks.
    da_all = load_obs(spec, var, slice(str(first), str(last)), version=version, freq=freq,
                         chunks=None if first == last else {})
    if da_all.sizes["time"] == 0:
        raise ValueError(f"{src} has no time steps in {label}")
    regridded, t = _regrid_blocks(da_all, resolutions, na_thresh, progress=first != last)
    t["save"] = 0.0
    print(f" load {t['load']:.1f}s", end="", flush=True)

    # Source state at regrid time: a later mtime or size on disk means the regridded file is out of date
    src_stat = src.stat()
    regrid_date = dt.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    for tag, out in regridded.items():
        out.attrs.pop("res", None)  # load_obs's res="native" describes the source, not this file
        out.attrs["target_grid"] = tag
        out.attrs["src_dims"] = da_all.dims
        out.attrs["src_shape"] = da_all.shape
        out.attrs["src_dlat_deg"], out.attrs["src_dlon_deg"] = approx_resolution(da_all)
        out.attrs["tgt_dlat_deg"], out.attrs["tgt_dlon_deg"] = approx_resolution(out)
        out.attrs["source_file"] = str(src)
        out.attrs["source_mtime"] = dt.fromtimestamp(src_stat.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        out.attrs["source_size_bytes"] = src_stat.st_size
        # Unit fixes and decoding applied by load_obs before regridding
        out.attrs["load_preprocess"] = spec.preprocess.__name__ if spec.preprocess is not None else "none"
        out.attrs["regrid_method"] = "xesmf conservative, skipna=True, unmapped_to_nan=True"
        out.attrs["na_thresh"] = na_thresh
        out.attrs["regrid_script"] = script
        out.attrs["regrid_date"] = regrid_date
        out.attrs.update(code_provenance())

        t0 = time.perf_counter()
        history = f"{regrid_date}: regridded to {tag} by {script} (etunc {out.attrs['etunc_git_commit']})"
        _save(out, var, src, outpaths[tag], complevel, history=history)
        t["save"] += time.perf_counter() - t0
        print(f" | {tag} {out.shape}", end="", flush=True)

    print(f" | regrid {t['regrid']:.1f}s save {t['save']:.1f}s")
    del da_all, regridded
    gc.collect()
    return t
