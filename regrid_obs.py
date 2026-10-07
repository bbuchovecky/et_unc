"""
Regrid gridded observational products (GLEAM v4.3, PML-V2.2, SiTHv2) from
their native 0.1 deg grid to the common 0.5 deg and 1 deg grids.

Each source file is loaded with `load_obs.load_obs` (so it gets that module's
unit fixes, decoding and lat/lon standardization) and regridded with xESMF
first-order conservative regridding (`regrid.conservative_regridder`) onto
the common grids of `regrid.target_grid`. Multi-year files (SiTHv2,
1982-2022) are read and regridded in blocks of time steps matching their on-disk chunks (so
each compressed chunk is read once and memory stays bounded), and the blocks
are concatenated into one output file. Missing values are skipped: each coarse
cell is the area-weighted mean of its valid fine cells, and becomes NaN when
more than `NA_THRES` of its area is missing. Target cells outside the source
domain (PML south of 60S) are NaN. Units are kept as stored (mm/month,
mm/year, ...).

Output mirrors the source tree under a resolution directory, one file per
source file:

    REGRID_ROOT/<dataset dir>/<res tag>/<source path relative to dataset dir>
    e.g. regridded/gleam-v4.3/1deg/v4.3a/monthly/E/E_1980_GLEAM_v4.3a_MO.nc
         regridded/pml-v2.2/0.5deg/V2.2c/monthly/ET/PML-V2.2c_ET_1982.nc
         regridded/sith-v2/1deg/Monthly/ET.SiTHv2.A1982_2022.M.nc

so a regridded product can be read back with the `load_obs` functions via
their `res` argument ("0.5" or "1"; see `load_obs.at_resolution`):

>>> import load_obs as lo
>>> et = lo.load_obs("pml", "ET", slice("2003", "2014"), res="1")

Regridded SiTHv2 files are float32 with NaN and dims (time, lat, lon); the
"sith" `_decode_sith` preprocess still reads them correctly (scale factor 1).

Existing outputs are skipped unless `OVERWRITE` is True, so the batch job
(`regrid_obs.pbs`) can be resubmitted after hitting its walltime.
"""
from __future__ import annotations

import gc
import os
import time
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe

import load_obs as lo
import regrid as rg


REGRID_ROOT = lo.REGRID_ROOT  # = /glade/campaign/univ/uwas0155/obs/regridded

# {dataset: {"versions": ..., "freqs": ..., "vars": ...}}; None = all available
SELECTION: dict[str, dict[str, tuple[str, ...] | None]] = {
    # "gleam": {"versions": ("v4.3b",), "freqs": ("monthly", "yearly"), "vars": None},
    # "gleam": {"versions": None, "freqs": ("monthly", "yearly"), "vars": None},
    ## V2.2a-VIIRS skipped: monthly E 2019 is a 0-byte file on the TPDC server
    # "pml":   {"versions": ("V2.2c", "V2.2b", "V2.2a-MODIS"), "freqs": ("monthly", "yearly"), "vars": None},
    # "sith":  {"versions": None, "freqs": ("monthly", "yearly"), "vars": None},
    "sith":  {"versions": None, "freqs": ("yearly",), "vars": None},
}
YEARS: list[int] | None = None  # optional subset of years; multi-year files touching it are regridded whole

RESOLUTIONS = rg.RESOLUTIONS  # output directory tag (= lo.RES_DIRS values) -> grid spacing [deg]
NA_THRES = 0.5     # coarse cells with more than this fraction of missing area are NaN
OVERWRITE = True  # False: skip source files whose outputs all exist
COMPLEVEL = 4


# ------------------------------------------------------------------
# Regridding
# ------------------------------------------------------------------

def regrid(da: xr.DataArray, regridder: xe.Regridder) -> xr.DataArray:
    """Area-weighted mean of the valid source cells; NaN where > NA_THRES is missing."""
    out = regridder(da, skipna=True, na_thres=NA_THRES, keep_attrs=True)
    out = out.astype(da.dtype)
    out.attrs.pop("grid_mapping", None)  # PML's "crs" variable is not carried over
    out["lat"].attrs = dict(rg.LAT_ATTRS)
    out["lon"].attrs = dict(rg.LON_ATTRS)
    return out


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------

def output_path(spec: lo.ObsDataset, res_tag: str, src: Path) -> Path:
    return REGRID_ROOT / spec.root.name / res_tag / src.relative_to(spec.root)


def save(da: xr.DataArray, var: str, src: Path, outpath: Path) -> None:
    """Write `da` next to `outpath` as .tmp, then move it into place."""
    with xr.open_dataset(src) as ds_src:
        global_attrs = dict(ds_src.attrs)
    da.encoding = {}
    ds = xr.Dataset({var: da}, attrs=global_attrs)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = outpath.with_name(outpath.name + ".tmp")
    ds.to_netcdf(tmp, encoding={var: {"zlib": True, "complevel": COMPLEVEL}})
    os.replace(tmp, outpath)


def regrid_file(
    spec: lo.ObsDataset, var: str, src: Path, years: tuple[int, int], version: str, freq: str
) -> dict[str, float] | None:
    """
    Regrid one source file, covering `years` (first, last), to every resolution.
    A multi-year file is read in blocks of its on-disk time chunks. Returns
    timings, or None if skipped.
    """
    outpaths = {tag: output_path(spec, tag, src) for tag in RESOLUTIONS}
    if not OVERWRITE and all(p.exists() for p in outpaths.values()):
        return None

    first, last = years
    label = str(first) if first == last else f"{first}-{last}"
    print(f"    {label}: load+regrid...", end="", flush=True)
    # Per-year file: read whole. Multi-year file: dask chunks = on-disk chunks.
    da_all = lo.load_obs(spec, var, slice(str(first), str(last)), version=version, freq=freq,
                         chunks=None if first == last else {})
    sizes = np.array(da_all.chunksizes.get("time", (da_all.sizes["time"],)))
    stops = np.cumsum(sizes)
    t = {"load": 0.0, "regrid": 0.0, "save": 0.0}
    pieces: dict[str, list[xr.DataArray]] = {tag: [] for tag in RESOLUTIONS}
    for i0, i1 in zip(stops - sizes, stops):
        t0 = time.perf_counter()
        da = da_all.isel(time=slice(i0, i1)).load()
        t["load"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        for tag, res in RESOLUTIONS.items():
            pieces[tag].append(regrid(da, rg.conservative_regridder(da, res)))
        t["regrid"] += time.perf_counter() - t0
        if first != last:
            print(f" {da.time.dt.year.values[-1]}", end="", flush=True)
        del da
    print(f" load {t['load']:.1f}s", end="", flush=True)
    if not any(pieces.values()):
        raise ValueError(f"{src} has no time steps in {label}")

    for tag in RESOLUTIONS:
        out = xr.concat(pieces[tag], dim="time")
        out.attrs["src_dims"] = da_all.dims
        out.attrs["src_shape"] = da_all.shape
        out.attrs["src_dlat_deg"], out.attrs["src_dlon_deg"] = rg.approx_resolution(da_all)
        out.attrs["tgt_dlat_deg"], out.attrs["tgt_dlon_deg"] = rg.approx_resolution(out)
        out.attrs["source_file"] = str(src)
        out.attrs["regrid_method"] = "xesmf conservative, skipna=True, unmapped_to_nan=True"
        out.attrs["na_thres"] = NA_THRES
        out.attrs["regrid_script"] = os.path.basename(__file__)
        out.attrs["regrid_date"] = dt.now().strftime("%Y-%m-%d %H:%M:%S")

        t0 = time.perf_counter()
        save(out, var, src, outpaths[tag])
        t["save"] += time.perf_counter() - t0
        print(f" | {tag} {out.shape}", end="", flush=True)

    print(f" | regrid {t['regrid']:.1f}s save {t['save']:.1f}s")
    del da_all, pieces
    gc.collect()
    return t


def main() -> None:
    run_t0 = time.perf_counter()
    for tag, res in RESOLUTIONS.items():
        grid = rg.target_grid(res)
        print(f"Target grid {tag}: nlat={grid.sizes['lat']}, nlon={grid.sizes['lon']}")
    print(f"Output root: {REGRID_ROOT}\n")

    # (dataset, version, freq, var) -> {"files", "skipped", "load", "regrid", "save"}
    timings: dict[tuple[str, str, str, str], dict[str, float]] = {}

    for name, sel in SELECTION.items():
        spec = lo.get_dataset(name)
        for version in sel.get("versions") or spec.versions:
            for freq in sel.get("freqs") or tuple(spec.freqs):
                if freq not in spec.freqs:
                    continue
                variables = sel.get("vars") or lo.list_variables(spec, version, freq)
                for var in variables:
                    files = lo.list_files(spec, var, version, freq)
                    if YEARS is not None:
                        files = {src: (y0, y1) for src, (y0, y1) in files.items()
                                 if any(y0 <= y <= y1 for y in YEARS)}
                    if not files:
                        continue
                    y0, y1 = min(y for y, _ in files.values()), max(y for _, y in files.values())
                    print(f"==== {spec.name} {version} {freq} {var} ({y0}-{y1}, {len(files)} files) ====")
                    tot = {"files": 0, "skipped": 0, "load": 0.0, "regrid": 0.0, "save": 0.0}
                    for src, years in files.items():
                        t = regrid_file(spec, var, src, years, version, freq)
                        if t is None:
                            tot["skipped"] += 1
                            continue
                        tot["files"] += 1
                        for k, v in t.items():
                            tot[k] += v
                    if tot["skipped"]:
                        print(f"    skipped {tot['skipped']} files with existing output")
                    timings[(spec.name, version, freq, var)] = tot

    run_elapsed = time.perf_counter() - run_t0

    print("\n==== Timing summary ====")
    print(f"{'dataset':<8}{'version':<13}{'freq':<9}{'var':<6}{'files':>6}{'skip':>6}"
          f"{'load [s]':>10}{'regrid [s]':>12}{'save [s]':>10}")
    total = {"load": 0.0, "regrid": 0.0, "save": 0.0}
    for (name, version, freq, var), t in timings.items():
        print(f"{name:<8}{version:<13}{freq:<9}{var:<6}{t['files']:>6}{t['skipped']:>6}"
              f"{t['load']:>10.1f}{t['regrid']:>12.1f}{t['save']:>10.1f}")
        for k in total:
            total[k] += t[k]
    print(f"\nTotal load time:        {total['load']:.2f}s")
    print(f"Total regridding time:  {total['regrid']:.2f}s")
    print(f"Total save time:        {total['save']:.2f}s")
    print(f"Total run time:         {run_elapsed:.2f}s")


if __name__ == "__main__":
    main()
