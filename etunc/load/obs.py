"""
etunc.load.obs
==============
Load gridded observational products stored as NetCDF files per variable and
year (GLEAM v4.3, PML-V2.2) or per variable and multi-year period (SiTHv2) into
a single DataArray over a time period.

Each product is described by an ``ObsDataset`` in ``DATASETS``: its root
directory, a file path template, its versions and temporal frequencies, and
the names of its native lat/lon coordinates. ``load_obs`` opens only the files
for the requested years, puts them on one grid, and returns a lazy (dask)
DataArray with dims (time, lat, lon), latitude ascending and longitude in
[-180, 180).

To add a product, register another ``ObsDataset`` (see ``register_dataset``).
The path template may use the fields {version}, {freq}, {freq_dir}, {freq_tag},
{var} and either {year} (one file per year) or {year_start} and {year_end}
(one file per period, e.g. A1982_2022). A template without any year field (one
file holding every year) also works. Only files overlapping the requested
years are opened, and the time selection is applied after opening.

Every function also takes ``res``: "native" (default) reads the original
files, "0.5" or "1" the copies regridded by ``regrid_obs.py`` to 0.5 or 1 deg,
stored under ``REGRID_ROOT/<dataset dir>/<0.5deg|1deg>`` with the same layout
as the dataset dir (see ``at_resolution``).

Example
-------
>>> import etunc.load.obs as lo
>>> et = lo.load_gleam("E", slice("1995-01", "2014-12"))                 # v4.3a, monthly
>>> et = lo.load_pml("ET", slice("1995", "2014"), version="V2.2c")
>>> et = lo.load_obs("pml", "ET", slice("2003", "2005"), version="V2.2a-MODIS", freq="8-day")
>>> et = lo.load_sith("ET", slice("1995", "2014"), freq="yearly")
>>> et = lo.load_gleam("E", slice("1995", "2014"), res="1")             # regridded to 1 deg
>>> lo.list_variables("gleam"), lo.list_years("pml", "ET", version="V2.2b")
>>> lo.list_files("sith", "ET")   # {path: (1982, 2022)}
>>> et_wm2 = units.latent_heat_to_wm2(units.accumulation_to_flux(et))  # import etunc.units as units

Notes
-----
- Monthly and yearly files of all products hold period *totals* (mm/month,
  mm/year). PML 8-day and half-month files hold daily rates (mm/day).
  ``etunc.units.accumulation_to_flux`` converts any of these to kg m-2 s-1.
- PML files from the AVHRR era (V2.2b, and V2.2c before 2001) have
  coordinates offset by up to ~1e-5 deg from the MODIS-era files. Coordinates
  are rounded to ``ObsDataset.coord_decimals`` so every year shares one grid
  exactly; files that still disagree raise an error instead of being
  outer-joined.
- GLEAM v4.3b files label their monthly and yearly totals "mm.day-1"; the
  loader relabels them mm.month-1 / mm.year-1 (yearly = sum of monthly).
- GLEAM v4.3b stores the ocean as 0 instead of NaN. The loader applies the
  NaN pattern of the matching v4.3a file (same variable, frequency and year),
  so loading v4.3b needs the v4.3a files for the same years.
- SiTHv2 files store int32 values x100 with fill -999 in non-CF attributes
  ("scale factor", global "Fill Value"), dims (time, lon, lat) and float32
  coordinates; the loader decodes them to float32 with NaN, transposes to
  (time, lat, lon) and makes lat/lon float64.
- Time stamps are kept as stored: GLEAM and SiTHv2 mark the end of each period
  (1980-01-31, 1980-12-31), PML the start (1980-01-01).
"""

from __future__ import annotations

import gc
import os
import re
import string
import time
import warnings
from datetime import datetime as dt
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import xarray as xr

import etunc.config as config
from etunc.grid import NA_THRES, RESOLUTIONS, approx_resolution, conservative_regridder, regrid_with_na_thres

OBS_ROOT = config.OBS_ROOT
REGRID_ROOT = config.OBS_REGRID_ROOT
RES_DIRS = {"0.5": "0.5deg", "1": "1deg"}  # res -> directory under REGRID_ROOT/<dataset dir>


# ------------------------------------------------------------------
# Dataset registry
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ObsDataset:
    """
    Layout of one observational product on disk.

    Parameters
    ----------
    name : registry key (case-insensitive).
    root : directory that `template` is relative to.
    template : file path relative to `root`, with fields {version}, {freq},
        {freq_dir}, {freq_tag}, {var} and optionally {year} or the pair
        {year_start}/{year_end}.
    versions : available versions; the first is the default.
    freqs : frequency (as it appears in {freq}) -> tag substituted for
        {freq_tag}. Use "" when the template has no {freq_tag}.
    freq_dirs : frequency -> text substituted for {freq_dir} (e.g. a
        directory named "Monthly"); defaults to the frequency itself.
    lat_name, lon_name : native coordinate names, renamed to lat/lon.
    coord_decimals : round lat/lon to this many decimals so files share one
        grid exactly; None leaves them as stored.
    preprocess : optional dataset-specific fix applied to each file before
        the variable is selected and the grid is standardized, called as
        ``preprocess(ds, var=..., version=..., freq=...)``.
    """

    name: str
    root: Path
    template: str
    versions: tuple[str, ...]
    freqs: Mapping[str, str]
    freq_dirs: Mapping[str, str] | None = None
    lat_name: str = "lat"
    lon_name: str = "lon"
    coord_decimals: int | None = 4
    preprocess: Callable[..., xr.Dataset] | None = None


DATASETS: dict[str, ObsDataset] = {}


def register_dataset(spec: ObsDataset) -> ObsDataset:
    """Add (or replace) a product in `DATASETS`."""
    DATASETS[spec.name.lower()] = spec
    return spec


def _fix_gleam_units(ds: xr.Dataset, var: str, version: str, freq: str) -> xr.Dataset:
    """v4.3b labels monthly and yearly totals "mm.day-1"; relabel to match v4.3a."""
    if ds[var].attrs.get("units") == "mm.day-1":
        ds[var].attrs["units"] = {"monthly": "mm.month-1", "yearly": "mm.year-1"}[freq]
    return ds


def _mask_gleam_like_v43a(ds: xr.Dataset, var: str, version: str, freq: str) -> xr.Dataset:
    """
    v4.3b stores the ocean as 0 instead of NaN; apply the NaN pattern of the
    matching v4.3a file (same variable, frequency and year). Both versions
    share one layout, so this also works for the regridded files.
    """
    if version != "v4.3b":
        return ds
    src = Path(ds[var].encoding.get("source") or ds.encoding.get("source", ""))
    # <root>/<version>/<freq>/<var>/<var>_<year>_GLEAM_<version>_<freq_tag>.nc
    if len(src.parents) < 4 or src.parents[2].name != version:
        raise ValueError(f"Cannot find the v4.3a file matching GLEAM {version} file {str(src)!r}")
    ref_path = (src.parents[3] / "v4.3a" / src.parents[1].name / src.parent.name
                / src.name.replace(f"_{version}_", "_v4.3a_"))
    if not ref_path.exists():
        raise FileNotFoundError(f"{ref_path} is needed for the ocean mask of {src}")
    ref = xr.open_dataset(ref_path, chunks={})[var]
    try:
        da, ref = xr.align(ds[var], ref, join="exact")
    except ValueError as err:
        raise ValueError(f"{src} and {ref_path} do not share time/lat/lon coordinates") from err
    with xr.set_options(keep_attrs=True):
        return ds.assign({var: da.where(ref.notnull())})


def _preprocess_gleam(ds: xr.Dataset, var: str, version: str, freq: str) -> xr.Dataset:
    ds = _fix_gleam_units(ds, var, version, freq)
    return _mask_gleam_like_v43a(ds, var, version, freq)


# GLEAM v4.3 (0.1 deg, 1980-2025 for v4.3a, 2003-2025 for v4.3b)
register_dataset(ObsDataset(
    name="gleam",
    root=OBS_ROOT / "gleam-v4.3",
    template="{version}/{freq}/{var}/{var}_{year}_GLEAM_{version}_{freq_tag}.nc",
    versions=("v4.3a", "v4.3b"),
    freqs={"monthly": "MO", "yearly": "YR"},
    preprocess=_preprocess_gleam,
))

# PML-V2.2 (0.1 deg, 60S-90N). V2.2c: 1982-2025, V2.2b: 1982-2020,
# V2.2a-MODIS: 2000-2024, V2.2a-VIIRS: 2012-2025. 8-day is only for V2.2a-*,
# half-month only for V2.2b/c.
register_dataset(ObsDataset(
    name="pml",
    root=OBS_ROOT / "pml-v2.2",
    template="{version}/{freq}/{var}/PML-{version}_{var}_{year}.nc",
    versions=("V2.2c", "V2.2b", "V2.2a-MODIS", "V2.2a-VIIRS"),
    freqs={"monthly": "", "yearly": "", "8-day": "", "half-month": ""},
    lat_name="latitude",
    lon_name="longitude",
))

def _decode_sith(ds: xr.Dataset, var: str, version: str, freq: str) -> xr.Dataset:
    """Apply SiTHv2's non-CF "scale factor" and "Fill Value" and put dims in (time, lat, lon) order."""
    da = ds[var]
    attrs = dict(da.attrs)
    scale = float(attrs.pop("scale factor", 1))
    fill = ds.attrs.get("Fill Value")
    da = da.astype("float32")
    if fill is not None:
        da = da.where(da != float(fill))
    da = (da / np.float32(scale)).transpose("time", "lat", "lon")
    da.attrs = attrs
    return ds.assign({var: da})


# SiTHv2 (0.1 deg, 1982-2022), one file per variable for the whole period
register_dataset(ObsDataset(
    name="sith",
    root=OBS_ROOT / "sith-v2",
    template="{freq_dir}/{var}.SiTH{version}.A{year_start}_{year_end}.{freq_tag}.nc",
    versions=("v2",),
    freqs={"monthly": "M", "yearly": "Y"},
    freq_dirs={"monthly": "Monthly", "yearly": "Yearly"},
    preprocess=_decode_sith,
))


def get_dataset(dataset: str | ObsDataset) -> ObsDataset:
    """Look up a registered product by name, or pass an `ObsDataset` through."""
    if isinstance(dataset, ObsDataset):
        return dataset
    try:
        return DATASETS[dataset.lower()]
    except KeyError:
        raise KeyError(f"Unknown dataset {dataset!r}; registered: {sorted(DATASETS)}") from None


def at_resolution(dataset: str | ObsDataset, res: str = "native") -> ObsDataset:
    """
    The product at resolution `res`: "native" returns it unchanged; "0.5" or
    "1" returns a copy rooted at its regridded tree, ``REGRID_ROOT/<root
    dir name>/<0.5deg|1deg>``, whose files have lat/lon coordinates.
    """
    spec = get_dataset(dataset)
    if res == "native":
        return spec
    if res not in RES_DIRS:
        raise ValueError(f"Unknown res {res!r}; options: {('native', *RES_DIRS)}")
    return replace(spec, root=REGRID_ROOT / spec.root.name / RES_DIRS[res], lat_name="lat", lon_name="lon")


def _resolve(
    dataset: str | ObsDataset, version: str | None, freq: str, res: str = "native"
) -> tuple[ObsDataset, str]:
    spec = at_resolution(dataset, res)
    version = spec.versions[0] if version is None else version
    if version not in spec.versions:
        raise ValueError(f"{spec.name}: unknown version {version!r}; options: {spec.versions}")
    if freq not in spec.freqs:
        raise ValueError(f"{spec.name}: unknown freq {freq!r}; options: {tuple(spec.freqs)}")
    return spec, version


def _fixed_fields(spec: ObsDataset, version: str, freq: str) -> dict[str, str]:
    """Template fields set by the version and frequency."""
    freq_dir = freq if spec.freq_dirs is None else spec.freq_dirs[freq]
    return {"version": version, "freq": freq, "freq_dir": freq_dir, "freq_tag": spec.freqs[freq]}


# ------------------------------------------------------------------
# File discovery
# ------------------------------------------------------------------

def _scan(spec: ObsDataset, **fields: str) -> list[tuple[dict[str, str], Path]]:
    """
    Files matching `spec.template` with `fields` filled in, and the values of the
    remaining template fields parsed from each path. Unfilled fields match any
    path component text (year fields match four digits).
    """
    glob_parts, regex_parts, seen = [], [], set()
    for literal, field, _, _ in string.Formatter().parse(spec.template):
        glob_parts.append(literal)
        regex_parts.append(re.escape(literal))
        if field is None:
            continue
        if field in fields:
            glob_parts.append(fields[field])
            regex_parts.append(re.escape(fields[field]))
        else:
            glob_parts.append("*")
            if field in seen:
                regex_parts.append(f"(?P={field})")
            else:
                regex_parts.append(rf"(?P<{field}>\d{{4}})" if field in _YEAR_FIELDS else rf"(?P<{field}>[^/]+)")
                seen.add(field)
    regex = re.compile("".join(regex_parts))

    out = []
    for path in sorted(spec.root.glob("".join(glob_parts))):
        m = regex.fullmatch(path.relative_to(spec.root).as_posix())
        if m:
            out.append((m.groupdict(), path))
    return out


_YEAR_FIELDS = ("year", "year_start", "year_end")


def _has_year(spec: ObsDataset) -> bool:
    return any(f in _YEAR_FIELDS for _, f, _, _ in string.Formatter().parse(spec.template))


def _file_years(groups: Mapping[str, str]) -> tuple[int, int]:
    """First and last year covered by a file, from its parsed template fields."""
    if "year" in groups:
        return int(groups["year"]), int(groups["year"])
    return int(groups["year_start"]), int(groups["year_end"])


def list_variables(
    dataset: str | ObsDataset, version: str | None = None, freq: str = "monthly", res: str = "native"
) -> list[str]:
    """Variables with at least one file for this version, frequency and resolution."""
    spec, version = _resolve(dataset, version, freq, res)
    found = _scan(spec, **_fixed_fields(spec, version, freq))
    return sorted({groups["var"] for groups, _ in found})


def list_files(
    dataset: str | ObsDataset, var: str, version: str | None = None, freq: str = "monthly",
    res: str = "native",
) -> dict[Path, tuple[int, int]]:
    """Every file for this variable, version, frequency and resolution -> (first, last) year it covers, sorted by year."""
    spec, version = _resolve(dataset, version, freq, res)
    if not _has_year(spec):
        raise ValueError(f"{spec.name}: template has no year field")
    found = _scan(spec, **_fixed_fields(spec, version, freq), var=var)
    return dict(sorted(((path, _file_years(groups)) for groups, path in found), key=lambda item: item[1]))


def list_years(
    dataset: str | ObsDataset, var: str, version: str | None = None, freq: str = "monthly",
    res: str = "native",
) -> list[int]:
    """Years covered by a file for this variable, version, frequency and resolution."""
    spans = list_files(dataset, var, version, freq, res).values()
    return sorted({y for first, last in spans for y in range(first, last + 1)})


def _year_bounds(time_slice: slice) -> tuple[int | None, int | None]:
    """First and last calendar year touched by a time slice (None if open-ended)."""
    def year(t):
        return None if t is None else pd.Timestamp(str(t)).year
    return year(time_slice.start), year(time_slice.stop)


def find_files(
    dataset: str | ObsDataset,
    var: str,
    time_slice: slice = slice(None, None),
    *,
    version: str | None = None,
    freq: str = "monthly",
    res: str = "native",
) -> list[Path]:
    """
    Files for `var` overlapping the years of `time_slice`, sorted by first year.
    Warns about years missing inside the requested (or, if open-ended, found)
    range.
    """
    spec, version = _resolve(dataset, version, freq, res)
    if not _has_year(spec):
        return [path for _, path in _scan(spec, **_fixed_fields(spec, version, freq), var=var)]

    y0, y1 = _year_bounds(time_slice)
    selected = {
        path: (first, last) for path, (first, last) in list_files(spec, var, version, freq).items()
        if (y0 is None or last >= y0) and (y1 is None or first <= y1)
    }
    if selected:
        lo = min(first for first, _ in selected.values()) if y0 is None else y0
        hi = max(last for _, last in selected.values()) if y1 is None else y1
        covered = {y for first, last in selected.values() for y in range(first, last + 1)}
        missing = sorted(set(range(lo, hi + 1)) - covered)
        if missing:
            warnings.warn(f"{spec.name} {version} {freq} {var} ({res}): no files for years {missing}")
    return list(selected)


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def standardize_grid(ds: xr.Dataset, spec: ObsDataset) -> xr.Dataset:
    """Rename coords to lat/lon, wrap lon to [-180, 180), round coords, sort both ascending."""
    ds = ds.rename({k: v for k, v in ((spec.lat_name, "lat"), (spec.lon_name, "lon")) if k != v})
    # float64 so rounding gives one exact grid (SiTHv2 stores float32 coords)
    lon = ((ds.lon.values.astype("float64") + 180) % 360) - 180
    lat = ds.lat.values.astype("float64")
    if spec.coord_decimals is not None:
        lat, lon = np.round(lat, spec.coord_decimals), np.round(lon, spec.coord_decimals)
    ds = ds.assign_coords(lat=("lat", lat, ds.lat.attrs), lon=("lon", lon, ds.lon.attrs))
    for dim in ("lat", "lon"):
        index = ds.indexes[dim]
        if index.is_monotonic_decreasing:
            ds = ds.isel({dim: slice(None, None, -1)})  # cheap reverse instead of a dask gather
        elif not index.is_monotonic_increasing:
            ds = ds.sortby(dim)
    return ds


def load_obs(
    dataset: str | ObsDataset,
    var: str,
    time_slice: slice = slice(None, None),
    *,
    version: str | None = None,
    freq: str = "monthly",
    res: str = "native",
    lat_bnds: slice | None = None,
    chunks: str | int | Mapping | None = "auto",
    parallel: bool = False,
) -> xr.DataArray:
    """
    Load one variable of an observational product as a single lazy DataArray.

    Parameters
    ----------
    dataset : registered name ("gleam", "pml", "sith", ...) or an `ObsDataset`.
    var : variable name as used in the directory and file names (e.g. "E", "ET").
    time_slice : passed to ``.sel(time=...)``; only files for the years it
        touches are opened.
    version : product version; defaults to the first in `ObsDataset.versions`.
    freq : temporal frequency, a key of `ObsDataset.freqs`.
    res : "native", or "0.5" / "1" for the files regridded to 0.5 / 1 deg.
    lat_bnds : optional latitude slice applied after sorting lat ascending,
        e.g. ``etunc.config.LAT_BNDS``.
    chunks : dask chunks for `xr.open_mfdataset`; dict keys may use lat/lon.
    parallel : open files in parallel with dask.

    Returns
    -------
    DataArray with dims (time, lat, lon) and attrs ``dataset``, ``version``,
    ``frequency`` and ``res`` added to the file's variable attrs.
    """
    spec, version = _resolve(dataset, version, freq, res)
    files = find_files(dataset, var, time_slice, version=version, freq=freq, res=res)
    if not files:
        y0, y1 = _year_bounds(time_slice)
        raise FileNotFoundError(
            f"No {spec.name} {version} {freq} ({res}) files for {var!r} in years {y0}-{y1} under "
            f"{spec.root}. Available variables: {list_variables(dataset, version, freq, res)}"
        )

    def preprocess(ds: xr.Dataset) -> xr.Dataset:
        if spec.preprocess is not None:
            ds = spec.preprocess(ds, var=var, version=version, freq=freq)
        if var not in ds:
            raise KeyError(f"{var!r} not in {ds.encoding.get('source')}; data vars: {list(ds.data_vars)}")
        return standardize_grid(ds[[var]], spec)

    if isinstance(chunks, Mapping):
        native = {"lat": spec.lat_name, "lon": spec.lon_name}
        chunks = {native.get(k, k): v for k, v in chunks.items()}

    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        preprocess=preprocess,
        chunks=chunks,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
        parallel=parallel,
    )
    da = ds[var].sel(time=time_slice)
    if lat_bnds is not None:
        da = da.sel(lat=lat_bnds)
    da.attrs.update(dataset=spec.name, version=version, frequency=freq, res=res)
    return da


def load_gleam(var: str, time_slice: slice = slice(None, None), **kwargs) -> xr.DataArray:
    """GLEAM v4.3 variable (E, Et, Ec, Es, Ei, Eb, Ew, Ep); see `load_obs` for kwargs."""
    return load_obs("gleam", var, time_slice, **kwargs)


def load_pml(var: str, time_slice: slice = slice(None, None), **kwargs) -> xr.DataArray:
    """PML-V2.2 variable (ET, Ec, Es, Ei, E, Ew, PET, GPP); see `load_obs` for kwargs."""
    return load_obs("pml", var, time_slice, **kwargs)


def load_sith(var: str, time_slice: slice = slice(None, None), **kwargs) -> xr.DataArray:
    """SiTHv2 variable (ET, Tr, Es, Ei, En); see `load_obs` for kwargs."""
    return load_obs("sith", var, time_slice, **kwargs)


# ------------------------------------------------------------------
# Writing regridded files (regrid_obs.py)
# ------------------------------------------------------------------

def output_path(spec: ObsDataset, res_tag: str, src: Path) -> Path:
    """Regridded file of `src` at `res_tag` ("0.5deg", "1deg"): the tree that `at_resolution` reads."""
    return REGRID_ROOT / spec.root.name / res_tag / src.relative_to(spec.root)


def _save(da: xr.DataArray, var: str, src: Path, outpath: Path, complevel: int) -> None:
    """Write `da` next to `outpath` as .tmp, then move it into place."""
    with xr.open_dataset(src) as ds_src:
        global_attrs = dict(ds_src.attrs)
    da.encoding = {}
    ds = xr.Dataset({var: da}, attrs=global_attrs)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = outpath.with_name(outpath.name + ".tmp")
    ds.to_netcdf(tmp, encoding={var: {"zlib": True, "complevel": complevel}})
    os.replace(tmp, outpath)


def regrid_file(
    spec: ObsDataset, var: str, src: Path, years: tuple[int, int], version: str, freq: str,
    *,
    resolutions: Mapping[str, float] = RESOLUTIONS,
    na_thres: float = NA_THRES,
    overwrite: bool = True,
    complevel: int = 4,
    script: str = "regrid_obs.py",
) -> dict[str, float] | None:
    """
    Conservatively regrid one source file, covering `years` (first, last), to
    every grid in `resolutions` ({tag: spacing}), and write each to
    `output_path` (`regrid_with_na_thres`). A multi-year file is read in
    blocks of its on-disk time chunks. Existing outputs are skipped unless
    `overwrite`. `script` is recorded in the "regrid_script" attr. Returns
    timings, or None if skipped.
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
    sizes = np.array(da_all.chunksizes.get("time", (da_all.sizes["time"],)))
    stops = np.cumsum(sizes)
    t = {"load": 0.0, "regrid": 0.0, "save": 0.0}
    pieces: dict[str, list[xr.DataArray]] = {tag: [] for tag in resolutions}
    for i0, i1 in zip(stops - sizes, stops):
        t0 = time.perf_counter()
        da = da_all.isel(time=slice(i0, i1)).load()
        t["load"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        for tag, res in resolutions.items():
            pieces[tag].append(regrid_with_na_thres(da, conservative_regridder(da, res), na_thres))
        t["regrid"] += time.perf_counter() - t0
        if first != last:
            print(f" {da.time.dt.year.values[-1]}", end="", flush=True)
        del da
    print(f" load {t['load']:.1f}s", end="", flush=True)
    if not any(pieces.values()):
        raise ValueError(f"{src} has no time steps in {label}")

    for tag in resolutions:
        out = xr.concat(pieces[tag], dim="time")
        out.attrs["src_dims"] = da_all.dims
        out.attrs["src_shape"] = da_all.shape
        out.attrs["src_dlat_deg"], out.attrs["src_dlon_deg"] = approx_resolution(da_all)
        out.attrs["tgt_dlat_deg"], out.attrs["tgt_dlon_deg"] = approx_resolution(out)
        out.attrs["source_file"] = str(src)
        out.attrs["regrid_method"] = "xesmf conservative, skipna=True, unmapped_to_nan=True"
        out.attrs["na_thres"] = na_thres
        out.attrs["regrid_script"] = script
        out.attrs["regrid_date"] = dt.now().strftime("%Y-%m-%d %H:%M:%S")

        t0 = time.perf_counter()
        _save(out, var, src, outpaths[tag], complevel)
        t["save"] += time.perf_counter() - t0
        print(f" | {tag} {out.shape}", end="", flush=True)

    print(f" | regrid {t['regrid']:.1f}s save {t['save']:.1f}s")
    del da_all, pieces
    gc.collect()
    return t
