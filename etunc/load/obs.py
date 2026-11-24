"""
etunc.load.obs
==============
Load gridded observational products stored as NetCDF files per variable and
year (GLEAM v4.3, PML-V2.2, NOAA NCEI AVHRR and VIIRS LAI, FLUXCOM-X-BASE,
GLDAS), per variable and multi-year period (SiTHv2), per variable and month
(CLARA-A3) or per month with every variable (GPCP v3), or as one GeoTIFF per
composite (GIMMS LAI4g, half-monthly; HiQ-LAI, 8-day; GLOBMAP LAI, half-monthly
to 2000 and 8-day from 2001) or one HDF4 file per composite (GLASS LAI, 8-day)
into a single DataArray
over a time period. ``load_ncei_lai`` joins the AVHRR and VIIRS LAI records,
with the sensor of each month as a coord.

Each product is described by an ``ObsDataset`` in ``DATASETS``: its root
directory, a file path template, its versions and temporal frequencies, and
the names of its native lat/lon coordinates. ``load_obs`` opens only the files
for the requested years, puts them on one grid, and returns a lazy (dask)
DataArray with dims (time, lat, lon), latitude ascending and longitude in
[-180, 180).

To add a product, register another ``ObsDataset`` (see ``register_dataset``).
The path template may use the fields {version}, {freq}, {freq_dir}, {freq_tag},
{var} and either {year} (one file per year) or {year_start} and {year_end}
(one file per period, e.g. A1982_2022), plus {month} with {year} (one file per
month), or {half} (01 or 02) or {doy} (day of year) for composites shorter
than a month. Any other field (e.g. {standard_name}) matches any text in that
part of the path. A template without any year field (one file holding every
year) also works, and so does one without {var} (every variable in each file,
as GPCP). Only files overlapping the requested years are opened, and the time
selection is applied after opening. Files without a time axis (the GeoTIFFs,
opened with ``engine="rasterio"`` of rioxarray) get their time stamp from
their name (``ObsDataset.time_from_fields``).

``load_obs_regridded`` reads the native files and conservatively regrids each time step in memory onto
``etunc.grid.target_grid(res)`` (used by make_mask.py).

The binning pipeline (scripts/prep_obs.py) reads the native monthly files
with ``load_obs`` (step 1) and converts, averages, masks and regrids them in
the script. No regridded copies are saved.

Example
-------
>>> import etunc.load.obs as lo
>>> et = lo.load_obs("gleam", "E", slice("1995-01", "2014-12"))          # v4.3a, monthly
>>> et = lo.load_obs("pml", "ET", slice("1995", "2014"), version="V2.2c")
>>> et = lo.load_obs("pml", "ET", slice("2003", "2005"), version="V2.2a-MODIS", freq="8-day")
>>> et = lo.load_obs("sith", "ET", slice("1995", "2014"), freq="yearly")
>>> rn = lo.load_obs("clara", "SRB", slice("1995", "2014"))             # W m-2, monthly means
>>> lai = lo.load_obs("avhrr", "lai", slice("2001", "2013"))            # m2 m-2, monthly means
>>> lai = lo.load_ncei_lai(slice("2001", "2014"))                        # AVHRR to 2013, VIIRS in 2014
>>> lai = lo.load_obs("gimms-lai4g", "lai", slice("2001", "2014"), freq="half-month")  # m2 m-2
>>> lai = lo.load_obs("gimms-lai4g-avhrr", "lai", slice("2001", "2014"), freq="half-month")  # AVHRR only
>>> lai = lo.load_obs("hiq-lai", "lai", slice("2001", "2014"), freq="8-day")  # m2 m-2
>>> lai = lo.load_obs("globmap-lai", "lai", slice("2001", "2014"), freq="8-day")  # MODIS years
>>> lai = lo.load_obs("globmap-lai", "lai", slice("1982", "2000"), freq="half-month")  # AVHRR years
>>> lai = lo.load_obs("glass-lai", "lai", slice("2001", "2014"), freq="8-day")  # m2 m-2
>>> et = lo.load_obs("fluxcom-x", "ET", slice("2001", "2014"))          # mm hr-1, monthly means
>>> et = lo.load_obs("gldas", "Evap_tavg", slice("2001", "2014"), version="2.0")  # kg m-2 s-1
>>> pr = lo.load_obs("gpcp", "sat_gauge_precip", slice("2001", "2014"))  # v3.3, mm/day, monthly means
>>> et = lo.load_obs_regridded("gleam", "E", slice("1995", "2014"), 0.5)  # regridded in memory
>>> lo.list_variables("gleam"), lo.list_years("pml", "ET", version="V2.2b")
>>> lo.list_files("sith", "ET")   # {path: (1982, 2022)}
>>> et_wm2 = units.latent_heat_to_wm2(units.accumulation_to_flux(et))  # import etunc.units as units

Notes
-----
- Monthly and yearly files of the ET products (GLEAM, PML, SiTHv2) hold period
  *totals* (mm/month, mm/year). PML 8-day and half-month files hold daily
  rates (mm/day). ``etunc.units.accumulation_to_flux`` converts any of these
  to kg m-2 s-1. CLARA-A3 files hold monthly mean fluxes in W m-2.
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
- CLARA-A3 shortwave (SIS, SNS, and so net radiation SRB) is missing in
  low-sun months (e.g. Nov-Feb at 45-70N; valid as 0 in polar night), and over
  much of the tropics and subtropics in Oct-Dec 2000 and Jan 1995. Longwave
  (SDL, SNL) is nearly complete. The loader leaves these gaps as NaN.
- NOAA NCEI LAI files (AVHRR v5.0, 1981-06 to 2013-12, and VIIRS v1.0,
  2014-01 to 2024-12) hold monthly means (units "1"), NaN over water and where
  there is no retrieval (more of the high latitudes in winter). Sep-Dec 1994
  are entirely NaN (no AVHRR data after the NOAA-11 failure). Both share one
  grid, whose lat/lon carry float32 noise (e.g. lon 0.0249939) that the
  rounding to ``coord_decimals`` removes. ``load_ncei_lai`` concatenates them
  without any adjustment at the 2014 switch of sensor and retrieval version.
- FLUXCOM-X-BASE ET (0.5 deg, 2001-2021) holds monthly means of the hourly
  rate (mm hr-1), NaN over the ocean; the files' static ``land_fraction`` is
  not read.
- GLDAS files are those of scripts/download/download_gldas.py: monthly means of
  Evap_tavg and Rainf_f_tavg (kg m-2 s-1, already fluxes) and Swnet_tavg and
  Lwnet_tavg (W m-2, positive downward), one file per band and year, on the
  native 0.25 deg grid (60S-90N), from Earth Engine, for GLDAS-2.0 Noah (to 2014), 2.1 Noah and 2.2 Catchment
  with GRACE data assimilation (from 2003-02; 2003-01 is NaN). The loader sets
  a month to NaN where it has fewer valid time steps than the full month
  (``n_steps`` < ``n_steps_full``), so every value is a complete-month mean.
- GPCP v3 (GPCPMON, 0.5 deg, v3.3 1983-2025, v3.2 1983-2023) has one file per
  month holding every variable (``sat_gauge_precip``: combined
  satellite-gauge precipitation, mm/day monthly mean), ocean included, with
  float32 coordinates. Each version is in its own folder (gpcp-v<version>).
- GIMMS LAI4g V1.2 (1/12 deg, half-monthly, 1982-2020) and HiQ-LAI v1 (5 km,
  8-day, 2000-02 to 2022) are the Zenodo zips of
  scripts/download/download_gimms_4g.py and download_hiq_lai.py, unpacked, in
  OBS_ROOT/gimms-lai4g-v1.2 and OBS_ROOT/hiq-lai-v1: GeoTIFFs of scaled
  integers (LAI times 1000 and 10) with fill values but no nodata or scale in
  their metadata, which ``_decode_geotiff_lai`` applies from the products'
  documentation. Only the LAI band is read, not the QC bands, and the variable
  is always "lai". GIMMS LAI4g has two records, each its own product:
  "gimms-lai4g" (AVHRR consolidated with the reprocessed MODIS LAI from 2004)
  and "gimms-lai4g-avhrr" (AVHRR only, to 2015). HiQ-LAI misses a few 8-day
  composites (MODIS Terra gaps, e.g. 2001 days 169 and 177), and its 5 km
  cells are nearest-neighbour samples of the 500 m product. Their lat/lon are
  the cell centres of the geotransform, not rounded.
- GLOBMAP LAI v3 (0.0727 deg, 62.8S-89.2N, 1981-07 to 2023) is the RAR
  archives of scripts/download/download_globmap.py, unpacked, in
  OBS_ROOT/globmap-lai-v3: one int16 GeoTIFF per composite, LAI times 100,
  decoded by ``_decode_geotiff_lai`` (NaN outside 0-1000: the -100 and -20
  fill of the AVHRR years). The AVHRR years (to 2000) are half-monthly, the
  MODIS years (from 2001) 8-day, each its own ``freq``. Water is stored as 0
  like bare ground (no fill), so the ocean reads as LAI 0 until masked to land.
- GLASS LAI V6.1 (MODIS, 0.05 deg, 8-day, 2000-02 to 2025) is the archive
  layout of scripts/download/download_glass.py in
  OBS_ROOT/glass/LAI/MODIS/0.05D: one HDF4 (HDF-EOS2) grid per composite,
  read by netCDF-C (``engine="netcdf4"``), uint8 LAI times 10 with fill 255
  (ocean, Antarctica, land without retrieval) as NaN. The files have no
  coordinate variables; ``_decode_glass_lai`` sets the cell centres of the
  global grid given in their StructMetadata.
- Time stamps are kept as stored: GLEAM and SiTHv2 mark the end of each period
  (1980-01-31, 1980-12-31), PML, CLARA-A3, FLUXCOM-X, GLDAS, GPCP, GIMMS LAI4g,
  HiQ-LAI, GLOBMAP and GLASS the start (1980-01-01; GIMMS and GLOBMAP's half months
  also 1980-01-16), NCEI LAI the
  middle (1981-06-15).
"""

from __future__ import annotations

import gc
import re
import string
import time
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from etunc.config import NA_THRESH, OBS_ROOT
from etunc.grid import (
    RESOLUTIONS, conservative_regridder, regrid_with_na_thresh,
)


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
        {year_start}/{year_end}, and with {year}: {month}, {half} (01 or 02,
        the half of the month) or {doy} (day of year).
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
    engine : xarray backend that opens the files; None lets xarray choose
        (NetCDF). "rasterio" (rioxarray) for GeoTIFFs.
    time_from_fields : for files without a time axis (one GeoTIFF per
        composite): builds each file's time stamp from the template fields
        parsed from its path, e.g. {"year": "2001", "month": "01", "half": "02"}.
    file_chunks : dask chunks of each file used in place of `load_obs`'s
        default "auto"; None keeps "auto". {} reads each file in its stored
        chunks (GLASS: one chunk per file, which "auto" would split and read
        several times).
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
    engine: str | None = None
    time_from_fields: Callable[[Mapping[str, str]], pd.Timestamp] | None = None
    file_chunks: Mapping | None = None


DATASETS: dict[str, ObsDataset] = {}


def register_dataset(spec: ObsDataset) -> ObsDataset:
    """Add (or replace) a product in `DATASETS`."""
    DATASETS[spec.name.lower()] = spec
    return spec


def _fix_gleam_units(ds: xr.Dataset, var: str, version: str, freq: str) -> xr.Dataset:
    """v4.3b labels monthly and yearly totals "mm.day-1"; relabel to match v4.3a."""
    # `accumulation_to_flux` keys on this label: left as mm.day-1, a monthly total would
    # be read as a daily rate and ET would come out ~30x (yearly ~365x) too large
    if ds[var].attrs.get("units") == "mm.day-1":
        ds[var].attrs["units"] = {"monthly": "mm.month-1", "yearly": "mm.year-1"}[freq]
    return ds


def _mask_gleam_like_v43a(ds: xr.Dataset, var: str, version: str, freq: str) -> xr.Dataset:
    """
    v4.3b stores the ocean as 0 instead of NaN; apply the NaN pattern of the
    matching v4.3a file (same variable, frequency and year).
    """
    if version != "v4.3b":
        return ds
    # Unmasked, ocean zeros would count as valid ET = 0 and pull down coastal cells when regridded
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
    # Unlike CF scale_factor, SiTHv2's "scale factor" is a divisor (stored = value x 100) to help with compression
    scale = float(attrs.pop("scale factor", 1))
    fill = ds.attrs.get("Fill Value")
    # Cast before masking: int32 cannot hold NaN
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

# CLARA-A3 surface radiation (EUMETSAT CM SAF, 0.25 deg, 1979-2020), one file per variable and month,
# e.g. surface_net_downward_radiative_flux/2000/SRBmm20000101000000319AVPOS01GL.nc. Variables (W m-2):
# SRB net radiation (= SNS + SNL), SNS/SNL net shortwave/longwave, SIS/SDL downwelling shortwave/longwave
register_dataset(ObsDataset(
    name="clara",
    root=OBS_ROOT / "clara-a3",
    # {standard_name} is the CF standard name directory of {var}, matched from the path
    template="{standard_name}/{year}/{var}{freq_tag}{year}{month}01000000319AVPOS01GL.nc",
    versions=("A3",),
    freqs={"monthly": "mm"},  # "mm": monthly mean
))

# NOAA NCEI LAI climate data record (obs4MIPs, 0.05 deg), published as one dataset per sensor: AVHRR LAI
# v5.0 (1981-06 to 2013-12) and VIIRS LAI v1.0 (2014-01 to 2024-12), on the same grid. One file per variable
# and year, in the obs4MIPs directory tree of ESGF (written by scripts/download/download_ncei_lai.py). The
# version is the ESGF publication date. `load_ncei_lai` joins the two records. "avhrr"
# is not the AVHRR LAI of ILAMB (etunc.load.ilamb, product "AVHRR", from Boston University); ILAMB's
# AVH15C1 is this AVHRR record averaged to 0.5 deg
register_dataset(ObsDataset(
    name="avhrr",
    root=OBS_ROOT / "avhrr-lai-v5",
    template=("data/obs4MIPs/NOAA-NCEI/NOAA-NCEI-LAI-AVHRR-5-0/{freq_tag}/{var}/gn/{version}/"
              "{var}_{freq_tag}_NOAA-NCEI-LAI-AVHRR-5-0_BE_gn_{year_start}{month_start}-{year_end}{month_end}.nc"),
    versions=("v20260110",),
    freqs={"monthly": "mon"},
))
register_dataset(ObsDataset(
    name="viirs",
    root=OBS_ROOT / "viirs-lai-v1",
    template=("data/obs4MIPs/NOAA-NCEI/NOAA-NCEI-LAI-VIIRS-1-0/{freq_tag}/{var}/gn/{version}/"
              "{var}_{freq_tag}_NOAA-NCEI-LAI-VIIRS-1-0_BE_gn_{year_start}{month_start}-{year_end}{month_end}.nc"),
    versions=("v20260110",),
    freqs={"monthly": "mon"},
))

# Sensors of the joined NOAA NCEI LAI record, in time order: (sensor label, registered dataset, version)
NCEI_LAI_SENSORS = (
    ("AVHRR LAI v5.0", "avhrr", "v20260110"),
    ("VIIRS LAI v1.0", "viirs", "v20260110"),
)

# FLUXCOM-X-BASE (Nelson et al. 2024, 0.5 deg, 2001-2021), from the ICOS Carbon Portal
# (scripts/download/download_fluxcom_x.py): one file per variable and year, ET in mm hr-1
register_dataset(ObsDataset(
    name="fluxcom-x",
    root=OBS_ROOT / "fluxcom-x",
    template="050_{freq}/{var}/{var}_{year}_050_{freq}.nc",
    versions=("X-BASE",),
    freqs={"monthly": ""},
))


def _mask_incomplete_gldas_months(ds: xr.Dataset, var: str, version: str, freq: str) -> xr.Dataset:
    """NaN where a monthly mean has fewer valid time steps than the month (n_steps < n_steps_full)."""
    # Written by download_gldas.py next to each monthly mean, so a partial month is not taken as a full one
    if "n_steps" not in ds:
        return ds
    with xr.set_options(keep_attrs=True):
        return ds.assign({var: ds[var].where(ds["n_steps"] >= ds["n_steps_full"])})


# GLDAS (0.25 deg, 60S-90N), monthly means of Evap_tavg, Rainf_f_tavg (kg m-2 s-1), Swnet_tavg and Lwnet_tavg
# (W m-2, positive downward) computed on Earth Engine by
# scripts/download/download_gldas.py: one folder per product version, one file per year.
# 2.1: Noah (2000 on), 2.0: Noah (to 2014), 2.2-grace-da: Catchment with GRACE data assimilation (2003-02 on)
register_dataset(ObsDataset(
    name="gldas",
    root=OBS_ROOT / "gldas",
    template="{version}/GLDAS_{version}_{var}_{freq}_{year}.nc",
    versions=("2.1", "2.0", "2.2-grace-da"),
    freqs={"monthly": ""},
    preprocess=_mask_incomplete_gldas_months,
))

# GPCP v3 monthly precipitation (GPCPMON, 0.5 deg) from NASA GES DISC (scripts/download/download_gpcp.py):
# one folder per version, one file per month holding every variable (no {var} in the file name), e.g.
# gpcp-v3.3/GPCPMON_L3_200101_V3.3.nc4. Precipitation is sat_gauge_precip (mm/day)
register_dataset(ObsDataset(
    name="gpcp",
    root=OBS_ROOT,
    template="gpcp-v{version}/GPCPMON_L3_{year}{month}_V{version}.nc4",
    versions=("3.3", "3.2"),
    freqs={"monthly": ""},
))


# GeoTIFF LAI products (GIMMS LAI4g, HiQ-LAI, GLOBMAP LAI): one single-time GeoTIFF per composite, unpacked from
# the Zenodo archives of scripts/download/download_gimms_4g.py, download_hiq_lai.py and download_globmap.py. Opened by rioxarray (engine "rasterio") as band_data (band, y, x); the time stamp comes from the file name

# HiQ-LAI headers declare their extra bands inconsistently; GDAL reads them as extra bands anyway, and rioxarray
# would warn about it for every file opened
warnings.filterwarnings("ignore", message="TIFFReadDirectory:Sum of Photometric", category=RuntimeWarning)

def _decode_geotiff_lai(
    ds: xr.Dataset, var: str, version: str, freq: str, *, scale: float, valid_max: int,
) -> xr.Dataset:
    """
    Band 1 as float32 LAI ("lai", m2 m-2): NaN below 0 or above `valid_max` (fill), times `scale`. The files
    have no nodata or scale in their metadata, so both come from the product's documentation. Other bands
    (QC) are not read.
    """
    raw = ds["band_data"].sel(band=1, drop=True)  # the stored integers as float32
    # Negative values are fill in the signed GLOBMAP files (-100, -20); the unsigned GIMMS and HiQ files have none
    lai = raw.where((raw >= 0) & (raw <= valid_max)) * np.float32(scale)
    # Named "lai" whatever `var` is, so load_obs raises for any other variable name
    return xr.Dataset({"lai": lai.assign_attrs(units="m2 m-2", long_name="leaf area index")}).drop_vars(
        "spatial_ref", errors="ignore")


def _half_month_start(fields: Mapping[str, str]) -> pd.Timestamp:
    """Start of a half month: day 1 for half 01 (days 1-15), day 16 for half 02 (16 to the end)."""
    return pd.Timestamp(int(fields["year"]), int(fields["month"]), 1 if fields["half"] == "01" else 16)


def _day_of_year_start(fields: Mapping[str, str]) -> pd.Timestamp:
    """First day of a composite named by its year and day of year."""
    return pd.Timestamp(int(fields["year"]), 1, 1) + pd.Timedelta(days=int(fields["doy"]) - 1)


# GIMMS LAI4g V1.2 (Cao et al. 2023, doi:10.5281/zenodo.8281930; 1/12 deg, half-monthly), one product per record:
# "gimms-lai4g" (1982-2020) is AVHRR consolidated with the reprocessed MODIS LAI from 2004 on (random forest
# fusion), "gimms-lai4g-avhrr" (1982-2015) is AVHRR alone. Files are e.g.
# GIMMS_LAI4g_AVHRR_MODIS_consolidated_2001_2010/<same>/GIMMS_LAI4g_V1.2_20010102.tif (<YYYY><MM><half>).
# Band 1 is LAI x 1000 as uint16 (valid 0-7000, fill 65535)
for _name, _folder, _prefix in (
    ("gimms-lai4g", "GIMMS_LAI4g_AVHRR_MODIS_consolidated", "GIMMS_LAI4g_V1.2_"),
    ("gimms-lai4g-avhrr", "GIMMS_LAI4g_AVHRR_solely", "GIMMS_LAI4g(AVHRR)_V1.2_"),
):
    register_dataset(ObsDataset(
        name=_name,
        root=OBS_ROOT / "gimms-lai4g-v1.2",
        # {years} is the zip's year range, matched from the path
        template=f"{_folder}_{{years}}/{_folder}_{{years}}/{_prefix}{{year}}{{month}}{{half}}.tif",
        versions=("V1.2",),
        freqs={"half-month": ""},
        lat_name="y",
        lon_name="x",
        coord_decimals=None,  # cell centres from the geotransform, the same in every file
        preprocess=partial(_decode_geotiff_lai, scale=0.001, valid_max=7000),
        engine="rasterio",
        time_from_fields=_half_month_start,
    ))

# HiQ-LAI Version 1 (Yan et al. 2023, doi:10.5281/zenodo.8296768; 5 km, 8-day, 2000-02 to 2022), reprocessed
# MODIS C6.1 LAI. Files are e.g. HiQ_LAI_WGS84_5km_8day_2000-2005/<same>/HiQ_LAI_WGS84_5km_8day_2001009.tif
# (<YYYY><DDD>, first day of the composite). Band 1 is LAI x 10 as uint8 (valid 0-100; fill 249-255, and a few
# stray values 101-248, set to NaN). The 5 km grid (0.0449 deg, 79.5S-84.6N) samples the 500 m product by
# nearest neighbour, so each cell is one 500 m pixel, not an area mean. A few composites are missing (MODIS
# Terra gaps, e.g. 2001 days 169 and 177)
register_dataset(ObsDataset(
    name="hiq-lai",
    root=OBS_ROOT / "hiq-lai-v1",
    template=("HiQ_LAI_WGS84_5km_8day_{years}/HiQ_LAI_WGS84_5km_8day_{years}/"
              "HiQ_LAI_WGS84_5km_8day_{year}{doy}.tif"),
    versions=("v1",),
    freqs={"8-day": ""},
    lat_name="y",
    lon_name="x",
    coord_decimals=None,
    preprocess=partial(_decode_geotiff_lai, scale=0.1, valid_max=100),
    engine="rasterio",
    time_from_fields=_day_of_year_start,
))

# GLOBMAP LAI Version 3 (Liu et al. 2012, doi:10.5281/zenodo.12698637; 0.0727 deg, 62.8S-89.2N), MODIS LAI
# (GLOBCARBON algorithm, MOD09A1 C6) from 2001, and AVHRR LAI back to 1981-07 from pixel-wise fits of GIMMS
# NDVI to it. Files are e.g. 2001-2005_TIFF/GlobMapLAIV3.A2001009.Global.LAI.tif (<YYYY><DDD>, first day of the
# composite), unpacked from the RAR archives of scripts/download/download_globmap.py. The AVHRR years
# (1981-1990_TIFF, 1991-2000_TIFF) are half-monthly, the MODIS years (2001-2005_TIFF, ...) 8-day, so `freq`
# picks the folders by the century of their first year ({freq_dir}). Band 1 is LAI x 100 as int16 (valid
# 0-1000); the AVHRR years also hold -100 (no retrieval, mostly water and bare ground) and -20 (a few dense
# forest pixels), set to NaN. Water is otherwise stored as 0, like bare ground, so the ocean reads as LAI 0
# until masked to land
register_dataset(ObsDataset(
    name="globmap-lai",
    root=OBS_ROOT / "globmap-lai-v3",
    template="{freq_dir}{years}_TIFF/GlobMapLAIV3.A{year}{doy}.Global.LAI.tif",
    versions=("v3",),
    freqs={"half-month": "", "8-day": ""},
    freq_dirs={"half-month": "19", "8-day": "20"},
    lat_name="y",
    lon_name="x",
    coord_decimals=None,  # cell centres from the geotransform, the same in every file
    preprocess=partial(_decode_geotiff_lai, scale=0.01, valid_max=1000),
    engine="rasterio",
    time_from_fields=_day_of_year_start,
))


def _decode_glass_lai(ds: xr.Dataset, var: str, version: str, freq: str) -> xr.Dataset:
    """
    LAI of a GLASS HDF-EOS grid file as float32 "lai" (m2 m-2) on cell-centre lat/lon. xarray has already
    applied the file's scale_factor (0.1) and _FillValue (255, ocean and no retrieval) as NaN.
    """
    lai = ds["LAI"]
    ny, nx = lai.shape
    # The grid has no coordinate variables: its StructMetadata gives a global GEO grid from (-180, 90) to
    # (180, -90) with values at cell centres (PixelRegistration=HDFE_CENTER)
    lat = 90 - 180 / ny * (np.arange(ny) + 0.5)
    lon = -180 + 360 / nx * (np.arange(nx) + 0.5)
    lai = lai.where(lai <= 10).astype("float32")  # valid range 0-100 before scaling
    lai = lai.assign_coords({lai.dims[0]: lat, lai.dims[1]: lon})
    # Named "lai" whatever `var` is, as the GeoTIFF LAI products
    return xr.Dataset({"lai": lai.assign_attrs(units="m2 m-2", long_name="leaf area index")})


# GLASS LAI (Global LAnd Surface Satellite, Beijing Normal University / HKU), MODIS 0.05 deg, 8-day,
# 2000-02 to 2025, Version 6.1, from https://glass.hku.hk/archive/LAI/MODIS/0.05D/ (scripts/download/
# download_glass.py, which keeps the archive's layout). Files are e.g.
# 2001/GLASS01B01.V61.A2001009.2025248.hdf (<YYYY><DDD> the first day of the composite, then the production
# date), HDF4 (HDF-EOS2) read by netCDF-C, with an XML sidecar each (not read). LAI is uint8 x 10 (valid 0-100,
# fill 255 over the ocean, Antarctica and other land without a retrieval)
register_dataset(ObsDataset(
    name="glass-lai",
    root=OBS_ROOT / "glass" / "LAI" / "MODIS" / "0.05D",
    template="{year}/GLASS01B01.{version}.A{year}{doy}.{production_date}.hdf",
    versions=("V61",),
    freqs={"8-day": ""},
    lat_name="YDim:GLASS01B01",
    lon_name="XDim:GLASS01B01",
    preprocess=_decode_glass_lai,
    engine="netcdf4",  # not guessed from the .hdf extension
    time_from_fields=_day_of_year_start,
    file_chunks={},  # each file is stored as one 3600 x 7200 chunk: read it whole
))


def get_dataset(dataset: str | ObsDataset) -> ObsDataset:
    """Look up a registered product by name, or pass an `ObsDataset` through."""
    if isinstance(dataset, ObsDataset):
        return dataset
    try:
        return DATASETS[dataset.lower()]
    except KeyError:
        raise KeyError(f"Unknown dataset {dataset!r}; registered: {sorted(DATASETS)}") from None


def _resolve(dataset: str | ObsDataset, version: str | None, freq: str) -> tuple[ObsDataset, str]:
    spec = get_dataset(dataset)
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

_YEAR_FIELDS = ("year", "year_start", "year_end")
# Unfilled template fields of a fixed form, so adjacent fields such as {year}{month}{half} can be parsed; any
# other field matches any text within one path component
_FIELD_PATTERNS = {
    **{f: r"\d{4}" for f in _YEAR_FIELDS},
    "month": r"\d{2}",
    "half": r"0[12]",  # half of a month: 01 days 1-15, 02 the rest
    "doy": r"\d{3}",   # day of year
}


def _template_patterns(spec: ObsDataset, **fields: str) -> tuple[str, re.Pattern]:
    """
    Glob pattern and regex (of the path relative to `spec.root`) of `spec.template` with `fields` filled in.
    The regex captures the remaining template fields.
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
                # A field used twice (e.g. {var}, {version}) must match the same text both times
                regex_parts.append(f"(?P={field})")
            else:
                regex_parts.append(rf"(?P<{field}>{_FIELD_PATTERNS.get(field, '[^/]+')})")
                seen.add(field)
    # Adjacent unfilled fields (e.g. {year}{month}) give "**", which pathlib reads as a recursive glob
    return re.sub(r"\*+", "*", "".join(glob_parts)), re.compile("".join(regex_parts))


def _scan(spec: ObsDataset, **fields: str) -> list[tuple[dict[str, str], Path]]:
    """
    Files matching `spec.template` with `fields` filled in, and the values of the
    remaining template fields parsed from each path (`_template_patterns`).
    """
    pattern, regex = _template_patterns(spec, **fields)
    out = []
    for path in sorted(spec.root.glob(pattern)):
        m = regex.fullmatch(path.relative_to(spec.root).as_posix())
        if m:
            out.append((m.groupdict(), path))
    return out


def _has_year(spec: ObsDataset) -> bool:
    return any(f in _YEAR_FIELDS for _, f, _, _ in string.Formatter().parse(spec.template))


def _file_years(groups: Mapping[str, str]) -> tuple[int, int]:
    """First and last year covered by a file, from its parsed template fields."""
    if "year" in groups:
        return int(groups["year"]), int(groups["year"])
    return int(groups["year_start"]), int(groups["year_end"])


def list_variables(dataset: str | ObsDataset, version: str | None = None, freq: str = "monthly") -> list[str]:
    """Variables with at least one file for this version and frequency."""
    spec, version = _resolve(dataset, version, freq)
    found = _scan(spec, **_fixed_fields(spec, version, freq))
    if not any(f == "var" for _, f, _, _ in string.Formatter().parse(spec.template)):
        # Every variable is in each file (e.g. GPCP): the numeric time-varying variables of the first file
        # (not time bounds)
        if not found:
            return []
        with xr.open_dataset(found[0][1], engine=spec.engine) as ds:
            if spec.time_from_fields is not None:
                # No time axis (one GeoTIFF per composite): the variables that the preprocess hook makes
                return sorted(spec.preprocess(ds, var=None, version=version, freq=freq).data_vars)
            return sorted(v for v in ds.data_vars if "time" in ds[v].dims and ds[v].dtype.kind in "fiu")
    return sorted({groups["var"] for groups, _ in found})


def list_files(
    dataset: str | ObsDataset, var: str, version: str | None = None, freq: str = "monthly",
) -> dict[Path, tuple[int, int]]:
    """Every file for this variable, version and frequency -> (first, last) year it covers, sorted by year."""
    spec, version = _resolve(dataset, version, freq)
    if not _has_year(spec):
        raise ValueError(f"{spec.name}: template has no year field")
    found = _scan(spec, **_fixed_fields(spec, version, freq), var=var)
    return dict(sorted(((path, _file_years(groups)) for groups, path in found), key=lambda item: item[1]))


def list_years(
    dataset: str | ObsDataset, var: str, version: str | None = None, freq: str = "monthly",
) -> list[int]:
    """Years covered by a file for this variable, version and frequency."""
    spans = list_files(dataset, var, version, freq).values()
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
) -> list[Path]:
    """
    Files for `var` overlapping the years of `time_slice`, sorted by first year.
    Warns about years missing inside the requested (or, if open-ended, found)
    range.
    """
    spec, version = _resolve(dataset, version, freq)
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
            warnings.warn(f"{spec.name} {version} {freq} {var}: no files for years {missing}")
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
    lat_bnds: slice | None = None,
    chunks: str | int | Mapping | None = "auto",
    parallel: bool = False,
) -> xr.DataArray:
    """
    Load one variable of an observational product as a single lazy DataArray.

    Parameters
    ----------
    dataset : registered name ("gleam", "pml", "sith", "clara", "avhrr", "viirs", "fluxcom-x", "gldas",
        "gpcp", "gimms-lai4g", "gimms-lai4g-avhrr", "hiq-lai", "globmap-lai", "glass-lai", ...) or an
        `ObsDataset`.
    var : variable name as used in the directory and file names (e.g. "E", "ET"); "lai" for the GeoTIFF
        LAI products.
    time_slice : passed to ``.sel(time=...)``; only files for the years it
        touches are opened.
    version : product version; defaults to the first in `ObsDataset.versions`.
    freq : temporal frequency, a key of `ObsDataset.freqs`.
    lat_bnds : optional latitude slice applied after sorting lat ascending,
        e.g. ``etunc.config.LAT_BNDS``.
    chunks : dask chunks for `xr.open_mfdataset`; dict keys may use lat/lon.
    parallel : open files in parallel with dask.

    Returns
    -------
    DataArray with dims (time, lat, lon) and attrs ``dataset``, ``version``,
    ``frequency`` and ``res`` ("native") added to the file's variable attrs.
    """
    spec, version = _resolve(dataset, version, freq)
    files = find_files(dataset, var, time_slice, version=version, freq=freq)
    if not files:
        y0, y1 = _year_bounds(time_slice)
        raise FileNotFoundError(
            f"No {spec.name} {version} {freq} files for {var!r} in years {y0}-{y1} under "
            f"{spec.root}. Available variables: {list_variables(dataset, version, freq)}"
        )

    if spec.time_from_fields is not None:
        regex = _template_patterns(spec, **_fixed_fields(spec, version, freq), var=var)[1]

    def preprocess(ds: xr.Dataset) -> xr.Dataset:
        source = ds.encoding.get("source")  # before the hook, which may return a new Dataset
        if spec.preprocess is not None:
            ds = spec.preprocess(ds, var=var, version=version, freq=freq)
        if var not in ds:
            raise KeyError(f"{var!r} not in {source}; data vars: {list(ds.data_vars)}")
        ds = ds[[var]]
        if spec.time_from_fields is not None:
            # A file without a time axis: its time stamp from the fields of its name
            fields = regex.fullmatch(Path(source).relative_to(spec.root).as_posix()).groupdict()
            ds = ds.expand_dims(time=[spec.time_from_fields(fields)])
        return standardize_grid(ds, spec)

    if chunks == "auto" and spec.file_chunks is not None:
        chunks = spec.file_chunks
    # Chunks apply when each file is opened, before `preprocess` renames the coords
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
        # Files whose grids differ after rounding raise, instead of being outer-joined into a NaN-padded grid
        join="exact",
        parallel=parallel,
        engine=spec.engine,
    )
    if spec.time_from_fields is not None and not ds.indexes["time"].is_unique:
        # e.g. one composite in two zips: open_mfdataset would stack both
        dup = ds.indexes["time"][ds.indexes["time"].duplicated()]
        raise ValueError(f"{spec.name}: several files for the time stamps {[str(t.date()) for t in dup]}")
    # Time stamps mark the end (GLEAM, SiTHv2) or start (PML, CLARA-A3, FLUXCOM-X, GLDAS, GPCP, GeoTIFF and GLASS LAI)
    # of each period, so slice with year or year-month strings rather than exact dates
    da = ds[var].sel(time=time_slice)
    if lat_bnds is not None:
        da = da.sel(lat=lat_bnds)
    da.attrs.update(dataset=spec.name, version=version, frequency=freq, res="native")
    return da


def load_ncei_lai(
    time_slice: slice = slice(None, None),
    *,
    lat_bnds: slice | None = None,
    chunks: str | int | Mapping | None = "auto",
) -> xr.DataArray:
    """
    The NOAA NCEI LAI record over `time_slice`, joined across sensors: AVHRR LAI v5.0 to 2013-12, then VIIRS
    LAI v1.0 from 2014-01 (`NCEI_LAI_SENSORS`). Each sensor is read with `load_obs`; the two share one grid
    and do not overlap in time, so they are concatenated as they are (no adjustment of either at the switch).

    Returns
    -------
    Lazy monthly DataArray "lai" (time, lat, lon), with a ``sensor`` coord on `time` naming the sensor of
    each month, and attrs ``dataset`` ("ncei_lai"), ``version`` (dataset and version of each sensor),
    ``sensors`` (each sensor and its months) and ``sensor_change`` (the first month of each new sensor;
    "none" if `time_slice` covers one sensor only).
    """
    y0, y1 = _year_bounds(time_slice)
    pieces = []
    for sensor, dataset, version in NCEI_LAI_SENSORS:
        years = list_years(dataset, "lai", version)
        if not years:
            raise FileNotFoundError(f"No {dataset} {version} LAI files under {get_dataset(dataset).root}")
        if (y0 is not None and years[-1] < y0) or (y1 is not None and years[0] > y1):
            continue  # this sensor's record is outside time_slice
        da = load_obs(dataset, "lai", time_slice, version=version, lat_bnds=lat_bnds, chunks=chunks)
        pieces.append(da.assign_coords(sensor=("time", np.full(da.sizes["time"], sensor))))
    if not pieces:
        raise FileNotFoundError(f"No NCEI LAI in {time_slice}")

    for a, b in zip(pieces, pieces[1:]):
        if b.time.values[0] <= a.time.values[-1]:
            raise ValueError(f"{b.sensor.values[0]} starts before {a.sensor.values[0]} ends")
    # join="exact": the sensors must share one grid exactly
    da = xr.concat(pieces, dim="time", join="exact", combine_attrs="override")

    def month(t):
        return pd.Timestamp(t).strftime("%Y-%m")
    da.attrs.update(
        dataset="ncei_lai",
        version=", ".join(f"{p.attrs['dataset']} {p.attrs['version']}" for p in pieces),
        sensors="; ".join(f"{p.sensor.values[0]}: {month(p.time.values[0])} to {month(p.time.values[-1])}"
                          for p in pieces),
        sensor_change="; ".join(f"{month(b.time.values[0])} ({a.sensor.values[0]} to {b.sensor.values[0]})"
                                for a, b in zip(pieces, pieces[1:])) or "none",
    )
    return da


def _regrid_blocks(
    da_all: xr.DataArray, resolutions: Mapping[str, float | str], na_thresh: float, progress: bool = False,
) -> tuple[dict[str, xr.DataArray], dict[str, float]]:
    """
    Conservatively regrid `da_all` onto every grid in `resolutions` ({key:
    spacing}), loading one block of its dask time chunks at a time
    (`regrid_with_na_thresh`). Returns ({key: regridded (time, lat, lon)},
    {"load": s, "regrid": s}). `progress` prints the last year of each block.
    """
    if da_all.sizes["time"] == 0:
        raise ValueError(f"{da_all.name}: no time steps to regrid")
    sizes = np.array(da_all.chunksizes.get("time", (da_all.sizes["time"],)))
    stops = np.cumsum(sizes)
    t = {"load": 0.0, "regrid": 0.0}
    pieces: dict[str, list[xr.DataArray]] = {key: [] for key in resolutions}
    for i0, i1 in zip(stops - sizes, stops):
        t0 = time.perf_counter()
        da = da_all.isel(time=slice(i0, i1)).load()
        t["load"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        for key, res in resolutions.items():
            # na_thresh decides which coastal cells keep a value: NaN if more than na_thresh of the area is missing
            pieces[key].append(regrid_with_na_thresh(da, conservative_regridder(da, res), na_thresh))
        t["regrid"] += time.perf_counter() - t0
        if progress:
            print(f" {da.time.dt.year.values[-1]}", end="", flush=True)
        del da
    return {key: xr.concat(p, dim="time") for key, p in pieces.items()}, t


def load_obs_regridded(
    dataset: str | ObsDataset,
    var: str,
    time_slice: slice = slice(None, None),
    res: float | str = 0.5,
    *,
    version: str | None = None,
    freq: str = "monthly",
    na_thresh: float = NA_THRESH,
) -> xr.DataArray:
    """
    Load one variable from its native files and conservatively regrid it onto
    `target_grid(res)` in memory, without writing regridded files.

    Each time step is regridded with `regrid_with_na_thresh` (NaN where more
    than `na_thresh` of a target cell's area is missing). Files are
    read one year at a time (per-year files whole, multi-year files in blocks of
    their on-disk time chunks, GeoTIFFs one file at a time), so only one block
    is in memory at native resolution.

    Parameters
    ----------
    dataset, var, time_slice, version, freq : as in `load_obs`.
    res : grid spacing [deg] or tag of `target_grid` (0.5, 1.0, "0.5deg", ...).
    na_thresh : maximum fraction of missing source area in a target cell.

    Returns
    -------
    Loaded DataArray (time, lat, lon) on `target_grid(res)`, with attrs
    ``dataset``, ``version``, ``frequency``, ``res`` (spacing, e.g. "0.5"),
    ``regrid_method`` and ``na_thresh``.
    """
    spec, version = _resolve(dataset, version, freq)
    files = find_files(spec, var, time_slice, version=version, freq=freq)
    if not files:
        y0, y1 = _year_bounds(time_slice)
        raise FileNotFoundError(
            f"No {spec.name} {version} {freq} files for {var!r} in years {y0}-{y1} under {spec.root}. "
            f"Available variables: {list_variables(spec, version, freq)}"
        )

    # (time slice, chunks) per year span of the files, clipped to the requested years. Each span once: the
    # files of one year (one per month for GPCP, per composite for GIMMS LAI4g, ...) are opened together.
    # Per-year files: read whole. Multi-year file: dask chunks = on-disk chunks, so each chunk is read once.
    # One GeoTIFF per composite: one block per file ("auto" chunks), not the whole year in memory.
    if _has_year(spec):
        y0, y1 = _year_bounds(time_slice)
        spans = list_files(spec, var, version, freq)
        reads = []
        for first, last in dict.fromkeys(spans[path] for path in files):
            chunks = {} if first != last else "auto" if spec.time_from_fields is not None else None
            reads.append((slice(str(max(first, y0 or first)), str(min(last, y1 or last))), chunks))
    else:
        reads = [(time_slice, {})]

    spacing = float(RESOLUTIONS.get(res, res))  # tag -> spacing
    t = {"load": 0.0, "regrid": 0.0}
    pieces = []
    for file_slice, chunks in reads:
        da_all = load_obs(spec, var, file_slice, version=version, freq=freq, chunks=chunks)
        out, t_file = _regrid_blocks(da_all, {"out": spacing}, na_thresh)
        pieces.append(out["out"])
        for k, v in t_file.items():
            t[k] += v
        del da_all, out
        gc.collect()

    da = xr.concat(pieces, dim="time").sel(time=time_slice)
    da.attrs.update(res=f"{spacing:g}", regrid_method="xesmf conservative, skipna=True, unmapped_to_nan=True",
                    na_thresh=na_thresh)
    print(f"{spec.name} {version} {freq} {var}: regridded {len(files)} files to {spacing:g} deg "
          f"(load {t['load']:.1f}s, regrid {t['regrid']:.1f}s)")
    return da
