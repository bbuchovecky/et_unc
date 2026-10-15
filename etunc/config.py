"""
etunc.config
============
Paths and domain constants shared by the package and the scripts, the
paths of the files passed between the pipeline scripts (`inputs_path`,
`area_mask_path`), and the two functions that print the constants
(`summarize`) and a script's own settings (`summarize_settings`) to the log. Dataset-specific settings stay at the top
of each script.

Every output root derives from ``WORK_ROOT``.
"""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import numpy as np


# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------

# Shared campaign storage: read-only unless writing regridded files on purpose.
CAMPAIGN_ROOT = Path("/glade/campaign/univ/uwas0155")
OBS_ROOT = CAMPAIGN_ROOT / "obs"
ILAMB_ROOT = OBS_ROOT / "ilamb"

# DEPRECATED: Regridded files, written by the regrid scripts and read by the loaders
OBS_REGRID_ROOT = OBS_ROOT / "regridded"
CMIP_REGRID_ROOT = CAMPAIGN_ROOT / "cmip6" / "regridded"

# CMIP6 catalogs from the external cmip-intake-esgf-fetch tool. The catalogs, and most file paths in them, are on purgeable Derecho scratch.
# Moving them to campaign storage only needs CMIP_FETCH_ROOT changed here.
CMIP_FETCH_ROOT = Path("/glade/derecho/scratch/bbuchovecky/cmip_intake_esgf_fetch")
CMIP_CATALOG_ROOT = CMIP_FETCH_ROOT / "catalogs"
CMIP_CATALOG = CMIP_CATALOG_ROOT / "cmip6_bin.csv"  # evspsbl, lai, pr, radiative fluxes
CMIP_FX_CATALOG = CMIP_CATALOG_ROOT / "cmip6_fx_glade.csv"  # one sftlf file per model

# ESGF_CACHE_CATALOG = CMIP_FETCH_ROOT / "manifests" / "esgf_cache_catalog.csv"  # I don't think this is necessary anymore


# ------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------

WORK_ROOT = Path("/glade/work/bbuchovecky/et_unc")
PROC_ROOT = WORK_ROOT / "proc"              # binned stats in PROC_ROOT/<family>/qbin/
FIG_ROOT = WORK_ROOT / "fig"
BIN_EDGES_ROOT = PROC_ROOT / "qbin_edges"


def inputs_path(family: str, grid: str, period: str) -> Path:
    """Binning inputs of one family, written by scripts/prep_<family>.py."""
    return PROC_ROOT / family / f"{family}.inputs.{grid}.{period}.nc"


def area_mask_path(grid: str, period: str) -> Path:
    """Common area mask of all families, written by scripts/make_edges.py."""
    return PROC_ROOT / "all" / f"all.area_mask.{grid}.{period}.nc"


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# Pipeline settings shared by every family (docs/binning-workflow.md)
TIME_SLICE = slice("1995-01", "2014-12")  # one period for every family
MIN_YEARS = 3          # minimum complete years shared by the four products of an obs combination
MIN_ANNUAL_ET = -1.0   # W/m2; gridcell-years with lower annual mean ET are NaN (step 6)
FAMILIES = ("obs", "cmip", "fppe", "lens")  # families pooled by make_edges.py and binned by bin_all.py

# Binning settings
N_XBINS = 15         # aridity index
N_YBINS = 15         # LAI
SIGNIF_ALPHA = 0.05  # t-test of bin mean ET against 0 (95% confidence)
SIGNIF_N_MIN = 10    # bins with <= this many samples are never significant

# Common grid and masking
TARGET_RES = 1             # target resolution (regrid all output to this resolution)
LAT_BNDS = slice(-58, 90)  # excludes Antarctica (Tierra del Fuego ends near -56)
NA_THRESH = 0.5            # for conservative regridding: target cells with more than this 
                           # fraction of missing source area are NaN
LF_THRESH = 0.5            # gridcell land fraction threshold (land where strictly greater)

# One Lv (near 20 C) for every water flux, ET and precipitation alike, with no
# sublimation or fusion term. It cancels in ET/P but sets the W/m2 scale of ET and AI
LATENT_HEAT_VAPORIZATION = 2.45e6  # J/kg
LIQ_WATER_DENSITY = 1e3            # kg/m3
EARTH_RADIUS = 6.371e6             # m, as in ILAMB, so cell areas match ILAMB's

# Fixed histogram bins of climatological LAI and AI, so every histogram figure has the same
# bin widths; values outside are counted in the end bins
HIST_BINS = {
    "lai": np.linspace(0, 8, 81),    # 0.1 m2/m2
    "ai": np.linspace(-1, 10, 111),  # 0.1
}

PROJECTION = ccrs.PlateCarree()
DPI = 120


def summarize():
    print("######### config.py")
    print("\nInput Paths:")
    print(f"{'CAMPAIGN_ROOT':18} = {CAMPAIGN_ROOT}")
    print(f"{'OBS_ROOT':18} = {OBS_ROOT}")
    print(f"{'ILAMB_ROOT':18} = {ILAMB_ROOT}")
    print(f"{'OBS_REGRID_ROOT':18} = {OBS_REGRID_ROOT}")
    print(f"{'CMIP_REGRID_ROOT':18} = {CMIP_REGRID_ROOT}")
    print(f"{'CMIP_FETCH_ROOT':18} = {CMIP_FETCH_ROOT}")
    print(f"{'CMIP_CATALOG_ROOT':18} = {CMIP_CATALOG_ROOT}")
    print(f"{'CMIP_CATALOG':18} = {CMIP_CATALOG}")
    print(f"{'CMIP_FX_CATALOG':18} = {CMIP_FX_CATALOG}")
    # print(f"{'ESGF_CACHE_CATALOG':18} = {ESGF_CACHE_CATALOG}")
    print("\nOutput Paths:")
    print(f"{'WORK_ROOT':18} = {WORK_ROOT}")
    print(f"{'PROC_ROOT':18} = {PROC_ROOT}")
    print(f"{'FIG_ROOT':18} = {FIG_ROOT}")
    print(f"{'BIN_EDGES_ROOT':18} = {BIN_EDGES_ROOT}")
    print("\nConstants:")
    print(f"{'TIME_SLICE':18} = {TIME_SLICE}")
    print(f"{'MIN_YEARS':18} = {MIN_YEARS}")
    print(f"{'MIN_ANNUAL_ET':18} = {MIN_ANNUAL_ET}")
    print(f"{'FAMILIES':18} = {FAMILIES}")
    print(f"{'N_XBINS':18} = {N_XBINS}")
    print(f"{'N_YBINS':18} = {N_YBINS}")
    print(f"{'TARGET_RES':18} = {TARGET_RES}")
    print(f"{'LAT_BNDS':18} = {LAT_BNDS}")
    print(f"{'NA_THRESH':18} = {NA_THRESH}")
    print(f"{'LF_THRESH':18} = {LF_THRESH}")
    print(f"{'LATENT_HEAT_VAPORIZATION':18} = {LATENT_HEAT_VAPORIZATION}")
    print(f"{'LIQ_WATER_DENSITY':18} = {LIQ_WATER_DENSITY}")
    print(f"{'EARTH_RADIUS':18} = {EARTH_RADIUS}")
    print("\n######### config.py")


def summarize_settings(namespace: dict):
    """
    Print the settings of a script: every UPPERCASE module-level name in
    `namespace` (call with `globals()`) that holds a plain value (number,
    string, path, slice, or a container of them), so the log of each run
    records them. Larger objects such as grids are skipped.
    """
    plain = (bool, int, float, str, Path, slice, tuple, list, dict, type(None))
    print("######### script settings")
    for name, value in namespace.items():
        if name.isupper() and isinstance(value, plain):
            print(f"{name:18} = {value!r}")
    print("#########")
