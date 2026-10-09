"""
etunc.config
============
Paths and domain constants shared by the package and the scripts. Nothing
else belongs here: no functions, no dataset-specific settings (those stay at
the top of each script).

Every output root derives from ``WORK_ROOT``.
"""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs


# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------

# Shared campaign storage: read-only unless writing regridded files on purpose.
CAMPAIGN_ROOT = Path("/glade/campaign/univ/uwas0155")
OBS_ROOT = CAMPAIGN_ROOT / "obs"
ILAMB_ROOT = OBS_ROOT / "ilamb"

# Regridded files, written by the regrid scripts and read by the loaders.
OBS_REGRID_ROOT = OBS_ROOT / "regridded"
CMIP_REGRID_ROOT = CAMPAIGN_ROOT / "cmip6" / "regridded"

# CMIP6 catalogs from the external cmip-intake-esgf-fetch tool. They (and about
# 2/3 of the file paths in cmip_evap.csv) are on purgeable Derecho scratch.
# Moving them to campaign storage only needs CMIP_FETCH_ROOT changed here.
CMIP_FETCH_ROOT = Path("/glade/derecho/scratch/bbuchovecky/cmip_intake_esgf_fetch")
CMIP_CATALOG_ROOT = CMIP_FETCH_ROOT / "catalogs"
CMIP_CATALOG = CMIP_CATALOG_ROOT / "cmip_evap.csv"  # evspsbl, lai, pr, radiative fluxes
CMIP_FX_CATALOG = CMIP_CATALOG_ROOT / "cmip6_fx_glade.csv"  # one sftlf file per model
ESGF_CACHE_CATALOG = CMIP_FETCH_ROOT / "manifests" / "esgf_cache_catalog.csv"


# ------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------

WORK_ROOT = Path("/glade/work/bbuchovecky/et_unc")
PROC_ROOT = WORK_ROOT / "proc"              # binned stats in PROC_ROOT/<dataset>/qbin/
FIG_ROOT = WORK_ROOT / "fig"
BIN_EDGES_ROOT = PROC_ROOT / "qbin_edges"


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

LAT_BNDS = slice(-58, 90)  # excludes Antarctica (Tierra del Fuego ends near -56)
LF_THRESH = 0.5            # gridcell land fraction threshold (land where strictly greater)

# One L (near 20 C) for every water flux, ET and precipitation alike, with no
# sublimation or fusion term. It cancels in ET/P but sets the W/m2 scale of ET and AI
LATENT_HEAT_VAPORIZATION = 2.45e6  # J/kg
LIQ_WATER_DENSITY = 1e3            # kg/m3
EARTH_RADIUS = 6.371e6             # m, as in ILAMB, so cell areas match ILAMB's

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
    print(f"{'ESGF_CACHE_CATALOG':18} = {ESGF_CACHE_CATALOG}")
    print("\nOutput Paths:")
    print(f"{'WORK_ROOT':18} = {WORK_ROOT}")
    print(f"{'PROC_ROOT':18} = {PROC_ROOT}")
    print(f"{'FIG_ROOT':18} = {FIG_ROOT}")
    print(f"{'BIN_EDGES_ROOT':18} = {BIN_EDGES_ROOT}")
    print("\nConstants:")
    print(f"{'LAT_BNDS':18} = {LAT_BNDS}")
    print(f"{'LF_THRESH':18} = {LF_THRESH}")
    print(f"{'LATENT_HEAT_VAPORIZATION':18} = {LATENT_HEAT_VAPORIZATION}")
    print(f"{'LIQ_WATER_DENSITY':18} = {LIQ_WATER_DENSITY}")
    print(f"{'EARTH_RADIUS':18} = {EARTH_RADIUS}")
    print("\n######### config.py")
