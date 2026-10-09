"""
etunc.units
===========
Unit conversions and net radiation. Water fluxes become their energy
equivalent [W/m2] with L = LATENT_HEAT_VAPORIZATION. There are two conversion
paths: `convert_units` keys on the variable name (CMIP, CESM, ERA5) and
`latent_heat_to_wm2` on the `units` attr (ILAMB; `flux_to_wm2` also takes
mm/day). `accumulation_to_flux` turns water depths per day, month or year
(the gridded obs products) into kg m-2 s-1. Net radiation sign
conventions differ by source, so each has its own `net_radiation_*`.
"""

from __future__ import annotations

import re

import xarray as xr

from etunc.config import LATENT_HEAT_VAPORIZATION, LIQ_WATER_DENSITY


def convert_units(v: str, da: xr.DataArray, verbose: bool = False) -> xr.DataArray:
    """Convert water fluxes to their energy equivalent [W/m2], keyed on variable name `v`."""

    # CESM2: precip from m/s -> W/m2
    if v in ("PRECT_calculated_month_1", "PRECT_month_1", "PRECT_calculated", "PRECT"):
        if verbose:
            print("Converting units from m/s -> W/m2")
        da = da * LATENT_HEAT_VAPORIZATION * LIQ_WATER_DENSITY
        da.attrs["units"] = "W/m2"

    # CMIP6, ERA5, ILAMB: precip and et from kg/m2/s -> W/m2
    if v in ("pr", "evspsbl", "mtpr", "mer", "et"):
        if verbose:
            print("Converting units from kg/m2/s -> W/m2")
        da = da * LATENT_HEAT_VAPORIZATION
        da.attrs["units"] = "W/m2"

    # ERA5: et sign convention
    if v == "mer":
        da = -1 * da

    return da


# Multiplicative factors to W/m2, keyed on normalized (lowercase, single-spaced) unit strings
# as they appear in the ILAMB evspsbl and hfls files
_UNITS_TO_WM2 = {
    "w m-2": 1.0,
    "w/m2": 1.0,
    "w/m^2": 1.0,
    "watt m-2": 1.0,
    "watt/m2": 1.0,
    "kg/m2/s": LATENT_HEAT_VAPORIZATION,
    "kg m-2 s-1": LATENT_HEAT_VAPORIZATION,
    "mj m-2 day-1": 1e6 / 86400,
}


def latent_heat_to_wm2(da: xr.DataArray) -> xr.DataArray:
    """Convert an ET [kg/m2/s] or latent heat [W/m2, MJ/m2/day] flux to W/m2, keyed on `da.attrs["units"]`."""
    units = da.attrs.get("units", "")
    key = " ".join(units.lower().split())
    if key not in _UNITS_TO_WM2:
        raise ValueError(f"Cannot convert units {units!r} of {da.name!r} to W/m2")
    out = da * _UNITS_TO_WM2[key]
    out.attrs = {**da.attrs, "units": "W/m2"}
    return out


def flux_to_wm2(da: xr.DataArray) -> xr.DataArray:
    """
    Convert an ET, precipitation or net radiation flux to W/m2: like
    `latent_heat_to_wm2`, plus water fluxes in mm/day (mm d-1).
    """
    key = " ".join(str(da.attrs.get("units", "")).lower().split())
    if key in ("mm d-1", "mm/day"):  # GPCCv2018; 1 mm of water = 1 kg/m2
        da = (da / 86400).assign_attrs({**da.attrs, "units": "kg m-2 s-1"})
    return latent_heat_to_wm2(da)


_ACCUMULATION_PERIODS = {"d": "day", "day": "day", "month": "month", "mon": "month", "year": "year", "yr": "year"}


def accumulation_to_flux(da: xr.DataArray) -> xr.DataArray:
    """
    Convert a water depth per day, month or year (mm/day, mm.month-1, mm/year, ...)
    to a mass flux [kg m-2 s-1], using each time step's days in month or year.
    The result can go straight into ``latent_heat_to_wm2``.
    """
    units = str(da.attrs.get("units", ""))
    m = re.fullmatch(r"mm\s*[./ ]\s*([a-z]+)(?:-1)?", units.strip().lower())
    period = _ACCUMULATION_PERIODS.get(m.group(1)) if m else None
    if period is None:
        raise ValueError(f"Cannot convert units {units!r} of {da.name!r} to kg m-2 s-1")

    if period == "day":
        days = 1
    elif period == "month":
        days = da.time.dt.days_in_month
    else:
        days = xr.where(da.time.dt.is_leap_year, 366, 365)
    out = da / (days * 86400)  # 1 mm of liquid water = 1 kg m-2
    out.attrs = {**da.attrs, "units": "kg m-2 s-1"}
    return out


# ------------------------------------------------------------------
# Net radiation
# ------------------------------------------------------------------

def net_radiation_cmip(
    rsds: xr.DataArray,
    rsus: xr.DataArray,
    rlds: xr.DataArray,
    rlus: xr.DataArray,
) -> xr.DataArray:
    """CMIP6 net surface radiation [W/m2]; all four fluxes are positive in their named direction."""
    rn = rsds - rsus + rlds - rlus
    rn.attrs = {"units": "W/m2", "long_name": "net surface radiation"}
    return rn.rename("rn")


def net_radiation_cesm(fsns: xr.DataArray, flns: xr.DataArray) -> xr.DataArray:
    """CESM net surface radiation [W/m2]; FSNS is net SW down, FLNS is net LW *up*."""
    rn = fsns - flns
    rn.attrs = {"units": "W/m2", "long_name": "net surface radiation"}
    return rn.rename("rn")


def net_radiation_era5(msnswrf: xr.DataArray, msnlwrf: xr.DataArray) -> xr.DataArray:
    """ERA5 net surface radiation [W/m2]; both net fluxes are positive downward."""
    rn = msnswrf + msnlwrf
    rn.attrs = {"units": "W/m2", "long_name": "net surface radiation"}
    return rn.rename("rn")
