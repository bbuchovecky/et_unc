"""
etunc.units
===========
Unit conversions and net radiation. Water fluxes become their energy
equivalent [W/m2] with L = LATENT_HEAT_VAPORIZATION. There are two conversion
paths: `convert_units` keys on the variable name (CMIP, CESM, CLM, ERA5,
ERA5-Land "e" and "tp") and
`latent_heat_to_wm2` on the `units` attr (ILAMB and gridded obs;
`flux_to_wm2` also takes water depths per period). `accumulation_to_flux`
turns water depths per hour, day, month or year (the gridded obs products)
into kg m-2 s-1, and `era5_land_evaporation_to_flux` and
`era5_land_precipitation_to_flux` do the same for ERA5-Land (m of water per
day, positive downward). Net radiation sign conventions differ by source, so
each has its own `net_radiation_*`.
"""

from __future__ import annotations

import re

import xarray as xr

from etunc.config import LATENT_HEAT_VAPORIZATION, LIQ_WATER_DENSITY


def convert_units(v: str, da: xr.DataArray, verbose: bool = False) -> xr.DataArray:
    """Convert water fluxes to their energy equivalent [W/m2], keyed on variable name `v`."""

    # Any other name (LAI, radiative fluxes, EFLX_LH_TOT) passes through unchanged, without
    # a check of its units. The `if`s are not exclusive: "mer" is scaled and then sign-flipped

    # CESM2: precip from m/s -> W/m2 (a volume flux, so times the water density too)
    if v in ("PRECT_calculated_month_1", "PRECT_month_1", "PRECT_calculated", "PRECT"):
        if verbose:
            print("Converting units from m/s -> W/m2")
        da = da * LATENT_HEAT_VAPORIZATION * LIQ_WATER_DENSITY
        da.attrs["units"] = "W/m2"

    # CLM (offline PPE): rain and snow from mm/s (= kg/m2/s) -> W/m2
    if v in ("RAIN", "SNOW"):
        if verbose:
            print("Converting units from mm/s -> W/m2")
        da = da * LATENT_HEAT_VAPORIZATION
        da.attrs["units"] = "W/m2"

    # CMIP6, ERA5, ILAMB: precip and et from kg/m2/s -> W/m2
    if v in ("pr", "evspsbl", "mtpr", "mer", "et"):
        if verbose:
            print("Converting units from kg/m2/s -> W/m2")
        da = da * LATENT_HEAT_VAPORIZATION
        da.attrs["units"] = "W/m2"

    # ERA5: et sign convention (fluxes are positive downward, so evaporation is negative)
    if v == "mer":
        with xr.set_options(keep_attrs=True):  # keep the "W/m2" set above
            da = -1 * da

    # ERA5-Land: evaporation "e" and precipitation "tp" from m of water per day (positive downward) -> W/m2,
    # evaporation positive upward. These generic names are safe to key on because both conversions check the
    # ERA5-Land units attr and raise otherwise
    if v == "e":
        if verbose:
            print("Converting units from m/day (positive downward) -> W/m2 (positive upward)")
        da = latent_heat_to_wm2(era5_land_evaporation_to_flux(da))
    if v == "tp":
        if verbose:
            print("Converting units from m/day -> W/m2")
        da = latent_heat_to_wm2(era5_land_precipitation_to_flux(da))

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
    `latent_heat_to_wm2`, plus water depths per hour, day, month or year
    (mm/day of GPCCv2018 and GPCP, monthly totals of PML, ...), converted by
    `accumulation_to_flux` first.
    """
    if _accumulation_period(str(da.attrs.get("units", ""))) is not None:
        da = accumulation_to_flux(da)
    return latent_heat_to_wm2(da)


_ACCUMULATION_PERIODS = {"h": "hour", "hr": "hour", "hour": "hour", "d": "day", "day": "day",
                         "month": "month", "mon": "month", "year": "year", "yr": "year"}


def _accumulation_period(units: str) -> str | None:
    """The period of a water depth unit ("mm/day" -> "day", "mm month-1" -> "month"), or None."""
    m = re.fullmatch(r"mm\s*[./ ]\s*([a-z]+)(?:-1)?", units.strip().lower())
    return _ACCUMULATION_PERIODS.get(m.group(1)) if m else None


def accumulation_to_flux(da: xr.DataArray) -> xr.DataArray:
    """
    Convert a water depth per hour, day, month or year (mm hr-1, mm/day, mm.month-1, mm/year, ...)
    to a mass flux [kg m-2 s-1], using each time step's days in month or year.
    The result can go straight into ``latent_heat_to_wm2``.
    """
    units = str(da.attrs.get("units", ""))
    period = _accumulation_period(units)
    if period is None:
        raise ValueError(f"Cannot convert units {units!r} of {da.name!r} to kg m-2 s-1")

    # Each time stamp must lie inside the month or year it covers (not at its end, as in
    # raw CESM output), or the wrong month length is used
    if period == "hour":  # a mean hourly rate (FLUXCOM-X)
        days = 1 / 24
    elif period == "day":
        days = 1
    elif period == "month":
        days = da.time.dt.days_in_month
    else:
        days = xr.where(da.time.dt.is_leap_year, 366, 365)
    out = da / (days * 86400)  # 1 mm of liquid water = 1 kg m-2
    out.attrs = {**da.attrs, "units": "kg m-2 s-1"}
    return out


def era5_land_evaporation_to_flux(da: xr.DataArray) -> xr.DataArray:
    """
    Convert ERA5-Land monthly mean evaporation "e" to a mass flux [kg m-2 s-1],
    positive upward. The monthly means of ERA5-Land accumulations are means of
    daily accumulations, so "m of water equivalent" is m of water per day, and
    ECMWF fluxes are positive downward, so evaporation is negative. The result
    can go straight into ``latent_heat_to_wm2``.
    """
    units = str(da.attrs.get("units", ""))
    if units != "m of water equivalent":
        raise ValueError(f"Cannot convert units {units!r} of {da.name!r}: expected 'm of water equivalent' per day")
    out = -da * LIQ_WATER_DENSITY / 86400  # m/day -> kg m-2 s-1, sign flipped to positive upward
    out.attrs = {**da.attrs, "units": "kg m-2 s-1"}
    return out


def era5_land_precipitation_to_flux(da: xr.DataArray) -> xr.DataArray:
    """
    Convert ERA5-Land monthly mean total precipitation "tp" (a mean of daily
    accumulations, so m of water per day, positive downward) to a mass flux
    [kg m-2 s-1]. The result can go straight into ``latent_heat_to_wm2``.
    """
    units = str(da.attrs.get("units", ""))
    if units != "m":
        raise ValueError(f"Cannot convert units {units!r} of {da.name!r}: expected 'm' (of water) per day")
    out = da * LIQ_WATER_DENSITY / 86400  # m/day -> kg m-2 s-1
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
    """CESM net surface radiation [W/m2]; FSNS is net SW *down*, FLNS is net LW *up*."""
    rn = fsns - flns
    rn.attrs = {"units": "W/m2", "long_name": "net surface radiation"}
    return rn.rename("rn")


def net_radiation_clm(fsa: xr.DataArray, fira: xr.DataArray) -> xr.DataArray:
    """CLM net surface radiation of the land [W/m2]; FSA is absorbed SW (down), FIRA is net LW *up*."""
    rn = fsa - fira
    rn.attrs = {"units": "W/m2", "long_name": "net surface radiation"}
    return rn.rename("rn")


def net_radiation_era5(msnswrf: xr.DataArray, msnlwrf: xr.DataArray) -> xr.DataArray:
    """ERA5 net surface radiation [W/m2]; both net fluxes are positive downward."""
    rn = msnswrf + msnlwrf
    rn.attrs = {"units": "W/m2", "long_name": "net surface radiation"}
    return rn.rename("rn")


def net_radiation_era5_land(ssr: xr.DataArray, str_: xr.DataArray) -> xr.DataArray:
    """
    ERA5-Land net surface radiation [W/m2] from the monthly means of ssr (net SW) and str (net LW):
    means of daily accumulations [J m-2 per day], both positive downward.
    """
    for da in (ssr, str_):
        if da.attrs.get("units") != "J m**-2":
            raise ValueError(f"Cannot convert units {da.attrs.get('units')!r} of {da.name!r}: expected 'J m**-2' per day")
    rn = (ssr + str_) / 86400
    rn.attrs = {"units": "W/m2", "long_name": "net surface radiation"}
    return rn.rename("rn")


def net_radiation_gldas(swnet: xr.DataArray, lwnet: xr.DataArray) -> xr.DataArray:
    """GLDAS net surface radiation [W/m2]; Swnet_tavg and Lwnet_tavg are both positive downward."""
    rn = swnet + lwnet
    rn.attrs = {"units": "W/m2", "long_name": "net surface radiation"}
    return rn.rename("rn")
