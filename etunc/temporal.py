"""
etunc.temporal
==============
Time handling: annual means and other temporal aggregations of monthly
fields, and the time-period strings used in file names.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr


Aggregation = Literal["mon", "year", "clim", "year_max", "clim_max"]


# ------------------------------------------------------------------
# Time-period strings
# ------------------------------------------------------------------

def format_time_period(time_slice: slice) -> str:
    """slice("1950-01", "2014-12") -> "195001-201412", for file names."""
    return f"{time_slice.start.replace('-', '')}-{time_slice.stop.replace('-', '')}"


def to_yyyymm(time) -> str:
    """One time stamp (numpy datetime64 or cftime) -> "YYYYMM"."""
    time_raw = time.values
    if isinstance(time_raw, np.datetime64):
        return pd.Timestamp(time_raw).strftime("%Y%m")
    elif isinstance(time_raw, np.ndarray):  # cftime dates come back as a 0-d object array
        return time.item().strftime("%Y%m")
    raise TypeError(f"Unsupported type {type(time)!r} for time")


def period_str(years) -> str:
    """Years [2003, ..., 2009] -> "200301-200912", for file names."""
    return format_time_period(slice(f"{min(years)}-01", f"{max(years)}-12"))


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------

def compute_annual_mean(da: xr.DataArray) -> xr.DataArray:
    """Days-in-month weighted annual mean of a monthly field; `time` -> `year`."""
    days_in_month = da.time.dt.days_in_month
    # Weights sum to 1 over the months *in the time axis*, so a year with time steps
    # missing averages only the months it has (see `complete_years`)
    weights = days_in_month.groupby('time.year') / days_in_month.groupby('time.year').sum()
    # The NaN-skipping sum counts a NaN month as 0 (biased low) and gives 0, not NaN,
    # for an all-NaN year. `annual_mean` masks both cases
    with xr.set_options(keep_attrs=True):
        return (da * weights).groupby('time.year').sum()


def aggregate(da: xr.DataArray, how: Aggregation = "clim") -> xr.DataArray:
    """
    Temporal aggregation of a monthly field.

    how : "mon"      -> unchanged
          "year"     -> annual mean (`year` dim)
          "clim"     -> time mean of the annual means
          "year_max" -> annual maximum of monthly values (`year` dim)
          "clim_max" -> maximum of the mean seasonal cycle

    A field without `time` is assumed to be aggregated already: "clim" then
    averages over `year` if present and otherwise returns `da` unchanged.
    """
    if "time" not in da.dims:
        if how == "clim":
            return da.mean("year", keep_attrs=True) if "year" in da.dims else da
        if how == "year" and "year" in da.dims:
            return da
        raise ValueError(f"aggregate(how={how!r}) needs a monthly `time` dimension; got {da.dims}")

    if how == "mon":
        return da
    if how == "year":
        return compute_annual_mean(da)
    if how == "clim":
        # Every year weighs equally (NaN years skipped)
        return compute_annual_mean(da).mean("year", keep_attrs=True)
    if how == "year_max":
        return da.groupby("time.year").max(keep_attrs=True)
    if how == "clim_max":
        return da.groupby("time.month").mean(keep_attrs=True).max("month", keep_attrs=True)
    raise ValueError(f"Unknown aggregation {how!r}")


def complete_years(da: xr.DataArray) -> list[int]:
    """Years with all 12 months in the time axis."""
    # Checks the time axis only, not the values: an all-NaN month still counts
    years, months = da.time.dt.year.values, da.time.dt.month.values
    return [int(y) for y in np.unique(years) if np.unique(months[years == y]).size == 12]


def annual_mean(da: xr.DataArray, require_all_months: bool) -> xr.DataArray:
    """
    Annual mean via `aggregate`, which counts missing months as 0. With
    `require_all_months`, years with any missing month are NaN instead;
    otherwise only years without any valid month are NaN.
    """
    # The loaders pass require_all_months=False only for LAI, so a missing LAI month counts
    # as LAI = 0. ET, pr and rns need all 12 months, since a 0 would bias the flux low
    n_valid = da.notnull().groupby("time.year").sum()
    with xr.set_options(keep_attrs=True):
        ann = aggregate(da, "year")
        return ann.where(n_valid == 12 if require_all_months else n_valid > 0)


def shared_years(*anns: xr.DataArray) -> list[int]:
    """Sorted years present in the `year` dim of every one of `anns`."""
    return sorted(set.intersection(*(set(a.year.values.tolist()) for a in anns)))


def yearly_to_annual(da: xr.DataArray) -> xr.DataArray:
    """Yearly field (one time step per year) with `time` replaced by `year`."""
    years = da.time.dt.year.values
    if len(set(years)) != len(years):
        raise ValueError(f"{da.name}: more than one time step in a year")
    return da.assign_coords(year=("time", years)).swap_dims(time="year").drop_vars("time")


def on_month_axis(da: xr.DataArray, months: pd.DatetimeIndex) -> xr.DataArray:
    """Time stamps set to the first of the month, reindexed to `months` (missing months are NaN)."""
    t = pd.to_datetime(pd.DataFrame({"year": da.time.dt.year.values, "month": da.time.dt.month.values, "day": 1}))
    if t.duplicated().any():
        raise ValueError(f"{da.name}: more than one time step in a month")
    return da.assign_coords(time=t.values).reindex(time=months)
