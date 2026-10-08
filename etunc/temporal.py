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
    time_raw = time.values
    if isinstance(time_raw, np.datetime64):
        return pd.Timestamp(time_raw).strftime("%Y%m")
    elif isinstance(time_raw, np.ndarray):
        return time.item().strftime("%Y%m")
    raise TypeError(f"Unsupported type {type(time)!r} for time")


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------

def compute_annual_mean(da: xr.DataArray) -> xr.DataArray:
    """Days-in-month weighted annual mean of a monthly field; `time` -> `year`."""
    days_in_month = da.time.dt.days_in_month
    weights = days_in_month.groupby('time.year') / days_in_month.groupby('time.year').sum()
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
        return compute_annual_mean(da).mean("year", keep_attrs=True)
    if how == "year_max":
        return da.groupby("time.year").max(keep_attrs=True)
    if how == "clim_max":
        return da.groupby("time.month").mean(keep_attrs=True).max("month", keep_attrs=True)
    raise ValueError(f"Unknown aggregation {how!r}")
