"""
etunc.temporal
==============
Time handling: annual means and climatologies of the binning pipeline
(`aggregate`, steps 4 and 10 of docs/binning-workflow.md), complete and shared
years, and the time-period strings used in file names.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr


# ------------------------------------------------------------------
# Time-period strings
# ------------------------------------------------------------------

def format_time_period(time_slice: slice) -> str:
    """slice("1950-01", "2014-12") -> "195001-201412", for file names."""
    return f"{time_slice.start.replace('-', '')}-{time_slice.stop.replace('-', '')}"


def period_str(years) -> str:
    """Years [2003, ..., 2009] -> "200301-200912", for file names."""
    return format_time_period(slice(f"{min(years)}-01", f"{max(years)}-12"))


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------

def complete_years(da: xr.DataArray) -> list[int]:
    """Years with all 12 months in the time axis."""
    # Checks the time axis only, not the values: an all-NaN month still counts
    years, months = da.time.dt.year.values, da.time.dt.month.values
    return [int(y) for y in np.unique(years) if np.unique(months[years == y]).size == 12]


def aggregate(
    da: xr.DataArray,
    how: Literal["year", "clim"],
    *,
    require_all_months: bool = True,
    label: str | None = None,
) -> xr.DataArray:
    """
    Temporal aggregation of the binning pipeline (docs/binning-workflow.md).

    how="year" (step 4): days-in-month weighted annual means of a monthly
        field, `time` -> `year`, over complete calendar years only (all 12
        months in the time axis; partial first/last years are dropped).
        require_all_months=True (ET, P, Rn): a gridcell-year with any
        missing month is NaN, since a missing month would bias the flux.
        require_all_months=False (LAI): a missing month counts as LAI = 0
        (winter gaps at high latitudes); only gridcell-years without any
        valid month are NaN.
    how="clim" (step 10): mean over `year` of annual means. Every year
        weighs equally, and a gridcell's missing (NaN) years are skipped, so
        its climatology covers fewer years.

    Prints a one-line report of what the rule did (`label`, default the
    name of `da`) and stores the counts in the attrs of the result.
    """
    label = label or da.name or "field"

    if how == "year":
        if "time" not in da.dims:
            raise ValueError(f"{label}: aggregate(how='year') needs a monthly `time` dimension; got {da.dims}")
        years = complete_years(da)
        dropped = sorted(set(da.time.dt.year.values.tolist()) - set(years))
        da = da.sel(time=da.time.dt.year.isin(years))
        # Days-in-month weights that sum to 1 over the 12 months of each year
        days = da.time.dt.days_in_month
        weights = days.groupby("time.year") / days.groupby("time.year").sum()
        # Weighted sum and valid-month count in one pass (one read of lazy data). The
        # NaN-skipping sum counts a missing month as 0
        ds = xr.Dataset({
            "sum": (da * weights).groupby("time.year").sum(keep_attrs=True),
            "n_valid": da.notnull().groupby("time.year").sum(),
        }).load()
        n_valid = ds["n_valid"]
        with xr.set_options(keep_attrs=True):
            out = ds["sum"].where(n_valid == 12 if require_all_months else n_valid > 0).rename(da.name)
        if {"lat", "lon"} <= set(out.dims):
            out = out.transpose(..., "lat", "lon")  # ([member,] year, lat, lon), as the binning expects
        n_partial = int(((n_valid > 0) & (n_valid < 12)).sum())
        rule = "set to NaN" if require_all_months else "missing months counted as 0"
        print(f"{label}: annual means of {len(years)} complete years"
              + (f" (incomplete years dropped: {dropped})" if dropped else "")
              + f"; {n_partial} gridcell-years with 1-11 valid months, {rule}")
        out.attrs.update(n_partial_gridcell_years=n_partial, partial_years=rule)
        return out

    if how == "clim":
        if "year" not in da.dims:
            raise ValueError(f"{label}: aggregate(how='clim') needs a `year` dimension; got {da.dims}")
        n_years = da.sizes["year"]
        n_valid = da.notnull().sum("year")
        missing = (n_valid > 0) & (n_valid < n_years)
        n_missing = int(missing.sum())
        out = da.mean("year", keep_attrs=True)
        report = f"{label}: climatology of {n_years} years; {n_missing} gridcells have missing years"
        if n_missing:
            report += f" (fewest valid years: {int(n_valid.where(missing).min())})"
        print(report)
        out.attrs.update(n_years=n_years, n_gridcells_missing_years=n_missing)
        return out

    raise ValueError(f"Unknown aggregation {how!r}; use 'year' or 'clim'")


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
