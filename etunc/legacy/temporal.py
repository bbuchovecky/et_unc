"""
etunc.legacy.temporal
=====================
Unmaintained and untested. The temporal aggregation of `etunc.temporal`
before the binning pipeline was restructured (docs/binning-workflow.md):
`aggregate` with its "mon", "year_max" and "clim_max" modes, and the annual
means of the former `load_native` loaders (`annual_means`). The pipeline now
uses `etunc.temporal.aggregate`.
"""

from __future__ import annotations

from typing import Literal

import xarray as xr

from etunc.temporal import complete_years, period_str


Aggregation = Literal["mon", "year", "clim", "year_max", "clim_max"]


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


def annual_mean(da: xr.DataArray, require_all_months: bool) -> xr.DataArray:
    """
    Annual mean via `aggregate`, which counts missing months as 0. With
    `require_all_months`, years with any missing month are NaN instead;
    otherwise only years without any valid month are NaN.
    """
    # The loaders pass require_all_months=False only for LAI, so a missing LAI month counts
    # as LAI = 0. ET, pr and Rn need all 12 months, since a 0 would bias the flux low
    n_valid = da.notnull().groupby("time.year").sum()
    with xr.set_options(keep_attrs=True):
        ann = aggregate(da, "year")
        return ann.where(n_valid == 12 if require_all_months else n_valid > 0)


def annual_means(monthly: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """
    Loaded annual means ([member,] year, lat, lon) of monthly binning
    variables {key: (..., time, lat, lon)} over complete years only, as every
    `load_native` takes them. A missing month counts as 0 for "lai", while
    every other key (ET, pr, Rn) needs all 12 months.
    """
    out = {}
    for k, da in monthly.items():
        # Drop partial first/last years: they would bias annual means toward the months present
        da = da.sel(time=da.time.dt.year.isin(complete_years(da)))
        # Key rule for which gridcell-years are binned: a missing LAI month (winter gaps at
        # high latitudes) counts as 0, while ET, pr and Rn need all 12 months
        ann = annual_mean(da, require_all_months=(k != "lai"))
        out[k] = ann.transpose(..., "lat", "lon").load()
    return out


def print_annual(da: xr.DataArray, label: str) -> None:
    """One-line summary of annual means `da` (year, lat, lon): dims, shape, period, range and units."""
    print(
        f"{label}: {da.dims} {da.shape} {period_str(da.year.values)} "
        f"[{float(da.min()):0.3g}, {float(da.max()):0.3g}] {da.attrs.get('units', '?')}"
    )


