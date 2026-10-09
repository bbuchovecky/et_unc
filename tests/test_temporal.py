"""
Tests for etunc/temporal.py helpers used by the drivers. Run from the project
root with:

    python -m pytest tests/test_temporal.py

Synthetic data only; takes a few seconds.
"""

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import etunc.temporal as temporal


def monthly(times, values=None):
    """(time, lat, lon) field with one value per time step on a 1x2 grid."""
    values = np.arange(1.0, len(times) + 1) if values is None else np.asarray(values, dtype=float)
    data = np.repeat(values[:, None, None], 2, axis=2)
    return xr.DataArray(data, dims=("time", "lat", "lon"),
                        coords={"time": times, "lat": [0.0], "lon": [0.0, 1.0]}, attrs={"units": "W/m2"})


def test_complete_years():
    """Only years with all 12 calendar months in the time axis count; a partial year or one missing month does not."""
    times = pd.date_range("2000-01-01", "2002-06-01", freq="MS")
    times = times[times != pd.Timestamp("2001-03-01")]
    assert temporal.complete_years(monthly(times)) == [2000]


@pytest.mark.parametrize("require_all_months", [True, False])
def test_annual_mean_missing_month(require_all_months):
    """
    A NaN month counts as 0 in the days-weighted annual mean; with `require_all_months` that year is NaN
    instead. A year with no valid month is NaN either way.
    """
    times = pd.date_range("2000-01-01", "2002-12-01", freq="MS")
    values = np.full(len(times), 2.0)
    values[3] = np.nan          # 2000-04 missing
    values[24:] = np.nan        # 2002 entirely missing
    ann = temporal.annual_mean(monthly(times, values), require_all_months=require_all_months)
    assert ann.dims == ("year", "lat", "lon") and ann.attrs["units"] == "W/m2"
    np.testing.assert_allclose(ann.sel(year=2001), 2.0)
    assert ann.sel(year=2002).isnull().all()
    if require_all_months:
        assert ann.sel(year=2000).isnull().all()
    else:
        np.testing.assert_allclose(ann.sel(year=2000), 2.0 * (366 - 30) / 366)  # April (30 days) as 0, leap year


def test_period_str_and_shared_years():
    """period_str spans the first to last year; shared_years is the sorted intersection of `year` coords."""
    assert temporal.period_str([2009, 2003, 2005]) == "200301-200912"
    a = xr.DataArray(np.zeros(4), dims="year", coords={"year": [2003, 2001, 2002, 2004]})
    b = xr.DataArray(np.zeros(3), dims="year", coords={"year": [2004, 2002, 2005]})
    assert temporal.shared_years(a, b) == [2002, 2004]


def test_yearly_to_annual():
    """One time step per year becomes a `year` dim; two steps in one year raise."""
    times = pd.to_datetime(["2001-07-01", "2002-07-01"])
    out = temporal.yearly_to_annual(monthly(times))
    assert out.dims == ("year", "lat", "lon") and "time" not in out.coords
    np.testing.assert_array_equal(out.year, [2001, 2002])
    with pytest.raises(ValueError, match="more than one time step in a year"):
        temporal.yearly_to_annual(monthly(pd.to_datetime(["2001-01-01", "2001-07-01"])))


def test_on_month_axis():
    """Mid-month (or cftime) stamps move to the first of the month and are reindexed onto `months`; two in a month raise."""
    months = pd.date_range("2000-01-01", "2000-06-01", freq="MS")
    times = [cftime.DatetimeNoLeap(2000, 2, 15), cftime.DatetimeNoLeap(2000, 4, 16)]
    out = temporal.on_month_axis(monthly(times, [5.0, 7.0]), months)
    np.testing.assert_array_equal(out.time.values, months.values)
    np.testing.assert_array_equal(out.isel(lat=0, lon=0).values, [np.nan, 5.0, np.nan, 7.0, np.nan, np.nan])
    with pytest.raises(ValueError, match="more than one time step in a month"):
        temporal.on_month_axis(monthly(pd.to_datetime(["2000-02-01", "2000-02-15"])), months)


def test_to_yyyymm():
    """Both numpy datetime64 and cftime time stamps format as YYYYMM."""
    da = monthly(pd.to_datetime(["2003-07-16"]))
    assert temporal.to_yyyymm(da.time.isel(time=0)) == "200307"
    da = monthly([cftime.DatetimeNoLeap(1850, 1, 16)])
    assert temporal.to_yyyymm(da.time.isel(time=0)) == "185001"
