"""
Tests for etunc/temporal.py: annual means, climatologies and time-period helpers. Run from the project
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
def test_aggregate_year_missing_month(require_all_months, capsys):
    """
    A NaN month counts as 0 in the days-weighted annual mean (LAI rule); with `require_all_months` (ET, P, Rn)
    that gridcell-year is NaN instead. A year with no valid month is NaN either way, and the number of
    gridcell-years with 1-11 valid months is printed and stored in attrs.
    """
    times = pd.date_range("2000-01-01", "2002-12-01", freq="MS")
    values = np.full(len(times), 2.0)
    values[3] = np.nan          # 2000-04 missing
    values[24:] = np.nan        # 2002 entirely missing
    ann = temporal.aggregate(monthly(times, values), "year", require_all_months=require_all_months, label="x")
    assert ann.dims == ("year", "lat", "lon") and ann.attrs["units"] == "W/m2"
    assert ann.attrs["n_partial_gridcell_years"] == 2  # 2000 at both gridcells
    np.testing.assert_allclose(ann.sel(year=2001), 2.0)
    assert ann.sel(year=2002).isnull().all()
    if require_all_months:
        assert ann.sel(year=2000).isnull().all()
    else:
        np.testing.assert_allclose(ann.sel(year=2000), 2.0 * (366 - 30) / 366)  # April (30 days) as 0, leap year
    assert "x: annual means of 3 complete years; 2 gridcell-years with partial (1-11) valid months" in capsys.readouterr().out


def test_aggregate_year_days_weighted_and_complete_years():
    """
    Annual means weight each month by its days, drop partial years from the time axis, keep extra dims, and
    return loaded data even for dask input.
    """
    times = pd.date_range("2000-01-01", "2001-06-01", freq="MS")  # 2001 is partial
    days = times.days_in_month.to_numpy(dtype=float)
    field = monthly(times, days).transpose("lat", "time", "lon").expand_dims(member=[0]).chunk()
    ann = temporal.aggregate(field, "year")
    assert ann.dims == ("member", "year", "lat", "lon")
    assert ann.year.values.tolist() == [2000]
    assert ann.chunks is None  # loaded
    d = days[:12]
    np.testing.assert_allclose(ann, np.sum(d * d) / np.sum(d))


def test_aggregate_clim_reports_missing_years(capsys):
    """
    The climatology is the mean over years, skipping a gridcell's NaN years; gridcells with some but not all
    years valid are counted.
    """
    ann = xr.DataArray([[1.0, 2.0], [3.0, np.nan], [5.0, np.nan]], dims=("year", "lat"),
                       coords={"year": [2000, 2001, 2002]}, name="lai")
    clim = temporal.aggregate(ann, "clim")
    np.testing.assert_allclose(clim, [3.0, 2.0])
    assert clim.attrs == {"n_years": 3, "n_gridcells_missing_years": 1}
    assert "lai: climatology of 3 years; 1 gridcells have missing years (fewest valid years: 1)" in \
        capsys.readouterr().out


def test_aggregate_errors():
    """'year' needs `time`, 'clim' needs `year`, and other modes raise."""
    times = pd.date_range("2000-01-01", "2000-12-01", freq="MS")
    with pytest.raises(ValueError, match="time"):
        temporal.aggregate(monthly(times).isel(time=0), "year")
    with pytest.raises(ValueError, match="year"):
        temporal.aggregate(monthly(times), "clim")
    with pytest.raises(ValueError, match="Unknown"):
        temporal.aggregate(monthly(times), "year_max")


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

