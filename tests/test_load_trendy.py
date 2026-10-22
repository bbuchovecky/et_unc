"""
Tests of the TRENDY loaders in `etunc.load.trendy` used by scripts/prep_trendy.py:
file lookup (`models`, `find_file`, `missing_variables`), `load_native` (raw
monthly fields with a rebuilt time axis) and `load_forcing` (monthly totals or
means of the 6-hourly forcing in crujra3/ and Radiation/).
Run from the project root with:

    python -m pytest tests/test_load_trendy.py

Synthetic files in tmp_path that mimic the quirks of the real ones (undecodable
time units, month-end stamps, descending lat, lon in [0, 360), ELM's index-less
dims); takes a few seconds.
"""

from datetime import timedelta

import cftime
import numpy as np
import pytest
import xarray as xr

import etunc.load.trendy as lt


# Coarse native grid as in many TRENDY files: lat descending, lon in [0, 360)
LAT = np.arange(80, -81, -20.0)
LON = np.arange(0, 360, 45.0)
N_MONTHS = 24  # Jan 2023 - Dec 2024, ending in LAST_MONTH as every GCB2025 run


def write(root, model, folder, filename, variable, time, time_attrs, *, value=1e-5,
          lat=LAT, lon=LON, dims=("time", "latitude", "longitude")):
    """One TRENDY-like file root/model/folder/filename with a constant field `variable`."""
    shape = (len(time), len(lat), len(lon))
    ds = xr.Dataset(
        {variable: (dims, np.full(shape, value, dtype="f4"), {"units": "kg m-2 s-1"})},
        coords={dims[0]: ("time", np.asarray(time, dtype="f8"), time_attrs), dims[1]: lat, dims[2]: lon},
    )
    path = root / model / folder / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    # Raw time values and attrs as given, not re-encoded
    ds.to_netcdf(path, encoding={"time": {"dtype": "f8"}})
    return path


def month_dates(day, calendar="noleap", start=(2023, 1), n=N_MONTHS):
    """Dates on `day` of each month from `start`; day="end" for the last day of the month."""
    out = []
    y, m = start
    for _ in range(n):
        if day == "end":
            out.append(cftime.datetime(y + (m == 12), m % 12 + 1, 1, calendar=calendar) - timedelta(days=1))
        else:
            out.append(cftime.datetime(y, m, day, calendar=calendar))
        y, m = y + (m == 12), m % 12 + 1
    return out


def days_since(dates, ref="1700-12-31", calendar="noleap"):
    units = f"days since {ref}"
    return cftime.date2num(dates, units, calendar=calendar), {"units": units, "calendar": calendar}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, "ROOT", tmp_path)
    return tmp_path


# ------------------------------------------------------------------
# Files
# ------------------------------------------------------------------

def test_file_lookup(root):
    """
    models() lists only directories with an experiment subdirectory, CLASSIC S2 is read from correctedS2/,
    files in subdirectories (LPJ-GUESS S3/v1.0/) are ignored, and two matching files raise.
    """
    t, attrs = days_since(month_dates(15))
    write(root, "CLM", "S3", "CLM6.0_S3_evapotrans.nc", "evapotrans", t, attrs)
    write(root, "CLASSIC", "S2", "CLASSIC_S2_lai.nc", "lai", t, attrs)
    corrected = write(root, "CLASSIC", "correctedS2", "CLASSIC_S2_lai.nc", "lai", t, attrs)
    write(root, "LPJ-GUESS", "S3/v1.0", "LPJ-GUESS_S3_lai.nc", "lai", t, attrs)
    (root / "crujra3").mkdir()
    (root / "Radiation").mkdir()

    assert lt.models() == ["CLASSIC", "CLM", "LPJ-GUESS"]
    assert lt.find_file("CLASSIC", "S2", "lai") == corrected
    assert lt.find_file("LPJ-GUESS", "S3", "lai") is None
    assert lt.missing_variables("CLM", "S3", ["evapotrans", "lai"]) == ["lai"]

    write(root, "CLM", "S3", "CLM6.0b_S3_evapotrans.nc", "evapotrans", t, attrs)
    with pytest.raises(ValueError, match="more than one"):
        lt.find_file("CLM", "S3", "evapotrans")


# ------------------------------------------------------------------
# load_native
# ------------------------------------------------------------------

def test_load_native_month_end_stamps_on_formatted_grid(root):
    """
    Month-end stamps (CLASSIC) become first-of-month stamps in the file's calendar, only the months of the
    time slice are kept, and the grid has lat ascending and lon in [-180, 180], with values and units unchanged.
    """
    t, attrs = days_since(month_dates("end"))
    write(root, "CLASSIC", "S3", "CLASSIC_S3_evapotrans.nc", "evapotrans", t, attrs)

    da = lt.load_native("CLASSIC", "S3", slice("2024-01", "2024-06"), ["evapotrans"])["evapotrans"]
    assert da.dims == ("time", "lat", "lon")
    assert [(d.year, d.month, d.day) for d in da.time.values] == [(2024, m, 1) for m in range(1, 7)]
    assert da.time.values[0].calendar == "noleap"
    assert np.all(np.diff(da.lat) > 0)
    assert da.lon.min() >= -180 and da.lon.max() < 180
    assert da.attrs["units"] == "kg m-2 s-1"
    np.testing.assert_allclose(da.values, 1e-5)


@pytest.mark.parametrize("time, attrs", [
    (np.arange(N_MONTHS), {"units": "months since 2023-1-16", "calendar": "365_day"}),  # CABLE-POP, LPJml
    (2023 + np.arange(N_MONTHS) / 12, {}),                                              # ELM-FATES: no units
    (2023 + (np.arange(N_MONTHS) + 0.5) / 12, {"units": "years since AD 0-Jan-1st"}),  # VISIT-UT
])
def test_load_native_undecodable_time_ends_in_last_month(root, time, attrs):
    """A time axis that cannot be CF-decoded is rebuilt as a monthly series ending in LAST_MONTH."""
    write(root, "M", "S3", "M_S3_lai.nc", "lai", time, attrs)
    da = lt.load_native("M", "S3", slice("2000-01", "2030-12"), ["lai"])["lai"]
    assert da.sizes["time"] == N_MONTHS
    first, last = da.time.values[0], da.time.values[-1]
    assert (first.year, first.month) == (2023, 1)
    assert (last.year, last.month) == (2024, 12)


def test_load_native_shifted_time_raises(root):
    """Stamps at the start of the next month (Jan stamped 1 Feb) disagree with the rebuilt axis."""
    t, attrs = days_since(month_dates(1, start=(2023, 2)), ref="1700-01-01")
    write(root, "M", "S3", "M_S3_lai.nc", "lai", t, attrs)
    with pytest.raises(ValueError, match="time step 0 is 2023-02"):
        lt.load_native("M", "S3", slice("2023-01", "2024-12"), ["lai"])


def test_load_native_elm_coordinates(root):
    """ELM: lat/lon dims without an index, with the coordinates in the variables latitude(lat), longitude(lon)."""
    t, attrs = days_since(month_dates(16), ref="0001-01-01")
    path = write(root, "ELM", "S3", "E3SM_S3_lai.nc", "lai", t, attrs, dims=("time", "lat", "lon"))
    # Replace the lat/lon coordinate variables by index-less dims and latitude/longitude variables
    ds = xr.open_dataset(path, decode_times=False).load()
    ds = ds.drop_vars(["lat", "lon"]).assign(latitude=("lat", LAT[::-1]), longitude=("lon", LON - 180))
    path.unlink()
    ds.to_netcdf(path)

    da = lt.load_native("ELM", "S3", slice("2023-01", "2024-12"), ["lai"])["lai"]
    np.testing.assert_array_equal(da.lat, LAT[::-1])
    np.testing.assert_array_equal(da.lon, LON - 180)
    assert "latitude" not in da.coords and "longitude" not in da.coords


def test_load_native_mismatched_grids_raise(root):
    """Variables of one model on different grids raise ValueError (prep_trendy.py skips the model)."""
    t, attrs = days_since(month_dates(15))
    write(root, "M", "S3", "M_S3_evapotrans.nc", "evapotrans", t, attrs)
    write(root, "M", "S3", "M_S3_lai.nc", "lai", t, attrs, lat=LAT[:-1])
    with pytest.raises(ValueError, match="grid shape"):
        lt.load_native("M", "S3", slice("2023-01", "2024-12"), ["evapotrans", "lai"])
    with pytest.raises(ValueError, match="no tran file"):
        lt.load_native("M", "S3", slice("2023-01", "2024-12"), ["tran"])


# ------------------------------------------------------------------
# load_forcing
# ------------------------------------------------------------------

def write_forcing(root, variable, year, day_values, units):
    """
    One yearly forcing file as in crujra3/ or Radiation/ (lt.FORCING): 6-hourly values in a 365-day calendar,
    stamped at the start of each 6 h, repeating `day_values` (the 4 steps of a day) on land and NaN on one
    ocean cell.
    """
    folder, filename, _ = lt.FORCING[variable]
    t_units = "days since 1901-01-01 00:00:00"
    start = cftime.date2num(cftime.datetime(year, 1, 1, calendar="365_day"), t_units, calendar="365_day")
    time = start + np.arange(365 * 4) / 4
    lat, lon = np.arange(-67.5, 90, 45.0), np.arange(-157.5, 180, 45.0)
    values = np.broadcast_to(np.tile(np.asarray(day_values, dtype=float), 365)[:, None, None], (time.size, lat.size, lon.size)).copy()
    values[:, 0, 0] = np.nan  # ocean
    ds = xr.Dataset(
        {variable: (("time", "lat", "lon"), values.astype("f4"), {"units": units})},
        coords={"time": ("time", time, {"units": t_units, "calendar": "365_day"}), "lat": lat, "lon": lon},
    )
    (root / folder).mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(root / folder / filename.format(year=year))


def test_load_forcing_pre_monthly_totals(root):
    """Precipitation totals of each month are summed (1 mm/6h -> 4 mm per day of the month), over the time slice only."""
    write_forcing(root, "pre", 2001, [1, 1, 1, 1], "mm/6h")
    write_forcing(root, "pre", 2002, [1, 1, 1, 1], "mm/6h")

    pr = lt.load_forcing("pre", slice("2001-11", "2002-02"))
    assert [(d.year, d.month, d.day) for d in pr.time.values] == [(2001, 11, 1), (2001, 12, 1), (2002, 1, 1), (2002, 2, 1)]
    assert pr.attrs["units"] == "mm/month"
    np.testing.assert_allclose(pr.isel(lat=1, lon=1).values, [4 * 30, 4 * 31, 4 * 31, 4 * 28])
    assert pr.isel(lat=0, lon=0).isnull().all()  # an ocean cell stays NaN, not a total of 0

    with pytest.raises(FileNotFoundError, match="2003"):
        lt.load_forcing("pre", slice("2002-01", "2003-12"))


@pytest.mark.parametrize("variable, units", [("dlwrf", "W/m^2"), ("fd", "1")])
def test_load_forcing_monthly_means(root, variable, units):
    """Fluxes (crujra3/) and fractions (Radiation/) are monthly means of every 6-hourly step, night included."""
    write_forcing(root, variable, 2001, [0, 1, 1, 0], units)  # e.g. a diffuse fraction of 0 at night

    da = lt.load_forcing(variable, slice("2001-01", "2001-12"))
    assert da.sizes["time"] == 12
    assert da.attrs["units"] == units
    np.testing.assert_allclose(da.isel(lat=1, lon=1).values, 0.5)
    assert da.isel(lat=0, lon=0).isnull().all()
