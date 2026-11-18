"""
Tests of the ERA5 loaders in `etunc.load.era5` used by scripts/prep_obs.py:
variable selection in `load_era5`, `load_native` (raw monthly fields) and
`load_land_fraction`.
Run from the project root with:

    python -m pytest tests/test_load_era5.py

Synthetic GDEX-like files on a coarse global grid (2.5 deg) in tmp_path;
takes a few seconds.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import etunc.load.era5 as le


# Native grid as in the GDEX files: lat descending and including the poles, lon in [0, 360)
LAT = np.arange(90, -90.1, -2.5)
LON = np.arange(0, 360, 2.5)
# Constant monthly values of each code (ERA5 sign convention: positive downward)
VALUES = {"mer": -2e-5, "mtpr": 3e-5, "msnswrf": 150.0, "msnlwrf": -60.0}
CODES = {"mer": "235_043", "mtpr": "235_055", "msnswrf": "235_037", "msnlwrf": "235_038"}
# Land is the western hemisphere in [-180, -90) only (lon 180-270 in GDEX coords)
LAND_LON = (180, 270)


def write_year(root, code, year, months=12):
    """One GDEX-like yearly file with the code's field and `utc_date`, as the real files have."""
    time = pd.date_range(f"{year}-01-01", periods=months, freq="MS")
    shape = (months, LAT.size, LON.size)
    ds = xr.Dataset(
        {
            code.upper(): (("time", "latitude", "longitude"), np.full(shape, VALUES[code], dtype="f4")),
            "utc_date": ("time", time.strftime("%Y%m%d00").astype(int)),
        },
        coords={"time": time, "latitude": LAT, "longitude": LON},
    )
    ds[code.upper()].attrs["units"] = "kg m**-2 s**-1" if code in ("mer", "mtpr") else "W m**-2"
    d = root / str(year)
    d.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(d / f"e5.moda.fc.sfc.meanflux.{CODES[code]}_{code}.ll025sc.{year}010100_{year}120100.nc")


@pytest.fixture
def gdex(tmp_path, monkeypatch):
    """GDEX tree with 2000 and 2001 complete and 2002 partial (6 months), plus a land fraction."""
    root = tmp_path / "meanflux"
    for code in VALUES:
        write_year(root, code, 2000)
        write_year(root, code, 2001)
        write_year(root, code, 2002, months=6)
    (root / "README").write_text("not a year")
    monkeypatch.setitem(le.GDEX_ROOTS, "meanflux", root)

    lon = xr.DataArray(LON, dims="lon")
    landfrac = xr.where((lon >= LAND_LON[0]) & (lon < LAND_LON[1]), 1.0, 0.0)
    # Scalar `time` coord, as in the real grid file
    landfrac = landfrac.expand_dims(lat=LAT).assign_coords(lat=LAT, lon=LON, time=np.datetime64("1979-01-01")).astype("f4")
    monkeypatch.setattr(le, "load_era5_grid", lambda: xr.Dataset({"LANDFRAC": landfrac}))
    return root


def test_load_era5_selects_code_variable(gdex):
    """With `utc_date` next to the field, load_era5 returns the field named by the code, with lat/lon dims."""
    da = le.load_era5("mer", "2000-01", "2001-12", kind="meanflux")
    assert isinstance(da, xr.DataArray)
    assert da.name == "MER"
    assert da.dims == ("time", "lat", "lon")
    assert da.sizes["time"] == 24


def test_load_native_raw_months_on_formatted_grid(gdex):
    """
    load_native returns each code's monthly field over the time slice only, with lon in [-180, 180] and lat
    ascending, in native units and sign (mer stays negative).
    """
    raw = le.load_native(("mer", "msnswrf"), slice("2000-03", "2001-12"))
    assert list(raw) == ["mer", "msnswrf"]
    mer = raw["mer"]
    assert mer.dims == ("time", "lat", "lon")
    assert mer.sizes["time"] == 22 and str(mer.time.values[0])[:7] == "2000-03"
    assert np.all(np.diff(mer.lat) > 0) and mer.lon.min() >= -180 and mer.lon.max() < 180
    np.testing.assert_allclose(mer, VALUES["mer"])
    assert mer.attrs["units"] == "kg m**-2 s**-1"


def test_load_land_fraction(gdex):
    """LANDFRAC on the formatted grid without the grid file's scalar time coord, and its grid with cell edges."""
    landfrac, src_grid = le.load_land_fraction()
    assert landfrac.dims == ("lat", "lon") and "time" not in landfrac.coords
    assert np.all(np.diff(landfrac.lat) > 0) and landfrac.lon.min() >= -180
    land = (landfrac.lon >= -180) & (landfrac.lon < -90)
    np.testing.assert_array_equal(landfrac.isel(lat=0) == 1, land)
    np.testing.assert_array_equal(src_grid.lat, landfrac.lat)
    assert src_grid.sizes["lat_b"] == landfrac.sizes["lat"] + 1


# ------------------------------------------------------------------
# ERA5-Land
# ------------------------------------------------------------------

# Evaporation in m of water per day, positive downward (so negative), as in the CDS monthly means
E_VALUE = -2e-3


def write_land_year(root, year):
    """One CDS-like ERA5-Land yearly file: `valid_time`, lat descending with the poles, the CDS aux coords."""
    time = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
    e = np.full((12, LAT.size, LON.size), E_VALUE, dtype="f4")
    e[:, :, LON < 180] = np.nan  # ocean (NaN) in the eastern hemisphere
    ds = xr.Dataset(
        {"e": (("valid_time", "latitude", "longitude"), e, {"units": "m of water equivalent"})},
        coords={"valid_time": time, "latitude": LAT, "longitude": LON, "number": 0,
                "expver": ("valid_time", np.full(12, "0001"))},
    )
    d = root / "monthly" / str(year)
    d.mkdir(parents=True)
    ds.to_netcdf(d / "data_stream-moda.nc")


@pytest.fixture
def era5_land(tmp_path, monkeypatch):
    """ERA5-Land tree with 2000 and 2001, plus a file that is not a year directory."""
    root = tmp_path / "era5-land"
    for year in (2000, 2001):
        write_land_year(root, year)
    (root / "monthly" / "README").write_text("not a year")
    monkeypatch.setattr(le, "ERA5_LAND_ROOT", root)
    return root


def test_load_era5_land_raw_months_on_formatted_grid(era5_land):
    """
    load_era5_land returns the variable's monthly field over the time slice only, with time/lat/lon dims, lon
    in [-180, 180) and lat ascending, in native units and sign (e stays negative), without the CDS aux coords.
    """
    e = le.load_era5_land("e", slice("2000-03", "2001-06"))
    assert e.dims == ("time", "lat", "lon")
    assert e.sizes["time"] == 16 and str(e.time.values[0])[:7] == "2000-03" and str(e.time.values[-1])[:7] == "2001-06"
    assert np.all(np.diff(e.lat) > 0) and e.lon.min() >= -180 and e.lon.max() < 180
    assert "expver" not in e.coords and "number" not in e.coords
    assert e.attrs["units"] == "m of water equivalent"
    # Land (GDEX lon 180-360) is now lon [-180, 0); ocean stays NaN
    np.testing.assert_allclose(e.where(e.lon < 0, drop=True), E_VALUE)
    assert e.where(e.lon >= 0, drop=True).isnull().all()


def test_load_era5_land_open_slice_and_missing_year(era5_land):
    """An open-ended slice reads every year on disk; a requested year without a file raises."""
    assert le.load_era5_land("e", slice(None, None)).sizes["time"] == 24
    with pytest.raises(FileNotFoundError, match="2002"):
        le.load_era5_land("e", slice("2001-01", "2002-12"))
