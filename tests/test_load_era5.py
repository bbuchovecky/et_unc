"""
Tests of the ERA5 loaders in `etunc.load.era5` used by scripts/bin_obs.py:
variable selection in `load_era5`, `available_years`, `to_wm2` and
`load_era5_annual` (annual means, land masking, conservative regridding).
Run from the project root with:

    python -m pytest tests/test_load_era5.py

Synthetic GDEX-like files on a coarse global grid (2.5 deg, regridded to
10 deg) in tmp_path; takes a few seconds.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import etunc.config as config
import etunc.grid as rg
import etunc.load.era5 as le


RES = 10  # target spacing [deg]; any spacing that divides 180
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
    landfrac = landfrac.expand_dims(lat=LAT).assign_coords(lat=LAT, lon=LON).astype("f4")
    monkeypatch.setattr(le, "load_era5_grid", lambda: xr.Dataset({"LANDFRAC": landfrac}))
    return root


def test_load_era5_selects_code_variable(gdex):
    """With `utc_date` next to the field, load_era5 returns the field named by the code, with lat/lon dims."""
    da = le.load_era5("mer", "2000-01", "2001-12", kind="meanflux")
    assert isinstance(da, xr.DataArray)
    assert da.name == "MER"
    assert da.dims == ("time", "lat", "lon")
    assert da.sizes["time"] == 24


def test_available_years(gdex):
    """Only year directories count."""
    assert le.available_years("meanflux") == [2000, 2001, 2002]


def test_to_wm2():
    """ET is -mer x L, pr is mtpr x L and rns is net SW + net LW, all in W/m2."""
    one = xr.DataArray([1.0])
    fields = {code: one * v for code, v in VALUES.items()}
    L = config.LATENT_HEAT_VAPORIZATION
    for v, expected in (("et", 2e-5 * L), ("pr", 3e-5 * L), ("rns", 90.0)):
        out = le.to_wm2(v, fields)
        np.testing.assert_allclose(out, expected)
        assert out.attrs["units"] == "W/m2"


@pytest.mark.parametrize("variable, expected", [
    ("et", 2e-5 * config.LATENT_HEAT_VAPORIZATION),
    ("pr", 3e-5 * config.LATENT_HEAT_VAPORIZATION),
    ("rns", 90.0),
])
def test_load_era5_annual(gdex, variable, expected):
    """
    Complete years only, on the target grid within LAT_BNDS, with the constant
    value on land target cells and NaN over ocean.
    """
    ann = le.load_era5_annual(variable, (2000, 2002), res=RES)
    grid = rg.target_grid(RES).sel(lat=config.LAT_BNDS)

    assert ann.name == variable
    assert ann.dims == ("year", "lat", "lon")
    assert ann.year.values.tolist() == [2000, 2001]  # 2002 has 6 months
    np.testing.assert_array_equal(ann.lat, grid.lat)
    np.testing.assert_array_equal(ann.lon, grid.lon)
    assert ann.attrs["units"] == "W/m2"

    # Land is lon [-180, -90): target cells inside it keep the value, cells outside are NaN
    land = (ann.lon > -180) & (ann.lon < -90)
    np.testing.assert_allclose(ann.where(land, drop=True), expected, rtol=1e-6)
    assert ann.where(~land, drop=True).isnull().all()


def test_regrid_annual_coast_na_thres():
    """A target cell that is part land is the mean of its land cells, or NaN if more than na_thres is not land."""
    lat, lon = np.arange(-88.75, 90, 2.5), np.arange(-178.75, 180, 2.5)
    # 4 x 4 native cells per 10 deg target cell; land in the first native column of each target cell (25%),
    # and in the first three west of lon 0 (75%)
    col = ((lon + 180) % 10) // 2.5
    land_lon = (col == 0) | ((col <= 2) & (lon < 0))
    landfrac = xr.DataArray(np.broadcast_to(land_lon, (lat.size, lon.size)).astype("f4"),
                            coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    # Land values 1, ocean values 100 (must not leak into land cells)
    ann = xr.where(landfrac > 0.5, 1.0, 100.0).expand_dims(year=[2000]).assign_attrs(units="W/m2")

    out = le.regrid_annual(ann, landfrac, RES, na_thres=0.5)
    west, east = out.lon < 0, out.lon > 0
    np.testing.assert_allclose(out.where(west, drop=True), 1.0)  # 75% land: kept
    assert out.where(east, drop=True).isnull().all()             # 25% land: NaN
    assert out.attrs["units"] == "W/m2"
