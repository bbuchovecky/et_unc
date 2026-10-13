"""
Tests that pin the behavior of the ILAMB product loaders in `etunc.load.ilamb`:
`load_native` and `load_annual` (annual means for binning; `load_annual` =
`etunc.grid.regrid_annual` of `load_native`) and `load_ilamb` (monthly; was
`mask.load_ilamb` and, with `wm2=True`, `obs_et_availability.load_ilamb`).
Run from the project root with:

    python -m pytest tests/test_load_ilamb.py

Small synthetic ILAMB-style NetCDF files only, with a synthetic land mask in
place of Natural Earth (`etunc.grid.land_mask`); takes a few seconds.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import etunc.config as config
import etunc.grid as rg
import etunc.load.ilamb as il
import etunc.temporal as temporal


# ------------------------------------------------------------------
# Fixtures and helpers
# ------------------------------------------------------------------

# 1 deg source grid as in some ILAMB files: lat descending, lon in [0, 360]
LAT_1DEG = np.arange(89.5, -90, -1.0)
LON_1DEG = np.arange(0.5, 360, 1.0)
TIME = pd.date_range("2001-01-01", "2003-06-01", freq="MS")  # 2003 is incomplete

LAI = 2.0      # m2/m2
PR = 3.0       # mm/day
ET = 2e-5      # kg m-2 s-1
FILL = 9.97e36  # undecoded fill value, as in GPCCv2018

# Source cells (lat, lon in [0, 360]) and months with missing or fill values
LAI_GAP = (10.5, 20.5, "2001-07")
PR_GAP = (30.5, 40.5, "2001-02")
PR_FILL = (-20.5, 200.5, "2002-03")

JULY_WEIGHT = 31 / 365  # weight of July in the days-in-month weighted annual mean of 2001

# Synthetic land: lon in (LAND_WEST, 180] (formatted coords), so the gaps above are on
# land. On a 0.5 deg grid it reaches into the 1 deg target cell [-1, 0], which is then half land
LAND_WEST = -0.3

# Settings the old drivers used, now passed explicitly
RES = 0.5                                           # 1 deg products are regridded onto 0.5 deg
GRID = rg.target_grid(RES)
MASK_TIME_SLICE = slice("1982-01", "2025-12")      # mask.TIME_SLICE
AV_TIME_SLICE = slice("2000-01", "2014-12")        # obs_et_availability.TIME_SLICE
AV_MONTHS = pd.date_range(AV_TIME_SLICE.start, AV_TIME_SLICE.stop, freq="MS")
ALL = slice(None, None)


def availability_ilamb(product):
    """obs_et_availability.load_ilamb: monthly ET in W/m2 on the AV_MONTHS axis."""
    return temporal.on_month_axis(il.load_ilamb("et", product, AV_TIME_SLICE, RES, wm2=True), AV_MONTHS)


def fake_land_mask(grid):
    """Stand-in for rg.land_mask (Natural Earth): land where lon > LAND_WEST."""
    return (grid["lat"] > -1000) & (grid["lon"] > LAND_WEST)  # (lat, lon) bool


def field(value, gaps=(), lat=LAT_1DEG, lon=LON_1DEG):
    """(time, lat, lon) constant field, with `gaps` = [(lat, lon, month, fill)]."""
    da = xr.DataArray(
        np.full((TIME.size, lat.size, lon.size), value, dtype="float64"),
        dims=("time", "lat", "lon"),
        coords={"time": TIME, "lat": lat, "lon": lon},
    )
    for glat, glon, month, fill in gaps:
        da.loc[{"time": month, "lat": glat, "lon": glon}] = fill
    return da


def write_product(root, variable, product, name, da, units):
    """Write `da` as an ILAMB file and register it in il.PRODUCTS[variable][product]."""
    relpath = f"{variable}/{product}/{name}.nc"
    (root / variable / product).mkdir(parents=True, exist_ok=True)
    da.rename(name).assign_attrs(units=units).to_dataset().to_netcdf(root / relpath)
    il.PRODUCTS[variable][product] = (relpath, name)


@pytest.fixture
def ilamb_root(tmp_path, monkeypatch):
    """ILAMB_DATA_ROOT in tmp_path, with fake 1 deg "FAKE" lai, pr and et products, and the synthetic land mask."""
    monkeypatch.setattr(il, "ILAMB_DATA_ROOT", tmp_path)
    monkeypatch.setattr(il, "land_mask", fake_land_mask)
    for variable in il.PRODUCTS:
        monkeypatch.setitem(il.PRODUCTS, variable, dict(il.PRODUCTS[variable]))
    write_product(tmp_path, "lai", "FAKE", "lai", field(LAI, [(*LAI_GAP, np.nan)]), "1")
    write_product(tmp_path, "pr", "FAKE", "pr", field(PR, [(*PR_GAP, np.nan), (*PR_FILL, FILL)]), "mm/day")
    write_product(tmp_path, "et", "FAKE", "et", field(ET, [(*PR_GAP, np.nan)]), "kg m-2 s-1")
    return tmp_path


def target_lat_lon(res=RES):
    grid = rg.target_grid(res).sel(lat=config.LAT_BNDS)
    return grid.lat.values, grid.lon.values


def inside(lat, lon, res=RES):
    """(lat, lon) mask of the target cells inside the 1 deg source cell centered at (lat, lon in [0, 360])."""
    lon = ((lon + 180) % 360) - 180
    tlat, tlon = target_lat_lon(res)
    return (np.abs(tlat - lat) < 0.5)[:, None] & (np.abs(tlon - lon) < 0.5)[None, :]


def land(res=RES):
    """(lat, lon) mask of the target cells whose 1 deg source cell is land."""
    tlat, tlon = target_lat_lon(res)
    return np.broadcast_to(tlon > 0, (tlat.size, tlon.size))


# ------------------------------------------------------------------
# load_native and load_annual: annual means for binning
# ------------------------------------------------------------------

def test_load_annual_grid_years_and_units(ilamb_root):
    """
    Annual means {variable: (year, lat, lon)} on the exact 0.5 deg grid within LAT_BNDS, the incomplete final
    year dropped, and LAI units set to m2/m2.
    """
    out = il.load_annual("lai", "FAKE", ALL, RES)
    assert list(out) == ["lai"]
    ann = out["lai"]
    assert ann.name == "lai"
    assert ann.dims == ("year", "lat", "lon")
    assert ann.year.values.tolist() == [2001, 2002]  # incomplete 2003 dropped
    assert ann.attrs["units"] == "m2/m2"             # set for LAI, whatever the file says
    tlat, tlon = target_lat_lon()
    np.testing.assert_array_equal(ann.lat, tlat)
    np.testing.assert_array_equal(ann.lon, tlon)


def test_load_annual_land_only_and_lai_gap_counts_as_zero(ilamb_root):
    """
    Values only on land (the native land mask), and a missing LAI month counts as 0, in the 4 target cells
    inside the source cell. Conservative regridding from 1 to 0.5 deg copies each source cell, up to ~1e-5
    (ESMF cell sides are great circles, so a 0.5 deg cell is not exactly inside its 1 deg cell).
    """
    ann = il.load_annual("lai", "FAKE", ALL, RES)["lai"]
    lat, lon, _ = LAI_GAP
    gap = inside(lat, lon)
    assert gap.sum() == 4
    for year in (2001, 2002):
        vals = ann.sel(year=year).values
        assert np.isnan(vals[~land()]).all()
        expected = LAI * (1 - JULY_WEIGHT) if year == 2001 else LAI
        np.testing.assert_allclose(vals[gap], expected, rtol=1e-4)
        np.testing.assert_allclose(vals[land() & ~gap], LAI, rtol=1e-4)


def test_load_annual_averages_before_regridding(ilamb_root):
    """
    0.5 deg product onto 1 deg: annual mean on the native grid first, so a
    missing LAI month in one of the 4 source cells lowers the target cell.
    Regridding each month first would fill the July gap with the other 3
    cells and give LAI.
    """
    lat05, lon05 = np.arange(89.75, -90, -0.5), np.arange(0.25, 360, 0.5)
    da = field(LAI, [(10.25, 20.25, "2001-07", np.nan)], lat=lat05, lon=lon05)
    write_product(ilamb_root, "lai", "FAKE05", "lai", da, "1")
    ann = il.load_annual("lai", "FAKE05", ALL, 1.0)["lai"]
    cell = float(ann.sel(year=2001, lat=10.5, lon=20.5))
    # One of 4 (nearly equal-area) source cells has LAI * (1 - July weight)
    assert cell == pytest.approx(LAI * (1 - JULY_WEIGHT / 4), rel=1e-4)
    assert abs(cell - LAI) > 1e-3
    np.testing.assert_allclose(ann.sel(year=2002, lat=10.5, lon=20.5), LAI)


def test_load_annual_masks_ocean_before_regridding(ilamb_root):
    """
    A global product (as GPCP or CERES) is masked to land on its native grid,
    so ocean values never enter a coastal target cell: the half-land 1 deg
    cell [-1, 0] keeps the land value, the ocean cell next to it is NaN.
    (na_thres 0.6: at exactly half land, great-circle cell sides decide 0.5.)
    """
    lat05, lon05 = np.arange(89.75, -90, -0.5), np.arange(0.25, 360, 0.5)
    land05 = ((lon05 + 180) % 360) - 180 > LAND_WEST
    da = field(PR, lat=lat05, lon=lon05).where(xr.DataArray(land05, dims="lon", coords={"lon": lon05}), 100 * PR)
    write_product(ilamb_root, "pr", "GLOBAL", "pr", da, "mm/day")
    ann = il.load_annual("pr", "GLOBAL", ALL, 1.0, na_thres=0.6)["pr"]
    wm2 = PR / 86400 * config.LATENT_HEAT_VAPORIZATION
    np.testing.assert_allclose(ann.sel(lon=-0.5), wm2, rtol=1e-12)  # half land, no ocean value
    assert ann.sel(lon=-1.5).isnull().all()
    np.testing.assert_allclose(ann.where(ann.notnull(), wm2), wm2, rtol=1e-12)


def test_load_annual_flux_needs_all_months(ilamb_root):
    """A pr year with a missing month is NaN in that source cell's target cells; elsewhere mm/day -> W/m2."""
    ann = il.load_annual("pr", "FAKE", ALL, RES)["pr"]
    assert ann.attrs["units"] == "W/m2"
    assert ann.year.values.tolist() == [2001, 2002]
    wm2 = PR / 86400 * config.LATENT_HEAT_VAPORIZATION
    lat, lon, _ = PR_GAP
    gap = inside(lat, lon)
    vals = ann.sel(year=2001).values
    assert np.isnan(vals[gap]).all()
    np.testing.assert_allclose(vals[land() & ~gap], wm2, rtol=1e-12)
    np.testing.assert_allclose(ann.sel(year=2002).values[land()], wm2, rtol=1e-12)


def test_load_annual_is_regrid_annual_of_load_native(ilamb_root):
    """load_annual is rg.regrid_annual(*load_native(...)); load_native is native, land-masked and complete years."""
    native, native_mask, src_grid = il.load_native("pr", "FAKE")
    assert list(native) == ["pr"]
    assert native["pr"].dims == ("year", "lat", "lon") and native["pr"].sizes["lon"] == LON_1DEG.size
    assert native["pr"].year.values.tolist() == [2001, 2002]
    assert native["pr"].lon.min() < 0 and (np.diff(native["pr"].lat) > 0).all()  # formatted grid
    assert native["pr"].where(~native_mask).isnull().all()
    assert native_mask.dtype == bool and src_grid.sizes["lat_b"] == LAT_1DEG.size + 1
    out = rg.regrid_annual(native, native_mask, src_grid, RES)
    xr.testing.assert_identical(il.load_annual("pr", "FAKE", ALL, RES)["pr"], out["pr"])


def test_load_annual_near_grid_product(ilamb_root):
    """A 0.5 deg product off by float round-off is regridded onto itself: the values barely change."""
    rng = np.random.default_rng(0)
    data = rng.uniform(0, 5, (TIME.size, GRID.sizes["lat"], GRID.sizes["lon"]))
    da = xr.DataArray(
        data, dims=("time", "lat", "lon"),
        coords={"time": TIME, "lat": GRID.lat.values + 1e-5, "lon": GRID.lon.values - 1e-5},
    )
    write_product(ilamb_root, "lai", "FAKE05", "lai", da, "1")
    ann = il.load_annual("lai", "FAKE05", ALL, RES)["lai"]

    tlat, tlon = target_lat_lon()
    np.testing.assert_array_equal(ann.lat, tlat)
    np.testing.assert_array_equal(ann.lon, tlon)
    days = np.asarray(TIME[:12].days_in_month, dtype=float)
    expected = np.tensordot(days / days.sum(), data[:12], axes=1)[GRID.lat.values > -58]
    expected = np.where(tlon > LAND_WEST, expected, np.nan)
    np.testing.assert_allclose(ann.isel(year=0).values, expected, atol=1e-3)


# ------------------------------------------------------------------
# load_ilamb (was mask.load_ilamb and obs_et_availability.load_ilamb): monthly
# ------------------------------------------------------------------

def test_mask_load_ilamb_regrids_each_month_in_native_units(ilamb_root):
    """
    load_ilamb (as in make_mask.py) keeps every month in TIME_SLICE (partial years too) in native units, and
    conservatively regrids each month, so a missing month is NaN in that source cell's target cells only.
    No land mask is applied.
    """
    da = il.load_ilamb("lai", "FAKE", MASK_TIME_SLICE, RES)
    assert da.dims == ("time", "lat", "lon")
    np.testing.assert_array_equal(da.time.values, TIME.values)  # partial 2003 kept (TIME_SLICE only)
    assert da.attrs["units"] == "1"                             # native units
    tlat, tlon = target_lat_lon()
    np.testing.assert_array_equal(da.lat, tlat)
    np.testing.assert_array_equal(da.lon, tlon)

    lat, lon, month = LAI_GAP
    gap = inside(lat, lon)
    july = da.sel(time=month).squeeze("time").values
    assert np.isnan(july[gap]).all()
    np.testing.assert_allclose(july[~gap], LAI, rtol=1e-12)
    np.testing.assert_allclose(da.sel(time="2001-08").values, LAI, rtol=1e-12)


def test_mask_load_ilamb_masks_fill_values(ilamb_root):
    """load_ilamb sets fill values to NaN before regridding, so they do not spread into other cells."""
    da = il.load_ilamb("pr", "FAKE", MASK_TIME_SLICE, RES)
    assert da.attrs["units"] == "mm/day"
    lat, lon, month = PR_FILL
    gap = inside(lat, lon)
    vals = da.sel(time=month).squeeze("time").values
    assert np.isnan(vals[gap]).all()
    np.testing.assert_allclose(vals[~gap], PR, rtol=1e-12)
    assert float(da.max()) == pytest.approx(PR)


def test_load_ilamb_native(ilamb_root):
    """With res=None, load_ilamb returns the native grid (formatted, within LAT_BNDS), fill values as NaN."""
    da = il.load_ilamb("pr", "FAKE", MASK_TIME_SLICE, None)
    assert da.sizes["lon"] == LON_1DEG.size
    np.testing.assert_array_equal(da.lat, np.sort(LAT_1DEG[(LAT_1DEG >= -58) & (LAT_1DEG <= 90)]))
    assert (np.diff(da.lon) > 0).all() and da.lon.min() < 0
    lat, lon, month = PR_FILL
    assert np.isnan(float(da.sel(time=month, lat=lat, lon=lon - 360).squeeze()))
    assert float(da.max()) == pytest.approx(PR)


def test_availability_load_ilamb_wm2_on_month_axis(ilamb_root):
    """
    load_ilamb with wm2=True (as in obs_et_availability.py) converts ET to W/m2, reindexes it to MONTHS (NaN
    outside the data), and regrids each month.
    """
    da = availability_ilamb("FAKE")
    np.testing.assert_array_equal(da.time.values, AV_MONTHS.values)  # reindexed to 2000-2014
    assert da.attrs["units"] == "W/m2"
    present = (AV_MONTHS >= TIME[0]) & (AV_MONTHS <= TIME[-1])
    assert da.isel(time=~present).isnull().all()

    lat, lon, month = PR_GAP
    gap = inside(lat, lon)
    vals = da.sel(time=month).squeeze("time").values
    assert np.isnan(vals[gap]).all()
    np.testing.assert_allclose(vals[~gap], ET * config.LATENT_HEAT_VAPORIZATION, rtol=1e-12)


def test_availability_load_ilamb_no_data_raises(ilamb_root):
    """A product with no data in TIME_SLICE raises."""
    old = field(ET).assign_coords(time=pd.date_range("1990-01-01", periods=TIME.size, freq="MS"))
    write_product(ilamb_root, "et", "OLD", "et", old, "kg m-2 s-1")
    with pytest.raises(FileNotFoundError, match="no data"):
        availability_ilamb("OLD")
