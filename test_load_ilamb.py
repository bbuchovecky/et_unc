"""
Tests that pin the behavior of the ILAMB product loaders before the refactor
merges them into `etunc.load.ilamb`: `ilamb_binned_et.load_product` (annual
means), `mask.load_ilamb` and `obs_et_availability.load_ilamb` (monthly). Run
from the project root with:

    python -m pytest test_load_ilamb.py

Small synthetic ILAMB-style NetCDF files only; takes a few seconds.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import etunc.config as config
import ilamb_binned_et as ib
import mask as mk
import obs_et_availability as oa
import etunc.grid as rg


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


def field(value, gaps=(), dtype="float64"):
    """(time, lat, lon) constant field on the 1 deg grid, with `gaps` = [(lat, lon, month, fill)]."""
    da = xr.DataArray(
        np.full((TIME.size, LAT_1DEG.size, LON_1DEG.size), value, dtype=dtype),
        dims=("time", "lat", "lon"),
        coords={"time": TIME, "lat": LAT_1DEG, "lon": LON_1DEG},
    )
    for lat, lon, month, fill in gaps:
        da.loc[{"time": month, "lat": lat, "lon": lon}] = fill
    return da


def write_product(root, variable, product, name, da, units):
    """Write `da` as an ILAMB file and register it in ib.PRODUCTS[variable][product]."""
    relpath = f"{variable}/{product}/{name}.nc"
    (root / variable / product).mkdir(parents=True, exist_ok=True)
    da.rename(name).assign_attrs(units=units).to_dataset().to_netcdf(root / relpath)
    ib.PRODUCTS[variable][product] = (relpath, name)


@pytest.fixture
def ilamb_root(tmp_path, monkeypatch):
    """ILAMB_DATA_ROOT in tmp_path, with fake 1 deg "FAKE" lai, pr and et products."""
    monkeypatch.setattr(ib, "ILAMB_DATA_ROOT", tmp_path)
    for variable in ib.PRODUCTS:
        monkeypatch.setitem(ib.PRODUCTS, variable, dict(ib.PRODUCTS[variable]))
    write_product(tmp_path, "lai", "FAKE", "lai", field(LAI, [(*LAI_GAP, np.nan)]), "1")
    write_product(tmp_path, "pr", "FAKE", "pr", field(PR, [(*PR_GAP, np.nan), (*PR_FILL, FILL)]), "mm/day")
    write_product(tmp_path, "et", "FAKE", "et", field(ET, [(*PR_GAP, np.nan)]), "kg m-2 s-1")
    return tmp_path


def target_lat_lon():
    grid = ib.TARGET_GRID.sel(lat=config.LAT_BNDS)
    return grid.lat.values, grid.lon.values


def near(lat, lon):
    """(lat, lon) mask of the target points whose bilinear stencil uses the source cell at (lat, lon)."""
    lon = ((lon + 180) % 360) - 180
    tlat, tlon = target_lat_lon()
    return (np.abs(tlat - lat) < 1)[:, None] & (np.abs(tlon - lon) < 1)[None, :]


def bilinear(da):
    """Bilinear regrid of `da` (on the formatted 1 deg grid) onto the 0.5 deg target grid within LAT_BNDS."""
    return rg.bilinear_regridder(da, ib.TARGET_RES)(da).sel(lat=config.LAT_BNDS)


def formatted(da):
    """`da` with lon in [-180, 180] and lat ascending, written out independently of rg.format_grid."""
    return da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon").sortby("lat")


# ------------------------------------------------------------------
# ib.load_product: annual means on the 0.5 deg grid
# ------------------------------------------------------------------

def test_load_product_grid_years_and_units(ilamb_root):
    """
    Annual means have dims (year, lat, lon) on the exact 0.5 deg grid within LAT_BNDS, the incomplete final year
    is dropped, and LAI units are set to m2/m2.
    """
    ann = ib.load_product("lai", "FAKE")
    assert ann.name == "lai"
    assert ann.dims == ("year", "lat", "lon")
    assert ann.year.values.tolist() == [2001, 2002]  # incomplete 2003 dropped
    assert ann.attrs["units"] == "m2/m2"             # set for LAI, whatever the file says
    tlat, tlon = target_lat_lon()
    np.testing.assert_array_equal(ann.lat, tlat)
    np.testing.assert_array_equal(ann.lon, tlon)


def test_load_product_averages_before_regridding(ilamb_root):
    """Annual mean first, then bilinear regrid. Regridding each month first gives different values."""
    ann = ib.load_product("lai", "FAKE")

    # A missing LAI month counts as 0, so the source cell's 2001 mean is LAI * (1 - July weight)
    src = formatted(field(LAI))
    src_ann = xr.concat([src.isel(time=0, drop=True)] * 2, dim=pd.Index([2001, 2002], name="year")).copy()
    lat, lon, _ = LAI_GAP
    src_ann.loc[{"year": 2001, "lat": lat, "lon": lon}] = LAI * (1 - JULY_WEIGHT)
    expected = bilinear(src_ann)
    np.testing.assert_allclose(ann.values, expected.values, rtol=1e-12)

    # Regridding each month first spreads the gap to every target point next to the source cell
    hit = near(lat, lon)
    assert hit.sum() == 16
    month_first = LAI * (1 - JULY_WEIGHT)
    np.testing.assert_allclose(ann.sel(year=2001).values[~hit], LAI, rtol=1e-12)
    assert np.all(np.abs(ann.sel(year=2001).values[hit] - month_first) > 1e-3)
    np.testing.assert_allclose(ann.sel(year=2002).values, LAI, rtol=1e-12)


def test_load_product_flux_needs_all_months_and_masks_fill(ilamb_root):
    """
    A pr year with a missing month or a fill value is NaN around that cell; elsewhere mm/day is converted to
    W/m2.
    """
    ann = ib.load_product("pr", "FAKE")
    assert ann.attrs["units"] == "W/m2"
    assert ann.year.values.tolist() == [2001, 2002]

    # A year with a missing month or a fill value is NaN at that source cell, so the
    # bilinear regrid makes the target points around it NaN; elsewhere mm/day -> W/m2
    wm2 = PR / 86400 * config.LATENT_HEAT_VAPORIZATION
    for year, (lat, lon, _) in [(2001, PR_GAP), (2002, PR_FILL)]:
        vals = ann.sel(year=year).values
        hit = near(lat, lon)
        assert np.isnan(vals[hit]).all()
        np.testing.assert_allclose(vals[~hit], wm2, rtol=1e-12)


def test_load_product_snaps_near_grid_without_regridding(ilamb_root):
    """A 0.5 deg product off by float round-off is not regridded, only given the exact target coords."""
    grid = ib.TARGET_GRID
    rng = np.random.default_rng(0)
    time = pd.date_range("2001-01-01", periods=12, freq="MS")
    data = rng.uniform(0, 5, (time.size, grid.sizes["lat"], grid.sizes["lon"]))
    da = xr.DataArray(
        data, dims=("time", "lat", "lon"),
        coords={"time": time, "lat": grid.lat.values + 1e-5, "lon": grid.lon.values - 1e-5},
    )
    write_product(ilamb_root, "lai", "FAKE05", "lai", da, "1")
    ann = ib.load_product("lai", "FAKE05")

    tlat, tlon = target_lat_lon()
    np.testing.assert_array_equal(ann.lat, tlat)
    np.testing.assert_array_equal(ann.lon, tlon)
    days = np.asarray(time.days_in_month, dtype=float)
    expected = np.tensordot(days / days.sum(), data, axes=1)[grid.lat.values > -58]
    np.testing.assert_allclose(ann.isel(year=0).values, expected, rtol=1e-12)


def test_load_product_off_grid_raises(ilamb_root):
    """A product whose grid is off by more than the tolerance raises."""
    grid = ib.TARGET_GRID
    time = pd.date_range("2001-01-01", periods=12, freq="MS")
    da = xr.DataArray(
        np.ones((time.size, grid.sizes["lat"], grid.sizes["lon"])), dims=("time", "lat", "lon"),
        coords={"time": time, "lat": grid.lat.values + 0.01, "lon": grid.lon.values},
    )
    write_product(ilamb_root, "lai", "OFFGRID", "lai", da, "1")
    with pytest.raises(ValueError, match="differ from the reference grid"):
        ib.load_product("lai", "OFFGRID")


# ------------------------------------------------------------------
# Monthly loaders: mask.load_ilamb and obs_et_availability.load_ilamb
# ------------------------------------------------------------------

def test_mask_load_ilamb_regrids_each_month_in_native_units(ilamb_root):
    """
    mask.load_ilamb keeps every month in TIME_SLICE (partial years too) in native units, and regrids each month,
    so a missing month is NaN around that cell.
    """
    da = mk.load_ilamb("lai", "FAKE")
    assert da.dims == ("time", "lat", "lon")
    np.testing.assert_array_equal(da.time.values, TIME.values)  # partial 2003 kept (TIME_SLICE only)
    assert da.attrs["units"] == "1"                             # native units
    tlat, tlon = target_lat_lon()
    np.testing.assert_array_equal(da.lat, tlat)
    np.testing.assert_array_equal(da.lon, tlon)

    lat, lon, month = LAI_GAP
    hit = near(lat, lon)
    july = da.sel(time=month).squeeze("time").values
    assert np.isnan(july[hit]).all()
    np.testing.assert_allclose(july[~hit], LAI, rtol=1e-12)
    np.testing.assert_allclose(da.sel(time="2001-08").values, LAI, rtol=1e-12)


def test_mask_load_ilamb_masks_fill_values(ilamb_root):
    """
    mask.load_ilamb sets fill values to NaN before regridding, so they do not spread into neighboring cells.
    """
    da = mk.load_ilamb("pr", "FAKE")
    assert da.attrs["units"] == "mm/day"
    lat, lon, month = PR_FILL
    hit = near(lat, lon)
    vals = da.sel(time=month).squeeze("time").values
    assert np.isnan(vals[hit]).all()
    np.testing.assert_allclose(vals[~hit], PR, rtol=1e-12)
    assert float(da.max()) == pytest.approx(PR)


def test_availability_load_ilamb_wm2_on_month_axis(ilamb_root):
    """
    obs_et_availability.load_ilamb converts ET to W/m2, reindexes it to MONTHS (NaN outside the data), and
    regrids each month.
    """
    da = oa.load_ilamb("FAKE")
    np.testing.assert_array_equal(da.time.values, oa.MONTHS.values)  # reindexed to 2000-2014
    assert da.attrs["units"] == "W/m2"
    present = (oa.MONTHS >= TIME[0]) & (oa.MONTHS <= TIME[-1])
    assert da.isel(time=~present).isnull().all()

    lat, lon, month = PR_GAP
    hit = near(lat, lon)
    gap = da.sel(time=month).squeeze("time").values
    assert np.isnan(gap[hit]).all()
    np.testing.assert_allclose(gap[~hit], ET * config.LATENT_HEAT_VAPORIZATION, rtol=1e-12)


def test_availability_load_ilamb_no_data_raises(ilamb_root):
    """A product with no data in TIME_SLICE raises."""
    old = field(ET).assign_coords(time=pd.date_range("1990-01-01", periods=TIME.size, freq="MS"))
    write_product(ilamb_root, "et", "OLD", "et", old, "kg m-2 s-1")
    with pytest.raises(FileNotFoundError, match="no data"):
        oa.load_ilamb("OLD")
