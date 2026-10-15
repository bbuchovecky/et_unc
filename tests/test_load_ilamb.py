"""
Tests that pin the behavior of the ILAMB product loaders in `etunc.load.ilamb`:
`load_native` (raw monthly fields for the binning pipeline, scripts/prep_obs.py)
and `load_ilamb` (monthly, regridded; used by make_mask.py and, with
`wm2=True`, obs_et_availability.py).
Run from the project root with:

    python -m pytest tests/test_load_ilamb.py

Small synthetic ILAMB-style NetCDF files only; takes a few seconds.
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
    """ILAMB_DATA_ROOT in tmp_path, with fake 1 deg "FAKE" lai, pr and et products."""
    monkeypatch.setattr(il, "ILAMB_DATA_ROOT", tmp_path)
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


# ------------------------------------------------------------------
# load_native: raw monthly fields for binning
# ------------------------------------------------------------------

def test_load_native_raw_monthly(ilamb_root):
    """
    The monthly field over the time slice on the native grid formatted to lon [-180, 180] and lat ascending,
    in native units, with gaps and undecoded fill values as NaN.
    """
    pr = il.load_native("pr", "FAKE", slice("2001-01", "2002-12"))
    assert pr.dims == ("time", "lat", "lon") and pr.sizes["time"] == 24
    assert pr.sizes["lon"] == LON_1DEG.size and pr.lon.min() < 0 and (np.diff(pr.lat) > 0).all()
    assert pr.attrs["units"] == "mm/day"
    lon = lambda x: ((x + 180) % 360) - 180  # noqa: E731
    for glat, glon, month in (PR_GAP, PR_FILL):
        assert np.isnan(float(pr.sel(time=month, lat=glat, lon=lon(glon))))
    assert int(pr.isnull().sum()) == 2
    with pytest.raises(FileNotFoundError, match="no data"):
        il.load_native("pr", "FAKE", slice("1990-01", "1990-12"))


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
