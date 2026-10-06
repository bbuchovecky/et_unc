"""
Tests for load_obs.py on small synthetic per-year NetCDF files. Run from the
project root with:

    python -m pytest test_load_obs.py
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import load_obs as lo


# ------------------------------------------------------------------
# Fixtures and helpers
# ------------------------------------------------------------------

LAT = np.array([45.05, 44.95, 44.85])   # descending, like GLEAM and PML
LON = np.array([0.05, 180.05, 359.95])  # 0-360, to check wrapping


def write_year(path, var, year, freq="monthly", lat=LAT, lon=LON, units="mm/month",
               lat_name="latitude", lon_name="longitude"):
    time = pd.date_range(f"{year}-01", periods=12, freq="MS") if freq == "monthly" else pd.DatetimeIndex([f"{year}-01-01"])
    data = np.full((len(time), len(lat), len(lon)), float(year), dtype="float32")
    ds = xr.Dataset(
        {var: (("time", lat_name, lon_name), data, {"units": units}), "crs": ((), 0)},
        coords={"time": time, lat_name: lat, lon_name: lon},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


@pytest.fixture
def spec(tmp_path):
    """PML-like layout: two versions, AVHRR-era files offset by ~1e-5 deg."""
    s = lo.ObsDataset(
        name="fake",
        root=tmp_path,
        template="{version}/{freq}/{var}/FAKE-{version}_{var}_{year}.nc",
        versions=("v1", "v2"),
        freqs={"monthly": "", "yearly": ""},
        lat_name="latitude",
        lon_name="longitude",
    )
    for year in range(2000, 2004):
        offset = 1.07e-5 if year < 2002 else 0.0
        for var in ("ET", "GPP"):
            write_year(tmp_path / f"v1/monthly/{var}/FAKE-v1_{var}_{year}.nc", var, year,
                       lat=LAT - offset, lon=LON + offset)
        write_year(tmp_path / f"v1/yearly/ET/FAKE-v1_ET_{year}.nc", "ET", year, freq="yearly", units="mm/year")
    return s


# ------------------------------------------------------------------
# File discovery
# ------------------------------------------------------------------

def test_list_variables_and_years(spec):
    assert lo.list_variables(spec) == ["ET", "GPP"]
    assert lo.list_variables(spec, freq="yearly") == ["ET"]
    assert lo.list_years(spec, "ET") == [2000, 2001, 2002, 2003]
    assert lo.list_variables(spec, version="v2") == []


def test_find_files_by_year(spec):
    files = lo.find_files(spec, "ET", slice("2001-06", "2002-02"))
    assert [f.name for f in files] == ["FAKE-v1_ET_2001.nc", "FAKE-v1_ET_2002.nc"]


def test_find_files_warns_missing_years(spec):
    with pytest.warns(UserWarning, match=r"\[1998, 1999\]"):
        files = lo.find_files(spec, "ET", slice("1998", "2000"))
    assert len(files) == 1


def test_scan_does_not_match_other_variable_prefix(tmp_path):
    """A {var} of "E" must not pick up "Ec" or "Ep" files."""
    s = lo.ObsDataset("g", tmp_path, "{var}/{var}_{year}_{freq_tag}.nc", ("v",), {"monthly": "MO"})
    for var in ("E", "Ec"):
        write_year(tmp_path / f"{var}/{var}_2000_MO.nc", var, 2000)
    assert [f.name for f in lo.find_files(s, "E")] == ["E_2000_MO.nc"]


def test_bad_names(spec):
    with pytest.raises(KeyError, match="Unknown dataset"):
        lo.get_dataset("nope")
    with pytest.raises(ValueError, match="unknown version"):
        lo.load_obs(spec, "ET", version="v9")
    with pytest.raises(ValueError, match="unknown freq"):
        lo.load_obs(spec, "ET", freq="8-day")
    with pytest.raises(FileNotFoundError, match="Available variables"):
        lo.load_obs(spec, "LAI")


def test_registered_datasets():
    assert {"gleam", "pml"} <= set(lo.DATASETS)
    assert lo.get_dataset("PML") is lo.DATASETS["pml"]


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def test_load_obs_single_grid_across_offset_files(spec):
    da = lo.load_obs(spec, "ET", slice("2001-07", "2002-06"))
    assert da.dims == ("time", "lat", "lon")
    assert da.sizes == {"time": 12, "lat": 3, "lon": 3}
    np.testing.assert_array_equal(da.lat, [44.85, 44.95, 45.05])
    np.testing.assert_array_equal(da.lon, [-179.95, -0.05, 0.05])
    assert np.isfinite(da.values).all()
    np.testing.assert_array_equal(da.isel(lat=0, lon=0), [2001] * 6 + [2002] * 6)
    assert da.attrs == {"units": "mm/month", "dataset": "fake", "version": "v1", "frequency": "monthly", "res": "native"}


def test_load_obs_lat_bnds_and_chunks(spec):
    da = lo.load_obs(spec, "ET", lat_bnds=slice(44.9, 50), chunks={"lat": 1, "time": 12})
    np.testing.assert_array_equal(da.lat, [44.95, 45.05])
    assert da.chunks[1] == (1, 1)


def test_load_obs_mismatched_grids_raise(spec, tmp_path):
    write_year(tmp_path / "v2/monthly/ET/FAKE-v2_ET_2000.nc", "ET", 2000)
    write_year(tmp_path / "v2/monthly/ET/FAKE-v2_ET_2001.nc", "ET", 2001, lat=LAT + 0.1)
    with pytest.raises(ValueError):
        lo.load_obs(spec, "ET", version="v2")


def test_load_obs_preprocess_hook(spec):
    def relabel(ds, var, version, freq):
        ds[var].attrs["units"] = f"{version}-{freq}"
        return ds

    s = lo.ObsDataset(**{**spec.__dict__, "preprocess": relabel})
    assert lo.load_obs(s, "ET", freq="yearly").attrs["units"] == "v1-yearly"


def test_load_obs_template_without_year(tmp_path):
    s = lo.ObsDataset("one", tmp_path, "{var}_{version}.nc", ("v",), {"monthly": ""},
                      lat_name="lat", lon_name="lon")
    time = pd.date_range("2000-01", periods=36, freq="MS")
    xr.Dataset({"et": (("time", "lat", "lon"), np.ones((36, 2, 2)))},
               coords={"time": time, "lat": [1.0, 0.0], "lon": [0.0, 1.0]}).to_netcdf(tmp_path / "et_v.nc")
    da = lo.load_obs(s, "et", slice("2001", "2001"))
    assert da.sizes["time"] == 12
    np.testing.assert_array_equal(da.lat, [0.0, 1.0])


def test_load_obs_period_files(tmp_path):
    """SiTH-like layout: multi-year files, {freq_dir} directories, non-CF encoding, (time, lon, lat) dims."""
    s = lo.ObsDataset("period", tmp_path, "{freq_dir}/{var}.P{version}.A{year_start}_{year_end}.{freq_tag}.nc",
                      ("v2",), {"monthly": "M"}, freq_dirs={"monthly": "Monthly"}, preprocess=lo._decode_sith)
    for y0, y1 in [(2000, 2001), (2004, 2005)]:
        time = pd.date_range(f"{y0}-01", periods=24, freq="ME")
        data = np.full((24, 3, 2), 150, dtype="int32")
        data[:, 0, 0] = -999
        ds = xr.Dataset({"ET": (("time", "lon", "lat"), data, {"units": "mm month-1", "scale factor": "100"})},
                        coords={"time": time, "lon": LON, "lat": [0.05, -0.05]}, attrs={"Fill Value": "-999"})
        (tmp_path / "Monthly").mkdir(exist_ok=True)
        ds.to_netcdf(tmp_path / f"Monthly/ET.Pv2.A{y0}_{y1}.M.nc")

    assert lo.list_variables(s) == ["ET"]
    assert lo.list_years(s, "ET") == [2000, 2001, 2004, 2005]
    assert [f.name for f in lo.find_files(s, "ET", slice("2001-06", "2001-12"))] == ["ET.Pv2.A2000_2001.M.nc"]
    with pytest.warns(UserWarning, match=r"\[2002, 2003\]"):
        assert len(lo.find_files(s, "ET", slice("2001", "2004"))) == 2

    da = lo.load_obs(s, "ET", slice("2001-07", "2004-06"))
    assert da.dims == ("time", "lat", "lon") and da.dtype == np.float32
    assert da.sizes["time"] == 12
    np.testing.assert_array_equal(da.lat, [-0.05, 0.05])
    assert np.isnan(da.sel(lat=0.05, lon=0.05)).all()
    np.testing.assert_allclose(da.sel(lat=-0.05), 1.5)
    assert "scale factor" not in da.attrs and da.attrs["units"] == "mm month-1"


def test_at_resolution():
    spec = lo.get_dataset("pml")
    assert lo.at_resolution(spec) is spec
    regridded = lo.at_resolution("pml", "0.5")
    assert regridded.root == lo.REGRID_ROOT / "pml-v2.2" / "0.5deg"
    assert (regridded.lat_name, regridded.lon_name) == ("lat", "lon")
    assert regridded.template == spec.template and regridded.preprocess is spec.preprocess
    with pytest.raises(ValueError, match="res"):
        lo.at_resolution("pml", "0.25")


def test_load_obs_regridded(spec, monkeypatch, tmp_path):
    """res="1" reads REGRID_ROOT/<root dir name>/1deg/<same relative path>, with lat/lon coords."""
    monkeypatch.setattr(lo, "REGRID_ROOT", tmp_path / "regridded")
    lat, lon = np.array([45.5, 44.5]), np.array([-0.5, 0.5])
    for year in (2001, 2002):
        write_year(tmp_path / f"regridded/{tmp_path.name}/1deg/v1/monthly/ET/FAKE-v1_ET_{year}.nc", "ET", year,
                   lat=lat, lon=lon, lat_name="lat", lon_name="lon")
    assert lo.list_years(spec, "ET", res="1") == [2001, 2002]
    assert lo.list_variables(spec, res="1") == ["ET"]
    da = lo.load_obs(spec, "ET", slice("2001", "2002"), res="1")
    assert da.sizes == {"time": 24, "lat": 2, "lon": 2}
    np.testing.assert_array_equal(da.lat, [44.5, 45.5])
    assert da.attrs["res"] == "1"
    assert lo.load_obs(spec, "ET", slice("2001", "2001")).attrs["res"] == "native"
    with pytest.raises(FileNotFoundError, match=r"\(0\.5\)"):
        lo.load_obs(spec, "ET", res="0.5")


def test_gleam_units_fix():
    ds = xr.Dataset({"E": ("time", [1.0], {"units": "mm.day-1"})})
    out = lo._fix_gleam_units(ds, "E", "v4.3b", "yearly")
    assert out["E"].attrs["units"] == "mm.year-1"


@pytest.fixture
def gleam(tmp_path):
    """GLEAM layout: v4.3a with NaN ocean, v4.3b with 0 ocean labelled mm.day-1."""
    for version, ocean, units in (("v4.3a", np.nan, "mm.month-1"), ("v4.3b", 0.0, "mm.day-1")):
        for year in (2003, 2004):
            path = tmp_path / f"{version}/monthly/E/E_{year}_GLEAM_{version}_MO.nc"
            write_year(path, "E", year, units=units, lat_name="lat", lon_name="lon")
            ds = xr.load_dataset(path)
            ds["E"][:, 0, 1] = ocean
            ds.to_netcdf(path)
    return lo.replace(lo.get_dataset("gleam"), root=tmp_path)


@pytest.mark.parametrize("chunks", ["auto", None])
def test_gleam_v43b_ocean_masked_like_v43a(gleam, chunks):
    a = lo.load_obs(gleam, "E", slice("2003", "2004"), version="v4.3a", chunks=chunks)
    b = lo.load_obs(gleam, "E", slice("2003", "2004"), version="v4.3b", chunks=chunks)
    xr.testing.assert_equal(a.isnull(), b.isnull())
    assert int(b.isnull().sum()) == 24  # one ocean cell, 24 months
    assert float(b.min()) == 2003.0
    assert b.attrs["units"] == "mm.month-1"


def test_gleam_v43b_needs_v43a_file(gleam, tmp_path):
    (tmp_path / "v4.3a/monthly/E/E_2004_GLEAM_v4.3a_MO.nc").unlink()
    with pytest.raises(FileNotFoundError, match="ocean mask"):
        lo.load_obs(gleam, "E", slice("2004", "2004"), version="v4.3b").load()


# ------------------------------------------------------------------
# Units
# ------------------------------------------------------------------

@pytest.mark.parametrize("units, time, days", [
    ("mm/month", ["2000-02-01", "2001-02-01"], [29, 28]),
    ("mm.month-1", ["2000-01-31"], [31]),
    ("mm/year", ["2000-01-01", "2001-12-31"], [366, 365]),
    ("mm.year-1", ["2004-12-31"], [366]),
    ("mm/day", ["2000-01-01"], [1]),
    ("mm.day-1", ["2000-01-01"], [1]),
])
def test_accumulation_to_flux(units, time, days):
    da = xr.DataArray(np.ones(len(time)), dims="time", coords={"time": pd.DatetimeIndex(time)},
                      attrs={"units": units, "long_name": "x"})
    out = lo.accumulation_to_flux(da)
    np.testing.assert_allclose(out, 1 / (np.array(days) * 86400))
    assert out.attrs == {"units": "kg m-2 s-1", "long_name": "x"}


def test_accumulation_to_flux_bad_units():
    da = xr.DataArray([1.0], dims="time", coords={"time": pd.DatetimeIndex(["2000-01-01"])},
                      attrs={"units": "gC/m2/month"})
    with pytest.raises(ValueError, match="Cannot convert"):
        lo.accumulation_to_flux(da)
