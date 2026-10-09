"""
Tests for the obs regridding of regrid_obs.py: grid.regrid_with_na_thres and
load.obs.output_path / regrid_file (the shared grids are tested in test_grid.py). Run from the project root with:

    python -m pytest tests/test_regrid_obs.py

The synthetic tests take a few seconds. The real-data tests compare global
area-weighted means of native and regridded GLEAM and PML months; they need
glade (skipped otherwise) and take ~6 min, mostly building the 0.1 deg -> 0.5
and 1 deg weights (cached across tests).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import etunc.load.obs as lo
import etunc.grid as rg


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def fine_field(data, lat0=10.0, lon0=20.0, res=0.1):
    """(lat, lon) DataArray on a 0.1 deg sub-domain whose SW corner is (lat0, lon0)."""
    nlat, nlon = data.shape
    return xr.DataArray(
        np.asarray(data, dtype="float32"),
        dims=("lat", "lon"),
        coords={
            "lat": np.round(lat0 + res * (np.arange(nlat) + 0.5), 4),
            "lon": np.round(lon0 + res * (np.arange(nlon) + 0.5), 4),
        },
        name="E",
        attrs={"units": "mm.month-1"},
    )


def cell_area(lat_b, lon_b):
    """Relative cell areas (lat, lon) on the sphere from 1-D bounds."""
    dsin = np.diff(np.sin(np.deg2rad(lat_b)))
    dlon = np.diff(np.deg2rad(lon_b))
    return np.outer(dsin, dlon)


def area_mean(values, area):
    valid = np.isfinite(values)
    return float(np.sum(np.where(valid, values, 0) * area) / np.sum(np.where(valid, area, 0)))


def coarse_cell(out, lat, lon):
    return float(out.sel(lat=lat, lon=lon).item())


# ------------------------------------------------------------------
# Regridding
# ------------------------------------------------------------------

@pytest.mark.parametrize("res", [0.5, 1.0])
def test_constant_field_and_unmapped_nan(res):
    """A constant stays constant; target cells outside the source domain are NaN, not 0."""
    da = fine_field(np.full((20, 20), 3.0))  # 10-12N, 20-22E
    out = rg.regrid_with_na_thres(da, rg.conservative_regridder(da, res))
    inside = out.sel(lat=slice(10, 12), lon=slice(20, 22))
    np.testing.assert_allclose(inside, 3.0, rtol=1e-6)
    assert int(out.notnull().sum()) == inside.size
    assert out.dtype == np.float32
    assert out.attrs["units"] == "mm.month-1"


@pytest.mark.parametrize("res", [0.5, 1.0])
def test_conserves_area_weighted_total(res):
    """The area-weighted total of a random field is the same before and after regridding."""
    rng = np.random.default_rng(0)
    da = fine_field(rng.uniform(0, 100, (20, 30)))  # 10-12N, 20-23E
    out = rg.regrid_with_na_thres(da, rg.conservative_regridder(da, res))
    src = rg.source_grid(da)
    tgt = rg.target_grid(res)
    native_total = np.sum(da.values * cell_area(src.lat_b.values, src.lon_b.values))
    regridded_total = np.nansum(out.values * cell_area(tgt.lat_b.values, tgt.lon_b.values))
    np.testing.assert_allclose(regridded_total, native_total, rtol=1e-5)


def test_missing_fraction_threshold():
    """1 deg cells: 40% missing keeps the mean of the valid cells, 60% missing is NaN."""
    # Values vary with lon only, so the area-weighted mean of valid columns is a plain mean.
    cols = np.arange(10, dtype="float32")
    data = np.tile(np.concatenate([cols, cols]), (10, 1))  # 10-11N, 20-22E
    data[:, :4] = np.nan          # cell (10.5, 20.5): 40% missing
    data[:, 10:16] = np.nan       # cell (10.5, 21.5): 60% missing
    da = fine_field(data)
    out = rg.regrid_with_na_thres(da, rg.conservative_regridder(da, 1.0))
    # ESMF uses great-circle cell edges, so equal-lon columns differ in area by ~1e-5
    np.testing.assert_allclose(coarse_cell(out, 10.5, 20.5), cols[4:].mean(), rtol=1e-4)
    assert np.isnan(coarse_cell(out, 10.5, 21.5))


def test_time_varying_missing_values():
    """The missing-value correction is applied per time step."""
    data = np.ones((2, 10, 10), dtype="float32")
    data[0, :, :3] = np.nan   # 30% missing -> kept
    data[1, :, :7] = np.nan   # 70% missing -> NaN
    da = xr.concat([fine_field(d) for d in data], dim="time")
    out = rg.regrid_with_na_thres(da, rg.conservative_regridder(da, 1.0))
    assert coarse_cell(out.isel(time=0), 10.5, 20.5) == pytest.approx(1.0)
    assert np.isnan(coarse_cell(out.isel(time=1), 10.5, 20.5))


def test_output_path_mirrors_source(monkeypatch, tmp_path):
    """The output path mirrors the source path under REGRID_ROOT/<dataset dir>/<res tag>."""
    monkeypatch.setattr(lo, "REGRID_ROOT", tmp_path)
    spec = lo.get_dataset("gleam")
    src = spec.root / "v4.3a/monthly/E/E_1980_GLEAM_v4.3a_MO.nc"
    assert lo.output_path(spec, "1deg", src) == tmp_path / "gleam-v4.3/1deg/v4.3a/monthly/E/E_1980_GLEAM_v4.3a_MO.nc"


@pytest.fixture
def sith_like(tmp_path):
    """SiTH-like 2000-2002 file (int32 x100, (time, lon, lat)) with on-disk time chunks of 10: (spec, ds)."""
    spec = lo.ObsDataset("per", tmp_path / "per", "{freq_dir}/{var}.P{version}.A{year_start}_{year_end}.{freq_tag}.nc",
                         ("v2",), {"monthly": "M"}, freq_dirs={"monthly": "Monthly"}, preprocess=lo._decode_sith)
    time = pd.date_range("2000-01", periods=36, freq="ME")
    data = np.broadcast_to((time.year.values * 100)[:, None, None], (36, 10, 10)).astype("int32")
    ds = xr.Dataset(
        {"ET": (("time", "lon", "lat"), data, {"units": "mm month-1", "scale factor": "100"})},
        coords={"time": time, "lon": 20.05 + 0.1 * np.arange(10), "lat": 10.95 - 0.1 * np.arange(10)},
        attrs={"Fill Value": "-999"},
    )
    (spec.root / "Monthly").mkdir(parents=True)
    # time chunks of 10 -> blocks of 10, 10, 10, 6 that straddle years
    ds.to_netcdf(spec.root / "Monthly/ET.Pv2.A2000_2002.M.nc", encoding={"ET": {"chunksizes": (10, 5, 5)}})
    return spec, ds


def test_regrid_file_multi_year(monkeypatch, tmp_path, sith_like):
    """SiTH-like file spanning 2000-2002 is regridded block by block into one output file."""
    monkeypatch.setattr(lo, "REGRID_ROOT", tmp_path / "out")
    spec, ds = sith_like
    ((src, years),) = lo.list_files(spec, "ET", "v2", "monthly").items()
    assert years == (2000, 2002)
    assert lo.regrid_file(spec, "ET", src, years, "v2", "monthly") is not None
    with xr.open_dataset(lo.output_path(spec, "1deg", src)) as out:
        assert out.ET.sizes == {"time": 36, "lat": 180, "lon": 360}
        np.testing.assert_array_equal(out.time, ds.time)
        np.testing.assert_allclose(out.ET.sel(lat=10.5, lon=20.5), np.repeat([2000.0, 2001.0, 2002.0], 12))
        assert np.isnan(out.ET.sel(lat=0.5, lon=0.5)).all()
        assert tuple(out.ET.attrs["src_shape"]) == (36, 10, 10)
        # Provenance: target grid, source state, preprocess and code versions
        attrs = out.ET.attrs
        assert attrs["target_grid"] == "1deg" and "res" not in attrs
        assert attrs["source_size_bytes"] == src.stat().st_size
        assert attrs["source_mtime"].endswith("UTC") and attrs["regrid_date"].endswith("UTC")
        assert attrs["load_preprocess"] == "_decode_sith"
        assert attrs["etunc_git_commit"] and attrs["xesmf_version"] and attrs["esmpy_version"]
        assert out.attrs["history"].splitlines()[0].endswith(f"(etunc {attrs['etunc_git_commit']})")
        assert out.attrs["Fill Value"] == "-999"  # source global attrs kept
    assert lo.output_path(spec, "0.5deg", src).exists()

    # Read back with the load functions
    back = lo.load_obs(spec, "ET", slice("2001", "2001"), res="1")
    assert back.sizes == {"time": 12, "lat": 180, "lon": 360}
    np.testing.assert_allclose(back.sel(lat=10.5, lon=20.5), 2001.0)
    assert lo.list_files(spec, "ET", "v2", "monthly", res="0.5") == {lo.output_path(spec, "0.5deg", src): (2000, 2002)}
    assert lo.regrid_file(spec, "ET", src, years, "v2", "monthly", overwrite=False) is None  # existing output skipped


@pytest.fixture
def per_year(tmp_path):
    """GLEAM-like per-year files 2000-2001 on a 0.1 deg patch, varying in space and time, with NaN."""
    spec = lo.ObsDataset("yr", tmp_path / "yr", "{version}/{freq}/{var}/{var}_{year}.nc", ("v1",), {"monthly": "M"})
    rng = np.random.default_rng(0)
    for year in (2000, 2001):
        time = pd.date_range(f"{year}-01", periods=12, freq="MS")
        data = rng.uniform(0, 100, (12, 20, 20)).astype("float32")
        data[:, :7, :5] = np.nan  # part of one 1 deg cell missing (kept), most of another (NaN at na_thres=0.5)
        data[:, 12:, 10:17] = np.nan
        ds = xr.Dataset({"E": (("time", "lat", "lon"), data, {"units": "mm/month"})},
                        coords={"time": time, "lat": np.round(11.95 - 0.1 * np.arange(20), 4),
                                "lon": np.round(20.05 + 0.1 * np.arange(20), 4)})
        path = spec.root / f"v1/monthly/E/E_{year}.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path)
    return spec


@pytest.mark.parametrize("which", ["per_year", "sith_like"])
def test_load_obs_regridded_matches_regrid_file(monkeypatch, tmp_path, request, which):
    """In-memory regridding equals the files regrid_file writes, read back with load_obs(res=)."""
    monkeypatch.setattr(lo, "REGRID_ROOT", tmp_path / "out")
    spec = request.getfixturevalue(which)
    if which == "sith_like":
        spec, var = spec[0], "ET"
    else:
        var = "E"
    version = spec.versions[0]
    for src, years in lo.list_files(spec, var, version, "monthly").items():
        lo.regrid_file(spec, var, src, years, version, "monthly")

    for res, saved_res in ((1.0, "1"), ("0.5deg", "0.5")):
        mem = lo.load_obs_regridded(spec, var, slice("2000", "2001"), res, version=version)
        saved = lo.load_obs(spec, var, slice("2000", "2001"), version=version, res=saved_res).load()
        xr.testing.assert_equal(mem, saved)
        assert mem.dtype == saved.dtype == np.float32
        assert mem.attrs["res"] == saved_res and mem.attrs["na_thres"] == rg.NA_THRES
        assert mem.attrs["dataset"] == spec.name and mem.attrs["version"] == version


def test_load_obs_regridded_time_slices(per_year, sith_like):
    """Month-precision slices and partial reads of a multi-year file return exactly the requested months."""
    mem = lo.load_obs_regridded(per_year, "E", slice("2000-03", "2001-02"), 1.0)
    np.testing.assert_array_equal(mem.time, pd.date_range("2000-03", "2001-02", freq="MS"))

    spec, ds = sith_like
    mem = lo.load_obs_regridded(spec, "ET", slice("2001", "2001"), 1.0)
    np.testing.assert_array_equal(mem.time, ds.time.sel(time="2001"))
    np.testing.assert_allclose(mem.sel(lat=10.5, lon=20.5), 2001.0)


def test_load_obs_regridded_missing_files(per_year):
    with pytest.raises(FileNotFoundError, match="No yr v1 monthly files"):
        lo.load_obs_regridded(per_year, "E", slice("1990", "1991"), 1.0)


# ------------------------------------------------------------------
# Real data: global area-weighted mean, native vs regridded
# ------------------------------------------------------------------

REAL_MONTHS = [
    ("gleam", "v4.3a", "E", "1980-01"),
    ("gleam", "v4.3a", "E", "2010-07"),
    ("pml", "V2.2c", "ET", "1982-01"),   # AVHRR-era coordinates
    ("pml", "V2.2c", "ET", "2015-07"),
]


# Regridding itself conserves exactly (test_conserves_area_weighted_total). The
# global means still differ because the NA_THRES coastal mask changes the land
# area: coarse cells < 50% valid are dropped and partly valid cells are counted
# at full area. That shifts the mean by up to ~1.2% (GLEAM Jan 1980 at 1 deg,
# when the dropped coastal band has high ET), hence the 2% tolerance.
GLOBAL_MEAN_RTOL = 0.02


@pytest.mark.skipif(not lo.OBS_ROOT.exists(), reason="needs glade obs data")
@pytest.mark.parametrize("res", [0.5, 1.0])
@pytest.mark.parametrize("dataset, version, var, month", REAL_MONTHS)
def test_global_area_weighted_mean_real_month(dataset, version, var, month, res):
    """
    On real GLEAM and PML months, regridding changes the global area-weighted mean by less than 2% (skipped
    without glade data).
    """
    da = lo.load_obs(dataset, var, slice(month, month), version=version, chunks=None).load()
    assert da.sizes["time"] == 1
    out = rg.regrid_with_na_thres(da, rg.conservative_regridder(da, res))

    src, tgt = rg.source_grid(da), rg.target_grid(res)
    native = area_mean(da.values[0], cell_area(src.lat_b.values, src.lon_b.values))
    regridded = area_mean(out.values[0], cell_area(tgt.lat_b.values, tgt.lon_b.values))
    rel_diff = (regridded - native) / native
    print(f"{dataset} {version} {var} {month} {res} deg: native {native:.5f}, "
          f"regridded {regridded:.5f} {da.attrs['units']}, rel. diff {rel_diff:+.3%}")
    assert regridded == pytest.approx(native, rel=GLOBAL_MEAN_RTOL)
