"""
Tests for load_obs.py on small synthetic per-year NetCDF files. Run from the
project root with:

    python -m pytest tests/test_load_obs.py
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import rasterio
import rasterio.transform
import xarray as xr

import etunc.load.obs as lo
from etunc.units import accumulation_to_flux


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
    """Variables and years are found from the files, per frequency and version."""
    assert lo.list_variables(spec) == ["ET", "GPP"]
    assert lo.list_variables(spec, freq="yearly") == ["ET"]
    assert lo.list_years(spec, "ET") == [2000, 2001, 2002, 2003]
    assert lo.list_variables(spec, version="v2") == []


def test_find_files_by_year(spec):
    """Only the files for the years in the time slice are returned."""
    files = lo.find_files(spec, "ET", slice("2001-06", "2002-02"))
    assert [f.name for f in files] == ["FAKE-v1_ET_2001.nc", "FAKE-v1_ET_2002.nc"]


def test_find_files_warns_missing_years(spec):
    """Requested years without a file trigger a warning that lists them."""
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
    """Unknown datasets, versions, frequencies and variables raise clear errors."""
    with pytest.raises(KeyError, match="Unknown dataset"):
        lo.get_dataset("nope")
    with pytest.raises(ValueError, match="unknown version"):
        lo.load_obs(spec, "ET", version="v9")
    with pytest.raises(ValueError, match="unknown freq"):
        lo.load_obs(spec, "ET", freq="8-day")
    with pytest.raises(FileNotFoundError, match="Available variables"):
        lo.load_obs(spec, "LAI")


def test_registered_datasets():
    """
    GLEAM, PML, SiTHv2, CLARA-A3, NCEI AVHRR/VIIRS LAI, FLUXCOM-X, GLDAS and GPCP are registered (GPCP v3.3 by
    default); names are case-insensitive.
    """
    assert {"gleam", "pml", "sith", "clara", "avhrr", "viirs", "fluxcom-x", "gldas", "gpcp"} <= set(lo.DATASETS)
    assert lo.get_dataset("gpcp").versions[0] == "3.3" and lo.get_dataset("gldas").versions[0] == "2.1"
    assert [d for _, d, _ in lo.NCEI_LAI_SENSORS] == ["avhrr", "viirs"]
    assert lo.get_dataset("PML") is lo.DATASETS["pml"]


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def test_load_obs_single_grid_across_offset_files(spec):
    """
    Files with coordinates offset by ~1e-5 deg load onto one grid (lat ascending, lon in [-180, 180]) without
    NaN, with the standard attrs.
    """
    da = lo.load_obs(spec, "ET", slice("2001-07", "2002-06"))
    assert da.dims == ("time", "lat", "lon")
    assert da.sizes == {"time": 12, "lat": 3, "lon": 3}
    np.testing.assert_array_equal(da.lat, [44.85, 44.95, 45.05])
    np.testing.assert_array_equal(da.lon, [-179.95, -0.05, 0.05])
    assert np.isfinite(da.values).all()
    np.testing.assert_array_equal(da.isel(lat=0, lon=0), [2001] * 6 + [2002] * 6)
    assert da.attrs == {"units": "mm/month", "dataset": "fake", "version": "v1", "frequency": "monthly", "res": "native"}


def test_load_obs_lat_bnds_and_chunks(spec):
    """lat_bnds selects a latitude range, and chunks sets the dask chunks."""
    da = lo.load_obs(spec, "ET", lat_bnds=slice(44.9, 50), chunks={"lat": 1, "time": 12})
    np.testing.assert_array_equal(da.lat, [44.95, 45.05])
    assert da.chunks[1] == (1, 1)


def test_load_obs_mismatched_grids_raise(spec, tmp_path):
    """Files whose grids differ by more than round-off raise."""
    write_year(tmp_path / "v2/monthly/ET/FAKE-v2_ET_2000.nc", "ET", 2000)
    write_year(tmp_path / "v2/monthly/ET/FAKE-v2_ET_2001.nc", "ET", 2001, lat=LAT + 0.1)
    with pytest.raises(ValueError):
        lo.load_obs(spec, "ET", version="v2")


def test_load_obs_preprocess_hook(spec):
    """The dataset's preprocess function is applied to each file."""
    def relabel(ds, var, version, freq):
        ds[var].attrs["units"] = f"{version}-{freq}"
        return ds

    s = lo.ObsDataset(**{**spec.__dict__, "preprocess": relabel})
    assert lo.load_obs(s, "ET", freq="yearly").attrs["units"] == "v1-yearly"


def test_load_obs_template_without_year(tmp_path):
    """A single file that holds every year is loaded and subset to the time slice."""
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


def test_load_obs_monthly_files(tmp_path):
    """CLARA-A3 layout: one file per month, {year} twice, {standard_name} directory matched from the path."""
    s = replace(lo.get_dataset("clara"), root=tmp_path)
    lat, lon = np.array([-0.125, 0.125]), np.array([-0.125, 0.125])
    for var, name in (("SRB", "surface_net_downward_radiative_flux"), ("SNS", "surface_net_downward_shortwave_flux")):
        for year in (2000, 2001):
            for month in range(1, 13):
                time = pd.DatetimeIndex([f"{year}-{month:02d}-01"])
                ds = xr.Dataset({var: (("time", "lat", "lon"), np.full((1, 2, 2), month, "float32"), {"units": "W m-2"})},
                                coords={"time": time, "lat": lat, "lon": lon})
                path = tmp_path / f"{name}/{year}/{var}mm{year}{month:02d}01000000319AVPOS01GL.nc"
                path.parent.mkdir(parents=True, exist_ok=True)
                ds.to_netcdf(path)

    assert lo.list_variables(s) == ["SNS", "SRB"]
    assert lo.list_years(s, "SRB") == [2000, 2001]
    files = lo.find_files(s, "SRB", slice("2001", "2001"))
    assert len(files) == 12 and all(f.parts[-3] == "surface_net_downward_radiative_flux" for f in files)

    da = lo.load_obs(s, "SRB", slice("2000-11", "2001-02"))
    assert da.sizes["time"] == 4 and da.attrs["units"] == "W m-2"
    np.testing.assert_array_equal(da.isel(lat=0, lon=0), [11, 12, 1, 2])  # months in time order


def write_ncei_lai(root, source_id, years, first_month=1, value=None):
    """Per-year NOAA NCEI LAI files (obs4MIPs tree) under `root`, with LAI = `value` or year - 1980."""
    # float32 noise in the coords, as in the real files, and lon in [0, 360)
    lat = np.array([-0.025, 0.025], dtype="float32").astype("float64")
    lon = np.array([0.025, 359.975], dtype="float32").astype("float64")
    folder = root / f"data/obs4MIPs/NOAA-NCEI/{source_id}/mon/lai/gn/v20260110"
    folder.mkdir(parents=True, exist_ok=True)
    for year in years:
        m0 = first_month if year == years[0] else 1
        time = pd.date_range(f"{year}-{m0:02d}", periods=13 - m0, freq="MS") + pd.Timedelta(days=14)
        data = np.full((len(time), 2, 2), year - 1980 if value is None else value, dtype="float32")
        ds = xr.Dataset({"lai": (("time", "lat", "lon"), data, {"units": "1"})},
                        coords={"time": time, "lat": lat, "lon": lon})
        ds.to_netcdf(folder / f"lai_mon_{source_id}_BE_gn_{year}{m0:02d}-{year}12.nc")


def test_load_obs_avhrr_lai(tmp_path):
    """AVHRR LAI layout: obs4MIPs tree, per-year files named by start and end month, a partial first year."""
    s = replace(lo.get_dataset("avhrr"), root=tmp_path)
    write_ncei_lai(tmp_path, "NOAA-NCEI-LAI-AVHRR-5-0", [1981, 1982], first_month=6)

    assert lo.list_variables(s) == ["lai"]
    assert lo.list_years(s, "lai") == [1981, 1982]
    assert [f.name for f in lo.find_files(s, "lai", slice("1982", "1982"))] == [
        "lai_mon_NOAA-NCEI-LAI-AVHRR-5-0_BE_gn_198201-198212.nc"]

    da = lo.load_obs(s, "lai")
    assert da.sizes["time"] == 7 + 12 and da.attrs["dataset"] == "avhrr"
    np.testing.assert_array_equal(da.lon, [-0.025, 0.025])  # wrapped, sorted and rounded
    np.testing.assert_array_equal(da.lat, [-0.025, 0.025])
    np.testing.assert_array_equal(da.isel(lat=0, lon=0), [1] * 7 + [2] * 12)


@pytest.fixture
def ncei_lai(tmp_path, monkeypatch):
    """AVHRR (2012-2013) and VIIRS (2014-2015) records, each under its own root as on disk."""
    for name, source_id, years, value in (("avhrr", "NOAA-NCEI-LAI-AVHRR-5-0", [2012, 2013], 1.0),
                                          ("viirs", "NOAA-NCEI-LAI-VIIRS-1-0", [2014, 2015], 2.0)):
        root = tmp_path / name
        write_ncei_lai(root, source_id, years, value=value)
        monkeypatch.setitem(lo.DATASETS, name, replace(lo.get_dataset(name), root=root))
    return tmp_path


def test_load_ncei_lai_joins_sensors(ncei_lai):
    """AVHRR then VIIRS on one time axis, each month labelled with its sensor, the switch in the attrs."""
    da = lo.load_ncei_lai(slice("2013-07", "2014-06"))
    assert da.sizes["time"] == 12 and da.dims == ("time", "lat", "lon")
    assert list(da.sensor.values) == ["AVHRR LAI v5.0"] * 6 + ["VIIRS LAI v1.0"] * 6
    np.testing.assert_array_equal(da.isel(lat=0, lon=0), [1.0] * 6 + [2.0] * 6)  # values unchanged
    assert da.attrs["dataset"] == "ncei_lai"
    assert da.attrs["version"] == "avhrr v20260110, viirs v20260110"
    assert da.attrs["sensors"] == "AVHRR LAI v5.0: 2013-07 to 2013-12; VIIRS LAI v1.0: 2014-01 to 2014-06"
    assert da.attrs["sensor_change"] == "2014-01 (AVHRR LAI v5.0 to VIIRS LAI v1.0)"


def test_load_ncei_lai_one_sensor(ncei_lai):
    """A period within one sensor's record reads only that sensor, and has no sensor change."""
    da = lo.load_ncei_lai(slice("2015", "2015"))
    assert da.sizes["time"] == 12 and set(da.sensor.values) == {"VIIRS LAI v1.0"}
    assert da.attrs["sensor_change"] == "none" and da.attrs["version"] == "viirs v20260110"


def test_load_ncei_lai_overlap_and_missing_raise(ncei_lai, monkeypatch):
    """Overlapping sensor records raise, and so does a sensor without any file."""
    write_ncei_lai(ncei_lai / "viirs", "NOAA-NCEI-LAI-VIIRS-1-0", [2013], value=2.0)
    with pytest.raises(ValueError, match="starts before"):
        lo.load_ncei_lai()
    monkeypatch.setitem(lo.DATASETS, "viirs", replace(lo.get_dataset("viirs"), root=ncei_lai / "empty"))
    with pytest.raises(FileNotFoundError, match="viirs"):
        lo.load_ncei_lai(slice("2012", "2013"))


def test_load_obs_fluxcom_x(tmp_path):
    """
    FLUXCOM-X layout: per-year files of one variable, lat descending, a static land_fraction that is not
    read, ET in mm hr-1 left as stored.
    """
    s = replace(lo.get_dataset("fluxcom-x"), root=tmp_path)
    lat, lon = np.array([0.25, -0.25]), np.array([-0.25, 0.25])
    for year in (2001, 2002):
        time = pd.date_range(f"{year}-01", periods=12, freq="MS")
        ds = xr.Dataset({"ET": (("time", "lat", "lon"), np.full((12, 2, 2), 0.1, "float32"), {"units": "mm hr-1"}),
                         "land_fraction": (("lat", "lon"), np.ones((2, 2)))},
                        coords={"time": time, "lat": lat, "lon": lon})
        path = tmp_path / f"050_monthly/ET/ET_{year}_050_monthly.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path)

    assert lo.list_variables(s) == ["ET"] and lo.list_years(s, "ET") == [2001, 2002]
    da = lo.load_obs(s, "ET", slice("2002", "2002"))
    assert da.name == "ET" and da.dims == ("time", "lat", "lon") and da.sizes["time"] == 12
    np.testing.assert_array_equal(da.lat, [-0.25, 0.25])
    assert da.attrs["units"] == "mm hr-1" and da.attrs["dataset"] == "fluxcom-x"


def write_gldas(root, version, year, n_steps_full=8, band="Evap_tavg", units="kg m-2 s-1", value=1e-5):
    """One year of a download_gldas.py file: `band`, and n_steps = n_steps_full except one cell-month."""
    time = pd.date_range(f"{year}-01", periods=12, freq="MS")
    n_steps = np.full((12, 2, 2), n_steps_full, dtype="int16")
    n_steps[0, 0, 0] = n_steps_full - 1  # an incomplete month
    n_steps[:, 1, 1] = 0                  # ocean: no time step, NaN
    values = np.where(n_steps > 0, value, np.nan).astype("float32")
    ds = xr.Dataset({band: (("time", "lat", "lon"), values, {"units": units}),
                     "n_steps": (("time", "lat", "lon"), n_steps),
                     "n_steps_full": ("time", np.full(12, n_steps_full, dtype="int16"))},
                    coords={"time": time, "lat": [-0.125, 0.125], "lon": [-0.125, 0.125]})
    path = root / f"{version}/GLDAS_{version}_{band}_monthly_{year}.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def test_load_obs_gldas(tmp_path):
    """
    GLDAS layout: one folder per version and one file per band; a month with fewer valid time steps than
    the full month is NaN, and the n_steps variables are not returned. The P and Rn bands of prep_lsmobs.py
    (OWN_FIELDS) load the same way, each from its own file with its own units.
    """
    s = replace(lo.get_dataset("gldas"), root=tmp_path)
    write_gldas(tmp_path, "2.1", 2001)
    write_gldas(tmp_path, "2.2-grace-da", 2003, n_steps_full=31)

    assert lo.list_variables(s) == ["Evap_tavg"] and lo.list_years(s, "Evap_tavg", version="2.2-grace-da") == [2003]
    da = lo.load_obs(s, "Evap_tavg").load()  # default version 2.1
    assert da.name == "Evap_tavg" and da.attrs["version"] == "2.1" and da.attrs["units"] == "kg m-2 s-1"
    assert np.isnan(da.isel(time=0, lat=0, lon=0)) and np.isfinite(da.isel(time=1, lat=0, lon=0))
    assert int(da.isnull().sum()) == 1 + 12  # the incomplete month and the ocean cell
    with pytest.raises(FileNotFoundError):
        lo.load_obs(s, "Evap_tavg", version="2.0")

    # Precipitation (kg m-2 s-1) and net shortwave and longwave radiation (W m-2, positive downward)
    for band, units, value in (("Rainf_f_tavg", "kg m-2 s-1", 3e-5), ("Swnet_tavg", "W m-2", 150.0),
                               ("Lwnet_tavg", "W m-2", -60.0)):
        write_gldas(tmp_path, "2.1", 2001, band=band, units=units, value=value)
    assert lo.list_variables(s) == ["Evap_tavg", "Lwnet_tavg", "Rainf_f_tavg", "Swnet_tavg"]
    pr = lo.load_obs(s, "Rainf_f_tavg").load()
    assert pr.name == "Rainf_f_tavg" and pr.attrs["units"] == "kg m-2 s-1" and int(pr.isnull().sum()) == 1 + 12
    assert float(pr.isel(time=1, lat=0, lon=0)) == pytest.approx(3e-5)
    sw, lw = (lo.load_obs(s, b).load() for b in ("Swnet_tavg", "Lwnet_tavg"))
    assert sw.attrs["units"] == lw.attrs["units"] == "W m-2"
    assert float((sw + lw).isel(time=1, lat=0, lon=0)) == pytest.approx(90.0)


def test_load_obs_gpcp(tmp_path):
    """
    GPCP layout: one folder per version, one file per month holding every variable (no {var} in the name),
    float32 coords with lat descending.
    """
    s = replace(lo.get_dataset("gpcp"), root=tmp_path)
    lat = np.array([0.25, -0.25], dtype="float32")
    lon = np.array([-0.25, 0.25], dtype="float32")
    for version, years in (("3.3", (2000, 2001)), ("3.2", (2000,))):
        (tmp_path / f"gpcp-v{version}").mkdir()
        for year in years:
            for month in range(1, 13):
                time = pd.DatetimeIndex([f"{year}-{month:02d}-01"])
                full = np.full((1, 2, 2), month, "float32")
                ds = xr.Dataset({"sat_gauge_precip": (("time", "lat", "lon"), full, {"units": "mm/day"}),
                                 "gauge_precip": (("time", "lat", "lon"), full, {"units": "mm/day"}),
                                 "time_bnds": (("time", "bnds"), np.array([[time[0], time[0]]]))},
                                coords={"time": time, "lat": lat, "lon": lon})
                ds.to_netcdf(tmp_path / f"gpcp-v{version}/GPCPMON_L3_{year}{month:02d}_V{version}.nc4")

    assert lo.list_variables(s) == ["gauge_precip", "sat_gauge_precip"]  # from the file; no time bounds
    assert lo.list_years(s, "sat_gauge_precip") == [2000, 2001]
    assert lo.list_years(s, "sat_gauge_precip", version="3.2") == [2000]
    da = lo.load_obs(s, "sat_gauge_precip", slice("2000-11", "2001-02"))
    assert da.sizes["time"] == 4 and da.attrs["version"] == "3.3" and da.attrs["units"] == "mm/day"
    assert da.lat.dtype == np.float64
    np.testing.assert_array_equal(da.lat, [-0.25, 0.25])
    np.testing.assert_array_equal(da.isel(lat=0, lon=0), [11, 12, 1, 2])
    with pytest.raises(FileNotFoundError, match="Available variables"):
        lo.load_obs(s, "sat_gauge_precip", slice("2005", "2005"))


def test_gleam_units_fix():
    """The mislabeled units of GLEAM v4.3b yearly files are corrected to mm.year-1."""
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
    return replace(lo.get_dataset("gleam"), root=tmp_path)


@pytest.mark.parametrize("chunks", ["auto", None])
def test_gleam_v43b_ocean_masked_like_v43a(gleam, chunks):
    """GLEAM v4.3b ocean cells (stored as 0) are NaN wherever v4.3a is NaN, with or without dask."""
    a = lo.load_obs(gleam, "E", slice("2003", "2004"), version="v4.3a", chunks=chunks)
    b = lo.load_obs(gleam, "E", slice("2003", "2004"), version="v4.3b", chunks=chunks)
    xr.testing.assert_equal(a.isnull(), b.isnull())
    assert int(b.isnull().sum()) == 24  # one ocean cell, 24 months
    assert float(b.min()) == 2003.0
    assert b.attrs["units"] == "mm.month-1"


def test_gleam_v43b_needs_v43a_file(gleam, tmp_path):
    """Loading v4.3b raises if the v4.3a file used as its ocean mask is missing."""
    (tmp_path / "v4.3a/monthly/E/E_2004_GLEAM_v4.3a_MO.nc").unlink()
    with pytest.raises(FileNotFoundError, match="ocean mask"):
        lo.load_obs(gleam, "E", slice("2004", "2004"), version="v4.3b").load()


# ------------------------------------------------------------------
# GeoTIFF LAI products (GIMMS LAI4g, HiQ-LAI)
# ------------------------------------------------------------------

def write_geotiff(path, bands, west=-180.0, north=90.0, res=60.0):
    """North-up EPSG:4326 GeoTIFF of `bands` (list of 2-D arrays, row 0 the northernmost)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = bands[0].shape
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=len(bands),
                       dtype=bands[0].dtype, crs="EPSG:4326",
                       transform=rasterio.transform.from_origin(west, north, res, res)) as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(band, i)


# 3 x 6 grid of 60 deg cells; row 0 (north) holds the fill, row 2 (south) the valid values
GIMMS_RAW = np.array([[65535] * 6, [0, 7000, 1, 2, 3, 4], [1000, 1500, 2000, 2500, 3000, 3500]], dtype="uint16")


@pytest.fixture
def gimms(tmp_path, monkeypatch):
    """GIMMS LAI4g layout: both records, two zips of each, half-monthly files."""
    monkeypatch.setattr(lo, "GIMMS_LAI4G_ROOT", tmp_path)
    for record, (folder, prefix) in lo.GIMMS_LAI4G_RECORDS.items():
        for years in ("2001_2002", "2003_2003"):
            for year in range(int(years[:4]), int(years[-4:]) + 1):
                for month, half in ((1, "01"), (1, "02"), (2, "01")):
                    zdir = f"{folder}_{years}"
                    write_geotiff(tmp_path / zdir / zdir / f"{prefix}{year}{month:02d}{half}.tif",
                                  [GIMMS_RAW, np.zeros_like(GIMMS_RAW)])
    return tmp_path


def test_gimms_lai4g_files(gimms):
    """Files of each record are found across zips, dated at the start of their half month."""
    files = lo.gimms_lai4g_files()
    assert list(files)[:3] == [pd.Timestamp("2001-01-01"), pd.Timestamp("2001-01-16"), pd.Timestamp("2001-02-01")]
    assert len(files) == 9
    assert not any("(AVHRR)" in p.name for p in files.values())
    assert all("(AVHRR)" in p.name for p in lo.gimms_lai4g_files("AVHRR").values())
    with pytest.raises(ValueError, match="Unknown GIMMS"):
        lo.gimms_lai4g_files("MODIS")


def test_load_gimms_lai4g(gimms):
    """LAI is band 1 times 0.001, fill NaN, lat ascending, lazy, only the files in the time slice."""
    da = lo.load_gimms_lai4g(slice("2002", "2003-01"))
    assert da.chunks is not None
    assert da.dims == ("time", "lat", "lon")
    assert list(da.time.dt.strftime("%Y-%m-%d").values) == [
        "2002-01-01", "2002-01-16", "2002-02-01", "2003-01-01", "2003-01-16"]
    np.testing.assert_allclose(da.lat, [-60, 0, 60])
    np.testing.assert_allclose(da.lon, [-150, -90, -30, 30, 90, 150])
    first = da.isel(time=0).values
    np.testing.assert_allclose(first[0], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5])   # south row first
    np.testing.assert_allclose(first[1], [0.0, 7.0, 0.001, 0.002, 0.003, 0.004], rtol=1e-6)
    assert np.isnan(first[2]).all()                                       # 65535 fill
    assert da.dtype == "float32"
    assert da.attrs["units"] == "m2 m-2" and da.attrs["frequency"] == "half-month"
    assert lo.load_gimms_lai4g(lat_bnds=slice(-90, 0)).sizes["lat"] == 2


def test_load_gimms_lai4g_other_grid_raises(gimms):
    """A file on another grid raises when read, instead of being put on the first file's grid."""
    folder, prefix = lo.GIMMS_LAI4G_RECORDS["AVHRR-MODIS"]
    zdir = f"{folder}_2003_2003"
    write_geotiff(gimms / zdir / zdir / f"{prefix}20030201.tif", [GIMMS_RAW], west=-179.0)
    da = lo.load_gimms_lai4g(slice("2003", "2003"))
    with pytest.raises(ValueError, match="not on the grid"):
        da.load()
    with pytest.raises(FileNotFoundError):
        lo.load_gimms_lai4g(slice("1990", "1995"))


@pytest.fixture
def hiq(tmp_path, monkeypatch):
    """HiQ-LAI layout: 8-day files named by day of year, with a __MACOSX folder as in the zips."""
    monkeypatch.setattr(lo, "HIQ_LAI_ROOT", tmp_path)
    raw = np.array([[255, 249, 101, 0, 0, 0], [0, 100, 55, 1, 2, 3], [10, 20, 30, 40, 50, 60]], dtype="uint8")
    for years, doys in (("2000-2001", ("2000049", "2000361", "2001001")), ("2002-2002", ("2002009",))):
        zdir = tmp_path / f"{lo.HIQ_LAI_PREFIX}{years}"
        for doy in doys:
            write_geotiff(zdir / zdir.name / f"{lo.HIQ_LAI_PREFIX}{doy}.tif", [raw, raw])
            (zdir / "__MACOSX" / zdir.name).mkdir(parents=True, exist_ok=True)
            (zdir / "__MACOSX" / zdir.name / f"._{lo.HIQ_LAI_PREFIX}{doy}.tif").write_bytes(b"apple")
    return tmp_path


def test_load_hiq_lai(hiq):
    """Dated at the composite's first day; band 1 times 0.1, NaN above 100; macOS metadata files ignored."""
    assert list(lo.hiq_lai_files()) == [pd.Timestamp(t) for t in
                                        ("2000-02-18", "2000-12-26", "2001-01-01", "2002-01-09")]
    da = lo.load_hiq_lai(slice("2000-12", "2001"))
    assert list(da.time.dt.strftime("%Y-%m-%d").values) == ["2000-12-26", "2001-01-01"]
    v = da.isel(time=0).values
    np.testing.assert_allclose(v[0], [1, 2, 3, 4, 5, 6], rtol=1e-6)
    np.testing.assert_allclose(v[1], [0, 10, 5.5, 0.1, 0.2, 0.3], rtol=1e-6)
    assert np.isnan(v[2, :3]).all() and (v[2, 3:] == 0).all()  # 255, 249 (fill) and 101 (out of range)
    assert da.attrs["frequency"] == "8-day" and da.attrs["dataset"] == "hiq_lai"


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
    ("mm hr-1", ["2000-01-01"], [1 / 24]),  # FLUXCOM-X: mean hourly rate
])
def test_accumulation_to_flux(units, time, days):
    """Water depths per month, year, day or hour are divided by the length of the period in seconds."""
    da = xr.DataArray(np.ones(len(time)), dims="time", coords={"time": pd.DatetimeIndex(time)},
                      attrs={"units": units, "long_name": "x"})
    out = accumulation_to_flux(da)
    np.testing.assert_allclose(out, 1 / (np.array(days) * 86400))
    assert out.attrs == {"units": "kg m-2 s-1", "long_name": "x"}


def test_accumulation_to_flux_bad_units():
    """Units other than mm per hour, day, month or year raise."""
    da = xr.DataArray([1.0], dims="time", coords={"time": pd.DatetimeIndex(["2000-01-01"])},
                      attrs={"units": "gC/m2/month"})
    with pytest.raises(ValueError, match="Cannot convert"):
        accumulation_to_flux(da)
