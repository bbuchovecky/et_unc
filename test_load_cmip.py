"""
Tests that pin the behavior of the CMIP6 member selection and model loading in
`etunc.load.cmip` (moved from `cmip_binned_et.py`, with `load_model` split into
`load_native` and `regrid_annual`), the member-ID helpers of `CMIPESGFLoader`,
and `regrid_to_target` (from `regrid_cmip_esgf.py`). The expected values were
written against the old driver functions and are unchanged. Run from the
project root with:

    python -m pytest test_load_cmip.py

Synthetic catalogs and a stub loader on a small native grid; takes a few seconds.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import etunc.config as config
import etunc.grid as rg
import etunc.load.cmip as cmip
from etunc.load.cmip import CMIPESGFLoader


# ------------------------------------------------------------------
# Member IDs and member selection
# ------------------------------------------------------------------

# Settings of cmip_binned_et, passed explicitly
VARIABLES = ["evspsbl", "lai", "pr", "rsds", "rsus", "rlds", "rlus"]
EXPERIMENT_ID = "historical"
TIME_SLICE = slice("1995-01", "2014-12")
TARGET_RES = 1.0
TARGET_GRID = rg.target_grid(TARGET_RES)
NA_THRES = 0.5
SETTINGS = dict(variables=VARIABLES, experiment_id=EXPERIMENT_ID, time_slice=TIME_SLICE, res=TARGET_RES, na_thres=NA_THRES)

def test_sort_member_ids_numeric():
    """Member IDs sort numerically by r, i, p, f (r10 after r2), and invalid IDs raise."""
    mids = ["r10i1p1f1", "r2i1p1f1", "r1i2p1f1", "r1i1p1f2", "r1i1p1f1"]
    assert CMIPESGFLoader.sort_member_ids(mids) == [
        "r1i1p1f1", "r1i1p1f2", "r1i2p1f1", "r2i1p1f1", "r10i1p1f1",
    ]
    with pytest.raises(ValueError, match="Invalid member_id"):
        CMIPESGFLoader.sort_member_ids(["r1i1p1f1", "member1"])


def test_group_member_ids_by_ipf():
    """Members are grouped by (i, p, f), each group in r order."""
    groups = CMIPESGFLoader.group_member_ids_by_ipf(["r3i1p2f1", "r1i1p1f1", "r2i1p2f1", "r10i1p1f1"])
    assert groups == {(1, 1, 1): ["r1i1p1f1", "r10i1p1f1"], (1, 2, 1): ["r2i1p2f1", "r3i1p2f1"]}


def catalog_rows(source_id, member_id, variables, experiment_id="historical"):
    return [
        {"experiment_id": experiment_id, "source_id": source_id, "member_id": member_id, "variable_id": v}
        for v in variables
    ]


def test_available_members():
    """
    A model's members are the historical members with every VARIABLE, sorted; models without any are left out.
    """
    allv = VARIABLES
    rows = (
        catalog_rows("A", "r2i1p1f1", allv)
        + catalog_rows("A", "r1i1p1f1", allv)
        + catalog_rows("A", "r3i1p1f1", [v for v in allv if v != "lai"])  # missing a variable
        + catalog_rows("A", "r4i1p1f1", allv, experiment_id="ssp585")     # other experiment
        + catalog_rows("B", "r1i1p1f1", allv[:-1])                         # no member has every variable
        + catalog_rows("C", "r10i1p1f1", allv)
        + catalog_rows("C", "r2i1p1f1", allv)
        + catalog_rows("D", "r1i1p1f1", allv, experiment_id="ssp585")
    )
    avail = cmip.available_members(pd.DataFrame(rows), VARIABLES, "historical")
    assert avail == {"A": ["r1i1p1f1", "r2i1p1f1"], "C": ["r2i1p1f1", "r10i1p1f1"]}


AVAIL = ["r1i1p1f1", "r1i1p2f1", "r2i1p1f1", "r2i1p2f1", "r3i1p2f1"]


@pytest.mark.parametrize("spec, expected", [
    ("top", ["r1i1p1f1"]),
    ("all", AVAIL),
    ("max_r", ["r1i1p2f1", "r2i1p2f1", "r3i1p2f1"]),
    (["r3i1p2f1", "r1i1p1f1"], ["r3i1p2f1", "r1i1p1f1"]),  # order of the list, not of AVAIL
    (["r1i1p1f1", "r9i1p1f1"], ["r1i1p1f1"]),              # unavailable members dropped
])
def test_select_members_default(spec, expected):
    """Each default mode (top, all, max_r, explicit list) selects the expected members."""
    assert cmip.select_members("A", AVAIL, {}, spec) == expected


def test_select_members_max_r_tie_takes_first_group():
    """With max_r, a tie between groups goes to the group of the first sorted member."""
    # Two groups of two: the group of the first sorted member wins
    assert cmip.select_members("A", ["r2i1p2f1", "r1i1p2f1", "r2i1p1f1", "r1i1p1f1"], {}, "max_r") == ["r1i1p1f1", "r2i1p1f1"]


def test_select_members_per_model_and_unknown():
    """member_ids overrides the default for a model, and an unknown mode raises."""
    member_ids = {"B": "all", "C": "first"}
    assert cmip.select_members("A", AVAIL, member_ids, "top") == ["r1i1p1f1"]
    assert cmip.select_members("B", AVAIL, member_ids, "top") == AVAIL
    with pytest.raises(ValueError, match="unknown member selection"):
        cmip.select_members("C", AVAIL, member_ids, "top")


def test_sftlf_files(tmp_path):
    """Each model's sftlf path is read from the fx catalog, and a model with two sftlf files raises."""
    rows = [
        {"source_id": "A", "variable_id": "sftlf", "path": "/a/sftlf.nc"},
        {"source_id": "A", "variable_id": "areacella", "path": "/a/areacella.nc"},
        {"source_id": "B", "variable_id": "sftlf", "path": "/b/sftlf.nc"},
    ]
    path = tmp_path / "fx.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    assert cmip.sftlf_files(path) == {"A": "/a/sftlf.nc", "B": "/b/sftlf.nc"}

    rows.append({"source_id": "B", "variable_id": "sftlf", "path": "/b/other/sftlf.nc"})
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match=r"more than one sftlf file for \['B'\]"):
        cmip.sftlf_files(path)


def test_member_tag():
    """One member is tagged by its ID, several by their count."""
    assert cmip.member_tag(["r1i1p1f1"]) == "r1i1p1f1"
    assert cmip.member_tag(["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]) == "3members"


# ------------------------------------------------------------------
# load_model: native annual means -> conservative regrid onto the 1 deg grid
# ------------------------------------------------------------------
# Native grid: 2.25 deg, lon in [0, 360], edges at multiples of 2.25 from -90
# and 0. Its edges fall a quarter of the way into 1 deg target cells, so a
# target cell on the edge of a land block is 25% or 75% land.

RES_N = 2.25
LAT_N = -90 + RES_N * (np.arange(80) + 0.5)
LON_N = RES_N * (np.arange(160) + 0.5)
TIME = xr.date_range("1995-01-01", "1997-06-01", freq="MS", calendar="noleap", use_cftime=True)  # 1997 incomplete
MEMBERS = ["r1i1p1f1", "r2i1p1f1"]

LAND = (slice(41, 45), slice(41, 45))       # native cells: lat 2.25-11.25, lon 92.25-101.25
GREENLAND = (slice(72, 76), slice(140, 144))  # native cells: lat 72-81, lon 315-324
LAI_GAP = (0, "1995-07", 42, 42)  # (member, month, lat index, lon index): lat 4.5-6.75, lon 94.5-96.75
ET_GAP = (0, "1996-03", 43, 43)   # lat 6.75-9, lon 96.75-99

# {variable: (land value, ocean value)}; ocean values must not leak into the result
VALUES = {
    "evspsbl": (2e-5, 5e-5),  # kg m-2 s-1
    "lai": (2.0, 0.0),
    "pr": (3e-5, 9e-5),
    "rsds": (200.0, 250.0),
    "rsus": (30.0, 20.0),
    "rlds": (350.0, 300.0),
    "rlus": (400.0, 410.0),
}
L = config.LATENT_HEAT_VAPORIZATION
EXPECTED = {"et": 2e-5 * L, "lai": 2.0, "pr": 3e-5 * L, "rn": 200.0 - 30.0 + 350.0 - 400.0}

# Target (1 deg) cells
VALID_LAT = np.arange(2.5, 11, 1.0)   # 75% land at 2.5, 25% land at 11.5 (NaN)
VALID_LON = np.arange(92.5, 101, 1.0)  # likewise for lon
MASKED_CELL = (6.5, 96.5)              # False in the target `mask`
ET_GAP_CELLS = [(7.5, 97.5), (7.5, 98.5), (8.5, 97.5), (8.5, 98.5)]  # inside the native ET_GAP cell
LAI_GAP_CELL = (5.5, 95.5)             # the only target cell inside the native LAI_GAP cell
LAI_GAP_TOUCHED = [(lat, lon) for lat in (4.5, 5.5, 6.5) for lon in (94.5, 95.5, 96.5)]


def land_fraction(scale=100.0):
    lf = np.zeros((LAT_N.size, LON_N.size))
    lf[LAND] = 1.0
    lf[GREENLAND] = 1.0
    return xr.DataArray(
        lf * scale, dims=("lat", "lon"), coords={"lat": LAT_N, "lon": LON_N}, name="sftlf",
        attrs={"units": "%"},
    )


def write_sftlf(path, scale=100.0, lat_shift=0.0):
    lf = land_fraction(scale)
    lf = lf.assign_coords(lat=lf.lat + lat_shift)
    lf.to_dataset().to_netcdf(path)
    return str(path)


def monthly_fields():
    """{variable: (member, time, lat, lon)} as returned by CMIPESGFLoader.load_data."""
    land = land_fraction(1.0).values > 0
    out = {}
    for v, (vland, vocean) in VALUES.items():
        data = np.where(land, vland, vocean)[None, None] * np.ones((len(MEMBERS), TIME.size, 1, 1))
        out[v] = xr.DataArray(
            data, dims=("member", "time", "lat", "lon"),
            coords={"member": np.arange(len(MEMBERS)), "member_id": ("member", MEMBERS),
                    "time": TIME, "lat": LAT_N, "lon": LON_N},
            name=v, attrs={"units": "test"},
        )
    for v, (m, month, i, j) in [("lai", LAI_GAP), ("evspsbl", ET_GAP)]:
        t = np.flatnonzero(out[v].time.dt.strftime("%Y-%m").values == month)[0]
        out[v][m, t, i, j] = np.nan
    return out


class StubLoader:
    """Returns fixed data in the structure of `CMIPESGFLoader.load_data` and records the call."""

    def __init__(self, data):
        self.data = data
        self.calls = []

    def load_data(self, variables, experiment_id, source_id=None, member_id=None, time_slice=None, **kwargs):
        self.calls.append({"variables": list(variables), "experiment_id": experiment_id,
                           "source_id": source_id, "member_id": list(member_id), "time_slice": time_slice})
        return {source_id: {v: self.data[v].sel(time=time_slice) for v in variables if v in self.data}}


def target_mask():
    grid = TARGET_GRID.sel(lat=config.LAT_BNDS)
    mask = xr.DataArray(
        np.ones((grid.sizes["lat"], grid.sizes["lon"])), dims=("lat", "lon"),
        coords={"lat": grid.lat, "lon": grid.lon},
    )
    mask.loc[{"lat": MASKED_CELL[0], "lon": MASKED_CELL[1]}] = 0.0
    return mask


def expected_field(k, member, year):
    """Expected (lat, lon) values of `k` away from the LAI gap, which is checked separately."""
    grid = TARGET_GRID.sel(lat=config.LAT_BNDS)
    out = xr.DataArray(np.full((grid.sizes["lat"], grid.sizes["lon"]), np.nan), dims=("lat", "lon"),
                       coords={"lat": grid.lat, "lon": grid.lon})
    out.loc[{"lat": VALID_LAT, "lon": VALID_LON}] = EXPECTED[k]
    out.loc[{"lat": MASKED_CELL[0], "lon": MASKED_CELL[1]}] = np.nan
    if k == "et" and member == 0 and year == 1996:
        for lat, lon in ET_GAP_CELLS:
            out.loc[{"lat": lat, "lon": lon}] = np.nan
    return out


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    """load_model output with a % sftlf file, plus the stub loader that it called."""
    path = write_sftlf(tmp_path_factory.mktemp("fx") / "sftlf.nc")
    loader = StubLoader(monthly_fields())
    ann = cmip.load_model(loader, "FAKE-ESM", MEMBERS, path, target_mask(), **SETTINGS)
    return ann, loader


def test_load_model_calls_loader(loaded):
    """load_model asks the loader for the variables, experiment, the model's members and the time slice."""
    _, loader = loaded
    assert loader.calls == [{
        "variables": VARIABLES, "experiment_id": EXPERIMENT_ID, "source_id": "FAKE-ESM",
        "member_id": MEMBERS, "time_slice": TIME_SLICE,
    }]


def test_load_model_dims_coords_names_units(loaded):
    """
    et, lai, pr and rn have dims (member, year, lat, lon), complete years only, member_id, the exact 1 deg
    coords within LAT_BNDS, and the right units.
    """
    ann, _ = loaded
    assert list(ann) == ["et", "lai", "pr", "rn"]
    grid = TARGET_GRID.sel(lat=config.LAT_BNDS)
    for k, da in ann.items():
        assert da.name == k
        assert da.dims == ("member", "year", "lat", "lon")
        assert da.year.values.tolist() == [1995, 1996]  # incomplete 1997 dropped
        assert da.member_id.values.tolist() == MEMBERS
        np.testing.assert_array_equal(da.lat, grid.lat)
        np.testing.assert_array_equal(da.lon, grid.lon)
    assert {k: da.attrs["units"] for k, da in ann.items()} == {
        "et": "W/m2", "lai": "m2/m2", "pr": "W/m2", "rn": "W/m2",
    }


@pytest.mark.parametrize("k", ["et", "lai", "pr", "rn"])
def test_load_model_values(loaded, k):
    """
    Land values only (ocean masked on the native grid before regridding),
    NaN where more than NA_THRES of a target cell is outside the native land
    mask, in Greenland, and where the target mask is False. A missing ET month
    makes that year NaN.
    """
    ann, _ = loaded
    for m in range(len(MEMBERS)):
        for year in (1995, 1996):
            got = ann[k].isel(member=m).sel(year=year)
            expected = expected_field(k, m, year)
            if k == "lai" and m == 0 and year == 1995:
                for lat, lon in LAI_GAP_TOUCHED:
                    expected.loc[{"lat": lat, "lon": lon}] = got.sel(lat=lat, lon=lon)  # checked below
            np.testing.assert_allclose(got.values, expected.values, rtol=1e-10, equal_nan=True)


def test_load_model_missing_lai_month_counts_as_zero(loaded):
    """A missing LAI month counts as 0 in the annual mean instead of making the year NaN."""
    ann, _ = loaded
    lai = ann["lai"].isel(member=0).sel(year=1995)
    july_weight = 31 / 365
    assert float(lai.sel(lat=LAI_GAP_CELL[0], lon=LAI_GAP_CELL[1])) == pytest.approx(2.0 * (1 - july_weight), rel=1e-10)
    for lat, lon in LAI_GAP_TOUCHED:
        if (lat, lon) in (LAI_GAP_CELL, MASKED_CELL):
            continue
        val = float(lai.sel(lat=lat, lon=lon))  # partly inside the gap cell
        assert 2.0 * (1 - july_weight) < val < 2.0
    np.testing.assert_allclose(ann["lai"].isel(member=1).sel(year=1995, lat=LAI_GAP_CELL[0], lon=LAI_GAP_CELL[1]), 2.0)


def test_load_model_greenland_excluded(loaded):
    """Land in Greenland is excluded by the native mask, so it is NaN on the target grid."""
    ann, _ = loaded
    native = rg.mask_greenland(land_fraction(1.0))
    assert not native.isel(lat=GREENLAND[0], lon=GREENLAND[1]).any()  # land in sftlf, excluded by the mask
    greenland = ann["et"].sel(lat=slice(72, 81), lon=slice(-45, -36))
    assert greenland.size > 0 and greenland.isnull().all()


def test_load_model_fraction_sftlf_same_as_percent(loaded, tmp_path):
    """A sftlf file stored as a fraction gives the same output as one stored in %."""
    ann, _ = loaded
    path = write_sftlf(tmp_path / "sftlf_frac.nc", scale=1.0)  # fraction labelled "%" (as E3SM-1-0)
    ann_frac = cmip.load_model(StubLoader(monthly_fields()), "FAKE-ESM", MEMBERS, path, target_mask(), **SETTINGS)
    for k in ann:
        xr.testing.assert_identical(ann_frac[k], ann[k])


def test_load_model_sftlf_on_other_grid_raises(tmp_path):
    """A sftlf file on a different grid from the data raises."""
    path = write_sftlf(tmp_path / "sftlf_shifted.nc", lat_shift=0.5)
    with pytest.raises(ValueError, match="differ from the reference grid"):
        cmip.load_model(StubLoader(monthly_fields()), "FAKE-ESM", MEMBERS, path, target_mask(), **SETTINGS)


def test_load_model_missing_variable_raises(tmp_path):
    """A variable the loader cannot provide raises, naming it."""
    path = write_sftlf(tmp_path / "sftlf.nc")
    data = monthly_fields()
    del data["rlus"]
    with pytest.raises(ValueError, match=r"could not load \['rlus'\]"):
        cmip.load_model(StubLoader(data), "FAKE-ESM", MEMBERS, path, target_mask(), **SETTINGS)


def test_load_model_is_regrid_annual_of_load_native(loaded, tmp_path):
    """load_model is regrid_annual(*load_native(...)): native annual means, then the conservative regrid."""
    ann, _ = loaded
    path = write_sftlf(tmp_path / "sftlf.nc")
    native, native_mask, src_grid = cmip.load_native(
        StubLoader(monthly_fields()), "FAKE-ESM", MEMBERS, path, VARIABLES, EXPERIMENT_ID, TIME_SLICE,
    )
    assert native["et"].dims == ("member", "year", "lat", "lon") and native["et"].sizes["lat"] == LAT_N.size
    assert native_mask.dtype == bool and src_grid.sizes["lat_b"] == LAT_N.size + 1
    out = cmip.regrid_annual(native, src_grid, TARGET_RES, target_mask(), NA_THRES, members=MEMBERS)
    for k in ann:
        xr.testing.assert_identical(out[k], ann[k])


def test_regrid_annual_na_thres_and_any_spacing(tmp_path):
    """
    A higher na_thres keeps target cells that are only 25% land (with the land value); any spacing that divides
    180 gives that grid's exact coords within LAT_BNDS.
    """
    path = write_sftlf(tmp_path / "sftlf.nc")
    native, _, src_grid = cmip.load_native(
        StubLoader(monthly_fields()), "FAKE-ESM", MEMBERS, path, VARIABLES, EXPERIMENT_ID, TIME_SLICE,
    )
    loose = cmip.regrid_annual(native, src_grid, TARGET_RES, target_mask(), na_thres=0.8)
    edge = loose["pr"].isel(member=1).sel(year=1995, lat=11.5, lon=VALID_LON[1:])  # 25% land (lon 92.5: 19%, NaN)
    np.testing.assert_allclose(edge, EXPECTED["pr"], rtol=1e-10)
    assert "member_id" not in loose["pr"].coords

    g2 = rg.target_grid(2.0).sel(lat=config.LAT_BNDS)
    mask2 = xr.ones_like(g2.lat * g2.lon)
    coarse = cmip.regrid_annual(native, src_grid, 2.0, mask2, NA_THRES)
    np.testing.assert_array_equal(coarse["et"].lat, g2.lat)
    np.testing.assert_array_equal(coarse["et"].lon, g2.lon)


def test_regrid_to_target_ec_earth_ij_grid():
    """
    A field on EC-Earth-style (j, i) dims with 2-D latitude/longitude coords is given 1-D lat/lon and
    conservatively regridded onto the target grid; a constant field stays constant.
    """
    lat = np.arange(-89.0, 90.0, 2.0)
    lon = np.arange(1.0, 360.0, 2.0)
    da = xr.DataArray(
        np.full((2, lat.size, lon.size), 3.0), dims=("time", "j", "i"),
        coords={"latitude": (("j", "i"), np.repeat(lat[:, None], lon.size, axis=1)),
                "longitude": (("j", "i"), np.repeat(lon[None, :], lat.size, axis=0))},
        name="lai", attrs={"units": "1"},
    )
    out = cmip.regrid_to_target(da, 1.0)
    assert out.dims == ("time", "lat", "lon") and out.attrs["units"] == "1"
    np.testing.assert_array_equal(out.lat, rg.target_grid(1.0).lat)
    np.testing.assert_allclose(out.sel(lat=slice(-87, 87)), 3.0, rtol=1e-10)
