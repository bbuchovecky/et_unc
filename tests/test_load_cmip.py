"""
Tests that pin the behavior of the CMIP6 member selection and model loading in
`etunc.load.cmip` (`load_native`, raw monthly fields for scripts/prep_cmip.py,
and `load_land_fraction`), the member-ID helpers of `CMIPESGFLoader`, and
`regrid_to_target` (from `regrid_cmip_esgf.py`). Run from the project root with:

    python -m pytest tests/test_load_cmip.py

Synthetic catalogs and a stub loader on a small native grid; takes a few seconds.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import etunc.grid as rg
import etunc.load.cmip as cmip
from etunc.load.cmip import CMIPESGFLoader


# ------------------------------------------------------------------
# Member IDs and member selection
# ------------------------------------------------------------------

# Settings of scripts/prep_cmip.py, passed explicitly
VARIABLES = ["evspsbl", "lai", "pr", "rsds", "rsus", "rlds", "rlus"]
EXPERIMENT_ID = "historical"
TIME_SLICE = slice("1995-01", "2014-12")

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
# load_native and load_land_fraction: raw monthly fields for binning
# ------------------------------------------------------------------
# Native grid: 2.25 deg, lon in [0, 360]

RES_N = 2.25
LAT_N = -90 + RES_N * (np.arange(80) + 0.5)
LON_N = RES_N * (np.arange(160) + 0.5)
TIME = xr.date_range("1995-01-01", "1997-06-01", freq="MS", calendar="noleap", use_cftime=True)
MEMBERS = ["r1i1p1f1", "r2i1p1f1"]
LAND = (slice(41, 45), slice(41, 45))  # native land cells


def land_fraction(scale=100.0):
    lf = np.zeros((LAT_N.size, LON_N.size))
    lf[LAND] = 1.0
    return xr.DataArray(
        lf * scale, dims=("lat", "lon"), coords={"lat": LAT_N, "lon": LON_N}, name="sftlf",
        attrs={"units": "%"},
    )


def write_sftlf(path, scale=100.0):
    land_fraction(scale).to_dataset().to_netcdf(path)
    return str(path)


def monthly_fields():
    """{variable: (member, time, lat, lon)} as returned by CMIPESGFLoader.load_data, with an extra coord."""
    out = {}
    for i, v in enumerate(VARIABLES):
        out[v] = xr.DataArray(
            np.full((len(MEMBERS), TIME.size, LAT_N.size, LON_N.size), float(i)),
            dims=("member", "time", "lat", "lon"),
            coords={"member": np.arange(len(MEMBERS)), "time": TIME, "lat": LAT_N, "lon": LON_N,
                    "height": 2.0},
            name=v, attrs={"units": "test"},
        )
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


def test_load_native_raw_fields_with_member_id():
    """
    load_native asks the loader for the model's variables, members and time slice, and returns them unchanged
    (native units) with only member_id kept of the non-index coords.
    """
    loader = StubLoader(monthly_fields())
    raw = cmip.load_native(loader, "FAKE-ESM", MEMBERS, TIME_SLICE, variables=VARIABLES, experiment_id=EXPERIMENT_ID)
    assert loader.calls == [{
        "variables": VARIABLES, "experiment_id": EXPERIMENT_ID, "source_id": "FAKE-ESM",
        "member_id": MEMBERS, "time_slice": TIME_SLICE,
    }]
    assert list(raw) == VARIABLES
    pr = raw["pr"]
    assert pr.dims == ("member", "time", "lat", "lon") and pr.attrs["units"] == "test"
    assert pr.member_id.values.tolist() == MEMBERS and "height" not in pr.coords
    np.testing.assert_allclose(pr, VARIABLES.index("pr"))


def test_load_native_missing_variable_raises():
    """A variable the loader cannot provide raises, naming it."""
    data = monthly_fields()
    del data["rlus"]
    with pytest.raises(ValueError, match=r"could not load \['rlus'\]"):
        cmip.load_native(StubLoader(data), "FAKE-ESM", MEMBERS, TIME_SLICE, variables=VARIABLES,
                         experiment_id=EXPERIMENT_ID)


@pytest.mark.parametrize("scale", [100.0, 1.0])
def test_load_land_fraction_percent_or_fraction(tmp_path, scale):
    """sftlf in % and sftlf stored as a fraction labelled "%" (as E3SM-1-0) both give a [0, 1] land fraction."""
    lf, src_grid = cmip.load_land_fraction(write_sftlf(tmp_path / "sftlf.nc", scale))
    xr.testing.assert_allclose(lf, land_fraction(1.0).assign_attrs(units="1"))
    assert lf.attrs["units"] == "1"
    assert src_grid.sizes["lat_b"] == LAT_N.size + 1 and src_grid.sizes["lon_b"] == LON_N.size + 1


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
