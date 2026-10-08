"""
Tests for regrid.py. Run from the project root with:

    python -m pytest test_regrid.py

Synthetic data only; takes a few seconds.
"""

import numpy as np
import pytest
import xarray as xr

import cmip_binned_et as cb
import ilamb_binned_et as ib
import etunc.load.obs as lo
import etunc.grid as rg
import regrid_cmip_esgf as rce
import regrid_obs as ro


def field(lat, lon, value=1.0):
    return xr.DataArray(np.full((len(lat), len(lon)), value), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


# ------------------------------------------------------------------
# Target grids
# ------------------------------------------------------------------

def test_target_grid_values():
    """
    The 1 deg and 0.5 deg grids have the expected centers and edges, with each center midway between its edges.
    """
    g1, g05 = rg.target_grid(1.0), rg.target_grid(0.5)
    np.testing.assert_array_equal(g1.lat, np.arange(-89.5, 90, 1.0))
    np.testing.assert_array_equal(g1.lon, np.arange(-179.5, 180, 1.0))
    np.testing.assert_array_equal(g05.lat, np.arange(-89.75, 90, 0.5))
    np.testing.assert_array_equal(g05.lon, np.arange(-179.75, 180, 0.5))
    for g, res in ((g1, 1.0), (g05, 0.5)):
        np.testing.assert_array_equal(g.lat_b, np.linspace(-90, 90, round(180 / res) + 1))
        np.testing.assert_array_equal(g.lon_b, np.linspace(-180, 180, round(360 / res) + 1))
        np.testing.assert_allclose(g.lat, (g.lat_b[:-1].values + g.lat_b[1:].values) / 2, atol=1e-12)


@pytest.mark.parametrize("res, tag", [(0.5, "0.5deg"), (1.0, "1deg"), (1, "1deg")])
def test_target_grid_by_tag_or_spacing(res, tag):
    """A grid can be requested by spacing or by tag, and grid_tag maps both to the tag."""
    xr.testing.assert_identical(rg.target_grid(res), rg.target_grid(tag))
    assert rg.grid_tag(res) == rg.grid_tag(tag) == tag


@pytest.mark.parametrize("res", [0.25, 2.0, "2deg", "0.5"])
def test_unsupported_resolution_raises(res):
    """Spacings or tags other than 0.5 and 1 deg raise."""
    with pytest.raises(ValueError, match="unsupported resolution"):
        rg.target_grid(res)


def test_scripts_use_target_grids():
    """Every script regrids onto the same `target_grid`, and load_obs reads the same tags."""
    xr.testing.assert_identical(ib.TARGET_GRID, rg.target_grid(0.5))
    xr.testing.assert_identical(cb.TARGET_GRID, rg.target_grid(1.0))
    assert cb.GRID_TAG == "1deg"
    assert rce.TARGET_RES in rg.RESOLUTIONS.values()
    assert ro.RESOLUTIONS == rg.RESOLUTIONS
    assert set(lo.RES_DIRS.values()) == set(rg.RESOLUTIONS)


# ------------------------------------------------------------------
# Source grids
# ------------------------------------------------------------------

def test_source_grid_bounds():
    """The edges of a regular 0.1 deg grid are halfway between its centers."""
    da = field(np.round(10.05 + 0.1 * np.arange(20), 4), np.round(20.05 + 0.1 * np.arange(30), 4))
    grid = rg.source_grid(da)
    np.testing.assert_allclose(grid.lat_b, 10 + 0.1 * np.arange(21), atol=1e-9)
    np.testing.assert_allclose(grid.lon_b, 20 + 0.1 * np.arange(31), atol=1e-9)


def test_source_grid_clips_poles():
    """Edges beyond the pole are clipped to 90."""
    da = field([89.85, 89.95], [0.05, 0.15])
    assert float(rg.source_grid(da).lat_b[-1]) == 90.0


def test_source_grid_irregular_raises():
    """An irregularly spaced grid raises."""
    da = field([10.05, 10.15, 10.35], [20.05, 20.15, 20.25])
    with pytest.raises(ValueError, match="regularly spaced"):
        rg.source_grid(da)


def test_cell_edges_from_bounds_and_midpoints():
    """
    Edges come from CF bounds when given and from midpoints otherwise; bounds that do not enclose the centers
    raise.
    """
    centers = np.array([-60.0, -20.0, 10.0, 70.0])
    bnds = np.array([[-90.0, -40.0], [-40.0, 0.0], [0.0, 30.0], [30.0, 90.0]])
    np.testing.assert_array_equal(rg.cell_edges(centers, bnds), [-90, -40, 0, 30, 90])
    np.testing.assert_array_equal(rg.cell_edges(centers), [-80, -40, -5, 40, 100])
    with pytest.raises(ValueError, match="do not enclose"):
        rg.cell_edges(centers, bnds[::-1])


def test_bounded_source_grid():
    """Irregular lat: edges from `lat_bnds` when present, else midpoints; lat clipped to [-90, 90]."""
    lat, lon = np.array([-60.0, -20.0, 10.0, 70.0]), np.arange(0.0, 360, 90)
    ds = field(lat, lon).to_dataset(name="sftlf")
    np.testing.assert_array_equal(rg.bounded_source_grid(ds).lat_b, [-80, -40, -5, 40, 90])
    np.testing.assert_array_equal(rg.bounded_source_grid(ds).lon_b, [-45, 45, 135, 225, 315])
    ds["lat_bnds"] = (("lat", "bnds"), [[-90, -40], [-40, 0], [0, 30], [30, 90]])
    np.testing.assert_array_equal(rg.bounded_source_grid(ds).lat_b, [-90, -40, 0, 30, 90])


def test_bounded_conservative_regridder_conserves():
    """Global Gaussian-like source: a constant stays constant, and the masked-area mean is conserved."""
    lat = np.sort(np.concatenate([[-88.0, 88.0], np.linspace(-80, 80, 30) + 0.3 * np.sin(np.arange(30))]))
    da = field(lat, np.arange(0.0, 360, 2.5), value=4.0)
    regridder = rg.bounded_conservative_regridder(da, 1.0)
    np.testing.assert_allclose(regridder(da), 4.0)
    # NaN over half the domain: target cells away from the edge are the mean of the valid part
    half = da.where(da.lon < 180)
    out = regridder(half, skipna=True, na_thres=0.5)
    np.testing.assert_allclose(out.sel(lon=slice(10, 170)), 4.0)
    assert out.sel(lon=slice(-170, -10)).isnull().all()


# ------------------------------------------------------------------
# Regridders
# ------------------------------------------------------------------

@pytest.mark.parametrize("res", [0.5, 1.0])
def test_regridded_coords_are_target_grid(res):
    """Conservative, bilinear and generic regridders all return exactly the target lat/lon."""
    grid = rg.target_grid(res)
    fine = field(np.round(10.05 + 0.1 * np.arange(20), 4), np.round(20.05 + 0.1 * np.arange(20), 4))
    coarse = field(np.arange(-88.75, 90, 2.5), np.arange(-178.75, 180, 2.5))
    outs = [
        rg.conservative_regridder(fine, res)(fine),
        rg.bilinear_regridder(coarse, res)(coarse),
        rce.regrid_to_target(coarse, res),
    ]
    for out in outs:
        np.testing.assert_array_equal(out.lat, grid.lat)
        np.testing.assert_array_equal(out.lon, grid.lon)


def test_conservative_regridder_cached():
    """The regridder is reused for the same source grid and resolution, but not for another resolution."""
    da = field(np.round(10.05 + 0.1 * np.arange(10), 4), np.round(20.05 + 0.1 * np.arange(10), 4))
    assert rg.conservative_regridder(da, 1.0) is rg.conservative_regridder(da * 2, "1deg")
    assert rg.conservative_regridder(da, 1.0) is not rg.conservative_regridder(da, 0.5)


def test_bilinear_constant_and_unmapped_nan():
    """Constant 1 deg sub-domain field stays constant at 0.5 deg; points outside it are NaN."""
    da = field(np.arange(10.5, 20, 1.0), np.arange(20.5, 30, 1.0), value=3.0)
    out = rg.bilinear_regridder(da, 0.5)(da)
    inside = out.sel(lat=slice(10.5, 19.5), lon=slice(20.5, 29.5))
    np.testing.assert_allclose(inside, 3.0)
    assert int(out.notnull().sum()) == inside.size


def test_bilinear_periodic_across_dateline():
    """Global source on [0, 360): target points between 359.5 and 0.5 E are interpolated, not NaN."""
    da = field(np.arange(-89.5, 90, 1.0), np.arange(0.5, 360, 1.0), value=2.0)
    out = rg.bilinear_regridder(da, 0.5)(da)
    np.testing.assert_allclose(out.sel(lat=slice(-89, 89)), 2.0)


# ------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------

def test_approx_resolution():
    """The grid spacing is found from lat/lon or latitude/longitude coords, and is (None, None) without them."""
    assert rg.approx_resolution(rg.target_grid(0.5)) == (0.5, 0.5)
    da = xr.DataArray(np.zeros((3, 4)), dims=("y", "x"),
                      coords={"latitude": ("y", [0.0, 2.0, 4.0]), "longitude": ("x", [0.0, 3.0, 6.0, 9.0])})
    assert rg.approx_resolution(da) == (2.0, 3.0)
    assert rg.approx_resolution(xr.DataArray([1.0], dims="t")) == (None, None)
