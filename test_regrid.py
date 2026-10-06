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
import load_obs as lo
import regrid as rg
import regrid_cmip_esgf as rce
import regrid_obs as ro


def field(lat, lon, value=1.0):
    return xr.DataArray(np.full((len(lat), len(lon)), value), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


# ------------------------------------------------------------------
# Target grids
# ------------------------------------------------------------------

def test_target_grid_values():
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
    xr.testing.assert_identical(rg.target_grid(res), rg.target_grid(tag))
    assert rg.grid_tag(res) == rg.grid_tag(tag) == tag


@pytest.mark.parametrize("res", [0.25, 2.0, "2deg", "0.5"])
def test_unsupported_resolution_raises(res):
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
    da = field(np.round(10.05 + 0.1 * np.arange(20), 4), np.round(20.05 + 0.1 * np.arange(30), 4))
    grid = rg.source_grid(da)
    np.testing.assert_allclose(grid.lat_b, 10 + 0.1 * np.arange(21), atol=1e-9)
    np.testing.assert_allclose(grid.lon_b, 20 + 0.1 * np.arange(31), atol=1e-9)


def test_source_grid_clips_poles():
    da = field([89.85, 89.95], [0.05, 0.15])
    assert float(rg.source_grid(da).lat_b[-1]) == 90.0


def test_source_grid_irregular_raises():
    da = field([10.05, 10.15, 10.35], [20.05, 20.15, 20.25])
    with pytest.raises(ValueError, match="regularly spaced"):
        rg.source_grid(da)


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
    assert rg.approx_resolution(rg.target_grid(0.5)) == (0.5, 0.5)
    da = xr.DataArray(np.zeros((3, 4)), dims=("y", "x"),
                      coords={"latitude": ("y", [0.0, 2.0, 4.0]), "longitude": ("x", [0.0, 3.0, 6.0, 9.0])})
    assert rg.approx_resolution(da) == (2.0, 3.0)
    assert rg.approx_resolution(xr.DataArray([1.0], dims="t")) == (None, None)
