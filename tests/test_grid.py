"""
Tests for etunc/grid.py. Run from the project root with:

    python -m pytest tests/test_grid.py

Synthetic data only; takes a few seconds.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import etunc.config as config
import etunc.load.obs as lo
import etunc.grid as rg
import etunc.load.cmip as cmip


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


@pytest.mark.parametrize("res", ["2deg", "0.5", 0.7, 0.0, -1.0])
def test_unsupported_resolution_raises(res):
    """Unknown tags, and spacings that do not divide 180, raise."""
    with pytest.raises(ValueError, match="unsupported resolution"):
        rg.target_grid(res)


@pytest.mark.parametrize("res, nlat, nlon", [(0.25, 720, 1440), (2.0, 90, 180)])
def test_target_grid_any_spacing(res, nlat, nlon):
    """Any spacing that divides 180 gives a global grid with centers midway between edges at multiples of res."""
    g = rg.target_grid(res)
    assert (g.sizes["lat"], g.sizes["lon"]) == (nlat, nlon)
    np.testing.assert_allclose(g.lat_b, np.linspace(-90, 90, nlat + 1))
    np.testing.assert_allclose(g.lon_b, np.linspace(-180, 180, nlon + 1))
    np.testing.assert_allclose(g.lat, 0.5 * (g.lat_b.values[:-1] + g.lat_b.values[1:]))
    np.testing.assert_allclose(g.lon, 0.5 * (g.lon_b.values[:-1] + g.lon_b.values[1:]))


@pytest.mark.parametrize("res", [0.25, 2.0, "2deg", "0.5"])
def test_grid_tag_only_standard_grids(res):
    """grid_tag (output directory names) only knows the standard 0.5 and 1 deg grids."""
    with pytest.raises(ValueError, match="unsupported resolution"):
        rg.grid_tag(res)


def test_load_obs_reads_the_standard_grid_tags():
    """load_obs reads (and regrid_file writes) regridded files under the same tags as grid.RESOLUTIONS."""
    assert set(lo.RES_DIRS.values()) == set(rg.RESOLUTIONS)
    assert {float(r) for r in lo.RES_DIRS} == set(rg.RESOLUTIONS.values())


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
        cmip.regrid_to_target(coarse, res),
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


# ------------------------------------------------------------------
# Grid formatting and snapping
# ------------------------------------------------------------------

def test_format_grid():
    """Lon in [0, 360] becomes [-180, 180] and both lon and lat end up ascending, values following their cells."""
    lat, lon = np.array([45.0, -45.0]), np.array([0.0, 90.0, 180.0, 270.0])
    da = xr.DataArray(np.arange(8.0).reshape(2, 4), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    out = rg.format_grid(da)
    np.testing.assert_array_equal(out.lat, [-45.0, 45.0])
    np.testing.assert_array_equal(out.lon, [-180.0, -90.0, 0.0, 90.0])
    assert float(out.sel(lat=45.0, lon=-90.0)) == float(da.sel(lat=45.0, lon=270.0))
    assert float(out.sel(lat=-45.0, lon=-180.0)) == float(da.sel(lat=-45.0, lon=180.0))


def test_on_grid_snaps_and_cuts_to_lat_bnds():
    """A field within round-off of the target grid gets its exact coords and is cut to LAT_BNDS."""
    g = rg.target_grid(1.0)
    da = field(g.lat.values + 1e-6, g.lon.values - 1e-6)
    out = rg.on_grid(da, 1.0, "test")
    xr.testing.assert_identical(out.lat, g.lat.sel(lat=config.LAT_BNDS))
    xr.testing.assert_identical(out.lon, g.lon)


def test_on_grid_off_grid_raises():
    """A field off the target grid by more than the tolerance, or of another shape, raises."""
    g = rg.target_grid(1.0)
    with pytest.raises(ValueError, match="differ from the reference grid"):
        rg.on_grid(field(g.lat.values + 0.01, g.lon.values), 1.0, "shifted")
    with pytest.raises(ValueError, match="does not match"):
        rg.on_grid(field(g.lat.values, g.lon.values), 0.5, "coarse")


# ------------------------------------------------------------------
# Cell areas (port of ILAMB's CellAreas)
# ------------------------------------------------------------------

R = config.EARTH_RADIUS


def band_areas(lat_edges, lon_edges):
    """Exact (lat, lon) cell areas from cell edges in degrees."""
    dy = np.diff(np.sin(np.deg2rad(lat_edges)))
    dx = np.diff(np.deg2rad(lon_edges))
    return R**2 * np.outer(dy, dx)


@pytest.mark.parametrize("res", [10.0, 1.0])
def test_cell_area_global_sum(res):
    """On a global regular grid the cell areas, from centers or from bounds, sum to 4 pi R^2."""
    lat = np.arange(-90 + res / 2, 90, res)
    lon = np.arange(-180 + res / 2, 180, res)
    lat_bnds = np.stack([lat - res / 2, lat + res / 2], axis=1)
    lon_bnds = np.stack([lon - res / 2, lon + res / 2], axis=1)
    for area in (rg.cell_area(lat, lon), rg.cell_area(lat, lon, lat_bnds, lon_bnds)):
        assert area.shape == (lat.size, lon.size)
        np.testing.assert_allclose(area.sum(), 4 * np.pi * R**2, rtol=1e-12)


def test_cell_area_regular_grid_exact():
    """
    On a regular 10 deg grid the midpoint edges are the true edges, so centers and bounds both give the exact
    band areas, and lon in [0, 360] gives the same areas as lon in [-180, 180].
    """
    lat = np.arange(-85.0, 90.0, 10.0)
    lon180 = np.arange(-175.0, 180.0, 10.0)
    expected = band_areas(np.arange(-90.0, 91.0, 10.0), np.arange(-180.0, 181.0, 10.0))
    bnds = lambda c: np.stack([c - 5.0, c + 5.0], axis=1)  # noqa: E731
    np.testing.assert_allclose(rg.cell_area(lat, lon180), expected, rtol=1e-12)
    np.testing.assert_allclose(rg.cell_area(lat, lon180, bnds(lat), bnds(lon180)), expected, rtol=1e-12)
    np.testing.assert_allclose(rg.cell_area(lat, lon180 + 180.0), expected, rtol=1e-12)


def test_cell_area_irregular_lat_extrapolates_and_clips():
    """
    Without bounds, interior lat edges are midpoints, and the outer edges are extrapolated by half a cell and
    clipped to +-90 (as in ILAMB): lat [-80, -30, 40, 85] has edges [-90, -55, 5, 62.5, 90].
    """
    lat = np.array([-80.0, -30.0, 40.0, 85.0])
    lon = np.arange(-170.0, 180.0, 20.0)
    expected = band_areas(np.array([-90.0, -55.0, 5.0, 62.5, 90.0]), np.arange(-180.0, 181.0, 20.0))
    np.testing.assert_allclose(rg.cell_area(lat, lon), expected, rtol=1e-12)


# ------------------------------------------------------------------
# Valid-month product masks (make_mask.py)
# ------------------------------------------------------------------

def test_product_mask_counts_only_the_products_valid_months():
    """
    A land cell is in the mask when valid in at least min_frac of the product's valid months; months with no
    valid land data at all are left out of the denominator, and ocean cells are never in the mask.
    """
    coords = {"time": pd.date_range("2000-01-01", periods=4, freq="MS"), "lat": [0.0, 1.0], "lon": [0.0, 1.0]}
    da = xr.DataArray(np.ones((4, 2, 2)), dims=("time", "lat", "lon"), coords=coords)
    land = xr.DataArray([[True, True], [True, False]], dims=("lat", "lon"), coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]})
    da[3] = np.nan                    # a month without data (e.g. a missing file): not counted
    da[0, 0, 1] = np.nan              # one missing month at (0, 1): 2 of 3 valid months
    ds = rg.product_mask(da, land, 1.0, "test")
    assert int(ds["n_product_months"]) == 3
    np.testing.assert_array_equal(ds["n_valid_months"], [[3, 2], [3, 0]])  # 0 over the ocean
    np.testing.assert_array_equal(ds["product_mask"], [[True, False], [True, False]])
    np.testing.assert_allclose(ds["frac_valid_months"], [[1.0, 2 / 3], [1.0, np.nan]])
    np.testing.assert_array_equal(rg.product_mask(da, land, 0.6, "test")["product_mask"], [[True, True], [True, False]])
    with pytest.raises(ValueError, match="no valid land data"):
        rg.product_mask(da * np.nan, land, 1.0, "empty")
