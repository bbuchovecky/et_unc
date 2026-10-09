"""
Tests for the code split out of binned_et.py: etunc.units, etunc.temporal,
etunc.binning, etunc.plotting, the coordinate checks in etunc.grid and
etunc.legacy. Run from the project root with:

    python -m pytest test_binned_et.py

The data loaders (load_cmip, load_cesm_*, load_ilamb_obs) and compute_cell_area
need external data/libraries and are not tested here.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import etunc.config as config
import etunc.units as units
import etunc.temporal as temporal
import etunc.grid as rg
import etunc.binning as binning
import etunc.plotting as plotting
import etunc.legacy as legacy


# ------------------------------------------------------------------
# Fixtures and helpers
# ------------------------------------------------------------------

LAT = np.linspace(-50, 80, 12)
LON = np.linspace(-180, 170, 18)
TIME = pd.date_range("2000-01", periods=36, freq="MS")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def monthly(rng, scale=1.0, members=None, units="W/m2"):
    dims = ("time", "lat", "lon")
    shape = (len(TIME), len(LAT), len(LON))
    coords = {"time": TIME, "lat": LAT, "lon": LON}
    if members is not None:
        dims = ("member",) + dims
        shape = (members,) + shape
        coords["member"] = np.arange(members)
        coords["member_id"] = ("member", [f"r{i + 1}i1p1f1" for i in range(members)])
    return xr.DataArray(rng.gamma(2.0, scale, shape), dims=dims, coords=coords, attrs={"units": units})


@pytest.fixture
def mask(rng):
    return xr.DataArray(rng.random((len(LAT), len(LON))) > 0.3, dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})


@pytest.fixture
def inputs(rng, mask):
    return binning.prepare_inputs(
        et=monthly(rng, 20), lai=monthly(rng, 1), precip=monthly(rng, 40), rn=monthly(rng, 30), mask=mask
    )


@pytest.fixture
def edges(inputs):
    y = binning.pooled_bin_edges(inputs["lai"], 5, name="lai", verbose=False)
    x = binning.pooled_bin_edges(inputs["ai"], 4, name="ai", verbose=False)
    return y, x


def brute_force_stats(target, y, x, y_edges, x_edges):
    """Reference implementation: loop over bins, same assignment rule as the module."""
    n_y, n_x = len(y_edges) - 1, len(x_edges) - 1
    valid = np.isfinite(target) & np.isfinite(y) & np.isfinite(x)
    t, y, x = target[valid], y[valid], x[valid]
    yi = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, n_y - 1)
    xi = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, n_x - 1)
    out = np.full((len(binning.STATS), n_y, n_x), np.nan)
    for j in range(n_y):
        for i in range(n_x):
            v = t[(yi == j) & (xi == i)]
            out[3, j, i] = v.size
            out[4, j, i] = np.sum(v > 0)
            if v.size:
                out[0, j, i] = v.mean()
                out[1, j, i] = v.var()
            if v.size > 1:
                out[2, j, i] = v.var(ddof=1)
    return out


# ------------------------------------------------------------------
# General helpers
# ------------------------------------------------------------------

def test_format_time_period():
    """A time slice is formatted as YYYYMM-YYYYMM."""
    assert temporal.format_time_period(slice("1950-01", "2014-12")) == "195001-201412"


def test_safe_squeeze_and_get_one_mid(rng):
    """
    safe_squeeze drops a length-1 dim and ignores a missing one; get_one_mid returns the member_id, or
    "onemember" without a member dim.
    """
    da = monthly(rng, members=1)
    assert "member" not in legacy.safe_squeeze(da, "member").dims
    assert legacy.safe_squeeze(da, "nonexistent").dims == da.dims
    assert legacy.get_one_mid(da) == "r1i1p1f1"
    assert legacy.get_one_mid(da.isel(member=0)) == "onemember"


def test_equal_coords_and_check_same_grid(inputs):
    """
    equal_coords accepts lat/lon differences within atol and rejects larger shifts or missing coords;
    check_same_grid raises on shifted values or a different shape.
    """
    lai = inputs["lai"]
    assert rg.equal_coords(lai, lai + 1, ("lat", "lon"))
    assert rg.equal_coords(lai, lai.assign_coords(lat=lai.lat + 1e-4), ("lat", "lon"))
    shifted = lai.assign_coords(lat=lai.lat + 0.1)
    assert not rg.equal_coords(lai, shifted, ("lat", "lon"))
    assert not rg.equal_coords(lai, lai, ("time",))
    with pytest.raises(ValueError, match="differ"):
        rg.check_same_grid(shifted, lai, "shifted")
    with pytest.raises(ValueError, match="grid shape"):
        rg.check_same_grid(lai.isel(lat=slice(1, None)), lai, "cropped")


def test_convert_units():
    """
    convert_units scales pr/et by L and CESM PRECT (m/s) by L times water density, flips the sign of ERA5 mer,
    and leaves other variables unchanged.
    """
    da = xr.DataArray([1.0, 2.0])
    np.testing.assert_allclose(units.convert_units("pr", da), da * config.LATENT_HEAT_VAPORIZATION)
    np.testing.assert_allclose(
        units.convert_units("PRECT_month_1", da), da * config.LATENT_HEAT_VAPORIZATION * config.LIQ_WATER_DENSITY
    )
    np.testing.assert_allclose(units.convert_units("mer", da), -da * config.LATENT_HEAT_VAPORIZATION)
    assert units.convert_units("et", da).attrs["units"] == "W/m2"
    xr.testing.assert_identical(units.convert_units("lai", da), da)


@pytest.mark.parametrize(
    "unit, factor",
    [
        ("W m-2", 1.0),
        ("W/m^2", 1.0),
        ("Watt m-2", 1.0),
        ("watt/m2", 1.0),
        ("kg/m2/s", config.LATENT_HEAT_VAPORIZATION),
        ("kg m-2 s-1", config.LATENT_HEAT_VAPORIZATION),
        ("MJ m-2 day-1", 1e6 / 86400),
    ],
)
def test_latent_heat_to_wm2(unit, factor):
    """
    Each supported ET or latent heat unit string (any case or spacing) is scaled to W/m2, keeping the other
    attrs and leaving the input unchanged.
    """
    da = xr.DataArray([1.0, 2.0], attrs={"units": unit, "long_name": "x"})
    out = units.latent_heat_to_wm2(da)
    np.testing.assert_allclose(out, da * factor)
    assert out.attrs == {"units": "W/m2", "long_name": "x"}
    assert da.attrs["units"] == unit


@pytest.mark.parametrize("unit, factor", [
    ("mm/day", config.LATENT_HEAT_VAPORIZATION / 86400),
    ("mm d-1", config.LATENT_HEAT_VAPORIZATION / 86400),
    ("kg m-2 s-1", config.LATENT_HEAT_VAPORIZATION),
    ("W m-2", 1.0),
])
def test_flux_to_wm2(unit, factor):
    """flux_to_wm2 adds mm/day (1 mm of water = 1 kg/m2 per 86400 s) to the units of latent_heat_to_wm2."""
    da = xr.DataArray([1.0, 2.0], attrs={"units": unit, "long_name": "x"})
    out = units.flux_to_wm2(da)
    np.testing.assert_allclose(out, da * factor, rtol=1e-12)
    assert out.attrs == {"units": "W/m2", "long_name": "x"}


def test_latent_heat_to_wm2_unknown_units():
    """An unsupported unit raises a ValueError that names it."""
    with pytest.raises(ValueError, match="mm d-1"):
        units.latent_heat_to_wm2(xr.DataArray([1.0], attrs={"units": "mm d-1"}))


def test_mask_greenland():
    """Land above the threshold is True except in Greenland; land below the threshold is False."""
    lat = np.array([-10.0, 72.0])
    lon = np.array([-60.0, -40.0])  # Amazon, Greenland
    lf = xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    mask = rg.mask_greenland(lf, 0.5)
    assert bool(mask.sel(lat=-10, lon=-60))
    assert not bool(mask.sel(lat=72, lon=-40))
    assert not bool(rg.mask_greenland(lf * 0.2, 0.5).sel(lat=-10, lon=-60))


def test_filter_all_variables_available(rng):
    """Only sources with every requested variable, all on matching grids, are kept."""
    a = monthly(rng)
    data = {
        "good": {"x": a, "y": a},
        "missing": {"x": a},
        "badgrid": {"x": a, "y": a.isel(lat=slice(1, None))},
    }
    out = legacy.filter_all_variables_available(data, ["x", "y"], coords=("lat", "lon"), verbose=False)
    assert list(out) == ["good"]


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------

def test_compute_annual_mean_is_days_weighted():
    """The annual mean weights each month by its number of days (2000 is a leap year)."""
    da = xr.DataArray(TIME.days_in_month.astype(float), dims="time", coords={"time": TIME})
    ann = temporal.compute_annual_mean(da)
    d = TIME.days_in_month[:12].to_numpy(dtype=float)  # 2000 (leap year)
    assert ann.dims == ("year",)
    np.testing.assert_allclose(ann.sel(year=2000), np.sum(d * d) / np.sum(d))


def test_compute_annual_mean_constant(rng):
    """The annual mean of a constant field is that constant."""
    da = xr.full_like(monthly(rng), 3.0)
    np.testing.assert_allclose(temporal.compute_annual_mean(da), 3.0)


@pytest.mark.parametrize(
    "how, dims",
    [
        ("mon", ("time", "lat", "lon")),
        ("year", ("year", "lat", "lon")),
        ("clim", ("lat", "lon")),
        ("year_max", ("year", "lat", "lon")),
        ("clim_max", ("lat", "lon")),
    ],
)
def test_aggregate_dims(rng, how, dims):
    """Each aggregation mode returns the expected dims."""
    assert temporal.aggregate(monthly(rng), how).dims == dims


def test_aggregate_values():
    """year_max, clim_max and clim give the right values for a known seasonal cycle plus trend."""
    # Seasonal cycle 0..11 every year, plus 12 * year index
    vals = np.tile(np.arange(12.0), 3) + np.repeat([0.0, 12.0, 24.0], 12)
    da = xr.DataArray(vals, dims="time", coords={"time": TIME})
    np.testing.assert_allclose(temporal.aggregate(da, "year_max"), [11, 23, 35])
    np.testing.assert_allclose(temporal.aggregate(da, "clim_max"), 11 + 12)  # max of mean seasonal cycle
    np.testing.assert_allclose(temporal.aggregate(da, "clim"), temporal.compute_annual_mean(da).mean())


def test_aggregate_preaggregated(rng):
    """
    For fields already aggregated, "clim" averages over year or passes through; modes that need time, and
    unknown modes, raise.
    """
    clim = temporal.aggregate(monthly(rng), "clim")
    xr.testing.assert_identical(temporal.aggregate(clim, "clim"), clim)
    ann = temporal.aggregate(monthly(rng), "year")
    xr.testing.assert_allclose(temporal.aggregate(ann, "clim"), ann.mean("year"))
    with pytest.raises(ValueError, match="time"):
        temporal.aggregate(clim, "year_max")
    with pytest.raises(ValueError, match="Unknown"):
        temporal.aggregate(monthly(rng), "decadal")


def test_net_radiation_sign_conventions():
    """CMIP, CESM and ERA5 net radiation each follow their source's flux sign convention."""
    one = xr.DataArray(1.0)
    assert float(units.net_radiation_cmip(one * 300, one * 50, one * 350, one * 400)) == 200
    assert float(units.net_radiation_cesm(one * 250, one * 50)) == 200  # FLNS is net LW up
    assert float(units.net_radiation_era5(one * 250, one * -50)) == 200  # both positive down


def test_compute_aridity_index():
    """AI = Rn / P, named "ai"; clip=True sets negative values to 0."""
    p = xr.DataArray([100.0, 100.0])
    rn = xr.DataArray([150.0, -20.0])
    np.testing.assert_allclose(binning.compute_aridity_index(p, rn), [1.5, -0.2])
    np.testing.assert_allclose(binning.compute_aridity_index(p, rn, clip=True), [1.5, 0.0])
    assert binning.compute_aridity_index(p, rn).name == "ai"


def test_prepare_inputs(rng, mask):
    """
    ET becomes annual means within time_slice and LAI and AI climatologies, all masked; lai_agg="year" keeps
    annual LAI.
    """
    et, lai, pr, rn = (monthly(rng, s) for s in (20, 1, 40, 30))
    out = binning.prepare_inputs(et, lai, pr, rn, mask=mask, time_slice=slice("2000-01", "2001-12"))
    assert out["et"].dims == ("year", "lat", "lon")
    assert list(out["et"].year.values) == [2000, 2001]
    assert out["lai"].dims == ("lat", "lon")
    assert out["ai"].dims == ("lat", "lon")
    for da in out.values():
        assert da.where(~mask).isnull().all()
        assert da.where(mask).notnull().sum() > 0
    expected_ai = temporal.aggregate(rn.sel(time=slice("2000-01", "2001-12")), "clim") / temporal.aggregate(
        pr.sel(time=slice("2000-01", "2001-12")), "clim"
    )
    xr.testing.assert_allclose(out["ai"], expected_ai.where(mask).rename("ai"), check_dim_order=True)

    out_ann = binning.prepare_inputs(et, lai, pr, rn, lai_agg="year")
    assert out_ann["lai"].dims == ("year", "lat", "lon")


# ------------------------------------------------------------------
# Bin edges
# ------------------------------------------------------------------

def test_build_edges_quantile_ignores_nan(rng):
    """Quantile edges ignore NaN and inf values."""
    vals = rng.random(1000)
    with_nan = np.append(vals, [np.nan, np.inf])
    edges = binning.build_edges(with_nan, 4)
    np.testing.assert_allclose(edges, np.quantile(vals, [0, 0.25, 0.5, 0.75, 1]))


def test_build_edges_linear():
    """Linear edges span the given range, or the data range when none is given."""
    np.testing.assert_allclose(binning.build_edges(np.array([0.3, 0.7]), 4, "linear", (0, 2)), [0, 0.5, 1, 1.5, 2])
    np.testing.assert_allclose(binning.build_edges(np.array([1.0, 3.0]), 2, "linear"), [1, 2, 3])


def test_build_edges_duplicates():
    """Duplicate quantile edges from many zeros are kept by default and merged with collapse_duplicates=True."""
    vals = np.concatenate([np.zeros(600), np.linspace(1, 2, 400)])
    kept = binning.build_edges(vals, 4)
    assert len(kept) == 5 and kept[0] == kept[1] == kept[2] == 0
    collapsed = binning.build_edges(vals, 4, collapse_duplicates=True)
    assert len(collapsed) == 3 and np.all(np.diff(collapsed) > 0)


def test_build_edges_errors():
    """All-NaN data, edges that collapse to a single value, and unknown strategies raise."""
    with pytest.raises(ValueError, match="no finite"):
        binning.build_edges(np.array([np.nan, np.nan]), 3)
    with pytest.raises(ValueError, match="collapsed"):
        binning.build_edges(np.zeros(10), 3, collapse_duplicates=True)
    with pytest.raises(ValueError, match="Unknown bin strategy"):
        binning.build_edges(np.arange(10.0), 3, strategy="log")


def test_pooled_bin_edges_pools_all_fields(rng):
    """Edges are built from the values of all fields together (NaN dropped), with name, dims and attrs set."""
    a = xr.DataArray(rng.random((3, 4)), dims=("lat", "lon"))
    b = xr.DataArray(rng.random((2, 5, 6)) + 1, dims=("member", "lat", "lon"))
    b[0, 0, 0] = np.nan
    edges = binning.pooled_bin_edges({"a": a, "b": b}, 5, name="lai", attrs={"units": "m2/m2"}, verbose=False)
    pooled = np.concatenate([a.values.ravel(), b.values.ravel()])
    np.testing.assert_allclose(edges.values, binning.build_edges(pooled, 5))
    assert edges.dims == ("edge",) and edges.name == "lai"
    assert edges.attrs["strategy"] == "quantile"
    assert edges.attrs["n_bins"] == 5
    assert edges.attrs["pooled_sources"] == ["a", "b"]
    assert edges.attrs["units"] == "m2/m2"


def test_pooled_bin_edges_warns_on_zeros():
    """A warning is raised when many values are exactly 0."""
    da = xr.DataArray(np.concatenate([np.zeros(50), np.arange(1.0, 51)]))
    with pytest.warns(UserWarning, match="are 0"):
        binning.pooled_bin_edges(da, 3, name="lai", verbose=False)


# ------------------------------------------------------------------
# Binning core
# ------------------------------------------------------------------

def test_bin_stats_flat_matches_brute_force(rng):
    """Binned statistics match a loop over bins on random data with NaNs."""
    n = 5000
    t = rng.normal(50, 10, n)
    y = rng.gamma(2, 1, n)
    x = rng.random(n) * 3
    t[::17] = np.nan
    y[::23] = np.nan
    y_edges = np.quantile(y[np.isfinite(y)], np.linspace(0, 1, 6))
    x_edges = np.linspace(0, 3, 5)
    result = binning._bin_stats_flat(t, y, x, y_edges, x_edges)
    assert result.shape == (len(binning.STATS), 5, 4)
    np.testing.assert_allclose(result, brute_force_stats(t, y, x, y_edges, x_edges), equal_nan=True)


def test_bin_stats_flat_spike_goes_to_last_duplicate_bin():
    """Values equal to a repeated edge fall in the last of the duplicate bins."""
    y_edges = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    x_edges = np.array([0.0, 1.0])
    y = np.array([0.0, 0.0, 0.0, 0.5, 1.5])
    result = binning._bin_stats_flat(np.ones(5), y, np.full(5, 0.5), y_edges, x_edges)
    np.testing.assert_array_equal(result[binning.STATS.index("count"), :, 0], [0, 0, 4, 1])


def test_bin_stats_flat_clips_out_of_range_values():
    """Values outside the edges are counted in the first or last bin."""
    edges = np.array([0.0, 1.0, 2.0])
    result = binning._bin_stats_flat(np.ones(2), np.array([-5.0, 99.0]), np.array([0.5, 0.5]), edges, edges)
    np.testing.assert_array_equal(result[binning.STATS.index("count"), :, 0], [1, 1])


def test_bin_stats_flat_count_pos_and_empty_bins():
    """
    count_pos counts positive values, mean and sample variance are right, and empty bins have NaN mean and count
    0.
    """
    edges = np.array([0.0, 1.0, 2.0])
    t = np.array([-1.0, 2.0, 3.0])
    y = np.array([0.5, 0.5, 0.5])
    result = binning._bin_stats_flat(t, y, y, edges, edges)
    stats = dict(zip(binning.STATS, result[:, 0, 0]))
    assert stats["count"] == 3 and stats["count_pos"] == 2
    np.testing.assert_allclose(stats["mean"], 4 / 3)
    np.testing.assert_allclose(stats["var_samp"], np.var(t, ddof=1))
    assert np.isnan(result[0, 1, 1]) and result[3, 1, 1] == 0


# ------------------------------------------------------------------
# bin_stats
# ------------------------------------------------------------------

def test_bin_stats_broadcasts_climatology_over_years(inputs, edges):
    """
    Each annual ET sample is binned by its gridcell's climatological LAI and AI, matching a brute-force
    reference.
    """
    y_edges, x_edges = edges
    bs = binning.bin_stats(inputs["et"], inputs["lai"], inputs["ai"], y_edges, x_edges)
    assert bs.dims == ("stats", "y_bin", "x_bin")
    assert bs.sizes == {"stats": len(binning.STATS), "y_bin": 5, "x_bin": 4}

    et = inputs["et"]
    lai_b = inputs["lai"].expand_dims(year=et.year)
    ai_b = inputs["ai"].expand_dims(year=et.year)
    expected = brute_force_stats(
        et.values.ravel(), lai_b.values.ravel(), ai_b.values.ravel(), y_edges.values, x_edges.values
    )
    np.testing.assert_allclose(bs.values, expected, equal_nan=True)
    assert bs.sel(stats="count").sum() == np.isfinite(et).sum()


def test_bin_stats_coords_and_attrs(inputs, edges):
    """
    The output has bin index, lower/upper/center and quantile label coords, plus the name, units, edge and
    caller attrs.
    """
    y_edges, x_edges = edges
    bs = binning.bin_stats(
        inputs["et"], inputs["lai"], inputs["ai"], y_edges, x_edges, name="evspsbl", attrs={"source_id": "M"}
    )
    assert bs.name == "evspsbl"
    np.testing.assert_array_equal(bs.y_bin, np.arange(5))
    np.testing.assert_allclose(bs.y_bin_lower, y_edges.values[:-1])
    np.testing.assert_allclose(bs.x_bin_upper, x_edges.values[1:])
    np.testing.assert_allclose(bs.y_bin_center, 0.5 * (y_edges.values[:-1] + y_edges.values[1:]))
    assert list(bs.y_bin_label.values) == ["Q0-Q20", "Q20-Q40", "Q40-Q60", "Q60-Q80", "Q80-Q100"]
    assert bs.attrs["units"] == "W/m2"
    assert bs.attrs["y_variable"] == "lai" and bs.attrs["x_variable"] == "ai"
    assert bs.attrs["y_strategy"] == "quantile"
    assert bs.attrs["source_id"] == "M"


def test_bin_stats_plain_array_edges_have_no_labels(inputs):
    """Edges given as plain arrays give no quantile labels and an "unknown" strategy."""
    bs = binning.bin_stats(inputs["et"], inputs["lai"], inputs["ai"], np.linspace(0, 5, 4), np.linspace(0, 5, 3))
    assert "y_bin_label" not in bs.coords
    assert bs.attrs["y_strategy"] == "unknown"


def test_bin_stats_mask(inputs, edges, mask):
    """Passing mask= gives the same result as masking the inputs beforehand."""
    y_edges, x_edges = edges
    unmasked = {k: v.fillna(1.0) for k, v in inputs.items()}
    bs = binning.bin_stats(unmasked["et"], unmasked["lai"], unmasked["ai"], y_edges, x_edges, mask=mask)
    expected = binning.bin_stats(inputs["et"], inputs["lai"], inputs["ai"], y_edges, x_edges)
    np.testing.assert_allclose(bs.values, expected.values, equal_nan=True)


def test_bin_stats_per_member(rng, mask, edges):
    """member_dim bins each member separately (same as binning it alone); without it, all members are pooled."""
    y_edges, x_edges = edges
    inp = binning.prepare_inputs(
        monthly(rng, 20, members=3), monthly(rng, 1, members=3),
        monthly(rng, 40, members=3), monthly(rng, 30, members=3), mask=mask,
    )
    bs = binning.bin_stats(inp["et"], inp["lai"], inp["ai"], y_edges, x_edges, member_dim="member")
    assert bs.dims == ("member", "stats", "y_bin", "x_bin")
    assert list(bs.member_id.values) == ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]
    for m in range(3):
        single = binning.bin_stats(
            inp["et"].sel(member=m), inp["lai"].sel(member=m), inp["ai"].sel(member=m), y_edges, x_edges
        )
        np.testing.assert_allclose(bs.sel(member=m).values, single.values, equal_nan=True)

    pooled = binning.bin_stats(inp["et"], inp["lai"], inp["ai"], y_edges, x_edges)
    assert pooled.dims == ("stats", "y_bin", "x_bin")
    np.testing.assert_allclose(pooled.sel(stats="count"), bs.sel(stats="count").sum("member"))


def test_pool_members_matches_binning_pooled_samples(rng, mask, edges):
    """Pooling per-member statistics gives the statistics of all members' samples binned together."""
    y_edges, x_edges = edges
    inp = binning.prepare_inputs(
        monthly(rng, 20, members=3), monthly(rng, 1, members=3),
        monthly(rng, 40, members=3), monthly(rng, 30, members=3), mask=mask,
    )
    bs = binning.bin_stats(inp["et"], inp["lai"], inp["ai"], y_edges, x_edges, member_dim="member", name="et")
    pooled = binning.pool_members(bs)
    direct = binning.bin_stats(inp["et"], inp["lai"], inp["ai"], y_edges, x_edges, name="et")
    assert pooled.dims == ("stats", "y_bin", "x_bin") and pooled.name == "et"
    assert list(pooled.stats.values) == list(binning.STATS)
    assert pooled.attrs["members"] == ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]
    assert "member_id" not in pooled.coords
    np.testing.assert_allclose(pooled.values, direct.values, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_bin_stats_rejects_mismatched_grids(inputs, edges):
    """Inputs whose lat/lon values differ raise instead of being aligned."""
    y_edges, x_edges = edges
    shifted = inputs["lai"].assign_coords(lat=inputs["lai"].lat + 0.01)
    with pytest.raises(ValueError, match="identical coordinates"):
        binning.bin_stats(inputs["et"], shifted, inputs["ai"], y_edges, x_edges)


def test_bin_stats_by_source(inputs, edges):
    """Per-source results are concatenated along `dim`, each matching bin_stats on that source alone."""
    y_edges, x_edges = edges
    other = {k: v * 2 for k, v in inputs.items()}
    bs = binning.bin_stats_by_source({"A": inputs, "B": other}, y_edges, x_edges, dim="sid", verbose=False)
    assert bs.dims == ("sid", "stats", "y_bin", "x_bin")
    assert list(bs.sid.values) == ["A", "B"]
    single = binning.bin_stats(inputs["et"], inputs["lai"], inputs["ai"], y_edges, x_edges)
    np.testing.assert_allclose(bs.sel(sid="A").values, single.values, equal_nan=True)


def test_valid_and_common_area(capsys):
    """
    valid_area needs ET in some year plus LAI and a finite AI (in every member with member_dim); common_area
    is the land cells valid in every dataset, and prints one line per dataset.
    """
    coords = {"lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]}
    et = xr.DataArray(np.ones((2, 2, 2, 3)), dims=("member", "year", "lat", "lon"), coords=coords)
    lai = xr.DataArray(np.ones((2, 2, 3)), dims=("member", "lat", "lon"), coords=coords)
    ai = lai.copy()
    et[:, :, 0, 0] = np.nan                 # no ET in any year
    et[0, 0, 0, 1] = np.nan                 # one missing year only: still valid
    lai[1, 0, 2] = np.nan                   # LAI missing in member 1 only
    ai[:, 1, 0] = np.inf                    # AI not finite
    inp = {"et": et, "lai": lai, "ai": ai}

    per_member = binning.valid_area(inp)
    assert per_member.dims == ("member", "lat", "lon")
    np.testing.assert_array_equal(per_member.sel(member=0), [[False, True, True], [False, True, True]])
    np.testing.assert_array_equal(per_member.sel(member=1), [[False, True, False], [False, True, True]])
    all_members = binning.valid_area(inp, member_dim="member")
    np.testing.assert_array_equal(all_members, [[False, True, False], [False, True, True]])

    land = xr.DataArray([[1, 1, 1], [1, 0, 1]], dims=("lat", "lon"), coords=coords)
    no_lai_at_lat1 = {**inp, "lai": inp["lai"].where(inp["lai"].lat == 0)}
    area = binning.common_area({"A": inp, "B": no_lai_at_lat1}, land, member_dim="member")
    assert area.name == "area_mask"
    np.testing.assert_array_equal(area, [[False, True, False], [False, False, False]])
    out = capsys.readouterr().out
    assert out.count("valid gridcells") == 2 and "common area: 1.0000e+00 of 5.0000e+00" in out


def test_open_bin_stats_roundtrip(tmp_path, inputs, edges):
    """A saved bin_stats file opens back unchanged."""
    bs = binning.bin_stats(inputs["et"], inputs["lai"], inputs["ai"], *edges, name="et")
    bs.to_netcdf(tmp_path / "bs.nc")
    opened = legacy.open_bin_stats(tmp_path / "bs.nc")
    xr.testing.assert_allclose(opened, bs)


def test_open_bin_stats_legacy_file(tmp_path):
    """Notebook-era files with edges only in attrs get their bin coords rebuilt."""
    # Older notebook output: no y_bin/x_bin coords, edges only in attrs
    old = xr.DataArray(
        np.ones((4, 3, 2)),
        name="evspsbl",
        dims=("stats", "y_bin", "x_bin"),
        coords={"stats": ["mean", "var_pop", "var_samp", "count"]},
        attrs={"y_bin_edges": [0.0, 0.0, 1.0, 2.0], "x_bin_edges": [0.0, 0.5, 1.0]},
    )
    old.to_netcdf(tmp_path / "legacy.nc")
    opened = legacy.open_bin_stats(tmp_path / "legacy.nc")
    np.testing.assert_array_equal(opened.y_bin, [0, 1, 2])
    np.testing.assert_allclose(opened.y_bin_lower, [0, 0, 1])
    np.testing.assert_allclose(opened.x_bin_upper, [0.5, 1.0])


def test_open_bin_stats_multiple_vars_needs_var(tmp_path):
    """A file with several variables needs var= to choose one."""
    xr.Dataset({"a": ("x", [1.0]), "b": ("x", [2.0])}).to_netcdf(tmp_path / "two.nc")
    with pytest.raises(ValueError, match="pass `var`"):
        legacy.open_bin_stats(tmp_path / "two.nc")
    assert legacy.open_bin_stats(tmp_path / "two.nc", var="b").name == "b"


# ------------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------------

def make_bs(means, counts, var_samp=None, y_edges=None, x_edges=None, dim="sid"):
    """Build a (dim, stats, y_bin, x_bin) bin-stats array from (dim, y, x) means and counts."""
    means = np.asarray(means, dtype=float)
    counts = np.broadcast_to(np.asarray(counts, dtype=float), means.shape)
    var_samp = np.ones_like(means) if var_samp is None else np.broadcast_to(var_samp, means.shape)
    data = np.stack([means, var_samp, var_samp, counts, counts], axis=1)
    n_y, n_x = means.shape[1:]
    y_edges = np.arange(n_y + 1.0) if y_edges is None else np.asarray(y_edges)
    x_edges = np.arange(n_x + 1.0) if x_edges is None else np.asarray(x_edges)
    return binning.ensure_bin_coords(xr.DataArray(
        data,
        dims=(dim, "stats", "y_bin", "x_bin"),
        coords={"stats": list(binning.STATS)},
        attrs={"y_bin_edges": y_edges, "x_bin_edges": x_edges, "units": "W/m2"},
    ))


def test_drop_zero_width_bins():
    """Bins whose lower and upper edges are equal are dropped from both axes."""
    bs = make_bs(np.ones((2, 4, 3)), 10, y_edges=[0, 0, 0, 1, 2], x_edges=[0, 1, 1, 2])
    dropped = binning.drop_zero_width_bins(bs)
    np.testing.assert_array_equal(dropped.y_bin, [2, 3])
    np.testing.assert_array_equal(dropped.x_bin, [0, 2])
    np.testing.assert_allclose(dropped.y_bin_lower, [0, 1])


def test_frac_count():
    """Each bin's count as a fraction of the total count, summing to 1."""
    bs = make_bs(np.ones((2, 2, 2)), [[1, 3], [2, 4]])
    fc = binning.frac_count(bs)
    np.testing.assert_allclose(fc.sum(["y_bin", "x_bin"]), 1.0)
    np.testing.assert_allclose(fc.isel(sid=0).values, [[0.1, 0.3], [0.2, 0.4]])


def test_frac_valid():
    """The fraction of entries along `dim` (e.g. models) with a valid mean in each bin."""
    means = np.ones((4, 1, 2))
    means[:3, 0, 1] = np.nan
    np.testing.assert_allclose(binning.frac_valid(make_bs(means, 10), "sid").values, [[1.0, 0.25]])


def test_significance():
    """A bin mean is significant only when it is far from 0 and has more than n_min samples."""
    # bin 0: mean far from 0 with many samples; bin 1: mean 0; bin 2: too few samples
    means = np.array([[[50.0, 0.0, 50.0]]])
    counts = np.array([[[100, 100, 5]]])
    sig = binning.test_significance(make_bs(means, counts, var_samp=4.0), alpha=0.05, n_min=10)
    np.testing.assert_array_equal(sig.values, [[[True, False, False]]])


def test_ensemble_spread():
    """The std or var of bin means along `dim`, with the min_frac and signif filters."""
    means = np.array([[[1.0, 5.0]], [[3.0, np.nan]], [[5.0, np.nan]]])
    bs = make_bs(means, 10)
    np.testing.assert_allclose(binning.ensemble_spread(bs, "sid").values[0, 0], np.std([1, 3, 5]))
    np.testing.assert_allclose(binning.ensemble_spread(bs, "sid", how="var").values[0, 0], np.var([1, 3, 5]))
    limited = binning.ensemble_spread(bs, "sid", min_frac=0.5)
    assert np.isfinite(limited.values[0, 0]) and np.isnan(limited.values[0, 1])
    signif = xr.DataArray([[[True, True]], [[True, True]], [[False, True]]], dims=("sid", "y_bin", "x_bin"))
    np.testing.assert_allclose(binning.ensemble_spread(bs, "sid", signif=signif).values[0, 0], np.std([1, 3]))


# ------------------------------------------------------------------
# Plotting (smoke tests)
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_finish_saves_and_closes(tmp_path):
    """_finish saves and closes a figure when given a path, and leaves it open without one."""
    fig = plt.figure()
    out = tmp_path / "sub" / "fig.png"
    assert plotting.finish(fig, out) is fig
    assert out.exists()
    assert not plt.fignum_exists(fig.number)
    fig2 = plt.figure()
    plotting.finish(fig2, None)
    assert plt.fignum_exists(fig2.number)


def test_plot_bin_field_hatch_and_ticks(tmp_path):
    """plot_bin_field hatches the flagged bins, labels the ticks with the edges, and saves to fout."""
    bs = make_bs(np.arange(6.0).reshape(1, 2, 3), 10, y_edges=[0, 0.5, 1.5], x_edges=[0, 1, 2, 3])
    field = bs.sel(stats="mean").isel(sid=0)
    hatch = field > 3
    fig = plotting.plot_bin_field(field, hatch=hatch)
    ax = fig.axes[0]
    assert len(ax.patches) == int(hatch.sum())
    assert [t.get_text() for t in ax.get_yticklabels()] == ["0", "0.5", "1.5"]
    plotting.plot_bin_field(field, fout=tmp_path / "field.png")
    assert (tmp_path / "field.png").exists()


def test_plot_bin_summary_and_facets(tmp_path, rng, mask, edges):
    """plot_bin_summary and plot_bin_facets run on per-member stats and save their figures."""
    y_edges, x_edges = edges
    inp = binning.prepare_inputs(
        monthly(rng, 20, members=2), monthly(rng, 1, members=2),
        monthly(rng, 40, members=2), monthly(rng, 30, members=2), mask=mask,
    )
    bs = binning.bin_stats(inp["et"], inp["lai"], inp["ai"], y_edges, x_edges, member_dim="member")
    assert len(plotting.plot_bin_summary(bs, dim="member").axes) == 6  # 3 panels + 3 colorbars
    plotting.plot_bin_summary(bs.isel(member=0), fout=tmp_path / "summary.png")
    fg = plotting.plot_bin_facets(bs, "member", signif=binning.test_significance(bs), fout=tmp_path / "facets.png")
    assert fg.axs.size >= 2
    assert (tmp_path / "summary.png").exists() and (tmp_path / "facets.png").exists()


def test_plot_edges_and_hist(tmp_path, inputs, edges):
    """plot_edges, plot_edges_compare and plot_input_hist (log scale, edge lines) run and save."""
    y_edges, x_edges = edges
    plotting.plot_edges(y_edges, fout=tmp_path / "edges.png")
    plotting.plot_edges_compare({"lai": y_edges, "ai": x_edges}, fout=tmp_path / "cmp.png")
    fig = plotting.plot_input_hist({"a": inputs["lai"], "b": inputs["lai"] * 1.1}, y_edges, log=True)
    assert fig.axes[0].get_yscale() == "log"
    assert len(fig.axes[0].get_lines()) >= 1  # edge lines
    for name in ("edges.png", "cmp.png"):
        assert (tmp_path / name).exists()


def test_map_plots(tmp_path, inputs):
    """quick_map saves a map, and plot_input_maps draws one map per input."""
    plotting.quick_map(inputs["lai"], tmp_path / "map.png", title="lai")
    fig = plotting.plot_input_maps(inputs, title="inputs")
    assert len([ax for ax in fig.axes if hasattr(ax, "projection")]) == 3
    assert (tmp_path / "map.png").exists()


def test_facets_removes_unused_panels():
    """facets returns exactly n axes, in rows of ncols, with the unused panels of the last row removed."""
    fig, axs = plotting.facets(5, ncols=3, sharex=True)
    assert len(axs) == 5 and len(fig.axes) == 5
    assert axs[0].get_subplotspec().get_gridspec().get_geometry() == (2, 3)
    fig, axs = plotting.facets(2, ncols=4, maps=True)
    assert len(axs) == 2 and all(hasattr(ax, "coastlines") for ax in axs)
    plt.close("all")


def test_hatch_bins_one_patch_per_true_bin():
    """hatch_bins draws one hatched rectangle centred on each True (y_bin, x_bin) cell; NaN counts as False."""
    hatch = xr.DataArray([[True, False, np.nan], [False, True, True]], dims=("y_bin", "x_bin"),
                         coords={"y_bin": [0, 1], "x_bin": [0, 1, 2]})
    fig, ax = plt.subplots()
    plotting.hatch_bins(ax, hatch.T)  # any dim order
    corners = sorted(p.get_xy() for p in ax.patches)
    assert corners == [(-0.5, -0.5), (0.5, 0.5), (1.5, 0.5)]
    assert all(p.get_hatch() == "///" for p in ax.patches)
    plt.close(fig)


def test_save_map_writes_styled_map(tmp_path, inputs, capsys):
    """save_map writes the map to fout (creating its directory), labels the colorbar with variable and units."""
    fout = tmp_path / "maps" / "lai.png"
    da = inputs["lai"].assign_attrs(units="m2/m2")
    plotting.save_map(da, "lai", fout, title="LAI")
    assert fout.exists() and str(fout) in capsys.readouterr().out
    with pytest.raises(KeyError):
        plotting.save_map(da, "not_a_variable", tmp_path / "x.png")
    plt.close("all")
