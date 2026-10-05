"""
Tests for binned_et.py. Run from the project root with:

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

import binned_et as be


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
    return be.prepare_inputs(
        et=monthly(rng, 20), lai=monthly(rng, 1), precip=monthly(rng, 40), rn=monthly(rng, 30), mask=mask
    )


@pytest.fixture
def edges(inputs):
    y = be.pooled_bin_edges(inputs["lai"], 5, name="lai", verbose=False)
    x = be.pooled_bin_edges(inputs["ai"], 4, name="ai", verbose=False)
    return y, x


def brute_force_stats(target, y, x, y_edges, x_edges):
    """Reference implementation: loop over bins, same assignment rule as the module."""
    n_y, n_x = len(y_edges) - 1, len(x_edges) - 1
    valid = np.isfinite(target) & np.isfinite(y) & np.isfinite(x)
    t, y, x = target[valid], y[valid], x[valid]
    yi = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, n_y - 1)
    xi = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, n_x - 1)
    out = np.full((len(be.STATS), n_y, n_x), np.nan)
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
    assert be.format_time_period(slice("1950-01", "2014-12")) == "195001-201412"


def test_safe_squeeze_and_get_one_mid(rng):
    da = monthly(rng, members=1)
    assert "member" not in be.safe_squeeze(da, "member").dims
    assert be.safe_squeeze(da, "nonexistent").dims == da.dims
    assert be.get_one_mid(da) == "r1i1p1f1"
    assert be.get_one_mid(da.isel(member=0)) == "onemember"


def test_equal_coords_and_check_same_grid(inputs):
    lai = inputs["lai"]
    assert be.equal_coords(lai, lai + 1, ("lat", "lon"))
    assert be.equal_coords(lai, lai.assign_coords(lat=lai.lat + 1e-4), ("lat", "lon"))
    shifted = lai.assign_coords(lat=lai.lat + 0.1)
    assert not be.equal_coords(lai, shifted, ("lat", "lon"))
    assert not be.equal_coords(lai, lai, ("time",))
    with pytest.raises(ValueError, match="differ"):
        be.check_same_grid(shifted, lai, "shifted")
    with pytest.raises(ValueError, match="grid shape"):
        be.check_same_grid(lai.isel(lat=slice(1, None)), lai, "cropped")


def test_convert_units():
    da = xr.DataArray([1.0, 2.0])
    np.testing.assert_allclose(be.convert_units("pr", da), da * be.LATENT_HEAT_VAPORIZATION)
    np.testing.assert_allclose(
        be.convert_units("PRECT_month_1", da), da * be.LATENT_HEAT_VAPORIZATION * be.LIQ_WATER_DENSITY
    )
    np.testing.assert_allclose(be.convert_units("mer", da), -da * be.LATENT_HEAT_VAPORIZATION)
    assert be.convert_units("et", da).attrs["units"] == "W/m2"
    xr.testing.assert_identical(be.convert_units("lai", da), da)


@pytest.mark.parametrize(
    "units, factor",
    [
        ("W m-2", 1.0),
        ("W/m^2", 1.0),
        ("Watt m-2", 1.0),
        ("watt/m2", 1.0),
        ("kg/m2/s", be.LATENT_HEAT_VAPORIZATION),
        ("kg m-2 s-1", be.LATENT_HEAT_VAPORIZATION),
        ("MJ m-2 day-1", 1e6 / 86400),
    ],
)
def test_latent_heat_to_wm2(units, factor):
    da = xr.DataArray([1.0, 2.0], attrs={"units": units, "long_name": "x"})
    out = be.latent_heat_to_wm2(da)
    np.testing.assert_allclose(out, da * factor)
    assert out.attrs == {"units": "W/m2", "long_name": "x"}
    assert da.attrs["units"] == units


def test_latent_heat_to_wm2_unknown_units():
    with pytest.raises(ValueError, match="mm d-1"):
        be.latent_heat_to_wm2(xr.DataArray([1.0], attrs={"units": "mm d-1"}))


def test_mask_greenland():
    lat = np.array([-10.0, 72.0])
    lon = np.array([-60.0, -40.0])  # Amazon, Greenland
    lf = xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    mask = be.mask_greenland(lf, 0.5)
    assert bool(mask.sel(lat=-10, lon=-60))
    assert not bool(mask.sel(lat=72, lon=-40))
    assert not bool(be.mask_greenland(lf * 0.2, 0.5).sel(lat=-10, lon=-60))


def test_filter_all_variables_available(rng):
    a = monthly(rng)
    data = {
        "good": {"x": a, "y": a},
        "missing": {"x": a},
        "badgrid": {"x": a, "y": a.isel(lat=slice(1, None))},
    }
    out = be.filter_all_variables_available(data, ["x", "y"], coords=("lat", "lon"), verbose=False)
    assert list(out) == ["good"]


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------

def test_compute_annual_mean_is_days_weighted():
    da = xr.DataArray(TIME.days_in_month.astype(float), dims="time", coords={"time": TIME})
    ann = be.compute_annual_mean(da)
    d = TIME.days_in_month[:12].to_numpy(dtype=float)  # 2000 (leap year)
    assert ann.dims == ("year",)
    np.testing.assert_allclose(ann.sel(year=2000), np.sum(d * d) / np.sum(d))


def test_compute_annual_mean_constant(rng):
    da = xr.full_like(monthly(rng), 3.0)
    np.testing.assert_allclose(be.compute_annual_mean(da), 3.0)


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
    assert be.aggregate(monthly(rng), how).dims == dims


def test_aggregate_values():
    # Seasonal cycle 0..11 every year, plus 12 * year index
    vals = np.tile(np.arange(12.0), 3) + np.repeat([0.0, 12.0, 24.0], 12)
    da = xr.DataArray(vals, dims="time", coords={"time": TIME})
    np.testing.assert_allclose(be.aggregate(da, "year_max"), [11, 23, 35])
    np.testing.assert_allclose(be.aggregate(da, "clim_max"), 11 + 12)  # max of mean seasonal cycle
    np.testing.assert_allclose(be.aggregate(da, "clim"), be.compute_annual_mean(da).mean())


def test_aggregate_preaggregated(rng):
    clim = be.aggregate(monthly(rng), "clim")
    xr.testing.assert_identical(be.aggregate(clim, "clim"), clim)
    ann = be.aggregate(monthly(rng), "year")
    xr.testing.assert_allclose(be.aggregate(ann, "clim"), ann.mean("year"))
    with pytest.raises(ValueError, match="time"):
        be.aggregate(clim, "year_max")
    with pytest.raises(ValueError, match="Unknown"):
        be.aggregate(monthly(rng), "decadal")


def test_net_radiation_sign_conventions():
    one = xr.DataArray(1.0)
    assert float(be.net_radiation_cmip(one * 300, one * 50, one * 350, one * 400)) == 200
    assert float(be.net_radiation_cesm(one * 250, one * 50)) == 200  # FLNS is net LW up
    assert float(be.net_radiation_era5(one * 250, one * -50)) == 200  # both positive down


def test_compute_aridity_index():
    p = xr.DataArray([100.0, 100.0])
    rn = xr.DataArray([150.0, -20.0])
    np.testing.assert_allclose(be.compute_aridity_index(p, rn), [1.5, -0.2])
    np.testing.assert_allclose(be.compute_aridity_index(p, rn, clip=True), [1.5, 0.0])
    assert be.compute_aridity_index(p, rn).name == "ai"


def test_prepare_inputs(rng, mask):
    et, lai, pr, rn = (monthly(rng, s) for s in (20, 1, 40, 30))
    out = be.prepare_inputs(et, lai, pr, rn, mask=mask, time_slice=slice("2000-01", "2001-12"))
    assert out["et"].dims == ("year", "lat", "lon")
    assert list(out["et"].year.values) == [2000, 2001]
    assert out["lai"].dims == ("lat", "lon")
    assert out["ai"].dims == ("lat", "lon")
    for da in out.values():
        assert da.where(~mask).isnull().all()
        assert da.where(mask).notnull().sum() > 0
    expected_ai = be.aggregate(rn.sel(time=slice("2000-01", "2001-12")), "clim") / be.aggregate(
        pr.sel(time=slice("2000-01", "2001-12")), "clim"
    )
    xr.testing.assert_allclose(out["ai"], expected_ai.where(mask).rename("ai"), check_dim_order=True)

    out_ann = be.prepare_inputs(et, lai, pr, rn, lai_agg="year")
    assert out_ann["lai"].dims == ("year", "lat", "lon")


# ------------------------------------------------------------------
# Bin edges
# ------------------------------------------------------------------

def test_build_edges_quantile_ignores_nan(rng):
    vals = rng.random(1000)
    with_nan = np.append(vals, [np.nan, np.inf])
    edges = be.build_edges(with_nan, 4)
    np.testing.assert_allclose(edges, np.quantile(vals, [0, 0.25, 0.5, 0.75, 1]))


def test_build_edges_linear():
    np.testing.assert_allclose(be.build_edges(np.array([0.3, 0.7]), 4, "linear", (0, 2)), [0, 0.5, 1, 1.5, 2])
    np.testing.assert_allclose(be.build_edges(np.array([1.0, 3.0]), 2, "linear"), [1, 2, 3])


def test_build_edges_duplicates():
    vals = np.concatenate([np.zeros(600), np.linspace(1, 2, 400)])
    kept = be.build_edges(vals, 4)
    assert len(kept) == 5 and kept[0] == kept[1] == kept[2] == 0
    collapsed = be.build_edges(vals, 4, collapse_duplicates=True)
    assert len(collapsed) == 3 and np.all(np.diff(collapsed) > 0)


def test_build_edges_errors():
    with pytest.raises(ValueError, match="no finite"):
        be.build_edges(np.array([np.nan, np.nan]), 3)
    with pytest.raises(ValueError, match="collapsed"):
        be.build_edges(np.zeros(10), 3, collapse_duplicates=True)
    with pytest.raises(ValueError, match="Unknown bin strategy"):
        be.build_edges(np.arange(10.0), 3, strategy="log")


def test_pooled_bin_edges_pools_all_fields(rng):
    a = xr.DataArray(rng.random((3, 4)), dims=("lat", "lon"))
    b = xr.DataArray(rng.random((2, 5, 6)) + 1, dims=("member", "lat", "lon"))
    b[0, 0, 0] = np.nan
    edges = be.pooled_bin_edges({"a": a, "b": b}, 5, name="lai", attrs={"units": "m2/m2"}, verbose=False)
    pooled = np.concatenate([a.values.ravel(), b.values.ravel()])
    np.testing.assert_allclose(edges.values, be.build_edges(pooled, 5))
    assert edges.dims == ("edge",) and edges.name == "lai"
    assert edges.attrs["strategy"] == "quantile"
    assert edges.attrs["n_bins"] == 5
    assert edges.attrs["pooled_sources"] == ["a", "b"]
    assert edges.attrs["units"] == "m2/m2"


def test_pooled_bin_edges_warns_on_zeros():
    da = xr.DataArray(np.concatenate([np.zeros(50), np.arange(1.0, 51)]))
    with pytest.warns(UserWarning, match="are 0"):
        be.pooled_bin_edges(da, 3, name="lai", verbose=False)


# ------------------------------------------------------------------
# Binning core
# ------------------------------------------------------------------

def test_bin_stats_flat_matches_brute_force(rng):
    n = 5000
    t = rng.normal(50, 10, n)
    y = rng.gamma(2, 1, n)
    x = rng.random(n) * 3
    t[::17] = np.nan
    y[::23] = np.nan
    y_edges = np.quantile(y[np.isfinite(y)], np.linspace(0, 1, 6))
    x_edges = np.linspace(0, 3, 5)
    result = be._bin_stats_flat(t, y, x, y_edges, x_edges)
    assert result.shape == (len(be.STATS), 5, 4)
    np.testing.assert_allclose(result, brute_force_stats(t, y, x, y_edges, x_edges), equal_nan=True)


def test_bin_stats_flat_spike_goes_to_last_duplicate_bin():
    y_edges = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    x_edges = np.array([0.0, 1.0])
    y = np.array([0.0, 0.0, 0.0, 0.5, 1.5])
    result = be._bin_stats_flat(np.ones(5), y, np.full(5, 0.5), y_edges, x_edges)
    np.testing.assert_array_equal(result[be.STATS.index("count"), :, 0], [0, 0, 4, 1])


def test_bin_stats_flat_clips_out_of_range_values():
    edges = np.array([0.0, 1.0, 2.0])
    result = be._bin_stats_flat(np.ones(2), np.array([-5.0, 99.0]), np.array([0.5, 0.5]), edges, edges)
    np.testing.assert_array_equal(result[be.STATS.index("count"), :, 0], [1, 1])


def test_bin_stats_flat_count_pos_and_empty_bins():
    edges = np.array([0.0, 1.0, 2.0])
    t = np.array([-1.0, 2.0, 3.0])
    y = np.array([0.5, 0.5, 0.5])
    result = be._bin_stats_flat(t, y, y, edges, edges)
    stats = dict(zip(be.STATS, result[:, 0, 0]))
    assert stats["count"] == 3 and stats["count_pos"] == 2
    np.testing.assert_allclose(stats["mean"], 4 / 3)
    np.testing.assert_allclose(stats["var_samp"], np.var(t, ddof=1))
    assert np.isnan(result[0, 1, 1]) and result[3, 1, 1] == 0


# ------------------------------------------------------------------
# bin_stats
# ------------------------------------------------------------------

def test_bin_stats_broadcasts_climatology_over_years(inputs, edges):
    y_edges, x_edges = edges
    bs = be.bin_stats(inputs["et"], inputs["lai"], inputs["ai"], y_edges, x_edges)
    assert bs.dims == ("stats", "y_bin", "x_bin")
    assert bs.sizes == {"stats": len(be.STATS), "y_bin": 5, "x_bin": 4}

    et = inputs["et"]
    lai_b = inputs["lai"].expand_dims(year=et.year)
    ai_b = inputs["ai"].expand_dims(year=et.year)
    expected = brute_force_stats(
        et.values.ravel(), lai_b.values.ravel(), ai_b.values.ravel(), y_edges.values, x_edges.values
    )
    np.testing.assert_allclose(bs.values, expected, equal_nan=True)
    assert bs.sel(stats="count").sum() == np.isfinite(et).sum()


def test_bin_stats_coords_and_attrs(inputs, edges):
    y_edges, x_edges = edges
    bs = be.bin_stats(
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
    bs = be.bin_stats(inputs["et"], inputs["lai"], inputs["ai"], np.linspace(0, 5, 4), np.linspace(0, 5, 3))
    assert "y_bin_label" not in bs.coords
    assert bs.attrs["y_strategy"] == "unknown"


def test_bin_stats_mask(inputs, edges, mask):
    y_edges, x_edges = edges
    unmasked = {k: v.fillna(1.0) for k, v in inputs.items()}
    bs = be.bin_stats(unmasked["et"], unmasked["lai"], unmasked["ai"], y_edges, x_edges, mask=mask)
    expected = be.bin_stats(inputs["et"], inputs["lai"], inputs["ai"], y_edges, x_edges)
    np.testing.assert_allclose(bs.values, expected.values, equal_nan=True)


def test_bin_stats_per_member(rng, mask, edges):
    y_edges, x_edges = edges
    inp = be.prepare_inputs(
        monthly(rng, 20, members=3), monthly(rng, 1, members=3),
        monthly(rng, 40, members=3), monthly(rng, 30, members=3), mask=mask,
    )
    bs = be.bin_stats(inp["et"], inp["lai"], inp["ai"], y_edges, x_edges, member_dim="member")
    assert bs.dims == ("member", "stats", "y_bin", "x_bin")
    assert list(bs.member_id.values) == ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]
    for m in range(3):
        single = be.bin_stats(
            inp["et"].sel(member=m), inp["lai"].sel(member=m), inp["ai"].sel(member=m), y_edges, x_edges
        )
        np.testing.assert_allclose(bs.sel(member=m).values, single.values, equal_nan=True)

    pooled = be.bin_stats(inp["et"], inp["lai"], inp["ai"], y_edges, x_edges)
    assert pooled.dims == ("stats", "y_bin", "x_bin")
    np.testing.assert_allclose(pooled.sel(stats="count"), bs.sel(stats="count").sum("member"))


def test_bin_stats_rejects_mismatched_grids(inputs, edges):
    y_edges, x_edges = edges
    shifted = inputs["lai"].assign_coords(lat=inputs["lai"].lat + 0.01)
    with pytest.raises(ValueError, match="identical coordinates"):
        be.bin_stats(inputs["et"], shifted, inputs["ai"], y_edges, x_edges)


def test_bin_stats_by_source(inputs, edges):
    y_edges, x_edges = edges
    other = {k: v * 2 for k, v in inputs.items()}
    bs = be.bin_stats_by_source({"A": inputs, "B": other}, y_edges, x_edges, dim="sid", verbose=False)
    assert bs.dims == ("sid", "stats", "y_bin", "x_bin")
    assert list(bs.sid.values) == ["A", "B"]
    single = be.bin_stats(inputs["et"], inputs["lai"], inputs["ai"], y_edges, x_edges)
    np.testing.assert_allclose(bs.sel(sid="A").values, single.values, equal_nan=True)


def test_open_bin_stats_roundtrip(tmp_path, inputs, edges):
    bs = be.bin_stats(inputs["et"], inputs["lai"], inputs["ai"], *edges, name="et")
    bs.to_netcdf(tmp_path / "bs.nc")
    opened = be.open_bin_stats(tmp_path / "bs.nc")
    xr.testing.assert_allclose(opened, bs)


def test_open_bin_stats_legacy_file(tmp_path):
    # Older notebook output: no y_bin/x_bin coords, edges only in attrs
    legacy = xr.DataArray(
        np.ones((4, 3, 2)),
        name="evspsbl",
        dims=("stats", "y_bin", "x_bin"),
        coords={"stats": ["mean", "var_pop", "var_samp", "count"]},
        attrs={"y_bin_edges": [0.0, 0.0, 1.0, 2.0], "x_bin_edges": [0.0, 0.5, 1.0]},
    )
    legacy.to_netcdf(tmp_path / "legacy.nc")
    opened = be.open_bin_stats(tmp_path / "legacy.nc")
    np.testing.assert_array_equal(opened.y_bin, [0, 1, 2])
    np.testing.assert_allclose(opened.y_bin_lower, [0, 0, 1])
    np.testing.assert_allclose(opened.x_bin_upper, [0.5, 1.0])


def test_open_bin_stats_multiple_vars_needs_var(tmp_path):
    xr.Dataset({"a": ("x", [1.0]), "b": ("x", [2.0])}).to_netcdf(tmp_path / "two.nc")
    with pytest.raises(ValueError, match="pass `var`"):
        be.open_bin_stats(tmp_path / "two.nc")
    assert be.open_bin_stats(tmp_path / "two.nc", var="b").name == "b"


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
    return be._ensure_bin_coords(xr.DataArray(
        data,
        dims=(dim, "stats", "y_bin", "x_bin"),
        coords={"stats": list(be.STATS)},
        attrs={"y_bin_edges": y_edges, "x_bin_edges": x_edges, "units": "W/m2"},
    ))


def test_drop_zero_width_bins():
    bs = make_bs(np.ones((2, 4, 3)), 10, y_edges=[0, 0, 0, 1, 2], x_edges=[0, 1, 1, 2])
    dropped = be.drop_zero_width_bins(bs)
    np.testing.assert_array_equal(dropped.y_bin, [2, 3])
    np.testing.assert_array_equal(dropped.x_bin, [0, 2])
    np.testing.assert_allclose(dropped.y_bin_lower, [0, 1])


def test_frac_count():
    bs = make_bs(np.ones((2, 2, 2)), [[1, 3], [2, 4]])
    fc = be.frac_count(bs)
    np.testing.assert_allclose(fc.sum(["y_bin", "x_bin"]), 1.0)
    np.testing.assert_allclose(fc.isel(sid=0).values, [[0.1, 0.3], [0.2, 0.4]])


def test_frac_valid():
    means = np.ones((4, 1, 2))
    means[:3, 0, 1] = np.nan
    np.testing.assert_allclose(be.frac_valid(make_bs(means, 10), "sid").values, [[1.0, 0.25]])


def test_significance():
    # bin 0: mean far from 0 with many samples; bin 1: mean 0; bin 2: too few samples
    means = np.array([[[50.0, 0.0, 50.0]]])
    counts = np.array([[[100, 100, 5]]])
    sig = be.test_significance(make_bs(means, counts, var_samp=4.0), alpha=0.05, n_min=10)
    np.testing.assert_array_equal(sig.values, [[[True, False, False]]])


def test_ensemble_spread():
    means = np.array([[[1.0, 5.0]], [[3.0, np.nan]], [[5.0, np.nan]]])
    bs = make_bs(means, 10)
    np.testing.assert_allclose(be.ensemble_spread(bs, "sid").values[0, 0], np.std([1, 3, 5]))
    np.testing.assert_allclose(be.ensemble_spread(bs, "sid", how="var").values[0, 0], np.var([1, 3, 5]))
    limited = be.ensemble_spread(bs, "sid", min_frac=0.5)
    assert np.isfinite(limited.values[0, 0]) and np.isnan(limited.values[0, 1])
    signif = xr.DataArray([[[True, True]], [[True, True]], [[False, True]]], dims=("sid", "y_bin", "x_bin"))
    np.testing.assert_allclose(be.ensemble_spread(bs, "sid", signif=signif).values[0, 0], np.std([1, 3]))


# ------------------------------------------------------------------
# Plotting (smoke tests)
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_finish_saves_and_closes(tmp_path):
    fig = plt.figure()
    out = tmp_path / "sub" / "fig.png"
    assert be._finish(fig, out) is fig
    assert out.exists()
    assert not plt.fignum_exists(fig.number)
    fig2 = plt.figure()
    be._finish(fig2, None)
    assert plt.fignum_exists(fig2.number)


def test_plot_bin_field_hatch_and_ticks(tmp_path):
    bs = make_bs(np.arange(6.0).reshape(1, 2, 3), 10, y_edges=[0, 0.5, 1.5], x_edges=[0, 1, 2, 3])
    field = bs.sel(stats="mean").isel(sid=0)
    hatch = field > 3
    fig = be.plot_bin_field(field, hatch=hatch)
    ax = fig.axes[0]
    assert len(ax.patches) == int(hatch.sum())
    assert [t.get_text() for t in ax.get_yticklabels()] == ["0", "0.5", "1.5"]
    be.plot_bin_field(field, fout=tmp_path / "field.png")
    assert (tmp_path / "field.png").exists()


def test_plot_bin_summary_and_facets(tmp_path, rng, mask, edges):
    y_edges, x_edges = edges
    inp = be.prepare_inputs(
        monthly(rng, 20, members=2), monthly(rng, 1, members=2),
        monthly(rng, 40, members=2), monthly(rng, 30, members=2), mask=mask,
    )
    bs = be.bin_stats(inp["et"], inp["lai"], inp["ai"], y_edges, x_edges, member_dim="member")
    assert len(be.plot_bin_summary(bs, dim="member").axes) == 6  # 3 panels + 3 colorbars
    be.plot_bin_summary(bs.isel(member=0), fout=tmp_path / "summary.png")
    fg = be.plot_bin_facets(bs, "member", signif=be.test_significance(bs), fout=tmp_path / "facets.png")
    assert fg.axs.size >= 2
    assert (tmp_path / "summary.png").exists() and (tmp_path / "facets.png").exists()


def test_plot_edges_and_hist(tmp_path, inputs, edges):
    y_edges, x_edges = edges
    be.plot_edges(y_edges, fout=tmp_path / "edges.png")
    be.plot_edges_compare({"lai": y_edges, "ai": x_edges}, fout=tmp_path / "cmp.png")
    fig = be.plot_input_hist({"a": inputs["lai"], "b": inputs["lai"] * 1.1}, y_edges, log=True)
    assert fig.axes[0].get_yscale() == "log"
    assert len(fig.axes[0].get_lines()) >= 1  # edge lines
    for name in ("edges.png", "cmp.png"):
        assert (tmp_path / name).exists()


def test_map_plots(tmp_path, inputs):
    be.quick_map(inputs["lai"], tmp_path / "map.png", title="lai")
    fig = be.plot_input_maps(inputs, title="inputs")
    assert len([ax for ax in fig.axes if hasattr(ax, "projection")]) == 3
    assert (tmp_path / "map.png").exists()
