"""
Tests for etunc.units, etunc.binning, etunc.plotting, the aridity index of
etunc.edges, and the coordinate checks, land masks and area masks of
etunc.grid. Run from the project root with:

    python -m pytest tests/test_binning.py
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
import etunc.edges as bin_edges


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


def make_inputs(rng, mask, members=None):
    """Binning inputs as the prep scripts build them: annual mean ET and climatological LAI and AI, masked."""
    def clim(scale):
        return temporal.aggregate(temporal.aggregate(monthly(rng, scale, members), "year"), "clim")
    inp = {
        "et": temporal.aggregate(monthly(rng, 20, members), "year").rename("et"),
        "lai": clim(1).rename("lai"),
        "ai": bin_edges.compute_aridity_index(clim(40), clim(30)),
    }
    return {k: v.where(mask) for k, v in inp.items()}


def quantile_edges(da, n_bins, name):
    """Equally weighted quantile edges of all finite values of `da`."""
    values = da.values[np.isfinite(da.values)]
    return bin_edges.pooled_bin_edges(values, np.ones(values.size), n_bins, name=name)


@pytest.fixture
def inputs(rng, mask):
    return make_inputs(rng, mask)


@pytest.fixture
def edges(inputs):
    return quantile_edges(inputs["lai"], 5, "lai"), quantile_edges(inputs["ai"], 4, "ai")


def brute_force_stats(target, y, x, y_edges, x_edges, weights=None):
    """Reference implementation: loop over bins, same assignment rule as the module."""
    n_y, n_x = len(y_edges) - 1, len(x_edges) - 1
    weights = np.ones_like(target) if weights is None else weights
    valid = np.isfinite(target) & np.isfinite(y) & np.isfinite(x) & np.isfinite(weights)
    t, y, x, w = target[valid], y[valid], x[valid], weights[valid]
    yi = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, n_y - 1)
    xi = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, n_x - 1)
    out = np.full((len(binning.STATS), n_y, n_x), np.nan)
    for j in range(n_y):
        for i in range(n_x):
            in_bin = (yi == j) & (xi == i)
            v, wb = t[in_bin], w[in_bin]
            out[3, j, i] = v.size
            out[4, j, i] = np.sum(v > 0)
            out[5, j, i] = wb.sum()
            out[6, j, i] = wb.sum() ** 2 / np.sum(wb**2) if v.size else 0  # Kish effective n
            if v.size:
                out[0, j, i] = np.average(v, weights=wb)
                out[1, j, i] = np.average((v - out[0, j, i]) ** 2, weights=wb)
            if out[6, j, i] > 1:
                out[2, j, i] = out[1, j, i] * out[6, j, i] / (out[6, j, i] - 1)
    return out


# ------------------------------------------------------------------
# General helpers
# ------------------------------------------------------------------

def test_format_time_period():
    """A time slice is formatted as YYYYMM-YYYYMM."""
    assert temporal.format_time_period(slice("1950-01", "2014-12")) == "195001-201412"


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
    convert_units scales pr/et and CLM RAIN/SNOW (mm/s) by L and CESM PRECT (m/s) by L times water density,
    flips the sign of ERA5 mer, and leaves other variables unchanged.
    """
    da = xr.DataArray([1.0, 2.0])
    np.testing.assert_allclose(units.convert_units("pr", da), da * config.LATENT_HEAT_VAPORIZATION)
    np.testing.assert_allclose(
        units.convert_units("PRECT_month_1", da), da * config.LATENT_HEAT_VAPORIZATION * config.LIQ_WATER_DENSITY
    )
    np.testing.assert_allclose(units.convert_units("mer", da), -da * config.LATENT_HEAT_VAPORIZATION)
    np.testing.assert_allclose(units.convert_units("RAIN", da), da * config.LATENT_HEAT_VAPORIZATION)
    assert units.convert_units("SNOW", da).attrs["units"] == "W/m2"
    assert units.convert_units("et", da).attrs["units"] == "W/m2"
    assert units.convert_units("mer", da).attrs["units"] == "W/m2"  # kept through the sign flip
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


def test_landfrac_and_greenland_masks():
    """
    landfrac_mask is True where land fraction > threshold, except in Greenland (greenland_mask False there);
    land at or below the threshold, and NaN land fraction, is False.
    """
    lat = np.array([-10.0, 72.0])
    lon = np.array([-60.0, -40.0])  # Amazon, Greenland
    lf = xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    assert not bool(rg.greenland_mask(lf).sel(lat=72, lon=-40))
    assert bool(rg.greenland_mask(lf).sel(lat=-10, lon=-60))
    mask = rg.landfrac_mask(lf, 0.5)
    assert mask.dtype == bool
    assert bool(mask.sel(lat=-10, lon=-60))
    assert not bool(mask.sel(lat=72, lon=-40))
    assert not bool(rg.landfrac_mask(lf * 0.5, 0.5).sel(lat=-10, lon=-60))
    assert not bool(rg.landfrac_mask(lf * np.nan, 0.5).sel(lat=-10, lon=-60))


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------

def test_net_radiation_sign_conventions():
    """CMIP, CESM, CLM and ERA5 net radiation each follow their source's flux sign convention."""
    one = xr.DataArray(1.0)
    assert float(units.net_radiation_cmip(one * 300, one * 50, one * 350, one * 400)) == 200
    assert float(units.net_radiation_cesm(one * 250, one * 50)) == 200  # FLNS is net LW up
    assert float(units.net_radiation_clm(one * 250, one * 50)) == 200  # FIRA is net LW up
    assert float(units.net_radiation_era5(one * 250, one * -50)) == 200  # both positive down


def test_compute_aridity_index():
    """AI = Rn / P, named "ai"; clip=True sets negative values to 0."""
    p = xr.DataArray([100.0, 100.0])
    rn = xr.DataArray([150.0, -20.0])
    np.testing.assert_allclose(bin_edges.compute_aridity_index(p, rn), [1.5, -0.2])
    np.testing.assert_allclose(bin_edges.compute_aridity_index(p, rn, clip=True), [1.5, 0.0])
    assert bin_edges.compute_aridity_index(p, rn).name == "ai"


def test_drop_et_outliers(capsys):
    """
    Gridcell-years below the threshold become NaN, attrs are kept, and the count is printed with the number
    inside LAT_BNDS (here 0: the outlier is at -70, in Antarctica).
    """
    et = xr.DataArray([[5.0, -3.0], [-1.0, 0.5]], dims=("year", "lat"), coords={"lat": [10.0, -70.0]},
                      name="et", attrs={"units": "W/m2"})
    out = binning.drop_et_outliers(et, -1.0)
    np.testing.assert_array_equal(out.isnull(), [[False, True], [False, False]])  # -1 itself is kept
    assert out.attrs["units"] == "W/m2"
    assert "et: 1 gridcell-years with annual ET < -1.0 W/m2 set to NaN (0 within LAT_BNDS" in capsys.readouterr().out


def test_evaporation_ratio(capsys):
    """ET/P is named "et_pr", dimensionless, NaN where P <= 0 (no inf), and the dropped count is printed."""
    coords = {"year": [2000, 2001], "lat": [10.0, 20.0]}
    et = xr.DataArray([[50.0, 20.0], [40.0, np.nan]], dims=("year", "lat"), coords=coords, name="et",
                      attrs={"units": "W/m2"})
    pr = xr.DataArray([[100.0, 0.0], [-1.0, 10.0]], dims=("year", "lat"), coords=coords, name="pr",
                      attrs={"units": "W/m2"})
    out = binning.evaporation_ratio(et, pr, label="t")
    np.testing.assert_array_equal(out.values, [[0.5, np.nan], [np.nan, np.nan]])
    assert out.name == "et_pr" and out.attrs == {"long_name": "evaporation ratio ET/P of annual means", "units": "1"}
    # The two gridcell-years with valid ET and P <= 0; the NaN ET does not count
    assert "t: 2 gridcell-years with valid ET and P <= 0 set to NaN" in capsys.readouterr().out
    # Climatological P broadcasts over years
    np.testing.assert_allclose(binning.evaporation_ratio(et, pr.isel(year=0, drop=True).clip(min=1)).isel(year=0), [0.5, 20.0])


def test_evaporation_ratio_rejects_mismatched_grids():
    """ET and P whose coords differ raise instead of being aligned."""
    et = xr.DataArray([1.0, 2.0], dims="lat", coords={"lat": [10.0, 20.0]})
    pr = et.assign_coords(lat=et.lat + 0.01)
    with pytest.raises(ValueError):
        binning.evaporation_ratio(et, pr)


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
    # Unweighted: sum_w and n_eff equal the count
    assert stats["sum_w"] == 3 and stats["n_eff"] == 3


def test_bin_stats_flat_weighted_matches_brute_force(rng):
    """With weights, the mean and variances are weighted and n_eff is Kish's; counts stay raw counts."""
    n = 2000
    t = rng.normal(size=n)
    y = rng.gamma(2, 1, n)
    x = rng.random(n) * 3
    w = rng.uniform(0.2, 3.0, n)
    t[::17] = np.nan
    w[::29] = np.nan  # a NaN weight drops the sample
    y_edges = np.quantile(y, np.linspace(0, 1, 6))
    x_edges = np.linspace(0, 3, 5)
    result = binning._bin_stats_flat(t, y, x, y_edges, x_edges, w)
    np.testing.assert_allclose(result, brute_force_stats(t, y, x, y_edges, x_edges, w), equal_nan=True)


def test_bin_stats_flat_unit_weights_are_unweighted(rng):
    """Unit weights give exactly the unweighted statistics, with sum_w = n_eff = count."""
    n = 500
    t, y, x = rng.normal(size=n), rng.random(n), rng.random(n)
    edges = np.linspace(0, 1, 4)
    unweighted = binning._bin_stats_flat(t, y, x, edges, edges)
    np.testing.assert_array_equal(binning._bin_stats_flat(t, y, x, edges, edges, np.ones(n)), unweighted)
    count = unweighted[binning.STATS.index("count")]
    np.testing.assert_array_equal(unweighted[binning.STATS.index("sum_w")], count)
    np.testing.assert_array_equal(unweighted[binning.STATS.index("n_eff")], count)


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
    inp = make_inputs(rng, mask, members=3)
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
    inp = make_inputs(rng, mask, members=3)
    bs = binning.bin_stats(inp["et"], inp["lai"], inp["ai"], y_edges, x_edges, member_dim="member", name="et")
    pooled = binning.pool_members(bs)
    direct = binning.bin_stats(inp["et"], inp["lai"], inp["ai"], y_edges, x_edges, name="et")
    assert pooled.dims == ("stats", "y_bin", "x_bin") and pooled.name == "et"
    assert list(pooled.stats.values) == list(binning.STATS)
    assert pooled.attrs["members"] == ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]
    assert "member_id" not in pooled.coords
    np.testing.assert_allclose(pooled.values, direct.values, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_bin_stats_land_area_weights(rng, mask, edges):
    """
    bin_all.py with AREA_WEIGHTED: a (lat, lon) land area weight is repeated over year and member, so each
    (year, gridcell) sample carries its gridcell's area, matching a weighted brute force per member.
    """
    y_edges, x_edges = edges
    inp = make_inputs(rng, mask, members=2)
    area = xr.DataArray(rg.cell_area(LAT, LON) / 1e6, dims=("lat", "lon"), coords={"lat": LAT, "lon": LON},
                        name="land_area")
    bs = binning.bin_stats(inp["et"], inp["lai"], inp["ai"], y_edges, x_edges, weights=area, mask=mask,
                           member_dim="member")
    assert bs.attrs["weights"] == "land_area"
    for m in range(2):
        et = inp["et"][m]
        lai, ai, w = (da.broadcast_like(et).values.ravel() for da in (inp["lai"][m], inp["ai"][m], area))
        expected = brute_force_stats(et.values.ravel(), lai, ai, y_edges.values, x_edges.values, w)
        np.testing.assert_allclose(bs.isel(member=m).values, expected, equal_nan=True)
    # Each gridcell's area is counted once per valid year
    n_years = inp["et"].sizes["year"]
    expected_w = float(area.where(mask).sum()) * n_years
    np.testing.assert_allclose(bs.sel(stats="sum_w").sum(("y_bin", "x_bin")), [expected_w] * 2)


def test_pool_members_weighted(rng, mask, edges):
    """Pooling weighted per-member statistics gives the weighted statistics of all members' samples together."""
    y_edges, x_edges = edges
    inp = make_inputs(rng, mask, members=3)
    area = xr.DataArray(rg.cell_area(LAT, LON) / 1e6, dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})
    kwargs = {"weights": area, "name": "et"}
    bs = binning.bin_stats(inp["et"], inp["lai"], inp["ai"], y_edges, x_edges, member_dim="member", **kwargs)
    direct = binning.bin_stats(inp["et"], inp["lai"], inp["ai"], y_edges, x_edges, **kwargs)
    np.testing.assert_allclose(binning.pool_members(bs).values, direct.values, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_bin_stats_climatological_et(rng, mask, edges):
    """
    bin_all_clim.py: the ET climatology (NaN years skipped) is one sample per (member, gridcell), binned by
    the gridcell's climatological LAI and AI, matching a brute-force reference.
    """
    y_edges, x_edges = edges
    inp = make_inputs(rng, mask, members=2)
    et = inp["et"].copy()
    j, i = np.argwhere(mask.values)[0]
    et[0, 0, j, i] = np.nan  # a missing year: the climatology covers the other years
    et_clim = temporal.aggregate(et, "clim")
    assert et_clim.dims == ("member", "lat", "lon")
    assert float(et_clim[0, j, i]) == pytest.approx(float(et[0, 1:, j, i].mean()))

    bs = binning.bin_stats(et_clim, inp["lai"], inp["ai"], y_edges, x_edges, member_dim="member")
    for m in range(2):
        expected = brute_force_stats(
            et_clim[m].values.ravel(), inp["lai"][m].values.ravel(), inp["ai"][m].values.ravel(),
            y_edges.values, x_edges.values,
        )
        np.testing.assert_allclose(bs.isel(member=m).values, expected, equal_nan=True)
    # One sample per valid gridcell, whatever its number of valid years
    np.testing.assert_array_equal(bs.sel(stats="count").sum(("y_bin", "x_bin")), [int(mask.sum())] * 2)


def test_bin_stats_land_area(rng, mask, edges):
    """
    bin_area.py: binning a (lat, lon) area field repeated over members gives, as mean x count, the total area
    of each member's masked gridcells in each bin, and keeps the member coords.
    """
    y_edges, x_edges = edges
    inp = make_inputs(rng, mask, members=2)
    area = xr.DataArray(rg.cell_area(LAT, LON) / 1e6, dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})
    area_m = area * xr.ones_like(inp["lai"])
    bs = binning.bin_stats(area_m, inp["lai"], inp["ai"], y_edges, x_edges, mask=mask, member_dim="member")
    assert list(bs.member_id.values) == ["r1i1p1f1", "r2i1p1f1"]
    total = (bs.sel(stats="mean") * bs.sel(stats="count")).fillna(0)
    for m in range(2):
        # Brute-force sum of the area of each bin's gridcells
        lai, ai = inp["lai"][m].values.ravel(), inp["ai"][m].values.ravel()
        a = area.where(mask).values.ravel()
        valid = np.isfinite(a) & np.isfinite(lai) & np.isfinite(ai)
        yi = np.clip(np.searchsorted(y_edges.values, lai[valid], side="right") - 1, 0, len(y_edges) - 2)
        xi = np.clip(np.searchsorted(x_edges.values, ai[valid], side="right") - 1, 0, len(x_edges) - 2)
        expected = np.zeros(total.shape[1:])
        np.add.at(expected, (yi, xi), a[valid])
        np.testing.assert_allclose(total.isel(member=m).values, expected, rtol=1e-12)
        # Every masked gridcell has valid LAI and AI, so the total is the masked area
        assert float(total.isel(member=m).sum()) == pytest.approx(float(area.where(mask).sum()), rel=1e-12)


def test_pool_by_source(rng, mask, edges):
    """
    pool_by_source pools the members of each source (as pool_members) along `source`, keeping the member
    coords that are the same within every source.
    """
    y_edges, x_edges = edges
    inp = make_inputs(rng, mask, members=3)
    member_coords = {
        "source": ("member", ["a", "a", "b"]),
        "product": ("member", ["p", "p", "q"]),       # one value per source: kept
        "member_id": ("member", ["r1", "r2", "r1"]),  # varies within source "a": dropped
    }
    et = inp["et"].assign_coords(member_coords)
    bs = binning.bin_stats(et, inp["lai"], inp["ai"], y_edges, x_edges, member_dim="member", name="et")
    by_source = binning.pool_by_source(bs)
    assert by_source.dims == ("source", "stats", "y_bin", "x_bin")
    assert list(by_source.source.values) == ["a", "b"]
    assert list(by_source["product"].values) == ["p", "q"]
    assert "member_id" not in by_source.coords
    np.testing.assert_allclose(by_source.sel(source="a").values, binning.pool_members(bs.isel(member=[0, 1])).values,
                               equal_nan=True)
    np.testing.assert_allclose(by_source.sel(source="b").values, binning.pool_members(bs.isel(member=[2])).values,
                               equal_nan=True)


def test_et_product_spread(tmp_path):
    """
    et_product_spread is the std (ddof=1) of the bin mean across ET products within each (lai, pr, rn)
    product set, NaN with fewer than two ET products; plot_et_product_spread draws one panel per set.
    """
    means = np.array([1.0, 3.0, 2.0, 5.0]).reshape(4, 1, 1) * np.ones((4, 2, 2))
    means[1, 0, 0] = np.nan  # one ET product missing in bin (0, 0) of the first product set
    stats = np.full((4, len(binning.STATS), 2, 2), np.nan)
    stats[:, 0] = means
    bs = xr.DataArray(
        stats, dims=("source", "stats", "y_bin", "x_bin"),
        coords={
            "source": ["s1", "s2", "s3", "s4"], "stats": list(binning.STATS),
            "et_product": ("source", ["E1", "E2", "E1", "E2"]),
            "lai_product": ("source", ["L1", "L1", "L2", "L2"]),
            "pr_product": ("source", ["P"] * 4), "rn_product": ("source", ["R"] * 4),
        },
        attrs={"units": "W/m2", "y_variable": "lai", "x_variable": "ai"},
    )
    spread = binning.et_product_spread(bs)
    assert list(spread.lai_pr_rn.values) == ["L1-P-R", "L2-P-R"]
    np.testing.assert_array_equal(spread.n_et_products, [2, 2])
    first, second = spread.sel(lai_pr_rn="L1-P-R"), spread.sel(lai_pr_rn="L2-P-R")
    assert np.isnan(first["et_std"][0, 0]) and int(first["n_products"][0, 0]) == 1
    assert float(first["et_std"][1, 1]) == pytest.approx(np.std([1.0, 3.0], ddof=1))
    np.testing.assert_allclose(second["et_std"], np.std([2.0, 5.0], ddof=1))

    fout = tmp_path / "spread.png"
    plotting.plot_et_product_spread(spread, title="t", fout=fout)
    assert fout.exists()


def test_bin_stats_rejects_mismatched_grids(inputs, edges):
    """Inputs whose lat/lon values differ raise instead of being aligned."""
    y_edges, x_edges = edges
    shifted = inputs["lai"].assign_coords(lat=inputs["lai"].lat + 0.01)
    with pytest.raises(ValueError, match="identical coordinates"):
        binning.bin_stats(inputs["et"], shifted, inputs["ai"], y_edges, x_edges)


def test_valid_and_common_area(capsys):
    """
    valid_area needs ET in some year plus LAI and a finite AI, in every member; common_area is the cells
    valid in every dataset, and prints each dataset's valid cells and what the common area loses.
    """
    coords = {"lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]}
    et = xr.DataArray(np.ones((2, 2, 2, 3)), dims=("member", "year", "lat", "lon"), coords=coords)
    lai = xr.DataArray(np.ones((2, 2, 3)), dims=("member", "lat", "lon"), coords=coords)
    ai = lai.copy()
    et[:, :, 0, 0] = np.nan                 # no ET in any year
    et[0, 0, 0, 1] = np.nan                 # one missing year only: still valid
    lai[1, 0, 2] = np.nan                   # LAI missing in member 1 only
    ai[:, 1, 0] = np.inf                    # AI not finite
    inp = xr.Dataset({"et": et, "lai": lai, "ai": ai})

    all_members = rg.valid_area(inp)
    assert all_members.dims == ("lat", "lon")
    np.testing.assert_array_equal(all_members, [[False, True, False], [False, True, True]])
    np.testing.assert_array_equal(rg.valid_area(inp.isel(member=0)), [[False, True, True], [False, True, True]])

    other = rg.valid_area(inp.assign(lai=inp["lai"].where(inp.lat == 0)))
    area = rg.common_area({"A": all_members, "B": other})
    assert area.name == "area_mask"
    np.testing.assert_array_equal(area, [[False, True, False], [False, False, False]])
    out = capsys.readouterr().out
    assert "A       :      3 valid gridcells" in out and "loses      2 gridcells" in out
    assert "common  :      1 gridcells" in out


def test_output_layout(tmp_path, monkeypatch):
    """Outputs go to <root>/<family>/<kind>/<period>[/<bins>]; file names keep the grid, period and bins."""
    monkeypatch.setattr(config, "PROC_ROOT", tmp_path)
    period = "199501-201412"
    assert config.bins_tag(15, 20) == "lai15_ai20"
    assert config.output_dir(tmp_path, "obs", "clim", period) == tmp_path / "obs" / "clim" / period
    assert config.output_dir(tmp_path, "obs", "qbin", period, "lai15_ai20") == tmp_path / "obs/qbin" / period / "lai15_ai20"
    assert config.inputs_path("obs", "1deg", period) == tmp_path / "obs/inputs" / period / f"obs.inputs.1deg.{period}.nc"
    assert config.area_mask_path("1deg", period) == tmp_path / "all/mask" / period / f"all.area_mask.1deg.{period}.nc"


# ------------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------------

def make_bs(means, counts, var_samp=None, y_edges=None, x_edges=None, dim="sid"):
    """Build a (dim, stats, y_bin, x_bin) bin-stats array from (dim, y, x) means and counts."""
    means = np.asarray(means, dtype=float)
    counts = np.broadcast_to(np.asarray(counts, dtype=float), means.shape)
    var_samp = np.ones_like(means) if var_samp is None else np.broadcast_to(var_samp, means.shape)
    # Unweighted: sum_w = n_eff = count
    data = np.stack([means, var_samp, var_samp, counts, counts, counts, counts], axis=1)
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
    inp = make_inputs(rng, mask, members=2)
    bs = binning.bin_stats(inp["et"], inp["lai"], inp["ai"], y_edges, x_edges, member_dim="member")
    assert len(plotting.plot_bin_summary(bs, dim="member").axes) == 6  # 3 panels + 3 colorbars
    plotting.plot_bin_summary(bs.isel(member=0), fout=tmp_path / "summary.png")
    fg = plotting.plot_bin_facets(bs, "member", signif=binning.test_significance(bs), fout=tmp_path / "facets.png")
    assert fg.axs.size >= 2
    assert (tmp_path / "summary.png").exists() and (tmp_path / "facets.png").exists()


def test_plot_edges_compare_and_quick_map(tmp_path, inputs, edges):
    """plot_edges_compare and quick_map run and save."""
    y_edges, x_edges = edges
    plotting.plot_edges_compare({"lai": y_edges, "ai": x_edges}, fout=tmp_path / "cmp.png")
    plotting.quick_map(inputs["lai"], tmp_path / "map.png", title="lai")
    for name in ("cmp.png", "map.png"):
        assert (tmp_path / name).exists()


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


def test_save_ensemble_map_three_panels(tmp_path, inputs, capsys):
    """save_ensemble_map draws mean, std and range maps over `dim`, each with a variable/units colorbar."""
    lai = inputs["lai"].assign_attrs(units="m2/m2")
    da = xr.concat([lai, lai + 1, lai + 3], dim="member")
    fout = tmp_path / "maps" / "lai.png"
    fig = plotting.save_ensemble_map(da, "lai", fout, title="LAI")
    assert fout.exists() and str(fout) in capsys.readouterr().out
    maps = [ax for ax in fig.axes if hasattr(ax, "projection")]
    assert [ax.get_title() for ax in maps] == [
        "ensemble mean", "ensemble standard deviation", "ensemble range (max - min)"
    ]
    # Range of members lai, lai + 1, lai + 3 is 3 wherever lai is valid
    rng_vals = maps[2].collections[0].get_array()
    assert np.allclose(rng_vals[~np.ma.getmaskarray(rng_vals)], 3)
    assert all(ax.get_ylabel() == "lai [m2/m2]" for ax in fig.axes if ax not in maps)
    assert fig._suptitle.get_text() == "LAI, 3 members"
    plt.close("all")


def test_plot_bin_means_titles_and_ticks(inputs, edges):
    """
    plot_bin_means titles each panel with its `dim` label (plus time_period when that is a coord along dim),
    uses edge values as ticks when all panels share edges and quantile levels otherwise, and hatches.
    """
    y_edges, x_edges = edges
    bs = binning.bin_stats(inputs["et"], inputs["lai"], inputs["ai"], y_edges, x_edges, name="et")
    shared = xr.concat([bs, bs * 2], dim="combo").assign_coords(
        combo=["a", "b"], time_period=("combo", ["200001-200212", "200101-200212"]))
    fig = plotting.plot_bin_means(shared, "combo", hatch=shared.sel(stats="count") > 0)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles[:2] == ["a\n200001-200212", "b\n200101-200212"]
    assert fig.axes[0].get_xticklabels()[0].get_text() != "Q0"
    assert len(fig.axes[0].patches) > 0
    plt.close(fig)

    own = xr.concat([bs, bs.assign_coords(y_bin_lower=bs.y_bin_lower + 1)], dim="source_id", compat="equals",
                    coords="different").assign_coords(source_id=["M1", "M2"])
    fig = plotting.plot_bin_means(own, "source_id")
    assert [ax.get_title() for ax in fig.axes if ax.get_title()][:2] == ["M1", "M2"]
    assert fig.axes[0].get_xticklabels()[0].get_text() == "Q0"
    plt.close(fig)


@pytest.mark.parametrize("highlight_common", [False, True])
def test_plot_mask_agreement_counts_common_mask(highlight_common, mask):
    """The title counts land cells in every product's mask; there is one colour per count 0..n."""
    land = mask
    masks = xr.concat([land, land, land.where(land.lat > 0, False)], dim="product").assign_coords(
        product=["a", "b", "c"])
    fig = plotting.plot_mask_agreement(masks, land, "t", highlight_common=highlight_common)
    expected = int((land & (land.lat > 0)).sum())
    assert f"common mask (all 3 products): {expected} of {int(land.sum())} land cells" in fig.axes[0].get_title()
    assert fig.axes[0].collections[0].cmap.N == 4
    plt.close(fig)


def test_plot_clim_hist_fixed_bins_and_outside_note(capsys):
    """
    One line per field (plus the weighted pooled line) on the given fixed bins; values outside the bins go
    to the end bins, and the largest fraction outside is printed and noted.
    """
    bins = np.linspace(0, 2, 5)
    fields = {"a": xr.DataArray([0.1, 0.6, 1.1, 1.6]), "b": xr.DataArray([0.1, 0.1, 5.0, np.nan])}
    values, weights = np.array([0.1, 1.6]), np.array([0.5, 0.5])
    fig = plotting.plot_clim_hist(fields, bins, pooled=(values, weights), edges=np.array([0.5]), xlabel="lai")
    ax = fig.axes[0]
    steps = [p for p in ax.patches if p.get_label() in ("a", "b", "pooled (weighted)")]
    assert [p.get_label() for p in steps] == ["a", "b", "pooled (weighted)"]
    np.testing.assert_allclose(steps[1].get_data().values, [2 / 3 / 0.5, 0, 0, 1 / 3 / 0.5])  # 5.0 in the last bin
    assert "up to 33.33% (b)" in capsys.readouterr().out
    assert ax.get_xlim() == (0.0, 2.0)
    plt.close(fig)


def test_save_clim_diagnostics_writes_maps_and_hists(tmp_path, inputs, capsys):
    """Five climatology maps and two histograms per family, named <family>.<var>.<grid>.{map,hist}.<period>.png."""
    members = [f"m{i}" for i in range(3)]
    ds = xr.concat([xr.Dataset({**inputs, "pr": inputs["lai"], "rn": inputs["lai"]})] * 3, dim="member")
    ds = ds.assign_coords(member=members, source=("member", ["s1", "s1", "s2"]))
    plotting.save_clim_diagnostics(ds, tmp_path, "fam", "1deg", "200001-200212", title="Fam")
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == sorted(
        [f"fam.{v}.1deg.map.200001-200212.png" for v in ("et", "lai", "pr", "rn", "ai")]
        + [f"fam.{v}_clim.1deg.hist.200001-200212.png" for v in ("lai", "ai")]
    )
    plotting.save_clim_diagnostics(ds.isel(member=[0]), tmp_path / "one", "fam", "1deg", "200001-200212")
    assert len(list((tmp_path / "one").iterdir())) == 7  # one member: single maps
    plt.close("all")

