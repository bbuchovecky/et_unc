"""
Tests for etunc.decomposition (state, function and interaction decomposition of
the area-mean ET), on synthetic fields binned with binning.bin_stats(weights=...). Run from the
project root with:

    python -m pytest tests/test_decomposition.py
"""

import numpy as np
import pytest
import xarray as xr

import etunc.binning as binning
import etunc.decomposition as dc

# ------------------------------------------------------------------
# Fixtures and helpers
# ------------------------------------------------------------------

LAT = np.linspace(-50, 80, 14)
LON = np.linspace(-180, 170, 20)
Y_EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0])  # LAI
X_EDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 4.0])  # AI
RTOL = 1e-10


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def field(values, members):
    return xr.DataArray(values, dims=("member", "lat", "lon"),
                        coords={"member": members, "lat": LAT, "lon": LON})


@pytest.fixture
def area(rng):
    """Land area of each gridcell, varying so that area weighting matters."""
    return xr.DataArray(rng.uniform(100, 2000, (LAT.size, LON.size)), dims=("lat", "lon"),
                        coords={"lat": LAT, "lon": LON}, name="land_area")


def make_fields(rng, members, lai_scale=None, et_scale=None, ai=None):
    """LAI, AI and climatological ET (member, lat, lon), ET increasing with LAI and decreasing with AI."""
    shape = (len(members), LAT.size, LON.size)
    lai = rng.uniform(0, 5.5, shape) if lai_scale is None else np.asarray(lai_scale)[:, None, None] * rng.uniform(0, 5, shape[1:])
    if ai is None:
        ai = rng.uniform(0, 3.5, shape)
    et = 20 + 10 * lai - 8 * ai + rng.normal(0, 3, shape)
    if et_scale is not None:
        et = np.asarray(et_scale)[:, None, None] * et
    return field(et, members), field(lai, members), field(np.broadcast_to(ai, shape).copy(), members)


def binned(et, lai, ai, area, sources=None):
    bs = binning.bin_stats(et, lai, ai, Y_EDGES, X_EDGES, weights=area, member_dim="member", name="et")
    if sources is not None:
        bs = bs.assign_coords(source=("member", sources))
    return bs


def direct_mean(et, area):
    """sum_i a_i e_i / sum_i a_i over each member's valid gridcells."""
    a = area.where(et.notnull())
    return (a * et).sum(("lat", "lon")) / a.sum(("lat", "lon"))


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.maximum(np.abs(np.asarray(b)), 1e-300)))


@pytest.fixture
def pair(rng, area):
    """p and f of a dataset m and a reference r, every bin occupied by both."""
    et, lai, ai = make_fields(rng, ["m", "r"])
    p, f = dc.occupancy_and_mean(binned(et, lai, ai, area))
    assert bool(f.notnull().all())
    return p.sel(member="m"), f.sel(member="m"), p.sel(member="r"), f.sel(member="r")


# ------------------------------------------------------------------
# 1-3. The identities
# ------------------------------------------------------------------

def test_terms_sum_to_E_mm_minus_E_rr(pair):
    d = dc.decompose(*pair)
    assert rel(d["FUNCTION"] + d["STATE"] + d["INTERACTION"], d["E_mm"] - d["E_rr"]) < RTOL


def test_full_support_gives_full_domain_means(pair):
    p_m, f_m, p_r, f_r = pair
    d = dc.decompose(*pair)
    assert rel(d["E_mm"], d["E_m"]) < RTOL and rel(d["E_rr"], d["E_r"]) < RTOL
    assert float(d["OFF_SUPPORT_m"]) == 0 and float(d["OFF_SUPPORT_r"]) == 0
    # The counterfactual means mix the state of one with the function of the other
    assert rel(d["E_rm"], (p_r * f_m).sum()) < RTOL and rel(d["E_mr"], (p_m * f_r).sum()) < RTOL


def test_equal_f_or_equal_p(pair):
    p_m, f_m, p_r, f_r = pair
    d = dc.decompose(p_m, f_r, p_r, f_r)  # same bin means
    assert float(d["FUNCTION"]) == 0 and float(d["INTERACTION"]) == 0
    assert float(d["E_rm"]) == float(d["E_rr"])
    assert float(d["STATE"]) != 0
    d = dc.decompose(p_r, f_m, p_r, f_r)  # same occupancy
    assert float(d["STATE"]) == 0 and float(d["INTERACTION"]) == 0
    assert float(d["E_mr"]) == float(d["E_rr"])
    assert float(d["FUNCTION"]) != 0


def test_state_shift(pair):
    p_m, f_m, p_r, f_r = pair
    sh, d = dc.state_shift(*pair), dc.decompose(*pair)
    assert rel(sh["STATE_GAINS"] + sh["STATE_LOSSES"], d["STATE"]) < RTOL
    assert float(sh["STATE_GAINS"]) > 0 > float(sh["STATE_LOSSES"])
    assert rel(sh["LAND_SHIFT"], abs(p_m - p_r).sum() / 2) < RTOL
    # No shift: no land in other bins and no STATE
    sh = dc.state_shift(p_r, f_m, p_r, f_r)
    assert float(sh["LAND_SHIFT"]) == 0 and float(sh["STATE_GAINS"]) == 0 and float(sh["STATE_LOSSES"]) == 0
    # Land moved between two bins with the same observed ET: a shift, no STATE
    dims = ("y_bin", "x_bin")
    p_r = xr.DataArray([[0.25, 0.25], [0.25, 0.25]], dims=dims)
    f_r = xr.DataArray([[10.0, 10.0], [20.0, 30.0]], dims=dims)
    p_m = xr.DataArray([[0.40, 0.10], [0.25, 0.25]], dims=dims)
    sh = dc.state_shift(p_m, f_r, p_r, f_r)
    assert float(sh["LAND_SHIFT"]) == pytest.approx(0.15)
    assert float(sh["STATE_GAINS"]) == pytest.approx(1.5) and float(sh["STATE_LOSSES"]) == pytest.approx(-1.5)
    assert float(dc.decompose(p_m, f_r, p_r, f_r)["STATE"]) == pytest.approx(0)


# ------------------------------------------------------------------
# 4. p and f reproduce the area-mean ET; count does not
# ------------------------------------------------------------------

def test_area_mean_from_bins(rng, area):
    et, lai, ai = make_fields(rng, ["a", "b"])
    et = et.where(rng.uniform(size=et.shape) > 0.1)  # some missing gridcells
    bs = binned(et, lai, ai, area)
    p, f = dc.occupancy_and_mean(bs)
    np.testing.assert_allclose((p * f).sum(dc.BIN_DIMS), direct_mean(et, area), rtol=RTOL)
    np.testing.assert_allclose(p.sum(dc.BIN_DIMS), 1, rtol=RTOL)
    # Occupancy from gridcell counts ignores the varying land area
    count = bs.sel(stats="count")
    p_count = count / count.sum(dc.BIN_DIMS)
    assert rel((p_count * f).sum(dc.BIN_DIMS), direct_mean(et, area)) > 1e-6


# ------------------------------------------------------------------
# 5. Pooled datasets
# ------------------------------------------------------------------

def test_pool_gives_mean_of_sources(rng, area):
    # Sources with 1, 2 and 3 members
    members = ["a0", "b0", "b1", "c0", "c1", "c2"]
    sources = ["a", "b", "b", "c", "c", "c"]
    et, lai, ai = make_fields(rng, members)
    et = et.where(rng.uniform(size=et.shape) > 0.2)
    p, f = dc.occupancy_and_mean(binned(et, lai, ai, area, sources))
    E_member = (p * f).sum(dc.BIN_DIMS)
    p_src, f_src = dc.members_to_sources(p, f)
    E_src = (p_src * f_src).sum(dc.BIN_DIMS)
    # Members are averaged within a source, each source weighs once
    expected_src = E_member.groupby("source").mean().sel(source=["a", "b", "c"])
    np.testing.assert_allclose(E_src, expected_src, rtol=RTOL)
    p_pool, f_pool = dc.pool(p_src, f_src, dc.source_weights(p_src["source"].values), "source")
    assert rel((p_pool * f_pool).sum(), E_src.mean()) < RTOL
    assert rel(p_pool.sum(), 1) < RTOL
    # Not the member-weighted mean, since the sources have unequal member counts
    assert rel((p_pool * f_pool).sum(), E_member.mean()) > 1e-8


# ------------------------------------------------------------------
# 6. Empty bins
# ------------------------------------------------------------------

def test_off_support(rng, area):
    et, lai, ai = make_fields(rng, ["m", "r"])
    # m has no land in the lowest LAI and AI bin, r none in the highest
    low = (lai < Y_EDGES[1]) & (ai < X_EDGES[1])
    high = (lai >= Y_EDGES[-2]) & (ai >= X_EDGES[-2])
    lai = xr.where(low & (lai.member == "m"), 1.5, lai)
    lai = xr.where(high & (lai.member == "r"), 3.5, lai)
    p, f = dc.occupancy_and_mean(binned(et, lai, ai, area))
    d = dc.decompose(p.sel(member="m"), f.sel(member="m"), p.sel(member="r"), f.sel(member="r"))
    for q in dc.QUANTITIES:
        assert np.isfinite(float(d[q])), q
    # Off the support: r's land in m's empty bin, and m's land in r's empty bin
    A = float(area.sum())
    off_r = low.sel(member="r")
    off_m = ((lai >= Y_EDGES[-2]) & (ai >= X_EDGES[-2])).sel(member="m")
    assert rel(d["OFF_SUPPORT_r"], float(area.where(off_r).sum()) / A) < RTOL
    assert rel(d["OFF_SUPPORT_m"], float(area.where(off_m).sum()) / A) < RTOL
    # The full-domain means minus the support means are the ET of the off-support land
    assert rel(d["E_r"] - d["E_rr"], float((area * et.sel(member="r")).where(off_r).sum()) / A) < RTOL
    assert rel(d["E_m"] - d["E_mm"], float((area * et.sel(member="m")).where(off_m).sum()) / A) < RTOL
    assert rel(d["FUNCTION"] + d["STATE"] + d["INTERACTION"], d["E_mm"] - d["E_rr"]) < RTOL


# ------------------------------------------------------------------
# 7. Variance partition
# ------------------------------------------------------------------

def test_variance_partition_sums(rng, area):
    members = [f"m{i}" for i in range(6)]
    et, lai, ai = make_fields(rng, members + ["r"])
    p, f = dc.occupancy_and_mean(binned(et, lai, ai, area))
    assert bool(f.notnull().all())
    terms = dc.decompose(p.sel(member=members), f.sel(member=members), p.sel(member="r"), f.sel(member="r"))
    vp = dc.variance_partition(terms, "member")
    var = vp["variance"]
    # Against a fixed reference with a complete support, E_rr is the same for every member
    target = float(terms["E_mm"].var("member", ddof=0))
    assert float(var.sel(component="var_E_rr")) < 1e-20
    assert rel(var.sel(component="var_E_mm"), target) < RTOL
    assert rel(var.sel(component="var_E_mm_minus_E_rr"), target) < RTOL
    assert rel(var.sel(component=list(dc.COMPONENTS)).sum(), target) < RTOL
    assert rel(vp["fraction"].sel(component=list(dc.COMPONENTS)).sum(), 1) < RTOL
    # Var(STATE) = Var(E_mr) and Var(FUNCTION) = Var(E_rm), since E_rr is fixed
    assert rel(var.sel(component="var_STATE"), terms["E_mr"].var("member", ddof=0)) < 1e-8
    assert rel(var.sel(component="var_FUNCTION"), terms["E_rm"].var("member", ddof=0)) < 1e-8


# ------------------------------------------------------------------
# 8. Shared AI
# ------------------------------------------------------------------

def test_shared_ai_marginal(rng, area):
    # Members differ only in LAI (and ET), as the TRENDY models do
    ai = rng.uniform(0, 3.5, (LAT.size, LON.size))
    et, lai, ai = make_fields(rng, ["a", "b", "c"], ai=ai)
    p, _ = dc.occupancy_and_mean(binned(et, lai, ai, area))
    ai_marginal = p.sum("y_bin")
    for m in ("b", "c"):
        np.testing.assert_allclose(ai_marginal.sel(member=m), ai_marginal.sel(member="a"), rtol=RTOL)
    assert float(abs(p.sel(member="b") - p.sel(member="a")).max()) > 0


# ------------------------------------------------------------------
# 9-10. Obs product axes and pooling weights
# ------------------------------------------------------------------

AXES = ("et", "lai", "pr", "rn")


@pytest.fixture
def obs(rng, area):
    """Two ET x two LAI products (full factorial), one P and one Rn product: p and f along source."""
    combos = [(e, l) for e in ("E1", "E2") for l in ("L1", "L2")]
    members = [f"{e}-{l}" for e, l in combos]
    ai = rng.uniform(0, 3.5, (LAT.size, LON.size))
    lai_by_product = {"L1": rng.uniform(0, 5.5, ai.shape), "L2": rng.uniform(0, 5.5, ai.shape)}
    lai = np.stack([lai_by_product[l] for _, l in combos])
    # The ET products differ in their ET, not in LAI or AI
    et = np.stack([(1.0 if e == "E1" else 1.3) * (20 + 10 * lai[i] - 8 * ai) for i, (e, _) in enumerate(combos)])
    shape = (len(members),) + ai.shape
    bs = binned(field(et, members), field(lai, members), field(np.broadcast_to(ai, shape).copy(), members),
                area, members)
    bs = bs.assign_coords(et_product=("member", [e for e, _ in combos]), lai_product=("member", [l for _, l in combos]),
                          pr_product=("member", ["P1"] * 4), rn_product=("member", ["R1"] * 4))
    return dc.members_to_sources(*dc.occupancy_and_mean(bs))


def test_obs_product_axes(obs, capsys):
    p, f = obs
    # Sources that differ only in their ET product share the state
    d = dc.decompose(p.sel(source="E2-L1"), f.sel(source="E2-L1"), p.sel(source="E1-L1"), f.sel(source="E1-L1"))
    assert abs(float(d["STATE"])) < 1e-12 and abs(float(d["INTERACTION"])) < 1e-12
    assert float(d["FUNCTION"]) != 0
    # Sources that differ in their LAI product do not
    d = dc.decompose(p.sel(source="E1-L2"), f.sel(source="E1-L2"), p.sel(source="E1-L1"), f.sel(source="E1-L1"))
    assert abs(float(d["STATE"])) > 1e-6

    # Ranges by axis: a quantity that depends on each axis separately
    products = {a: p[f"{a}_product"].values for a in AXES}
    value = xr.DataArray([0.0, 2.0, 5.0, 7.0], dims="obs_reference")  # E1-L1, E1-L2, E2-L1, E2-L2: +5 ET, +2 LAI
    rng_axis = dc.product_axis_range(value, "obs_reference", AXES, products)
    assert list(rng_axis["axis"].values) == ["all", "et", "lai"]
    np.testing.assert_allclose(rng_axis.sel(axis="all"), 7)
    np.testing.assert_allclose(rng_axis.sel(axis="et"), 5)
    np.testing.assert_allclose(rng_axis.sel(axis="lai"), 2)
    out = capsys.readouterr().out
    assert "pr: one product" in out and "rn: one product" in out


def test_non_factorial_weights():
    # Three of the four ET x LAI combinations
    products = {"et": ["E1", "E1", "E2"], "lai": ["L1", "L2", "L1"], "pr": ["P1"] * 3, "rn": ["R1"] * 3}
    assert not dc.is_full_factorial(products)
    assert dc.is_full_factorial({"et": ["E1", "E2"], "lai": ["L1", "L1"]})
    w = dc.source_weights(["E1-L1", "E1-L2", "E2-L1"], "source")
    eff = dc.product_weights(w, products)
    assert eff["et"] == pytest.approx({"E1": 2 / 3, "E2": 1 / 3})
    assert eff["lai"] == pytest.approx({"L1": 2 / 3, "L2": 1 / 3})
    assert eff["pr"] == pytest.approx({"P1": 1.0})
    with pytest.raises(ValueError):
        dc.source_weights(["a"], "hierarchical")


def test_variant_source_weights():
    """'variant': each group weighs once, its sources share it; pooling with them is the mean of the group means."""
    w = dc.source_weights(["a1", "a2", "b"], "variant", groups=["A", "A", "B"])
    np.testing.assert_allclose(w, [0.25, 0.25, 0.5])
    assert list(w["source"].values) == ["a1", "a2", "b"]
    with pytest.raises(ValueError, match="group of every source"):
        dc.source_weights(["a1", "a2"], "variant")
