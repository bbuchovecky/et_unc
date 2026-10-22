"""
Tests for etunc.regression (OLS of ET on LAI and AI) and
binning.between_bin_fraction (eta2 of the bins). Run from the project root with:

    python -m pytest tests/test_regression.py
"""

import numpy as np
import pytest
import xarray as xr

import etunc.binning as binning
import etunc.regression as regression


# ------------------------------------------------------------------
# Fixtures and helpers
# ------------------------------------------------------------------

LAT = np.linspace(-50, 80, 12)
LON = np.linspace(-180, 170, 18)
YEARS = np.arange(2000, 2006)
MEMBERS = ["m0", "m1", "m2"]


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def fields(rng):
    """
    Synthetic inputs: correlated climatological LAI and AI (member, lat, lon), and
    ET = 10 + 8 LAI - 5 AI + noise (member, year, lat, lon), with NaN gridcells.
    """
    shape = (len(MEMBERS), len(LAT), len(LON))
    coords = {"member": MEMBERS, "lat": LAT, "lon": LON}
    lai_np = rng.uniform(0, 6, shape)
    ai_np = 3 - 0.3 * lai_np + rng.uniform(0, 2, shape)  # negatively correlated with LAI, as in the data
    lai_np[:, 0, :3] = np.nan
    lai = xr.DataArray(lai_np, dims=("member", "lat", "lon"), coords=coords, name="lai")
    ai = xr.DataArray(ai_np, dims=("member", "lat", "lon"), coords=coords, name="ai")
    noise = rng.normal(0, 2, (len(MEMBERS), len(YEARS), len(LAT), len(LON)))
    et = (10 + 8 * lai - 5 * ai).expand_dims(year=YEARS, axis=1) + noise
    et = et.rename("et").assign_attrs(units="W/m2").assign_coords(source=("member", ["a", "a", "b"]))
    return et, lai, ai


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_ols_recovers_exact_linear_relation(rng):
    X = rng.normal(size=(100, 2))
    y = 3 + 2 * X[:, 0] - 0.5 * X[:, 1]
    coef, r2 = regression.ols(y, X)
    np.testing.assert_allclose(coef, [3, 2, -0.5], atol=1e-10)
    assert r2 == pytest.approx(1)


def test_ols_r2_matches_squared_correlation(rng):
    # With one predictor, R2 is the squared Pearson correlation
    x = rng.normal(size=500)
    y = x + rng.normal(size=500)
    _, r2 = regression.ols(y, x[:, None])
    assert r2 == pytest.approx(np.corrcoef(x, y)[0, 1] ** 2)


def test_ols_constant_target_has_nan_r2():
    _, r2 = regression.ols(np.ones(10), np.arange(10.0)[:, None])
    assert np.isnan(r2)


def test_lai_ai_regression_coefficients_and_partition(fields):
    et, lai, ai = fields
    fit = regression.lai_ai_regression(et, lai, ai)
    assert fit["r2"].dims == ("member", "predictors")
    np.testing.assert_allclose(fit["slope_lai"], 8, atol=0.2)
    np.testing.assert_allclose(fit["slope_ai"], -5, atol=0.3)
    np.testing.assert_allclose(fit["intercept"], 10, atol=1)
    # The partition adds up to the joint R2, which is at least each single-predictor R2
    r2_both = fit["r2"].sel(predictors="lai+ai")
    np.testing.assert_allclose(fit["r2_unique_lai"] + fit["r2_unique_ai"] + fit["r2_shared"], r2_both)
    assert (r2_both >= fit["r2"].sel(predictors="lai") - 1e-12).all()
    assert (r2_both >= fit["r2"].sel(predictors="ai") - 1e-12).all()
    # Member coords are kept
    assert list(fit["source"].values) == ["a", "a", "b"]


def test_lai_ai_regression_matches_pooled_samples(fields):
    # Climatological LAI/AI are repeated over year: the fit of one member equals an OLS
    # on its flattened (year, gridcell) samples
    et, lai, ai = fields
    fit = regression.lai_ai_regression(et, lai, ai)
    e = et.isel(member=1)
    l, a = (da.isel(member=1).broadcast_like(e) for da in (lai, ai))
    y, x1, x2 = (da.values.ravel() for da in (e, l, a))
    ok = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    coef, r2 = regression.ols(y[ok], np.column_stack([x1[ok], x2[ok]]))
    assert fit["r2"].sel(predictors="lai+ai").isel(member=1) == pytest.approx(r2)
    assert fit["slope_ai"].isel(member=1) == pytest.approx(coef[2])
    assert int(fit["n_samples"].isel(member=1)) == ok.sum()


def test_lai_ai_regression_uses_bin_stats_samples(fields):
    # With a mask, the fitted samples are exactly the samples bin_stats bins
    et, lai, ai = fields
    mask = xr.DataArray(np.ones((len(LAT), len(LON)), bool), dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})
    mask[5:, :] = False
    fit = regression.lai_ai_regression(et, lai, ai, mask=mask)
    bs = binning.bin_stats(et, lai, ai, np.linspace(0, 6, 4), np.linspace(0, 5, 4), mask=mask, member_dim="member")
    np.testing.assert_array_equal(bs.sel(stats="count").sum(("y_bin", "x_bin")).values, fit["n_samples"].values)
    # 5 unmasked latitudes, minus the 3 NaN LAI gridcells at the first one, over every year
    assert int(fit["n_samples"].isel(member=0)) == (5 * len(LON) - 3) * len(YEARS)


def test_lai_ai_regression_rejects_mismatched_grids(fields):
    et, lai, ai = fields
    with pytest.raises(ValueError):
        regression.lai_ai_regression(et, lai.assign_coords(lat=LAT + 0.1), ai)


def test_lai_ai_regression_member_without_samples(fields):
    et, lai, ai = fields
    et = et.where(et["member"] != "m2")
    fit = regression.lai_ai_regression(et, lai, ai)
    assert fit["r2"].sel(member="m2").isnull().all()
    assert int(fit["n_samples"].sel(member="m2")) == 0


def test_between_bin_fraction_matches_direct_computation(fields):
    et, lai, ai = fields
    y_edges, x_edges = np.linspace(0, 6, 5), np.linspace(0, 5, 4)
    bs = binning.bin_stats(et, lai, ai, y_edges, x_edges, member_dim="member")
    eta2 = binning.between_bin_fraction(bs)

    # Direct: 1 - SS of the residuals from each sample's bin mean / total SS
    e = et.isel(member=0)
    l, a = (da.isel(member=0).broadcast_like(e) for da in (lai, ai))
    y, x1, x2 = (da.values.ravel() for da in (e, l, a))
    ok = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    y, x1, x2 = y[ok], x1[ok], x2[ok]
    iy = np.clip(np.searchsorted(y_edges, x1, side="right") - 1, 0, len(y_edges) - 2)
    ix = np.clip(np.searchsorted(x_edges, x2, side="right") - 1, 0, len(x_edges) - 2)
    b = iy * (len(x_edges) - 1) + ix
    bin_mean = np.bincount(b, weights=y) / np.maximum(np.bincount(b), 1)  # empty bins are never indexed
    expected = 1 - np.sum((y - bin_mean[b]) ** 2) / np.sum((y - y.mean()) ** 2)
    assert float(eta2.isel(member=0)) == pytest.approx(expected)
    assert eta2.dims == ("member",)


def test_between_bin_fraction_weighted_matches_direct_computation(fields):
    """With weighted bin stats, eta2 is the weighted R2 of predicting each sample by its bin's weighted mean."""
    et, lai, ai = fields
    y_edges, x_edges = np.linspace(0, 6, 5), np.linspace(0, 5, 4)
    area = xr.DataArray(np.cos(np.deg2rad(LAT))[:, None] * np.ones(len(LON)), dims=("lat", "lon"),
                        coords={"lat": LAT, "lon": LON})
    bs = binning.bin_stats(et, lai, ai, y_edges, x_edges, weights=area, member_dim="member")
    eta2 = binning.between_bin_fraction(bs)

    e = et.isel(member=0)
    l, a, w = (da.broadcast_like(e) for da in (lai.isel(member=0), ai.isel(member=0), area))
    y, x1, x2, w = (da.values.ravel() for da in (e, l, a, w))
    ok = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    y, x1, x2, w = y[ok], x1[ok], x2[ok], w[ok]
    iy = np.clip(np.searchsorted(y_edges, x1, side="right") - 1, 0, len(y_edges) - 2)
    ix = np.clip(np.searchsorted(x_edges, x2, side="right") - 1, 0, len(x_edges) - 2)
    b = iy * (len(x_edges) - 1) + ix
    sum_w = np.bincount(b, weights=w)
    bin_mean = np.bincount(b, weights=w * y) / np.where(sum_w > 0, sum_w, 1)  # empty bins are never indexed
    grand_mean = np.average(y, weights=w)
    expected = 1 - np.sum(w * (y - bin_mean[b]) ** 2) / np.sum(w * (y - grand_mean) ** 2)
    assert float(eta2.isel(member=0)) == pytest.approx(expected)


def test_between_bin_fraction_is_one_without_within_bin_variance():
    # A target constant within each bin is fully explained by the bins
    lai = xr.DataArray(np.repeat([0.5, 1.5, 2.5], 4), dims="cell")
    ai = xr.DataArray(np.tile([0.5, 1.5], 6), dims="cell")
    et = 3 * np.floor(lai) + np.floor(ai)
    bs = binning.bin_stats(et, lai, ai, np.array([0, 1, 2, 3.0]), np.array([0, 1, 2.0]))
    assert float(binning.between_bin_fraction(bs)) == pytest.approx(1)
