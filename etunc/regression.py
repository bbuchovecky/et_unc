"""
etunc.regression
================
Ordinary least squares (OLS) regression of ET on climatological leaf area
index (LAI) and aridity index (AI), to measure how much of the ET variance
binned in the LAI-AI heatmaps (`binning.bin_stats`) LAI and AI explain.

- `ols`: OLS fit of y on the columns of X, with an intercept.
- `lai_ai_regression`: per member, ET ~ LAI, ET ~ AI and ET ~ LAI + AI on
  the samples that `binning.bin_stats` bins, and the variance partition.

Variance partition between LAI and AI
-------------------------------------
LAI and AI are correlated, so the R2 of the two single-predictor fits do not
add up to the R2 of the joint fit. With two predictors, the joint R2 splits
exactly into (commonality analysis):

    unique LAI = R2(LAI + AI) - R2(AI)     explained by LAI only
    unique AI  = R2(LAI + AI) - R2(LAI)    explained by AI only
    shared     = R2(LAI) + R2(AI) - R2(LAI + AI)

with unique LAI + unique AI + shared = R2(LAI + AI). `shared` is negative
when one predictor makes the other more useful (suppression).
"""

from __future__ import annotations

import numpy as np
import xarray as xr


# Fits of `lai_ai_regression`, along its `predictors` dimension
PREDICTORS = ("lai", "ai", "lai+ai")


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """
    OLS fit of `y` (n,) on the columns of `X` (n, k), with an intercept.

    Returns
    -------
    coef : (k + 1,) array, intercept first
    r2 : coefficient of determination, 1 - SS_res / SS_tot (NaN if `y` is constant)
    """
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_res = np.sum((y - A @ coef) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coef, r2


def lai_ai_regression(
    target: xr.DataArray,
    lai: xr.DataArray,
    ai: xr.DataArray,
    *,
    mask: xr.DataArray | None = None,
    member_dim: str = "member",
) -> xr.Dataset:
    """
    OLS regressions of `target` (e.g. annual mean ET, (member, year, lat, lon))
    on the climatological `lai` and `ai` (member, lat, lon) of the same member,
    one set of fits per member.

    The samples are those of `binning.bin_stats`: LAI and AI are broadcast
    over `year`, every (year, gridcell) sample counts once (unweighted), and
    samples with NaN in any of the three fields, or outside `mask`, are dropped.

    Returns
    -------
    xr.Dataset along `member_dim`, with the member coords of `target`:
        r2 (member, predictors) : R2 of ET ~ LAI, ET ~ AI and ET ~ LAI + AI
        r2_unique_lai, r2_unique_ai, r2_shared : variance partition of R2(LAI + AI)
        intercept, slope_lai, slope_ai : coefficients of ET ~ LAI + AI
        n_samples : number of samples fitted
    Members with fewer than 3 samples are NaN.
    """
    # Coordinates must match exactly, as in `binning.bin_stats`
    if mask is not None:
        target, lai, ai, mask = xr.align(target, lai, ai, mask, join="exact")
        keep = mask == 1
        target, lai, ai = target.where(keep), lai.where(keep), ai.where(keep)
    else:
        target, lai, ai = xr.align(target, lai, ai, join="exact")

    names = ("intercept", "slope_lai", "slope_ai", "r2_lai", "r2_ai", "r2_lai+ai", "n_samples")
    rows = []
    for m in range(target.sizes[member_dim]):
        # LAI and AI repeated over `year`: each (year, gridcell) ET value is one sample
        t, x1, x2 = xr.broadcast(*(da.isel({member_dim: m}) for da in (target, lai, ai)))
        y, x1, x2 = (np.asarray(da.transpose(*t.dims).values, dtype=float).ravel() for da in (t, x1, x2))
        valid = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
        y, x1, x2 = y[valid], x1[valid], x2[valid]
        if y.size < 3:
            rows.append([np.nan] * (len(names) - 1) + [y.size])
            continue
        _, r2_lai = ols(y, x1[:, None])
        _, r2_ai = ols(y, x2[:, None])
        coef, r2_both = ols(y, np.column_stack([x1, x2]))
        rows.append([*coef, r2_lai, r2_ai, r2_both, y.size])
    res = dict(zip(names, np.array(rows, dtype=float).T))

    r2 = xr.DataArray(
        np.column_stack([res["r2_lai"], res["r2_ai"], res["r2_lai+ai"]]), dims=(member_dim, "predictors"),
        coords={"predictors": list(PREDICTORS)}, attrs={"long_name": "R2 of the OLS fit of ET on the predictors"},
    )
    units = target.attrs.get("units", "W/m2")

    def var(values, **attrs):
        return xr.DataArray(values, dims=member_dim, attrs=attrs)

    out = xr.Dataset({
        "r2": r2,
        "r2_unique_lai": var(res["r2_lai+ai"] - res["r2_ai"], long_name="R2 explained by LAI only"),
        "r2_unique_ai": var(res["r2_lai+ai"] - res["r2_lai"], long_name="R2 explained by AI only"),
        "r2_shared": var(res["r2_lai"] + res["r2_ai"] - res["r2_lai+ai"],
                         long_name="R2 shared by LAI and AI (negative under suppression)"),
        "intercept": var(res["intercept"], long_name="intercept of ET ~ LAI + AI", units=units),
        "slope_lai": var(res["slope_lai"], long_name="LAI coefficient of ET ~ LAI + AI", units=f"{units} per m2/m2"),
        "slope_ai": var(res["slope_ai"], long_name="AI coefficient of ET ~ LAI + AI", units=units),
        "n_samples": var(res["n_samples"].astype("i8"), long_name="number of (year, gridcell) samples fitted"),
    })
    member_coords = {c: target[c] for c in target.coords if target[c].dims == (member_dim,)}
    return out.assign_coords(member_coords)
