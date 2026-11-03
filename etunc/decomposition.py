"""
etunc.decomposition
===================
State, function and interaction decomposition of the area-mean ET of a dataset m
relative to an observational reference r, on a shared LAI-AI partition (the bins
of `binning.bin_stats`) of a shared domain (the common area mask).

With a_i the land area and e_i the climatological ET of gridcell i, and b a
LAI-AI bin, from land-area weighted `bin_stats` outputs:

    f(b) = `mean`(b) = sum_{i in b} a_i e_i / sum_{i in b} a_i    bin-mean ET ("function")
    p(b) = `sum_w`(b) / sum_b `sum_w`                           land-area occupancy ("state")
    E    = sum_b p(b) f(b)                                      domain-mean ET

On the common support S (bins where f_m and f_r are both finite), four
counterfactual domain means (first index: state, second: function):

    E_rr = sum_S p_r f_r    observed state, observed function
    E_rm = sum_S p_r f_m    observed state, m's function
    E_mr = sum_S p_m f_r    m's state, observed function
    E_mm = sum_S p_m f_m    m's state, m's function

    FUNCTION    = E_rm - E_rr
    STATE       = E_mr - E_rr
    INTERACTION = E_mm - E_rm - E_mr + E_rr   (> 0: the biases reinforce; < 0: they compensate)
    FUNCTION + STATE + INTERACTION = E_mm - E_rr

The occupancy outside S (OFF_SUPPORT_m, OFF_SUPPORT_r) holds the rest of the
full-domain means: E_m - E_mm and E_r - E_rr.

This is an accounting identity, not a causal attribution: in coupled runs
LAI and AI respond to ET. FUNCTION holds every control of ET that is not on
the two axes (soil moisture, seasonality, VPD, rooting depth, ...).

- `occupancy_and_mean`: p and f of a weighted `bin_stats` output.
- `pool`: p and f of several datasets pooled with given weights.
- `members_to_sources`: the members of each source pooled with equal weights.
- `source_weights`, `product_weights`: weights of the obs sources in the obs pooled
  reference, and the weight this gives each product of each product axis.
- `counterfactual_mean`: sum_S p f for one state and one function.
- `decompose`: the scalar quantities above (QUANTITIES).
- `product_axis_range`: range of a quantity across obs references, overall and by product axis.
- `variance_partition`: variance of E_mm - E_rr across datasets, split into the terms.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import xarray as xr

BIN_DIMS = ("y_bin", "x_bin")

# Variables of `decompose`
QUANTITIES = ("E_rr", "E_rm", "E_mr", "E_mm", "FUNCTION", "STATE", "INTERACTION", "E_m", "E_r",
              "OFF_SUPPORT_m", "OFF_SUPPORT_r")

# Components of `variance_partition`, which sum to var_E_mm_minus_E_rr
COMPONENTS = ("var_STATE", "var_FUNCTION", "var_INTERACTION", "2cov_STATE_FUNCTION",
              "2cov_STATE_INTERACTION", "2cov_FUNCTION_INTERACTION")

# Weightings of the obs sources in the obs pooled reference (`source_weights`)
OBS_POOL_WEIGHTINGS = ("source",)


def occupancy_and_mean(bs: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Occupancy p and bin-mean f of a `bin_stats` output ([member,] stats, y_bin, x_bin).

    Assumes land-area weighted statistics: then `sum_w` is the land area in each bin,
    so p is the fraction of the domain's land area in each bin (sum_b p = 1) and
    sum_b p f is the area-mean ET. Never `count` or `n_eff`, which count gridcells
    whatever their area. f is NaN in empty bins.
    """
    sum_w = bs.sel(stats="sum_w", drop=True)
    p = sum_w / sum_w.sum(BIN_DIMS)
    f = bs.sel(stats="mean", drop=True)
    return p.rename("p"), f.rename("f")


def pool(p: xr.DataArray, f: xr.DataArray, weights: xr.DataArray, dim: str) -> tuple[xr.DataArray, xr.DataArray]:
    """
    p and f of the datasets along `dim` pooled with `weights` (along `dim`, sum 1):
    p_pool = sum_s w_s p_s and f_pool = sum_s w_s p_s f_s / p_pool, so that
    E_pool = sum_s w_s E_s exactly. Not `binning.pool_members`, which weights each
    dataset by its sum of weights. f_pool is NaN where no dataset occupies the bin.
    """
    p_pool = (weights * p).sum(dim)
    # Empty bins have p = 0 and f = NaN; they add nothing to the sum
    pf = (weights * p * f.fillna(0)).sum(dim)
    f_pool = (pf / p_pool).where(p_pool > 0)
    return p_pool.rename("p"), f_pool.rename("f")


def members_to_sources(p: xr.DataArray, f: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    p and f along `member` pooled within each source (the `source` coord), each member of
    a source weighing equally (`pool`). Returns them along `source`, with the member
    coords that are the same for every member of a source (e.g. obs product coords).
    """
    sources = list(dict.fromkeys(p["source"].values.tolist()))
    p_out, f_out = [], []
    for s in sources:
        idx = np.flatnonzero(p["source"].values == s)
        w = xr.DataArray(np.full(idx.size, 1 / idx.size), dims="member")
        ps, fs = pool(p.isel(member=idx), f.isel(member=idx), w, "member")
        p_out.append(ps)
        f_out.append(fs)
    # Member coords constant within every source
    same = [c for c in p.coords if p[c].dims == ("member",) and c not in ("member", "source")
            and all(len(set(p[c].values[p["source"].values == s].tolist())) == 1 for s in sources)]
    first = [int(np.flatnonzero(p["source"].values == s)[0]) for s in sources]
    coords = {"source": sources, **{c: ("source", p[c].values[first]) for c in same}}
    p_src = xr.concat(p_out, dim="source").assign_coords(coords)
    f_src = xr.concat(f_out, dim="source").assign_coords(coords)
    return p_src, f_src


def source_weights(sources: Sequence[str], weighting: str = "source") -> xr.DataArray:
    """
    Weight of each obs source in the obs pooled reference, along `source` (sum 1).
    "source": every source weighs equally. If the product combinations are not a full
    factorial, this gives products unequal weight (see `product_weights`).
    """
    if weighting not in OBS_POOL_WEIGHTINGS:
        raise ValueError(f"Unknown obs pool weighting {weighting!r}; one of {OBS_POOL_WEIGHTINGS}")
    n = len(sources)
    return xr.DataArray(np.full(n, 1 / n), dims="source", coords={"source": list(sources)}, name="weight")


def product_weights(weights: xr.DataArray, products: dict[str, Sequence[str]]) -> dict[str, dict[str, float]]:
    """
    Effective weight of each product on each axis: the sum of the `weights` of the sources
    that use it. `products` is {axis: product of each source, in the order of `weights`}.
    """
    w = np.asarray(weights.values, dtype=float)
    out = {}
    for axis, prods in products.items():
        prods = np.asarray(prods)
        out[axis] = {str(q): float(w[prods == q].sum()) for q in dict.fromkeys(prods.tolist())}
    return out


def counterfactual_mean(p: xr.DataArray, f: xr.DataArray, support: xr.DataArray) -> xr.DataArray:
    """
    sum_{b in S} p(b) f(b): the domain-mean ET of the state p with the function f, over the
    bins of `support` only (both p and f must belong to datasets with ET in those bins).
    """
    return (p * f).where(support).sum(BIN_DIMS)


def decompose(p_m: xr.DataArray, f_m: xr.DataArray, p_r: xr.DataArray, f_r: xr.DataArray) -> xr.Dataset:
    """
    Scalar quantities (QUANTITIES) of m relative to r, summed over the bins; other dims
    (e.g. dataset and obs_reference) broadcast. The four counterfactual means are over
    the common support S (f_m and f_r both finite); bins outside it are never filled, and
    their occupancy is reported as OFF_SUPPORT_m/_r. E_m and E_r are the full-domain means.
    Non-index coords of the inputs are dropped (those of m and r would clash); label the result.
    """
    p_m, f_m, p_r, f_r = (da.reset_coords(drop=True) for da in (p_m, f_m, p_r, f_r))
    support = f_m.notnull() & f_r.notnull()
    E_rr = counterfactual_mean(p_r, f_r, support)
    E_rm = counterfactual_mean(p_r, f_m, support)
    E_mr = counterfactual_mean(p_m, f_r, support)
    E_mm = counterfactual_mean(p_m, f_m, support)
    E_m = (p_m * f_m).sum(BIN_DIMS)
    E_r = (p_r * f_r).sum(BIN_DIMS)
    out = xr.Dataset({
        "E_rr": E_rr, "E_rm": E_rm, "E_mr": E_mr, "E_mm": E_mm,
        "FUNCTION": E_rm - E_rr, "STATE": E_mr - E_rr, "INTERACTION": E_mm - E_rm - E_mr + E_rr,
        "E_m": E_m, "E_r": E_r,
        "OFF_SUPPORT_m": p_m.where(~support).sum(BIN_DIMS),
        "OFF_SUPPORT_r": p_r.where(~support).sum(BIN_DIMS),
    })
    # Every variable on the dims of m and r alike, e.g. (dataset, obs_reference)
    return xr.broadcast(out)[0]


def product_axis_range(
    da: xr.DataArray, dim: str, axes: Sequence[str], products: dict[str, Sequence[str]],
) -> xr.DataArray:
    """
    Range (max - min) of `da` across the obs references along `dim`, along a new `axis`:
    "all" (across every reference), and for each product axis with more than one product,
    the range across that axis's products with the other axes held fixed, averaged over
    the fixed combinations (combinations with one reference are left out). `products` is
    {axis: product of each reference along `dim`}. Axes with one product are skipped, with a note.
    """
    out = {"all": da.max(dim) - da.min(dim)}
    prods = {a: np.asarray(products[a]) for a in axes}
    for a in axes:
        if len(set(prods[a].tolist())) < 2:
            print(f"product axis {a}: one product ({prods[a][0]}), skipped")
            continue
        others = [b for b in axes if b != a]
        keys = list(zip(*(prods[b].tolist() for b in others))) if others else [()] * da.sizes[dim]
        ranges = []
        for key in dict.fromkeys(keys):
            idx = [i for i, k in enumerate(keys) if k == key]
            if len(idx) > 1:
                sub = da.isel({dim: idx})
                ranges.append(sub.max(dim) - sub.min(dim))
        if not ranges:
            print(f"product axis {a}: no two references differ only in {a}, skipped")
            continue
        out[a] = xr.concat(ranges, dim="fixed").mean("fixed")
    return xr.concat(list(out.values()), dim="axis").assign_coords(axis=list(out))


def variance_partition(terms: xr.Dataset, dim: str) -> xr.Dataset:
    """
    Variance (ddof=0) across `dim` of E_mm - E_rr, split into the terms (COMPONENTS):
      Var(STATE) + Var(FUNCTION) + Var(INTERACTION)
      + 2 Cov(STATE, FUNCTION) + 2 Cov(STATE, INTERACTION) + 2 Cov(FUNCTION, INTERACTION)
    Against one reference, E_rr changes across datasets only through the support S, so
    this is Var(E_mm) when every dataset's support is complete. Var(E_mm) and Var(E_rr)
    are reported alongside. Returns `variance` and `fraction` (of Var(E_mm - E_rr)) along
    `component`.
    """
    def cov(a, b):
        return ((a - a.mean(dim)) * (b - b.mean(dim))).mean(dim)

    s, f, i = terms["STATE"], terms["FUNCTION"], terms["INTERACTION"]
    d = terms["E_mm"] - terms["E_rr"]
    comps = {
        "var_E_mm_minus_E_rr": cov(d, d),
        "var_STATE": cov(s, s),
        "var_FUNCTION": cov(f, f),
        "var_INTERACTION": cov(i, i),
        "2cov_STATE_FUNCTION": 2 * cov(s, f),
        "2cov_STATE_INTERACTION": 2 * cov(s, i),
        "2cov_FUNCTION_INTERACTION": 2 * cov(f, i),
        "var_E_mm": cov(terms["E_mm"], terms["E_mm"]),
        "var_E_rr": cov(terms["E_rr"], terms["E_rr"]),
    }
    variance = xr.concat(list(comps.values()), dim="component").assign_coords(component=list(comps))
    total = comps["var_E_mm_minus_E_rr"]
    fraction = (variance / total).where(total > 0)
    return xr.Dataset({"variance": variance, "fraction": fraction}, attrs={"ddof": 0})


def is_full_factorial(products: dict[str, Sequence[str]]) -> bool:
    """Whether the product combinations of the obs sources are every combination of the products."""
    combos = set(zip(*(list(v) for v in products.values())))
    full = set(itertools.product(*(dict.fromkeys(v) for v in products.values())))
    return combos == full
