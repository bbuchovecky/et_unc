"""
etunc.binning
=============
Statistics of annual mean ET in 2-D bins of climatological leaf area index
(LAI, y-axis) and aridity index (AI = Rn / L*P, x-axis), and their
post-processing. The full pipeline, from loading to binning, is described in
docs/binning-workflow.md and written out step by step in the scripts.

- `drop_et_outliers`: the plausibility filter on annual mean ET (step 6).
- `bin_stats`: statistics of every (year, gridcell) ET sample, binned by its
  gridcell's climatological LAI and AI (step 14), optionally per member.
- Post-processing: `pool_members`, `drop_zero_width_bins`, `frac_count`,
  `frac_valid`, `test_significance`, `ensemble_spread`.
"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import xarray as xr

from etunc.config import LAT_BNDS


# Order of the `stats` dimension returned by `bin_stats`
STATS = ("mean", "var_pop", "var_samp", "count", "count_pos")


# ------------------------------------------------------------------
# ET outlier filter
# ------------------------------------------------------------------

def drop_et_outliers(et: xr.DataArray, min_annual_et: float, label: str | None = None) -> xr.DataArray:
    """
    Annual mean ET `et` [W/m2] with gridcell-years below `min_annual_et` set
    to NaN (step 6). Strongly negative annual ET is not physical (e.g. GLEAM
    v4.3b at high latitudes in winter). Prints how many gridcell-years are dropped.
    """
    outlier = et < min_annual_et
    n = int(outlier.sum())
    # Before step 8 the field still covers Antarctica, where ice can give negative annual ET
    n_binned = int(outlier.sel(lat=slice(LAT_BNDS.start, LAT_BNDS.stop)).sum()) if "lat" in et.dims else n
    print(f"{label or et.name}: {n} gridcell-years with annual ET < {min_annual_et} W/m2 set to NaN"
          + (f" ({n_binned} within LAT_BNDS; min {float(et.min()):0.3g})" if n else ""))
    with xr.set_options(keep_attrs=True):
        return et.where(~outlier)


# ------------------------------------------------------------------
# Binning
# ------------------------------------------------------------------

def _bin_stats_flat(
    target: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    y_edges: np.ndarray,
    x_edges: np.ndarray,
) -> np.ndarray:
    """
    2-D bin statistics of `target` over (y, x) for flat arrays, via np.bincount.

    Samples are assigned with searchsorted(side='right'), so values equal to
    duplicate edges go to the last duplicate bin, and values outside the edges
    are clipped into the first/last bin. Samples with NaN in any of the three
    arrays are dropped.

    Returns
    -------
    np.ndarray, shape (len(STATS), n_y, n_x); NaN mean/variance in empty bins
    """
    n_y = len(y_edges) - 1
    n_x = len(x_edges) - 1

    # Bins are [lower, upper): a value on an interior edge goes to the bin above it. The clip
    # puts the maximum (== last edge, which side="right" places past the last bin) into the
    # last bin, and out-of-range values into the edge bins
    y_idx = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, n_y - 1)
    x_idx = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, n_x - 1)

    # NaN samples got a (clipped) index above; they are dropped here
    valid = np.isfinite(target) & np.isfinite(y) & np.isfinite(x)
    v = target[valid]
    lin_idx = y_idx[valid] * n_x + x_idx[valid]
    total_bins = n_y * n_x

    # Unweighted: every (year, gridcell) sample counts once, whatever its cell area
    count = np.bincount(lin_idx, minlength=total_bins).astype(np.float64)
    bin_sum = np.bincount(lin_idx, weights=v, minlength=total_bins)
    bin_sum2 = np.bincount(lin_idx, weights=v * v, minlength=total_bins)
    count_pos = np.bincount(lin_idx, weights=(v > 0).astype(np.float64), minlength=total_bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, bin_sum / count, np.nan)
        ex2 = np.where(count > 0, bin_sum2 / count, np.nan)

    # Var(X) = E[X^2] - E[X]^2, floored at 0 against round-off
    var_pop = np.maximum(ex2 - mean * mean, 0.0)
    # Unbiased sample variance (ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        var_samp = np.where(count > 1, var_pop * count / (count - 1.0), np.nan)

    return np.stack((mean, var_pop, var_samp, count, count_pos), axis=0).reshape(len(STATS), n_y, n_x)


def edges_and_attrs(edges: xr.DataArray | np.ndarray) -> tuple[np.ndarray, dict]:
    """Bin edges as a float array, and their attrs ({} for a plain array)."""
    if isinstance(edges, xr.DataArray):
        return np.asarray(edges.values, dtype=float), dict(edges.attrs)
    return np.asarray(edges, dtype=float), {}


def _quantile_labels(n: int) -> list[str]:
    lo = np.linspace(0, 100, n + 1)[:-1]
    hi = np.linspace(0, 100, n + 1)[1:]
    return [f"Q{a:.0f}-Q{b:.0f}" for a, b in zip(lo, hi)]


def _align_exact(*arrays: xr.DataArray) -> list[xr.DataArray]:
    try:
        return list(xr.align(*arrays, join="exact"))
    except ValueError as err:
        raise ValueError(
            "target, y_var, x_var (and mask) must share identical coordinates. Put them on one grid "
            "first, e.g. da.reindex_like(ref, method='nearest', tolerance=1e-3)."
        ) from err


def bin_stats(
    target: xr.DataArray,
    y_var: xr.DataArray,
    x_var: xr.DataArray,
    y_edges: xr.DataArray | np.ndarray,
    x_edges: xr.DataArray | np.ndarray,
    *,
    mask: xr.DataArray | None = None,
    member_dim: str | None = None,
    y_name: str = "lai",
    x_name: str = "ai",
    name: str | None = None,
    attrs: Mapping | None = None,
    verbose: bool = False,
) -> xr.DataArray:
    """
    Statistics of `target` in 2-D (y_var, x_var) bins.

    `y_var` and `x_var` are broadcast against `target`, so a climatological
    (lat, lon) LAI/AI field is repeated for every year of an annual mean ET
    field: every (year, gridcell) ET sample is binned by its gridcell's
    climatological LAI and AI. Coordinates must match exactly.

    Parameters
    ----------
    target : variable to bin, e.g. annual mean ET (year, lat, lon[, member])
    y_var, x_var : binning variables, e.g. climatological LAI and AI
    y_edges, x_edges : bin edges (from `pooled_bin_edges` or a plain array)
    mask : optional mask (True/1 = keep) applied to all three fields
    member_dim : if given and in the broadcast dims, statistics are computed
        separately for each member; otherwise all samples are pooled
    y_name, x_name : names of the binning variables, stored in attrs
    name : name of the output (default: target.name)
    attrs : extra attributes (e.g. source_id, time_period)

    Returns
    -------
    xr.DataArray, dims ([member_dim,] stats, y_bin, x_bin), with bin index
    coords `y_bin`/`x_bin` and per-bin `*_bin_lower`, `*_bin_upper`,
    `*_bin_center` (and `*_bin_label` for quantile edges).
    """
    y_edges, y_edge_attrs = edges_and_attrs(y_edges)
    x_edges, x_edge_attrs = edges_and_attrs(x_edges)
    n_y, n_x = len(y_edges) - 1, len(x_edges) - 1

    # join="exact" raises on any coordinate mismatch instead of silently reindexing to NaN
    if mask is not None:
        target, y_var, x_var, mask = _align_exact(target, y_var, x_var, mask)
        keep = mask == 1
        target, y_var, x_var = target.where(keep), y_var.where(keep), x_var.where(keep)
    else:
        target, y_var, x_var = _align_exact(target, y_var, x_var)

    # The key step: (lat, lon) LAI and AI are repeated over `year` (and `member`), so each
    # (year, gridcell) ET value is one sample. A gridcell's weight in its bin therefore
    # scales with its number of valid ET years
    tgt_b, y_b, x_b = xr.broadcast(target, y_var, x_var)
    dims = tgt_b.dims
    y_b, x_b = y_b.transpose(*dims), x_b.transpose(*dims)
    if verbose:
        print(f"target: {target.dims} {target.shape} -> {dims} {tgt_b.shape}")
        print(f"y_var: {y_var.dims} {y_var.shape}\nx_var: {x_var.dims} {x_var.shape}")

    per_member = member_dim is not None and member_dim in dims
    if per_member:
        order = (member_dim,) + tuple(d for d in dims if d != member_dim)
        tgt_b, y_b, x_b = (da.transpose(*order) for da in (tgt_b, y_b, x_b))

    tgt_np = np.asarray(tgt_b.values, dtype=float)
    y_np = np.asarray(y_b.values, dtype=float)
    x_np = np.asarray(x_b.values, dtype=float)

    if per_member:
        result = np.stack([
            _bin_stats_flat(tgt_np[m].ravel(), y_np[m].ravel(), x_np[m].ravel(), y_edges, x_edges)
            for m in range(tgt_np.shape[0])
        ])
        out_dims = (member_dim, "stats", "y_bin", "x_bin")
    else:
        # Without member_dim, the samples of all members (if any) are pooled into one set of stats
        result = _bin_stats_flat(tgt_np.ravel(), y_np.ravel(), x_np.ravel(), y_edges, x_edges)
        out_dims = ("stats", "y_bin", "x_bin")

    if verbose:
        n_valid = np.nansum(result[..., STATS.index("count"), :, :])
        print(f"samples binned: {n_valid:.3e} of {tgt_np.size:.3e}")

    coords = {
        "stats": list(STATS),
        "y_bin": np.arange(n_y),
        "x_bin": np.arange(n_x),
        "y_bin_lower": ("y_bin", y_edges[:-1]),
        "y_bin_upper": ("y_bin", y_edges[1:]),
        "y_bin_center": ("y_bin", 0.5 * (y_edges[:-1] + y_edges[1:])),
        "x_bin_lower": ("x_bin", x_edges[:-1]),
        "x_bin_upper": ("x_bin", x_edges[1:]),
        "x_bin_center": ("x_bin", 0.5 * (x_edges[:-1] + x_edges[1:])),
    }
    if y_edge_attrs.get("strategy") == "quantile":
        coords["y_bin_label"] = ("y_bin", _quantile_labels(n_y))
    if x_edge_attrs.get("strategy") == "quantile":
        coords["x_bin_label"] = ("x_bin", _quantile_labels(n_x))

    name = name or target.name or "target"
    out = xr.DataArray(
        result,
        name=name,
        dims=out_dims,
        coords=coords,
        attrs={
            "long_name": f"2-D binned statistics of {name}",
            "units": target.attrs.get("units", "unknown"),
            "y_variable": y_name,
            "x_variable": x_name,
            "y_strategy": y_edge_attrs.get("strategy", "unknown"),
            "x_strategy": x_edge_attrs.get("strategy", "unknown"),
            "y_bin_edges": y_edges,
            "x_bin_edges": x_edges,
            "n_y_bins": n_y,
            "n_x_bins": n_x,
            **(attrs or {}),
        },
    )
    if per_member:
        member_coords = {c: target[c] for c in target.coords if target[c].dims == (member_dim,)}
        out = out.assign_coords(member_coords)
    return out


def pool_members(bs: xr.DataArray) -> xr.DataArray:
    """Combine per-member bin statistics into the statistics of all members' samples pooled."""
    mean, var_pop, count, count_pos = (bs.sel(stats=s, drop=True) for s in ("mean", "var_pop", "count", "count_pos"))
    n = count.sum("member")
    # Count-weighted, so the result equals binning all members' samples together. A member
    # with an empty bin has a NaN mean and count 0; the sums skip the NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled_mean = (mean * count).sum("member") / n
        # Pooled E[X^2] from each member's var_pop + mean^2, then Var = E[X^2] - E[X]^2
        pooled_var = ((var_pop + mean**2) * count).sum("member") / n - pooled_mean**2
        pooled_var = pooled_var.clip(min=0)
        var_samp = xr.where(n > 1, pooled_var * n / (n - 1), np.nan)
    out = xr.concat(
        [pooled_mean.where(n > 0), pooled_var.where(n > 0), var_samp, n, count_pos.sum("member")], dim="stats",
    ).assign_coords(stats=list(STATS)).transpose("stats", "y_bin", "x_bin")
    out = out.drop_vars([c for c in out.coords if "member" in out[c].dims or c == "member_id"], errors="ignore")
    # Members named by member_id if present (CMIP6), else by the member labels
    names = bs["member_id"] if "member_id" in bs.coords else bs["member"]
    out.attrs = {**bs.attrs, "members": [str(m) for m in names.values]}
    return out.rename(bs.name)


def ensure_bin_coords(bs: xr.DataArray) -> xr.DataArray:
    """Add y_bin/x_bin index coords and lower/upper edge coords from attrs, if missing."""
    for ax in ("y", "x"):
        dim = f"{ax}_bin"
        if dim not in bs.dims:
            continue
        if dim not in bs.coords:
            bs = bs.assign_coords({dim: np.arange(bs.sizes[dim])})
        edges = bs.attrs.get(f"{ax}_bin_edges")
        if f"{ax}_bin_lower" not in bs.coords and edges is not None:
            edges = np.asarray(edges, dtype=float)
            # Only when no bins were dropped; the bin index values then index into `edges`
            if len(edges) - 1 == bs.sizes[dim]:
                idx = bs[dim].values
                bs = bs.assign_coords({
                    f"{ax}_bin_lower": (dim, edges[idx]),
                    f"{ax}_bin_upper": (dim, edges[idx + 1]),
                })
    return bs


# ------------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------------

def drop_zero_width_bins(bs: xr.DataArray) -> xr.DataArray:
    """
    Drop bins whose lower and upper edges are equal. These are always empty:
    duplicate quantile edges (e.g. from LAI = 0) put every tied value in the
    last duplicate bin.
    """
    bs = ensure_bin_coords(bs)
    for ax in ("y", "x"):
        dim = f"{ax}_bin"
        if f"{ax}_bin_lower" in bs.coords:
            # Exception: if the *last* edges are duplicated (a spike at the maximum), the clip in
            # `_bin_stats_flat` puts those maximum values in the zero-width last bin, which is
            # then dropped with its samples. The kept bins keep their original y_bin/x_bin
            # values. Needs 1-D (shared) edge coords
            keep = (bs[f"{ax}_bin_upper"] > bs[f"{ax}_bin_lower"]).values
            bs = bs.isel({dim: np.flatnonzero(keep)})
    return bs


def frac_count(bs: xr.DataArray) -> xr.DataArray:
    """Fraction of all samples (per member/source) falling in each bin."""
    count = bs.sel(stats="count")
    return count / count.sum(dim=["y_bin", "x_bin"])


def frac_valid(bs: xr.DataArray, dim: str) -> xr.DataArray:
    """Fraction of members/sources along `dim` with a non-NaN mean in each bin."""
    return bs.sel(stats="mean").notnull().sum(dim=dim) / bs.sizes[dim]


def test_significance(
    bs: xr.DataArray,
    alpha: float = 0.05,
    n_min: int = 10,
) -> xr.DataArray:
    """
    Two-sided t-test of the binned means against H0: mu = 0. A bin is also
    required to hold more than `n_min` samples to count as significant.
    """
    import scipy.stats as stats

    means = bs.sel(stats="mean")
    stds = np.sqrt(bs.sel(stats="var_samp"))
    ns = bs.sel(stats="count")

    # Treats every (year, gridcell) sample as independent. Successive years of a gridcell and
    # neighbouring gridcells are correlated, so the effective sample size is smaller than
    # `count` and significance is overstated. Bins with count <= 1 get a NaN statistic or
    # critical value, so they compare False (not significant)
    t_critical = xr.apply_ufunc(stats.t.ppf, 1 - alpha / 2, ns - 1)
    t_statistic = means / (stds / np.sqrt(ns))

    return (np.abs(t_statistic) > t_critical) & (ns > n_min)


def ensemble_spread(
    bs: xr.DataArray,
    dim: str,
    *,
    how: Literal["std", "var"] = "std",
    signif: xr.DataArray | None = None,
    min_frac: float | None = None,
) -> xr.DataArray:
    """
    Spread of the binned means across members/sources along `dim`.

    signif : keep only bins where this is True (e.g. from `test_significance`)
    min_frac : keep only bins where more than this fraction of members/sources
        have values (`frac_valid`)
    """
    mean = bs.sel(stats="mean")
    if signif is not None:
        mean = mean.where(signif)
    # xarray's std/var: ddof=0 (population spread) and NaN skipped, so a bin's spread uses
    # only the members/sources that have a value in it (see `min_frac`)
    spread = getattr(mean, how)(dim=dim)
    if min_frac is not None:
        spread = spread.where(frac_valid(bs, dim) > min_frac)
    return spread
