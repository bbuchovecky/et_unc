"""
etunc.binning
=============
Bin evapotranspiration (ET) in a 2-D space of climatological leaf area index
(LAI, y-axis) and aridity index (AI = Rn / L*P, x-axis), and plot intermediate
diagnostics. Dataset-agnostic: works for CMIP6 models, CESM ensembles (FHIST
PPE, GOGA2, LENS2), and observations, as long as each dataset's fields share a
lat/lon grid.

Pipeline
--------
1. Load monthly fields and a land mask on one grid (``etunc.load``, or your own
   loader). Water fluxes should be energy fluxes [W/m2] (``etunc.units``).
2. Build the binning inputs per dataset/model (``prepare_inputs``):
   annual mean ET, climatological LAI, climatological AI.
3. Build bin edges pooled across every dataset that should share bins
   (``pooled_bin_edges``).
4. Compute per-bin statistics of ET (``bin_stats`` for one dataset, optionally
   per member; ``bin_stats_by_source`` for a dict of models).
5. Post-process and plot (``drop_zero_width_bins``, ``test_significance``,
   ``frac_valid``, ``ensemble_spread``; ``etunc.plotting``).

Example
-------
>>> import etunc.binning as binning
>>> import etunc.units as units
>>> inputs = {
...     sid: binning.prepare_inputs(
...         et=d["evspsbl"], lai=d["lai"], precip=d["pr"],
...         rn=units.net_radiation_cmip(d["rsds"], d["rsus"], d["rlds"], d["rlus"]),
...         mask=d["mask"], time_slice=slice("1995-01", "2014-12"),
...     )
...     for sid, d in cmip.items()
... }
>>> y_edges = binning.pooled_bin_edges({s: i["lai"] for s, i in inputs.items()}, 15, name="lai")
>>> x_edges = binning.pooled_bin_edges({s: i["ai"] for s, i in inputs.items()}, 15, name="ai")
>>> bs = binning.bin_stats_by_source(inputs, y_edges, x_edges, dim="source_id")
>>> import etunc.plotting as plotting
>>> plotting.plot_bin_summary(binning.drop_zero_width_bins(bs), dim="source_id")
"""

from __future__ import annotations

from typing import Literal, Mapping
import warnings

import numpy as np
import xarray as xr

from etunc.temporal import Aggregation, aggregate


# Order of the `stats` dimension returned by `bin_stats`
STATS = ("mean", "var_pop", "var_samp", "count", "count_pos")


BinStrategy = Literal["quantile", "linear"]

# Attributes of the LAI and AI bin edges written by the drivers
EDGE_ATTRS = {
    "lai": {"long_name": "leaf area index bin edges", "units": "m2/m2", "variable": "leaf area index"},
    "ai":  {"long_name": "aridity index bin edges", "units": "1", "variable": "aridity index (Rn/L*P)"},
}


# ------------------------------------------------------------------
# Binning inputs
# ------------------------------------------------------------------

def compute_aridity_index(
    precip_wm2: xr.DataArray,
    rn: xr.DataArray,
    clip: bool = False,
) -> xr.DataArray:
    """
    Compute the aridity index, AI = Rn / (L*P).

    Parameters
    ----------
    precip_wm2 : xr.DataArray
        Mean precipitation expressed as an energy flux, L*P [W/m2].
    rn : xr.DataArray
        Mean net surface radiation [W/m2], used as a proxy for
        PET (PET ~ Rn) per the Budyko framework.
    clip : bool, optional
        If True, floor `rn` at zero before dividing.
    """
    if clip:
        rn = rn.clip(min=0)
    ai = rn / precip_wm2
    ai = ai.rename("ai")
    ai.attrs = {
        "long_name": "Budyko dryness index (Rn / L*P)",
        "units": "1",
        "description": (
            "computed from mean net surface radiation and "
            f"precipitation{' (Rn floor at zero)' if clip else ''}"
        ),
    }
    return ai


def prepare_inputs(
    et: xr.DataArray,
    lai: xr.DataArray,
    precip: xr.DataArray,
    rn: xr.DataArray,
    *,
    mask: xr.DataArray | None = None,
    time_slice: slice | None = None,
    lai_agg: Aggregation = "clim",
    clip_rn: bool = False,
) -> dict[str, xr.DataArray]:
    """
    Build the binning inputs for one dataset from monthly fields on one grid.

    Parameters
    ----------
    et, lai, precip, rn : monthly fields with a `time` dim (precip and rn may
        already be time means). ET, precip and rn in W/m2.
    mask : land mask (True/1 = keep) applied to every output.
    time_slice : applied to each input that has a `time` dim before aggregating.
        Pre-slice the inputs yourself if they need different periods (e.g. obs).
    lai_agg : how LAI is aggregated (see `aggregate`).
    clip_rn : floor Rn at zero in the aridity index.

    Returns
    -------
    {"et": annual mean ET, "lai": aggregated LAI, "ai": climatological AI}
    """
    def _prep(da):
        if time_slice is not None and "time" in da.dims:
            da = da.sel(time=time_slice)
        return da

    out = {
        "et": aggregate(_prep(et), "year").rename("et"),
        "lai": aggregate(_prep(lai), lai_agg).rename("lai"),
        "ai": compute_aridity_index(
            aggregate(_prep(precip), "clim"), aggregate(_prep(rn), "clim"), clip=clip_rn
        ),
    }
    if mask is not None:
        out = {k: v.where(mask == 1) for k, v in out.items()}
    return out


def valid_area(inputs: Mapping[str, xr.DataArray], member_dim: str | None = None) -> xr.DataArray:
    """
    Gridcells where ET (in any year), LAI and AI are all valid; with
    `member_dim`, valid in every member along it.
    """
    valid = inputs["et"].notnull().any("year") & inputs["lai"].notnull() & np.isfinite(inputs["ai"])
    return valid.all(member_dim) if member_dim is not None else valid


def common_area(
    inputs: Mapping[str, Mapping[str, xr.DataArray]],
    mask: xr.DataArray,
    member_dim: str | None = None,
    label_width: int = 16,
) -> xr.DataArray:
    """
    Land gridcells (`mask` == 1) valid (`valid_area`) in every dataset of
    `inputs` ({label: {"et", "lai", "ai"}}, e.g. every combination or model),
    so that all datasets cover the same area. The mask is static: ET years
    that are NaN inside this area stay NaN, so per-year data availability is
    kept. Prints each dataset's valid gridcells, labels padded to `label_width`.
    """
    area = mask == 1
    for label, inp in inputs.items():
        valid = valid_area(inp, member_dim)
        print(f"{label:{label_width}}: {int(valid.sum()):.4e} valid gridcells")
        area = area & valid
    print(f"common area: {int(area.sum()):.4e} of {int((mask == 1).sum()):.4e} land gridcells")
    return area.rename("area_mask")


# ------------------------------------------------------------------
# Bin edges
# ------------------------------------------------------------------

def build_edges(
    values: np.ndarray,
    n_bins: int,
    strategy: BinStrategy = "quantile",
    value_range: tuple[float, float] | None = None,
    collapse_duplicates: bool = False,
) -> np.ndarray:
    """
    Build `n_bins + 1` bin edges from an array of values (NaN tolerated).

    strategy : "quantile" -> edges at equally spaced quantiles of the finite data
               "linear"   -> edges linearly spaced over `value_range`
                             (default: finite data min/max)
    collapse_duplicates : for "quantile", remove duplicate edges produced by
        tied values (e.g. many LAI = 0), reducing the number of bins.

    Duplicate quantile edges are kept by default. With searchsorted(side='right')
    in the binning, a spike value (e.g. LAI = 0) goes to the *last* of the
    zero-width bins at that value, so the spike lands in one bin and the bins
    before it stay empty (see `drop_zero_width_bins`).
    """
    values = np.asarray(values).ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Binning variable contains no finite values.")

    if strategy == "quantile":
        edges = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1))
        if collapse_duplicates:
            edges = np.unique(edges)
            if edges.size < 2:
                raise ValueError(
                    "Quantile edges collapsed to a single value; cannot form bins. "
                    "Use fewer bins or disable collapse_duplicates."
                )
    elif strategy == "linear":
        lo, hi = value_range if value_range is not None else (finite.min(), finite.max())
        edges = np.linspace(float(lo), float(hi), n_bins + 1)
    else:
        raise ValueError(f"Unknown bin strategy {strategy!r}. Choose 'quantile' or 'linear'.")

    return edges


def as_field_dict(fields: xr.DataArray | Mapping[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    if isinstance(fields, xr.DataArray):
        return {fields.name or "field": fields}
    return dict(fields)


def finite_flat(da: xr.DataArray) -> np.ndarray:
    flat = np.asarray(da.values, dtype=float).ravel()
    return flat[np.isfinite(flat)]


def pooled_bin_edges(
    fields: xr.DataArray | Mapping[str, xr.DataArray],
    n_bins: int,
    *,
    name: str,
    strategy: BinStrategy = "quantile",
    value_range: tuple[float, float] | None = None,
    collapse_duplicates: bool = False,
    attrs: Mapping | None = None,
    zero_warn_frac: float = 0.10,
    verbose: bool = True,
) -> xr.DataArray:
    """
    Bin edges from the finite values of all `fields` pooled together (over
    every dim, including member/time), so all datasets share bin definitions.
    Mask the fields beforehand (`prepare_inputs` does).

    Parameters
    ----------
    fields : one DataArray or {label: DataArray}, e.g. {source_id: lai_clim}
    n_bins : number of bins requested
    name : name of the returned edges DataArray (e.g. "lai", "ai")
    attrs : extra attributes (e.g. units, long_name, time_period)
    zero_warn_frac : warn when more than this fraction of a field is exactly 0

    Returns
    -------
    xr.DataArray with dim `edge` (n_bins + 1 values unless duplicates collapsed)
    """
    fields = as_field_dict(fields)

    flat_list = []
    for label, da in fields.items():
        flat = finite_flat(da)
        flat_list.append(flat)
        if flat.size == 0:
            warnings.warn(f"{label}: no finite values")
            continue
        zero_frac = np.sum(flat == 0) / flat.size
        if verbose:
            print(
                f"{label:20}: {da.dims} {da.shape} -> {flat.size:.3e} finite "
                f"[{flat.min():0.3e}, {flat.max():0.3e}] zeros={zero_frac * 100:0.3f}%"
            )
        if zero_frac > zero_warn_frac:
            warnings.warn(f"{label}: {zero_frac * 100:0.3f}% of values are 0 (> {zero_warn_frac * 100:g}%)")

    pooled = np.concatenate(flat_list)
    edges = build_edges(pooled, n_bins, strategy, value_range, collapse_duplicates)
    n_eff = len(edges) - 1
    if verbose:
        print(f"pooled: {pooled.size:.3e} values -> {n_eff} bins\nedges: {edges}")

    return xr.DataArray(
        data=edges,
        dims=["edge"],
        coords=dict(edge=np.arange(n_eff + 1)),
        name=name,
        attrs={
            "long_name": f"{name} bin edges",
            "strategy": strategy,
            "n_bins": n_eff,
            "n_bins_requested": n_bins,
            "collapse_duplicate_quantile_bins": int(collapse_duplicates),
            "pool_edges": 1,
            "pooled_sources": list(fields.keys()),
            **(attrs or {}),
        },
    )


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

    y_idx = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, n_y - 1)
    x_idx = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, n_x - 1)

    valid = np.isfinite(target) & np.isfinite(y) & np.isfinite(x)
    v = target[valid]
    lin_idx = y_idx[valid] * n_x + x_idx[valid]
    total_bins = n_y * n_x

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

    if mask is not None:
        target, y_var, x_var, mask = _align_exact(target, y_var, x_var, mask)
        keep = mask == 1
        target, y_var, x_var = target.where(keep), y_var.where(keep), x_var.where(keep)
    else:
        target, y_var, x_var = _align_exact(target, y_var, x_var)

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


def bin_stats_by_source(
    inputs: Mapping[str, Mapping[str, xr.DataArray]],
    y_edges: xr.DataArray | np.ndarray,
    x_edges: xr.DataArray | np.ndarray,
    *,
    dim: str = "source_id",
    target: str = "et",
    y: str = "lai",
    x: str = "ai",
    masks: Mapping[str, xr.DataArray] | None = None,
    verbose: bool = True,
    **kwargs,
) -> xr.DataArray:
    """
    `bin_stats` for each source (e.g. CMIP6 model) on its own grid, concatenated along `dim`.

    Parameters
    ----------
    inputs : {source: {"et": ..., "lai": ..., "ai": ...}}, e.g. from `prepare_inputs`
    target, y, x : keys of the target and binning variables in each inputs[source]
    masks : optional {source: mask}
    **kwargs : passed to `bin_stats` (e.g. y_name, x_name, name, attrs)
    """
    results = []
    for src, fields in inputs.items():
        bs = bin_stats(
            fields[target], fields[y], fields[x], y_edges, x_edges,
            mask=None if masks is None else masks[src],
            **kwargs,
        )
        if verbose:
            print(f"{src:20}: {int(bs.sel(stats='count').sum()):.3e} samples binned")
        results.append(bs)
    out = xr.concat(results, dim=dim, combine_attrs="drop_conflicts")
    return out.assign_coords({dim: list(inputs.keys())})


def pool_members(bs: xr.DataArray) -> xr.DataArray:
    """Combine per-member bin statistics into the statistics of all members' samples pooled."""
    mean, var_pop, count, count_pos = (bs.sel(stats=s, drop=True) for s in ("mean", "var_pop", "count", "count_pos"))
    n = count.sum("member")
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled_mean = (mean * count).sum("member") / n
        pooled_var = ((var_pop + mean**2) * count).sum("member") / n - pooled_mean**2
        pooled_var = pooled_var.clip(min=0)
        var_samp = xr.where(n > 1, pooled_var * n / (n - 1), np.nan)
    out = xr.concat(
        [pooled_mean.where(n > 0), pooled_var.where(n > 0), var_samp, n, count_pos.sum("member")], dim="stats",
    ).assign_coords(stats=list(STATS)).transpose("stats", "y_bin", "x_bin")
    out = out.drop_vars([c for c in out.coords if "member" in out[c].dims or c == "member_id"], errors="ignore")
    out.attrs = {**bs.attrs, "members": list(bs["member_id"].values)}
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
    spread = getattr(mean, how)(dim=dim)
    if min_frac is not None:
        spread = spread.where(frac_valid(bs, dim) > min_frac)
    return spread
