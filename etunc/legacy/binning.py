"""
etunc.legacy.binning
====================
Unmaintained and untested. Moved from `etunc.binning` when the binning
pipeline was restructured (docs/binning-workflow.md):

- `prepare_inputs`: annual mean ET and climatological LAI and AI of one
  dataset in one call. The prep scripts now write out these steps.
- `valid_area` / `common_area` with a target land mask. The maintained
  versions are in `etunc.grid`.
- `build_edges` / `pooled_bin_edges`: unweighted quantile edges with linear
  interpolation (every finite value counts once). The maintained edges are
  in `etunc.edges` (hierarchically weighted).
- `bin_stats_by_source`: `bin_stats` for a dict of models.
- `frac_count`: fraction of the samples in each bin.
"""

from __future__ import annotations

from typing import Literal, Mapping
import warnings

import numpy as np
import xarray as xr

from etunc.binning import bin_stats
from etunc.edges import compute_aridity_index
from etunc.legacy.temporal import Aggregation, aggregate


BinStrategy = Literal["quantile", "linear"]


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

    # ET that already has a `year` dim (e.g. from `temporal.annual_mean`) passes through
    # unchanged. Monthly ET goes through `compute_annual_mean`, which counts NaN months as 0
    out = {
        "et": aggregate(_prep(et), "year").rename("et"),
        "lai": aggregate(_prep(lai), lai_agg).rename("lai"),
        # Ratio of the climatological means, not the mean of annual ratios
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
    # ET only needs one valid year (its missing years stay NaN samples); LAI and AI must
    # be valid, and np.isfinite also drops the inf AI of zero-precipitation gridcells
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
        # Quantiles of gridcell values, not area-weighted: small high-latitude cells count
        # as much as large tropical ones. Linear interpolation, so edges can fall between values
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
    """One DataArray (keyed by its name) or a mapping of them, as a {label: DataArray} dict."""
    if isinstance(fields, xr.DataArray):
        return {fields.name or "field": fields}
    return dict(fields)


def finite_flat(da: xr.DataArray) -> np.ndarray:
    """Finite values of `da`, flattened over every dim."""
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

    # Every finite value counts once, so each field weighs in proportion to its size: a field
    # with more gridcells (finer grid, larger valid area), members or years pulls the edges
    # towards its own distribution
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
    # Attrs that differ between sources (e.g. per-source `attrs`) are dropped
    out = xr.concat(results, dim=dim, combine_attrs="drop_conflicts")
    return out.assign_coords({dim: list(inputs.keys())})


def frac_count(bs: xr.DataArray) -> xr.DataArray:
    """Fraction of all samples (per member/source) falling in each bin."""
    count = bs.sel(stats="count")
    return count / count.sum(dim=["y_bin", "x_bin"])
