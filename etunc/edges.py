"""
etunc.edges
===========
The binning variables and their bin edges (steps 11 and 13 of
docs/binning-workflow.md).

- `compute_aridity_index`: AI = Rn / (L*P) from climatological means.
- `hierarchical_weights`: one weight per (member, gridcell) value, so that
  every family, every source within a family and every member within a
  source counts equally.
- `pooled_bin_edges`: weighted quantile edges of the pooled values.
- `edges_path` / `load_bin_edges`: where `make_edges.py` saves the edges and
  how `bin_all.py` reads them back.

How the pooled edges are computed
---------------------------------
1. Values: one climatological LAI (or AI) value per (member, gridcell) on the
   common area. Years don't add values, and values are not area-weighted.
2. Weights: w = 1/N_families x 1/N_sources(family) x 1/N_members(source) x
   1/N_values(member). A source is an obs combination, a CMIP6 model, or a
   PPE/LENS member; only CMIP6 models can have several members.
3. Edges: np.quantile(values, q, weights=w, method="inverted_cdf") at
   q = 0, 1/n, ..., 1. Edge q is the smallest value at which the cumulative
   weight of the sorted values reaches q, so edges fall on data values, and
   the first and last edges are the pooled minimum and maximum.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping
import warnings

import numpy as np
import xarray as xr

from etunc.config import BIN_EDGES_ROOT


# Attributes of the LAI and AI bin edges
EDGE_ATTRS = {
    "lai": {"long_name": "leaf area index bin edges", "units": "m2/m2", "variable": "leaf area index"},
    "ai":  {"long_name": "aridity index bin edges", "units": "1", "variable": "aridity index (Rn/L*P)"},
}


# ------------------------------------------------------------------
# Aridity index
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
    # P = 0 gives inf (or NaN for 0/0), which `valid_area` and the binning drop.
    # Without `clip`, a negative mean Rn (e.g. ice, high latitudes) gives a negative AI
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


# ------------------------------------------------------------------
# Pooled quantile edges
# ------------------------------------------------------------------

def hierarchical_weights(fields: Mapping[str, xr.DataArray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Finite values of all `fields` ({family: (member, lat, lon)}, each with a
    `source` coord on `member`) flattened into one array, and the weight of
    each value:

        w = 1/N_families x 1/N_sources(family) x 1/N_members(source) x 1/N_values(member)

    so every family's weights sum to 1/N_families, every source's to an equal
    share of its family's, and so on. With one family this weighs its
    sources equally.
    """
    values, weights = [], []
    for family, da in fields.items():
        sources = da["source"].values
        unique_sources = list(dict.fromkeys(sources))
        for src in unique_sources:
            members = np.flatnonzero(sources == src)
            for m in members:
                x = np.asarray(da.isel(member=m).values, dtype=float).ravel()
                x = x[np.isfinite(x)]
                if x.size == 0:
                    # Its share is lost; np.quantile renormalizes the remaining weights
                    warnings.warn(f"{family}/{src}: a member has no finite values")
                    continue
                values.append(x)
                weights.append(np.full(x.size, 1 / (len(fields) * len(unique_sources) * len(members) * x.size)))
    return np.concatenate(values), np.concatenate(weights)


def pooled_bin_edges(
    values: np.ndarray,
    weights: np.ndarray,
    n_bins: int,
    *,
    name: str,
    attrs: Mapping | None = None,
    zero_warn_frac: float = 0.10,
) -> xr.DataArray:
    """
    `n_bins + 1` quantile edges of `values` with `weights` (from
    `hierarchical_weights`), at equally spaced weighted quantiles.

    Ties are kept: many equal values (e.g. LAI = 0) give duplicate edges. With
    searchsorted(side="right") in `bin_stats` the tied values all go to the
    last of the zero-width bins, and `drop_zero_width_bins` removes the empty ones.

    Returns
    -------
    xr.DataArray with dim `edge`, named `name`, with `attrs` added.
    """
    # The only np.quantile method that takes weights. Edge 0 is the minimum, edge n_bins the maximum
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1), weights=weights, method="inverted_cdf")

    zero_frac = weights[values == 0].sum() / weights.sum()
    if zero_frac > zero_warn_frac:
        warnings.warn(f"{name}: {zero_frac * 100:0.1f}% of the weight is on values equal to 0")
    print(f"{name}: {values.size:.3e} values, [{values.min():0.3g}, {values.max():0.3g}], "
          f"{zero_frac * 100:0.1f}% of weight at 0 -> edges {np.round(edges, 3)}")

    return xr.DataArray(
        edges,
        dims=["edge"],
        coords={"edge": np.arange(n_bins + 1)},
        name=name,
        attrs={
            "long_name": f"{name} bin edges",
            "strategy": "quantile",
            "quantile_method": "inverted_cdf",
            "n_bins": n_bins,
            **(attrs or {}),
        },
    )


# ------------------------------------------------------------------
# Saved edges
# ------------------------------------------------------------------

def edges_path(kind: str, variable: str, n_bins: int, grid: str, period: str) -> Path:
    """File of the `kind` edges ("pooled_all", "pooled_cmip", ...) of `variable` ("lai", "ai")."""
    return BIN_EDGES_ROOT / f"{kind}.{variable}_clim.{n_bins}_quantiles.{grid}.{period}.nc"


def load_bin_edges(kind: str, variable: str, n_bins: int, grid: str, period: str) -> xr.DataArray:
    """Edges saved by make_edges.py; raises FileNotFoundError if make_edges.py has not written them."""
    path = edges_path(kind, variable, n_bins, grid, period)
    if not path.exists():
        raise FileNotFoundError(f"No {kind} {variable} edges at {path}; run scripts/make_edges.py first")
    with xr.open_dataarray(path) as da:
        return da.load()
