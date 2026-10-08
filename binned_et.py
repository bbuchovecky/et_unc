"""
binned_et.py
============
Bin evapotranspiration (ET) in a 2-D space of climatological leaf area index
(LAI, y-axis) and aridity index (AI = Rn / L*P, x-axis), and plot intermediate
diagnostics. Dataset-agnostic: works for CMIP6 models, CESM ensembles (FHIST
PPE, GOGA2, LENS2), and observations, as long as each dataset's fields share a
lat/lon grid.

Pipeline
--------
1. Load monthly fields and a land mask on one grid (``load_*`` helpers, or
   your own loader). Water fluxes should be energy fluxes [W/m2]
   (``convert_units``).
2. Build the binning inputs per dataset/model (``prepare_inputs``):
   annual mean ET, climatological LAI, climatological AI.
3. Build bin edges pooled across every dataset that should share bins
   (``pooled_bin_edges``).
4. Compute per-bin statistics of ET (``bin_stats`` for one dataset, optionally
   per member; ``bin_stats_by_source`` for a dict of models).
5. Post-process and plot (``drop_zero_width_bins``, ``test_significance``,
   ``frac_valid``, ``ensemble_spread``, ``plot_*``).

Example
-------
>>> import binned_et as be
>>> inputs = {
...     sid: be.prepare_inputs(
...         et=d["evspsbl"], lai=d["lai"], precip=d["pr"],
...         rn=be.net_radiation_cmip(d["rsds"], d["rsus"], d["rlds"], d["rlus"]),
...         mask=d["mask"], time_slice=slice("1995-01", "2014-12"),
...     )
...     for sid, d in cmip.items()
... }
>>> y_edges = be.pooled_bin_edges({s: i["lai"] for s, i in inputs.items()}, 15, name="lai")
>>> x_edges = be.pooled_bin_edges({s: i["ai"] for s, i in inputs.items()}, 15, name="ai")
>>> bs = be.bin_stats_by_source(inputs, y_edges, x_edges, dim="source_id")
>>> be.plot_bin_summary(be.drop_zero_width_bins(bs), dim="source_id")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import regionmask as regmask
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs

import load_cesm as lc


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

LATENT_HEAT_VAPORIZATION = 2.45e6  # J/kg
LIQ_WATER_DENSITY = 1e3            # kg/m3

LAT_BNDS = slice(-58, 90)
LF_THRESH = 0.5  # gridcell land fraction threshold
ILAMB_ROOT = Path("/glade/campaign/univ/uwas0155/obs/ilamb")

PROJECTION = ccrs.PlateCarree()
DPI = 120

# Order of the `stats` dimension returned by `bin_stats`
STATS = ("mean", "var_pop", "var_samp", "count", "count_pos")

Aggregation = Literal["mon", "year", "clim", "year_max", "clim_max"]
BinStrategy = Literal["quantile", "linear"]


# ------------------------------------------------------------------
# General helpers
# ------------------------------------------------------------------

def format_time_period(time_slice: slice) -> str:
    """slice("1950-01", "2014-12") -> "195001-201412", for file names."""
    return f"{time_slice.start.replace('-', '')}-{time_slice.stop.replace('-', '')}"


def to_yyyymm(time) -> str:
    time_raw = time.values
    if isinstance(time_raw, np.datetime64):
        return pd.Timestamp(time_raw).strftime("%Y%m")
    elif isinstance(time_raw, np.ndarray):
        return time.item().strftime("%Y%m")
    raise TypeError(f"Unsupported type {type(time)!r} for time")


def get_one_mid(da: xr.DataArray) -> str:
    """Member id of a single-member CMIP DataArray, for file names."""
    if ("member" in da.dims):
        if ("member_id" in da.coords):
            return str(da.member_id[0].item())
    return "onemember"


def safe_squeeze(da: xr.DataArray, dim: str, drop: bool = True) -> xr.DataArray:
    if dim in da.dims:
        return da.squeeze(dim=dim, drop=drop)
    return da


def check_coords(da: xr.DataArray, coords: Iterable[str]) -> bool:
    """Check that da has non-empty coords."""
    for coord in coords:
        if coord not in da.coords:
            return False
        if da[coord].ndim == 0:
            continue  # scalar coord counts as present
        if len(da[coord]) == 0:
            return False
    return True


def equal_coords(
    a: xr.DataArray | xr.Dataset,
    b: xr.DataArray | xr.Dataset,
    coords: Iterable[str],
    atol: float = 1e-3,
) -> bool:
    """Check that a and b have the same coords (numeric coords within atol)."""
    if not check_coords(a, coords) or not check_coords(b, coords):
        return False
    for crd in coords:
        if a[crd].shape != b[crd].shape:
            return False
        if np.issubdtype(a[crd].dtype, np.number):
            if not np.allclose(a[crd], b[crd], atol=atol):
                return False
    return True


def check_same_grid(
    da: xr.DataArray,
    ref: xr.DataArray | xr.Dataset,
    label: str,
    atol: float = 1e-3,
) -> None:
    """Raise if `da` does not share its lat/lon grid with `ref`, rather than silently reindex."""
    if da.sizes.get("lat") != ref.sizes.get("lat") or da.sizes.get("lon") != ref.sizes.get("lon"):
        raise ValueError(
            f"{label}: grid shape lat={da.sizes.get('lat')}, lon={da.sizes.get('lon')} does not match "
            f"the reference grid lat={ref.sizes.get('lat')}, lon={ref.sizes.get('lon')}."
        )
    if not equal_coords(da, ref, ("lat", "lon"), atol=atol):
        raise ValueError(f"{label}: lat/lon values differ from the reference grid by more than {atol}.")


def compute_cell_area(ds: xr.Dataset | xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Grid cell area and land grid cell area [m2], using ILAMB's CellAreas and
    the Natural Earth land mask. Uses `lat_bounds`/`lon_bounds` if present.
    """
    from ILAMB import ilamblib

    land = regmask.defined_regions.natural_earth_v5_1_2.land_50

    if "lat_bounds" in ds and "lon_bounds" in ds:
        method = "bounds"
        lat_bounds = ds["lat_bounds"].values
        lon_bounds = ds["lon_bounds"].values
    else:
        method = "coords"
        lat_bounds = None
        lon_bounds = None

    area = ilamblib.CellAreas(
        lat=ds["lat"].values,
        lon=ds["lon"].values,
        lat_bnds=lat_bounds,
        lon_bnds=lon_bounds,
    )

    area = xr.DataArray(area, dims=["lat", "lon"], coords={"lat": ds["lat"], "lon": ds["lon"]})
    area.attrs["units"] = "m2"
    area.attrs["long_name"] = "grid cell area"
    area.attrs["method"] = method

    mask = xr.where(np.isnan(land.mask(lon_or_obj=area.lon, lat=area.lat)), 0, 1)
    la = area * mask
    la.attrs["units"] = "m2"
    la.attrs["long_name"] = "land grid cell area"
    la.attrs["method"] = method

    return area, la


def mask_greenland(landfrac: xr.DataArray, lf_thresh: float = LF_THRESH) -> xr.DataArray:
    """Land mask: True where landfrac > lf_thresh, excluding Greenland/Iceland (AR6 region 0)."""
    mask = regmask.defined_regions.ar6.land.mask(landfrac.lon, landfrac.lat)
    return xr.where((mask == 0) & (landfrac > lf_thresh), False, landfrac > lf_thresh)


def convert_units(v: str, da: xr.DataArray, verbose: bool = False) -> xr.DataArray:
    """Convert water fluxes to their energy equivalent [W/m2], keyed on variable name `v`."""

    # CESM2: precip from m/s -> W/m2
    if v in ("PRECT_calculated_month_1", "PRECT_month_1", "PRECT_calculated", "PRECT"):
        if verbose:
            print("Converting units from m/s -> W/m2")
        da = da * LATENT_HEAT_VAPORIZATION * LIQ_WATER_DENSITY
        da.attrs["units"] = "W/m2"

    # CMIP6, ERA5, ILAMB: precip and et from kg/m2/s -> W/m2
    if v in ("pr", "evspsbl", "mtpr", "mer", "et"):
        if verbose:
            print("Converting units from kg/m2/s -> W/m2")
        da = da * LATENT_HEAT_VAPORIZATION
        da.attrs["units"] = "W/m2"

    # ERA5: et sign convention
    if v == "mer":
        da = -1 * da

    return da


# Multiplicative factors to W/m2, keyed on normalized (lowercase, single-spaced) unit strings
# as they appear in the ILAMB evspsbl and hfls files
_UNITS_TO_WM2 = {
    "w m-2": 1.0,
    "w/m2": 1.0,
    "w/m^2": 1.0,
    "watt m-2": 1.0,
    "watt/m2": 1.0,
    "kg/m2/s": LATENT_HEAT_VAPORIZATION,
    "kg m-2 s-1": LATENT_HEAT_VAPORIZATION,
    "mj m-2 day-1": 1e6 / 86400,
}


def latent_heat_to_wm2(da: xr.DataArray) -> xr.DataArray:
    """Convert an ET [kg/m2/s] or latent heat [W/m2, MJ/m2/day] flux to W/m2, keyed on `da.attrs["units"]`."""
    units = da.attrs.get("units", "")
    key = " ".join(units.lower().split())
    if key not in _UNITS_TO_WM2:
        raise ValueError(f"Cannot convert units {units!r} of {da.name!r} to W/m2")
    out = da * _UNITS_TO_WM2[key]
    out.attrs = {**da.attrs, "units": "W/m2"}
    return out


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------

def compute_annual_mean(da: xr.DataArray) -> xr.DataArray:
    """Days-in-month weighted annual mean of a monthly field; `time` -> `year`."""
    days_in_month = da.time.dt.days_in_month
    weights = days_in_month.groupby('time.year') / days_in_month.groupby('time.year').sum()
    with xr.set_options(keep_attrs=True):
        return (da * weights).groupby('time.year').sum()


def aggregate(da: xr.DataArray, how: Aggregation = "clim") -> xr.DataArray:
    """
    Temporal aggregation of a monthly field.

    how : "mon"      -> unchanged
          "year"     -> annual mean (`year` dim)
          "clim"     -> time mean of the annual means
          "year_max" -> annual maximum of monthly values (`year` dim)
          "clim_max" -> maximum of the mean seasonal cycle

    A field without `time` is assumed to be aggregated already: "clim" then
    averages over `year` if present and otherwise returns `da` unchanged.
    """
    if "time" not in da.dims:
        if how == "clim":
            return da.mean("year", keep_attrs=True) if "year" in da.dims else da
        if how == "year" and "year" in da.dims:
            return da
        raise ValueError(f"aggregate(how={how!r}) needs a monthly `time` dimension; got {da.dims}")

    if how == "mon":
        return da
    if how == "year":
        return compute_annual_mean(da)
    if how == "clim":
        return compute_annual_mean(da).mean("year", keep_attrs=True)
    if how == "year_max":
        return da.groupby("time.year").max(keep_attrs=True)
    if how == "clim_max":
        return da.groupby("time.month").mean(keep_attrs=True).max("month", keep_attrs=True)
    raise ValueError(f"Unknown aggregation {how!r}")


def net_radiation_cmip(
    rsds: xr.DataArray,
    rsus: xr.DataArray,
    rlds: xr.DataArray,
    rlus: xr.DataArray,
) -> xr.DataArray:
    """CMIP6 net surface radiation [W/m2]; all four fluxes are positive in their named direction."""
    rn = rsds - rsus + rlds - rlus
    rn.attrs = {"units": "W/m2", "long_name": "net surface radiation"}
    return rn.rename("rn")


def net_radiation_cesm(fsns: xr.DataArray, flns: xr.DataArray) -> xr.DataArray:
    """CESM net surface radiation [W/m2]; FSNS is net SW down, FLNS is net LW *up*."""
    rn = fsns - flns
    rn.attrs = {"units": "W/m2", "long_name": "net surface radiation"}
    return rn.rename("rn")


def net_radiation_era5(msnswrf: xr.DataArray, msnlwrf: xr.DataArray) -> xr.DataArray:
    """ERA5 net surface radiation [W/m2]; both net fluxes are positive downward."""
    rn = msnswrf + msnlwrf
    rn.attrs = {"units": "W/m2", "long_name": "net surface radiation"}
    return rn.rename("rn")


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


def _as_field_dict(fields: xr.DataArray | Mapping[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    if isinstance(fields, xr.DataArray):
        return {fields.name or "field": fields}
    return dict(fields)


def _finite_flat(da: xr.DataArray) -> np.ndarray:
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
    fields = _as_field_dict(fields)

    flat_list = []
    for label, da in fields.items():
        flat = _finite_flat(da)
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


def _edges_and_attrs(edges: xr.DataArray | np.ndarray) -> tuple[np.ndarray, dict]:
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
    y_edges, y_edge_attrs = _edges_and_attrs(y_edges)
    x_edges, x_edge_attrs = _edges_and_attrs(x_edges)
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


def open_bin_stats(path: str | Path, var: str | None = None) -> xr.DataArray:
    """
    Open a saved bin-stats file (from `bin_stats` or the older notebook
    version) and ensure it has the bin coords used by the functions below.
    """
    ds = xr.open_dataset(path)
    if var is None:
        if len(ds.data_vars) != 1:
            raise ValueError(f"{path} has several variables {list(ds.data_vars)}; pass `var`.")
        var = next(iter(ds.data_vars))
    return _ensure_bin_coords(ds[var])


def _ensure_bin_coords(bs: xr.DataArray) -> xr.DataArray:
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
    bs = _ensure_bin_coords(bs)
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


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def _finish(fig, fout: str | Path | None):
    """Save and close `fig` if `fout` is given; otherwise leave it open for display."""
    if fout is not None:
        fout = Path(fout)
        fout.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fout, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return fig


def add_gridlines(ax, lat_bnds: slice = LAT_BNDS):
    x_gls = np.arange(-180, 181, 30)
    y_gls = np.arange(-90, 91, 30)
    y_gls = y_gls[(y_gls >= lat_bnds.start) & (y_gls <= lat_bnds.stop)]
    ax.gridlines(
        draw_labels=False,
        xlocs=x_gls,
        ylocs=y_gls,
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )


def _map_ax(ax, lat_bnds: slice):
    ax.coastlines(color="k", lw=0.8)
    ax.set_extent((-180, 180, lat_bnds.start, lat_bnds.stop), crs=PROJECTION)
    add_gridlines(ax, lat_bnds)


def _spatial_mean_over_other_dims(da: xr.DataArray) -> xr.DataArray:
    other = [d for d in da.dims if d not in ("lat", "lon")]
    return da.mean(other) if other else da


def quick_map(
    da: xr.DataArray,
    fout: str | Path | None = None,
    title: str = "",
    robust: bool = True,
    vmin: float | None = 0,
    lat_bnds: slice = LAT_BNDS,
    **kwargs,
):
    """Map of a (lat, lon) field; non-spatial dims are averaged. Saved and closed if `fout` is given."""
    fig, ax = plt.subplots(figsize=(8, 3), layout="constrained", subplot_kw={"projection": PROJECTION})
    kwargs.setdefault("cmap", "viridis")
    _spatial_mean_over_other_dims(da).plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(), vmin=vmin, robust=robust, **kwargs
    )
    ax.set_title(title)
    _map_ax(ax, lat_bnds)
    return _finish(fig, fout)


def plot_input_maps(
    inputs: Mapping[str, xr.DataArray],
    title: str = "",
    fout: str | Path | None = None,
    lat_bnds: slice = LAT_BNDS,
    cmaps: Mapping[str, str] | None = None,
):
    """
    One map per binning input (e.g. the dict from `prepare_inputs`); non-spatial
    dims are averaged. Useful to check masking, units and grids before binning.
    """
    cmaps = {"et": "YlGnBu", "lai": "Greens", "ai": "BrBG_r", **(cmaps or {})}
    n = len(inputs)
    fig, axs = plt.subplots(
        n, 1, figsize=(8, 3 * n), layout="constrained", squeeze=False,
        subplot_kw={"projection": PROJECTION},
    )
    for ax, (key, da) in zip(axs[:, 0], inputs.items()):
        _spatial_mean_over_other_dims(da).plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(), robust=True, cmap=cmaps.get(key, "viridis"),
            cbar_kwargs={"label": f"{key} [{da.attrs.get('units', '?')}]"},
        )
        ax.set_title(key)
        _map_ax(ax, lat_bnds)
    fig.suptitle(title)
    return _finish(fig, fout)


def plot_input_hist(
    fields: xr.DataArray | Mapping[str, xr.DataArray],
    edges: xr.DataArray | np.ndarray | None = None,
    *,
    bins: int | np.ndarray = 100,
    value_range: tuple[float, float] | None = None,
    log: bool = False,
    ax=None,
    title: str = "",
    fout: str | Path | None = None,
):
    """
    Density histograms of each field's finite values, with bin edges as
    vertical lines. Use to compare e.g. LAI distributions across models
    against the pooled edges.
    """
    fields = _as_field_dict(fields)
    flats = {k: _finite_flat(v) for k, v in fields.items()}
    if value_range is None:
        pooled = np.concatenate(list(flats.values()))
        value_range = tuple(np.quantile(pooled, [0, 0.995]))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    else:
        fig = ax.figure
    for label, flat in flats.items():
        ax.hist(flat, bins=bins, range=value_range, density=True, histtype="step", lw=1, label=label)
    if edges is not None:
        for e in _edges_and_attrs(edges)[0]:
            if value_range[0] <= e <= value_range[1]:
                ax.axvline(e, color="k", lw=0.6, ls="--", alpha=0.6)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel("density")
    ax.set_title(title)
    if len(flats) <= 12:
        ax.legend(fontsize=7, ncols=2)
    return _finish(fig, fout)


def plot_edges(edges: xr.DataArray, fout: str | Path | None = None, title: str = ""):
    """Bin edges vs edge index; bottom panel omits the last (maximum) edge."""
    fig, axs = plt.subplots(2, 1, layout="constrained")
    edges.plot(ax=axs[0], marker="o")
    edges[:-1].plot(ax=axs[1], marker="o")
    for ax in axs:
        ax.set_xlim(0, len(edges))
    axs[0].set_title(title)
    return _finish(fig, fout)


def plot_edges_compare(
    edges: Mapping[str, xr.DataArray],
    *,
    drop_last: bool = True,
    ax=None,
    title: str = "",
    fout: str | Path | None = None,
):
    """Overlay several sets of bin edges, e.g. {"FPPE": ..., "CMIP6": ...} or LAI aggregations."""
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = ax.figure
    markers = "o^sPdvX*"
    for i, (label, e) in enumerate(edges.items()):
        (e[:-1] if drop_last else e).plot(ax=ax, marker=markers[i % len(markers)], alpha=0.75, label=label)
    ax.legend()
    ax.set_title(title)
    return _finish(fig, fout)


def _edge_ticks(ax, da: xr.DataArray, fmt: str):
    for ax_name, dim, set_ticks in (("x", "x_bin", ax.set_xticks), ("y", "y_bin", ax.set_yticks)):
        lower, upper = f"{ax_name}_bin_lower", f"{ax_name}_bin_upper"
        if lower not in da.coords:
            continue
        idx = da[dim].values
        pos = np.append(idx - 0.5, idx[-1] + 0.5)
        vals = np.append(da[lower].values, da[upper].values[-1])
        set_ticks(pos, [fmt.format(v) for v in vals], fontsize=7, rotation=90 if ax_name == "x" else 0)


def plot_bin_field(
    da: xr.DataArray,
    *,
    ax=None,
    hatch: xr.DataArray | None = None,
    edge_ticks: bool = True,
    fmt: str = "{:.2g}",
    title: str = "",
    fout: str | Path | None = None,
    **kwargs,
):
    """
    Heatmap of a 2-D (y_bin, x_bin) field, e.g. bs.sel(stats="mean"), a
    `frac_valid` or `ensemble_spread` result, or a ratio of spreads.

    hatch : boolean (y_bin, x_bin) field; True cells are hatched
        (e.g. ~significant, or a ratio range)
    edge_ticks : label bin boundaries with the edge values
    **kwargs : passed to DataArray.plot.pcolormesh
    """
    da = _ensure_bin_coords(da)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    else:
        fig = ax.figure
    da.plot.pcolormesh(ax=ax, x="x_bin", y="y_bin", **kwargs)

    if hatch is not None:
        hatch = _ensure_bin_coords(hatch).transpose("y_bin", "x_bin")
        yb, xb = hatch["y_bin"].values, hatch["x_bin"].values
        for j, i in zip(*np.nonzero(hatch.fillna(False).astype(bool).values)):
            ax.add_patch(Rectangle(
                (xb[i] - 0.5, yb[j] - 0.5), 1, 1, fill=False, hatch="///", lw=0, edgecolor="0.3",
            ))

    if edge_ticks:
        _edge_ticks(ax, da, fmt)
    ax.set_xlabel(f"{da.attrs.get('x_variable', 'x')} $\\rightarrow$")
    ax.set_ylabel(f"{da.attrs.get('y_variable', 'y')} $\\rightarrow$")
    ax.set_title(title)
    return _finish(fig, fout)


def plot_bin_facets(
    bs: xr.DataArray,
    dim: str,
    *,
    stat: str = "mean",
    signif: xr.DataArray | None = None,
    col_wrap: int = 6,
    fout: str | Path | None = None,
    **kwargs,
):
    """One panel per member/source along `dim` of statistic `stat`, optionally masked by `signif`."""
    da = bs.sel(stats=stat)
    if signif is not None:
        da = da.where(signif)
    kwargs.setdefault("cmap", "BrBG")
    fg = da.plot.pcolormesh(x="x_bin", y="y_bin", col=dim, col_wrap=col_wrap, **kwargs)
    _finish(fg.fig, fout)
    return fg


def plot_bin_summary(
    bs: xr.DataArray,
    dim: str | None = None,
    *,
    title: str = "",
    fout: str | Path | None = None,
):
    """
    Three-panel diagnostic of a bin-stats result: mean ET (averaged over `dim`),
    total sample count (log scale), and spread (std across `dim` if given,
    otherwise the within-bin sample std).
    """
    bs = _ensure_bin_coords(bs)
    mean = bs.sel(stats="mean")
    count = bs.sel(stats="count")
    if dim is not None:
        mean = mean.mean(dim)
        count = count.sum(dim)
        spread = ensemble_spread(bs, dim)
        spread_label = f"std of bin mean across {dim}"
    else:
        spread = np.sqrt(bs.sel(stats="var_samp"))
        spread_label = "within-bin std"
    units = bs.attrs.get("units", "?")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    plot_bin_field(mean, ax=axs[0], cmap="YlGnBu", cbar_kwargs={"label": f"mean [{units}]"})
    plot_bin_field(
        count.where(count > 0), ax=axs[1], cmap="magma_r",
        norm=mcolors.LogNorm(), cbar_kwargs={"label": "sample count"},
    )
    plot_bin_field(spread, ax=axs[2], cmap="viridis", cbar_kwargs={"label": f"{spread_label} [{units}]"})
    for ax in axs:
        ax.set_title("")
    fig.suptitle(title)
    return _finish(fig, fout)


# ------------------------------------------------------------------
# Data loading (optional; imports dataset-specific libraries lazily)
# ------------------------------------------------------------------

def load_ilamb_obs(
    variable: str,
    product: str,
    *,
    root: Path = ILAMB_ROOT,
    time_slice: slice = slice(None, None),
    lat_bnds: slice = LAT_BNDS,
    verbose: bool = True,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Load an ILAMB observational product, converted to W/m2 if a water flux.

    Returns
    -------
    (field, cell area, land cell area)
    """
    ds = xr.open_dataset(root / variable / f"{variable}_{product}.nc")
    if product == "WECANN":
        ds = ds.sortby("lat", ascending=True)  # flip the lat dimension for WECANN

    obs = ds[variable].sel(time=time_slice)
    area, la = compute_cell_area(ds)
    if verbose:
        print(
            f"{product:11}: {obs.dims} {obs.shape} {to_yyyymm(obs.time[0])}-{to_yyyymm(obs.time[-1])} "
            f"(cell area from {area.attrs['method']})"
        )
    return (
        convert_units(variable, obs).sel(lat=lat_bnds),
        area.sel(lat=lat_bnds),
        la.sel(lat=lat_bnds),
    )


def load_cesm_grid(source: Literal["fppe", "goga", "lens"], lat_bnds: slice = LAT_BNDS) -> xr.Dataset:
    """Grid dataset (LANDFRAC, LANDAREA, ...) of a CESM ensemble."""
    return lc.load_grid(source).sel(lat=lat_bnds)


def load_cesm_variable(
    source: Literal["fppe", "goga", "lens"],
    variable: str,
    grid: xr.Dataset | xr.DataArray,
    *,
    mask: xr.DataArray | bool = True,
    time_slice: slice = slice(None, None),
    lat_bnds: slice = LAT_BNDS,
    gcomp: str = "lnd",
    stream: str = "h0",
    bb: str = "cmip6",
    verbose: bool = True,
) -> xr.DataArray:
    """
    Load a CESM ensemble variable on `grid`, masked and converted to W/m2.

    source : "fppe" (FHIST PPE), "goga" (GOGA2) or "lens" (LENS2)
    variable : name with frequency suffix, e.g. "EFLX_LH_TOT_month_1"
    gcomp, stream : model component ("lnd" or "atm") and history stream of
        `variable`, for every source. For "fppe", PRECT is read as
        PRECT_calculated from "atm" whatever `gcomp` is.
    """
    if verbose:
        print(f"{source.upper()}: Loading {variable}")
    v = "_".join(variable.split("_")[:-2])
    frq = "_".join(variable.split("_")[-2:])

    if source == "fppe":
        if v == "PRECT":
            v = "PRECT_calculated"
            variable = f"PRECT_calculated_{frq}"
            gcomp = "atm"
        ds = lc.load_fhist_ppe(v, gcomp, frq, stream, verbose=verbose)
    elif source == "goga":
        ds = lc.load_goga2(v, gcomp, frq, stream)
    elif source == "lens":
        ds = lc.load_cesm2le(v, gcomp, frq, stream, bb=bb, verbose=verbose)
    else:
        raise ValueError(f"Unknown CESM source {source!r}")

    da = ds[v].sel(time=time_slice, lat=lat_bnds)
    if not equal_coords(da, grid, ("lat", "lon")):
        raise IndexError(f"{variable} and grid do not have the same 'lat', 'lon' coordinates")
    da = da.reindex_like(grid, method="nearest", tolerance=1e-3)
    if isinstance(mask, xr.DataArray):
        if not equal_coords(mask, grid, ("lat", "lon")):
            raise IndexError("mask and grid do not have the same 'lat', 'lon' coordinates")
        da = da.where(mask.reindex_like(grid, method="nearest", tolerance=1e-3))
    return convert_units(variable, da, verbose=verbose)


def filter_all_variables_available(
    data_dict: dict,
    variables: Sequence[str],
    coords: Iterable[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Keep sources in {source: {var: da}} that have all `variables` on matching `coords`."""
    sid_avail = []
    missing = {}
    for sid, vardict in data_dict.items():
        absent = set(variables) - set(vardict.keys())
        if absent:
            missing[sid] = f"missing {absent}"
            continue
        test_da = vardict[variables[0]]
        if coords is not None:
            bad = [v for v in variables[1:] if not equal_coords(vardict[v], test_da, coords)]
            if bad:
                missing[sid] = f"{bad} mismatched coords {tuple(coords)}"
                continue
        sid_avail.append(sid)

    if verbose:
        print("=== Not available ===")
        for sid, reason in missing.items():
            print(f"{sid:16}: {reason}")
        print(f"\n=== Available: {len(sid_avail)} ===\n{sid_avail}")

    return {sid: data_dict[sid] for sid in sid_avail}


def load_cmip(
    variables: Sequence[str],
    catalog: str | Path,
    *,
    experiment_id: str = "historical",
    time_slice: slice = slice(None, None),
    lat_bnds: slice = LAT_BNDS,
    lf_thresh: float = LF_THRESH,
    member_id: str | Sequence[str] | None = "top",
    source_id: str | Sequence[str] | None = None,
    omit_source_id: str | Sequence[str] | None = None,
    grid_variables: Sequence[str] = ("areacella", "sftlf"),
    verbose: bool = True,
    **loader_kwargs,
) -> dict[str, dict[str, xr.DataArray]]:
    """
    Load CMIP6 variables with CMIPESGFLoader, keeping models with all
    variables and grid variables. Every variable is put on the areacella grid,
    masked with `mask_greenland`, and converted to W/m2 where applicable.

    Returns
    -------
    {source_id: {variable: da, "la": land area, "lf": land fraction, "mask": mask}}
    """
    try:
        from load_cmip_esgf import CMIPESGFLoader
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from load_cmip_esgf import CMIPESGFLoader

    loader = CMIPESGFLoader(catalog)
    cmip = loader.load_data(
        variables=list(variables) + list(grid_variables),
        experiment_id=experiment_id,
        source_id=source_id,
        omit_source_id=omit_source_id,
        member_id=member_id,
        time_slice=time_slice,
        **loader_kwargs,
    )
    cmip = filter_all_variables_available(cmip, list(variables), coords=("lat", "lon", "time"), verbose=verbose)
    cmip = filter_all_variables_available(cmip, list(grid_variables), coords=None, verbose=verbose)

    for sid, vardict in cmip.items():
        area = vardict.pop("areacella").sel(lat=lat_bnds)
        lf = vardict.pop("sftlf").sel(lat=lat_bnds).reindex_like(area, method="nearest", tolerance=1e-3)
        if lf.attrs.get("units") == "%":  # sftlf is in %, lf_thresh is a fraction
            lf = lf / 100
            lf.attrs["units"] = "1"
        mask = mask_greenland(lf, lf_thresh)
        mask = mask.where(mask.notnull(), other=False)
        for v, da in vardict.items():
            vardict[v] = convert_units(
                v,
                da.sel(lat=lat_bnds).reindex_like(mask, method="nearest", tolerance=1e-3).where(mask),
                verbose=verbose,
            )
        vardict["lf"] = lf
        vardict["la"] = (area * lf).compute()
        vardict["mask"] = mask
        if verbose:
            print(f"{sid:20}: {[(v, vardict[v].shape) for v in variables]}")

    return cmip
