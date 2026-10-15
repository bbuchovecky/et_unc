"""
bin_obs.py
==========
Bin evapotranspiration (ET) in a 2-D space of climatological leaf area index
(LAI, y-axis) and aridity index (AI = Rn / L*P, x-axis), and plot intermediate
diagnostics, for every (ET, LAI, precipitation, net radiation) combination of
observational products: the ILAMB products in ILAMB_PRODUCTS, ERA5 as a product
of each variable in ERA5_VARIABLES (ET, pr and rn; ERA5 has no LAI), plus the
gridded ET products in GRIDDED_ET_PRODUCTS (PML-V2.2, GLEAM v4.3, SiTHv2). Set
ERA5_VARIABLES = () and GRIDDED_ET_PRODUCTS = {} for an ILAMB-only run. (Merges
the former ilamb_binned_et.py and obs_binned_et.py.)

Steps
-----
1. Load annual means of each product onto the common TARGET_RES grid (lon in
   [-180, 180], lat ascending), all by the same rule: `load_annual` of the
   product's load module, i.e. annual means over complete years on the native
   grid, masked to land there (`load_native`), then conservatively regridded
   (`rg.regrid_annual`: NaN where more than NA_THRESH of a target cell is
   missing). ET, pr and rn are in W/m2 and need all 12 months for a valid
   annual mean, while a missing LAI month counts as LAI = 0 (winter gaps at
   high latitudes).
   - ILAMB products (`il.load_annual`): native land mask from Natural Earth
     (removes the ocean of the global pr and rn products).
   - ERA5 (`le.load_annual`, GDEX monthly mean rates: ET = -mer, pr = mtpr,
     rn = msnswrf + msnlwrf): native land mask LANDFRAC > LF_THRESH, as for
     CMIP6. Loaded only over the years in which every variable that ERA5 does
     not provide (LAI) has some product.
   - Gridded ET products (`lo.load_annual`), from their native 0.1 deg files
     (mm/month or mm/year): native land mask from Natural Earth. Loaded only
     over the years shared by the LAI, pr and rn products (ERA5 included). For
     yearly products the yearly total is the annual mean.
   Gridcell-years of any ET product with annual mean ET < MIN_ANNUAL_ET
   (-1 W/m2) are NaN (GLEAM v4.3b has extreme negative values at high
   latitudes).
2. Map the climatological mean of each product, and of AI for each pr/rn pair.
3. Each (ET, LAI, pr, rn) combination uses the complete years shared by all
   four products, and is skipped if there are fewer than MIN_YEARS. Binning
   inputs are annual mean ET and climatological LAI and AI
   (`lb.prepare_inputs`), restricted to one common area mask: land gridcells
   where ET (in any year), LAI and AI are valid in *every* combination, so
   all ET products cover the same area. The mask is static; ET years that
   are missing inside it stay NaN, so per-year availability still differs
   between products.
4. Bin ET with (a) quantile edges pooled across all combinations
   ("pooled_obs") and (b) each combination's own quantile edges ("combo").

Outputs
-------
<combo>  : "<et>-<lai>-<pr>-<rn>" product names
<period> : YYYYMM-YYYYMM of a combination / product; <span> covers all combinations
<kind>   : "pooled_obs" or "combo"

PROC_ROOT/obs.area_mask.all<N>.<span>.nc                         common area mask
PROC_ROOT/qbin/obs.<combo>.et.qbin_clim_<kind>.<period>.nc      per combination
PROC_ROOT/qbin/obs.all<N>.et.qbin_clim_<kind>.<span>.nc          all combinations along `combo`
PROC_ROOT/qbin/obs.all<N>.et.qbin_clim_pooled_obs.<span>.et_product_std.nc   std across ET products
BIN_EDGES_ROOT/obs.{lai,ai}_clim.15_quantiles_{pooled,combo}.all<N>.<span>.nc
FIG_ROOT/obs/<var>/obs.<product>.<var>.map.<period>.png          var in et, lai, pr, rn
FIG_ROOT/obs/ai/obs.<pr>-<rn>.ai.map.<period>.png
FIG_ROOT/obs/qbin_edges/obs.{lai,ai}_clim.15_quantiles_{pooled,combo}.all<N>.<span>.png
FIG_ROOT/obs/qbin/obs.all<N>.et.qbin_clim_<kind>.<span>.summary.png
FIG_ROOT/obs/qbin/obs.all<N>.et.qbin_clim_<kind>.<span>.bin_mean.png   one heatmap per combination
FIG_ROOT/obs/qbin/obs.all<N>.et.qbin_clim_<kind>.<span>.bin_mean_signif.png
    as bin_mean, with bins whose mean is not significantly different from 0 hatched
FIG_ROOT/obs/qbin/obs.all<N>.et.qbin_clim_pooled_obs.<span>.et_product_std.png
    std of bin mean ET across ET products, one heatmap per (lai, pr, rn) product set
"""

from __future__ import annotations

import itertools
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import etunc.config as cfg
import etunc.temporal as temporal
import etunc.binning as binning
from etunc.edges import EDGE_ATTRS, compute_aridity_index
import etunc.legacy.binning as lb  # old edges and inputs, until bin_all.py replaces this script
import etunc.plotting as plotting
import etunc.grid as rg
import etunc.load.era5 as le
import etunc.load.ilamb as il
import etunc.load.obs as lo

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS. Will affect performance.", category=UserWarning)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROC_ROOT = cfg.PROC_ROOT / "obs"
BIN_EDGES_ROOT = cfg.BIN_EDGES_ROOT
FIG_ROOT = cfg.FIG_ROOT


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

N_XBINS = cfg.N_XBINS
N_YBINS = cfg.N_YBINS
SIGNIF_ALPHA = cfg.SIGNIF_ALPHA
SIGNIF_N_MIN = cfg.SIGNIF_N_MIN

# Common grid
TARGET_RES = cfg.TARGET_RES
TARGET_GRID = rg.target_grid(TARGET_RES)

# Minimum years shared by all four products
MIN_YEARS = 3

FACTORS = tuple(il.PRODUCTS)  # order of products in a combination

# ILAMB products used, by variable (keys of il.PRODUCTS[variable]); a variable
# missing here (or None) uses all of its il.PRODUCTS
ILAMB_PRODUCTS = {
    "et":  list(il.PRODUCTS["et"]),  # every ILAMB ET product
    "lai": ["MODIS"],
    "pr":  ["GPCPv2.3"],
    "rn":  ["CERESed4.2"],
}

# Variables for which ERA5 is added as a product (labelled "ERA5"), any of
# le.VARIABLES ("et", "pr", "rn"; ERA5 has no LAI). () = no ERA5
ERA5_VARIABLES = ("et", "pr", "rn")

# Gridded ET products added to the ILAMB ET products, read from their native files:
# {label: (load_obs dataset, version, variable, freq)}. {} = ILAMB only.
GRIDDED_ET_PRODUCTS = {
    "PMLv2.2a_MODIS": ("pml", "V2.2a-MODIS", "ET", "monthly"),
    # "PMLv2.2b":       ("pml", "V2.2b", "ET", "monthly"),
    "PMLv2.2c":       ("pml", "V2.2c", "ET", "monthly"),
    "GLEAMv4.3a":     ("gleam", "v4.3a", "E", "monthly"),
    "GLEAMv4.3b":     ("gleam", "v4.3b", "E", "monthly"),
    "SiTHv2":         ("sith", "v2", "ET", "monthly"),
}
# Gridcell-years of any ET product with annual mean ET below this [W/m2] are
# NaN; GLEAM v4.3b has extreme negative monthly E at high latitudes in winter
MIN_ANNUAL_ET = -1.0

# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def era5_span(ann: dict[str, dict[str, xr.DataArray]]) -> slice:
    """
    Years to load ERA5 for: from the first to the last year on GDEX in which
    every variable that ERA5 does not provide (always LAI) has some product in
    `ann`. ERA5 years outside them can never be part of a combination.
    """
    years = set(le.available_years("meanflux"))
    for v in FACTORS:
        if v not in ERA5_VARIABLES:
            years &= set().union(*(da.year.values.tolist() for da in ann[v].values()))
    if not years:
        raise RuntimeError("No ERA5 year is covered by a product of every variable that ERA5 does not provide.")
    return slice(str(min(years)), str(max(years)))


def drop_et_outliers(da: xr.DataArray, label: str) -> xr.DataArray:
    """Annual mean ET `da` with gridcell-years below MIN_ANNUAL_ET set to NaN."""
    outlier = da < MIN_ANNUAL_ET
    if not outlier.any():
        return da
    print(f"et/{label}: {int(outlier.sum())} gridcell-years with annual ET < {MIN_ANNUAL_ET} W/m2 set to NaN "
          f"(min {float(da.min()):0.3g})")
    with xr.set_options(keep_attrs=True):
        return da.where(~outlier)


# ------------------------------------------------------------------
# Combinations
# ------------------------------------------------------------------

def combo_attrs(combo: tuple[str, ...], years: list[int]) -> dict:
    return {
        **{f"{v}_product": p for v, p in zip(FACTORS, combo)},
        "combo": "-".join(combo),
        "time_period": temporal.period_str(years),
        "n_years": len(years),
    }


def combo_inputs(
    ann: dict[str, dict[str, xr.DataArray]],
    combo: tuple[str, ...],
    years: list[int],
    mask: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """`lb.prepare_inputs` for one (et, lai, pr, rn) combination over `years`, on land gridcells."""
    fields = {v: ann[v][p].sel(year=years) for v, p in zip(FACTORS, combo)}
    return lb.prepare_inputs(
        et=fields["et"], lai=fields["lai"], precip=fields["pr"], rn=fields["rn"], mask=mask,
    )


def concat_combos(das: list[xr.DataArray], combos: dict[str, tuple]) -> xr.DataArray:
    """Concatenate per-combination results along `combo`, with each factor's product as a coord."""
    out = xr.concat(
        das, dim="combo", coords="different", compat="equals", join="exact", combine_attrs="drop_conflicts",
    )
    return out.assign_coords(
        combo=list(combos),
        **{f"{v}_product": ("combo", [c[i] for c, _ in combos.values()]) for i, v in enumerate(FACTORS)},
        time_period=("combo", [temporal.period_str(y) for _, y in combos.values()]),
        n_years=("combo", [len(y) for _, y in combos.values()]),
    )


def et_product_spread(bs_all: xr.DataArray) -> xr.Dataset:
    """
    Sample std (ddof=1) of the bin mean ET across ET products, separately for
    each (lai, pr, rn) product set along `lai_pr_rn`, so only the ET product
    varies. Needs edges shared by all combinations ("pooled_obs"). Bins with
    fewer than two ET products are NaN; `n_products` counts the ET products
    with a bin mean and `n_et_products` the ET products in each set.
    """
    labels = ["-".join(c) for c in zip(*(bs_all[f"{v}_product"].values for v in FACTORS[1:]))]
    grouped = (
        bs_all.sel(stats="mean", drop=True)
        .assign_coords(lai_pr_rn=("combo", labels))
        .groupby("lai_pr_rn")
    )
    with xr.set_options(keep_attrs=True):
        n = grouped.count("combo")
        std = grouped.std("combo", ddof=1).where(n >= 2)
    out = xr.Dataset({"et_std": std, "n_products": n.astype("i4")}, attrs=bs_all.attrs)
    out["et_std"].attrs["long_name"] = "std of bin mean ET across ET products (ddof=1)"
    return out.assign_coords(n_et_products=("lai_pr_rn", [labels.count(g) for g in out.lai_pr_rn.values]))


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def save_clim_map(da: xr.DataArray, variable: str, label: str, period: str, mask: xr.DataArray):
    """Map of the climatology `da` of one product (`label`) on land (`mask`)."""
    fout = FIG_ROOT / "obs" / variable / f"obs.{label}.{variable}.map.{period}.png"
    plotting.save_map(da.where(mask), variable, fout, title=f"{label}, {period}")


def plot_combo_edges(combo_edges: xr.DataArray, pooled: xr.DataArray, title: str = "", fout: Path | None = None):
    """
    Per-combination bin edges (without the maximum edge), coloured by the
    product of each factor in turn, with the pooled edges in black.
    """
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True, layout="constrained")
    cmap = plt.get_cmap("tab10")
    edge = combo_edges.edge.values[:-1]
    for ax, factor in zip(axs.ravel(), FACTORS):
        products = combo_edges[f"{factor}_product"].values
        for i, p in enumerate(dict.fromkeys(products)):
            sub = combo_edges.isel(combo=np.flatnonzero(products == p))
            for j, e in enumerate(sub.values[:, :-1]):
                ax.plot(edge, e, color=cmap(i % 10), lw=0.8, alpha=0.5, label=p if j == 0 else None)
        ax.plot(edge, pooled.values[:-1], color="k", lw=2, ls="--", marker="o", ms=3, label="pooled")
        ax.set_title(f"coloured by {factor} product")
        ax.legend(fontsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("edge")
    for ax in axs[:, 0]:
        ax.set_ylabel(f"{combo_edges.name} [{pooled.attrs.get('units', '?')}]")
    fig.suptitle(title)
    return plotting.finish(fig, fout)


def plot_et_product_spread(spread: xr.Dataset, title: str = "", fout: Path | None = None):
    """
    One heatmap per (lai, pr, rn) product set of the std of bin mean ET across
    ET products (`et_product_spread`), on a shared colour scale. Hatched bins
    lack a bin mean from at least one ET product.
    """
    groups = spread.lai_pr_rn.values
    fig, axs = plt.subplots(
        1, len(groups), figsize=(5.5 * len(groups), 4.5), squeeze=False, layout="constrained",
    )
    units = spread.attrs.get("units", "?")
    vmax = float(spread["et_std"].quantile(0.98))
    for ax, g in zip(axs.ravel(), groups):
        s = spread.sel(lai_pr_rn=g)
        plotting.plot_bin_field(
            s["et_std"], ax=ax, hatch=s["n_products"] < s["n_et_products"],
            cmap="viridis", vmin=0, vmax=vmax, extend="max",
            cbar_kwargs={"label": f"std of bin mean et [{units}]"},
            title=f"{g}\n{int(s.n_et_products)} ET products",
        )
        ax.set_xlabel("ai $\\rightarrow$")
        ax.set_ylabel("lai $\\rightarrow$")
    fig.suptitle(title)
    return plotting.finish(fig, fout)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    mask = rg.landne_mask(TARGET_GRID).sel(lat=cfg.LAT_BNDS)

    # ------------------------------------------------------------------
    # Annual means of every product: {variable: {product: (year, lat, lon)}}
    # ------------------------------------------------------------------
    print("=== Load ILAMB products ===")
    ann = {
        v: {p: il.load_annual(v, p, slice(None, None), TARGET_RES)[v] for p in (ILAMB_PRODUCTS.get(v) or products)}
        for v, products in il.PRODUCTS.items()
    }

    # ERA5 before the gridded ET products, so that ERA5 pr and rn count toward their years
    if ERA5_VARIABLES:
        span = era5_span(ann)
        print(f"\n=== Load ERA5 ({span.start}-{span.stop}) ===")
        for v, da in le.load_annual(ERA5_VARIABLES, span, TARGET_RES).items():
            ann[v]["ERA5"] = da

    # Gridded ET only over the years that some (lai, pr, rn) product set shares
    if GRIDDED_ET_PRODUCTS:
        years = set().union(*(
            temporal.shared_years(*das) for das in itertools.product(*(ann[v].values() for v in FACTORS[1:]))
        ))
        if not years:
            raise RuntimeError("No (lai, pr, rn) product set shares any year.")
        span = slice(str(min(years)), str(max(years)))
        print(f"\n=== Load gridded ET products ({span.start}-{span.stop}) ===")
        for label, (dataset, version, var, freq) in GRIDDED_ET_PRODUCTS.items():
            et = lo.load_annual(dataset, var, span, TARGET_RES, version=version, freq=freq)[var]
            ann["et"][label] = et.rename("et")

    # One plausibility rule for every ET product
    ann["et"] = {p: drop_et_outliers(da, p) for p, da in ann["et"].items()}

    # ------------------------------------------------------------------
    # Maps of climatological means
    # ------------------------------------------------------------------
    print("\n=== Maps of climatological means ===")
    for v, products in ann.items():
        for p, da in products.items():
            save_clim_map(temporal.aggregate(da, "clim"), v, p, temporal.period_str(da.year.values), mask)

    for pr_p, rn_p in itertools.product(ann["pr"], ann["rn"]):
        years = temporal.shared_years(ann["pr"][pr_p], ann["rn"][rn_p])
        if not years:
            print(f"{pr_p}-{rn_p}: no shared years, skipping AI map")
            continue
        ai = compute_aridity_index(
            temporal.aggregate(ann["pr"][pr_p].sel(year=years), "clim"),
            temporal.aggregate(ann["rn"][rn_p].sel(year=years), "clim"),
        )
        save_clim_map(ai, "ai", f"{pr_p}-{rn_p}", temporal.period_str(years), mask)

    # ------------------------------------------------------------------
    # Combinations and their shared years
    # ------------------------------------------------------------------
    print("\n=== Combinations ===")
    combos = {}
    for combo in itertools.product(*(ann[v] for v in FACTORS)):
        cid = "-".join(combo)
        years = temporal.shared_years(*(ann[v][p] for v, p in zip(FACTORS, combo)))
        if len(years) < MIN_YEARS:
            print(f"{cid:50}: {len(years)} shared years < {MIN_YEARS}, skipping")
            continue
        combos[cid] = (combo, years)
    ncombo = len(combos)
    if ncombo == 0:
        raise RuntimeError(f"No combination has at least {MIN_YEARS} shared years.")
    span = temporal.period_str([y for _, years in combos.values() for y in years])
    print(f"{ncombo} of {np.prod([len(ann[v]) for v in FACTORS])} combinations have >= {MIN_YEARS} shared years ({span})")

    # ------------------------------------------------------------------
    # Binning inputs, restricted to the area valid in every combination
    # ------------------------------------------------------------------
    print("\n=== Common area mask ===")
    inputs = {cid: combo_inputs(ann, combo, years, mask) for cid, (combo, years) in combos.items()}
    area = lb.common_area(inputs, mask, label_width=50)
    with xr.set_options(keep_attrs=True):
        inputs = {cid: {k: da.where(area) for k, da in inp.items()} for cid, inp in inputs.items()}
    fout = PROC_ROOT / f"obs.area_mask.all{ncombo}.{span}.nc"
    fout.parent.mkdir(exist_ok=True, parents=True)
    area.astype("i1").assign_attrs(
        long_name="land gridcells where ET (any year), LAI and AI are valid in every combination",
        combos=list(combos),
    ).to_netcdf(fout)
    print(fout)

    # ------------------------------------------------------------------
    # Quantile bin edges pooled across all combinations
    # ------------------------------------------------------------------
    print("\n=== Pooled bin edges ===")
    n_bins = {"lai": N_YBINS, "ai": N_XBINS}
    pooled_edges = {}
    for v in ("lai", "ai"):
        print(f"\n{v}")
        pooled_edges[v] = lb.pooled_bin_edges(
            {cid: inp[v] for cid, inp in inputs.items()}, n_bins[v], name=v,
            attrs={**EDGE_ATTRS[v], "time_period": span},
        )
        fstem = f"obs.{v}_clim.{n_bins[v]}_quantiles_pooled.all{ncombo}.{span}"
        fout = BIN_EDGES_ROOT / f"{fstem}.nc"
        fout.parent.mkdir(exist_ok=True, parents=True)
        pooled_edges[v].to_netcdf(fout)
        print(fout)
        fout = FIG_ROOT / "obs" / "qbin_edges" / f"{fstem}.png"
        plotting.plot_edges(pooled_edges[v], fout, title=f"obs {v}, pooled across {ncombo} combinations, {span}")
        print(fout)

    # ------------------------------------------------------------------
    # Binned mean ET, with pooled and with per-combination edges
    # ------------------------------------------------------------------
    print("\n=== Binned ET ===")
    bs = {"pooled_obs": [], "combo": []}
    combo_edges = {"lai": [], "ai": []}
    for cid, (combo, years) in combos.items():
        inp = inputs[cid]
        attrs = combo_attrs(combo, years)

        edges = {}
        for v in ("lai", "ai"):
            edges[v] = lb.pooled_bin_edges(
                inp[v], n_bins[v], name=v, verbose=False,
                attrs={**EDGE_ATTRS[v], **attrs, "pool_edges": 0, "pooled_sources": [cid]},
            )
            combo_edges[v].append(edges[v])

        for kind, (y_edges, x_edges) in (
            ("pooled_obs", (pooled_edges["lai"], pooled_edges["ai"])),
            ("combo", (edges["lai"], edges["ai"])),
        ):
            bs_c = binning.bin_stats(
                inp["et"], inp["lai"], inp["ai"], y_edges, x_edges,
                y_name="lai", x_name="ai", name="et",
                attrs={
                    **attrs,
                    "n_y_bins_requested": N_YBINS,
                    "n_x_bins_requested": N_XBINS,
                    "collapse_duplicate_quantile_bins": 0,
                    "pool_edges": int(kind == "pooled_obs"),
                },
            )
            fout = PROC_ROOT / "qbin" / f"obs.{cid}.et.qbin_clim_{kind}.{attrs['time_period']}.nc"
            fout.parent.mkdir(exist_ok=True, parents=True)
            bs_c.to_netcdf(fout)
            bs[kind].append(bs_c)

        print(f"{cid:50}: {attrs['time_period']} ({len(years)} yr), "
              f"{int(bs[kind][-1].sel(stats='count').sum()):.3e} samples binned")

    for kind, bs_list in bs.items():
        bs_all = concat_combos(bs_list, combos)
        fstem = f"obs.all{ncombo}.et.qbin_clim_{kind}.{span}"
        fout = PROC_ROOT / "qbin" / f"{fstem}.nc"
        bs_all.to_netcdf(fout)
        print(fout)

        # Zero-width bins only exist (and can only be dropped) for the shared pooled edges
        if kind == "pooled_obs":
            bs_all = binning.drop_zero_width_bins(bs_all)
        fout = FIG_ROOT / "obs" / "qbin" / f"{fstem}.summary.png"
        plotting.plot_bin_summary(bs_all, dim="combo", title=f"obs ET, {ncombo} combinations, {kind} edges", fout=fout)
        print(fout)
        fout = FIG_ROOT / "obs" / "qbin" / f"{fstem}.bin_mean.png"
        plotting.plot_bin_means(bs_all, "combo", col_wrap=4, size=3.5, title=f"obs ET bin mean, {ncombo} combinations, {kind} edges", fout=fout)
        print(fout)

        # Hatch non-empty bins whose mean is not significantly different from 0
        signif = binning.test_significance(bs_all, alpha=SIGNIF_ALPHA, n_min=SIGNIF_N_MIN)
        not_signif = ~signif & bs_all.sel(stats="mean").notnull()
        print(f"{kind}: {int(not_signif.sum())} of {int(bs_all.sel(stats='mean').notnull().sum())} "
              f"non-empty bins not significant at {1 - SIGNIF_ALPHA:.0%}")
        fout = FIG_ROOT / "obs" / "qbin" / f"{fstem}.bin_mean_signif.png"
        plotting.plot_bin_means(
            bs_all, "combo", col_wrap=4, size=3.5, fout=fout, hatch=not_signif,
            title=(f"obs ET bin mean, {ncombo} combinations, {kind} edges; hatched: not significantly "
                   f"different from 0 at {1 - SIGNIF_ALPHA:.0%} (t-test, n > {SIGNIF_N_MIN})"),
        )
        print(fout)

        # Spread across ET products needs bins that mean the same LAI/AI range in every combination
        if kind == "pooled_obs":
            spread = et_product_spread(bs_all)
            fout = PROC_ROOT / "qbin" / f"{fstem}.et_product_std.nc"
            spread.to_netcdf(fout)
            print(fout)
            fout = FIG_ROOT / "obs" / "qbin" / f"{fstem}.et_product_std.png"
            plot_et_product_spread(spread, title=f"obs ET, std of bin mean across ET products, {kind} edges", fout=fout)
            print(fout)

    # ------------------------------------------------------------------
    # Per-combination bin edges
    # ------------------------------------------------------------------
    for v in ("lai", "ai"):
        edges_all = concat_combos(combo_edges[v], combos)
        fstem = f"obs.{v}_clim.{n_bins[v]}_quantiles_combo.all{ncombo}.{span}"
        fout = BIN_EDGES_ROOT / f"{fstem}.nc"
        edges_all.to_netcdf(fout)
        print(fout)
        fout = FIG_ROOT / "obs" / "qbin_edges" / f"{fstem}.png"
        plot_combo_edges(edges_all, pooled_edges[v], title=f"obs {v}, per-combination edges, {span}", fout=fout)
        print(fout)


if __name__ == "__main__":
    main()
