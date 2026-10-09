"""
ilamb_binned_et.py
==================
Bin evapotranspiration (ET) in a 2-D space of climatological leaf area index
(LAI, y-axis) and aridity index (AI = Rn / L*P, x-axis), and plot intermediate
diagnostics, for every (ET, LAI, precipitation, net radiation) combination of
ILAMB products.

Steps
-----
1. Load each product onto a common 0.5 deg grid (lon in [-180, 180], lat
   ascending in [-90, 90]); 1 deg products are bilinearly interpolated
   (xESMF). ET (evspsbl, hfls), pr and rns are converted to W/m2, LAI is m2/m2.
   Annual means are computed over complete years only. A missing month counts
   as LAI = 0 (winter gaps at high latitudes), while ET, pr and rns need all
   12 months for the annual mean to be valid.
2. Map the climatological mean of each product, and of AI for each pr/rns pair.
3. Each (ET, LAI, pr, rns) combination uses the complete years shared by all
   four products, and is skipped if there are fewer than MIN_YEARS. Binning
   inputs are annual mean ET and climatological LAI and AI
   (`binning.prepare_inputs`), restricted to one common area mask: land gridcells
   where ET (in any year), LAI and AI are valid in *every* combination, so
   all ET products cover the same area. The mask is static; ET years that
   are missing inside it stay NaN, so per-year availability still differs
   between products.
4. Bin ET with (a) quantile edges pooled across all combinations
   ("pooled_obs") and (b) each combination's own quantile edges ("combo").

Outputs
-------
<combo>  : "<et>-<lai>-<pr>-<rns>" product names
<period> : YYYYMM-YYYYMM of a combination / product; <span> covers all combinations
<kind>   : "pooled_obs" or "combo"

PROC_ROOT/obs.area_mask.all<N>.<span>.nc                         common area mask
PROC_ROOT/qbin/obs.<combo>.et.qbin_clim_<kind>.<period>.nc      per combination
PROC_ROOT/qbin/obs.all<N>.et.qbin_clim_<kind>.<span>.nc          all combinations along `combo`
PROC_ROOT/qbin/obs.all<N>.et.qbin_clim_pooled_obs.<span>.et_product_std.nc   std across ET products
BIN_EDGES_ROOT/obs.{lai,ai}_clim.15_quantiles_{pooled,combo}.all<N>.<span>.nc
FIG_ROOT/obs/<var>/obs.<product>.<var>.map.<period>.png          var in et, lai, pr, rns
FIG_ROOT/obs/ai/obs.<pr>-<rns>.ai.map.<period>.png
FIG_ROOT/obs/qbin_edges/obs.{lai,ai}_clim.15_quantiles_{pooled,combo}.all<N>.<span>.png
FIG_ROOT/obs/qbin/obs.all<N>.et.qbin_clim_<kind>.<span>.summary.png
FIG_ROOT/obs/qbin/obs.all<N>.et.qbin_clim_<kind>.<span>.bin_mean.png   one heatmap per combination
FIG_ROOT/obs/qbin/obs.all<N>.et.qbin_clim_<kind>.<span>.bin_mean_signif.png
    as bin_mean, with bins whose mean is not significantly different from 0 hatched
FIG_ROOT/obs/qbin/obs.all<N>.et.qbin_clim_pooled_obs.<span>.et_product_std.png
    std of bin mean ET across ET products, one heatmap per (lai, pr, rns) product set
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import xarray as xr

import etunc.config as config
import etunc.units as units
import etunc.temporal as temporal
import etunc.binning as binning
import etunc.plotting as plotting
import etunc.grid as rg


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

ILAMB_DATA_ROOT = config.ILAMB_ROOT / "ILAMB-Data"
PROC_ROOT = Path("/glade/work/bbuchovecky/et_unc/proc/obs")
BIN_EDGES_ROOT = Path("/glade/work/bbuchovecky/et_unc/proc/qbin_edges")
FIG_ROOT = Path("/glade/work/bbuchovecky/et_unc/fig")


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

N_XBINS = 15    # aridity index
N_YBINS = 15    # LAI
MIN_YEARS = 3   # minimum number of complete years shared by a combination
SIGNIF_ALPHA = 0.05  # t-test of bin mean ET against 0 (95% confidence)
SIGNIF_N_MIN = 10    # bins with <= this many samples are never significant
FILL_THRESH = 1e30  # GPCCv2018 stores undecoded ~9.97e36 fill values

# Common 0.5 deg grid; 1 deg products (WECANN, CERESed4.2) are interpolated onto it
TARGET_RES = 0.5
TARGET_GRID = rg.target_grid(TARGET_RES)

# {variable: {product: (file relative to ILAMB_DATA_ROOT, variable name in file)}}
# ET products come from both evspsbl and hfls. Not used: FLUXNET2015 and
# WRMC.BSRN (site data), CARDAMOM (4x5 deg), FLUXCOM le.nc (hfls.nc used).
PRODUCTS = {
    "et": {
        "CLASS":      ("hfls/CLASS/hfls.nc", "hfls"),
        "DOLCE":      ("evspsbl/DOLCE/DOLCE.nc", "hfls"),
        "FLUXCOM":    ("hfls/FLUXCOM/hfls.nc", "hfls"),
        "GLEAMv3.3a": ("evspsbl/GLEAMv3.3a/et.nc", "et"),
        "MOD16A2":    ("evspsbl/MOD16A2/et.nc", "et"),
        "MODIS":      ("evspsbl/MODIS/et_0.5x0.5.nc", "et"),
        "WECANN":     ("hfls/WECANN/hfls.nc", "hfls"),
    },
    "lai": {
        "AVH15C1":     ("lai/AVH15C1/lai.nc", "lai"),
        "AVHRR":       ("lai/AVHRR/lai_0.5x0.5.nc", "lai"),
        "GIMMS_LAI4g": ("lai/GIMMS_LAI4g/cao2023_lai.nc", "lai"),
        "MODIS":       ("lai/MODIS/lai_0.5x0.5.nc", "lai"),
    },
    "pr": {
        "CLASS":     ("pr/CLASS/pr.nc", "pr"),
        "CMAPv1904": ("pr/CMAPv1904/pr.nc", "pr"),
        "GPCCv2018": ("pr/GPCCv2018/pr.nc", "pr"),
        "GPCPv2.3":  ("pr/GPCPv2.3/pr.nc", "pr"),
    },
    "rns": {
        "CERESed4.2": ("rns/CERESed4.2/rns.nc", "rns"),
        "CLASS":      ("rns/CLASS/rns.nc", "rns"),
        "GEWEX.SRB":  ("rns/GEWEX.SRB/rns_0.5x0.5.nc", "rns"),
    },
}
FACTORS = tuple(PRODUCTS)  # order of products in a combination

# Products used in this run; a variable missing here (or None) uses all of its PRODUCTS
RUN_PRODUCTS = {
    "et":  [
        "CLASS",
        "DOLCE",
        "FLUXCOM",
        "GLEAMv3.3a",
        "MOD16A2",
        "MODIS",
        "WECANN",
    ],
    "lai": ["MODIS"],
    "pr":  ["GPCPv2.3"],
    "rns": ["CERESed4.2"],
}

MAP_KWARGS = {
    "et":  {"cmap": "YlGnBu"},
    "lai": {"cmap": "Greens"},
    "pr":  {"cmap": "Blues"},
    "rns": {"cmap": "YlOrRd", "vmin": None},
    "ai":  {"cmap": "BrBG_r"},
}

EDGE_ATTRS = {
    "lai": {"long_name": "leaf area index bin edges", "units": "m2/m2", "variable": "leaf area index"},
    "ai":  {"long_name": "aridity index bin edges", "units": "1", "variable": "aridity index (Rn/L*P)"},
}


# ------------------------------------------------------------------
# Loading and formatting
# ------------------------------------------------------------------

def to_wm2(da: xr.DataArray) -> xr.DataArray:
    """Convert an ET, precipitation or net radiation flux to W/m2."""
    key = " ".join(str(da.attrs.get("units", "")).lower().split())
    if key in ("mm d-1", "mm/day"):  # GPCCv2018; 1 mm of water = 1 kg/m2
        da = (da / 86400).assign_attrs({**da.attrs, "units": "kg m-2 s-1"})
    return units.latent_heat_to_wm2(da)


def regrid_to_target(da: xr.DataArray) -> xr.DataArray:
    """Bilinear interpolation onto TARGET_GRID; target points outside the source grid are NaN."""
    return rg.bilinear_regridder(da, TARGET_RES)(da, keep_attrs=True)


def load_product(variable: str, product: str) -> xr.DataArray:
    """Annual means (year, lat, lon) of one product on TARGET_GRID within LAT_BNDS."""
    relpath, name = PRODUCTS[variable][product]
    ds = xr.open_dataset(ILAMB_DATA_ROOT / relpath)
    da = rg.format_grid(ds[name])
    da = da.sel(time=da.time.dt.year.isin(temporal.complete_years(da))).load()
    with xr.set_options(keep_attrs=True):
        da = da.where(np.abs(da) < FILL_THRESH)

    if variable == "lai":
        da.attrs["units"] = "m2/m2"
    else:
        da = to_wm2(da)
    ann = temporal.annual_mean(da, require_all_months=(variable != "lai"))

    if ann.sizes["lat"] != TARGET_GRID.sizes["lat"] or ann.sizes["lon"] != TARGET_GRID.sizes["lon"]:
        print(f"{variable}/{product}: regridding {ann.sizes['lat']}x{ann.sizes['lon']} -> 0.5 deg")
        ann = regrid_to_target(ann)
    # Exact coordinate values so that fields from different products align
    ann = rg.on_grid(ann, TARGET_RES, f"{variable}/{product}").rename(variable)

    print(
        f"{variable:3} {product:12}: {ann.dims} {ann.shape} {temporal.period_str(ann.year.values)} "
        f"[{float(ann.min()):0.3g}, {float(ann.max()):0.3g}] {ann.attrs['units']}"
    )
    return ann


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
    """`binning.prepare_inputs` for one (et, lai, pr, rns) combination over `years`, on land gridcells."""
    fields = {v: ann[v][p].sel(year=years) for v, p in zip(FACTORS, combo)}
    return binning.prepare_inputs(
        et=fields["et"], lai=fields["lai"], precip=fields["pr"], rn=fields["rns"], mask=mask,
    )


def valid_area(inputs: dict[str, xr.DataArray]) -> xr.DataArray:
    """Gridcells where ET (in any year), LAI and AI are all valid."""
    return inputs["et"].notnull().any("year") & inputs["lai"].notnull() & np.isfinite(inputs["ai"])


def common_area(inputs: dict[str, dict[str, xr.DataArray]], mask: xr.DataArray) -> xr.DataArray:
    """
    Land gridcells valid (`valid_area`) in every combination, so that all ET
    products cover the same area. The mask is static: ET years that are NaN
    inside this area stay NaN, so per-year data availability is kept.
    """
    area = mask == 1
    for cid, inp in inputs.items():
        valid = valid_area(inp)
        print(f"{cid:50}: {int(valid.sum()):.4e} valid gridcells")
        area = area & valid
    print(f"common area: {int(area.sum()):.4e} of {int((mask == 1).sum()):.4e} land gridcells")
    return area.rename("area_mask")


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
    each (lai, pr, rns) product set along `lai_pr_rns`, so only the ET product
    varies. Needs edges shared by all combinations ("pooled_obs"). Bins with
    fewer than two ET products are NaN; `n_products` counts the ET products
    with a bin mean and `n_et_products` the ET products in each set.
    """
    labels = ["-".join(c) for c in zip(*(bs_all[f"{v}_product"].values for v in FACTORS[1:]))]
    grouped = (
        bs_all.sel(stats="mean", drop=True)
        .assign_coords(lai_pr_rns=("combo", labels))
        .groupby("lai_pr_rns")
    )
    with xr.set_options(keep_attrs=True):
        n = grouped.count("combo")
        std = grouped.std("combo", ddof=1).where(n >= 2)
    out = xr.Dataset({"et_std": std, "n_products": n.astype("i4")}, attrs=bs_all.attrs)
    out["et_std"].attrs["long_name"] = "std of bin mean ET across ET products (ddof=1)"
    return out.assign_coords(n_et_products=("lai_pr_rns", [labels.count(g) for g in out.lai_pr_rns.values]))


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def save_map(da: xr.DataArray, variable: str, label: str, period: str, mask: xr.DataArray):
    fout = FIG_ROOT / "obs" / variable / f"obs.{label}.{variable}.map.{period}.png"
    plotting.quick_map(
        da.where(mask), fout, title=f"{label}, {period}",
        cbar_kwargs={"label": f"{variable} [{da.attrs.get('units', '?')}]"}, **MAP_KWARGS[variable],
    )
    print(fout)


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


def hatch_bins(ax, hatch: xr.DataArray):
    """Hatch the cells of a (y_bin, x_bin) heatmap where `hatch` is True."""
    hatch = hatch.transpose("y_bin", "x_bin")
    yb, xb = hatch["y_bin"].values, hatch["x_bin"].values
    for j, i in zip(*np.nonzero(hatch.fillna(False).astype(bool).values)):
        ax.add_patch(Rectangle(
            (xb[i] - 0.5, yb[j] - 0.5), 1, 1, fill=False, hatch="///", lw=0, edgecolor="0.3",
        ))


def plot_combo_bin_means(
    bs_all: xr.DataArray,
    title: str = "",
    fout: Path | None = None,
    col_wrap: int = 4,
    hatch: xr.DataArray | None = None,
):
    """
    One heatmap of the bin mean ET per combination on a shared colour scale.
    Panels share axes, so ticks show the bin edge values when all combinations
    share edges, and quantile levels when each has its own edges.

    hatch : boolean (combo, y_bin, x_bin) field; True bins are hatched
        (e.g. ~`binning.test_significance`)
    """
    fg = plotting.plot_bin_facets(
        bs_all, dim="combo", col_wrap=min(col_wrap, bs_all.sizes["combo"]), size=3.5,
        cmap="YlGnBu", robust=True, cbar_kwargs={"label": f"bin mean et [{bs_all.attrs.get('units', '?')}]"},
    )
    shared_edges = "combo" not in bs_all["y_bin_lower"].dims and "combo" not in bs_all["x_bin_lower"].dims
    for ax, name_dict in zip(fg.axs.flat, fg.name_dicts.flat):
        if name_dict is None:
            continue
        panel = bs_all.sel(**name_dict)
        ax.set_title(f"{name_dict['combo']}\n{panel.time_period.item()}", fontsize=8)
        if hatch is not None:
            hatch_bins(ax, hatch.sel(**name_dict))
    if shared_edges:
        plotting.set_edge_ticks(fg.axs.flat[0], bs_all, "{:.2g}")
    else:
        for dim, set_ticks in (("x_bin", fg.axs.flat[0].set_xticks), ("y_bin", fg.axs.flat[0].set_yticks)):
            n = bs_all.sizes[dim]
            set_ticks(np.arange(n + 1) - 0.5, [f"Q{q:.0f}" for q in np.linspace(0, 100, n + 1)])
    for ax in fg.axs.flat:
        ax.tick_params(axis="both", labelsize=7)
        ax.tick_params(axis="x", labelrotation=90)
    fg.set_axis_labels("ai $\\rightarrow$", "lai $\\rightarrow$")
    fg.fig.suptitle(title, y=1.02)
    return plotting.finish(fg.fig, fout)


def plot_et_product_spread(spread: xr.Dataset, title: str = "", fout: Path | None = None):
    """
    One heatmap per (lai, pr, rns) product set of the std of bin mean ET across
    ET products (`et_product_spread`), on a shared colour scale. Hatched bins
    lack a bin mean from at least one ET product.
    """
    groups = spread.lai_pr_rns.values
    fig, axs = plt.subplots(
        1, len(groups), figsize=(5.5 * len(groups), 4.5), squeeze=False, layout="constrained",
    )
    units = spread.attrs.get("units", "?")
    vmax = float(spread["et_std"].quantile(0.98))
    for ax, g in zip(axs.ravel(), groups):
        s = spread.sel(lai_pr_rns=g)
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
    mask = rg.land_mask(TARGET_GRID).sel(lat=config.LAT_BNDS)

    # ------------------------------------------------------------------
    # Load annual means of every product
    # ------------------------------------------------------------------
    print("=== Load ILAMB products ===")
    ann = {
        v: {p: load_product(v, p) for p in (RUN_PRODUCTS.get(v) or products)}
        for v, products in PRODUCTS.items()
    }
    run(ann, mask)


def run(ann: dict[str, dict[str, xr.DataArray]], mask: xr.DataArray):
    """
    Steps 2-4 for annual means `ann` ({variable: {product: (year, lat, lon)}},
    variables in FACTORS order, on TARGET_GRID within LAT_BNDS) and land `mask`.
    Also used by `obs_binned_et.py`.
    """
    # ------------------------------------------------------------------
    # Maps of climatological means
    # ------------------------------------------------------------------
    print("\n=== Maps of climatological means ===")
    for v, products in ann.items():
        for p, da in products.items():
            save_map(temporal.aggregate(da, "clim"), v, p, temporal.period_str(da.year.values), mask)

    for pr_p, rns_p in itertools.product(ann["pr"], ann["rns"]):
        years = temporal.shared_years(ann["pr"][pr_p], ann["rns"][rns_p])
        if not years:
            print(f"{pr_p}-{rns_p}: no shared years, skipping AI map")
            continue
        ai = binning.compute_aridity_index(
            temporal.aggregate(ann["pr"][pr_p].sel(year=years), "clim"),
            temporal.aggregate(ann["rns"][rns_p].sel(year=years), "clim"),
        )
        save_map(ai, "ai", f"{pr_p}-{rns_p}", temporal.period_str(years), mask)

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
    area = common_area(inputs, mask)
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
        pooled_edges[v] = binning.pooled_bin_edges(
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
            edges[v] = binning.pooled_bin_edges(
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
        plot_combo_bin_means(bs_all, title=f"obs ET bin mean, {ncombo} combinations, {kind} edges", fout=fout)
        print(fout)

        # Hatch non-empty bins whose mean is not significantly different from 0
        signif = binning.test_significance(bs_all, alpha=SIGNIF_ALPHA, n_min=SIGNIF_N_MIN)
        not_signif = ~signif & bs_all.sel(stats="mean").notnull()
        print(f"{kind}: {int(not_signif.sum())} of {int(bs_all.sel(stats='mean').notnull().sum())} "
              f"non-empty bins not significant at {1 - SIGNIF_ALPHA:.0%}")
        fout = FIG_ROOT / "obs" / "qbin" / f"{fstem}.bin_mean_signif.png"
        plot_combo_bin_means(
            bs_all, fout=fout, hatch=not_signif,
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
