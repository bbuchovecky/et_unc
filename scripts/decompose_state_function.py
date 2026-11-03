"""
decompose_state_function.py
===========================
How much of each dataset's area-mean ET difference from the observations comes
from where its land sits in LAI-AI space ("state"), how much from the ET it has
at a given LAI and AI ("function"), and how much from the two together
("interaction")? An analysis of the step 14 outputs, not part of the pipeline
(`scripts/run_all.sh` does not run it).

Inputs: the land-area weighted climatological ET binned with the pooled_all
edges (bin_all_clim.py with ETUNC_AREA_WEIGHTED=1, ETUNC_BIN_VARIABLE=et), per
member, for the reference family obs and the evaluated families (cfg.FAMILIES
without obs, and TRENDY_FAMILIES). From each member's bins (`decomposition`):

    f(b) = bin-mean ET [W m-2]     p(b) = fraction of the domain's land area in bin b
    E    = sum_b p f = area-mean ET over the common area mask

For a dataset m against an obs reference r, on the common support S (bins with
ET in both), four counterfactual domain means [W m-2] (state, then function):

    E_rr = sum_S p_r f_r   observed state, observed function
    E_rm = sum_S p_r f_m   the ET m's ET response would give on the observed state
    E_mr = sum_S p_m f_r   the ET the observed response would give on m's state
    E_mm = sum_S p_m f_m   m's state, m's function

and the terms (every difference m minus r):

    FUNCTION    = E_rm - E_rr     STATE = E_mr - E_rr
    INTERACTION = E_mm - E_rm - E_mr + E_rr  (> 0: the biases reinforce; < 0: they compensate)
    FUNCTION + STATE + INTERACTION = E_mm - E_rr

with the full-domain means E_m and E_r and the occupancy outside S,
OFF_SUPPORT_m/_r (the off-support ET is E_m - E_mm and E_r - E_rr). The bins
only condition the sums; no per-bin contribution is saved.

Comparisons
1. Every member of every evaluated family, and each family pooled
   ("<family>_pooled", each source weighted once, the members of a source
   averaged first, so E_pool = mean_s E_s), against each obs source and the
   obs pooled reference ("obs_pooled", weighted by OBS_POOL_WEIGHTING).
2. Observational spread: each obs source against obs_pooled. STATE and
   INTERACTION measure the observed-state uncertainty, FUNCTION the ET-response
   uncertainty. For each evaluated dataset (against obs_pooled), whether each
   term lies within the obs-source range of that term.
3. Sensitivity to the obs reference: for each evaluated dataset and quantity,
   the range across the obs sources, overall and by product axis (the range
   across one axis's products with the others fixed, averaged over the fixed
   combinations; axes with one product skipped).
4. Variance partition across the sources of each family with more than one
   source, against obs_pooled (ddof=0): Var(E_mm - E_rr) = Var(STATE) +
   Var(FUNCTION) + Var(INTERACTION) + 2 Cov of each pair. This is Var(E_mm)
   when E_rr is the same for every source (a complete support); Var(E_mm) and
   Var(E_rr) are saved alongside.
5. Bin-count sensitivity: the variance partitions of every bin count already
   run (folders under all/decomp/<period>/), with fraction shifts > 10 points flagged.

Obs product combinations are read from the data (the <axis>_product coords of
prep_obs.py, for each axis of OBS_PRODUCT_AXES): adding obs LAI, P or Rn
products needs no change here. Rerun the pipeline, then this script.

Interpretation
- Domain means are over the common area mask (make_edges.py), not all land.
- The terms are reference-weighted by design: FUNCTION uses the observed state
  and STATE the observed response, so both inherit the reference's uncertainty
  (comparison 3).
- The observed state is sampled only as far as the obs sources vary in LAI, P
  and Rn; an axis with one product adds no observational state uncertainty, so
  the obs spread understates it (attrs `obs_varying_axes`). Each obs source's
  climatological LAI, P and Rn are over its own shared years (prep_obs.py), so
  sources that differ only in their ET product can still differ in state when
  their years differ.
- In coupled runs LAI and AI respond to ET (P recycling, Rn, prognostic LAI):
  this is an accounting identity on a shared partition of a shared domain, not
  a causal attribution. FUNCTION includes every control not on the two axes
  (soil moisture, seasonality, VPD, rooting depth).
- TRENDY (trendy_s3, trendy_s2) are offline DGVMs, one member per model, all
  driven by the CRU-JRA P and given the ERA5 Rn: the AI is identical across
  TRENDY models (their state differs only in LAI), and prescribed (Rn not from
  the models' own energy balance), so their STATE term holds no climate feedback.
  ippe is offline CLM too (P prescribed). trendy_s3 and ippe spread is
  structural vs. parametric; the lens spread is the internal-variability floor.

Outputs
-------
<bins>   : lai<N_YBINS>_ai<N_XBINS>, the bin counts, e.g. "lai15_ai15"
<grid>   : grid tag of TARGET_RES, e.g. "1deg"
<period> : YYYYMM-YYYYMM of TIME_SLICE
<stem>   : all.et_clim.area_weighted.decomp_<kind>.qbin_pooled_all

PROC_ROOT/all/decomp/<period>/<bins>/<stem: vs_obs>.<bins>.<grid>.<period>.nc
    dc.QUANTITIES (E_rr, E_rm, E_mr, E_mm, FUNCTION, STATE, INTERACTION, E_m, E_r,
    OFF_SUPPORT_m, OFF_SUPPORT_r) along (dataset, obs_reference); `dataset` with coords
    family, source and <axis>_product ("none" outside obs), `obs_reference` (each obs source
    and "obs_pooled") with ref_<axis>_product ("pooled" for obs_pooled); obs_spread_min/max
    (quantity) and within_obs_spread (dataset, term) of comparison 2
PROC_ROOT/all/decomp/<period>/<bins>/<stem: vs_obs>.<bins>.<grid>.<period>.csv
    the quantities, one row per dataset x obs_reference, with the coords
PROC_ROOT/all/decomp/<period>/<bins>/<stem: obs_axis_range>.<bins>.<grid>.<period>.nc
    range (dataset, quantity, axis), axis = "all" and each product axis with > 1 product
PROC_ROOT/all/decomp/<period>/<bins>/<stem: variance>.<bins>.<grid>.<period>.nc
    variance and fraction (family, component), within_lens (ppe_family, component)
PROC_ROOT/all/decomp/<period>/<stem: variance>.all_bins.<grid>.<period>.nc
    variance and fraction (family, component, bins), fraction_shift and shift_gt_10pp
FIG_ROOT/all/decomp/<period>/<bins>/<stem: vs_obs>.<bins>.<grid>.<period>.<family>.png
    each dataset of a family against obs_pooled: E_rr, E_rm, E_mr, E_mm (top), then
    FUNCTION, STATE, INTERACTION, one panel each, with E_mm - E_rr marked
FIG_ROOT/all/decomp/<period>/<bins>/<stem: obs_axis_range>.<bins>.<grid>.<period>.png
    the ranges of comparison 3, one panel per quantity
"""

from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import etunc.config as cfg
import etunc.decomposition as dc
import etunc.grid as rg
import etunc.plotting as plotting
import etunc.temporal as temporal

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

# Pinned, independent of cfg: sum_b p f is the area-mean ET only for area-weighted bins of ET,
# and only a shared partition (the pooled_all edges) makes the datasets comparable bin by bin
AREA_WEIGHTED = True
BIN_VARIABLE = "et"
EDGES = "pooled_all"

REFERENCE_FAMILY = "obs"
TRENDY_FAMILIES = ("trendy_s3", "trendy_s2")
OPTIONAL_FAMILIES = ("trendy_s2",)  # skipped with a note if not binned
EVALUATED_FAMILIES = tuple(dict.fromkeys(
    [f for f in cfg.FAMILIES if f != REFERENCE_FAMILY] + list(TRENDY_FAMILIES)))
OBS_PRODUCT_AXES = ("et", "lai", "pr", "rn")
OBS_POOL_WEIGHTING = "source"  # one weight per obs source (dc.source_weights)
LENS_FAMILY = "lens"            # internal-variability floor of comparison 4
PPE_FAMILIES = ("fppe", "ippe")  # compared with it
SHIFT_THRESHOLD = 0.10          # comparison 5: flag fraction shifts across bin counts above this

TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES
N_YBINS = cfg.N_YBINS        # LAI
N_XBINS = cfg.N_XBINS        # AI

# The three terms (comparison 2 and the figures), and the four counterfactual means
TERMS = ("FUNCTION", "STATE", "INTERACTION")
MEANS = ("E_rr", "E_rm", "E_mr", "E_mm")
COLORS = {"E_rr": "#808080", "E_rm": "#eb6834", "E_mr": "#2a78d6", "E_mm": "#6250d6",
          "FUNCTION": "#eb6834", "STATE": "#2a78d6", "INTERACTION": "#1baf7a"}
UNITS = {q: "1" if q.startswith("OFF_SUPPORT") else "W m-2" for q in dc.QUANTITIES}


# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------

def stem(kind: str) -> str:
    return f"all.{BIN_VARIABLE}_clim.area_weighted.decomp_{kind}.qbin_{EDGES}"


def load_binned(family: str, bins: str, grid: str, period: str, area_path) -> xr.DataArray | None:
    """The area-weighted pooled_all clim ET bins of bin_all_clim.py; None if optional and missing."""
    path = (cfg.output_dir(cfg.PROC_ROOT, family, "qbin", period, bins)
            / f"{family}.{BIN_VARIABLE}_clim.area_weighted.qbin_{EDGES}.{bins}.{grid}.{period}.nc")
    if not path.exists():
        if family in OPTIONAL_FAMILIES:
            print(f"{family}: no {path.name}, skipped")
            return None
        raise FileNotFoundError(
            f"No {path}; run bin_all_clim.py for {family} with ETUNC_AREA_WEIGHTED=1 ETUNC_BIN_VARIABLE={BIN_VARIABLE} "
            f"ETUNC_N_YBINS={N_YBINS} ETUNC_N_XBINS={N_XBINS} (pixi run bin-all-clim)")
    bs = xr.load_dataarray(path)
    # The decomposition needs land-area weights, the pooled_all edges and the current common area mask
    if bs.attrs.get("weights") != "land_area" or bs.attrs.get("edges") != EDGES or bs.attrs.get("area_mask") != str(area_path):
        raise ValueError(f"{path}: weights {bs.attrs.get('weights')}, edges {bs.attrs.get('edges')}, "
                         f"area mask {bs.attrs.get('area_mask')}; expected land_area, {EDGES}, {area_path}")
    print(f"{family}: {bs.sizes['member']} members, {len(set(bs['source'].values.tolist()))} sources  {path}")
    return bs


def select(ds, cond: xr.DataArray):
    """The entries of `ds` along `dataset` where `cond` (along `dataset`) is True."""
    return ds.isel(dataset=np.flatnonzero(cond.values))


def label_datasets(p: xr.DataArray, f: xr.DataArray, dim: str, labels, family: str, sources,
                   products: dict | None) -> tuple[xr.DataArray, xr.DataArray]:
    """p and f along `dataset`, with the family, source and obs product coords ("none" outside obs)."""
    coords = {"dataset": list(labels), "family": ("dataset", [family] * len(labels)), "source": ("dataset", list(sources))}
    for a in OBS_PRODUCT_AXES:
        coords[f"{a}_product"] = ("dataset", list(products[a]) if products else ["none"] * len(labels))
    p, f = (da.reset_coords(drop=True).rename({dim: "dataset"}).assign_coords(coords) for da in (p, f))
    return p, f


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_family(vs_obs: xr.Dataset, lo: xr.Dataset, hi: xr.Dataset, family: str, lens_range: dict | None,
                title: str, fout):
    """
    Each dataset of `family` against obs_pooled. Top: the four counterfactual means, with the
    range of the obs sources' E shaded. Below, one panel per term: bars, with E_mm - E_rr
    marked and the obs-source range of the term shaded (and the lens range, for the PPE
    families). Error bars: min to max across the obs sources as references (`lo`, `hi`).
    """
    ds = select(vs_obs.sel(obs_reference="obs_pooled"), vs_obs["family"] == family)
    labels = [str(d).split(".", 1)[-1] for d in ds["dataset"].values]
    x = np.arange(len(labels))

    def errors(q):
        v = ds[q].values
        return v, np.abs(np.stack([v - lo[q].sel(dataset=ds["dataset"]).values,
                                   hi[q].sel(dataset=ds["dataset"]).values - v]))

    fig, axs = plt.subplots(1 + len(TERMS), 1, figsize=(max(6.0, 0.22 * len(x) + 2.5), 10.5), sharex=True,
                            layout="constrained", gridspec_kw={"height_ratios": [1.3] + [1] * len(TERMS)})
    # Top: E_rr, E_rm, E_mr, E_mm, offset within each dataset
    ax = axs[0]
    ax.axhspan(float(vs_obs["obs_spread_min"].sel(quantity="E_m")), float(vs_obs["obs_spread_max"].sel(quantity="E_m")),
               color="0.88", zorder=0, label="E of the obs sources (range)")
    for k, q in enumerate(MEANS):
        v, err = errors(q)
        xk = x + (k - 1.5) * 0.18
        ax.errorbar(xk, v, yerr=err, fmt="none", ecolor=COLORS[q], elinewidth=0.7, alpha=0.6, zorder=2)
        ax.scatter(xk, v, s=16, color=COLORS[q], edgecolor="white", linewidth=0.4, zorder=3, label=f"{q} vs obs_pooled")
    ax.set_ylabel("domain-mean ET [W m$^{-2}$]")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7, frameon=False)
    # Below: one panel per term
    for ax, term in zip(axs[1:], TERMS):
        ax.axhspan(float(vs_obs["obs_spread_min"].sel(quantity=term)), float(vs_obs["obs_spread_max"].sel(quantity=term)),
                   color="0.85", zorder=0, label="obs sources vs obs_pooled (range)")
        if lens_range is not None:
            ax.axhspan(*lens_range[term], facecolor="none", edgecolor="0.45", hatch="///", lw=0, zorder=0,
                       label="lens members (range)")
        v, err = errors(term)
        ax.bar(x, v, width=0.7, color=COLORS[term], edgecolor="white", linewidth=0.5, zorder=2, label=f"{term} vs obs_pooled")
        ax.errorbar(x, v, yerr=err, fmt="none", ecolor="0.25", elinewidth=0.8, capsize=1.5, zorder=3,
                    label="range across obs sources")
        ax.scatter(x, (ds["E_mm"] - ds["E_rr"]).values, marker="_", s=60, color="0.1", linewidth=1.5, zorder=4,
                   label="E_mm - E_rr")
        ax.axhline(0, color="0.4", lw=0.6, zorder=1)
        ax.set_ylabel(f"{term} [W m$^{{-2}}$]")
    for ax in axs:
        ax.grid(axis="y", color="0.92", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7, frameon=False)
    axs[-1].set_xticks(x, labels, rotation=90, fontsize=6)
    fig.suptitle(title, fontsize=9)
    return plotting.finish(fig, fout)


def plot_axis_ranges(axis_range: xr.DataArray, title: str, fout):
    """One panel per quantity: the range across obs references of each evaluated dataset, by product axis."""
    families = list(dict.fromkeys(axis_range["family"].values.tolist()))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    markers = ["o", "s", "D", "^", "v", "P", "X"]
    quantities = list(axis_range["quantity"].values)
    axes = list(axis_range["axis"].values)
    ncol = 4
    nrow = -(-len(quantities) // ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.4 * ncol + 1.5, 2.6 * nrow), squeeze=False, layout="constrained")
    width = 0.8 / len(families)
    for ax, q in zip(axs.flat, quantities):
        for i, fam in enumerate(families):
            sel = select(axis_range.sel(quantity=q), axis_range["family"] == fam)
            for j, a in enumerate(axes):
                v = sel.sel(axis=a).values
                xj = j + (i - (len(families) - 1) / 2) * width
                ax.scatter(np.full(v.size, xj), v, s=12, color=colors[i % len(colors)], marker=markers[i % len(markers)],
                           edgecolor="white", linewidth=0.4, alpha=0.8, zorder=2, label=fam if j == 0 else None)
        ax.set_xticks(range(len(axes)), axes)
        ax.set_title(f"{q} [{UNITS[q]}]", fontsize=8)
        ax.grid(axis="y", color="0.92", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs.flat[len(quantities):]:
        ax.axis("off")
    for ax in axs[:, 0]:
        ax.set_ylabel("range across obs references", fontsize=8)
    fig.legend(*axs.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=len(families),
               fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=9)
    return plotting.finish(fig, fout)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.summarize_settings(globals())
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)
    bins = cfg.bins_tag(N_YBINS, N_XBINS)  # bin counts, in the folder and file names
    out_dir = cfg.output_dir(cfg.PROC_ROOT, "all", "decomp", period, bins)
    fig_dir = cfg.output_dir(cfg.FIG_ROOT, "all", "decomp", period, bins)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Common area mask of all families (make_edges.py), which every binned file must have used
    area_path = cfg.area_mask_path(grid, period)
    if not area_path.exists():
        raise FileNotFoundError(f"No common area mask at {area_path}; run scripts/make_edges.py first")

    print("\n=== inputs ===")
    binned = {f: load_binned(f, bins, grid, period, area_path) for f in (REFERENCE_FAMILY,) + EVALUATED_FAMILIES}
    binned = {f: bs for f, bs in binned.items() if bs is not None}
    # Every dataset must share the LAI-AI partition
    ref_edges = (binned[REFERENCE_FAMILY].attrs["y_bin_edges"], binned[REFERENCE_FAMILY].attrs["x_bin_edges"])
    for f, bs in binned.items():
        if not (np.array_equal(bs.attrs["y_bin_edges"], ref_edges[0]) and np.array_equal(bs.attrs["x_bin_edges"], ref_edges[1])):
            raise ValueError(f"{f} was binned with other {EDGES} edges than {REFERENCE_FAMILY}; rerun bin_all_clim.py")

    # ------------------------------------------------------------------
    # Obs references: each obs source, and the obs sources pooled
    # ------------------------------------------------------------------
    print("\n=== obs references ===")
    obs_bs = binned[REFERENCE_FAMILY]
    missing = [a for a in OBS_PRODUCT_AXES if f"{a}_product" not in obs_bs.coords]
    if missing:
        raise KeyError(f"obs inputs have no {[f'{a}_product' for a in missing]} coords; rerun scripts/prep_obs.py")
    p_obs, f_obs = dc.members_to_sources(*dc.occupancy_and_mean(obs_bs))
    obs_sources = [str(s) for s in p_obs["source"].values]
    products = {a: [str(v) for v in p_obs[f"{a}_product"].values] for a in OBS_PRODUCT_AXES}
    varying = [a for a in OBS_PRODUCT_AXES if len(set(products[a])) > 1]
    for a in OBS_PRODUCT_AXES:
        print(f"product axis {a}: {len(set(products[a]))} products {sorted(set(products[a]))}")
    full_factorial = dc.is_full_factorial(products)
    print(f"obs product combinations: {len(obs_sources)}, full factorial: {full_factorial}")

    # ... pooled with one weight per source; the weight this gives each product
    weights = dc.source_weights(obs_sources, OBS_POOL_WEIGHTING)
    eff = dc.product_weights(weights, products)
    for a, w in eff.items():
        print(f"effective weight of each {a} product in obs_pooled: " + ", ".join(f"{q} {v:.3f}" for q, v in w.items()))
    p_obs_pool, f_obs_pool = dc.pool(p_obs, f_obs, weights.assign_coords(source=p_obs["source"]), "source")

    ref_coords = {"obs_reference": obs_sources + ["obs_pooled"]}
    for a in OBS_PRODUCT_AXES:
        ref_coords[f"ref_{a}_product"] = ("obs_reference", products[a] + ["pooled"])
    p_ref, f_ref = (
        xr.concat([src.reset_coords(drop=True).drop_vars("source").rename(source="obs_reference"),
                   pool.expand_dims(obs_reference=1)],
                  dim="obs_reference").assign_coords(ref_coords)
        for src, pool in ((p_obs, p_obs_pool), (f_obs, f_obs_pool))
    )

    # Attrs of every output: the obs configuration, so runs with other obs products are told apart
    attrs = {
        "time_period": period, "grid": grid, "bins": bins, "edges": EDGES, "area_mask": str(area_path),
        "et_samples": "climatological ET per gridcell, bins weighted by land area (bin_all_clim.py)",
        "difference": "dataset minus obs reference",
        "counterfactual_means": "E_xy = sum over the common support of p_x f_y (x: state, y: function; m dataset, r reference)",
        "obs_sources": [f"{s}: " + ", ".join(f"{a}={products[a][i]}" for a in OBS_PRODUCT_AXES)
                        for i, s in enumerate(obs_sources)],
        "obs_product_axes": list(OBS_PRODUCT_AXES),
        "obs_varying_axes": varying,
        "obs_full_factorial": int(full_factorial),
        "obs_pool_weighting": OBS_POOL_WEIGHTING,
        "obs_effective_product_weights": [f"{a}: " + ", ".join(f"{q}={v:.4f}" for q, v in w.items()) for a, w in eff.items()],
        "domain": "common area mask of all families, not all land",
        "note_reference": "FUNCTION uses the observed state and STATE the observed response: both inherit the reference's uncertainty",
        "note_obs_state": ("the observed state is sampled only as far as the obs sources vary in LAI, P and Rn "
                           "(obs_varying_axes); each obs source's LAI, P and Rn climatology is over its own years"),
        "note_causality": "accounting identity on a shared partition of a shared domain, not a causal attribution",
        "note_function": "FUNCTION includes every control not on the LAI and AI axes (soil moisture, seasonality, VPD, rooting depth)",
        "note_offline": "ippe and TRENDY are offline (P prescribed); TRENDY Rn is ERA5, so its AI is shared and its STATE term holds no climate feedback",
    }

    # ------------------------------------------------------------------
    # Datasets: every member of every family, and each evaluated family pooled
    # ------------------------------------------------------------------
    print("\n=== datasets ===")
    p_list, f_list, p_sources, f_sources = [], [], {}, {}
    for family, bs in binned.items():
        p_mem, f_mem = dc.occupancy_and_mean(bs)
        if family == REFERENCE_FAMILY:
            # The obs sources as datasets (one member each, the same as the obs references)
            p, f = label_datasets(p_obs, f_obs, "source", [f"obs.{s}" for s in obs_sources], family, obs_sources, products)
        else:
            members = [str(m) for m in bs["member"].values]
            p, f = label_datasets(p_mem, f_mem, "member", [f"{family}.{m}" for m in members], family,
                                  [str(s) for s in bs["source"].values], None)
        p_list.append(p)
        f_list.append(f)
        # Members averaged within each source, then sources pooled with equal weights
        p_src, f_src = dc.members_to_sources(p_mem, f_mem)
        p_sources[family], f_sources[family] = p_src, f_src
        if family != REFERENCE_FAMILY:
            w = dc.source_weights(p_src["source"].values, "source").assign_coords(source=p_src["source"])
            p_pool, f_pool = dc.pool(p_src, f_src, w, "source")
            E_pool = float((p_pool * f_pool).sum())
            E_src = (p_src * f_src).sum(dc.BIN_DIMS)
            print(f"{family}_pooled: E {E_pool:.4f} W m-2, mean of the {E_src.sizes['source']} sources' E "
                  f"{float(E_src.mean()):.4f}")
            p, f = label_datasets(p_pool.expand_dims(pooled=1), f_pool.expand_dims(pooled=1), "pooled",
                                  [f"{family}_pooled"], family, ["pooled"], None)
            p_list.append(p)
            f_list.append(f)
    p_ds, f_ds = xr.concat(p_list, dim="dataset"), xr.concat(f_list, dim="dataset")
    ds_coords = {c: p_ds[c] for c in p_ds.coords if c != "dataset" and p_ds[c].dims == ("dataset",)}

    # ------------------------------------------------------------------
    # 1. Every dataset against every obs reference
    # ------------------------------------------------------------------
    print("\n=== 1. datasets vs obs references ===")
    vs_obs = dc.decompose(p_ds, f_ds, p_ref, f_ref).assign_coords(ds_coords).assign_coords(ref_coords)
    vs_obs = vs_obs.transpose("dataset", "obs_reference")
    for q in dc.QUANTITIES:
        vs_obs[q].attrs["units"] = UNITS[q]
    pooled_rows = select(vs_obs.sel(obs_reference="obs_pooled"), vs_obs["source"] == "pooled")
    print(f"E of obs_pooled: {float(vs_obs['E_r'].sel(obs_reference='obs_pooled').isel(dataset=0)):.3f} W m-2")
    print(f"{'vs obs_pooled':18} {'E_rr':>8} {'E_rm':>8} {'E_mr':>8} {'E_mm':>8} {'FUNCTION':>9} {'STATE':>8} "
          f"{'INTERACT':>9} {'E_m':>8} {'OFF_m':>7} {'OFF_r':>7}")
    for d in pooled_rows["dataset"].values:
        r = pooled_rows.sel(dataset=d)
        print(f"{d:18} {float(r.E_rr):8.3f} {float(r.E_rm):8.3f} {float(r.E_mr):8.3f} {float(r.E_mm):8.3f} "
              f"{float(r.FUNCTION):9.3f} {float(r.STATE):8.3f} {float(r.INTERACTION):9.3f} {float(r.E_m):8.3f} "
              f"{float(r.OFF_SUPPORT_m):7.4f} {float(r.OFF_SUPPORT_r):7.4f}")

    # ------------------------------------------------------------------
    # 2. Observational spread: each obs source against obs_pooled
    # ------------------------------------------------------------------
    print("\n=== 2. obs spread ===")
    is_obs = vs_obs["family"] == REFERENCE_FAMILY
    obs_spread = select(vs_obs[list(dc.QUANTITIES)].sel(obs_reference="obs_pooled"), is_obs).to_array("quantity")
    spread_min, spread_max = obs_spread.min("dataset"), obs_spread.max("dataset")
    for q in dc.QUANTITIES:
        print(f"obs sources vs obs_pooled, {q:13}: [{float(spread_min.sel(quantity=q)):7.3f}, "
              f"{float(spread_max.sel(quantity=q)):7.3f}] {UNITS[q]}")
    # STATE and INTERACTION: observed-state uncertainty; FUNCTION: ET-response uncertainty
    terms_pooled = vs_obs[list(TERMS)].sel(obs_reference="obs_pooled", drop=True).to_array("term")
    lo_t = spread_min.sel(quantity=list(TERMS)).rename(quantity="term")
    hi_t = spread_max.sel(quantity=list(TERMS)).rename(quantity="term")
    within = ((terms_pooled >= lo_t) & (terms_pooled <= hi_t)).where(~is_obs, False)
    vs_obs["obs_spread_min"], vs_obs["obs_spread_max"] = spread_min, spread_max
    vs_obs["within_obs_spread"] = within.transpose("dataset", "term")
    vs_obs["within_obs_spread"].attrs["description"] = (
        "term of the dataset vs obs_pooled within the range of the obs sources vs obs_pooled (False for obs)")
    for family in EVALUATED_FAMILIES:
        if family not in binned:
            continue
        w = select(within, vs_obs["family"] == family)
        print(f"{family:10} datasets within the obs spread: " + ", ".join(
            f"{t} {int(w.sel(term=t).sum())}/{w.sizes['dataset']}" for t in TERMS))
    vs_obs.attrs = {**attrs, "comparison": "1. every dataset vs every obs reference; 2. obs spread (vs obs_pooled)"}
    fout = out_dir / f"{stem('vs_obs')}.{bins}.{grid}.{period}.nc"
    vs_obs.to_netcdf(fout)
    print(fout)
    fout = fout.with_suffix(".csv")
    vs_obs[list(dc.QUANTITIES)].to_dataframe().to_csv(fout)
    print(fout)

    # ------------------------------------------------------------------
    # 3. Sensitivity to the obs reference, by product axis
    # ------------------------------------------------------------------
    print("\n=== 3. obs reference sensitivity ===")
    evaluated = select(vs_obs, ~is_obs).drop_sel(obs_reference="obs_pooled")
    per_ref = evaluated[list(dc.QUANTITIES)].to_array("quantity")
    axis_range = dc.product_axis_range(per_ref, "obs_reference", OBS_PRODUCT_AXES, products).rename("range")
    axis_range.attrs = {"units": "W m-2 (OFF_SUPPORT_*: 1)", "description": (
        "max - min across the obs sources; for a product axis, across its products with the other axes "
        "fixed, averaged over the fixed combinations")}
    out = axis_range.transpose("dataset", "quantity", "axis").to_dataset()
    out.attrs = {**attrs, "comparison": "3. range across obs references, by product axis"}
    for a in out["axis"].values:
        print(f"axis {a:4}: median range across datasets, " + ", ".join(
            f"{q} {float(out['range'].sel(axis=a, quantity=q).median()):.3f}" for q in MEANS + TERMS))
    fout = out_dir / f"{stem('obs_axis_range')}.{bins}.{grid}.{period}.nc"
    out.to_netcdf(fout)
    print(fout)

    # ------------------------------------------------------------------
    # 4. Variance partition across the sources of each family, against obs_pooled
    # ------------------------------------------------------------------
    print("\n=== 4. variance partition across sources (vs obs_pooled, ddof=0) ===")
    partitions = {}
    for family, p_src in p_sources.items():
        if p_src.sizes["source"] < 2:
            print(f"{family}: one source, skipped")
            continue
        terms = dc.decompose(p_src, f_sources[family], p_obs_pool, f_obs_pool)
        partitions[family] = dc.variance_partition(terms, "source")
    variance = xr.concat(list(partitions.values()), dim="family").assign_coords(family=list(partitions))
    variance = variance.transpose("family", "component")
    # The PPE components within the lens values: not distinguishable from internal variability
    ppe = [f for f in PPE_FAMILIES if f in partitions]
    if LENS_FAMILY in partitions and ppe:
        lens_var = abs(variance["variance"].sel(family=LENS_FAMILY))
        variance["within_lens"] = xr.concat([abs(variance["variance"].sel(family=f)) <= lens_var for f in ppe],
                                            dim="ppe_family").assign_coords(ppe_family=ppe)
        variance["within_lens"].attrs["description"] = f"|component| <= |component of {LENS_FAMILY}|"
    variance.attrs = {
        **attrs, "comparison": "4. variance across the sources of each family, all vs obs_pooled", "ddof": 0,
        "partition": ("var_E_mm_minus_E_rr = var_STATE + var_FUNCTION + var_INTERACTION + 2cov_STATE_FUNCTION + "
                      "2cov_STATE_INTERACTION + 2cov_FUNCTION_INTERACTION; fraction = component / var_E_mm_minus_E_rr. "
                      "Equal to var_E_mm when E_rr is the same for every source (complete support): var_E_mm and "
                      "var_E_rr alongside"),
        "note_obs": "the obs partition separates observed-state (STATE) from ET-product (FUNCTION) uncertainty",
        "note_trendy": "Var(STATE) within TRENDY reflects LAI only (the AI is shared)",
        "note_offline": "trendy_s3 and ippe are offline: structural (TRENDY) vs. parametric (ippe) spread",
        "note_lens": "the lens partition is the internal-variability floor (within_lens for fppe and ippe)",
    }
    comps = ["var_E_mm_minus_E_rr", *dc.COMPONENTS, "var_E_mm", "var_E_rr"]
    print(f"{'variance':10} " + " ".join(f"{c.replace('2cov_', 'cv_').replace('_minus_', '-')[:14]:>14}" for c in comps))
    for family in variance["family"].values:
        r = variance.sel(family=family)
        print(f"{family:10} " + " ".join(f"{float(r['variance'].sel(component=c)):14.4f}" for c in comps))
    print(f"{'fraction':10} " + " ".join(f"{c.replace('2cov_', 'cv_')[:14]:>14}" for c in dc.COMPONENTS))
    for family in variance["family"].values:
        r = variance.sel(family=family)
        print(f"{family:10} " + " ".join(f"{float(r['fraction'].sel(component=c)):14.3f}" for c in dc.COMPONENTS))
    for f in (variance["ppe_family"].values if "within_lens" in variance else []):
        flagged = [str(c) for c in dc.COMPONENTS if bool(variance["within_lens"].sel(ppe_family=f, component=c))]
        print(f"{f}: components within the {LENS_FAMILY} values: {flagged or 'none'}")
    fout = out_dir / f"{stem('variance')}.{bins}.{grid}.{period}.nc"
    variance.to_netcdf(fout)
    print(fout)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    # Range of each dataset's quantities across the obs sources, and the lens range for the PPE families
    per_source = vs_obs[list(MEANS + TERMS)].drop_sel(obs_reference="obs_pooled")
    lo, hi = per_source.min("obs_reference"), per_source.max("obs_reference")
    lens_vs_pool = select(vs_obs.sel(obs_reference="obs_pooled"),
                          (vs_obs["family"] == LENS_FAMILY) & (vs_obs["source"] != "pooled"))
    lens_range = {t: (float(lens_vs_pool[t].min()), float(lens_vs_pool[t].max())) for t in TERMS}
    for family in EVALUATED_FAMILIES:
        if family not in binned:
            continue
        fout = fig_dir / f"{stem('vs_obs')}.{bins}.{grid}.{period}.{family}.png"
        plot_family(vs_obs, lo, hi, family, lens_range if family in PPE_FAMILIES else None, fout=fout,
                    title=f"{family} area-mean ET vs obs_pooled ({len(obs_sources)} obs sources), "
                          f"{EDGES} {bins}, {grid}, {period}")
        print(fout)
    fout = fig_dir / f"{stem('obs_axis_range')}.{bins}.{grid}.{period}.png"
    plot_axis_ranges(out["range"], fout=fout,
                     title=f"Range of each quantity across the obs references, by product axis, {bins}, {grid}, {period}")
    print(fout)

    # ------------------------------------------------------------------
    # 5. Bin-count sensitivity of the variance partitions
    # ------------------------------------------------------------------
    print("\n=== 5. bin-count sensitivity ===")
    decomp_dir = cfg.output_dir(cfg.PROC_ROOT, "all", "decomp", period)
    runs = {}
    for d in sorted(decomp_dir.iterdir()):
        m = re.fullmatch(r"lai(\d+)_ai(\d+)", d.name)
        path = d / f"{stem('variance')}.{d.name}.{grid}.{period}.nc"
        if m and path.exists():
            runs[d.name] = xr.load_dataset(path)[["variance", "fraction"]]
    if len(runs) < 2:
        print(f"one bin count ({list(runs)}), no cross-bin summary")
        return
    summary = xr.concat(list(runs.values()), dim="bins", join="outer").assign_coords(bins=list(runs))
    summary = summary.transpose("family", "component", "bins")
    summary["fraction_shift"] = summary["fraction"].max("bins") - summary["fraction"].min("bins")
    summary["shift_gt_10pp"] = summary["fraction_shift"] > SHIFT_THRESHOLD
    summary.attrs = {**attrs, "bins": list(runs), "comparison": "5. variance partition by bin count",
                     "shift_threshold": SHIFT_THRESHOLD}
    print(f"bin counts: {list(runs)}")
    for family in summary["family"].values:
        flagged = [f"{c} ({float(summary['fraction_shift'].sel(family=family, component=c)):.2f})"
                   for c in dc.COMPONENTS if bool(summary["shift_gt_10pp"].sel(family=family, component=c))]
        print(f"{family:10} fractions shifting by > {100 * SHIFT_THRESHOLD:.0f} percentage points: {flagged or 'none'}")
    fout = decomp_dir / f"{stem('variance')}.all_bins.{grid}.{period}.nc"
    summary.to_netcdf(fout)
    print(fout)


if __name__ == "__main__":
    main()
