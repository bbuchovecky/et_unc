"""
gridded_state_function_decomp.py
================================
Where on land does each dataset's ET difference from the observations come
from its state (where its land sits in LAI-AI space), its function (its ET at
a given LAI and AI), or the two together? The state-function decomposition of
decompose_state_function.py, mapped back from the LAI-AI bins to the lat/lon
grid. An analysis of the step 14 outputs, not part of the pipeline
(`scripts/run_all.sh` does not run it).

Run ID: RUN_ID (required) fixes the inputs and the settings, as in
decompose_state_function.py: those registered under it in config/runs.csv
(`cfg.run_settings`), not those of the current config. ETUNC_RUN_ID, if set,
replaces it. The run must have binning.area_weighted true and
binning.bin_variable et (e.g. the run of config/runs/aw_bins15.yml).

Inputs, per family (the reference family obs and the evaluated families, as in
decompose_state_function.py):
- the land-area weighted climatological ET binned with the pooled_all edges by
  bin_all_clim.py in run RUN_ID: the bin means f and occupancy p of each member;
- the binning inputs of prep_<family>.py (the run's PREP_ID): each member's
  climatological ET e (mean over the years, as bin_all_clim.py), LAI and AI,
  from which each gridcell's bin b(i) (`binning.bin_index`, the rule of `bin_stats`).

Method (`decomposition.decompose_gridded`): for a dataset m against an obs
reference r, on the common support S (bins with ET in both), each gridcell i
takes the bin means at its own bins in m and r, b_m(i) and b_r(i):

    E_xy(i)        = f_y(b_x(i))      x: state, y: function (0 where b_x(i) is off S)
    STATE(i)       = E_mr - E_rr = f_r(b_m) - f_r(b_r)   ET change from the gridcell's move in LAI-AI space
    FUNCTION(i)    = E_rm - E_rr = f_m(b_r) - f_r(b_r)   m's minus the observed ET at the observed bin
    INTERACTION(i) = E_mm - E_rm - E_mr + E_rr
    WITHIN(i)      = (e_m - E_mm) - (e_r - E_rr)          the difference of the within-bin anomalies
    DIFF(i)        = e_m - e_r = STATE + FUNCTION + INTERACTION + WITHIN

The land-area mean of each map over the common area is the scalar of
decompose_state_function.py (checked here, the script raises otherwise): the
STATE, FUNCTION and INTERACTION maps say where the area-mean terms come from.
WITHIN averages to 0 on a complete support (the bin mean is the area-weighted
mean of its gridcells), so it only moves ET around within a bin. OFF_SUPPORT_m/_r(i)
is the weight of the gridcell's bins off S (their area mean is OFF_SUPPORT_m/_r).

Pooled datasets: obs_pooled and <family>_pooled have no single LAI-AI map. Their
state is the mixture of their members' bin maps with the pool weights
(OBS_POOL_WEIGHTING, FAMILY_POOL_WEIGHTING: each group of variants once, each
source once within its group, each member of a source equally), and their e the
same weighted mean: E_xy(i) = sum_k w_k f_y(b_k(i)). Since p_pool = sum_k w_k p_k,
the area means are still those of the pooled datasets of decompose_state_function.py.

Every dataset (each member of each evaluated family and the family pooled, and
each obs source) is mapped against each obs source and obs_pooled. The figures
are against obs_pooled only.

Interpretation (see also decompose_state_function.py)
- A gridcell's STATE is f_r(b_m) - f_r(b_r): the observed ET of the bin the
  gridcell is in in m minus that of its bin in obs. Its area mean is STATE, but
  a gridcell's value is not "its share" of STATE in any unique sense: the
  attribution follows the gridcells, which carry their LAI and AI from one bin
  to another.
- FUNCTION includes every control not on the two axes; WITHIN holds what the
  bins do not resolve (the gridcell's ET differs from its bin's mean).
- Accounting identity on a shared partition of a shared domain (the common area
  mask), not a causal attribution.

Outputs
-------
<run_id> : RUN_ID (its settings in config/runs.csv)
<stem>   : all.et_clim.decomp_gridded.qbin_pooled_all

PROC_ROOT/<run_id>/all/decomp/<stem>.<family>.<run_id>.nc
    the maps of dc.GRIDDED_QUANTITIES (dataset, obs_reference, lat, lon), float32; `dataset` the members
    of the family and <family>_pooled (obs: each source), with coords family, source and <axis>_product,
    `obs_reference` each obs source and obs_pooled, with ref_<axis>_product
FIG_ROOT/<run_id>/all/decomp/<stem>.<family>.pooled.<run_id>.png
    <family>_pooled vs obs_pooled: DIFF, STATE, FUNCTION, INTERACTION, WITHIN (not for obs)
FIG_ROOT/<run_id>/all/decomp/<stem>.<family>.group_std.<run_id>.png
    std across the groups of variants (vs obs_pooled) of DIFF, STATE, FUNCTION, INTERACTION
    (families with > 1 group; for obs, the obs spread)
FIG_ROOT/<run_id>/all/decomp/<stem>.<family>.members_<term>.<run_id>.png
    every member's map of STATE, FUNCTION and INTERACTION vs obs_pooled
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs
import numpy as np
import xarray as xr

import etunc.binning as binning
import etunc.config as cfg
import etunc.decomposition as dc
import etunc.grid as rg
import etunc.plotting as plotting
import etunc.temporal as temporal
import etunc.variants as variants

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

# Required: the run ID whose settings fix the inputs (must be registered in config/runs.csv);
# ETUNC_RUN_ID, if set, replaces it. Default: the run of config/runs/aw_bins15.yml
RUN_ID = os.environ.get("ETUNC_RUN_ID") or "r59533f4"
_run = cfg.run_settings(RUN_ID)  # raises if RUN_ID is not registered
PREP_ID = next(r["prep_id"] for r in cfg.read_runs() if r["run_id"] == RUN_ID)  # its binning inputs

# Required of the run (checked in main), as in decompose_state_function.py
AREA_WEIGHTED = _run["binning"]["area_weighted"]  # must be True
BIN_VARIABLE = _run["binning"]["bin_variable"]    # must be "et"
EDGES = "pooled_all"

REFERENCE_FAMILY = "obs"
TRENDY_FAMILIES = ("trendy_s3", "trendy_s2")
OPTIONAL_FAMILIES = ("trendy_s2",)  # skipped with a note if not binned
EVALUATED_FAMILIES = tuple(dict.fromkeys(
    [f for f in _run["binning"]["families"] if f != REFERENCE_FAMILY] + list(TRENDY_FAMILIES)))
OBS_PRODUCT_AXES = ("et", "lai", "pr", "rn")
OBS_POOL_WEIGHTING = "variant"     # obs_pooled: each group of variants weighs once (dc.source_weights)
FAMILY_POOL_WEIGHTING = "variant"  # <family>_pooled: the same for the sources of each evaluated family

TIME_SLICE = slice(*_run["prep"]["time_slice"])
TARGET_RES = _run["prep"]["target_res"]
LAT_BNDS = slice(*_run["prep"]["lat_bnds"])
N_YBINS = _run["binning"]["n_ybins"]  # LAI
N_XBINS = _run["binning"]["n_xbins"]  # AI

TERMS = ("STATE", "FUNCTION", "INTERACTION")
POOLED_PANELS = ("DIFF",) + TERMS + ("WITHIN",)  # the pooled figure: DIFF = the sum of the other four
STD_PANELS = ("DIFF",) + TERMS
UNITS = {q: "1" if q.startswith("OFF_SUPPORT") else "W m-2" for q in dc.GRIDDED_QUANTITIES}
ROBUST_PCT = 98  # color scales: this percentile of |value| over the panels of a figure
ATOL = 1e-6      # W m-2 (OFF_SUPPORT: fraction): area mean of a map vs the scalar of dc.decompose


def stem() -> str:
    return f"all.{BIN_VARIABLE}_clim.decomp_gridded.qbin_{EDGES}"


# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------

def load_binned(family: str, area_path) -> xr.DataArray | None:
    """The area-weighted pooled_all clim ET bins of bin_all_clim.py in run RUN_ID; None if optional and missing."""
    path = cfg.output_dir(cfg.PROC_ROOT, family, "qbin", RUN_ID) / f"{family}.{BIN_VARIABLE}_clim.qbin_{EDGES}.{RUN_ID}.nc"
    if not path.exists():
        if family in OPTIONAL_FAMILIES:
            print(f"{family}: no {path.name}, skipped")
            return None
        raise FileNotFoundError(
            f"No {path}; run bin_all_clim.py for {family} in run {RUN_ID} (ETUNC_RUN=<its run file> "
            "pixi run bin-all-clim)")
    bs = xr.load_dataarray(path)
    if bs.attrs.get("weights") != "land_area" or bs.attrs.get("edges") != EDGES or bs.attrs.get("area_mask") != str(area_path):
        raise ValueError(f"{path}: weights {bs.attrs.get('weights')}, edges {bs.attrs.get('edges')}, "
                         f"area mask {bs.attrs.get('area_mask')}; expected land_area, {EDGES}, {area_path}")
    print(f"{family}: {bs.sizes['member']} members  {path}")
    return bs


def load_gridded(family: str, bs: xr.DataArray, area: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Each member's climatological ET e and LAI-AI bin map (member, lat, lon) in the common area, from the
    binning inputs of PREP_ID, binned with the edges of `bs`.
    """
    path = cfg.inputs_path(family, PREP_ID)
    if not path.exists():
        raise FileNotFoundError(f"No {family} inputs at {path}; run scripts/prep_{family}.py in run {RUN_ID}")
    inputs = xr.load_dataset(path)
    if [str(m) for m in inputs["member"].values] != [str(m) for m in bs["member"].values]:
        raise ValueError(f"{path}: members differ from those of the binned {family} file; rerun bin_all_clim.py")
    rg.check_same_grid(inputs["lai"], area, f"{family} inputs")
    keep = area.reindex_like(inputs["lai"], method="nearest", tolerance=1e-3)
    # As bin_all_clim.py: the mean over the years, NaN years skipped
    e = temporal.aggregate(inputs[BIN_VARIABLE], "clim", label=f"{family} {BIN_VARIABLE}").where(keep)
    bins = binning.bin_index(inputs["lai"].where(keep), inputs["ai"].where(keep),
                             bs.attrs["y_bin_edges"], bs.attrs["x_bin_edges"])
    bins = bins.where(keep & e.notnull(), -1)  # a gridcell without ET is no sample of bin_stats either
    # Only the member coords of the binned file (source, products), and the common lat/lon
    e, bins = (da.reset_coords(drop=True).assign_coords(lat=area.lat, lon=area.lon) for da in (e, bins))
    return e, bins


def check_occupancy(family: str, bins: xr.DataArray, p: xr.DataArray, land_area: xr.DataArray):
    """The land-area histogram of each member's bin map is its occupancy p of bin_stats, or raise."""
    a = land_area.values.ravel()
    for k in range(bins.sizes["member"]):
        idx = bins.isel(member=k).values.ravel()
        hist = np.bincount(idx[idx >= 0], weights=a[idx >= 0], minlength=p.isel(member=k).size)
        hist = hist / hist.sum()
        if not np.allclose(hist, p.isel(member=k).transpose(*dc.BIN_DIMS).values.ravel(), rtol=1e-8, atol=1e-12):
            raise ValueError(f"{family} member {str(bins['member'].values[k])}: the gridcell bins do not reproduce "
                             "the occupancy of bin_all_clim.py (other inputs, edges or mask?)")


def as_state(bins: xr.DataArray, e: xr.DataArray, weights) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Bin maps, weights and ET maps of a mixture of members, along `state` (dc.decompose_gridded)."""
    w = xr.DataArray(np.asarray(weights, dtype=float), dims="state")
    return bins.rename(member="state").drop_vars("state", errors="ignore"), w, e.rename(member="state").drop_vars(
        "state", errors="ignore")


def member_weights(p_mem: xr.DataArray, family: str, weighting: str) -> np.ndarray:
    """
    Weight of each member in <family>_pooled: the weight of its source (dc.source_weights, the groups
    of variants of variants.source_groups) shared equally by the source's members, as in
    dc.members_to_sources then dc.pool.
    """
    sources = [str(s) for s in p_mem["source"].values]
    p_src = p_mem.isel(member=[sources.index(s) for s in dict.fromkeys(sources)]).swap_dims(member="source")
    groups = variants.source_groups(family, p_src, "source")
    w_src = dc.source_weights(list(dict.fromkeys(sources)), weighting, groups=groups)
    return np.array([float(w_src.sel(source=s)) / sources.count(s) for s in sources])


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_maps(maps: dict, title: str, fout, *, diverging: bool = True, off_support: xr.DataArray | None = None,
              land_area: xr.DataArray | None = None, ncols: int = 2, panel_size=(5.5, 2.6), label: str = "W m$^{-2}$"):
    """
    One map per entry of `maps` ({panel title: (lat, lon)}), on one shared color scale (ROBUST_PCT of
    |value|): BrBG around 0 if `diverging`, else from 0. With `land_area`, each title gives the land-area
    mean; with `off_support` (lat, lon), gridcells with bins off the common support are hatched.
    """
    values = np.concatenate([np.abs(da.values[np.isfinite(da.values)]) for da in maps.values()])
    vmax = float(np.percentile(values, ROBUST_PCT)) if values.size else 1.0
    kwargs = ({"cmap": "BrBG", "vmin": -vmax, "vmax": vmax} if diverging else {"cmap": "viridis", "vmin": 0, "vmax": vmax})
    fig, axs = plotting.facets(len(maps), ncols=ncols, maps=True, panel_size=panel_size)
    for ax, (name, da) in zip(axs, maps.items()):
        mesh = da.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, add_labels=False, **kwargs)
        if off_support is not None and bool((off_support > 0).any()):
            ax.contourf(off_support["lon"], off_support["lat"], (off_support > 0).astype(float), levels=[0.5, 1.5],
                        colors="none", hatches=["...."], transform=ccrs.PlateCarree())
        if land_area is not None:
            name = f"{name}\narea mean {float((da * land_area).sum() / land_area.where(da.notnull()).sum()):.2f}"
        ax.set_title(name, fontsize=8 if len(maps) <= 6 else 6)
        plotting.map_ax(ax, LAT_BNDS)
    fig.colorbar(mesh, ax=axs, orientation="horizontal", shrink=0.5, aspect=40, pad=0.02, extend="both" if diverging else "max",
                 label=label)
    fig.suptitle(title, fontsize=9)
    return plotting.finish(fig, fout)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    # The settings come from RUN_ID, not the current config; RUN_ID is registered already
    cfg.summarize_settings(globals())
    if not AREA_WEIGHTED or BIN_VARIABLE != "et":
        raise ValueError(
            f"run {RUN_ID} has area_weighted {AREA_WEIGHTED} and bin_variable {BIN_VARIABLE!r}; the decomposition "
            "needs a run with binning.area_weighted: true and binning.bin_variable: et, e.g. "
            "the run of config/runs/aw_bins15.yml")
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)
    run = RUN_ID
    bins_tag = f"lai{N_YBINS}_ai{N_XBINS}"
    out_dir = cfg.output_dir(cfg.PROC_ROOT, "all", "decomp", run)
    fig_dir = cfg.output_dir(cfg.FIG_ROOT, "all", "decomp", run)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Common area mask (make_edges.py) and each gridcell's land area, the weights of bin_all_clim.py
    area_path = cfg.area_mask_path(RUN_ID)
    if not area_path.exists():
        raise FileNotFoundError(f"No common area mask at {area_path}; run scripts/make_edges.py first")
    area = xr.open_dataset(area_path)["area_mask"] == 1
    land_area = rg.land_area(rg.target_grid(TARGET_RES)).sel(lat=area.lat, lon=area.lon)
    rg.check_same_grid(land_area, area, "land area")
    land_area = land_area.where(area)  # NaN outside the common area

    def area_mean(da):
        return (da * land_area).sum(("lat", "lon")) / land_area.sum()

    # ------------------------------------------------------------------
    # Obs references: each obs source (one state), and obs_pooled (the mixture of every source)
    # ------------------------------------------------------------------
    print("\n=== obs references ===")
    obs_bs = load_binned(REFERENCE_FAMILY, area_path)
    p_obs, f_obs = (da.reset_coords(drop=True).assign_coords(
        {c: obs_bs[c] for c in obs_bs.coords if obs_bs[c].dims == ("member",)}) for da in dc.occupancy_and_mean(obs_bs))
    obs_sources = [str(s) for s in p_obs["source"].values]
    if len(set(obs_sources)) != len(obs_sources):
        raise ValueError("expected one obs member per source (prep_obs.py)")
    e_obs, bins_obs = load_gridded(REFERENCE_FAMILY, obs_bs, area)
    check_occupancy(REFERENCE_FAMILY, bins_obs, p_obs, land_area.fillna(0))
    products = {a: [str(v) for v in p_obs[f"{a}_product"].values] for a in OBS_PRODUCT_AXES}
    w_obs = member_weights(p_obs, REFERENCE_FAMILY, OBS_POOL_WEIGHTING)
    p_obs_pool, f_obs_pool = dc.pool(p_obs, f_obs, xr.DataArray(w_obs, dims="member"), "member")
    # Each reference: (p, f) of the bins, and its state (bin maps, weights, ET maps)
    refs = {s: (p_obs.isel(member=k), f_obs.isel(member=k),
                as_state(bins_obs.isel(member=[k]), e_obs.isel(member=[k]), [1.0]))
            for k, s in enumerate(obs_sources)}
    refs["obs_pooled"] = (p_obs_pool, f_obs_pool, as_state(bins_obs, e_obs, w_obs))
    ref_coords = {"obs_reference": list(refs)}
    for a in OBS_PRODUCT_AXES:
        ref_coords[f"ref_{a}_product"] = ("obs_reference", products[a] + ["pooled"])
    print(f"{len(obs_sources)} obs sources; obs_pooled weights " + ", ".join(f"{w:.3f}" for w in w_obs))

    attrs = {
        "time_period": period, "grid": grid, "bins": bins_tag, "run_id": run, "prep_id": PREP_ID, "edges": EDGES,
        "area_mask": str(area_path),
        "difference": "dataset minus obs reference",
        "maps": ("E_xy(i) = sum_k w_k f_y(b_k(i)) over the states k of x (b: the gridcell's LAI-AI bin; one state of "
                 "weight 1 for a single dataset), 0 where the bin is off the common support (its weight in "
                 "OFF_SUPPORT_x); FUNCTION = E_rm - E_rr; STATE = E_mr - E_rr; INTERACTION = E_mm - E_rm - E_mr + E_rr; "
                 "WITHIN = (e_m - E_mm) - (e_r - E_rr); DIFF = e_m - e_r = STATE + FUNCTION + INTERACTION + WITHIN"),
        "area_means": ("the land-area mean of each map over the common area is the quantity of the same name of "
                       "decompose_state_function.py (DIFF: E_m - E_r; WITHIN: 0 on a complete support)"),
        "obs_pool_weighting": OBS_POOL_WEIGHTING, "family_pool_weighting": FAMILY_POOL_WEIGHTING,
        "pooled_maps": "obs_pooled and <family>_pooled: the mixture of their members' bin maps and ET with the pool weights",
        "note_causality": "accounting identity on a shared partition of a shared domain, not a causal attribution",
    }

    # ------------------------------------------------------------------
    # Every dataset of each family against every obs reference, one family at a time
    # ------------------------------------------------------------------
    for family in (REFERENCE_FAMILY,) + EVALUATED_FAMILIES:
        print(f"\n=== {family} ===")
        if family == REFERENCE_FAMILY:
            p_mem, f_mem, e_mem, bins_mem = p_obs, f_obs, e_obs, bins_obs
        else:
            bs = load_binned(family, area_path)
            if bs is None:
                continue
            p_mem, f_mem = (da.reset_coords(drop=True).assign_coords(
                {c: bs[c] for c in bs.coords if bs[c].dims == ("member",)}) for da in dc.occupancy_and_mean(bs))
            if not (np.array_equal(bs.attrs["y_bin_edges"], obs_bs.attrs["y_bin_edges"])
                    and np.array_equal(bs.attrs["x_bin_edges"], obs_bs.attrs["x_bin_edges"])):
                raise ValueError(f"{family} was binned with other {EDGES} edges than obs; rerun bin_all_clim.py")
            e_mem, bins_mem = load_gridded(family, bs, area)
            check_occupancy(family, bins_mem, p_mem, land_area.fillna(0))
        print("the gridcell bins reproduce the occupancy of every member")

        # The datasets: (label, source, p, f, state); each member, then the family pooled (not for obs)
        members = [str(m) for m in p_mem["member"].values]
        datasets = [(f"{family}.{m}", str(p_mem["source"].values[k]), p_mem.isel(member=k), f_mem.isel(member=k),
                     as_state(bins_mem.isel(member=[k]), e_mem.isel(member=[k]), [1.0])) for k, m in enumerate(members)]
        if family != REFERENCE_FAMILY:
            w = member_weights(p_mem, family, FAMILY_POOL_WEIGHTING)
            p_pool, f_pool = dc.pool(p_mem, f_mem, xr.DataArray(w, dims="member"), "member")
            datasets.append((f"{family}_pooled", "pooled", p_pool, f_pool, as_state(bins_mem, e_mem, w)))

        # The maps, and the scalars of dc.decompose they must average to
        rows, max_err = [], {q: 0.0 for q in dc.GRIDDED_QUANTITIES}
        for label, _, p_m, f_m, state_m in datasets:
            per_ref = []
            for ref, (p_r, f_r, state_r) in refs.items():
                maps = dc.decompose_gridded(f_m, f_r, *state_m, *state_r)
                d = dc.decompose(p_m, f_m, p_r, f_r)
                expected = {"DIFF": d["E_m"] - d["E_r"], "WITHIN": d["E_m"] - d["E_mm"] - d["E_r"] + d["E_rr"]}
                for q in dc.GRIDDED_QUANTITIES:
                    err = abs(float(area_mean(maps[q])) - float(expected.get(q, d.get(q))))
                    max_err[q] = max(max_err[q], err)
                per_ref.append(maps)
            rows.append(xr.concat(per_ref, dim="obs_reference"))
        bad = {q: e for q, e in max_err.items() if e > ATOL}
        if bad:
            raise ValueError(f"{family}: area means of the maps differ from dc.decompose by {bad}")
        print("max |area mean of map - scalar of dc.decompose|: " + ", ".join(f"{q} {e:.1e}" for q, e in max_err.items()))

        out = xr.concat(rows, dim="dataset").assign_coords(ref_coords)
        out = out.assign_coords({"dataset": [d[0] for d in datasets], "family": ("dataset", [family] * len(datasets)),
                                 "source": ("dataset", [d[1] for d in datasets])})
        for a in OBS_PRODUCT_AXES:
            vals = products[a] if family == REFERENCE_FAMILY else ["none"] * len(datasets)
            out = out.assign_coords({f"{a}_product": ("dataset", vals)})
        out = out.transpose("dataset", "obs_reference", "lat", "lon")
        for q in dc.GRIDDED_QUANTITIES:
            out[q].attrs["units"] = UNITS[q]
        out.attrs = {**attrs, "family": family}

        # Area means vs obs_pooled (as the table of decompose_state_function.py)
        vs_pool = out.sel(obs_reference="obs_pooled")
        show = [d for d in vs_pool["dataset"].values if str(d).endswith("_pooled")] or list(vs_pool["dataset"].values)
        print(f"{'area mean vs obs_pooled':34} " + " ".join(f"{q:>11}" for q in ("DIFF",) + TERMS + ("WITHIN",)))
        for d in show:
            r = vs_pool.sel(dataset=d)
            print(f"{str(d):34} " + " ".join(f"{float(area_mean(r[q])):11.3f}" for q in ("DIFF",) + TERMS + ("WITHIN",)))

        fout = out_dir / f"{stem()}.{family}.{run}.nc"
        encoding = {q: {"dtype": "float32", "zlib": True, "complevel": 4} for q in out.data_vars}
        out.to_netcdf(fout, encoding=encoding)
        print(fout)

        # ------------------------------------------------------------------
        # Figures, against obs_pooled
        # ------------------------------------------------------------------
        context = f"vs obs_pooled ({len(obs_sources)} obs sources), {EDGES} {bins_tag}, {grid}, {period}, {run}"
        is_member = vs_pool["source"] != "pooled"
        if family != REFERENCE_FAMILY:
            r = vs_pool.sel(dataset=f"{family}_pooled")
            fout = fig_dir / f"{stem()}.{family}.pooled.{run}.png"
            plot_maps({q: r[q] for q in POOLED_PANELS}, fout=fout, land_area=land_area,
                      off_support=np.maximum(r["OFF_SUPPORT_m"], r["OFF_SUPPORT_r"]),
                      title=f"{family}_pooled minus obs: DIFF = STATE + FUNCTION + INTERACTION + WITHIN; hatched: "
                            f"bins off the common support\n{context}")
            print(fout)

        # Spread across the groups of variants (one value per group, its sources and members averaged)
        member_maps = vs_pool[list(STD_PANELS)].isel(dataset=np.flatnonzero(is_member.values)).rename(dataset="member")
        grouped = variants.group_means(member_maps, family, "member")
        if grouped.sizes["group"] > 1:
            std = grouped.std("group")
            fout = fig_dir / f"{stem()}.{family}.group_std.{run}.png"
            plot_maps({q: std[q] for q in STD_PANELS}, fout=fout, diverging=False, land_area=land_area,
                      title=f"{family}: std across its {grouped.sizes['group']} groups of variants\n{context}")
            print(fout)
        else:
            print(f"{family}: one group of variants, no std figure")

        # Every member's map of each term
        for term in TERMS:
            fout = fig_dir / f"{stem()}.{family}.members_{term}.{run}.png"
            plot_maps({str(d).split(".", 1)[-1]: member_maps[term].sel(member=d) for d in member_maps["member"].values},
                      fout=fout, land_area=land_area, ncols=6, panel_size=(3.0, 1.5),
                      title=f"{family} {term}, each member\n{context}")
            print(fout)


if __name__ == "__main__":
    main()
