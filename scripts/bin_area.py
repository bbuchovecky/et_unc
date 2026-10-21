"""
bin_area.py
===========
Step 14 of docs/binning-workflow.md for land area: the total land area in
each LAI-aridity bin, for every family.

Each gridcell of the common area mask is one sample, as in bin_all_clim.py,
binned by its climatological LAI and AI of the member. Its weight is its land
area (`rg.land_area`): the target-grid cell area times the fraction of the
cell covered by Natural Earth land. This is one field for every family (the
obs products have no land fraction of their own), so the binned area differs
between families and members only through their LAI and AI. Every member has
valid LAI and AI over the whole common area, so each member's total binned area
equals the land area of the common area mask.

Binned with the same two sets of edges and the same common area mask as
bin_all.py:
- "pooled_all": the edges pooled across all families by make_edges.py;
- "pooled_<family>": the edges pooled within the family by make_edges.py.

Outputs
-------
<family> : obs, era5, cmip, fppe, ippe or lens
<edges>  : pooled_all or pooled_<family>
<bins>   : lai<N_YBINS>_ai<N_XBINS>, the bin counts, e.g. "lai15_ai15"
<grid>   : grid tag of TARGET_RES, e.g. "1deg"
<period> : YYYYMM-YYYYMM of TIME_SLICE

PROC_ROOT/<family>/qbin/<family>.land_area.qbin_<edges>.<bins>.<grid>.<period>.nc
    per member, (member, y_bin, x_bin): land_area [km2] (0 in empty bins), land_area_frac (fraction of
    the member's binned land area) and n_cells (number of gridcells)
PROC_ROOT/<family>/qbin/<family>.land_area.qbin_<edges>.<bins>.<grid>.<period>.sources_pooled.nc
    the same, averaged over the members of each source: (source, y_bin, x_bin)
FIG_ROOT/<family>/qbin/<family>.land_area.qbin_<edges>.<bins>.<grid>.<period>.sources_pooled.png
    land_area_frac of each source
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np
import xarray as xr

import etunc.config as cfg
import etunc.binning as binning
import etunc.edges as edges
import etunc.grid as rg
import etunc.plotting as plotting
import etunc.temporal as temporal

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

BIN_FAMILIES = cfg.FAMILIES  # families to bin, e.g. ("lens",) for one
TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES
N_YBINS = cfg.N_YBINS        # LAI
N_XBINS = cfg.N_XBINS        # AI


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_area_frac(frac: xr.DataArray, title: str = "", fout=None, col_wrap: int = 6):
    """One heatmap per source of the % of its binned land area in each bin, on a shared colour scale."""
    frac = binning.drop_zero_width_bins(100 * frac)
    col_wrap = min(col_wrap, frac.sizes["source"])
    # Colorbar about two panels tall, as in plotting.plot_bin_means
    shrink = min(1.0, 2 / -(-frac.sizes["source"] // col_wrap))
    fg = frac.plot.pcolormesh(
        x="x_bin", y="y_bin", col="source", col_wrap=col_wrap, size=3, cmap="YlGnBu", robust=True,
        cbar_kwargs={"label": "% of the binned land area", "shrink": shrink, "aspect": 20 * shrink},
    )
    for ax, name_dict in zip(fg.axs.flat, fg.name_dicts.flat):
        if name_dict is not None:
            ax.set_title(name_dict["source"], fontsize=8)
    # FacetGrid axes are shared, so ticks set on the first panel apply to every panel
    plotting.set_edge_ticks(fg.axs.flat[0], frac, "{:.2g}")
    for ax in fg.axs.flat:
        ax.tick_params(axis="both", labelsize=7)
        ax.tick_params(axis="x", labelrotation=90)
    fg.set_axis_labels("ai $\\rightarrow$", "lai $\\rightarrow$")
    fg.fig.suptitle(title, y=1.02)
    return plotting.finish(fg.fig, fout)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.summarize_settings(globals())
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)

    # Common area mask of all families (make_edges.py)
    area_path = cfg.area_mask_path(grid, period)
    if not area_path.exists():
        raise FileNotFoundError(f"No common area mask at {area_path}; run scripts/make_edges.py first")
    area = xr.open_dataset(area_path)["area_mask"] == 1

    # Land area of every target-grid cell, on the latitudes of the inputs (LAT_BNDS, step 8)
    land = rg.land_area(rg.target_grid(TARGET_RES)).sel(lat=area.lat, lon=area.lon)
    rg.check_same_grid(land, area, "land area")
    land_common = float(land.where(area).sum())
    print(f"common area: {int(area.sum())} gridcells, {land_common:0.4g} km2 of Natural Earth land")

    for family in BIN_FAMILIES:
        print(f"\n=== {family} ===")
        path = cfg.inputs_path(family, grid, period)
        if not path.exists():
            raise FileNotFoundError(f"No {family} inputs at {path}; run scripts/prep_{family}.py first")
        inputs = xr.load_dataset(path)
        n_members = inputs.sizes["member"]

        # The land area repeated for every member, with the member coords (source, member_id, ...) of the inputs
        land_m = land * xr.ones_like(inputs["lai"])

        for kind in ("pooled_all", f"pooled_{family}"):
            y_edges = edges.load_bin_edges(kind, "lai", N_YBINS, grid, period)
            x_edges = edges.load_bin_edges(kind, "ai", N_XBINS, grid, period)

            # 14. Every gridcell's land area, in the common area, binned by that gridcell's climatological LAI
            # and AI of the member
            bs = binning.bin_stats(
                land_m, inputs["lai"], inputs["ai"], y_edges, x_edges,
                mask=area, member_dim="member", y_name="lai", x_name="ai", name="land_area",
                attrs={
                    "family": family, "edges": kind, "time_period": inputs.attrs["time_period"], "grid": grid,
                    "n_members": n_members, "area_mask": str(area_path),
                    "land_area": "target-grid cell area times its Natural Earth land fraction (rg.land_area)",
                },
            )
            # Total area in a bin = mean area of its gridcells x their number; 0 in empty bins
            count = bs.sel(stats="count", drop=True)
            total = (bs.sel(stats="mean", drop=True) * count).fillna(0)
            ds = xr.Dataset(
                {
                    "land_area": total.assign_attrs(long_name="land area in the bin", units="km2"),
                    "land_area_frac": (total / total.sum(("y_bin", "x_bin"))).assign_attrs(
                        long_name="fraction of the member's binned land area in the bin", units="1"),
                    "n_cells": count.assign_attrs(long_name="number of gridcells in the bin", units="1"),
                },
                attrs={k: v for k, v in bs.attrs.items() if k not in ("long_name", "units")},
            )
            stem = f"{family}.land_area.qbin_{kind}.lai{N_YBINS}_ai{N_XBINS}.{grid}.{period}"
            fout = cfg.PROC_ROOT / family / "qbin" / f"{stem}.nc"
            fout.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(fout)
            # Every member covers the whole common area, so its total is the common land area
            per_member = ds["land_area"].sum(("y_bin", "x_bin"))
            print(f"{kind}: {float(per_member.min()):0.6g} to {float(per_member.max()):0.6g} km2 binned per member "
                  f"(common area: {land_common:0.6g} km2)\n{fout}")
            if not np.allclose(per_member, land_common, rtol=1e-6):
                print(f"WARNING: the binned land area of some {family} members differs from the common land area")

            # Mean over the members of each source (not the sum), in the order of the inputs
            sources = list(dict.fromkeys(ds["source"].values.tolist()))
            by_source = ds.groupby("source").mean("member").sel(source=sources)
            fout = cfg.PROC_ROOT / family / "qbin" / f"{stem}.sources_pooled.nc"
            by_source.to_netcdf(fout)
            print(fout)

            fig = cfg.FIG_ROOT / family / "qbin" / f"{stem}.sources_pooled.png"
            plot_area_frac(
                by_source["land_area_frac"],
                title=f"{family} land area by bin, {len(sources)} sources, {kind} edges, {inputs.attrs['time_period']}",
                fout=fig,
            )
            print(fig)


if __name__ == "__main__":
    main()
