"""
make_edges.py
=============
Steps 12-13 of docs/binning-workflow.md, after every prep_<family>.py has
written its binning inputs: the area mask shared by all families, and the
LAI and AI quantile edges that bin_all.py bins with.

12. Common area mask: each family's own valid area (`rg.valid_area`: ET in
    any year, LAI and a finite AI, in every member of the family), and their
    intersection (`rg.common_area`), so every family is binned over the same
    gridcells. The gridcells and land area each family loses are printed.
13. Quantile edges (N_YBINS for LAI, N_XBINS for AI) of the climatological
    LAI and AI on the common area (`edges.pooled_bin_edges`):
    - "pooled_all": all families, with hierarchical weights
      (`edges.hierarchical_weights`): each family weighs equally, each source
      within a family equally (obs combination, CMIP6 model, PPE/LENS member),
      and each member within a source equally.
    - "pooled_<family>": one family, its sources weighted equally.

Outputs
-------
<grid>   : grid tag of TARGET_RES, e.g. "1deg"
<period> : YYYYMM-YYYYMM of TIME_SLICE
<n>      : number of bins (N_YBINS for LAI, N_XBINS for AI)

PROC_ROOT/all/all.area_mask.<grid>.<period>.nc
    area_mask (lat, lon), and valid_area (family, lat, lon): each family's own valid area
BIN_EDGES_ROOT/{pooled_all,pooled_<family>}.{lai,ai}_clim.<n>_quantiles.<grid>.<period>.nc
FIG_ROOT/all/all.area_mask.<grid>.<period>.png
    number of families whose valid area includes each gridcell
FIG_ROOT/all/all.{lai,ai}_clim.<n>_quantiles.<grid>.<period>.png
    pooled_all and pooled_<family> edges
FIG_ROOT/all/all.{lai,ai}_clim.<grid>.hist.<period>.png
    weighted histograms on the common area, one line per family and the pooled total, with the
    pooled_all edges
FIG_ROOT/all/all.{lai,ai}_clim.<grid>.hist_by_family.<period>.png
    one panel per family: one line per source, with the pooled_<family> edges
FIG_ROOT/all/all.{lai,ai}_clim.<n>_quantiles.<grid>.bin_fraction.<period>.png
    % of each family's weighted values in each pooled_all bin
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import etunc.config as cfg
import etunc.edges as edges
import etunc.grid as rg
import etunc.plotting as plotting
import etunc.temporal as temporal

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

FAMILIES = cfg.FAMILIES  # every family pooled into the common area and the pooled_all edges
TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES
N_BINS = {"lai": cfg.N_YBINS, "ai": cfg.N_XBINS}
FIG_ROOT = cfg.FIG_ROOT / "all"


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_bin_fraction(values: dict[str, tuple[np.ndarray, np.ndarray]], bin_edges: xr.DataArray, title: str = "",
                      fout=None):
    """
    Weighted fraction of each family's values in each bin of `bin_edges` (bins assigned as in
    `binning.bin_stats`). With quantile edges of its own, every bin would hold 1/n of a family.
    """
    e = bin_edges.values
    n = len(e) - 1
    frac = np.stack([
        np.bincount(np.clip(np.searchsorted(e, x, side="right") - 1, 0, n - 1), weights=w, minlength=n) / w.sum()
        for x, w in values.values()
    ])
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(values) + 1.5), layout="constrained")
    pm = ax.pcolormesh(np.arange(n + 1), np.arange(len(values) + 1), frac, cmap="RdBu_r", vmin=0, vmax=2 / n)
    for (i, j), f in np.ndenumerate(frac):
        ax.text(j + 0.5, i + 0.5, f"{100 * f:.2f}", ha="center", va="center", fontsize=7)
    ax.set_yticks(np.arange(len(values)) + 0.5, list(values))
    ax.set_xlabel(f"{bin_edges.name} bin")
    ax.set_title(f"{title}; even = {100 / n:.1f}%")
    fig.colorbar(pm, ax=ax, label="fraction of the family's weighted values")
    return plotting.finish(fig, fout)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.summarize_settings(globals())
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)

    print("=== Binning inputs ===")
    inputs = {}
    for family in FAMILIES:
        path = cfg.inputs_path(family, grid, period)
        if not path.exists():
            raise FileNotFoundError(f"No {family} inputs at {path}; run scripts/prep_{family}.py first")
        inputs[family] = xr.open_dataset(path)
        n_sources = len(set(inputs[family]["source"].values.tolist()))
        print(f"{family:5}: {inputs[family].sizes['member']} members, {n_sources} sources, "
              f"{inputs[family].attrs['time_period']}  {path}")

    # 12. Common area mask: gridcells valid in every member of every family
    print("\n=== 12. Common area mask ===")
    valid = {family: rg.valid_area(ds) for family, ds in inputs.items()}
    area = rg.common_area(valid)
    fout = cfg.area_mask_path(grid, period)
    fout.parent.mkdir(parents=True, exist_ok=True)
    valid_all = xr.concat(list(valid.values()), dim="family").assign_coords(family=list(valid))
    xr.Dataset({
        "area_mask": area.astype("i1").assign_attrs(
            long_name="gridcells where ET (any year), LAI and AI are valid in every member of every family"),
        "valid_area": valid_all.astype("i1").assign_attrs(long_name="each family's own valid area"),
    }, attrs={"families": list(inputs), "time_period": period, "grid": grid}).to_netcdf(fout)
    print(fout)
    plotting.plot_mask_agreement(
        valid_all.rename(family="product"), valid_all.any("family"),
        title=f"Valid areas of {', '.join(inputs)} ({grid}, {period})", label="families with the gridcell valid",
        highlight_common=True, fout=FIG_ROOT / f"all.area_mask.{grid}.{period}.png",
    )
    print(FIG_ROOT / f"all.area_mask.{grid}.{period}.png")

    # 13. Quantile edges on the common area: pooled across all families, and pooled within each family
    for v, n_bins in N_BINS.items():
        print(f"\n=== 13. {v} edges ===")
        fields = {family: ds[v].where(area) for family, ds in inputs.items()}
        attrs = {**edges.EDGE_ATTRS[v], "time_period": period, "grid": grid, "area_mask": str(fout)}

        # pooled_all: each family, each source within a family and each member within a source weighs equally
        values, weights = edges.hierarchical_weights(fields)
        edge_sets = {"pooled_all": edges.pooled_bin_edges(
            values, weights, n_bins, name=v, attrs={**attrs, "weighting": "hierarchical", "families": list(fields)},
        )}
        # pooled_<family>: the sources of one family weigh equally
        family_values = {family: edges.hierarchical_weights({family: da}) for family, da in fields.items()}
        for family, (fv, fw) in family_values.items():
            edge_sets[f"pooled_{family}"] = edges.pooled_bin_edges(
                fv, fw, n_bins, name=v, attrs={**attrs, "weighting": "sources equal", "families": [family]},
            )
        for kind, e in edge_sets.items():
            path = edges.edges_path(kind, v, n_bins, grid, period)
            path.parent.mkdir(parents=True, exist_ok=True)
            e.to_netcdf(path)
            print(path)

        # Diagnostics
        stem = f"all.{v}_clim.{n_bins}_quantiles.{grid}"
        plotting.plot_edges_compare(edge_sets, title=f"{v} quantile edges on the common area, {period}",
                                    fout=FIG_ROOT / f"{stem}.{period}.png")
        plotting.plot_clim_hist(
            family_values, cfg.HIST_BINS[v], pooled=(values, weights), edges=edge_sets["pooled_all"],
            xlabel=f"climatological {v}", title=f"{v} on the common area, with the pooled_all edges",
            fout=FIG_ROOT / f"all.{v}_clim.{grid}.hist.{period}.png",
        )
        fig, axs = plt.subplots(len(fields), 1, figsize=(8, 3 * len(fields)), layout="constrained", squeeze=False)
        for ax, (family, da) in zip(axs[:, 0], fields.items()):
            sources = list(dict.fromkeys(da["source"].values.tolist()))
            plotting.plot_clim_hist(
                {s: da.where(da["source"] == s) for s in sources}, cfg.HIST_BINS[v],
                pooled=family_values[family], edges=edge_sets[f"pooled_{family}"], xlabel=f"climatological {v}",
                title=f"{family}: {len(sources)} sources, with the pooled_{family} edges", ax=ax,
            )
        plotting.finish(fig, FIG_ROOT / f"all.{v}_clim.{grid}.hist_by_family.{period}.png")
        plot_bin_fraction(family_values, edge_sets["pooled_all"], title=f"{v}: % of values per pooled_all bin",
                          fout=FIG_ROOT / f"{stem}.bin_fraction.{period}.png")


if __name__ == "__main__":
    main()
