"""
regress_lai_aridity.py
======================
How much of the variance in annual mean ET do climatological LAI and aridity
index (AI) explain? Ordinary least squares (OLS) regressions of ET on LAI and
AI, on exactly the samples that bin_all.py bins in the LAI-AI heatmaps.

ET, LAI and AI are the binning inputs of steps 1-11 (prep_<family>.py), read
from the saved inputs, on the common area mask of make_edges.py. For every
member of every family in FAMILIES, separately:

- Samples: every (year, gridcell) annual mean ET with that gridcell's
  climatological LAI and AI of the same member, unweighted, as in bin_all.py
  ("annual"); and each gridcell's climatological ET (mean over its valid
  years), as in bin_all_clim.py ("clim"). The climatological predictors cannot
  explain interannual variability, so "annual" R2 is at most the fraction of
  ET variance that is between gridcells.
- Fits (`regression.lai_ai_regression`): ET ~ LAI, ET ~ AI and
  ET ~ LAI + AI (linear, with an intercept), and the partition of
  R2(LAI + AI) into unique LAI, unique AI and shared parts (LAI and AI are
  correlated, so the single-predictor R2 do not add up).
- For comparison, the fraction of the variance the heatmap bins explain
  (eta2, `binning.between_bin_fraction`): the R2 of predicting each sample by
  its bin mean, with the pooled_all and pooled_<family> edges of make_edges.py.
  The bins are a nonparametric fit, so eta2 - R2(LAI + AI) measures how much the
  linear fit misses.

Outputs
-------
<family> : obs, era5, cmip, fppe, ippe or lens
<bins>   : lai<N_YBINS>_ai<N_XBINS>, the bin counts, e.g. "lai15_ai15"
<grid>   : grid tag of TARGET_RES, e.g. "1deg"
<period> : YYYYMM-YYYYMM of TIME_SLICE

PROC_ROOT/<family>/regress/<family>.et.regress_lai_ai.<bins>.<grid>.<period>.nc
    along (et_samples, member), with the member coords of the inputs (source, ...):
    r2 (predictors = lai, ai, lai+ai), r2_unique_lai, r2_unique_ai, r2_shared,
    intercept, slope_lai, slope_ai, n_samples; eta2 (edges = pooled_all, pooled_<family>)
FIG_ROOT/all/all.et.regress_lai_ai.<bins>.<grid>.<period>.png
    R2 of LAI, AI, LAI + AI and eta2 of the pooled_all bins, one point per member
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import etunc.config as cfg
import etunc.binning as binning
import etunc.edges as edges
import etunc.grid as rg
import etunc.plotting as plotting
import etunc.regression as regression
import etunc.temporal as temporal

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

FAMILIES = cfg.FAMILIES  # families to fit, e.g. ("lens",) for one
TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES
N_YBINS = cfg.N_YBINS        # LAI, for eta2
N_XBINS = cfg.N_XBINS        # AI, for eta2
FIG_ROOT = cfg.FIG_ROOT / "all"


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

# (label, colour, marker) of each metric in the figure; markers keep them apart without colour
METRICS = {
    "lai": ("R$^2$ LAI", "#2a78d6", "o"),
    "ai": ("R$^2$ AI", "#eb6834", "s"),
    "lai+ai": ("R$^2$ LAI + AI", "#1baf7a", "D"),
    "eta2": ("$\\eta^2$ pooled_all bins", "#eda100", "^"),
}


def plot_r2_summary(results: dict[str, xr.Dataset], title: str = "", fout=None):
    """One panel per ET sample type: each metric of each family, one point per member, line at the median."""
    samples = list(next(iter(results.values()))["et_samples"].values)
    fig, axs = plt.subplots(1, len(samples), figsize=(5.5 * len(samples), 4.5), sharey=True,
                            squeeze=False, layout="constrained")
    width = 0.8 / len(METRICS)
    for ax, s in zip(axs[0], samples):
        for i, ds in enumerate(results.values()):
            ds = ds.sel(et_samples=s)
            values = {k: ds["r2"].sel(predictors=k) for k in regression.PREDICTORS}
            values["eta2"] = ds["eta2"].sel(edges="pooled_all")
            for j, (k, (label, color, marker)) in enumerate(METRICS.items()):
                x = i + (j - (len(METRICS) - 1) / 2) * width
                v = values[k].values
                ax.scatter(np.full(v.size, x), v, s=20, color=color, marker=marker, alpha=0.7,
                           edgecolor="white", linewidth=0.5, label=label if i == 0 else None, zorder=2)
                ax.hlines(np.nanmedian(v), x - 0.4 * width, x + 0.4 * width, color="0.15", lw=1.5, zorder=3)
        ax.set_xticks(range(len(results)), [f"{f}\n({ds.sizes['member']})" for f, ds in results.items()])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="0.9", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{s} ET samples")
    axs[0, 0].set_ylabel("fraction of ET variance explained")
    axs[0, -1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(f"{title}\n(members per family in parentheses; line: median across members)")
    return plotting.finish(fig, fout)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.summarize_settings(globals())
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)

    # Common area mask of all families (make_edges.py), as in bin_all.py
    area_path = cfg.area_mask_path(grid, period)
    if not area_path.exists():
        raise FileNotFoundError(f"No common area mask at {area_path}; run scripts/make_edges.py first")
    area = xr.open_dataset(area_path)["area_mask"] == 1

    results = {}
    for family in FAMILIES:
        print(f"\n=== {family} ===")
        path = cfg.inputs_path(family, grid, period)
        if not path.exists():
            raise FileNotFoundError(f"No {family} inputs at {path}; run scripts/prep_{family}.py first")
        inputs = xr.load_dataset(path)

        # ET samples of bin_all.py (annual means) and bin_all_clim.py (climatology, NaN years skipped)
        et_samples = {
            "annual": inputs["et"],
            "clim": temporal.aggregate(inputs["et"], "clim", label=f"{family} et"),
        }
        kinds = ("pooled_all", f"pooled_{family}")
        bin_edges = {k: (edges.load_bin_edges(k, "lai", N_YBINS, grid, period),
                         edges.load_bin_edges(k, "ai", N_XBINS, grid, period)) for k in kinds}

        per_samples = []
        for s, et in et_samples.items():
            # OLS fits of ET on the member's climatological LAI and AI, one set per member
            fit = regression.lai_ai_regression(et, inputs["lai"], inputs["ai"], mask=area, member_dim="member")

            # Variance explained by the heatmap bins, on the same samples
            eta2 = []
            for k, (y_edges, x_edges) in bin_edges.items():
                bs = binning.bin_stats(et, inputs["lai"], inputs["ai"], y_edges, x_edges,
                                       mask=area, member_dim="member", y_name="lai", x_name="ai", name="et")
                # Both must use the same samples, or R2 and eta2 are not comparable
                if not (bs.sel(stats="count").sum(("y_bin", "x_bin")) == fit["n_samples"]).all():
                    raise RuntimeError(f"{family}/{s}/{k}: binned and fitted sample counts differ")
                eta2.append(binning.between_bin_fraction(bs))
            fit["eta2"] = xr.concat(eta2, dim="edges").assign_coords(edges=list(kinds))
            per_samples.append(fit)

            # Median [min, max] across members of each metric
            print(f"{s} ET, {int(fit['n_samples'].sum()):.3e} samples, median [min, max] across "
                  f"{fit.sizes['member']} members:")
            rows = {f"r2 {p}": fit["r2"].sel(predictors=p) for p in regression.PREDICTORS}
            rows |= {v: fit[v] for v in ("r2_unique_lai", "r2_unique_ai", "r2_shared")}
            rows |= {f"eta2 {k}": fit["eta2"].sel(edges=k) for k in kinds}
            for name, da in rows.items():
                print(f"  {name:20} {float(da.median()):6.3f} [{float(da.min()):6.3f}, {float(da.max()):6.3f}]")

        out = xr.concat(per_samples, dim="et_samples").assign_coords(et_samples=list(et_samples))
        out.attrs = {
            "family": family, "time_period": inputs.attrs["time_period"], "grid": grid,
            "area_mask": str(area_path), "model": "OLS, ET ~ LAI + AI with intercept, unweighted samples",
            "et_samples": "annual: every (year, gridcell) annual mean ET; clim: climatological ET per gridcell",
        }
        fout = cfg.PROC_ROOT / family / "regress" / f"{family}.et.regress_lai_ai.lai{N_YBINS}_ai{N_XBINS}.{grid}.{period}.nc"
        fout.parent.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(fout)
        print(fout)
        results[family] = out

    fout = FIG_ROOT / f"all.et.regress_lai_ai.lai{N_YBINS}_ai{N_XBINS}.{grid}.{period}.png"
    plot_r2_summary(results, title=f"ET variance explained by climatological LAI and AI, {grid}, {period}",
                    fout=fout)
    print(fout)


if __name__ == "__main__":
    main()
