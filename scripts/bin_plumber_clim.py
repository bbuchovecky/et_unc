"""
bin_plumber_clim.py
===================
Step 14 for the PLUMBER2 flux towers with climatological ET: as
bin_plumber.py, but each tower's climatological annual mean ET (mean over its
years) is binned instead of each of its annual means, as bin_all_clim.py does
for the gridded families. One sample per tower instead of one per
(year, tower), so every tower weighs equally in its bin, whatever its number
of years, and the within-bin variance is across towers only (no interannual
variability).

The towers and their years come from the inputs of bin_plumber.py, so they
follow the same consecutive-year requirement: only each tower's longest run
of consecutive complete years in TIME_SLICE (after the ET outlier filter) is
kept, and towers whose run is shorter than MIN_CONSECUTIVE_YEARS of
bin_plumber.py (stored in the inputs attrs) are left out. The ET climatology
is the mean over that run (`temporal.aggregate(et, "clim")`).

LAI and AI are the climatologies of the obs products at the gridcell nearest
to each tower, separately for each (lai, pr, rn) product set, as in
bin_plumber.py. The towers are not masked by the common area mask, and a
tower whose nearest gridcell has no obs data is not binned.

Outputs
-------
<edges>  : pooled_all or pooled_obs
<grid>   : grid tag of TARGET_RES, e.g. "1deg"
<period> : YYYYMM-YYYYMM of TIME_SLICE

PROC_ROOT/plumber/qbin/plumber.et_clim.qbin_<edges>.<grid>.<period>.nc
    (lai_pr_rn, stats, y_bin, x_bin), all towers pooled; `count` is the number of towers
FIG_ROOT/plumber/qbin/plumber.et_clim.qbin_<edges>.<grid>.<period>.{summary,bin_mean,bin_mean_signif}.png
    as in bin_plumber.py, one panel per product set; bin_mean_signif hatches bins whose mean is not
    significantly different from 0
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
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

FAMILY = "plumber"
TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES
N_YBINS = cfg.N_YBINS                      # LAI
N_XBINS = cfg.N_XBINS                      # AI
EDGE_KINDS = ("pooled_all", "pooled_obs")  # edge sets of make_edges.py
SIGNIF_ALPHA = cfg.SIGNIF_ALPHA
SIGNIF_N_MIN = 3                           # as in bin_plumber.py


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.summarize_settings(globals())
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)

    # Tower ET and the obs LAI, P, Rn, AI at the nearest gridcell (bin_plumber.py)
    path = cfg.inputs_path(FAMILY, grid, period)
    if not path.exists():
        raise FileNotFoundError(f"No {FAMILY} inputs at {path}; run scripts/bin_plumber.py first")
    inputs = xr.load_dataset(path)
    # The consecutive-year requirement of bin_plumber.py; inputs written before it existed must be remade
    if "min_consecutive_years" not in inputs.attrs:
        raise ValueError(f"{path} has no consecutive-year requirement; rerun scripts/bin_plumber.py")
    min_consecutive = int(inputs.attrs["min_consecutive_years"])
    n_towers = inputs.sizes["member"]
    print(f"{n_towers} towers with >= {min_consecutive} consecutive complete years in {inputs.attrs['time_period']}, "
          f"{int(inputs['n_years'].min())} to {int(inputs['n_years'].max())} years per tower")

    # Climatological ET of each tower: (member, year) -> (member), the mean over its longest run of complete years
    et_clim = temporal.aggregate(inputs["et"], "clim", label="PLUMBER2 Qle")

    # ------------------------------------------------------------------
    # 14. Every tower's climatological ET binned by the tower's climatological LAI and AI, per product set
    # ------------------------------------------------------------------
    for kind in EDGE_KINDS:
        print(f"\n=== {kind} edges ===")
        y_edges = edges.load_bin_edges(kind, "lai", N_YBINS, grid, period)
        x_edges = edges.load_bin_edges(kind, "ai", N_XBINS, grid, period)
        bs = binning.bin_stats(
            et_clim, inputs["lai"], inputs["ai"], y_edges, x_edges,
            member_dim="lai_pr_rn", y_name="lai", x_name="ai", name="et",
            attrs={
                "family": FAMILY, "edges": kind, "time_period": inputs.attrs["time_period"], "grid": grid,
                "n_towers": n_towers, "min_consecutive_years": min_consecutive, "plumber_inputs": str(path),
                "et_samples": "climatological annual mean ET, one sample per tower",
            },
        )
        # bin_stats keeps only the coords of `et` along the member dim, so add those of the product sets
        bs = bs.assign_coords({c: inputs[c] for c in inputs.coords if inputs[c].dims == ("lai_pr_rn",)})
        stem = f"{FAMILY}.et_clim.qbin_{kind}.{grid}.{period}"
        fout = cfg.PROC_ROOT / FAMILY / "qbin" / f"{stem}.nc"
        fout.parent.mkdir(parents=True, exist_ok=True)
        bs.to_netcdf(fout)
        count = bs.sel(stats="count").sum(("y_bin", "x_bin"))
        for key in bs["lai_pr_rn"].values:
            n_bins = int((bs.sel(lai_pr_rn=key, stats="count") > 0).sum())
            print(f"{key}: {int(count.sel(lai_pr_rn=key))} towers binned in {n_bins} bins")
        print(fout)

        # Figures as in bin_plumber.py, one panel per product set; every set shares the edges, so the
        # zero-width bins can be dropped
        n_sets = bs.sizes["lai_pr_rn"]
        title = (f"PLUMBER2 tower climatological ET, {n_sets} (lai, pr, rn) product sets, {kind} edges, "
                 f"{inputs.attrs['time_period']}")
        bs = binning.drop_zero_width_bins(bs)
        fig_dir = cfg.FIG_ROOT / FAMILY / "qbin"
        plotting.plot_bin_summary(bs, dim="lai_pr_rn", title=title, fout=fig_dir / f"{stem}.summary.png")
        plotting.plot_bin_means(bs, "lai_pr_rn", title=f"{title}, bin mean", fout=fig_dir / f"{stem}.bin_mean.png")
        # Hatch non-empty bins whose mean is not significantly different from 0 (towers as independent samples)
        signif = binning.test_significance(bs, alpha=SIGNIF_ALPHA, n_min=SIGNIF_N_MIN)
        not_signif = ~signif & bs.sel(stats="mean").notnull()
        plotting.plot_bin_means(
            bs, "lai_pr_rn", hatch=not_signif, fout=fig_dir / f"{stem}.bin_mean_signif.png",
            title=(f"{title}; hatched: not significantly different from 0 at {1 - SIGNIF_ALPHA:.0%} "
                   f"(t-test, n > {SIGNIF_N_MIN})"),
        )
        for suffix in ("summary", "bin_mean", "bin_mean_signif"):
            print(fig_dir / f"{stem}.{suffix}.png")


if __name__ == "__main__":
    main()
