"""
bin_plumber_clim.py
===================
Step 14 for the PLUMBER2 flux towers with climatological ET: as
bin_plumber.py, but each tower's climatological annual mean ET (mean over its
years) is binned instead of each of its annual means, as bin_all_clim.py does
for the gridded families. One sample per tower instead of one per
(year, tower), so every tower weighs equally in its bin, whatever its number
of years, and the within-bin variance is across towers only (no interannual
variability). With BIN_VARIABLE = "et_pr", each tower's annual evaporation
ratio ET/P of bin_plumber.py is averaged instead (the mean of the annual
ratios).

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
<var>    : BIN_VARIABLE, et or et_pr
<edges>  : pooled_all or pooled_obs
<run_id> : RUN_ID of the run's settings (period, grid, bins, weighting; etunc.config, config/runs.csv)

PROC_ROOT/<run_id>/plumber/qbin/plumber.<var>_clim.qbin_<edges>.<run_id>.nc
    (lai_pr_rn, stats, y_bin, x_bin), all towers pooled; `count` is the number of towers
FIG_ROOT/<run_id>/plumber/qbin/plumber.<var>_clim.qbin_<edges>.{summary,bin_mean,bin_mean_signif}.<run_id>.png
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
BIN_VARIABLE = cfg.BIN_VARIABLE              # "et": tower ET; "et_pr": tower ET/P (saved by bin_plumber.py)
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
    cfg.register_run()
    cfg.summarize_settings(globals())
    grid = rg.grid_tag(TARGET_RES)
    run = cfg.RUN_ID  # folder and file tag of every output (the run's settings set the period and bins)

    # Tower ET and the obs LAI, P, Rn, AI at the nearest gridcell (bin_plumber.py)
    path = cfg.output_dir(cfg.PROC_ROOT, FAMILY, "inputs", run) / f"{FAMILY}.inputs.{run}.nc"
    if not path.exists():
        raise FileNotFoundError(f"No {FAMILY} inputs at {path}; run scripts/bin_plumber.py first")
    inputs = xr.load_dataset(path)
    # The consecutive-year requirement of bin_plumber.py; inputs written before it existed must be remade
    if "min_consecutive_years" not in inputs.attrs:
        raise ValueError(f"{path} has no consecutive-year requirement; rerun scripts/bin_plumber.py")
    if BIN_VARIABLE not in inputs:
        raise KeyError(f"No {BIN_VARIABLE} in {path}; rerun scripts/bin_plumber.py with BIN_VARIABLE = {BIN_VARIABLE!r}")
    min_consecutive = int(inputs.attrs["min_consecutive_years"])
    n_towers = inputs.sizes["member"]
    print(f"{n_towers} towers with >= {min_consecutive} consecutive complete years in {inputs.attrs['time_period']}, "
          f"{int(inputs['n_years'].min())} to {int(inputs['n_years'].max())} years per tower")

    # Climatological ET (or mean ET/P) of each tower: (member, year) -> (member), the mean over its longest run
    # of complete years
    target_clim = temporal.aggregate(inputs[BIN_VARIABLE], "clim", label=f"PLUMBER2 {BIN_VARIABLE}")
    label = binning.TARGET_LABELS[BIN_VARIABLE]

    # ------------------------------------------------------------------
    # 14. Every tower's climatological ET (or mean ET/P) binned by the tower's climatological LAI and AI, per product set
    # ------------------------------------------------------------------
    qbin_dir = cfg.output_dir(cfg.PROC_ROOT, FAMILY, "qbin", run)
    fig_dir = cfg.output_dir(cfg.FIG_ROOT, FAMILY, "qbin", run)
    for kind in EDGE_KINDS:
        print(f"\n=== {kind} edges ===")
        y_edges = edges.load_bin_edges(kind, "lai")
        x_edges = edges.load_bin_edges(kind, "ai")
        bs = binning.bin_stats(
            target_clim, inputs["lai"], inputs["ai"], y_edges, x_edges,
            member_dim="lai_pr_rn", y_name="lai", x_name="ai", name=BIN_VARIABLE,
            attrs={
                "family": FAMILY, "edges": kind, "time_period": inputs.attrs["time_period"], "grid": grid, "run_id": run,
                "n_towers": n_towers, "min_consecutive_years": min_consecutive, "plumber_inputs": str(path),
                "et_samples": f"climatological annual mean {label}, one sample per tower",
            },
        )
        # bin_stats keeps only the coords of the binned variable along the member dim, so add those of the product sets
        bs = bs.assign_coords({c: inputs[c] for c in inputs.coords if inputs[c].dims == ("lai_pr_rn",)})
        stem = f"{FAMILY}.{BIN_VARIABLE}_clim.qbin_{kind}"
        fout = qbin_dir / f"{stem}.{run}.nc"
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
        title = (f"PLUMBER2 tower climatological {label}, {n_sets} (lai, pr, rn) product sets, {kind} edges, "
                 f"{inputs.attrs['time_period']}")
        bs = binning.drop_zero_width_bins(bs)
        plotting.plot_bin_summary(bs, dim="lai_pr_rn", title=title, fout=fig_dir / f"{stem}.summary.{run}.png")
        plotting.plot_bin_means(bs, "lai_pr_rn", title=f"{title}, bin mean", fout=fig_dir / f"{stem}.bin_mean.{run}.png")
        # Hatch non-empty bins whose mean is not significantly different from 0 (towers as independent samples)
        signif = binning.test_significance(bs, alpha=SIGNIF_ALPHA, n_min=SIGNIF_N_MIN)
        not_signif = ~signif & bs.sel(stats="mean").notnull()
        plotting.plot_bin_means(
            bs, "lai_pr_rn", hatch=not_signif, fout=fig_dir / f"{stem}.bin_mean_signif.{run}.png",
            title=(f"{title}; hatched: not significantly different from 0 at {1 - SIGNIF_ALPHA:.0%} "
                   f"(t-test, n > {SIGNIF_N_MIN})"),
        )
        for suffix in ("summary", "bin_mean", "bin_mean_signif"):
            print(fig_dir / f"{stem}.{suffix}.{run}.png")


if __name__ == "__main__":
    main()
