"""
bin_all.py
==========
Step 14 of docs/binning-workflow.md: bin annual mean ET (or, with BIN_VARIABLE
= "et_pr", the annual evaporation ratio ET/P) by climatological LAI and AI,
for every family in BIN_FAMILIES and two sets of edges:

- "pooled_all": the edges pooled across all families by make_edges.py;
- "pooled_<family>": the edges pooled within the family by make_edges.py.

Every family is binned the same way, over the common area mask of
make_edges.py: each (year, gridcell) annual mean ET sample of a member is
binned by that gridcell's climatological LAI and AI of the same member
(`binning.bin_stats`, per member). Set BIN_FAMILIES to e.g. ("lens",) to bin
one family; it still uses the common area mask and the edges of all families.

With AREA_WEIGHTED = True, each sample is weighted by its gridcell's land area
(`rg.land_area`: target-grid cell area times its Natural Earth land fraction,
one field for every family) in the bin mean and variances, so a bin's mean is
the land-area weighted mean ET of its gridcell-years. The edges stay the same.

Outputs
-------
<family>    : obs, era5, cmip, fppe, ippe or lens
<var>       : BIN_VARIABLE, et or et_pr
<edges>     : pooled_all or pooled_<family>
<run_id>    : RUN_ID of the run's settings (period, grid, bins, weighting; etunc.config, config/runs.csv)

PROC_ROOT/<run_id>/<family>/qbin/<family>.<var>.qbin_<edges>.<run_id>.nc
    per member: (member, stats, y_bin, x_bin), with the member coords of the inputs (source, ...)
PROC_ROOT/<run_id>/<family>/qbin/<family>.<var>.qbin_<edges>.sources_pooled.<run_id>.nc
    the members of each source pooled (`binning.pool_members`): (source, stats, y_bin, x_bin).
    Equal to the per-member statistics when every source has one member
FIG_ROOT/<run_id>/<family>/qbin/<family>.<var>.qbin_<edges>.{summary,bin_mean,bin_mean_signif}.<run_id>.png
    of the sources_pooled statistics; the summary's mean and std are across the groups of variants
    (`variants.group_means`: CMIP6 models of one base model, versions of one obs ET product), each
    counting once; bin_mean_signif hatches bins whose mean is not significantly different from 0

Family-specific outputs
PROC_ROOT/<run_id>/obs/qbin/obs.<var>.qbin_<edges>.et_product_std.<run_id>.nc (and FIG_ROOT .png)
    std of the bin mean ET (or ET/P) across ET products, for each (lai, pr, rn) product set, the
    versions of one ET product (e.g. GLEAMv4.3a and b) averaged first
PROC_ROOT/<run_id>/lens/qbin/lens.<var>.qbin_<edges>.members_pooled.<run_id>.nc
    statistics of all LENS2 members' samples pooled
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

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

BIN_FAMILIES = cfg.FAMILIES  # families to bin, e.g. ("lens",) for one
TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES
BIN_VARIABLE = cfg.BIN_VARIABLE  # "et": annual mean ET; "et_pr": annual evaporation ratio ET/P (saved by prep_<family>.py)
AREA_WEIGHTED = cfg.AREA_WEIGHTED  # True: bin statistics weighted by each gridcell's land area
N_YBINS = cfg.N_YBINS        # LAI
N_XBINS = cfg.N_XBINS        # AI
SIGNIF_ALPHA = cfg.SIGNIF_ALPHA
SIGNIF_N_MIN = cfg.SIGNIF_N_MIN


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.register_run()
    cfg.summarize_settings(globals())
    grid = rg.grid_tag(TARGET_RES)
    run = cfg.RUN_ID  # folder and file tag of every output (the run's settings set the period and bins)

    # Common area mask of all families (make_edges.py)
    area_path = cfg.area_mask_path()
    if not area_path.exists():
        raise FileNotFoundError(f"No common area mask at {area_path}; run scripts/make_edges.py first")
    area = xr.open_dataset(area_path)["area_mask"] == 1

    # Weight of each sample: its gridcell's land area [km2], or none (every sample counts once)
    weights = None
    if AREA_WEIGHTED:
        weights = rg.land_area(rg.target_grid(TARGET_RES)).sel(lat=area.lat, lon=area.lon)
        rg.check_same_grid(weights, area, "land area")
    weight_title = ", land-area weighted" if AREA_WEIGHTED else ""

    for family in BIN_FAMILIES:
        print(f"\n=== {family} ===")
        path = cfg.inputs_path(family)
        if not path.exists():
            raise FileNotFoundError(f"No {family} inputs at {path}; run scripts/prep_{family}.py first")
        inputs = xr.load_dataset(path)
        if BIN_VARIABLE not in inputs:
            raise KeyError(f"No {BIN_VARIABLE} in {path}; rerun scripts/prep_{family}.py (it saves et and et_pr)")
        n_members = inputs.sizes["member"]

        qbin_dir = cfg.output_dir(cfg.PROC_ROOT, family, "qbin", run)
        fig_dir = cfg.output_dir(cfg.FIG_ROOT, family, "qbin", run)
        for kind in ("pooled_all", f"pooled_{family}"):
            y_edges = edges.load_bin_edges(kind, "lai")
            x_edges = edges.load_bin_edges(kind, "ai")

            # 14. Every (year, gridcell) ET (or ET/P) sample of a member, in the common area, binned by that
            # gridcell's climatological LAI and AI of the same member (weighted by its land area if AREA_WEIGHTED)
            bs = binning.bin_stats(
                inputs[BIN_VARIABLE], inputs["lai"], inputs["ai"], y_edges, x_edges, weights=weights,
                mask=area, member_dim="member", y_name="lai", x_name="ai", name=BIN_VARIABLE,
                attrs={
                    "family": family, "edges": kind, "time_period": inputs.attrs["time_period"], "grid": grid, "run_id": run,
                    "n_members": n_members, "area_mask": str(area_path),
                },
            )
            stem = f"{family}.{BIN_VARIABLE}.qbin_{kind}"
            fout = qbin_dir / f"{stem}.{run}.nc"
            fout.parent.mkdir(parents=True, exist_ok=True)
            bs.to_netcdf(fout)
            count = bs.sel(stats="count").sum(("y_bin", "x_bin"))
            print(f"{kind}: {int(count.sum()):.3e} samples binned, {int(count.min()):.3e} to {int(count.max()):.3e} "
                  f"per member\n{fout}")

            by_source = binning.pool_by_source(bs)
            fout = qbin_dir / f"{stem}.sources_pooled.{run}.nc"
            by_source.to_netcdf(fout)
            print(fout)

            # Figures of the sources; every source shares the edges, so the zero-width bins can be dropped
            n_sources = by_source.sizes["source"]
            title = (f"{family} {binning.TARGET_LABELS[BIN_VARIABLE]}{weight_title}, {n_sources} sources, {kind} edges, "
                     f"{inputs.attrs['time_period']}")
            by_source = binning.drop_zero_width_bins(by_source)
            plotting.plot_bin_summary(by_source, dim="source", family=family, title=title, fout=fig_dir / f"{stem}.summary.{run}.png")
            plotting.plot_bin_means(by_source, "source", title=f"{title}, bin mean", fout=fig_dir / f"{stem}.bin_mean.{run}.png")
            # Hatch non-empty bins whose mean is not significantly different from 0
            signif = binning.test_significance(by_source, alpha=SIGNIF_ALPHA, n_min=SIGNIF_N_MIN)
            not_signif = ~signif & by_source.sel(stats="mean").notnull()
            print(f"{int(not_signif.sum())} of {int(by_source.sel(stats='mean').notnull().sum())} non-empty bins "
                  f"not significant at {1 - SIGNIF_ALPHA:.0%}")
            plotting.plot_bin_means(
                by_source, "source", hatch=not_signif, fout=fig_dir / f"{stem}.bin_mean_signif.{run}.png",
                title=(f"{title}; hatched: not significantly different from 0 at {1 - SIGNIF_ALPHA:.0%} "
                       f"(t-test, n > {SIGNIF_N_MIN})"),
            )
            print(fig_dir / f"{stem}.summary.{run}.png")
            print(fig_dir / f"{stem}.bin_mean.{run}.png")
            print(fig_dir / f"{stem}.bin_mean_signif.{run}.png")

            # ------------------------------------------------------------------
            # Family-specific outputs
            # ------------------------------------------------------------------
            if family == "obs":
                # Spread across ET products, with the other three products fixed
                spread = binning.et_product_spread(by_source)
                fout = qbin_dir / f"{stem}.et_product_std.{run}.nc"
                spread.to_netcdf(fout)
                plotting.plot_et_product_spread(
                    spread, title=(f"obs {binning.TARGET_LABELS[BIN_VARIABLE]}{weight_title}, std of bin mean across "
                                   f"ET products, {kind} edges"),
                    fout=fig_dir / f"{stem}.et_product_std.{run}.png", label=BIN_VARIABLE,
                )
                print(fout)
            if family == "lens":
                # All members' samples pooled: the LENS2 initial-condition ensemble as one distribution
                fout = qbin_dir / f"{stem}.members_pooled.{run}.nc"
                binning.pool_members(bs).to_netcdf(fout)
                print(fout)


if __name__ == "__main__":
    main()
