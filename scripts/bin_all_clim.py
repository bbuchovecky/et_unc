"""
bin_all_clim.py
===============
Step 14 of docs/binning-workflow.md with climatological ET: as bin_all.py,
but each gridcell's climatological annual mean ET (mean over the years of the
inputs) is binned instead of each of its annual means. With BIN_VARIABLE =
"et_pr", it is the mean over the years of each gridcell's annual evaporation
ratio ET/P (the mean of the annual ratios). One sample per
(member, gridcell) instead of one per (member, year, gridcell), so every
gridcell weighs equally in its bin (unless AREA_WEIGHTED), whatever its number
of valid ET years, and the within-bin variance is spatial only (no interannual
variability).

With AREA_WEIGHTED = True, each gridcell is weighted by its land area
(`rg.land_area`: target-grid cell area times its Natural Earth land fraction,
one field for every family) in the bin mean and variances, so a bin's mean is
the land-area weighted mean of its gridcells. The edges stay the same.

The ET climatology follows the rule of the LAI, P and Rn climatologies (step
10, `temporal.aggregate(et, "clim")`): every year weighs equally and a
gridcell's NaN years are skipped (incomplete years and the years dropped by
the ET outlier filter of step 6; for obs, also the years outside a source's
own shared years, so the printed count of gridcells with missing years
includes those).

Binned with the same two sets of edges and the same common area mask as
bin_all.py:
- "pooled_all": the edges pooled across all families by make_edges.py;
- "pooled_<family>": the edges pooled within the family by make_edges.py.

Outputs
-------
<family>    : obs, era5, cmip, fppe, ippe or lens
<var>       : BIN_VARIABLE, et or et_pr
<weighting> : ".area_weighted" with AREA_WEIGHTED, else empty
<edges>     : pooled_all or pooled_<family>
<bins>      : lai<N_YBINS>_ai<N_XBINS>, the bin counts, e.g. "lai15_ai15"
<grid>      : grid tag of TARGET_RES, e.g. "1deg"
<period>    : YYYYMM-YYYYMM of TIME_SLICE

PROC_ROOT/<family>/qbin/<period>/<bins>/<family>.<var>_clim<weighting>.qbin_<edges>.<bins>.<grid>.<period>.nc
    per member: (member, stats, y_bin, x_bin); `count` is the number of gridcells
PROC_ROOT/<family>/qbin/<period>/<bins>/<family>.<var>_clim<weighting>.qbin_<edges>.<bins>.<grid>.<period>.sources_pooled.nc
    the members of each source pooled (`binning.pool_members`): (source, stats, y_bin, x_bin)
FIG_ROOT/<family>/qbin/<period>/<bins>/<family>.<var>_clim<weighting>.qbin_<edges>.<bins>.<grid>.<period>.{summary,bin_mean,bin_mean_signif}.png
    of the sources_pooled statistics; bin_mean_signif hatches bins whose mean is not significantly
    different from 0

Family-specific outputs
PROC_ROOT/obs/qbin/<period>/<bins>/obs.<var>_clim<weighting>.qbin_<edges>.<bins>.<grid>.<period>.et_product_std.nc (and FIG_ROOT .png)
    std of the bin mean ET (or ET/P) across ET products, for each (lai, pr, rn) product set
PROC_ROOT/lens/qbin/<period>/<bins>/lens.<var>_clim<weighting>.qbin_<edges>.<bins>.<grid>.<period>.members_pooled.nc
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
import etunc.temporal as temporal

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
    cfg.summarize_settings(globals())
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)
    bins = cfg.bins_tag(N_YBINS, N_XBINS)  # bin counts, in the folder and file names

    # Common area mask of all families (make_edges.py)
    area_path = cfg.area_mask_path(grid, period)
    if not area_path.exists():
        raise FileNotFoundError(f"No common area mask at {area_path}; run scripts/make_edges.py first")
    area = xr.open_dataset(area_path)["area_mask"] == 1

    # Weight of each gridcell: its land area [km2], or none (every gridcell counts once)
    weights = None
    if AREA_WEIGHTED:
        weights = rg.land_area(rg.target_grid(TARGET_RES)).sel(lat=area.lat, lon=area.lon)
        rg.check_same_grid(weights, area, "land area")
    weight_tag = ".area_weighted" if AREA_WEIGHTED else ""
    weight_title = ", land-area weighted" if AREA_WEIGHTED else ""

    for family in BIN_FAMILIES:
        print(f"\n=== {family} ===")
        path = cfg.inputs_path(family, grid, period)
        if not path.exists():
            raise FileNotFoundError(f"No {family} inputs at {path}; run scripts/prep_{family}.py first")
        inputs = xr.load_dataset(path)
        if BIN_VARIABLE not in inputs:
            raise KeyError(f"No {BIN_VARIABLE} in {path}; rerun scripts/prep_{family}.py with BIN_VARIABLE = {BIN_VARIABLE!r}")
        n_members = inputs.sizes["member"]

        # Climatological ET (or mean ET/P) of each member: (member, year, lat, lon) -> (member, lat, lon), NaN
        # years skipped
        target_clim = temporal.aggregate(inputs[BIN_VARIABLE], "clim", label=f"{family} {BIN_VARIABLE}")
        label = binning.TARGET_LABELS[BIN_VARIABLE]

        qbin_dir = cfg.output_dir(cfg.PROC_ROOT, family, "qbin", period, bins)
        fig_dir = cfg.output_dir(cfg.FIG_ROOT, family, "qbin", period, bins)
        for kind in ("pooled_all", f"pooled_{family}"):
            y_edges = edges.load_bin_edges(kind, "lai", N_YBINS, N_XBINS, grid, period)
            x_edges = edges.load_bin_edges(kind, "ai", N_YBINS, N_XBINS, grid, period)

            # 14. Every gridcell's climatological ET (or mean ET/P) of a member, in the common area, binned by
            # that gridcell's climatological LAI and AI of the same member (weighted by its land area if AREA_WEIGHTED)
            bs = binning.bin_stats(
                target_clim, inputs["lai"], inputs["ai"], y_edges, x_edges, weights=weights,
                mask=area, member_dim="member", y_name="lai", x_name="ai", name=BIN_VARIABLE,
                attrs={
                    "family": family, "edges": kind, "time_period": inputs.attrs["time_period"], "grid": grid,
                    "n_members": n_members, "area_mask": str(area_path),
                    "et_samples": f"climatological annual mean {label}, one sample per gridcell",
                },
            )
            stem = f"{family}.{BIN_VARIABLE}_clim{weight_tag}.qbin_{kind}.{bins}.{grid}.{period}"
            fout = qbin_dir / f"{stem}.nc"
            fout.parent.mkdir(parents=True, exist_ok=True)
            bs.to_netcdf(fout)
            count = bs.sel(stats="count").sum(("y_bin", "x_bin"))
            print(f"{kind}: {int(count.sum()):.3e} gridcells binned, {int(count.min()):.3e} to {int(count.max()):.3e} "
                  f"per member\n{fout}")

            by_source = binning.pool_by_source(bs)
            fout = qbin_dir / f"{stem}.sources_pooled.nc"
            by_source.to_netcdf(fout)
            print(fout)

            # Figures of the sources; every source shares the edges, so the zero-width bins can be dropped
            n_sources = by_source.sizes["source"]
            title = f"{family} climatological {label}{weight_title}, {n_sources} sources, {kind} edges, {inputs.attrs['time_period']}"
            by_source = binning.drop_zero_width_bins(by_source)
            plotting.plot_bin_summary(by_source, dim="source", title=title, fout=fig_dir / f"{stem}.summary.png")
            plotting.plot_bin_means(by_source, "source", title=f"{title}, bin mean", fout=fig_dir / f"{stem}.bin_mean.png")
            # Hatch non-empty bins whose mean is not significantly different from 0 (gridcells as independent samples)
            signif = binning.test_significance(by_source, alpha=SIGNIF_ALPHA, n_min=SIGNIF_N_MIN)
            not_signif = ~signif & by_source.sel(stats="mean").notnull()
            print(f"{int(not_signif.sum())} of {int(by_source.sel(stats='mean').notnull().sum())} non-empty bins "
                  f"not significant at {1 - SIGNIF_ALPHA:.0%}")
            plotting.plot_bin_means(
                by_source, "source", hatch=not_signif, fout=fig_dir / f"{stem}.bin_mean_signif.png",
                title=(f"{title}; hatched: not significantly different from 0 at {1 - SIGNIF_ALPHA:.0%} "
                       f"(t-test, n > {SIGNIF_N_MIN})"),
            )
            print(fig_dir / f"{stem}.summary.png")
            print(fig_dir / f"{stem}.bin_mean.png")
            print(fig_dir / f"{stem}.bin_mean_signif.png")

            # ------------------------------------------------------------------
            # Family-specific outputs
            # ------------------------------------------------------------------
            if family == "obs":
                # Spread across ET products, with the other three products fixed
                spread = binning.et_product_spread(by_source)
                fout = qbin_dir / f"{stem}.et_product_std.nc"
                spread.to_netcdf(fout)
                plotting.plot_et_product_spread(
                    spread, title=f"obs climatological {label}{weight_title}, std of bin mean across ET products, {kind} edges",
                    fout=fig_dir / f"{stem}.et_product_std.png", label=BIN_VARIABLE,
                )
                print(fout)
            if family == "lens":
                # All members' samples pooled: the LENS2 initial-condition ensemble as one distribution
                fout = qbin_dir / f"{stem}.members_pooled.nc"
                binning.pool_members(bs).to_netcdf(fout)
                print(fout)


if __name__ == "__main__":
    main()
