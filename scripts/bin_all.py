"""
bin_all.py
==========
Step 14 of docs/binning-workflow.md: bin annual mean ET by climatological LAI
and AI, for every family in BIN_FAMILIES and two sets of edges:

- "pooled_all": the edges pooled across all families by make_edges.py;
- "pooled_<family>": the edges pooled within the family by make_edges.py.

Every family is binned the same way, over the common area mask of
make_edges.py: each (year, gridcell) annual mean ET sample of a member is
binned by that gridcell's climatological LAI and AI of the same member
(`binning.bin_stats`, per member). Set BIN_FAMILIES to e.g. ("lens",) to bin
one family; it still uses the common area mask and the edges of all families.

Outputs
-------
<family> : obs, cmip, fppe or lens
<edges>  : pooled_all or pooled_<family>
<grid>   : grid tag of TARGET_RES, e.g. "1deg"
<period> : YYYYMM-YYYYMM of TIME_SLICE

PROC_ROOT/<family>/qbin/<family>.et.qbin_<edges>.<grid>.<period>.nc
    per member: (member, stats, y_bin, x_bin), with the member coords of the inputs (source, ...)
PROC_ROOT/<family>/qbin/<family>.et.qbin_<edges>.<grid>.<period>.sources_pooled.nc
    the members of each source pooled (`binning.pool_members`): (source, stats, y_bin, x_bin).
    Equal to the per-member statistics when every source has one member
FIG_ROOT/<family>/qbin/<family>.et.qbin_<edges>.<grid>.<period>.{summary,bin_mean,bin_mean_signif}.png
    of the sources_pooled statistics; bin_mean_signif hatches bins whose mean is not significantly
    different from 0

Family-specific outputs
PROC_ROOT/obs/qbin/obs.et.qbin_<edges>.<grid>.<period>.et_product_std.nc (and FIG_ROOT .png)
    std of the bin mean ET across ET products, for each (lai, pr, rn) product set
PROC_ROOT/lens/qbin/lens.et.qbin_<edges>.<grid>.<period>.members_pooled.nc
    statistics of all LENS2 members' samples pooled
"""

from __future__ import annotations

from pathlib import Path

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
import etunc.temporal as temporal

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

BIN_FAMILIES = cfg.FAMILIES  # families to bin, e.g. ("lens",) for one
TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES
N_YBINS = cfg.N_YBINS        # LAI
N_XBINS = cfg.N_XBINS        # AI
SIGNIF_ALPHA = cfg.SIGNIF_ALPHA
SIGNIF_N_MIN = cfg.SIGNIF_N_MIN


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def pool_by_source(bs: xr.DataArray) -> xr.DataArray:
    """
    Statistics of the members of each source pooled (`binning.pool_members`), along `source`. Member
    coords that are the same for every member of a source (e.g. the products of an obs combination) are kept.
    """
    sources = list(dict.fromkeys(bs["source"].values.tolist()))
    members = {s: bs.isel(member=np.flatnonzero(bs["source"].values == s)) for s in sources}
    pooled = xr.concat([binning.pool_members(m) for m in members.values()], dim="source",
                       combine_attrs="drop_conflicts")
    same = [
        c for c in bs.coords if bs[c].dims == ("member",) and c not in ("member", "source")
        and all(len(set(m[c].values.tolist())) == 1 for m in members.values())
    ]
    return pooled.assign_coords(source=sources, **{c: ("source", [m[c].values[0] for m in members.values()]) for c in same})


def et_product_spread(bs: xr.DataArray) -> xr.Dataset:
    """
    obs only: sample std (ddof=1) of the bin mean ET across ET products, separately for each
    (lai, pr, rn) product set along `lai_pr_rn`, so only the ET product varies. Bins with fewer
    than two ET products are NaN; `n_products` counts the ET products with a bin mean.
    """
    labels = ["-".join(c) for c in zip(*(bs[f"{v}_product"].values for v in ("lai", "pr", "rn")))]
    grouped = bs.sel(stats="mean", drop=True).assign_coords(lai_pr_rn=("source", labels)).groupby("lai_pr_rn")
    with xr.set_options(keep_attrs=True):
        n = grouped.count("source")
        std = grouped.std("source", ddof=1).where(n >= 2)
    out = xr.Dataset({"et_std": std, "n_products": n.astype("i4")}, attrs=bs.attrs)
    out["et_std"].attrs["long_name"] = "std of bin mean ET across ET products (ddof=1)"
    return out.assign_coords(n_et_products=("lai_pr_rn", [labels.count(g) for g in out.lai_pr_rn.values]))


def plot_et_product_spread(spread: xr.Dataset, title: str = "", fout: Path | None = None):
    """One heatmap per (lai, pr, rn) product set of `et_product_spread`; hatched bins lack an ET product."""
    groups = spread.lai_pr_rn.values
    fig, axs = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 4.5), squeeze=False, layout="constrained")
    vmax = float(spread["et_std"].quantile(0.98))
    for ax, g in zip(axs.ravel(), groups):
        s = spread.sel(lai_pr_rn=g)
        plotting.plot_bin_field(
            s["et_std"], ax=ax, hatch=s["n_products"] < s["n_et_products"], cmap="viridis", vmin=0, vmax=vmax,
            extend="max", cbar_kwargs={"label": f"std of bin mean et [{spread.attrs.get('units', '?')}]"},
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
    cfg.summarize()
    cfg.summarize_settings(globals())
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)

    # Common area mask of all families (make_edges.py)
    area_path = cfg.area_mask_path(grid, period)
    if not area_path.exists():
        raise FileNotFoundError(f"No common area mask at {area_path}; run scripts/make_edges.py first")
    area = xr.open_dataset(area_path)["area_mask"] == 1

    for family in BIN_FAMILIES:
        print(f"\n=== {family} ===")
        path = cfg.inputs_path(family, grid, period)
        if not path.exists():
            raise FileNotFoundError(f"No {family} inputs at {path}; run scripts/prep_{family}.py first")
        inputs = xr.load_dataset(path)
        n_members = inputs.sizes["member"]

        for kind in ("pooled_all", f"pooled_{family}"):
            y_edges = edges.load_bin_edges(kind, "lai", N_YBINS, grid, period)
            x_edges = edges.load_bin_edges(kind, "ai", N_XBINS, grid, period)

            # 14. Every (year, gridcell) ET sample of a member, in the common area, binned by that gridcell's
            # climatological LAI and AI of the same member
            bs = binning.bin_stats(
                inputs["et"], inputs["lai"], inputs["ai"], y_edges, x_edges,
                mask=area, member_dim="member", y_name="lai", x_name="ai", name="et",
                attrs={
                    "family": family, "edges": kind, "time_period": inputs.attrs["time_period"], "grid": grid,
                    "n_members": n_members, "area_mask": str(area_path),
                },
            )
            stem = f"{family}.et.qbin_{kind}.{grid}.{period}"
            fout = cfg.PROC_ROOT / family / "qbin" / f"{stem}.nc"
            fout.parent.mkdir(parents=True, exist_ok=True)
            bs.to_netcdf(fout)
            count = bs.sel(stats="count").sum(("y_bin", "x_bin"))
            print(f"{kind}: {int(count.sum()):.3e} samples binned, {int(count.min()):.3e} to {int(count.max()):.3e} "
                  f"per member\n{fout}")

            by_source = pool_by_source(bs)
            fout = cfg.PROC_ROOT / family / "qbin" / f"{stem}.sources_pooled.nc"
            by_source.to_netcdf(fout)
            print(fout)

            # Figures of the sources; every source shares the edges, so the zero-width bins can be dropped
            n_sources = by_source.sizes["source"]
            title = f"{family} ET, {n_sources} sources, {kind} edges, {inputs.attrs['time_period']}"
            by_source = binning.drop_zero_width_bins(by_source)
            fig_dir = cfg.FIG_ROOT / family / "qbin"
            plotting.plot_bin_summary(by_source, dim="source", title=title, fout=fig_dir / f"{stem}.summary.png")
            plotting.plot_bin_means(by_source, "source", title=f"{title}, bin mean", fout=fig_dir / f"{stem}.bin_mean.png")
            # Hatch non-empty bins whose mean is not significantly different from 0
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
                spread = et_product_spread(by_source)
                fout = cfg.PROC_ROOT / family / "qbin" / f"{stem}.et_product_std.nc"
                spread.to_netcdf(fout)
                plot_et_product_spread(spread, title=f"obs ET, std of bin mean across ET products, {kind} edges",
                                       fout=fig_dir / f"{stem}.et_product_std.png")
                print(fout)
            if family == "lens":
                # All members' samples pooled: the LENS2 initial-condition ensemble as one distribution
                fout = cfg.PROC_ROOT / family / "qbin" / f"{stem}.members_pooled.nc"
                binning.pool_members(bs).to_netcdf(fout)
                print(fout)


if __name__ == "__main__":
    main()
