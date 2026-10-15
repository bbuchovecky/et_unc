"""
bin_lens.py
===========
Bin evapotranspiration (ET) in a 2-D space of climatological leaf area index
(LAI, y-axis) and aridity index (AI = Rn / L*P, x-axis), and plot intermediate
diagnostics, for the CESM2 Large Ensemble (LENS2) historical simulations.
Follows bin_cmip.py, with the LENS2 members in place of the CMIP6 models.

Steps
-----
1. Load MEMBERS (default: all 50 members with CMIP6 biomass burning forcing,
   not smoothed) over TIME_SLICE on the native f09 grid with `lc.load_annual`
   (`lc.load_native`, then `rg.regrid_annual`), and mask them with the LENS2 land fraction (LANDFRAC > LF_THRESH, no Greenland
   or Iceland). ET is EFLX_LH_TOT [W/m2], LAI is TLAI, PRECT is converted to
   W/m2, and Rn = FSNS - FLNS. Annual means use complete years only: a missing
   month counts as LAI = 0, while ET, pr and Rn need all 12 months.
   The annual means are conservatively regridded (xESMF,
   `rg.regrid_annual`) onto the common 1 deg grid of
   bin_cmip.py (lon in [-180, 180], lat ascending). A target cell is NaN if
   more than NA_THRESH of its area is outside the native land mask. The Natural
   Earth land mask is then applied.
2. Map the climatological ET, LAI, pr, Rn and AI: member mean, standard
   deviation and range (max - min) of each gridcell (one map for one member).
3. The binning inputs per member are annual mean ET and climatological LAI and
   AI (`lb.prepare_inputs`). They are restricted to one common area mask:
   land gridcells where ET (in any year), LAI and AI are valid in every member.
4. Bin ET separately for each member with quantile edges pooled over all
   members ("pooled_lens"). The per-member statistics are then pooled over members.

Outputs
-------
<mtag>  : ens_name ("<branch year>-<member>") for one member, "<n>members" otherwise
<et>    : ET_VAR (EFLX_LH_TOT)

PROC_ROOT/lens2.area_mask.1deg.<period>.<mtag>.nc                               common area mask
PROC_ROOT/qbin/lens2.<et>.1deg.qbin_clim_pooled_lens.<period>.<mtag>.nc         per member, `member` dim
PROC_ROOT/qbin/lens2.<et>.1deg.qbin_clim_pooled_lens.<period>.<mtag>.members_pooled.nc
    statistics of all members' samples pooled
BIN_EDGES_ROOT/lens2.{lai,ai}_clim.15_quantiles_pooled.1deg.<mtag>.<period>.nc
FIG_ROOT/lens2/<var>/lens2.<var>.1deg.map.<period>.<mtag>.png                   var in et, lai, pr, rn, ai
    member mean, standard deviation and range (one panel each)
FIG_ROOT/lens2/qbin_edges/lens2.{lai,ai}_clim.15_quantiles_pooled.1deg.<mtag>.<period>.png
FIG_ROOT/lens2/qbin/lens2.<et>.1deg.qbin_clim_pooled_lens.<period>.<mtag>.summary.png
FIG_ROOT/lens2/qbin/lens2.<et>.1deg.qbin_clim_pooled_lens.<period>.<mtag>.bin_mean.png
FIG_ROOT/lens2/qbin/lens2.<et>.1deg.qbin_clim_pooled_lens.<period>.<mtag>.bin_mean_signif.png
    as bin_mean, with bins whose mean is not significantly different from 0 hatched
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import xarray as xr

import etunc.config as cfg
import etunc.temporal as temporal
import etunc.binning as binning
from etunc.edges import EDGE_ATTRS
import etunc.legacy.binning as lb  # old edges and inputs, until bin_all.py replaces this script
import etunc.plotting as plotting
import etunc.grid as rg
import etunc.load.cesm as lc
import etunc.load.cmip as cmip

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS. Will affect performance.", category=UserWarning)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

DATASET = "lens2"  # file-name prefix and output folder
PROC_ROOT = cfg.PROC_ROOT / DATASET
BIN_EDGES_ROOT = cfg.BIN_EDGES_ROOT
FIG_ROOT = cfg.FIG_ROOT / DATASET


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

N_XBINS = cfg.N_XBINS
N_YBINS = cfg.N_YBINS
SIGNIF_ALPHA = cfg.SIGNIF_ALPHA
SIGNIF_N_MIN = cfg.SIGNIF_N_MIN

TIME_SLICE = slice("1995-01", "2014-12")
FREQUENCY = "month_1"
STREAM = "h0"
# Biomass burning forcing. Only "cmip6" (the forcing of the FHIST PPE): load_cesm2le only
# knows the cmip6 member names when it loads every member
BB = "cmip6"

# {binning input: (CESM variable, component)}. EFLX_LH_TOT is already in W/m2
ET_VAR = "EFLX_LH_TOT"
VARIABLES = {
    "et": (ET_VAR, "lnd"),
    "lai": ("TLAI", "lnd"),
    "pr": ("PRECT", "atm"),
    "fsns": ("FSNS", "atm"),
    "flns": ("FLNS", "atm"),
}

# None = all 50 members, or a list of ens_name ("<branch year>-<member>", e.g. "1001-001", "1231-005")
MEMBERS: list[str] | None = None

# Common grid
TARGET_RES = cfg.TARGET_RES
GRID_TAG = rg.grid_tag(TARGET_RES)
TARGET_GRID = rg.target_grid(TARGET_RES)


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def save_clim_map(da: xr.DataArray, variable: str, period: str, mtag: str):
    """
    Maps of the member mean, standard deviation and range of the climatology
    `da` (member, lat, lon); a single map for one member.
    """
    fout = FIG_ROOT / variable / f"{DATASET}.{variable}.{GRID_TAG}.map.{period}.{mtag}.png"
    title = f"CESM2-LE {variable}, {period}"
    if da.sizes["member"] > 1:
        plotting.save_ensemble_map(da, variable, fout, dim="member", title=title)
    else:
        plotting.save_map(da.squeeze("member"), variable, fout, title=title)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    period = temporal.format_time_period(TIME_SLICE)
    mask = rg.landne_mask(TARGET_GRID).sel(lat=cfg.LAT_BNDS)

    # ------------------------------------------------------------------
    # Load and regrid annual means, build binning inputs, and map climatologies
    # ------------------------------------------------------------------
    print("=== Load LENS2 ===")
    ann = lc.load_annual(
        "lens", MEMBERS, TIME_SLICE, TARGET_RES, variables=VARIABLES, frequency=FREQUENCY, stream=STREAM,
        bb=BB, mask=mask, na_thresh=cfg.NA_THRESH,
    )
    ens = ann["et"]["member_id"].values.tolist()
    nmem, mtag = len(ens), cmip.member_tag(ens)
    print(f"LENS2 ({BB}): {nmem} member(s)")
    inputs = lb.prepare_inputs(et=ann["et"], lai=ann["lai"], precip=ann["pr"], rn=ann["rn"], mask=mask)

    # Maps of the climatologies' member mean, std and range (before the common area mask)
    clim = {
        "et": temporal.aggregate(ann["et"], "clim"),
        "lai": inputs["lai"],
        "pr": temporal.aggregate(ann["pr"], "clim"),
        "rn": temporal.aggregate(ann["rn"], "clim"),
        "ai": inputs["ai"],
    }
    for v, da in clim.items():
        save_clim_map(da, v, period, mtag)
    del ann

    # ------------------------------------------------------------------
    # Binning inputs, restricted to the area valid in every member
    # ------------------------------------------------------------------
    print("\n=== Common area mask ===")
    area = lb.common_area({"CESM2-LE": inputs}, mask, member_dim="member")
    with xr.set_options(keep_attrs=True):
        inputs = {k: da.where(area) for k, da in inputs.items()}
    fout = PROC_ROOT / f"{DATASET}.area_mask.{GRID_TAG}.{period}.{mtag}.nc"
    fout.parent.mkdir(exist_ok=True, parents=True)
    area.astype("i1").assign_attrs(
        long_name="land gridcells where ET (any year), LAI and AI are valid in every member",
        members=ens,
    ).to_netcdf(fout)
    print(fout)

    # ------------------------------------------------------------------
    # Quantile bin edges, pooled over members
    # ------------------------------------------------------------------
    print("\n=== Bin edges ===")
    n_bins = {"lai": N_YBINS, "ai": N_XBINS}
    attrs = {"source_id": "CESM2-LE", "biomass_burning": BB, "time_period": period, "grid": GRID_TAG}
    edges = {}
    for v in ("lai", "ai"):
        print(f"\n{v}")
        edges[v] = lb.pooled_bin_edges(
            {"CESM2-LE": inputs[v]}, n_bins[v], name=v,
            attrs={**EDGE_ATTRS[v], **attrs, "members": ens},
        )
        fstem = f"{DATASET}.{v}_clim.{n_bins[v]}_quantiles_pooled.{GRID_TAG}.{mtag}.{period}"
        fout = BIN_EDGES_ROOT / f"{fstem}.nc"
        fout.parent.mkdir(exist_ok=True, parents=True)
        edges[v].to_netcdf(fout)
        print(fout)
        fout = FIG_ROOT / "qbin_edges" / f"{fstem}.png"
        plotting.plot_edges(edges[v], fout, title=f"CESM2-LE {v}, pooled over {nmem} members, {period}")
        print(fout)

    # ------------------------------------------------------------------
    # Binned ET per member, with the pooled edges
    # ------------------------------------------------------------------
    print("\n=== Binned ET ===")
    kind = "pooled_lens"
    bin_attrs = {
        **attrs,
        "n_members": nmem,
        "n_y_bins_requested": N_YBINS,
        "n_x_bins_requested": N_XBINS,
        "collapse_duplicate_quantile_bins": 0,
        "pool_edges": 1,
    }
    bs = binning.bin_stats(
        inputs["et"], inputs["lai"], inputs["ai"], edges["lai"], edges["ai"],
        member_dim="member", y_name="lai", x_name="ai", name=ET_VAR, attrs=bin_attrs,
    )

    fstem = f"{DATASET}.{ET_VAR}.{GRID_TAG}.qbin_clim_{kind}.{period}.{mtag}"
    fout = PROC_ROOT / "qbin" / f"{fstem}.nc"
    fout.parent.mkdir(exist_ok=True, parents=True)
    bs.to_netcdf(fout)
    print(fout)
    count = bs.sel(stats="count").sum(("y_bin", "x_bin"))
    print(f"{int(count.sum()):.3e} samples binned, {int(count.min()):.3e} to {int(count.max()):.3e} per member")

    fout = PROC_ROOT / "qbin" / f"{fstem}.members_pooled.nc"
    binning.pool_members(bs).to_netcdf(fout)
    print(fout)

    # Figure panels are labelled by ens_name
    bs = binning.drop_zero_width_bins(bs).swap_dims(member="member_id")
    fout = FIG_ROOT / "qbin" / f"{fstem}.summary.png"
    plotting.plot_bin_summary(
        bs, dim="member_id", title=f"CESM2-LE {ET_VAR}, {nmem} members, {kind} edges, {period}", fout=fout,
    )
    print(fout)
    fout = FIG_ROOT / "qbin" / f"{fstem}.bin_mean.png"
    plotting.plot_bin_means(
        bs, "member_id", title=f"CESM2-LE {ET_VAR} bin mean, {nmem} members, {kind} edges, {period}", fout=fout,
    )
    print(fout)

    # Hatch non-empty bins whose mean is not significantly different from 0
    signif = binning.test_significance(bs, alpha=SIGNIF_ALPHA, n_min=SIGNIF_N_MIN)
    not_signif = ~signif & bs.sel(stats="mean").notnull()
    print(f"{int(not_signif.sum())} of {int(bs.sel(stats='mean').notnull().sum())} "
          f"non-empty bins not significant at {1 - SIGNIF_ALPHA:.0%}")
    fout = FIG_ROOT / "qbin" / f"{fstem}.bin_mean_signif.png"
    plotting.plot_bin_means(
        bs, "member_id", fout=fout, hatch=not_signif,
        title=(f"CESM2-LE {ET_VAR} bin mean, {nmem} members, {kind} edges, {period}; hatched: not significantly "
               f"different from 0 at {1 - SIGNIF_ALPHA:.0%} (t-test, n > {SIGNIF_N_MIN})"),
    )
    print(fout)


if __name__ == "__main__":
    main()
