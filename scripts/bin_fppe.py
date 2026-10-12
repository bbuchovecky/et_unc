"""
bin_fppe.py
===========
Bin evapotranspiration (ET) in a 2-D space of climatological leaf area index
(LAI, y-axis) and aridity index (AI = Rn / L*P, x-axis), and plot intermediate
diagnostics, for the CESM2 FHIST perturbed-parameter ensemble (FHIST PPE,
"fppe"). Follows bin_cmip.py.

Steps
-----
1. Load MEMBERS (default: every member except DROP_OUTLIERS) over TIME_SLICE
   with `lc.load_fhist_ppe`: EFLX_LH_TOT and TLAI from "lnd", PRECT_calculated,
   FSNS and FLNS from "atm". Only members with every variable are used.
2. Mask the fields on the native f19 grid with the grid file's land fraction
   (LANDFRAC > LF_THRESH, no Greenland or Iceland). ET is EFLX_LH_TOT [W/m2],
   PRECT_calculated is converted to W/m2, and Rn = FSNS - FLNS. Annual means
   use complete years only: a missing month counts as LAI = 0, while ET, pr
   and Rn need all 12 months.
   The annual means are conservatively regridded (xESMF,
   `rg.bounded_conservative_regridder`) onto the common 1 deg grid of
   bin_cmip.py (lon in [-180, 180], lat ascending). A target cell is NaN if
   more than NA_THRESH of its area is outside the native land mask. The Natural
   Earth land mask is then applied.
3. Map the member-mean climatological ET, LAI, pr, Rn and AI.
4. The binning inputs per member are annual mean ET and climatological LAI and
   AI (`binning.prepare_inputs`). They are restricted to one common area mask:
   land gridcells where ET (in any year), LAI and AI are valid in every member.
5. Bin ET separately for each member with two sets of quantile edges:
   (a) "pooled_fppe": edges pooled across the members;
   (b) "pooled_cmip": the CMIP6 pooled edges saved by bin_cmip.py for the same
       period and grid (skipped if that file does not exist).

Outputs
-------
<mtag> : "<n>members" (member id for one member)
<kind> : "pooled_fppe" or "pooled_cmip"

PROC_ROOT/fppe.area_mask.1deg.<period>.<mtag>.nc                               common area mask
PROC_ROOT/qbin/fppe.EFLX_LH_TOT.1deg.qbin_clim_<kind>.<period>.<mtag>.nc        `member` dim (member_id = parameter name)
BIN_EDGES_ROOT/fppe.{lai,ai}_clim.15_quantiles_pooled.1deg.<mtag>.<period>.nc
FIG_ROOT/fppe/<var>/fppe.<var>.1deg.map.<period>.<mtag>.png                     var in EFLX_LH_TOT, lai, pr, rn, ai
FIG_ROOT/fppe/qbin_edges/fppe.{lai,ai}_clim.15_quantiles_pooled.1deg.<mtag>.<period>.png
FIG_ROOT/fppe/qbin_edges/fppe.{lai,ai}_clim.15_quantiles_pooled_vs_cmip.1deg.<mtag>.<period>.png
FIG_ROOT/fppe/qbin/fppe.EFLX_LH_TOT.1deg.qbin_clim_<kind>.<period>.<mtag>.summary.png
FIG_ROOT/fppe/qbin/fppe.EFLX_LH_TOT.1deg.qbin_clim_<kind>.<period>.<mtag>.bin_mean.png
FIG_ROOT/fppe/qbin/fppe.EFLX_LH_TOT.1deg.qbin_clim_<kind>.<period>.<mtag>.bin_mean_signif.png
    as bin_mean, with bins whose mean is not significantly different from 0 hatched
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import xarray as xr

import etunc.config as cfg
import etunc.temporal as temporal
import etunc.units as units
import etunc.binning as binning
import etunc.plotting as plotting
import etunc.grid as rg
import etunc.load.cesm as lc
import etunc.load.cmip as cmip

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS. Will affect performance.", category=UserWarning)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROC_ROOT = cfg.PROC_ROOT / "fppe"
BIN_EDGES_ROOT = cfg.BIN_EDGES_ROOT
FIG_ROOT = cfg.FIG_ROOT


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
# Model component of each variable (one `load_fhist_ppe` call per component)
VARIABLES = {
    "lnd": ["EFLX_LH_TOT", "TLAI"],
    "atm": ["PRECT_calculated", "FSNS", "FLNS"],
}
ET_NAME = "EFLX_LH_TOT"  # name of the binned ET in outputs

# Members (0-28): None = every member not in DROP_OUTLIERS, or an explicit list
MEMBERS: list[int] | None = None
DROP_OUTLIERS = (13, 28)  # unreasonable LAI and ET

# Common grid
TARGET_RES = cfg.TARGET_RES
GRID_TAG = rg.grid_tag(TARGET_RES)
TARGET_GRID = rg.target_grid(TARGET_RES)

# CMIP6 pooled edges written by bin_cmip.py: "all<N>" of their file names, or
# None for the only file matching the period and grid
CMIP_EDGES_TAG: str | None = None


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_members() -> dict[str, xr.Dataset]:
    """{gcomp: lazy Dataset (member, time, lat, lon)} of VARIABLES, for the members with every variable."""
    ds = {}
    for gcomp, variables in VARIABLES.items():
        d = lc.load_fhist_ppe(variables, gcomp, FREQUENCY, STREAM, members=MEMBERS, drop_outliers=DROP_OUTLIERS)
        if not d.data_vars:
            raise RuntimeError(f"No FHIST PPE files found for {variables} ({gcomp})")
        # load_fhist_ppe squeezes out `member` when only one member is found
        ds[gcomp] = d if "member" in d.dims else d.expand_dims("member")

    # A member counts only if it has every variable of every component
    ids = sorted(set.intersection(*(set(d.member.values.tolist()) for d in ds.values())))
    for gcomp, d in ds.items():
        missing = [v for v in VARIABLES[gcomp] if v not in d]
        if missing:
            raise RuntimeError(f"Missing FHIST PPE variables {missing} ({gcomp})")
        dropped = sorted(set(d.member.values.tolist()) - set(ids))
        if dropped:
            print(f"{gcomp}: members {dropped} lack another component's variables, skipping them")
    if not ids:
        raise RuntimeError("No FHIST PPE member has every variable.")
    return {gcomp: d.sel(member=ids) for gcomp, d in ds.items()}


def load_annual(mask: xr.DataArray) -> dict[str, xr.DataArray]:
    """
    Annual means (member, year, lat, lon) of ET, LAI, pr and Rn [W/m2, m2/m2],
    masked with the native land fraction, conservatively regridded onto
    TARGET_GRID within LAT_BNDS and masked with `mask`. `member` holds the
    integer member ids, `member_id` the parameter names ("2.dleaf.max").
    """
    grid = lc.load_grid("fppe")
    # Native cells with LANDFRAC > LF_THRESH, minus Greenland/Iceland: keeps ocean and
    # ice-sheet values out of the area-weighted means of the regridding
    native_mask = rg.mask_greenland(grid["LANDFRAC"], cfg.LF_THRESH)

    ds = load_members()
    fields = {}
    for gcomp, variables in VARIABLES.items():
        for v in variables:
            da = ds[gcomp][v].sel(time=TIME_SLICE)
            rg.check_same_grid(da, grid, f"fppe/{v}")
            # Exact grid-file coords, so `.where(native_mask)` cannot misalign on round-off
            da = da.assign_coords(lat=grid.lat, lon=grid.lon)
            # Drop partial first/last years: they would bias annual means toward the months present
            fields[v] = da.sel(time=da.time.dt.year.isin(temporal.complete_years(da)))

    monthly = {
        # EFLX_LH_TOT is the total latent heat flux, already in W/m2
        "et": fields["EFLX_LH_TOT"].assign_attrs(units="W/m2"),
        "lai": fields["TLAI"].assign_attrs(units="m2/m2"),
        "pr": units.convert_units("PRECT_calculated", fields["PRECT_calculated"]),
        "rn": units.net_radiation_cesm(fields["FSNS"], fields["FLNS"]),
    }

    ann = {}
    for k, da in monthly.items():
        with xr.set_options(keep_attrs=True):
            da = da.where(native_mask).load()
        # A missing LAI month counts as 0, while ET, pr and Rn need all 12 months
        ann[k] = temporal.annual_mean(da, require_all_months=(k != "lai")).transpose(..., "lat", "lon")

    ids = ann["et"].member.values.tolist()
    names = [lc.ppe_member_name(m) for m in ids]
    ann = cmip.regrid_annual(ann, rg.bounded_source_grid(grid), TARGET_RES, mask, cfg.NA_THRESH, members=names)
    print(
        f"fppe: {len(ids)} member(s) {ids}, native {native_mask.sizes['lat']}x{native_mask.sizes['lon']} "
        f"-> {ann['et'].dims} {ann['et'].shape}, years {temporal.period_str(ann['et'].year.values)}"
    )
    return ann


def load_cmip_edges(v: str, n_bins: int, period: str) -> xr.DataArray | None:
    """CMIP6 pooled edges of `v` saved by bin_cmip.py for `period` and GRID_TAG, or None if there are none."""
    tag = CMIP_EDGES_TAG or "all*"
    files = sorted(BIN_EDGES_ROOT.glob(f"cmip6.{v}_clim.{n_bins}_quantiles_pooled.{GRID_TAG}.{tag}.{period}.nc"))
    if not files:
        print(f"No CMIP6 pooled {v} edges for {period} ({tag}) in {BIN_EDGES_ROOT}")
        return None
    if len(files) > 1:
        raise ValueError(f"More than one CMIP6 pooled {v} edges file, set CMIP_EDGES_TAG: {[f.name for f in files]}")
    print(files[0])
    with xr.open_dataarray(files[0]) as da:
        return da.load()


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def save_clim_map(da: xr.DataArray, variable: str, period: str, mtag: str):
    """Map of the member-mean climatology `da`."""
    fout = FIG_ROOT / "fppe" / variable / f"fppe.{variable}.{GRID_TAG}.map.{period}.{mtag}.png"
    title = f"FHIST PPE, {period}" + (f", mean of {da.attrs['n_members']} members" if da.attrs.get("n_members", 1) > 1 else "")
    # MAP_KWARGS has no CESM names; style ET like evspsbl
    plotting.save_map(da, "evspsbl" if variable == ET_NAME else variable, fout, title=title)


def by_name(bs: xr.DataArray) -> xr.DataArray:
    """`bs` with the parameter names as `member` labels, for plot panel titles."""
    return bs.assign_coords(member=bs["member_id"].values)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    period = temporal.format_time_period(TIME_SLICE)
    mask = rg.land_mask(TARGET_GRID).sel(lat=cfg.LAT_BNDS)

    # ------------------------------------------------------------------
    # Load and regrid annual means, build binning inputs, and map
    # climatologies
    # ------------------------------------------------------------------
    print("=== Load FHIST PPE ===")
    ann = load_annual(mask)
    inputs = binning.prepare_inputs(et=ann["et"], lai=ann["lai"], precip=ann["pr"], rn=ann["rn"], mask=mask)
    inputs["et"] = inputs["et"].rename(ET_NAME)
    names = inputs["et"]["member_id"].values.tolist()
    nmem = len(names)
    mtag = cmip.member_tag(names)

    # Maps of member-mean climatologies (before the common area mask)
    clim = {
        ET_NAME: temporal.aggregate(ann["et"], "clim"),
        "lai": inputs["lai"],
        "pr": temporal.aggregate(ann["pr"], "clim"),
        "rn": temporal.aggregate(ann["rn"], "clim"),
        "ai": inputs["ai"],
    }
    for v, da in clim.items():
        da = da.mean("member", keep_attrs=True).assign_attrs(n_members=nmem)
        save_clim_map(da, v, period, mtag)
    del ann

    # ------------------------------------------------------------------
    # Binning inputs, restricted to the area valid in every member
    # ------------------------------------------------------------------
    print("\n=== Common area mask ===")
    area = binning.common_area({"fppe": inputs}, mask, member_dim="member")
    with xr.set_options(keep_attrs=True):
        inputs = {k: da.where(area) for k, da in inputs.items()}
    fout = PROC_ROOT / f"fppe.area_mask.{GRID_TAG}.{period}.{mtag}.nc"
    fout.parent.mkdir(exist_ok=True, parents=True)
    area.astype("i1").assign_attrs(
        long_name="land gridcells where ET (any year), LAI and AI are valid in every FHIST PPE member",
        members=names,
    ).to_netcdf(fout)
    print(fout)

    # ------------------------------------------------------------------
    # Quantile bin edges: pooled across members, and the CMIP6 pooled edges
    # ------------------------------------------------------------------
    print("\n=== Bin edges ===")
    n_bins = {"lai": N_YBINS, "ai": N_XBINS}
    edges = {"pooled_fppe": {}, "pooled_cmip": {}}
    for v in ("lai", "ai"):
        print(f"\n{v}")
        edges["pooled_fppe"][v] = binning.pooled_bin_edges(
            inputs[v], n_bins[v], name=v,
            attrs={**binning.EDGE_ATTRS[v], "time_period": period, "grid": GRID_TAG,
                   "pooled_sources": ["fppe"], "members": names},
        )
        fstem = f"fppe.{v}_clim.{n_bins[v]}_quantiles_pooled.{GRID_TAG}.{mtag}.{period}"
        fout = BIN_EDGES_ROOT / f"{fstem}.nc"
        fout.parent.mkdir(exist_ok=True, parents=True)
        edges["pooled_fppe"][v].to_netcdf(fout)
        print(fout)
        fout = FIG_ROOT / "fppe" / "qbin_edges" / f"{fstem}.png"
        plotting.plot_edges(edges["pooled_fppe"][v], fout, title=f"FHIST PPE {v}, pooled across {nmem} members, {period}")
        print(fout)

        cmip_edges = load_cmip_edges(v, n_bins[v], period)
        if cmip_edges is not None:
            edges["pooled_cmip"][v] = cmip_edges
            fout = FIG_ROOT / "fppe" / "qbin_edges" / f"fppe.{v}_clim.{n_bins[v]}_quantiles_pooled_vs_cmip.{GRID_TAG}.{mtag}.{period}.png"
            plotting.plot_edges_compare(
                {"FHIST PPE": edges["pooled_fppe"][v], "CMIP6": cmip_edges},
                title=f"{v} pooled edges, {period}", fout=fout,
            )
            print(fout)
    # Both LAI and AI CMIP6 edges are needed to bin
    if len(edges["pooled_cmip"]) < 2:
        del edges["pooled_cmip"]

    # ------------------------------------------------------------------
    # Binned mean ET per member
    # ------------------------------------------------------------------
    print("\n=== Binned ET ===")
    attrs = {"source": "CESM2 FHIST PPE", "time_period": period, "grid": GRID_TAG, "n_members": nmem}
    for kind, kind_edges in edges.items():
        bs = binning.bin_stats(
            inputs["et"], inputs["lai"], inputs["ai"], kind_edges["lai"], kind_edges["ai"],
            member_dim="member", y_name="lai", x_name="ai", name=ET_NAME,
            attrs={
                **attrs,
                "n_y_bins_requested": N_YBINS,
                "n_x_bins_requested": N_XBINS,
                "collapse_duplicate_quantile_bins": 0,
                "pool_edges": 1,
                "edges": kind,
            },
        )
        print(f"{kind}: {int(bs.sel(stats='count').sum()):.3e} samples binned")
        fstem = f"fppe.{ET_NAME}.{GRID_TAG}.qbin_clim_{kind}.{period}.{mtag}"
        fout = PROC_ROOT / "qbin" / f"{fstem}.nc"
        fout.parent.mkdir(exist_ok=True, parents=True)
        bs.to_netcdf(fout)
        print(fout)

        # Every member shares the edges, so the zero-width bins can be dropped
        bs = by_name(binning.drop_zero_width_bins(bs))
        fout = FIG_ROOT / "fppe" / "qbin" / f"{fstem}.summary.png"
        plotting.plot_bin_summary(
            bs, dim="member", title=f"FHIST PPE {ET_NAME}, {nmem} members, {kind} edges, {period}", fout=fout,
        )
        print(fout)
        fout = FIG_ROOT / "fppe" / "qbin" / f"{fstem}.bin_mean.png"
        plotting.plot_bin_means(
            bs, "member", title=f"FHIST PPE {ET_NAME} bin mean, {nmem} members, {kind} edges, {period}", fout=fout,
        )
        print(fout)

        # Hatch non-empty bins whose mean is not significantly different from 0
        signif = binning.test_significance(bs, alpha=SIGNIF_ALPHA, n_min=SIGNIF_N_MIN)
        not_signif = ~signif & bs.sel(stats="mean").notnull()
        print(f"{kind}: {int(not_signif.sum())} of {int(bs.sel(stats='mean').notnull().sum())} "
              f"non-empty bins not significant at {1 - SIGNIF_ALPHA:.0%}")
        fout = FIG_ROOT / "fppe" / "qbin" / f"{fstem}.bin_mean_signif.png"
        plotting.plot_bin_means(
            bs, "member", fout=fout, hatch=not_signif,
            title=(f"FHIST PPE {ET_NAME} bin mean, {nmem} members, {kind} edges, {period}; hatched: not "
                   f"significantly different from 0 at {1 - SIGNIF_ALPHA:.0%} (t-test, n > {SIGNIF_N_MIN})"),
        )
        print(fout)


if __name__ == "__main__":
    main()
