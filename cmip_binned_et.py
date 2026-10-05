"""
cmip_binned_et.py
=================
Bin evapotranspiration (ET) in a 2-D space of climatological leaf area index
(LAI, y-axis) and aridity index (AI = Rn / L*P, x-axis), and plot intermediate
diagnostics, for CMIP6 historical simulations. Follows ilamb_binned_et.py.

Steps
-----
1. Find the models with every variable (evspsbl, lai, pr, rsds, rsus, rlds,
   rlus) for at least one member plus GRID_VARIABLES, and pick each model's
   members (MEMBER_IDS / DEFAULT_MEMBERS).
2. For each model, load the members over TIME_SLICE on the native grid, mask
   with the model's land fraction (sftlf > LF_THRESH, no Greenland/Iceland),
   convert evspsbl and pr to W/m2, compute Rn = rsds - rsus + rlds - rlus and
   annual means over complete years. Annual means are bilinearly interpolated
   (xESMF) onto a common 1 deg grid (lon in [-180, 180], lat ascending) and
   masked with the Natural Earth land mask.
3. Binning inputs per member are annual mean ET and climatological LAI and AI
   (`be.prepare_inputs`), restricted to gridcells where all three are valid.
4. Map the member-mean climatological ET, LAI, pr, Rn and AI of each model.
5. Bin ET separately for each member with (a) quantile edges pooled across
   models, each model contributing its member-mean LAI and AI climatology once
   ("pooled_cmip"), and (b) each model's own quantile edges pooled over its
   members ("model").

Outputs
-------
<mtag>   : member_id for one member, "<n>members" otherwise
<kind>   : "pooled_cmip" or "model"
<runtag> : "onemember" if every model uses one member, "multimember" otherwise

PROC_ROOT/qbin/cmip6.<sid>.evspsbl.1deg.qbin_clim_<kind>.<period>.<mtag>.nc     per model, `member` dim
PROC_ROOT/qbin/cmip6.all<N>.evspsbl.1deg.qbin_clim_<kind>.<period>.<runtag>.nc  members pooled per model, `source_id` dim
BIN_EDGES_ROOT/cmip6.{lai,ai}_clim.15_quantiles_{pooled,model}.1deg.all<N>.<period>.nc
FIG_ROOT/cmip6/<var>/cmip6.<sid>.<var>.1deg.map.<period>.<mtag>.png           var in evspsbl, lai, pr, rn, ai
FIG_ROOT/cmip6/qbin_edges/cmip6.{lai,ai}_clim.15_quantiles_{pooled,model}.1deg.all<N>.<period>.png
FIG_ROOT/cmip6/qbin/cmip6.all<N>.evspsbl.1deg.qbin_clim_<kind>.<period>.<runtag>.summary.png
FIG_ROOT/cmip6/qbin/cmip6.all<N>.evspsbl.1deg.qbin_clim_<kind>.<period>.<runtag>.bin_mean.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import regionmask as regmask
import xarray as xr
import xesmf as xe

import binned_et as be
from load_cmip_esgf import CMIPESGFLoader


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

CATALOG = Path("/glade/campaign/univ/uwas0155/catalogs/cmip6_all.csv")
PROC_ROOT = Path("/glade/work/bbuchovecky/et_unc/proc/cmip6")
BIN_EDGES_ROOT = Path("/glade/work/bbuchovecky/et_unc/proc/qbin_edges")
FIG_ROOT = Path("/glade/work/bbuchovecky/et_unc/fig")


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

N_XBINS = 15    # aridity index
N_YBINS = 15    # LAI

EXPERIMENT_ID = "historical"
TIME_SLICE = slice("1995-01", "2014-12")
VARIABLES = ["evspsbl", "lai", "pr", "rsds", "rsus", "rlds", "rlus"]
GRID_VARIABLES = ["areacella", "sftlf"]  # required, as in the notebook; only sftlf is used

# None = every model with all VARIABLES and GRID_VARIABLES, minus OMIT_SOURCE_IDS
SOURCE_IDS: list[str] | None = None
OMIT_SOURCE_IDS = ["EC-Earth3", "EC-Earth3-AerChem", "EC-Earth3-ESM-1", "EC-Earth3-HR", "EC-Earth3-Veg", "EC-Earth3-Veg-LR"]

# Members per model: a list of member_ids, or
#   "top"   : first member sorted by r/i/p/f
#   "max_r" : the largest group of members sharing i/p/f (initial-condition ensemble)
#   "all"   : every member with all VARIABLES
# Models not listed use DEFAULT_MEMBERS.
DEFAULT_MEMBERS: str | list[str] = "top"
MEMBER_IDS: dict[str, str | list[str]] = {
    # "CESM2": ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"],
    # "CanESM5": "max_r",
}

# Common 1 deg grid that every model is interpolated onto
GRID_TAG = "1deg"
TARGET_GRID = xr.Dataset(coords={
    "lat": ("lat", np.arange(-89.5, 90, 1.0), {"units": "degrees_north"}),
    "lon": ("lon", np.arange(-179.5, 180, 1.0), {"units": "degrees_east"}),
})

MAP_KWARGS = {
    "evspsbl": {"cmap": "YlGnBu"},
    "lai":     {"cmap": "Greens"},
    "pr":      {"cmap": "Blues"},
    "rn":      {"cmap": "YlOrRd", "vmin": None},
    "ai":      {"cmap": "BrBG_r"},
}

EDGE_ATTRS = {
    "lai": {"long_name": "leaf area index bin edges", "units": "m2/m2", "variable": "leaf area index"},
    "ai":  {"long_name": "aridity index bin edges", "units": "1", "variable": "aridity index (Rn/L*P)"},
}


# ------------------------------------------------------------------
# Model and member selection
# ------------------------------------------------------------------

def available_members(catalog: pd.DataFrame) -> dict[str, list[str]]:
    """
    {source_id: sorted members with every VARIABLE} for models that also have
    each GRID_VARIABLE for at least one member.
    """
    hist = catalog[catalog["experiment_id"] == EXPERIMENT_ID]
    avail = {}
    for sid, g in hist.groupby("source_id"):
        mids = set.intersection(*(set(g.loc[g["variable_id"] == v, "member_id"]) for v in VARIABLES))
        if mids and all((g["variable_id"] == v).any() for v in GRID_VARIABLES):
            avail[sid] = CMIPESGFLoader.sort_member_ids(mids)
    return avail


def select_members(sid: str, avail: list[str]) -> list[str]:
    """Resolve MEMBER_IDS / DEFAULT_MEMBERS for one model against its available members."""
    spec = MEMBER_IDS.get(sid, DEFAULT_MEMBERS)
    if spec == "top":
        return avail[:1]
    if spec == "all":
        return avail
    if spec == "max_r":
        return max(CMIPESGFLoader.group_member_ids_by_ipf(avail).values(), key=len)
    if isinstance(spec, str):
        raise ValueError(f"{sid}: unknown member selection {spec!r}")
    missing = [m for m in spec if m not in avail]
    if missing:
        print(f"{sid}: members {missing} do not have all of {VARIABLES}, skipping them")
    return [m for m in spec if m in avail]


def grid_member(catalog: pd.DataFrame, sid: str, members: list[str]) -> str:
    """Member to take sftlf from: the first selected member that has it, else any member that has it."""
    rows = catalog[
        (catalog["experiment_id"] == EXPERIMENT_ID) & (catalog["source_id"] == sid) & (catalog["variable_id"] == "sftlf")
    ]
    have = set(rows["member_id"])
    for m in members:
        if m in have:
            return m
    return CMIPESGFLoader.sort_member_ids(have)[0]


def member_tag(members: list[str]) -> str:
    return members[0] if len(members) == 1 else f"{len(members)}members"


# ------------------------------------------------------------------
# Loading and formatting
# ------------------------------------------------------------------

def format_grid(da: xr.DataArray) -> xr.DataArray:
    """Longitude in [-180, 180] and latitude ascending."""
    return da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon").sortby("lat")


def complete_years(da: xr.DataArray) -> list[int]:
    """Years with all 12 months in the time axis."""
    years, months = da.time.dt.year.values, da.time.dt.month.values
    return [int(y) for y in np.unique(years) if np.unique(months[years == y]).size == 12]


def annual_mean(da: xr.DataArray, require_all_months: bool) -> xr.DataArray:
    """
    Annual mean via `be.aggregate`, which counts missing months as 0. With
    `require_all_months`, years with any missing month are NaN instead;
    otherwise only years without any valid month are NaN.
    """
    n_valid = da.notnull().groupby("time.year").sum()
    with xr.set_options(keep_attrs=True):
        ann = be.aggregate(da, "year")
        return ann.where(n_valid == 12 if require_all_months else n_valid > 0)


def land_mask(grid: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """
    Natural Earth land mask without Greenland/Iceland. (Same land mask as
    `be.compute_cell_area`, which is not used because importing ILAMB
    initializes MPI.)
    """
    land = regmask.defined_regions.natural_earth_v5_1_2.land_50.mask(grid.lon, grid.lat)
    return be.mask_greenland(xr.where(land.notnull(), 1.0, 0.0))


def load_model(
    loader: CMIPESGFLoader,
    sid: str,
    members: list[str],
    fx_member: str,
    mask: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """
    Annual means (member, year, lat, lon) of ET, LAI, pr and Rn [W/m2, m2/m2]
    for one model on TARGET_GRID within LAT_BNDS, masked with `mask`.
    """
    lf = loader.load_data(["sftlf"], EXPERIMENT_ID, source_id=sid, member_id=[fx_member], verbose=False)[sid]["sftlf"]
    lf = format_grid(lf.isel(member=0, drop=True)).load()
    if lf.attrs.get("units") == "%":  # sftlf is in %, LF_THRESH is a fraction
        lf = lf / 100
    native_mask = be.mask_greenland(lf, be.LF_THRESH)

    data = loader.load_data(VARIABLES, EXPERIMENT_ID, source_id=sid, member_id=members, time_slice=TIME_SLICE)[sid]
    missing = set(VARIABLES) - set(data)
    if missing:
        raise ValueError(f"{sid}: could not load {sorted(missing)}")

    fields = {}
    for v in VARIABLES:
        da = format_grid(data[v]).reset_coords(drop=True)  # member_id is re-added after regridding
        be.check_same_grid(da, lf, f"{sid}/{v}")
        da = da.assign_coords(lat=lf.lat, lon=lf.lon)
        fields[v] = da.sel(time=da.time.dt.year.isin(complete_years(da)))

    monthly = {
        "et": be.convert_units("evspsbl", fields["evspsbl"]),
        "lai": fields["lai"].assign_attrs(units="m2/m2"),
        "pr": be.convert_units("pr", fields["pr"]),
        "rn": be.net_radiation_cmip(fields["rsds"], fields["rsus"], fields["rlds"], fields["rlus"]),
    }

    src = xr.Dataset(coords={"lat": lf.lat, "lon": lf.lon})
    regridder = xe.Regridder(src, TARGET_GRID, "bilinear", periodic=True, unmapped_to_nan=True)

    ann = {}
    for k, da in monthly.items():
        with xr.set_options(keep_attrs=True):
            da = da.where(native_mask).load()
        a = annual_mean(da, require_all_months=(k != "lai")).transpose(..., "lat", "lon")
        # Model ocean is NaN, so skipna keeps coastal target cells from land values only
        a = regridder(a, skipna=True, na_thres=1.0, keep_attrs=True)
        a = a.assign_coords(lat=TARGET_GRID.lat, lon=TARGET_GRID.lon, member_id=("member", members))
        with xr.set_options(keep_attrs=True):
            ann[k] = a.sel(lat=be.LAT_BNDS).where(mask).rename(k)

    print(
        f"{sid:16}: {len(members)} member(s), native {lf.sizes['lat']}x{lf.sizes['lon']} "
        f"(sftlf from {fx_member}) -> {ann['et'].dims} {ann['et'].shape}"
    )
    return ann


# ------------------------------------------------------------------
# Binning helpers
# ------------------------------------------------------------------

def period_str(years) -> str:
    """[1995, ..., 2014] -> "199501-201412"."""
    return be.format_time_period(slice(f"{min(years)}-01", f"{max(years)}-12"))


def model_inputs(ann: dict[str, xr.DataArray], mask: xr.DataArray) -> dict[str, xr.DataArray]:
    """
    `be.prepare_inputs` for each member of one model, restricted to gridcells
    where ET (in any year), LAI and AI are all valid.
    """
    inputs = be.prepare_inputs(et=ann["et"], lai=ann["lai"], precip=ann["pr"], rn=ann["rn"], mask=mask)
    valid = inputs["et"].notnull().any("year") & inputs["lai"].notnull() & np.isfinite(inputs["ai"])
    with xr.set_options(keep_attrs=True):
        return {k: da.where(valid) for k, da in inputs.items()}


def pool_members(bs: xr.DataArray) -> xr.DataArray:
    """Combine per-member bin statistics into the statistics of all members' samples pooled."""
    mean, var_pop, count, count_pos = (bs.sel(stats=s, drop=True) for s in ("mean", "var_pop", "count", "count_pos"))
    n = count.sum("member")
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled_mean = (mean * count).sum("member") / n
        pooled_var = ((var_pop + mean**2) * count).sum("member") / n - pooled_mean**2
        pooled_var = pooled_var.clip(min=0)
        var_samp = xr.where(n > 1, pooled_var * n / (n - 1), np.nan)
    out = xr.concat(
        [pooled_mean.where(n > 0), pooled_var.where(n > 0), var_samp, n, count_pos.sum("member")], dim="stats",
    ).assign_coords(stats=list(be.STATS)).transpose("stats", "y_bin", "x_bin")
    out = out.drop_vars([c for c in out.coords if "member" in out[c].dims or c == "member_id"], errors="ignore")
    out.attrs = {**bs.attrs, "members": list(bs["member_id"].values)}
    return out.rename(bs.name)


def concat_models(das: list[xr.DataArray], sids: list[str]) -> xr.DataArray:
    return xr.concat(
        das, dim="source_id", coords="different", compat="equals", join="exact", combine_attrs="drop_conflicts",
    ).assign_coords(source_id=sids)


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def save_map(da: xr.DataArray, variable: str, sid: str, period: str, mtag: str):
    fout = FIG_ROOT / "cmip6" / variable / f"cmip6.{sid}.{variable}.{GRID_TAG}.map.{period}.{mtag}.png"
    title = f"{sid}, {period}" + (f", mean of {da.attrs['n_members']} members" if da.attrs.get("n_members", 1) > 1 else "")
    be.quick_map(
        da, fout, title=title,
        cbar_kwargs={"label": f"{variable} [{da.attrs.get('units', '?')}]"}, **MAP_KWARGS[variable],
    )
    print(fout)


def plot_model_bin_means(bs_all: xr.DataArray, title: str = "", fout: Path | None = None, col_wrap: int = 6):
    """
    One heatmap of the bin mean ET per model on a shared colour scale. Panels
    share axes, so ticks show the bin edge values when all models share edges,
    and quantile levels when each has its own edges.
    """
    fg = be.plot_bin_facets(
        bs_all, dim="source_id", col_wrap=min(col_wrap, bs_all.sizes["source_id"]), size=3,
        cmap="YlGnBu", robust=True, cbar_kwargs={"label": f"bin mean evspsbl [{bs_all.attrs.get('units', '?')}]"},
    )
    for ax, name_dict in zip(fg.axs.flat, fg.name_dicts.flat):
        if name_dict is not None:
            ax.set_title(name_dict["source_id"], fontsize=8)
    if "source_id" not in bs_all["y_bin_lower"].dims and "source_id" not in bs_all["x_bin_lower"].dims:
        be._edge_ticks(fg.axs.flat[0], bs_all, "{:.2g}")
    else:
        for dim, set_ticks in (("x_bin", fg.axs.flat[0].set_xticks), ("y_bin", fg.axs.flat[0].set_yticks)):
            n = bs_all.sizes[dim]
            set_ticks(np.arange(n + 1) - 0.5, [f"Q{q:.0f}" for q in np.linspace(0, 100, n + 1)])
    for ax in fg.axs.flat:
        ax.tick_params(axis="both", labelsize=7)
        ax.tick_params(axis="x", labelrotation=90)
    fg.set_axis_labels("ai $\\rightarrow$", "lai $\\rightarrow$")
    fg.fig.suptitle(title, y=1.02)
    return be._finish(fg.fig, fout)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    period = be.format_time_period(TIME_SLICE)
    mask = land_mask(TARGET_GRID).sel(lat=be.LAT_BNDS)
    loader = CMIPESGFLoader(CATALOG)

    # ------------------------------------------------------------------
    # Models and members
    # ------------------------------------------------------------------
    print("=== Models and members ===")
    avail = available_members(loader.catalog)
    sids = sorted(set(SOURCE_IDS or avail) - set(OMIT_SOURCE_IDS))
    unavailable = [s for s in sids if s not in avail]
    if unavailable:
        print(f"Not all of {VARIABLES + GRID_VARIABLES} available, skipping: {unavailable}")
    members = {}
    for sid in sids:
        if sid in avail:
            members[sid] = select_members(sid, avail[sid])
            print(f"{sid:16}: {len(members[sid])} of {len(avail[sid])} members {members[sid]}")
    members = {sid: m for sid, m in members.items() if m}

    # ------------------------------------------------------------------
    # Load annual means and build binning inputs, one model at a time
    # ------------------------------------------------------------------
    print("\n=== Load CMIP6 models ===")
    inputs = {}
    for sid, mids in members.items():
        try:
            ann = load_model(loader, sid, mids, grid_member(loader.catalog, sid, mids), mask)
        except ValueError as err:
            print(f"{sid}: {err}, skipping")
            continue
        inputs[sid] = model_inputs(ann, mask)

        # Maps of member-mean climatologies
        mtag = member_tag(mids)
        clim = {
            "evspsbl": be.aggregate(ann["et"], "clim"),
            "lai": inputs[sid]["lai"],
            "pr": be.aggregate(ann["pr"], "clim"),
            "rn": be.aggregate(ann["rn"], "clim"),
            "ai": inputs[sid]["ai"],
        }
        for v, da in clim.items():
            da = da.mean("member", keep_attrs=True).assign_attrs(n_members=len(mids))
            save_map(da, v, sid, period, mtag)
        del ann

    sids = list(inputs)
    nsid = len(sids)
    if nsid == 0:
        raise RuntimeError("No model could be loaded.")
    runtag = "onemember" if all(len(members[s]) == 1 for s in sids) else "multimember"

    # ------------------------------------------------------------------
    # Quantile bin edges pooled across models (member-mean field per model)
    # ------------------------------------------------------------------
    print("\n=== Pooled bin edges ===")
    n_bins = {"lai": N_YBINS, "ai": N_XBINS}
    pooled_edges = {}
    for v in ("lai", "ai"):
        print(f"\n{v}")
        pooled_edges[v] = be.pooled_bin_edges(
            {sid: inputs[sid][v].mean("member") for sid in sids}, n_bins[v], name=v,
            attrs={**EDGE_ATTRS[v], "time_period": period, "grid": GRID_TAG,
                   "members": [f"{sid}.{m}" for sid in sids for m in members[sid]]},
        )
        fstem = f"cmip6.{v}_clim.{n_bins[v]}_quantiles_pooled.{GRID_TAG}.all{nsid}.{period}"
        fout = BIN_EDGES_ROOT / f"{fstem}.nc"
        fout.parent.mkdir(exist_ok=True, parents=True)
        pooled_edges[v].to_netcdf(fout)
        print(fout)
        fout = FIG_ROOT / "cmip6" / "qbin_edges" / f"{fstem}.png"
        be.plot_edges(pooled_edges[v], fout, title=f"CMIP6 {v}, pooled across {nsid} models, {period}")
        print(fout)

    # ------------------------------------------------------------------
    # Binned mean ET per member, with pooled and with per-model edges
    # ------------------------------------------------------------------
    print("\n=== Binned ET ===")
    bs = {"pooled_cmip": [], "model": []}
    model_edges = {"lai": {}, "ai": {}}
    for sid in sids:
        mids, mtag = members[sid], member_tag(members[sid])
        attrs = {"source_id": sid, "time_period": period, "grid": GRID_TAG, "n_members": len(mids)}

        edges = {}
        for v in ("lai", "ai"):
            edges[v] = be.pooled_bin_edges(
                inputs[sid][v], n_bins[v], name=v, verbose=False,
                attrs={**EDGE_ATTRS[v], **attrs, "pool_edges": 0, "pooled_sources": [sid], "members": mids},
            )
            model_edges[v][sid] = edges[v]

        for kind, (y_edges, x_edges) in (
            ("pooled_cmip", (pooled_edges["lai"], pooled_edges["ai"])),
            ("model", (edges["lai"], edges["ai"])),
        ):
            bs_m = be.bin_stats(
                inputs[sid]["et"], inputs[sid]["lai"], inputs[sid]["ai"], y_edges, x_edges,
                member_dim="member", y_name="lai", x_name="ai", name="evspsbl",
                attrs={
                    **attrs,
                    "n_y_bins_requested": N_YBINS,
                    "n_x_bins_requested": N_XBINS,
                    "collapse_duplicate_quantile_bins": 0,
                    "pool_edges": int(kind == "pooled_cmip"),
                },
            )
            fout = PROC_ROOT / "qbin" / f"cmip6.{sid}.evspsbl.{GRID_TAG}.qbin_clim_{kind}.{period}.{mtag}.nc"
            fout.parent.mkdir(exist_ok=True, parents=True)
            bs_m.to_netcdf(fout)
            bs[kind].append(pool_members(bs_m))

        print(f"{sid:16}: {len(mids)} member(s), {int(bs['model'][-1].sel(stats='count').sum()):.3e} samples binned")

    for kind, bs_list in bs.items():
        bs_all = concat_models(bs_list, sids)
        fstem = f"cmip6.all{nsid}.evspsbl.{GRID_TAG}.qbin_clim_{kind}.{period}.{runtag}"
        fout = PROC_ROOT / "qbin" / f"{fstem}.nc"
        bs_all.to_netcdf(fout)
        print(fout)

        # Zero-width bins only exist (and can only be dropped) for the shared pooled edges
        if kind == "pooled_cmip":
            bs_all = be.drop_zero_width_bins(bs_all)
        fout = FIG_ROOT / "cmip6" / "qbin" / f"{fstem}.summary.png"
        be.plot_bin_summary(bs_all, dim="source_id", title=f"CMIP6 evspsbl, {nsid} models, {kind} edges", fout=fout)
        print(fout)
        fout = FIG_ROOT / "cmip6" / "qbin" / f"{fstem}.bin_mean.png"
        plot_model_bin_means(bs_all, title=f"CMIP6 evspsbl bin mean, {nsid} models, {kind} edges, {period}", fout=fout)
        print(fout)

    # ------------------------------------------------------------------
    # Per-model bin edges
    # ------------------------------------------------------------------
    for v in ("lai", "ai"):
        edges_all = concat_models(list(model_edges[v].values()), sids)
        fstem = f"cmip6.{v}_clim.{n_bins[v]}_quantiles_model.{GRID_TAG}.all{nsid}.{period}"
        fout = BIN_EDGES_ROOT / f"{fstem}.nc"
        edges_all.to_netcdf(fout)
        print(fout)
        fout = FIG_ROOT / "cmip6" / "qbin_edges" / f"{fstem}.png"
        be.plot_edges_compare(
            {**model_edges[v], "pooled": pooled_edges[v]}, title=f"CMIP6 {v}, per-model edges, {period}", fout=fout,
        )
        print(fout)


if __name__ == "__main__":
    main()
