"""
cmip_binned_et.py
=================
Bin evapotranspiration (ET) in a 2-D space of climatological leaf area index
(LAI, y-axis) and aridity index (AI = Rn / L*P, x-axis), and plot intermediate
diagnostics, for CMIP6 historical simulations. Follows ilamb_binned_et.py and
obs_binned_et.py.

Steps
-----
1. Use every model in CATALOG that has all VARIABLES for at least one member
   and an sftlf file in FX_CATALOG, minus OMIT_SOURCE_IDS. Each model's members
   come from MEMBER_IDS / DEFAULT_MEMBERS.
2. For each model, load the members over TIME_SLICE on the native grid and mask
   them with the model's land fraction (sftlf > LF_THRESH, no Greenland or
   Iceland). evspsbl and pr are converted to W/m2, and Rn = rsds - rsus + rlds
   - rlus. Annual means use complete years only: a missing month counts as
   LAI = 0, while ET, pr and Rn need all 12 months.
   The annual means are conservatively regridded (xESMF,
   `rg.bounded_conservative_regridder`) onto the common 1 deg grid (lon in
   [-180, 180], lat ascending). A target cell is NaN if more than NA_THRES of
   its area is outside the native land mask. The Natural Earth land mask is
   then applied. A model whose sftlf is not on its data grid is skipped
   (MIROC-ES2H).
3. Map each model's member-mean climatological ET, LAI, pr, Rn and AI.
4. The binning inputs per member are annual mean ET and climatological LAI and
   AI (`be.prepare_inputs`). They are restricted to one common area mask: land
   gridcells where ET (in any year), LAI and AI are valid in every member of
   every model, so all models cover the same area.
5. Bin ET separately for each member with two sets of quantile edges:
   (a) "pooled_cmip": edges pooled across models, where each model contributes
       its member-mean LAI and AI once;
   (b) "model": each model's own edges, pooled over its members.
   The per-member statistics are then pooled within each model.

Outputs
-------
<mtag>   : member_id for one member, "<n>members" otherwise
<kind>   : "pooled_cmip" or "model"
<runtag> : "onemember" if every model uses one member, "multimember" otherwise

PROC_ROOT/cmip6.area_mask.all<N>.1deg.<period>.<runtag>.nc                    common area mask
PROC_ROOT/qbin/cmip6.<sid>.evspsbl.1deg.qbin_clim_<kind>.<period>.<mtag>.nc     per model, `member` dim
PROC_ROOT/qbin/cmip6.all<N>.evspsbl.1deg.qbin_clim_<kind>.<period>.<runtag>.nc  members pooled per model, `source_id` dim
BIN_EDGES_ROOT/cmip6.{lai,ai}_clim.15_quantiles_{pooled,model}.1deg.all<N>.<period>.nc
FIG_ROOT/cmip6/<var>/cmip6.<sid>.<var>.1deg.map.<period>.<mtag>.png           var in evspsbl, lai, pr, rn, ai
FIG_ROOT/cmip6/qbin_edges/cmip6.{lai,ai}_clim.15_quantiles_{pooled,model}.1deg.all<N>.<period>.png
FIG_ROOT/cmip6/qbin/cmip6.all<N>.evspsbl.1deg.qbin_clim_<kind>.<period>.<runtag>.summary.png
FIG_ROOT/cmip6/qbin/cmip6.all<N>.evspsbl.1deg.qbin_clim_<kind>.<period>.<runtag>.bin_mean.png
FIG_ROOT/cmip6/qbin/cmip6.all<N>.evspsbl.1deg.qbin_clim_<kind>.<period>.<runtag>.bin_mean_signif.png
    as bin_mean, with bins whose mean is not significantly different from 0 hatched
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import regionmask as regmask
import xarray as xr

import binned_et as be
import regrid as rg
from load_cmip_esgf import CMIPESGFLoader

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS. Will affect performance.", category=UserWarning)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

CATALOG_ROOT = Path("/glade/derecho/scratch/bbuchovecky/cmip_intake_esgf_fetch/catalogs")
CATALOG = CATALOG_ROOT / "cmip_evap.csv"          # evspsbl, lai, pr, radiative fluxes
FX_CATALOG = CATALOG_ROOT / "cmip6_fx_glade.csv"  # one sftlf file per model, from any experiment
PROC_ROOT = Path("/glade/work/bbuchovecky/et_unc/proc/cmip6")
BIN_EDGES_ROOT = Path("/glade/work/bbuchovecky/et_unc/proc/qbin_edges")
FIG_ROOT = Path("/glade/work/bbuchovecky/et_unc/fig")


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

N_XBINS = 15    # aridity index
N_YBINS = 15    # LAI
SIGNIF_ALPHA = 0.05  # t-test of bin mean ET against 0 (95% confidence)
SIGNIF_N_MIN = 10    # bins with <= this many samples are never significant

EXPERIMENT_ID = "historical"
TIME_SLICE = slice("1995-01", "2014-12")
VARIABLES = ["evspsbl", "lai", "pr", "rsds", "rsus", "rlds", "rlus"]

# None = every model with all VARIABLES and sftlf, minus OMIT_SOURCE_IDS
SOURCE_IDS: list[str] | None = None
OMIT_SOURCE_IDS: list[str] = []

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

# Common 1 deg grid that every model is conservatively regridded onto
TARGET_RES = 1.0
GRID_TAG = rg.grid_tag(TARGET_RES)
TARGET_GRID = rg.target_grid(TARGET_RES)
NA_THRES = 0.5  # target cells with more than this fraction of area outside the native land mask are NaN

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
    """{source_id: sorted members with every VARIABLE} for EXPERIMENT_ID."""
    hist = catalog[catalog["experiment_id"] == EXPERIMENT_ID]
    avail = {}
    for sid, g in hist.groupby("source_id"):
        mids = set.intersection(*(set(g.loc[g["variable_id"] == v, "member_id"]) for v in VARIABLES))
        if mids:
            avail[sid] = CMIPESGFLoader.sort_member_ids(mids)
    return avail


def sftlf_files(fx_catalog: Path) -> dict[str, str]:
    """{source_id: sftlf file}. The fx catalog holds one file per model (often from piControl)."""
    fx = pd.read_csv(fx_catalog)
    rows = fx[fx["variable_id"] == "sftlf"]
    dup = rows["source_id"][rows["source_id"].duplicated()].tolist()
    if dup:
        raise ValueError(f"{fx_catalog}: more than one sftlf file for {dup}")
    return dict(zip(rows["source_id"], rows["path"]))


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


def member_tag(members: list[str]) -> str:
    return members[0] if len(members) == 1 else f"{len(members)}members"


# ------------------------------------------------------------------
# Loading and regridding
# ------------------------------------------------------------------

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


def load_land_fraction(path: str | Path) -> tuple[xr.DataArray, xr.Dataset]:
    """Land fraction [0-1] on a model's native grid, and that grid with cell edges for regridding."""
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    lf = ds["sftlf"].reset_coords(drop=True)
    # sftlf should be in %, but some files store a fraction labelled "%" (E3SM-1-0), so check the values
    if float(lf.max()) > 1.5:
        lf = lf / 100
    return lf.assign_attrs(units="1"), rg.bounded_source_grid(ds)


def load_model(
    loader: CMIPESGFLoader,
    sid: str,
    members: list[str],
    sftlf_path: str,
    mask: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """
    Annual means (member, year, lat, lon) of ET, LAI, pr and Rn [W/m2, m2/m2]
    for one model, conservatively regridded onto TARGET_GRID within LAT_BNDS
    and masked with `mask`.
    """
    lf, src_grid = load_land_fraction(sftlf_path)
    native_mask = be.mask_greenland(lf, be.LF_THRESH)

    data = loader.load_data(
        VARIABLES, EXPERIMENT_ID, source_id=sid, member_id=members, time_slice=TIME_SLICE, verbose=False,
    )[sid]
    missing = set(VARIABLES) - set(data)
    if missing:
        raise ValueError(f"{sid}: could not load {sorted(missing)}")

    fields = {}
    for v in VARIABLES:
        da = data[v].reset_coords(drop=True)  # member_id is re-added after regridding
        # sftlf comes from another experiment, so check that it is on the same grid
        be.check_same_grid(da, lf, f"{sid}/{v}")
        da = da.assign_coords(lat=lf.lat, lon=lf.lon)
        fields[v] = da.sel(time=da.time.dt.year.isin(complete_years(da)))

    monthly = {
        "et": be.convert_units("evspsbl", fields["evspsbl"]),
        "lai": fields["lai"].assign_attrs(units="m2/m2"),
        "pr": be.convert_units("pr", fields["pr"]),
        "rn": be.net_radiation_cmip(fields["rsds"], fields["rsus"], fields["rlds"], fields["rlus"]),
    }

    regridder = rg.bounded_conservative_regridder(src_grid, TARGET_RES)

    ann = {}
    for k, da in monthly.items():
        with xr.set_options(keep_attrs=True):
            da = da.where(native_mask).load()
        a = annual_mean(da, require_all_months=(k != "lai")).transpose(..., "lat", "lon")
        # Area-weighted mean of the native land cells in each target cell (ocean is NaN and skipped)
        a = regridder(a, skipna=True, na_thres=NA_THRES, keep_attrs=True)
        a = a.assign_coords(lat=TARGET_GRID.lat, lon=TARGET_GRID.lon, member_id=("member", members))
        with xr.set_options(keep_attrs=True):
            ann[k] = a.sel(lat=be.LAT_BNDS).where(mask).rename(k)

    print(
        f"{sid:16}: {len(members)} member(s), native {lf.sizes['lat']}x{lf.sizes['lon']} "
        f"-> {ann['et'].dims} {ann['et'].shape}, years {period_str(ann['et'].year.values)}"
    )
    return ann


# ------------------------------------------------------------------
# Binning helpers
# ------------------------------------------------------------------

def period_str(years) -> str:
    """[1995, ..., 2014] -> "199501-201412"."""
    return be.format_time_period(slice(f"{min(years)}-01", f"{max(years)}-12"))


def valid_area(inputs: dict[str, xr.DataArray]) -> xr.DataArray:
    """Gridcells where ET (in any year), LAI and AI are valid in every member of one model."""
    valid = inputs["et"].notnull().any("year") & inputs["lai"].notnull() & np.isfinite(inputs["ai"])
    return valid.all("member")


def common_area(inputs: dict[str, dict[str, xr.DataArray]], mask: xr.DataArray) -> xr.DataArray:
    """
    Land gridcells valid (`valid_area`) in every model, so that all models
    cover the same area. The mask is static: ET years that are NaN inside this
    area stay NaN.
    """
    area = mask == 1
    for sid, inp in inputs.items():
        valid = valid_area(inp)
        print(f"{sid:16}: {int(valid.sum()):.4e} valid gridcells")
        area = area & valid
    print(f"common area: {int(area.sum()):.4e} of {int((mask == 1).sum()):.4e} land gridcells")
    return area.rename("area_mask")


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


def plot_model_edges(
    model_edges: dict[str, xr.DataArray], pooled: xr.DataArray, title: str = "", fout: Path | None = None,
):
    """Per-model bin edges (without the maximum edge) as thin lines, with the pooled edges in black."""
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    cmap = plt.get_cmap("tab20")
    styles = ("-", ":", "-.")  # cycle line styles so that more than 20 models stay distinguishable
    edge = pooled.edge.values[:-1]
    for i, (sid, e) in enumerate(model_edges.items()):
        ax.plot(edge, e.values[:-1], color=cmap(i % 20), ls=styles[(i // 20) % 3], lw=0.8, alpha=0.8, label=sid)
    ax.plot(edge, pooled.values[:-1], color="k", lw=2, marker="o", ms=3, label="pooled")
    ax.set_xlabel("edge")
    ax.set_ylabel(f"{pooled.name} [{pooled.attrs.get('units', '?')}]")
    ax.legend(fontsize=6, ncols=2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(title)
    return be._finish(fig, fout)


def hatch_bins(ax, hatch: xr.DataArray):
    """Hatch the cells of a (y_bin, x_bin) heatmap where `hatch` is True."""
    hatch = hatch.transpose("y_bin", "x_bin")
    yb, xb = hatch["y_bin"].values, hatch["x_bin"].values
    for j, i in zip(*np.nonzero(hatch.fillna(False).astype(bool).values)):
        ax.add_patch(Rectangle(
            (xb[i] - 0.5, yb[j] - 0.5), 1, 1, fill=False, hatch="///", lw=0, edgecolor="0.3",
        ))


def plot_model_bin_means(
    bs_all: xr.DataArray,
    title: str = "",
    fout: Path | None = None,
    col_wrap: int = 6,
    hatch: xr.DataArray | None = None,
):
    """
    One heatmap of the bin mean ET per model on a shared colour scale. Panels
    share axes, so ticks show the bin edge values when all models share edges,
    and quantile levels when each has its own edges.

    hatch : boolean (source_id, y_bin, x_bin) field; True bins are hatched
        (e.g. ~`be.test_significance`)
    """
    fg = be.plot_bin_facets(
        bs_all, dim="source_id", col_wrap=min(col_wrap, bs_all.sizes["source_id"]), size=3,
        cmap="YlGnBu", robust=True, cbar_kwargs={"label": f"bin mean evspsbl [{bs_all.attrs.get('units', '?')}]"},
    )
    for ax, name_dict in zip(fg.axs.flat, fg.name_dicts.flat):
        if name_dict is None:
            continue
        ax.set_title(name_dict["source_id"], fontsize=8)
        if hatch is not None:
            hatch_bins(ax, hatch.sel(**name_dict))
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
    # Models (all VARIABLES and sftlf) and members
    # ------------------------------------------------------------------
    print("=== Models and members ===")
    avail = available_members(loader.catalog)
    sftlf = sftlf_files(FX_CATALOG)
    sids = sorted(set(SOURCE_IDS or avail) - set(OMIT_SOURCE_IDS))
    no_vars = [s for s in sids if s not in avail]
    no_sftlf = [s for s in sids if s in avail and s not in sftlf]
    if no_vars:
        print(f"Not all of {VARIABLES} available, skipping: {no_vars}")
    if no_sftlf:
        print(f"No sftlf in {FX_CATALOG.name}, skipping: {no_sftlf}")
    members = {}
    for sid in sids:
        if sid in avail and sid in sftlf:
            members[sid] = select_members(sid, avail[sid])
            print(f"{sid:16}: {len(members[sid])} of {len(avail[sid])} members {members[sid]}")
    members = {sid: m for sid, m in members.items() if m}

    # ------------------------------------------------------------------
    # Load and regrid annual means, build binning inputs, and map
    # climatologies, one model at a time
    # ------------------------------------------------------------------
    print("\n=== Load CMIP6 models ===")
    inputs = {}
    for sid, mids in members.items():
        try:
            ann = load_model(loader, sid, mids, sftlf[sid], mask)
        except ValueError as err:  # e.g. sftlf on a different grid than the data
            print(f"{sid}: {err}, skipping")
            continue
        inputs[sid] = be.prepare_inputs(et=ann["et"], lai=ann["lai"], precip=ann["pr"], rn=ann["rn"], mask=mask)

        # Maps of member-mean climatologies (each model's own area, before the common area mask)
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
    # Binning inputs, restricted to the area valid in every model
    # ------------------------------------------------------------------
    print("\n=== Common area mask ===")
    area = common_area(inputs, mask)
    with xr.set_options(keep_attrs=True):
        inputs = {sid: {k: da.where(area) for k, da in inp.items()} for sid, inp in inputs.items()}
    fout = PROC_ROOT / f"cmip6.area_mask.all{nsid}.{GRID_TAG}.{period}.{runtag}.nc"
    fout.parent.mkdir(exist_ok=True, parents=True)
    area.astype("i1").assign_attrs(
        long_name="land gridcells where ET (any year), LAI and AI are valid in every member of every model",
        source_ids=sids,
        members=[f"{sid}.{m}" for sid in sids for m in members[sid]],
    ).to_netcdf(fout)
    print(fout)

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

        # This model's own edges, pooled over its members
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
        be.plot_bin_summary(
            bs_all, dim="source_id", title=f"CMIP6 evspsbl, {nsid} models, {kind} edges, {period}", fout=fout,
        )
        print(fout)
        fout = FIG_ROOT / "cmip6" / "qbin" / f"{fstem}.bin_mean.png"
        plot_model_bin_means(bs_all, title=f"CMIP6 evspsbl bin mean, {nsid} models, {kind} edges, {period}", fout=fout)
        print(fout)

        # Hatch non-empty bins whose mean is not significantly different from 0
        signif = be.test_significance(bs_all, alpha=SIGNIF_ALPHA, n_min=SIGNIF_N_MIN)
        not_signif = ~signif & bs_all.sel(stats="mean").notnull()
        print(f"{kind}: {int(not_signif.sum())} of {int(bs_all.sel(stats='mean').notnull().sum())} "
              f"non-empty bins not significant at {1 - SIGNIF_ALPHA:.0%}")
        fout = FIG_ROOT / "cmip6" / "qbin" / f"{fstem}.bin_mean_signif.png"
        plot_model_bin_means(
            bs_all, fout=fout, hatch=not_signif,
            title=(f"CMIP6 evspsbl bin mean, {nsid} models, {kind} edges, {period}; hatched: not significantly "
                   f"different from 0 at {1 - SIGNIF_ALPHA:.0%} (t-test, n > {SIGNIF_N_MIN})"),
        )
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
        plot_model_edges(model_edges[v], pooled_edges[v], title=f"CMIP6 {v}, per-model edges, {period}", fout=fout)
        print(fout)


if __name__ == "__main__":
    main()
