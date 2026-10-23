"""
bin_plumber.py
==============
Bin the annual mean ET of the PLUMBER2 flux towers in the LAI-aridity space
of bin_all.py, so the towers can be compared bin by bin with the gridded
families. Needs the obs inputs (prep_obs.py) and the edges and common area
mask (make_edges.py).

- ET: latent heat flux Qle [W/m2] of each tower's flux file. A calendar year
  in TIME_SLICE is complete if every time step is valid, and its annual mean
  is filtered like the gridded ET (step 6: annual ET < MIN_ANNUAL_ET dropped).
  Only each tower's longest run of consecutive complete years is kept, and
  towers whose run is shorter than MIN_CONSECUTIVE_YEARS are left out; each
  year of the run is one annual mean ET sample.
- LAI, P, Rn, AI: the climatologies of the obs products (prep_obs.py, steps
  10-11) at the gridcell nearest to each tower (`.sel(..., method="nearest")`).
  The obs sources only differ in LAI, P and Rn by their (lai, pr, rn) product
  set and their climatology years, so each product set is taken from the obs
  source with the most shared years, along the dim `lai_pr_rn`. Product sets
  with a product in EXCLUDE_PRODUCTS are left out.

- ET/P (BIN_VARIABLE = "et_pr" only): each year's tower ET over that year's P
  at the nearest gridcell, from the pr product of each product set (the P of
  its AI). Annual P is computed here over TIME_SLICE (steps 1-8 of
  prep_obs.py), not taken from the obs inputs, whose years end with the obs
  climatology (2007), so every tower-year keeps its ratio. NaN where P <= 0.

As in bin_all.py, every (year, tower) ET (or ET/P) sample is binned by the tower's
climatological LAI and AI (`binning.bin_stats`), separately for each product
set; the samples of all towers are pooled. The towers are not masked by the
common area mask: `in_common_area` records which towers are in it, and a
tower whose nearest gridcell has no obs data is not binned.

Outputs
-------
<var>    : BIN_VARIABLE, et or et_pr
<edges>  : pooled_all or pooled_obs
<bins>   : lai<N_YBINS>_ai<N_XBINS>, the bin counts, e.g. "lai15_ai15"
<grid>   : grid tag of TARGET_RES, e.g. "1deg"
<period> : YYYYMM-YYYYMM of TIME_SLICE

PROC_ROOT/plumber/inputs/<period>/plumber.inputs.<grid>.<period>.nc
    et (member, year), NaN outside a tower's longest run of consecutive complete years; lai, pr, rn, ai (lai_pr_rn, member).
    `member` is the site name, with coords lat/lon (tower), grid_lat/grid_lon (nearest gridcell),
    n_years, time_period and in_common_area; `lai_pr_rn` has coords obs_source, <v>_product,
    clim_n_years and clim_period; with BIN_VARIABLE = "et_pr", also et_pr (lai_pr_rn, member, year).
    Also the input of bin_plumber_clim.py
PROC_ROOT/plumber/qbin/<period>/<bins>/plumber.<var>.qbin_<edges>.<bins>.<grid>.<period>.nc
    (lai_pr_rn, stats, y_bin, x_bin), the samples of all towers pooled
FIG_ROOT/plumber/qbin/<period>/<bins>/plumber.<var>.qbin_<edges>.<bins>.<grid>.<period>.{summary,bin_mean,bin_mean_signif}.png
    as in bin_all.py, one panel per product set; bin_mean_signif hatches bins whose mean is not
    significantly different from 0

Scatter of the towers
---------------------
Each tower's mean ET over its full record (its longest run of consecutive complete years, not only TIME_SLICE) against
the climatological LAI and AI of SCATTER_PRODUCTS at its nearest gridcell. The climatology
covers the years of the longest tower record, so it is computed here from the products
(steps 1-11 of prep_obs.py) rather than taken from the obs inputs (MODIS LAI ends in 2005).

<clim_period> : YYYYMM-YYYYMM of the longest tower record (the folder is <period>, of TIME_SLICE)

PROC_ROOT/plumber/clim/<period>/plumber.tower_clim.<grid>.<clim_period>.nc
    et, n_years (member): mean annual ET and number of years of each tower's longest run of complete years;
    lai, pr, rn, ai (member): climatology of SCATTER_PRODUCTS at the nearest gridcell
FIG_ROOT/plumber/clim/<period>/plumber.tower_scatter.<grid>.<clim_period>.png
    AI (x) vs LAI (y), colored by tower ET, marker area proportional to the record length
"""

from __future__ import annotations

import warnings
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
import etunc.units as units
import etunc.load.era5 as le
import etunc.load.ilamb as il

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS. Will affect performance.", category=UserWarning)

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

FAMILY = "plumber"
PLUMBER_ROOT = cfg.PLUMBER_ROOT
TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES
BIN_VARIABLE = cfg.BIN_VARIABLE              # "et": annual tower ET; "et_pr": annual tower ET/P
N_YBINS = cfg.N_YBINS                      # LAI
N_XBINS = cfg.N_XBINS                      # AI
EDGE_KINDS = ("pooled_all", "pooled_obs")  # edge sets of make_edges.py
SIGNIF_ALPHA = cfg.SIGNIF_ALPHA
SIGNIF_N_MIN = 3

# Minimum consecutive years for valid tower data
MIN_CONSECUTIVE_YEARS = 3

# Products of the obs inputs left out of the binned (lai, pr, rn) product sets, by variable
EXCLUDE_PRODUCTS: dict[str, list[str]] = {}

# Products of the scatter's LAI/AI climatology: ILAMB product names (il.PRODUCTS), or "ERA5" for rn.
# They cover the longest tower record (1992-2012), which MODIS LAI and CERES Rn do not
SCATTER_PRODUCTS = {"lai": "GIMMS_LAI4g", "pr": "GPCPv2.3", "rn": "GEWEX.SRB"}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def tower_annual_et(path: Path, time_slice: slice) -> tuple[xr.DataArray | None, float, float]:
    """
    Annual mean Qle [W/m2] of one flux file over the calendar years in `time_slice`, NaN in years with
    a missing time step (None if the record is outside `time_slice`), and the tower's latitude and longitude.
    """
    ds = xr.open_dataset(path)
    lat, lon = float(ds["latitude"].squeeze()), float(ds["longitude"].squeeze())
    qle = ds["Qle"].squeeze(("y", "x"), drop=True).sel(time=time_slice)
    if qle.sizes["time"] == 0:
        return None, lat, lon
    complete = qle.groupby("time.year").count() == qle["time"].groupby("time.year").count()
    return qle.groupby("time.year").mean().where(complete).assign_attrs(units="W/m2"), lat, lon


def longest_run(years: np.ndarray) -> np.ndarray:
    """Years of the longest run of consecutive integers in the sorted `years` (the earliest run on a tie)."""
    if years.size == 0:
        return years
    # Split wherever the step between successive years is not 1
    runs = np.split(years, np.flatnonzero(np.diff(years) != 1) + 1)
    return max(runs, key=len)


def load_towers(time_slice: slice, min_consecutive: int = MIN_CONSECUTIVE_YEARS) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Annual mean tower ET (member, year) over `time_slice`, NaN outside a tower's longest run of consecutive
    complete years (after the ET outlier filter, step 6), and the towers' latitudes and longitudes (member).
    Towers whose longest run is shorter than `min_consecutive` years are skipped.
    """
    flux_files = sorted((PLUMBER_ROOT / "flux").glob("*.nc"))
    et, lats, lons = [], [], []
    for f in flux_files:
        site = f.stem.split("_")[0]
        ann, lat, lon = tower_annual_et(f, time_slice)
        if ann is None or int(ann.notnull().sum()) == 0:
            print(f"{site}: no complete year in {time_slice}, skipping")
            continue
        et.append(ann.expand_dims(member=[site]))
        lats.append(lat)
        lons.append(lon)
    et = xr.concat(et, dim="member", join="outer").sortby("year")
    et = binning.drop_et_outliers(et, cfg.MIN_ANNUAL_ET, label="PLUMBER2 Qle")
    et = et.assign_attrs(long_name="annual mean latent heat flux (Qle) of the flux tower")
    lat = xr.DataArray(lats, dims="member", coords={"member": et["member"]})
    lon = xr.DataArray(lons, dims="member", coords={"member": et["member"]})

    # Keep only each tower's longest run of consecutive complete years, and only towers whose run is at least
    # `min_consecutive` years. Runs are found on the year values, not positions, since a year with no data
    # at any tower is missing from `year`
    years = et["year"].values
    in_run = xr.DataArray(
        [np.isin(years, longest_run(years[et.sel(member=m).notnull().values])) for m in et["member"].values],
        dims=("member", "year"), coords={"member": et["member"], "year": et["year"]},
    )
    n_consec = in_run.sum("year")
    short = n_consec < min_consecutive
    for m in et["member"].values[short.values]:
        print(f"{m}: longest run of {int(n_consec.sel(member=m))} consecutive complete years "
              f"(< {min_consecutive}), skipping")
    n_out = int((et.notnull() & ~in_run).sel(member=~short).sum())
    print(f"{n_out} complete tower-years outside the longest runs set to NaN")
    with xr.set_options(keep_attrs=True):
        et = et.where(in_run)
    et, lat, lon = (da.sel(member=~short) for da in (et, lat, lon))
    print(f"{et.sizes['member']} of {len(flux_files)} towers have >= {min_consecutive} consecutive complete "
          f"years in {time_slice}")
    return et, lat, lon


def nearest_to_towers(da: xr.Dataset | xr.DataArray, lat: xr.DataArray, lon: xr.DataArray):
    """
    `da` at the gridcell nearest to each tower (vectorized `.sel(method="nearest")` along `member`), with the
    gridcell coords as grid_lat/grid_lon. Raises if a tower is outside the grid (e.g. south of LAT_BNDS),
    where it would snap to the edge of the grid.
    """
    out = da.sel(lat=lat, lon=lon, method="nearest").rename(lat="grid_lat", lon="grid_lon")
    far = (np.abs(out["grid_lat"] - lat) > TARGET_RES) | (np.abs(out["grid_lon"] - lon) > TARGET_RES)
    if far.any():
        raise ValueError(f"Towers outside the grid: {far['member'].values[far.values].tolist()}")
    return out


def obs_annual(v: str, product: str, time_slice: slice) -> xr.DataArray:
    """
    Steps 1-8 of prep_obs.py for one LAI, P or Rn product: annual means on the target grid. `product` is an
    ILAMB product or "ERA5" (Rn only).
    """
    label = f"{v}/{product}"
    print(f"\n=== {label} ===")
    # 1-3. Monthly native field over time_slice; P and Rn in W/m2
    if product == "ERA5":
        raw = le.load_native(["msnswrf", "msnlwrf"], time_slice)
        da = units.net_radiation_era5(raw["msnswrf"], raw["msnlwrf"])
    else:
        da = il.load_native(v, product, time_slice)
        da = da.assign_attrs(units="m2/m2") if v == "lai" else units.flux_to_wm2(da)

    # 4. Annual means over complete years: P and Rn need all 12 months, a missing LAI month counts as 0
    a = temporal.aggregate(da, "year", require_all_months=(v != "lai"), label=label)

    # 5. Native land mask without Greenland/Iceland: ERA5 LANDFRAC > LF_THRESH, or Natural Earth land
    if product == "ERA5":
        landfrac, src_grid = le.load_land_fraction()
        land = rg.landfrac_mask(landfrac, cfg.LF_THRESH)
    else:
        land, src_grid = rg.landne_mask(a), rg.bounded_src_grid(a)
    a = rg.mask_to_land(a, land, label)

    # 7. Conservative regridding onto the target grid; 8. domain without Antarctica
    regridder = rg.bounded_conservative_regridder(src_grid, TARGET_RES)
    a = rg.on_grid(rg.regrid_with_na_thresh(a, regridder, cfg.NA_THRESH), TARGET_RES, label)
    return a.sel(lat=cfg.LAT_BNDS).rename(v)


def plot_tower_scatter(towers: xr.Dataset, title: str = "", fout: Path | None = None):
    """Climatological AI (x) vs LAI (y) of each tower, colored by tower ET, marker area ∝ record length."""
    t = towers.sortby("n_years", ascending=False)  # long records first, so short ones stay visible on top
    size = lambda n: 12 * n
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    sc = ax.scatter(t["ai"], t["lai"], c=t["et"], s=size(t["n_years"]), cmap="YlGnBu",
                    edgecolor="0.3", linewidth=0.5, alpha=0.8)
    fig.colorbar(sc, ax=ax, label=f"tower ET, mean over record [{towers['et'].attrs.get('units', '?')}]")
    for n in (1, 5, 10, 20):
        ax.scatter([], [], s=size(n), c="none", edgecolor="0.3", label=f"{n} yr")
    ax.legend(title="Record length", loc="upper right", frameon=False)
    ax.set_xlabel(f"climatological aridity index Rn/LP ({towers.attrs['pr_product']}, {towers.attrs['rn_product']})")
    ax.set_ylabel(f"climatological LAI [m2/m2] ({towers.attrs['lai_product']})")
    ax.set_title(title)
    return plotting.finish(fig, fout)


def product_set_sources(obs: xr.Dataset, exclude: dict[str, list[str]]) -> dict[str, str]:
    """
    {"<lai>-<pr>-<rn>": obs source}: for each product set, the obs source with the most shared years.
    Sources using a product in `exclude` ({variable: [products]}) are skipped.
    """
    best = {}
    for m in obs["member"].values:
        products = {v: str(obs[f"{v}_product"].sel(member=m).values) for v in ("lai", "pr", "rn")}
        if any(products[v] in exclude.get(v, ()) for v in products):
            continue
        key = "-".join(products.values())
        if key not in best or obs["n_years"].sel(member=m) > obs["n_years"].sel(member=best[key]):
            best[key] = m
    return best


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.summarize_settings(globals())
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)
    bins = cfg.bins_tag(N_YBINS, N_XBINS)  # bin counts, in the folder and file names

    obs_path = cfg.inputs_path("obs", grid, period)
    area_path = cfg.area_mask_path(grid, period)
    for path, script in ((obs_path, "prep_obs.py"), (area_path, "make_edges.py")):
        if not path.exists():
            raise FileNotFoundError(f"No file at {path}; run scripts/{script} first")

    # ------------------------------------------------------------------
    # Tower ET: annual means of the complete years in TIME_SLICE
    # ------------------------------------------------------------------
    print("\n=== PLUMBER2 towers ===")
    et, site_lat, site_lon = load_towers(TIME_SLICE)
    n_years = et.notnull().sum("year")
    time_period = [temporal.period_str(et["year"].values[et.sel(member=m).notnull().values]) for m in et["member"].values]

    # ------------------------------------------------------------------
    # Obs LAI, P, Rn and AI at the gridcell nearest to each tower
    # ------------------------------------------------------------------
    obs = xr.open_dataset(obs_path)
    sets = product_set_sources(obs, EXCLUDE_PRODUCTS)
    print(f"\n=== {len(sets)} (lai, pr, rn) product sets ===")
    for key, m in sets.items():
        print(f"{key}: {m}, {int(obs['n_years'].sel(member=m))} years ({obs['time_period'].sel(member=m).values})")
    clim = (
        obs[["lai", "pr", "rn", "ai"]].sel(member=list(sets.values()))
        .drop_vars(["source", "et_product"])
        .rename(member="obs_source", n_years="clim_n_years", time_period="clim_period")
        .swap_dims(obs_source="lai_pr_rn")
        .assign_coords(lai_pr_rn=list(sets))
    )

    at_towers = nearest_to_towers(clim, site_lat, site_lon)
    area = xr.open_dataset(area_path)["area_mask"] == 1
    in_area = nearest_to_towers(area, site_lat, site_lon).reset_coords(drop=True)
    no_obs = at_towers["ai"].isnull() | at_towers["lai"].isnull()
    print(f"{int(in_area.sum())} of {in_area.size} towers in the common area mask")
    for key in sets:
        missing = no_obs.sel(lai_pr_rn=key)
        print(f"{key}: {int(missing.sum())} towers without obs LAI or AI at the nearest gridcell, not binned: "
              f"{missing['member'].values[missing.values].tolist()}")

    # ------------------------------------------------------------------
    # Annual evaporation ratio ET/P of each tower (BIN_VARIABLE = "et_pr" only)
    # ------------------------------------------------------------------
    # Tower ET over the annual P at the nearest gridcell, from each product set's pr product (the P of its AI).
    # P is computed over TIME_SLICE (steps 1-8 of prep_obs.py), not only over the obs climatology years
    ratio = {}
    if BIN_VARIABLE == "et_pr":
        pr_products = clim["pr_product"].values.tolist()
        pr_at_towers = {pr: nearest_to_towers(obs_annual("pr", pr, TIME_SLICE), site_lat, site_lon)
                        for pr in dict.fromkeys(pr_products)}
        # Annual P of each product set on the tower years; NaN in a tower year without P
        pr_ann = xr.concat(
            [pr_at_towers[pr].reindex(year=et["year"].values) for pr in pr_products], dim="lai_pr_rn",
        ).assign_coords(lai_pr_rn=list(sets))
        ratio["et_pr"] = binning.evaporation_ratio(et, pr_ann, label="PLUMBER2 Qle/P").transpose("lai_pr_rn", "member", "year")

    inputs = xr.Dataset({"et": et, **at_towers.data_vars, **ratio}).assign_coords(
        lat=site_lat, lon=site_lon, n_years=n_years, time_period=("member", time_period),
        in_common_area=in_area,
    )
    inputs.attrs = {
        "family": FAMILY, "dataset": "PLUMBER2 flux towers: tower ET, obs LAI/P/Rn/AI at the nearest gridcell",
        "time_period": temporal.period_str(inputs["year"].values), "grid": grid,
        "plumber_root": str(PLUMBER_ROOT), "obs_inputs": str(obs_path), "min_annual_et": cfg.MIN_ANNUAL_ET,
        "min_consecutive_years": MIN_CONSECUTIVE_YEARS,  # read by bin_plumber_clim.py
    }
    fout = cfg.inputs_path(FAMILY, grid, period)
    fout.parent.mkdir(parents=True, exist_ok=True)
    inputs.to_netcdf(fout)
    print(fout)

    # ------------------------------------------------------------------
    # 14. Every (year, tower) ET (or ET/P) sample binned by the tower's climatological LAI and AI, per product set
    # ------------------------------------------------------------------
    qbin_dir = cfg.output_dir(cfg.PROC_ROOT, FAMILY, "qbin", period, bins)
    fig_dir = cfg.output_dir(cfg.FIG_ROOT, FAMILY, "qbin", period, bins)
    for kind in EDGE_KINDS:
        print(f"\n=== {kind} edges ===")
        y_edges = edges.load_bin_edges(kind, "lai", N_YBINS, N_XBINS, grid, period)
        x_edges = edges.load_bin_edges(kind, "ai", N_YBINS, N_XBINS, grid, period)
        bs = binning.bin_stats(
            inputs[BIN_VARIABLE], inputs["lai"], inputs["ai"], y_edges, x_edges,
            member_dim="lai_pr_rn", y_name="lai", x_name="ai", name=BIN_VARIABLE,
            attrs={
                "family": FAMILY, "edges": kind, "time_period": inputs.attrs["time_period"], "grid": grid,
                "n_towers": inputs.sizes["member"], "obs_inputs": str(obs_path),
            },
        )
        # bin_stats keeps only the coords of the binned variable along the member dim, so add those of the product sets
        bs = bs.assign_coords({c: inputs[c] for c in inputs.coords if inputs[c].dims == ("lai_pr_rn",)})
        stem = f"{FAMILY}.{BIN_VARIABLE}.qbin_{kind}.{bins}.{grid}.{period}"
        fout = qbin_dir / f"{stem}.nc"
        fout.parent.mkdir(parents=True, exist_ok=True)
        bs.to_netcdf(fout)
        count = bs.sel(stats="count").sum(("y_bin", "x_bin"))
        for key in sets:
            n_bins = int((bs.sel(lai_pr_rn=key, stats="count") > 0).sum())
            print(f"{key}: {int(count.sel(lai_pr_rn=key))} tower-years binned in {n_bins} bins")
        print(fout)

        # Figures as in bin_all.py, one panel per product set; every set shares the edges, so the
        # zero-width bins can be dropped
        title = f"PLUMBER2 tower {binning.TARGET_LABELS[BIN_VARIABLE]}, {len(sets)} (lai, pr, rn) product sets, {kind} edges, {inputs.attrs['time_period']}"
        bs = binning.drop_zero_width_bins(bs)
        plotting.plot_bin_summary(bs, dim="lai_pr_rn", title=title, fout=fig_dir / f"{stem}.summary.png")
        plotting.plot_bin_means(bs, "lai_pr_rn", title=f"{title}, bin mean", fout=fig_dir / f"{stem}.bin_mean.png")
        # Hatch non-empty bins whose mean is not significantly different from 0
        signif = binning.test_significance(bs, alpha=SIGNIF_ALPHA, n_min=SIGNIF_N_MIN)
        not_signif = ~signif & bs.sel(stats="mean").notnull()
        plotting.plot_bin_means(
            bs, "lai_pr_rn", hatch=not_signif, fout=fig_dir / f"{stem}.bin_mean_signif.png",
            title=(f"{title}; hatched: not significantly different from 0 at {1 - SIGNIF_ALPHA:.0%} "
                   f"(t-test, n > {SIGNIF_N_MIN})"),
        )
        for suffix in ("summary", "bin_mean", "bin_mean_signif"):
            print(fig_dir / f"{stem}.{suffix}.png")

    # ------------------------------------------------------------------
    # Scatter: tower ET over its full record vs climatological obs LAI and AI over the longest tower record
    # ------------------------------------------------------------------
    print("\n=== PLUMBER2 towers, full records ===")
    et_rec, rec_lat, rec_lon = load_towers(slice(None, None))
    # Longest record: first to last complete year of a tower
    valid_years = et_rec["year"].where(et_rec.notnull())
    first, last = valid_years.min("year"), valid_years.max("year")
    longest = (last - first).argmax("member")
    clim_years = list(range(int(first[longest]), int(last[longest]) + 1))
    clim_slice = slice(f"{clim_years[0]}-01", f"{clim_years[-1]}-12")
    clim_period = temporal.period_str(clim_years)
    print(f"Longest record: {et_rec['member'].values[int(longest)]}, {clim_years[0]}-{clim_years[-1]}")

    # Steps 1-11 of prep_obs.py for SCATTER_PRODUCTS over the years of the longest record
    ann = {v: obs_annual(v, p, clim_slice) for v, p in SCATTER_PRODUCTS.items()}
    years = [int(y) for y in temporal.shared_years(*ann.values())]
    if years != clim_years:
        print(f"Warning: the products share {len(years)} of the {len(clim_years)} years of the longest record")
    clim = {v: temporal.aggregate(a.sel(year=years), "clim", label=f"{v}/{SCATTER_PRODUCTS[v]}") for v, a in ann.items()}
    clim["ai"] = edges.compute_aridity_index(clim["pr"], clim["rn"])
    clim = nearest_to_towers(xr.Dataset(clim), rec_lat, rec_lon)

    towers = xr.Dataset({
        "et": et_rec.mean("year").assign_attrs(units="W/m2", long_name="mean annual tower ET over its record"),
        "n_years": et_rec.notnull().sum("year").assign_attrs(long_name="complete years in the tower record"),
        **clim.data_vars,
    }).assign_coords(lat=rec_lat, lon=rec_lon)
    towers.attrs = {
        "dataset": "PLUMBER2 flux towers: record-mean tower ET, obs climatology at the nearest gridcell",
        "clim_period": temporal.period_str(years), "grid": grid,
        **{f"{v}_product": p for v, p in SCATTER_PRODUCTS.items()},
    }
    no_obs = towers["lai"].isnull() | towers["ai"].isnull()
    print(f"{int(no_obs.sum())} towers without LAI or AI at the nearest gridcell, not plotted: "
          f"{towers['member'].values[no_obs.values].tolist()}")
    fout = cfg.output_dir(cfg.PROC_ROOT, FAMILY, "clim", period) / f"{FAMILY}.tower_clim.{grid}.{clim_period}.nc"
    fout.parent.mkdir(parents=True, exist_ok=True)
    towers.to_netcdf(fout)
    print(fout)

    fout = cfg.output_dir(cfg.FIG_ROOT, FAMILY, "clim", period) / f"{FAMILY}.tower_scatter.{grid}.{clim_period}.png"
    plot_tower_scatter(
        towers.where(~no_obs, drop=True),
        title=f"PLUMBER2 towers (n = {int((~no_obs).sum())}), LAI and AI climatology {towers.attrs['clim_period']}",
        fout=fout,
    )
    print(fout)


if __name__ == "__main__":
    main()
