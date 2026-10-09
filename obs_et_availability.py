"""
obs_et_availability.py
======================
Compare the data availability of every observational ET product (ILAMB
products, PML-V2.2, GLEAM v4.3, and SiTHv2 once it is regridded) over one
period, with the masking and processing of `ilamb_binned_et.py`.

Processing
----------
- ILAMB products are read with `il.load_ilamb`: lon in
  [-180, 180], lat ascending, undecoded fill values removed, converted to W/m2,
  and 1 deg products (WECANN) bilinearly interpolated to the 0.5 deg grid.
- PML (ET = Ec + Es + Ei) and GLEAM (E) are read from the 0.5 deg files written
  by `regrid_obs.py` (conservative, `NA_THRES`) and converted from mm/month
  to W/m2. PML V2.2a-VIIRS is skipped (its monthly E 2019 file is empty).
- Every product is put on one monthly time axis over `TIME_SLICE`, so months
  outside a product's record count as missing, and restricted to `config.LAT_BNDS`
  and the land mask (Natural Earth, without Greenland/Iceland).
- A gridcell-year is valid when all 12 months are valid (`temporal.annual_mean` with
  `require_all_months=True`), as for ET in the binning. The bin mask of a
  product is the land gridcells with at least one valid gridcell-year (the ET
  part of `binning.valid_area`; LAI and AI are not used).

Counts are numbers of 0.5 deg gridcells, not areas.

Outputs
-------
<period> : YYYYMM-YYYYMM of TIME_SLICE

PROC_ROOT/obs.et.availability.<period>.nc       per-product counts and bin masks
PROC_ROOT/obs.et.availability.<period>.csv      summary table
FIG_ROOT/obs/availability/obs.et.<diag>.<period>.png, with <diag>:
    hist_valid_months       land gridcells binned by their number of valid months
    map_valid_months        number of valid months at each gridcell
    line_calendar_month     valid gridcell-years for each calendar month
    map_bin_mask            bin mask of each product
    heatmap_complete_years  fraction of land gridcells with a valid year, per year
    map_mask_agreement      number of products whose bin mask includes each gridcell
    zonal_mean_et           zonal mean of climatological ET, own and common mask
    box_annual_et           distribution of annual mean ET in the common mask
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr

import etunc.config as config
import etunc.units as units
import etunc.temporal as temporal
import etunc.grid as rg
import etunc.plotting as plotting
import ilamb_binned_et as ib
import etunc.load.ilamb as il
import etunc.load.obs as lo
import regrid_obs as ro


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

TIME_SLICE = slice("2000-01", "2014-12")

ILAMB_PRODUCTS = list(il.PRODUCTS["et"])  # every ILAMB ET product

# {label: (load_obs dataset, version, variable)}, read from the 0.5 deg regridded files.
# Products without regridded files (e.g. SiTHv2 before regrid_obs.py has run) are skipped.
GRIDDED_PRODUCTS = {
    "PML-V2.2a-MODIS": ("pml", "V2.2a-MODIS", "ET"),
    # "PML-V2.2b":       ("pml", "V2.2b", "ET"),
    "PML-V2.2c":       ("pml", "V2.2c", "ET"),
    "GLEAM-v4.3a":     ("gleam", "v4.3a", "E"),
    "GLEAM-v4.3b":     ("gleam", "v4.3b", "E"),
    "SiTHv2":          ("sith", "v2", "ET"),
}
REGRID_TAG = "0.5deg"

PROC_ROOT = ib.PROC_ROOT
FIG_DIR = ib.FIG_ROOT / "obs" / "availability"

NCOLS = 4  # panels per row in the per-product figures
BOX_WHIS = (5, 95)  # whisker percentiles in box_annual_et


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

MONTHS = pd.date_range(TIME_SLICE.start, TIME_SLICE.stop, freq="MS")
GRID = ib.TARGET_GRID.sel(lat=config.LAT_BNDS)


def load_ilamb(product: str) -> xr.DataArray:
    """Monthly ET [W/m2] of an ILAMB product (converted before regridding) on the MONTHS axis."""
    return temporal.on_month_axis(il.load_ilamb("et", product, TIME_SLICE, ib.TARGET_RES, wm2=True), MONTHS)


def regridded_dataset(dataset: str) -> lo.ObsDataset:
    """`load_obs` spec of the REGRID_TAG files written by `regrid_obs.py` for `dataset`."""
    spec = lo.get_dataset(dataset)
    return lo.register_dataset(dataclasses.replace(
        spec, name=f"{spec.name}-{REGRID_TAG}", root=ro.REGRID_ROOT / spec.root.name / REGRID_TAG,
        lat_name="lat", lon_name="lon",
    ))


def load_gridded(label: str) -> xr.DataArray:
    """Monthly ET [W/m2] of a PML/GLEAM/SiTH product from its regridded files."""
    dataset, version, var = GRIDDED_PRODUCTS[label]
    spec = regridded_dataset(dataset)
    if not lo.list_years(spec, var, version, "monthly"):
        raise FileNotFoundError(f"{label}: no {REGRID_TAG} files under {spec.root} (run regrid_obs.py)")
    da = lo.load_obs(spec, var, TIME_SLICE, version=version, freq="monthly").load()
    da = units.latent_heat_to_wm2(units.accumulation_to_flux(da))
    return temporal.on_month_axis(rg.on_grid(da, ib.TARGET_RES, label), MONTHS)


# ------------------------------------------------------------------
# Availability
# ------------------------------------------------------------------

def availability(et: xr.DataArray, land: xr.DataArray) -> tuple[xr.Dataset, xr.DataArray]:
    """
    Availability counts of monthly ET on land, and the annual mean ET
    (NaN for gridcell-years without all 12 months).
    """
    et = et.where(land)
    valid = et.notnull()
    ann = temporal.annual_mean(et, require_all_months=True)
    complete = ann.notnull()
    ds = xr.Dataset({
        "n_valid_months": valid.sum("time").where(land),
        "n_valid_years": complete.sum("year").where(land),
        "bin_mask": complete.any("year"),
        "valid_by_calendar_month": valid.groupby("time.month").sum().sum(("lat", "lon")),
        "valid_by_year": complete.sum(("lat", "lon")),
    })
    return ds, ann


def summary_table(av: xr.Dataset, n_land: int) -> pd.DataFrame:
    n_years = av.sizes["year"]
    rows = []
    for p in av["product"].values:
        a = av.sel(product=p)
        years = a["year"].values[(a["valid_by_year"] > 0).values]
        rows.append({
            "product": p,
            "first_year": int(years.min()) if years.size else None,
            "last_year": int(years.max()) if years.size else None,
            "bin_mask_cells": int(a["bin_mask"].sum()),
            "bin_mask_pct_land": 100 * float(a["bin_mask"].sum()) / n_land,
            "valid_gridcell_years": int(a["valid_by_year"].sum()),
            "valid_gridcell_years_pct_max": 100 * float(a["valid_by_year"].sum()) / (n_land * n_years),
            "mean_valid_months": float(a["n_valid_months"].mean()),
        })
    return pd.DataFrame(rows).set_index("product")


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.92", lw=0.6)
    ax.set_axisbelow(True)


def plot_hist_valid_months(av: xr.Dataset, title: str, fout: Path):
    n_months = len(MONTHS)
    bins = np.arange(-0.5, n_months + 1.5, 1)
    fig, axs = plotting.facets(av.sizes["product"], ncols=NCOLS, sharex=True, sharey=True)
    for ax, p in zip(axs, av["product"].values):
        n = av["n_valid_months"].sel(product=p).values.ravel()
        n = n[np.isfinite(n)]
        ax.hist(n, bins=bins, color="C0")
        ax.set_yscale("log")
        ax.set_title(f"{p}\n{int((n == n_months).sum())} of {n.size} land cells with all {n_months} months",
                     fontsize=9)
        ax.set_xticks(np.arange(0, n_months + 1, 12 * max(1, n_months // 120)))
        _clean(ax)
    for ax in axs:
        ax.set_xlabel("valid months")
        ax.set_ylabel("land gridcells")
        ax.label_outer()
    fig.suptitle(title)
    return plotting.finish(fig, fout)


def plot_map_valid_months(av: xr.Dataset, title: str, fout: Path):
    n_months = len(MONTHS)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_under("0.55")
    fig, axs = plotting.facets(av.sizes["product"], ncols=NCOLS, maps=True, panel_size=(4.6, 2.3))
    for ax, p in zip(axs, av["product"].values):
        pm = av["n_valid_months"].sel(product=p).plot.pcolormesh(
            ax=ax, transform=config.PROJECTION, cmap=cmap, vmin=0.5, vmax=n_months, add_colorbar=False,
        )
        plotting.map_ax(ax, config.LAT_BNDS)
        ax.set_title(p, fontsize=9)
    fig.colorbar(pm, ax=axs, shrink=0.6, extend="min",
                 label=f"valid months of {n_months} (gray: land, none valid)")
    fig.suptitle(title)
    return plotting.finish(fig, fout)


def plot_line_calendar_month(av: xr.Dataset, n_land: int, title: str, fout: Path):
    n_years = av.sizes["year"]
    fig, axs = plotting.facets(av.sizes["product"], ncols=NCOLS, sharex=True, sharey=True, panel_size=(4.2, 2.8))
    for ax, p in zip(axs, av["product"].values):
        a = av.sel(product=p)
        ax.plot(a["month"], a["valid_by_calendar_month"], "-o", color="C0", lw=2, ms=4,
                label="gridcell-years with the month valid")
        ax.axhline(int(a["valid_by_year"].sum()), color="C1", lw=1.5, ls="--",
                   label="gridcell-years with all 12 months valid")
        ax.axhline(n_land * n_years, color="0.5", lw=1, ls=":", label="land gridcells x years")
        ax.set_title(p, fontsize=9)
        ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
        ax.set_ylim(bottom=0)
        _clean(ax)
    for ax in axs:
        ax.set_xlabel("calendar month")
        ax.set_ylabel("gridcell-years")
        ax.label_outer()
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(title)
    return plotting.finish(fig, fout)


MASK_COLORS = {"land, no valid year": "#f0b67f", "bin mask": "#2b6a99"}


def plot_map_bin_mask(av: xr.Dataset, land: xr.DataArray, title: str, fout: Path):
    cmap = mcolors.ListedColormap(list(MASK_COLORS.values()))
    fig, axs = plotting.facets(av.sizes["product"], ncols=NCOLS, maps=True, panel_size=(4.6, 2.3))
    for ax, p in zip(axs, av["product"].values):
        m = av["bin_mask"].sel(product=p)
        m.astype(float).where(land).plot.pcolormesh(
            ax=ax, transform=config.PROJECTION, cmap=cmap, vmin=0, vmax=1, add_colorbar=False,
        )
        plotting.map_ax(ax, config.LAT_BNDS)
        ax.set_title(f"{p}: {int(m.sum())} cells", fontsize=9)
    fig.legend(handles=[Patch(color=c, label=k) for k, c in MASK_COLORS.items()],
               loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle(title)
    return plotting.finish(fig, fout)


def plot_heatmap_complete_years(av: xr.Dataset, n_land: int, title: str, fout: Path):
    frac = av["valid_by_year"] / n_land
    fig, ax = plt.subplots(figsize=(1.0 + 0.45 * av.sizes["year"], 0.9 + 0.38 * av.sizes["product"]),
                           layout="constrained")
    pm = ax.pcolormesh(np.arange(av.sizes["year"] + 1), np.arange(av.sizes["product"] + 1),
                       frac.transpose("product", "year").values, cmap="Blues", vmin=0, vmax=1,
                       edgecolors="white", linewidth=1)
    ax.set_xticks(np.arange(av.sizes["year"]) + 0.5, av["year"].values, rotation=90)
    ax.set_yticks(np.arange(av.sizes["product"]) + 0.5, av["product"].values)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(pm, ax=ax, label="fraction of land gridcells with all 12 months valid")
    ax.set_title(title)
    return plotting.finish(fig, fout)


FAMILIES = {
    "ILAMB": lambda p: p in ILAMB_PRODUCTS,
    "PML": lambda p: p.startswith("PML"),
    "GLEAM v4.3 / SiTHv2": lambda p: p.startswith(("GLEAM-", "SiTH")),
}


def plot_zonal_mean_et(clim: xr.DataArray, av: xr.Dataset, common: xr.DataArray, title: str, fout: Path):
    """Zonal mean of climatological ET per product, on its own bin mask (top) and the common mask (bottom)."""
    products = list(clim["product"].values)
    fams = {f: [p for p in products if is_f(p)] for f, is_f in FAMILIES.items()}
    fams = {f: ps for f, ps in fams.items() if ps}
    masks = {"own bin mask": av["bin_mask"], "common mask": common}
    fig, axs = plt.subplots(len(masks), len(fams), figsize=(4.6 * len(fams), 3.6 * len(masks)),
                            sharex=True, sharey=True, squeeze=False, layout="constrained")
    for i, (mname, mask) in enumerate(masks.items()):
        zm = clim.where(mask).mean("lon")
        median = zm.median("product")
        for j, (fam, ps) in enumerate(fams.items()):
            ax = axs[i, j]
            ax.plot(median, zm["lat"], color="k", lw=1.2, ls="--", label="median of all products")
            for k, p in enumerate(ps):
                ax.plot(zm.sel(product=p), zm["lat"], color=f"C{k}", lw=1.6, label=p)
            ax.set_title(f"{fam}, {mname}", fontsize=10)
            _clean(ax)
            if i == 0:
                ax.legend(fontsize=7, frameon=False, loc="upper right")
    for ax in axs[-1]:
        ax.set_xlabel(f"zonal mean ET [{clim.attrs.get('units', 'W/m2')}]")
    for ax in axs[:, 0]:
        ax.set_ylabel("latitude")
    fig.suptitle(title)
    return plotting.finish(fig, fout)


def plot_box_annual_et(ann: xr.DataArray, common: xr.DataArray, title: str, fout: Path):
    """Annual mean ET of every valid gridcell-year inside the common mask, per product."""
    products = list(ann["product"].values)
    data = []
    for p in products:
        v = ann.sel(product=p).where(common).values.ravel()
        data.append(v[np.isfinite(v)])
    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.42 * len(products)), layout="constrained")
    ax.boxplot(data, vert=False, whis=BOX_WHIS, showfliers=False, widths=0.6,
               medianprops={"color": "C1", "lw": 2}, boxprops={"color": "C0"},
               whiskerprops={"color": "C0"}, capprops={"color": "C0"})
    ax.set_yticks(np.arange(1, len(products) + 1), [f"{p} (n={d.size})" for p, d in zip(products, data)])
    ax.invert_yaxis()
    ax.set_xlabel(f"annual mean ET [{ann.attrs.get('units', 'W/m2')}]; whiskers at "
                  f"{BOX_WHIS[0]}th/{BOX_WHIS[1]}th percentiles")
    _clean(ax)
    ax.set_title(title)
    return plotting.finish(fig, fout)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    period = temporal.format_time_period(TIME_SLICE)
    land = (rg.land_mask(ib.TARGET_GRID).sel(lat=config.LAT_BNDS) == 1).rename("land")
    n_land = int(land.sum())
    print(f"{period}: {len(MONTHS)} months, {n_land} land gridcells\n")

    print("=== Load products ===")
    loaders = {**{p: load_ilamb for p in ILAMB_PRODUCTS}, **{p: load_gridded for p in GRIDDED_PRODUCTS}}
    avs, anns = {}, {}
    for label, load in loaders.items():
        try:
            et = load(label)
        except FileNotFoundError as err:
            print(f"{label:16}: skipped ({err})")
            continue
        avs[label], anns[label] = availability(et, land)
        print(f"{label:16}: {int(avs[label]['bin_mask'].sum())} bin-mask cells, "
              f"{int(avs[label]['valid_by_year'].sum())} valid gridcell-years")
        del et

    products = np.array(list(avs))  # numpy str, not pandas' StringDtype, which netCDF cannot write
    av = xr.concat(list(avs.values()), dim="product").assign_coords(product=products)
    av = av.assign_attrs(time_period=period, n_land_cells=n_land, lat_bnds=str(config.LAT_BNDS))
    ann = xr.concat(list(anns.values()), dim="product").assign_coords(product=products).assign_attrs(units="W/m2")
    clim = ann.mean("year", keep_attrs=True)
    common = av["bin_mask"].all("product")
    print(f"\ncommon mask: {int(common.sum())} of {n_land} land gridcells")

    PROC_ROOT.mkdir(parents=True, exist_ok=True)
    fout = PROC_ROOT / f"obs.et.availability.{period}.nc"
    av.assign(bin_mask=av["bin_mask"].astype("i1"), common_mask=common.astype("i1")).to_netcdf(fout)
    print(fout)
    table = summary_table(av, n_land)
    fout = PROC_ROOT / f"obs.et.availability.{period}.csv"
    table.to_csv(fout, float_format="%.2f")
    print(fout)
    print(table.round(1).to_string(), "\n")

    print("=== Figures ===")
    span = f"{TIME_SLICE.start} to {TIME_SLICE.stop}"
    figs = {
        "hist_valid_months": lambda f: plot_hist_valid_months(
            av, f"Land gridcells by number of valid ET months, {span}", f),
        "map_valid_months": lambda f: plot_map_valid_months(
            av, f"Valid ET months, {span}", f),
        "line_calendar_month": lambda f: plot_line_calendar_month(
            av, n_land, f"Valid ET gridcell-years by calendar month, {span}", f),
        "map_bin_mask": lambda f: plot_map_bin_mask(
            av, land, f"Bin mask: land gridcells with at least one year of 12 valid ET months, {span}", f),
        "heatmap_complete_years": lambda f: plot_heatmap_complete_years(
            av, n_land, f"Land gridcells with all 12 ET months valid, {span}", f),
        "map_mask_agreement": lambda f: plotting.plot_mask_agreement(
            av["bin_mask"], land, f"Agreement of ET bin masks, {span}", f,
            label="products with the gridcell in their bin mask"),
        "zonal_mean_et": lambda f: plot_zonal_mean_et(
            clim, av, common, f"Zonal mean of climatological ET (mean of valid years), {span}", f),
        "box_annual_et": lambda f: plot_box_annual_et(
            ann, common, f"Annual mean ET of valid gridcell-years in the common mask, {span}", f),
    }
    for diag, make in figs.items():
        fout = FIG_DIR / f"obs.et.{diag}.{period}.png"
        make(fout)
        print(fout)


if __name__ == "__main__":
    main()
