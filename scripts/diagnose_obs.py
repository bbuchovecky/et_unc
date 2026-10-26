"""
diagnose_obs.py
===============
Spatial and temporal coverage of every observational product of ET, LAI, P
and Rn, with the same diagnostics for each variable (formerly in
notebooks/check-et-obs.ipynb and notebooks/check-clara-a3.ipynb).

Products
--------
- ILAMB products in ILAMB_PRODUCTS (`il.load_native`), every variable.
- Gridded products in GRIDDED_PRODUCTS (`lo.load_obs`, native monthly files):
  ET of PML-V2.2, GLEAM v4.3 and SiTHv2, and Rn of CLARA-A3.
A product without files is skipped.

Processing
----------
Each product is read once over RECORD_SLICE (its full record by default) on
its native grid within LAT_BNDS, in W/m2 (LAI in m2/m2), and reduced as it is
read, so a 0.1 deg record is never held in memory:
- land is Natural Earth land by cell center without Greenland/Iceland
  (`rg.landne_mask`, step 5 of prep_obs.py), and each native cell weighs its
  area (`rg.cell_area`);
- a month is valid at a land cell if the product has a value there. Over
  TIME_SLICE, a year is valid under the rule of step 4: 12 valid months for
  ET, P and Rn, at least one for LAI (REQUIRE_ALL_MONTHS). Only the complete
  calendar years of the product's record inside TIME_SLICE count;
- on the target grid, the annual means over TIME_SLICE are those of steps
  4-8 of prep_obs.py (annual means, land mask, ET outlier filter for ET,
  conservative regridding with NA_THRESH, LAT_BNDS), and the climatology that
  of step 10 (mean of the valid years). Target land area is fractional
  Natural Earth land (`rg.land_area`), without Greenland/Iceland.

Outputs
-------
<var>    : et, lai, pr, rn
<period> : YYYYMM-YYYYMM of TIME_SLICE
<grid>   : grid tag of TARGET_RES, e.g. "1deg"

PROC_ROOT/obs/coverage/<period>/obs.<var>.coverage.<period>.csv
    per product: record, valid years, land area covered on the native and target grids (global and by
    region), and the QUANTILES of the valid monthly values
FIG_ROOT/obs/coverage/<period>/obs.<var>.<diag>.<period>.png, with <diag>:
    1. Land covered on the native grids and on the target grid
       land_area              land area with any valid month, a valid year (native) and a climatology (target)
       map_n_products         number of products with a climatology at each target gridcell
    2. Coverage maps on the native grids, over TIME_SLICE
       map_valid_months       fraction of months valid
       map_valid_years        fraction of years valid (rule of step 4)
       map_seasonal_gaps      calendar months valid in fewer than SEASONAL_GAP_FRAC of the years
    3. Valid fraction of the land area of each latitude row
       zonal_time             by month, over RECORD_SLICE
       zonal_calendar_month   by calendar month, mean over the years in TIME_SLICE
    4. Global and regional (REGIONS) timeseries, over RECORD_SLICE
       ts_valid_land          valid fraction of the region's land area
       ts_mean                area-weighted mean over the valid land: monthly, and annual (12 valid months)
    5. Climatologies on the target grid (<grid>), over TIME_SLICE
       map_clim.<grid>        climatology of each product
       map_diff.<grid>        every pair of products: mean difference over their shared valid years
    6. hist                   land-area weighted histogram of the valid monthly values in TIME_SLICE, with
                              vertical lines at QUANTILES
"""

from __future__ import annotations

import warnings

import dask
import dask.array
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr

import etunc.config as cfg
import etunc.binning as binning
import etunc.grid as rg
import etunc.plotting as plotting
import etunc.temporal as temporal
import etunc.units as units
import etunc.load.ilamb as il
import etunc.load.obs as lo

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS. Will affect performance.", category=UserWarning)
# Means over years or regions without any valid value are NaN, as intended
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

FAMILY = "obs"

VARIABLES = ("et", "lai", "pr", "rn")
TIME_SLICE = cfg.TIME_SLICE         # coverage statistics, climatologies and histograms (the pipeline's period)
RECORD_SLICE = slice(None, None)    # zonal and regional timeseries: each product's full record
TARGET_RES = cfg.TARGET_RES

# ILAMB products of each variable (keys of il.PRODUCTS[variable])
ILAMB_PRODUCTS = {v: list(il.PRODUCTS[v]) for v in VARIABLES}

# Gridded products, read from their native monthly files: {variable: {label: (load_obs dataset, version,
# variable)}}. ET as GRIDDED_ET_PRODUCTS of prep_obs.py (monthly totals); CLARA-A3 holds monthly mean W/m2
GRIDDED_PRODUCTS = {
    "et": {
        "PMLv2.2a_MODIS": ("pml", "V2.2a-MODIS", "ET"),
        "PMLv2.2c":       ("pml", "V2.2c", "ET"),
        "GLEAMv4.3a":     ("gleam", "v4.3a", "E"),
        "GLEAMv4.3b":     ("gleam", "v4.3b", "E"),
        "SiTHv2":         ("sith", "v2", "ET"),
    },
    "rn": {
        "CLARA-A3": ("clara", "A3", "SRB"),
    },
}

# Valid year (step 4 of prep_obs.py): all 12 months valid (ET, P, Rn), or any valid month (LAI, whose
# missing months count as 0)
REQUIRE_ALL_MONTHS = {"et": True, "lai": False, "pr": True, "rn": True}

# Latitude bands of the regional timeseries and the coverage table: high-latitude (winter) gaps apart
# from the tropics
REGIONS = {
    "Global":                    cfg.LAT_BNDS,
    "NH high (50N-90N)":         slice(50, 90),
    "NH mid (23N-50N)":          slice(23, 50),
    "Tropics (23S-23N)":         slice(-23, 23),
    "SH extratropics (58S-23S)": slice(-58, -23),
}

# A calendar month is a seasonal gap at a gridcell if it is valid in fewer than this fraction of the years
SEASONAL_GAP_FRAC = 0.5

# Histograms of the valid monthly values: fixed bins (first edge, last edge, number of bins; values outside
# are counted in the end bins), and the quantiles drawn and tabulated (interpolated within the bins)
HIST_BINS = {
    "et":  (-50, 250, 300),   # W/m2
    "lai": (0, 8, 160),       # m2/m2
    "pr":  (0, 800, 320),     # W/m2 (1 mm/day = 28.4 W/m2)
    "rn":  (-100, 250, 350),  # W/m2
}
QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)

# Axis labels, climatology color ranges and the range of the differences between products
LABELS = {"et": "ET [W m$^{-2}$]", "lai": "LAI [m$^2$ m$^{-2}$]", "pr": "P [W m$^{-2}$]", "rn": "Rn [W m$^{-2}$]"}
CLIM_RANGE = {"et": (0, 120), "lai": (0, 6), "pr": (0, 300), "rn": (-20, 160)}
DIFF_MAX = {"et": 30, "lai": 1.5, "pr": 60, "rn": 30}

# Fixed categorical order (validated for color-vision deficiency): a product keeps its color in every
# figure of its variable. Line plots show at most 8 products per panel (ILAMB and gridded apart if needed)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Valid fractions: one hue, with land without any valid value (0) in gray
FRAC_KW = {"cmap": plt.get_cmap("Blues").with_extremes(under="0.6"), "vmin": 1e-6, "vmax": 1}
FRAC_NOTE = "gray: land, none valid"


# ------------------------------------------------------------------
# Loading and reduction
# ------------------------------------------------------------------

def load(v: str, p: str) -> xr.DataArray:
    """
    Monthly field of product `p` of variable `v` over RECORD_SLICE on its native grid within LAT_BNDS,
    in W/m2 (LAI m2/m2) as in step 3 of prep_obs.py, lazy with one year per chunk.
    """
    if p in GRIDDED_PRODUCTS.get(v, {}):
        dataset, version, var = GRIDDED_PRODUCTS[v][p]
        da = lo.load_obs(dataset, var, RECORD_SLICE, version=version, freq="monthly")
        if v == "et":
            # Monthly totals -> kg m-2 s-1 -> W/m2
            da = units.latent_heat_to_wm2(units.accumulation_to_flux(da))
    else:
        da = il.load_native(v, p, RECORD_SLICE)
        # Keyed on the file's units attr (kg m-2 s-1, W m-2, MJ m-2 day-1, mm/day)
        da = da.assign_attrs(units="m2/m2") if v == "lai" else units.flux_to_wm2(da)
    # Pipeline domain without Antarctica; LAT_BNDS (-58) is a cell edge of every product's grid
    return da.sel(lat=cfg.LAT_BNDS).rename(v).chunk({"time": 12, "lat": -1, "lon": -1})


def month_axis(da: xr.DataArray) -> xr.DataArray:
    """`da` on a monthly axis (first of the month) from its first to its last month; missing months are NaN."""
    year, month = da.time.dt.year.values, da.time.dt.month.values
    first, last = f"{year[0]}-{month[0]:02d}", f"{year[-1]}-{month[-1]:02d}"
    return temporal.on_month_axis(da, pd.date_range(first, last, freq="MS"))


def hist_bins(v: str) -> np.ndarray:
    """Histogram bin edges of variable `v` (HIST_BINS)."""
    first, last, n = HIST_BINS[v]
    return np.linspace(first, last, n + 1)


def reduce_native(da: xr.DataArray, land: xr.DataArray, area: xr.DataArray, years: list[int], v: str):
    """
    Lazy reductions of the monthly field `da` (time, lat, lon) on its native grid, computed in one read:
    - zonal (time, lat), over the whole record: land area [km2] of each latitude row with valid data, and
      the sum of value x area over it
    - maps (lat, lon), over the complete `years` in TIME_SLICE: valid months, valid years (rule of
      REQUIRE_ALL_MONTHS) and calendar months valid in fewer than SEASONAL_GAP_FRAC of the years
    - land-area weighted histogram of the valid values in those years, on HIST_BINS, and the area outside
      the bins
    """
    valid = da.notnull() & land
    zonal = xr.Dataset({
        "valid_area": xr.dot(valid, area, dim="lon"),
        "value_area": xr.dot(da.where(valid, 0), area, dim="lon"),
    })

    in_years = da.time.dt.year.isin(years)
    v_months = valid.sel(time=in_years)
    n_valid = v_months.groupby("time.year").sum()
    maps = xr.Dataset({
        "n_valid_months": v_months.sum("time"),
        "n_valid_years": (n_valid == 12 if REQUIRE_ALL_MONTHS[v] else n_valid > 0).sum("year"),
        "n_gap_months": (v_months.groupby("time.month").mean() < SEASONAL_GAP_FRAC).sum("month"),
    })

    # Weight 0 for invalid values, so they can be put in any bin
    bins = hist_bins(v)
    x = da.sel(time=in_years)
    w = (x.notnull() * area).transpose(*x.dims)
    values = x.fillna(bins[0]).clip(bins[0], bins[-1])
    hist, _ = dask.array.histogram(values.data, bins=bins, weights=w.data.rechunk(values.data.chunks))
    outside = (w * ((x < bins[0]) | (x > bins[-1]))).sum()
    return zonal, maps, hist, outside


def to_target(da: xr.DataArray, land: xr.DataArray, v: str, label: str) -> xr.DataArray:
    """Annual means on the target grid over TIME_SLICE, as steps 4-8 of prep_obs.py."""
    a = temporal.aggregate(da.sel(time=TIME_SLICE), "year", require_all_months=REQUIRE_ALL_MONTHS[v], label=label)
    a = rg.mask_to_land(a, land, label)
    if v == "et":
        a = binning.drop_et_outliers(a, cfg.MIN_ANNUAL_ET, label=label)
    regridder = rg.bounded_conservative_regridder(rg.bounded_src_grid(a), TARGET_RES)
    a = rg.on_grid(rg.regrid_with_na_thresh(a, regridder, cfg.NA_THRESH), TARGET_RES, label)
    return a.sel(lat=cfg.LAT_BNDS)


def hist_quantiles(hist: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """QUANTILES of a weighted histogram, interpolated linearly within its bins."""
    cdf = np.concatenate([[0], np.cumsum(hist)]) / hist.sum()
    return np.interp(QUANTILES, cdf, bins)


def land_km2(mask: xr.DataArray, area: xr.DataArray) -> float:
    """Land area [10^6 km2] of the cells in `mask`."""
    return float(area.where(mask).sum()) / 1e6


def area_mean(da: xr.DataArray, land_area: xr.DataArray) -> float:
    """Land-area weighted mean of `da` (lat, lon) over its valid cells."""
    w = land_area.where(da.notnull(), 0)
    return float((da.fillna(0) * w).sum() / w.sum())


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_land_area(table: pd.DataFrame, land_total: float, grid: str, title: str, fout):
    """Land area [10^6 km2] covered by each product: native (any valid month, a valid year) and target grid."""
    kinds = {
        "native: any valid month": "native any valid month",
        "native: >= 1 valid year": "native >= 1 valid year",
        f"{grid}: climatology": f"{grid} clim Global",
    }
    products = list(table.index)
    x, width = np.arange(len(products)), 0.27
    fig, ax = plt.subplots(figsize=(max(7.0, 1.7 * len(products)), 4), layout="constrained")
    for i, (label, col) in enumerate(kinds.items()):
        bars = ax.bar(x + (i - 1) * width, table[col].astype(float), width * 0.92, color=PALETTE[i], label=label)
        ax.bar_label(bars, fmt="%.0f", fontsize=7)
    ax.axhline(land_total, color="0.4", ls="--", lw=1, label=f"land area ({grid})")
    ax.set_xticks(x, [f"{p}\n{table.loc[p, 'years']} yr, {table.loc[p, f'{grid} clim cells']} cells" for p in products],
                  fontsize=8)
    ax.set_ylabel("land area [10$^6$ km$^2$]")
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(fontsize=8, loc="outside lower center", ncols=4, frameon=False)
    ax.set_title(title)
    return plotting.finish(fig, fout)


def plot_maps(fields: dict, title: str, fout, *, cbar_label: str, extend: str = "neither", ticks=None, **kwargs):
    """One map per product ({label: (lat, lon)}), sharing one colorbar."""
    fig, axs = plotting.facets(len(fields), ncols=4, maps=True, panel_size=(4.6, 2.2))
    for ax, (label, da) in zip(axs, fields.items()):
        pm = da.plot.pcolormesh(ax=ax, transform=cfg.PROJECTION, add_colorbar=False, **kwargs)
        plotting.map_ax(ax, cfg.LAT_BNDS)
        ax.set_title(label, fontsize=9)
    fig.colorbar(pm, ax=axs, location="bottom", shrink=0.5, aspect=40, extend=extend, ticks=ticks,
                 label=cbar_label)
    fig.suptitle(title)
    return plotting.finish(fig, fout)


def plot_zonal(fields: dict, x: str, title: str, fout, *, time_box: bool = False):
    """Valid land fraction (lat, `x`) of each product, `x` being time or calendar month."""
    n = len(fields)
    if x == "time":  # long panels, in two columns for many products
        fig, axs = plotting.facets(n, ncols=2 if n > 4 else 1, sharex=True, sharey=True, panel_size=(7.0, 1.9))
    else:
        fig, axs = plotting.facets(n, ncols=4, sharex=True, sharey=True, panel_size=(3.6, 2.4))
    for ax, (p, z) in zip(axs, fields.items()):
        pm = ax.pcolormesh(z[x], z.lat, z.transpose("lat", x), shading="nearest", **FRAC_KW)
        if time_box:
            ax.axvspan(pd.Timestamp(TIME_SLICE.start), pd.Timestamp(TIME_SLICE.stop), facecolor="none",
                       edgecolor="k", lw=1.2)
        if x == "month":
            ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
        ax.set_title(p, loc="left", fontsize=9)
        ax.set_ylim(cfg.LAT_BNDS.start, cfg.LAT_BNDS.stop)
    for ax in axs:
        ax.set_ylabel("latitude")
        ax.label_outer()
    fig.colorbar(pm, ax=axs, location="bottom", shrink=0.5, aspect=40, extend="min",
                 label=f"valid fraction of land ({FRAC_NOTE})")
    fig.suptitle(title)
    return plotting.finish(fig, fout)


def annual(ts: xr.DataArray) -> xr.DataArray:
    """Calendar-year mean of a monthly series, only for years with 12 valid months, stamped mid-year."""
    ann = ts.groupby("time.year").mean().where(ts.notnull().groupby("time.year").sum() == 12)
    return ann.assign_coords(year=pd.to_datetime([f"{y}-07-01" for y in ann.year.values])).rename(year="time")


def plot_regions(series: dict, groups: dict, colors: dict, ylabel: str, title: str, fout, *, annual_mean=False):
    """
    One row per region and one column per group of products ({group: [products]}), one line per product:
    monthly (thin), and with `annual_mean` also annual means (thick). TIME_SLICE is shaded.
    """
    fig, axs = plt.subplots(len(series), len(groups), figsize=(9 * len(groups), 2.3 * len(series)),
                            sharex=True, sharey="row", squeeze=False, layout="constrained")
    for i, (r, by_product) in enumerate(series.items()):
        for j, (g, products) in enumerate(groups.items()):
            ax = axs[i, j]
            for p in products:
                ts = by_product[p]
                ax.plot(ts.time, ts, color=colors[p], lw=0.6 if annual_mean else 1.2,
                        alpha=0.45 if annual_mean else 1, label=None if annual_mean else p)
                if annual_mean:
                    ann = annual(ts)
                    ax.plot(ann.time, ann, color=colors[p], lw=2, marker="o", ms=3, label=p)
            ax.axvspan(pd.Timestamp(TIME_SLICE.start), pd.Timestamp(TIME_SLICE.stop), color="0.92", zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(r if len(groups) == 1 else f"{r}, {g}", loc="left", fontsize=10)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(series) - 1:
                ax.legend(fontsize=8, ncols=4, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.2))
    fig.suptitle(f"{title} (gray: {TIME_SLICE.start} to {TIME_SLICE.stop})")
    return plotting.finish(fig, fout)


def plot_diff_matrix(ann1: dict, land_area1: xr.DataArray, v: str, title: str, fout):
    """
    Every pair of products (row minus column, upper triangle): the mean over their shared valid years of the
    difference of their annual means on the target grid, with its land-area weighted global mean.
    """
    products = list(ann1)
    n = len(products) - 1
    fig, axs = plt.subplots(n, n, figsize=(2.9 * n + 1, 1.6 * n + 0.8), squeeze=False, layout="constrained",
                            subplot_kw={"projection": cfg.PROJECTION})
    kwargs = {"cmap": "RdBu_r", "vmin": -DIFF_MAX[v], "vmax": DIFF_MAX[v]}
    used = []
    for i, a in enumerate(products[:-1]):
        for j, b in enumerate(products[1:]):
            ax = axs[i, j]
            if j < i:  # each pair once
                ax.remove()
                continue
            x, y = xr.align(ann1[a], ann1[b], join="inner")  # shared years
            plotting.map_ax(ax, cfg.LAT_BNDS)
            used.append(ax)
            if x.sizes["year"] == 0:
                ax.set_title(f"{a} - {b}\nno shared years", fontsize=7)
                continue
            # Mean over the years in which both are valid, so both cover the same years in each cell
            d = (x - y).mean("year")
            d.plot.pcolormesh(ax=ax, transform=cfg.PROJECTION, add_colorbar=False, **kwargs)
            ax.set_title(f"{a} - {b}\n{int(x.year.min())}-{int(x.year.max())}, "
                         f"global {area_mean(d, land_area1):+.2f}", fontsize=7)
    fig.colorbar(plt.cm.ScalarMappable(mcolors.Normalize(kwargs["vmin"], kwargs["vmax"]), kwargs["cmap"]),
                 ax=used, shrink=0.5, extend="both", label=f"difference of climatological {LABELS[v]}")
    fig.suptitle(title)
    return plotting.finish(fig, fout)


def plot_hist(hists: dict, v: str, title: str, fout):
    """Density of the valid values of each product on HIST_BINS, with vertical lines at QUANTILES."""
    bins = hist_bins(v)
    fig, axs = plotting.facets(len(hists), ncols=4, sharex=True, panel_size=(4.2, 2.6))
    for ax, (p, h) in zip(axs, hists.items()):
        ax.stairs(h["hist"] / h["hist"].sum() / np.diff(bins), bins, color=PALETTE[0], lw=1.2)
        for q, x in zip(QUANTILES, h["quantiles"]):
            ax.axvline(x, color="k", lw=1 if q == 0.5 else 0.7, ls="-" if q == 0.5 else "--")
        text = "\n".join(f"q{100 * q:g} = {x:.3g}" for q, x in zip(QUANTILES, h["quantiles"]))
        ax.text(0.98, 0.95, text, transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set_title(f"{p}: {100 * h['outside']:.2g}% outside the bins", fontsize=9)
        ax.set_xlim(bins[0], bins[-1])
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs:
        ax.set_xlabel(f"monthly {LABELS[v]}")
        ax.set_ylabel("density (land-area weighted)")
        ax.label_outer()
    fig.suptitle(title)
    return plotting.finish(fig, fout)


def zonal_fraction(z: xr.Dataset) -> xr.DataArray:
    """Valid fraction of the land area of each latitude row (time, lat); NaN for rows without land."""
    return (z.valid_area / z.land_area).where(z.land_area > 0)


def product_colors(products: list[str], info: dict) -> tuple[dict, dict]:
    """
    Color of each product in fixed PALETTE order, and the groups of the line plots: all products together,
    or ILAMB and gridded products apart when there are more than len(PALETTE).
    """
    if len(products) <= len(PALETTE):
        groups = {"all": products}
    else:
        groups = {g: [p for p in products if info[p]["source"] == g] for g in ("ILAMB", "gridded")}
    if any(len(ps) > len(PALETTE) for ps in groups.values()):
        raise ValueError(f"more than {len(PALETTE)} products in a group: {groups}")
    colors = {p: PALETTE[i] for ps in groups.values() for i, p in enumerate(ps)}
    return colors, groups


# ------------------------------------------------------------------
# Diagnostics of one variable
# ------------------------------------------------------------------

def diagnose(v: str, land_area1: xr.DataArray, fig_dir, proc_dir, period: str, grid: str):
    products = [*ILAMB_PRODUCTS[v], *GRIDDED_PRODUCTS.get(v, {})]

    # Load each product once and reduce it: native-grid coverage, histogram, and annual means on the target grid
    info, zonal, maps, hists, ann1 = {}, {}, {}, {}, {}
    for p in products:
        label = f"{v}/{p}"
        print(f"\n=== {label} ===")
        try:
            da = load(v, p)
        except FileNotFoundError as err:
            print(f"{err}; skipping {label}")
            continue
        dm = month_axis(da)
        # Complete calendar years of the record inside TIME_SLICE
        years = temporal.complete_years(dm.sel(time=TIME_SLICE))
        if not years:
            print(f"{label}: no complete year in {period}, skipping")
            continue

        # Land (step 5) and the land area of each native cell [km2], 0 over the ocean
        land = rg.landne_mask(da)
        area = xr.DataArray(rg.cell_area(da.lat.values, da.lon.values) / 1e6, dims=("lat", "lon"),
                            coords={"lat": da.lat, "lon": da.lon}) * land

        z, m, h, out = dask.compute(*reduce_native(dm.where(land), land, area, years, v))
        zonal[p] = z.assign(land_area=area.sum("lon"))
        maps[p] = m.assign(land=land, area=area)
        hists[p] = {"hist": h, "outside": float(out) / float(h.sum()), "quantiles": hist_quantiles(h, hist_bins(v))}
        info[p] = {
            "source": "gridded" if p in GRIDDED_PRODUCTS.get(v, {}) else "ILAMB",
            "native res": rg.native_res(da),
            "record": f"{dm.time.dt.strftime('%Y-%m').values[0]} to {dm.time.dt.strftime('%Y-%m').values[-1]}",
            "years": len(years),
        }
        print(f"{label}: {info[p]['native res']}, {info[p]['record']}, {len(years)} complete years in {period}")

        ann1[p] = to_target(da, land, v, label)
        del da, dm

    products = list(info)  # products with files and a complete year in TIME_SLICE
    if not products:
        print(f"{v}: no product with a complete year in {TIME_SLICE}, no diagnostics")
        return
    clim1 = {p: temporal.aggregate(a, "clim", label=f"{v}/{p}") for p, a in ann1.items()}
    colors, groups = product_colors(products, info)

    # 1. Land covered, native and target grid
    rows = {}
    for p in products:
        m = maps[p]
        n_years = info[p]["years"]
        row = {
            **info[p],
            "native land cells": int(m.land.sum()),
            "native land area": land_km2(m.land, m.area),
            "native any valid month": land_km2(m.n_valid_months > 0, m.area),
            "native >= 1 valid year": land_km2(m.n_valid_years > 0, m.area),
            "native all years valid": land_km2(m.n_valid_years == n_years, m.area),
            f"{grid} clim cells": int(clim1[p].notnull().sum()),
        }
        for r, b in REGIONS.items():
            row[f"{grid} clim {r}"] = land_km2(clim1[p].notnull().sel(lat=b), land_area1.sel(lat=b))
            row[f"{grid} clim {r} [%]"] = 100 * row[f"{grid} clim {r}"] / land_km2(land_area1.sel(lat=b) > 0,
                                                                                    land_area1.sel(lat=b))
        row.update({f"q{100 * q:g}": x for q, x in zip(QUANTILES, hists[p]["quantiles"])})
        rows[p] = row
    table = pd.DataFrame(rows).T.infer_objects().rename_axis("product")
    fout = proc_dir / f"{FAMILY}.{v}.coverage.{period}.csv"
    table.to_csv(fout, float_format="%.4g")
    print(f"\nLand area in 10^6 km2, {period}\n{table.round(2).to_string()}\n{fout}")

    span = f"{TIME_SLICE.start} to {TIME_SLICE.stop}"
    land_total = land_km2(land_area1 > 0, land_area1)
    figs = {
        "land_area": lambda f: plot_land_area(table, land_total, grid, f"Land area with {v} data, {span}", f),
        # 2. Coverage maps on the native grids
        "map_valid_months": lambda f: plot_maps(
            {f"{p} ({12 * info[p]['years']} months)": (maps[p].n_valid_months / (12 * info[p]["years"])).where(maps[p].land)
             for p in products},
            f"Fraction of months with valid {v}, {span}", f, cbar_label=f"fraction of months ({FRAC_NOTE})",
            extend="min", **FRAC_KW),
        "map_valid_years": lambda f: plot_maps(
            {f"{p} ({info[p]['years']} years)": (maps[p].n_valid_years / info[p]["years"]).where(maps[p].land)
             for p in products},
            f"Fraction of valid years of {v} ({'12 valid months' if REQUIRE_ALL_MONTHS[v] else '>= 1 valid month'})"
            f", {span}", f, cbar_label=f"fraction of years ({FRAC_NOTE})", extend="min", **FRAC_KW),
        "map_seasonal_gaps": lambda f: plot_maps(
            {p: maps[p].n_gap_months.where(maps[p].land) for p in products},
            f"Seasonal gaps of {v}: calendar months valid in fewer than {100 * SEASONAL_GAP_FRAC:g}% of the years, "
            f"{span}", f, cbar_label="calendar months", ticks=range(13),
            cmap=mcolors.ListedColormap(plt.get_cmap("Reds")(np.linspace(0.08, 1, 13))),
            norm=mcolors.BoundaryNorm(np.arange(-0.5, 13), 13)),
        # 3. Valid fraction of the land area of each latitude row
        "zonal_time": lambda f: plot_zonal(
            {p: zonal_fraction(zonal[p]) for p in products}, "time",
            f"Valid fraction of the land area of each latitude row, {v} (box: {span})", f, time_box=True),
        "zonal_calendar_month": lambda f: plot_zonal(
            {p: zonal_fraction(zonal[p]).sel(time=TIME_SLICE).groupby("time.month").mean() for p in products},
            "month",
            f"Valid fraction of the land area of each latitude row by calendar month, {v}, {span}", f),
    }
    # 4. Regional timeseries, from the zonal sums
    cover, mean = {}, {}
    for r, b in REGIONS.items():
        band = {p: zonal[p].sel(lat=b).sum("lat") for p in products}
        cover[r] = {p: 100 * z.valid_area / z.land_area for p, z in band.items()}
        mean[r] = {p: (z.value_area / z.valid_area).where(z.valid_area > 0) for p, z in band.items()}
    figs |= {
        "ts_valid_land": lambda f: plot_regions(
            cover, groups, colors, "valid land [%]", f"Monthly valid fraction of the land area, {v}", f),
        "ts_mean": lambda f: plot_regions(
            mean, groups, colors, LABELS[v],
            f"Area-weighted mean {v} over each product's valid land: monthly (thin), annual (thick, 12 valid months)",
            f, annual_mean=True),
        # 6. Histograms of the valid values
        "hist": lambda f: plot_hist(
            hists, v,
            f"Valid monthly {v} on land (native grids, land-area weighted), {span}", f),
    }
    # 1 and 5 on the target grid
    masks = xr.concat([c.notnull() for c in clim1.values()], "product").assign_coords(product=list(clim1))
    vmin, vmax = CLIM_RANGE[v]
    figs |= {
        f"map_n_products.{grid}": lambda f: plotting.plot_mask_agreement(
            masks, land_area1 > 0, f"Products with a climatological {v} ({grid}), {span}", f,
            label="products", highlight_common=True),
        f"map_clim.{grid}": lambda f: plot_maps(
            {f"{p} ({info[p]['years']} years, global {area_mean(c, land_area1):.3g})": c for p, c in clim1.items()},
            f"Climatological {v} ({grid}), {span}", f, cbar_label=LABELS[v], extend="both",
            cmap=plotting.MAP_KWARGS[v]["cmap"], vmin=vmin, vmax=vmax),
    }
    if len(ann1) > 1:
        figs[f"map_diff.{grid}"] = lambda f: plot_diff_matrix(
            ann1, land_area1, v, f"Climatological {v}: row minus column over their shared valid years "
            f"({grid}, {span})", f)

    for diag, make in figs.items():
        fout = fig_dir / f"{FAMILY}.{v}.{diag}.{period}.png"
        make(fout)
        print(fout)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.summarize_settings(globals())
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)
    fig_dir = cfg.output_dir(cfg.FIG_ROOT, FAMILY, "coverage", period)
    proc_dir = cfg.output_dir(cfg.PROC_ROOT, FAMILY, "coverage", period)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Land area [km2] of each target cell within LAT_BNDS, without Greenland/Iceland
    land_area1 = rg.land_area(rg.target_grid(TARGET_RES).sel(lat=cfg.LAT_BNDS))
    land_area1 = land_area1.where(rg.greenland_mask(land_area1), 0)

    for v in VARIABLES:
        diagnose(v, land_area1, fig_dir, proc_dir, period, grid)


if __name__ == "__main__":
    main()
