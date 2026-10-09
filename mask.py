"""
mask.py
=======
Build one static lat/lon mask shared by every data product listed in
ILAMB_PRODUCTS and GRIDDED_PRODUCTS: the land gridcells where each product is
valid in (enough of) its own valid months over TIME_SLICE.

Use the common mask to compare products over exactly the same area, so that
differences between them are not caused by one product covering a region
(e.g. a desert) that another leaves out.

Usage
-----
Edit the settings below (period, grid, product list, thresholds), then run

    $PY mask.py

Steps
-----
1. Load the monthly field of each product over TIME_SLICE on the common RES
   grid (`rg.target_grid`), restricted to `config.LAT_BNDS` and the land mask
   (Natural Earth without Greenland/Iceland, `rg.land_mask`).
   - ILAMB products are read with `il.load_ilamb` (lon in [-180, 180], lat
     ascending, undecoded fill values removed) and bilinearly interpolated
     when their grid is not RES.
   - PML, GLEAM and SiTHv2 are read from the RES files written by
     `regrid_obs.py` (`lo.load_obs(..., res=...)`).
2. The valid months of a product are its time steps in TIME_SLICE with at
   least one valid land gridcell. Months outside the product's record, and
   empty months inside it (e.g. a missing file), are not counted, so a product
   with a shorter record or a gap is not penalized for it.
3. A land gridcell is in a product's mask when it is valid in at least
   MIN_VALID_FRAC[variable] of the product's valid months (1.0: all of them).
4. The common mask is the land gridcells in the mask of every product.

Only validity (not NaN) matters, so no units are converted. Every listed
product must load; a missing product raises instead of being left out of the
common mask.

Example: a product with data from 2000-01 to 2014-12 within a 1982-2025
TIME_SLICE has 180 valid months. With MIN_VALID_FRAC = 1.0, a gridcell is in
its mask only if it has data in all 180 of those months; the 1982-1999 and
2015-2025 months do not count against it.

Outputs
-------
<period> : YYYYMM-YYYYMM of TIME_SLICE
<res>    : RES ("0.5deg" or "1deg")

PROC_ROOT/<MASK_NAME>.common_mask.<res>.<period>.nc
    common_mask, land (lat, lon); product_mask, n_valid_months,
    frac_valid_months (product, lat, lon); n_product_months (product)
FIG_ROOT/<MASK_NAME>.common_mask.<res>.<period>.png
    number of products whose mask includes each land gridcell
FIG_ROOT/<MASK_NAME>.product_masks_any_month.<res>.<period>.png
    one map per product of the land gridcells with data in any month of its
    record (not the MIN_VALID_FRAC mask above)
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write figures to file without a display
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import xarray as xr

import etunc.config as config
import etunc.temporal as temporal
import etunc.plotting as plotting
import ilamb_binned_et as ib  # output roots
import etunc.load.ilamb as il  # ILAMB product table and loader
import etunc.load.obs as lo        # loader for PML / GLEAM / SiTH files
import etunc.grid as rg          # common target grids and regridders


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

# Period over which validity is assessed. Each product only counts the months
# it actually has inside this window (see `rg.product_mask`).
TIME_SLICE = slice("1982-01", "2025-12")
RES = "0.5deg"     # common grid, a key of rg.RESOLUTIONS
MASK_NAME = "obs"  # file name prefix; change it when changing the product set

# ILAMB products to include, per variable: {variable: [product, ...]}.
# Names must be keys of il.PRODUCTS[variable].
ILAMB_PRODUCTS = {
    "et":  ["CLASS", "DOLCE", "FLUXCOM", "GLEAMv3.3a", "MOD16A2", "MODIS", "WECANN"],
    "lai": ["MODIS"],
    "pr":  ["GPCPv2.3"],
    "rns": ["CERESed4.2"],
}

# Other gridded products, read from their regridded RES files:
# {variable: {label: (load_obs dataset, version, variable in files)}}
GRIDDED_PRODUCTS = {
    "et": {
        "PML-V2.2a-MODIS": ("pml", "V2.2a-MODIS", "ET"),
        "PML-V2.2c":       ("pml", "V2.2c", "ET"),
        "GLEAM-v4.3a":     ("gleam", "v4.3a", "E"),
        "GLEAM-v4.3b":     ("gleam", "v4.3b", "E"),
        # "SiTHv2":          ("sith", "v2", "ET"),  # no regridded files until regrid_obs.py has run
    },
}

# Minimum fraction of a product's valid months in which a gridcell must be
# valid, per variable. 1.0 = every month; e.g. 0.9 tolerates 10% missing months.
MIN_VALID_FRAC = {"et": 1.0, "lai": 1.0, "pr": 1.0, "rns": 1.0}

PROC_ROOT = ib.PROC_ROOT                  # NetCDF output directory
FIG_ROOT = ib.FIG_ROOT / "obs" / "mask"   # figure output directory

NCOLS = 4  # panels per row in plot_product_masks

# Derived from the settings above (no need to edit).
# FULL_GRID is the global grid that loaded data are checked against; GRID is
# the same grid cut to LAT_BNDS (no Antarctica), on which the masks are built.
FULL_GRID = rg.target_grid(RES)
GRID = FULL_GRID.sel(lat=config.LAT_BNDS)
# load_obs names resolutions differently from regrid.py
LO_RES = {tag: res for res, tag in lo.RES_DIRS.items()}[RES]  # "0.5deg" -> "0.5"


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------
# Each loader returns a monthly (time, lat, lon) DataArray over TIME_SLICE on
# GRID. Values stay in their native units, since only NaN vs. not NaN is used.

def load_gridded(variable: str, label: str) -> xr.DataArray:
    """Monthly field of a PML/GLEAM/SiTH product over TIME_SLICE from its RES files."""
    dataset, version, var = GRIDDED_PRODUCTS[variable][label]
    # load_obs only opens the files for the years in TIME_SLICE. Years without
    # a file are simply absent from the time axis (load_obs warns about them).
    # .load() reads everything once, instead of re-reading for each reduction.
    da = lo.load_obs(dataset, var, TIME_SLICE, version=version, freq="monthly", res=LO_RES).load()
    return rg.on_grid(da, RES, f"{variable}/{label}")


def product_loaders() -> dict[str, tuple[str, partial]]:
    """{"<variable>/<product>": (variable, loader)} of every listed product."""
    # Each loader is a zero-argument function, so `main` can load one product
    # at a time and free it before loading the next (to limit memory use).
    loaders = {}
    for variable, products in ILAMB_PRODUCTS.items():
        for p in products:
            loaders[f"{variable}/{p}"] = (variable, partial(il.load_ilamb, variable, p, TIME_SLICE, RES))
    for variable, products in GRIDDED_PRODUCTS.items():
        for p in products:
            loaders[f"{variable}/{p}"] = (variable, partial(load_gridded, variable, p))
    return loaders


# ------------------------------------------------------------------
# Masks
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

# Legend labels and colors of plot_product_masks (land without / with data)
ANY_MONTH_COLORS = {"land, no data in any month": "#f0b67f", "data in at least one month": "#2b6a99"}


def plot_product_masks(ds: xr.Dataset, title: str, fout: Path):
    """
    Mask of each product, one map per product: land gridcells with data in any
    month of the product's record within TIME_SLICE (`n_valid_months > 0`).
    This is looser than `rg.product_mask`, which needs MIN_VALID_FRAC of the months.
    """
    # Grid of map panels, NCOLS per row (unused panels of the last row removed)
    fig, axs = plotting.facets(ds.sizes["product"], ncols=NCOLS, maps=True, panel_size=(4.6, 2.5))

    land = ds["land"] == 1
    n_land = int(land.sum())
    # Two colors: 0 = land without data (orange), 1 = land with data (blue)
    cmap = mcolors.ListedColormap(list(ANY_MONTH_COLORS.values()))
    for ax, p in zip(axs, ds["product"].values):
        valid = (ds["n_valid_months"].sel(product=p) > 0) & land
        # Cast to float and blank out the ocean, so only land is colored
        valid.astype(float).where(land).plot.pcolormesh(
            ax=ax, transform=config.PROJECTION, cmap=cmap, vmin=0, vmax=1, add_colorbar=False,
        )
        plotting.map_ax(ax, config.LAT_BNDS)
        ax.set_title(f"{p} ({int(ds['n_product_months'].sel(product=p))} months)\n"
                     f"{int(valid.sum())} cells, {100 * int(valid.sum()) / n_land:.1f}% of land", fontsize=9)
    fig.legend(handles=[Patch(color=c, label=k) for k, c in ANY_MONTH_COLORS.items()],
               loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle(title)
    return plotting.finish(fig, fout)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    period = temporal.format_time_period(TIME_SLICE)  # e.g. "198201-202512", for file names

    # Land gridcells of GRID (Natural Earth, without Greenland/Iceland)
    land = (rg.land_mask(GRID) == 1).rename("land")
    n_land = int(land.sum())
    print(f"{period}, {RES}: {n_land} land gridcells\n")

    # 1. Load each product in turn and compute its mask. Only the small
    #    (lat, lon) results are kept, not the monthly data.
    masks = {}
    for label, (variable, load) in product_loaders().items():
        masks[label] = rg.product_mask(load(), land, MIN_VALID_FRAC[variable], label)
        m = masks[label]
        print(f"{label:22}: {int(m['n_product_months']):4d} valid months, "
              f"{int(m['product_mask'].sum()):6d} cells in mask "
              f"({100 * float(m['product_mask'].sum()) / n_land:5.1f}% of land)")

    # 2. Stack the per-product results along a new `product` dimension, with
    #    the variable (et, lai, pr, rns) of each product as a coordinate.
    labels = list(masks)
    ds = xr.concat(list(masks.values()), dim="product", coords="minimal", compat="override")
    ds = ds.assign_coords(
        product=np.array(labels),  # numpy str, not pandas' StringDtype, which netCDF cannot write
        variable=("product", np.array([label.split("/")[0] for label in labels])),
    )

    # 3. Common mask: land gridcells that are in the mask of every product
    common = (land & ds["product_mask"].all("product")).rename("common_mask")
    print(f"\ncommon mask: {int(common.sum())} of {n_land} land gridcells")

    # 4. Save. Masks are stored as int8 (0/1), since netCDF has no boolean
    #    type, and the settings are recorded in the attributes.
    ds = ds.assign(
        product_mask=ds["product_mask"].astype("i1"),
        common_mask=common.astype("i1"),
        land=land.astype("i1"),
    ).assign_attrs(
        time_period=period,
        res=RES,
        lat_bnds=str(config.LAT_BNDS),
        min_valid_frac=str(MIN_VALID_FRAC),
        products=", ".join(labels),
        description=(
            "Land gridcells valid in at least min_valid_frac of each product's valid months "
            "(months with any valid land gridcell) within time_period, for every product."
        ),
    )

    stem = f"{MASK_NAME}.common_mask.{RES}.{period}"
    PROC_ROOT.mkdir(parents=True, exist_ok=True)
    fout = PROC_ROOT / f"{stem}.nc"
    ds.to_netcdf(fout)
    print(fout)

    # 5. Figures
    # Agreement between the product masks (the common mask is the top class)
    fout = FIG_ROOT / f"{stem}.png"
    plotting.plot_mask_agreement(
        ds["product_mask"], ds["land"], f"Product masks, {TIME_SLICE.start} to {TIME_SLICE.stop}", fout,
        highlight_common=True,
    )
    print(fout)

    # Per-product coverage: gridcells with data in any month of the record
    fout = FIG_ROOT / f"{MASK_NAME}.product_masks_any_month.{RES}.{period}.png"
    plot_product_masks(ds, f"Land gridcells with data in any month, {TIME_SLICE.start} to {TIME_SLICE.stop}", fout)
    print(fout)


if __name__ == "__main__":
    main()
