"""
prep_era5.py
============
Binning inputs of the ERA5 reanalysis: steps 1-11 of docs/binning-workflow.md,
written out in `main()`. ET, P and Rn come from ERA5 and LAI from the ILAMB
product LAI_PRODUCT (ERA5 has no LAI). The (ERA5, LAI_PRODUCT) pair is one
source. make_edges.py and bin_all.py then use the saved inputs.

ERA5 monthly mean rates from GDEX (`le.load_native`): ET = -mer, P = mtpr,
Rn = msnswrf + msnlwrf. LAI from `il.load_native`. Native land mask: ERA5
LANDFRAC > LF_THRESH for ET, P and Rn, and Natural Earth land for LAI (no land
fraction), both without Greenland and Iceland.

Outputs
-------
<grid>   : grid tag of TARGET_RES, e.g. "1deg"
<period> : YYYYMM-YYYYMM of TIME_SLICE

PROC_ROOT/era5/era5.inputs.<grid>.<period>.nc
    et (member, year, lat, lon); lai, pr, rn, ai (member, lat, lon); with BIN_VARIABLE = "et_pr",
    also et_pr (member, year, lat, lon); `member` labelled
    "ERA5-<LAI_PRODUCT>", `source` = member, coord `lai_product`
FIG_ROOT/era5/clim/era5.<var>.<grid>.map.<period>.png        var in et, lai, pr, rn, ai (and et_pr)
    climatology maps
FIG_ROOT/era5/clim/era5.{lai,ai}_clim.<grid>.hist.<period>.png
    histograms of climatological LAI and AI
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
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

FAMILY = "era5"
FIG_ROOT = cfg.FIG_ROOT / FAMILY

TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES
BIN_VARIABLE = cfg.BIN_VARIABLE  # "et_pr" also saves the annual evaporation ratio ET/P (binned by bin_all.py)

# GDEX "meanflux" parameter codes of each ERA5 variable
ERA5_CODES = {"et": ["mer"], "pr": ["mtpr"], "rn": ["msnswrf", "msnlwrf"]}
# ILAMB LAI product (key of il.PRODUCTS["lai"]); GIMMS_LAI4g covers all of TIME_SLICE
LAI_PRODUCT = "GIMMS_LAI4g"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.summarize_settings(globals())
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)

    # 1-2. Monthly native fields over TIME_SLICE, and the ERA5 land fraction
    print("=== 1. Load ERA5 and LAI ===")
    raw = le.load_native([c for codes in ERA5_CODES.values() for c in codes], TIME_SLICE)
    landfrac, era5_grid = le.load_land_fraction()
    lai = il.load_native("lai", LAI_PRODUCT, TIME_SLICE)

    # 3. ET, P and Rn in W/m2. Mean rates are positive downward: convert_units also flips the sign of mer
    monthly = {
        "et": units.convert_units("mer", raw["mer"]),
        "lai": lai.assign_attrs(units="m2/m2"),
        "pr": units.convert_units("mtpr", raw["mtpr"]),
        "rn": units.net_radiation_era5(raw["msnswrf"], raw["msnlwrf"]),
    }

    # 4. Annual means over complete years: ET, P and Rn need all 12 months, a missing LAI month counts as 0
    print("\n=== 4. Annual means ===")
    ann = {k: temporal.aggregate(da, "year", require_all_months=(k != "lai"), label=k) for k, da in monthly.items()}

    # 5. Native land mask, without Greenland/Iceland: ERA5 LANDFRAC > LF_THRESH for ET, P and Rn,
    # Natural Earth land for LAI (no land fraction). LAI is on its own native grid
    src_grids = {k: era5_grid for k in ("et", "pr", "rn")}
    src_grids["lai"] = rg.bounded_src_grid(ann["lai"])
    lands = {k: rg.landfrac_mask(landfrac, cfg.LF_THRESH) for k in ("et", "pr", "rn")}
    lands["lai"] = rg.landne_mask(ann["lai"])
    ann = {k: rg.mask_to_land(da, lands[k], k) for k, da in ann.items()}

    # 6. ET outlier filter (on land, before regridding, so an outlier does not enter its target cell)
    ann["et"] = binning.drop_et_outliers(ann["et"], cfg.MIN_ANNUAL_ET)

    # 7. Conservative regridding onto the target grid (NaN where more than NA_THRESH is missing)
    regridders = {k: rg.bounded_conservative_regridder(g, TARGET_RES) for k, g in src_grids.items()}
    ann = {k: rg.on_grid(rg.regrid_with_na_thresh(da, regridders[k], cfg.NA_THRESH), TARGET_RES, k)
           for k, da in ann.items()}

    # 8. Domain without Antarctica
    ann = {k: da.sel(lat=cfg.LAT_BNDS).rename(k) for k, da in ann.items()}

    # 9. Complete years shared by ET, LAI, P and Rn
    years = temporal.shared_years(*ann.values())
    if len(years) < cfg.MIN_YEARS:
        raise RuntimeError(f"Only {len(years)} shared years, fewer than MIN_YEARS = {cfg.MIN_YEARS}")
    ann = {k: da.sel(year=years) for k, da in ann.items()}
    print(f"n_shared_years = {len(years)}")

    # 10. Climatological LAI, P and Rn over those years
    print("\n=== 10. Climatologies ===")
    clim = {k: temporal.aggregate(ann[k], "clim", label=k) for k in ("lai", "pr", "rn")}

    # 11. Aridity index from the climatological Rn and P
    ai = edges.compute_aridity_index(clim["pr"], clim["rn"])

    # 11b. Annual evaporation ratio ET/P (BIN_VARIABLE = "et_pr" only), with the annual P of the AI's
    # P (same product and shared years)
    data = {"et": ann["et"], **clim, "ai": ai}
    if BIN_VARIABLE == "et_pr":
        data["et_pr"] = binning.evaporation_ratio(ann["et"], ann["pr"], label="et_pr")

    # ERA5 with LAI_PRODUCT is one source with one member
    name = f"ERA5-{LAI_PRODUCT}"
    inputs = xr.Dataset(data).expand_dims(member=[name]).assign_coords(
        source=("member", [name]), lai_product=("member", [LAI_PRODUCT]),
    )
    inputs.attrs = {
        "family": FAMILY, "dataset": f"ERA5 (ET, P, Rn) with ILAMB {LAI_PRODUCT} LAI",
        "time_period": temporal.period_str(years), "grid": grid,
        "native_res": f"ERA5: {rg.native_res(era5_grid)}; lai/{LAI_PRODUCT}: {rg.native_res(src_grids['lai'])}",
        "na_thresh": cfg.NA_THRESH, "lf_thresh": cfg.LF_THRESH, "min_annual_et": cfg.MIN_ANNUAL_ET,
        "land_mask": "ERA5: LANDFRAC > lf_thresh; LAI: Natural Earth land; both without Greenland/Iceland",
    }
    fout = cfg.inputs_path(FAMILY, grid, period)
    fout.parent.mkdir(parents=True, exist_ok=True)
    inputs.to_netcdf(fout)
    print(f"\n{name}, {inputs.attrs['time_period']}\n{fout}")

    # Diagnostics: climatology maps and LAI/AI histograms
    plotting.save_clim_diagnostics(inputs, FIG_ROOT / "clim", FAMILY, grid, period, title="ERA5")


if __name__ == "__main__":
    main()
