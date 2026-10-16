"""
prep_obs.py
===========
Binning inputs of the observational products: steps 1-11 of
docs/binning-workflow.md, written out in `main()`. Steps 1-8 run once per
product, steps 9-11 once per (ET, LAI, P, Rn) combination of products. Every
combination is one source. make_edges.py and bin_all.py then use the saved
inputs.

Products
--------
- ILAMB products in ILAMB_PRODUCTS (`il.load_native`): ET, LAI, P and Rn.
- Gridded ET products in GRIDDED_ET_PRODUCTS (`lo.load_obs`, native 0.1 deg
  files of monthly or yearly totals): PML-V2.2, GLEAM v4.3, SiTHv2.
ERA5 is its own family (prep_era5.py). Native land mask: Natural Earth land
(the products have no land fraction), without Greenland and Iceland. A product
without data in TIME_SLICE is skipped.

Each combination uses the complete years shared by its four products inside
TIME_SLICE, and is skipped if there are fewer than MIN_YEARS.

Outputs
-------
<grid>   : grid tag of TARGET_RES, e.g. "1deg"
<period> : YYYYMM-YYYYMM of TIME_SLICE

PROC_ROOT/obs/obs.inputs.<grid>.<period>.nc
    et (member, year, lat, lon), NaN outside a combination's years; lai, pr, rn, ai (member, lat, lon);
    `member` labelled "<et>-<lai>-<pr>-<rn>" product names, with coords `source` (= member),
    `et_product`, `lai_product`, `pr_product`, `rn_product`, `n_years` and `time_period`
FIG_ROOT/obs/clim/obs.<var>.<grid>.map.<period>.png        var in et, lai, pr, rn, ai
    mean, standard deviation and range of the climatology over the combinations
FIG_ROOT/obs/clim/obs.{lai,ai}_clim.<grid>.hist.<period>.png
    histograms of climatological LAI and AI, one line per combination
"""

from __future__ import annotations

import itertools
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
import etunc.load.ilamb as il
import etunc.load.obs as lo

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS. Will affect performance.", category=UserWarning)

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

FAMILY = "obs"
FIG_ROOT = cfg.FIG_ROOT / FAMILY

TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES
MIN_YEARS = cfg.MIN_YEARS  # minimum complete years shared by the four products of a combination

FACTORS = ("et", "lai", "pr", "rn")  # order of the products in a combination

# ILAMB products used, by variable (keys of il.PRODUCTS[variable]); a variable missing here
# (or None) uses all of its il.PRODUCTS
ILAMB_PRODUCTS = {
    "et":  list(il.PRODUCTS["et"]),  # every ILAMB ET product
    "lai": ["GIMMS_LAI4g"],
    "pr":  ["GPCPv2.3"],
    "rn":  ["GEWEX.SRB"],
}

# Gridded ET products, read from their native files: {label: (load_obs dataset, version, variable, freq)}
GRIDDED_ET_PRODUCTS = {
    "PMLv2.2a_MODIS": ("pml", "V2.2a-MODIS", "ET", "monthly"),
    # "PMLv2.2b":       ("pml", "V2.2b", "ET", "monthly"),
    "PMLv2.2c":       ("pml", "V2.2c", "ET", "monthly"),
    "GLEAMv4.3a":     ("gleam", "v4.3a", "E", "monthly"),
    "GLEAMv4.3b":     ("gleam", "v4.3b", "E", "monthly"),
    "SiTHv2":         ("sith", "v2", "ET", "monthly"),
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.summarize_settings(globals())
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)

    # (variable, product, kind) of every product
    products = (
        [(v, p, "ilamb") for v in FACTORS for p in (ILAMB_PRODUCTS.get(v) or il.PRODUCTS[v])]
        + [("et", p, "gridded") for p in GRIDDED_ET_PRODUCTS]
    )

    # ------------------------------------------------------------------
    # Steps 1-8, once per product: annual means on the target grid
    # ------------------------------------------------------------------
    ann = {v: {} for v in FACTORS}  # {variable: {product: (year, lat, lon)}}
    native_res = {}  # {(variable, product): native resolution}
    for v, p, kind in products:
        label = f"{v}/{p}"
        print(f"\n=== {label} ===")

        # 1-2. Monthly native field over TIME_SLICE
        try:
            if kind == "ilamb":
                da = il.load_native(v, p, TIME_SLICE)
            else:
                dataset, version, var, freq = GRIDDED_ET_PRODUCTS[p]
                da = lo.load_obs(dataset, var, TIME_SLICE, version=version, freq=freq)
        except FileNotFoundError as err:
            print(f"{err}; skipping {label}")
            continue

        # 3. ET, P and Rn in W/m2
        if kind == "ilamb":
            # Keyed on the file's units attr (kg m-2 s-1, W m-2, MJ m-2 day-1, mm/day)
            da = da.assign_attrs(units="m2/m2") if v == "lai" else units.flux_to_wm2(da)
        else:
            # Totals per month (or year) -> kg m-2 s-1 -> W/m2
            da = units.latent_heat_to_wm2(units.accumulation_to_flux(da))

        # 4. Annual means over complete years: ET, P and Rn need all 12 months, a missing LAI month
        # counts as 0. A yearly total (gridded products only) is already the annual mean
        if kind == "gridded" and freq == "yearly":
            a = temporal.yearly_to_annual(da).load()
        else:
            a = temporal.aggregate(da, "year", require_all_months=(v != "lai"), label=label)
        if a.sizes["year"] == 0:
            print(f"{label}: no complete year in {TIME_SLICE}, skipping")
            continue

        # 5. Native land mask, without Greenland/Iceland: Natural Earth land, as the products have no land
        # fraction (removes the ocean of global P and Rn products)
        land, src_grid = rg.landne_mask(a), rg.bounded_src_grid(a)
        a = rg.mask_to_land(a, land, label)

        # 6. ET outlier filter (on land, before regridding, so an outlier does not enter its target cell)
        if v == "et":
            a = binning.drop_et_outliers(a, cfg.MIN_ANNUAL_ET, label=label)

        # 7. Conservative regridding onto the target grid (NaN where more than NA_THRESH is missing)
        regridder = rg.bounded_conservative_regridder(src_grid, TARGET_RES)
        a = rg.on_grid(rg.regrid_with_na_thresh(a, regridder, cfg.NA_THRESH), TARGET_RES, label)

        # 8. Domain without Antarctica
        ann[v][p] = a.sel(lat=cfg.LAT_BNDS).rename(v)
        native_res[(v, p)] = rg.native_res(src_grid)

    # ------------------------------------------------------------------
    # Steps 9-11, once per (et, lai, pr, rn) combination of products
    # ------------------------------------------------------------------
    print("\n=== Combinations ===")
    combos = []
    for combo in itertools.product(*(ann[v] for v in FACTORS)):
        cid = "-".join(combo)
        fields = {v: ann[v][p] for v, p in zip(FACTORS, combo)}

        # 9. Complete years shared by the four products
        years = temporal.shared_years(*fields.values())
        if len(years) < MIN_YEARS:
            print(f"{cid}: {len(years)} shared years < MIN_YEARS = {MIN_YEARS}, skipping")
            continue
        fields = {v: da.sel(year=years) for v, da in fields.items()}

        # 10. Climatological LAI, P and Rn over those years
        clim = {k: temporal.aggregate(fields[k], "clim", label=f"{cid}/{k}") for k in ("lai", "pr", "rn")}

        # 11. Aridity index from the climatological Rn and P
        ai = edges.compute_aridity_index(clim["pr"], clim["rn"])

        # The combination is a source with one member
        combos.append(xr.Dataset({"et": fields["et"], **clim, "ai": ai}).expand_dims(member=[cid]).assign_coords(
            source=("member", [cid]),
            **{f"{v}_product": ("member", [p]) for v, p in zip(FACTORS, combo)},
            n_years=("member", [len(years)]),
            time_period=("member", [temporal.period_str(years)]),
        ))
    if not combos:
        raise RuntimeError(f"No combination has at least {MIN_YEARS} shared years.")
    n_all = len(list(itertools.product(*(ann[v] for v in FACTORS))))
    print(f"{len(combos)} of {n_all} combinations have >= {MIN_YEARS} shared years")

    # ET is NaN in the years outside a combination's shared years
    inputs = xr.concat(combos, dim="member", join="outer", combine_attrs="drop_conflicts")
    inputs.attrs = {
        "family": FAMILY, "dataset": "observational products (ILAMB, gridded ET)",
        "time_period": temporal.period_str(inputs.year.values), "grid": grid,
        # Products in at least one saved combination
        "native_res": "; ".join(
            f"{v}/{p}: {res}" for (v, p), res in native_res.items() if p in inputs[f"{v}_product"].values
        ),
        "na_thresh": cfg.NA_THRESH, "min_annual_et": cfg.MIN_ANNUAL_ET,
        "min_years": MIN_YEARS,
        "land_mask": "Natural Earth land, without Greenland/Iceland",
    }
    fout = cfg.inputs_path(FAMILY, grid, period)
    fout.parent.mkdir(parents=True, exist_ok=True)
    inputs.to_netcdf(fout)
    print(f"{fout}")

    # Diagnostics: climatology maps and LAI/AI histograms
    plotting.save_clim_diagnostics(inputs, FIG_ROOT / "clim", FAMILY, grid, period, title="obs")


if __name__ == "__main__":
    main()
