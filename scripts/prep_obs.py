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
PRODUCTS, one mapping per variable (prep.obs.<v>_products of config/pipeline.yml).
Native files are monthly or yearly, or composites averaged to monthly in step 1:
half-monthly (GIMMS LAI4g; the mean of each month's valid halves) or 8-day
(HiQ-LAI; the mean of the valid composites that start in each month).

Native land mask: Natural Earth land (the products have no land fraction), without
Greenland and Iceland. A product without data in TIME_SLICE is skipped.

Each combination uses the complete years shared by its four products inside
TIME_SLICE, and is skipped if there are fewer than MIN_YEARS.

Outputs
-------
<prep_id> : PREP_ID of the prep settings (etunc.config; config/runs.csv)

PROC_ROOT/<prep_id>/obs/inputs/obs.inputs.<prep_id>.nc
    et (member, year, lat, lon), NaN outside a combination's years; lai, pr, rn, ai (member, lat, lon);
    and et_pr (member, year, lat, lon), ET/P with the combination's P product;
    `member` labelled "<et>-<lai>-<pr>-<rn>" product names, with coords `source` (= member),
    `et_product`, `lai_product`, `pr_product`, `rn_product`, `n_years` and `time_period`
FIG_ROOT/<prep_id>/obs/clim/obs.<var>.map.<prep_id>.png        var in et, et_pr, lai, pr, rn, ai
    mean, standard deviation and range of the climatology over the combinations
FIG_ROOT/<prep_id>/obs/clim/obs.<var>.map_members.<prep_id>.png    var in lai, pr, rn
    the climatology of every member, on one color scale (more than one member only)
FIG_ROOT/<prep_id>/obs/clim/obs.{lai,ai}_clim.hist.<prep_id>.png
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

TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES
MIN_YEARS = cfg.MIN_YEARS  # minimum complete years shared by the four products of a combination

FACTORS = ("et", "lai", "pr", "rn")  # order of the products in a combination

# Products of each variable: prep.obs of config/pipeline.yml, {label: {dataset, version, variable, freq, source}}.
# source "ilamb": il.load_native with the label; null: lo.load_obs with dataset, variable, version and freq
PRODUCTS = {
    "et": cfg.OBS_ET_PRODUCTS,
    "lai": cfg.OBS_LAI_PRODUCTS,
    "pr": cfg.OBS_PR_PRODUCTS,
    "rn": cfg.OBS_RN_PRODUCTS,
}
# Frequencies of native files that steps 1-4 handle: half-month (GIMMS LAI4g) and 8-day (HiQ-LAI) are
# averaged to monthly first
NATIVE_FREQS = ("monthly", "yearly", "half-month", "8-day")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.register_run()
    cfg.save_settings("prep")  # PROC_ROOT/FIG_ROOT/<prep_id>/settings.<prep_id>.json
    cfg.summarize_settings(globals())
    grid = rg.grid_tag(TARGET_RES)

    # (variable, label, settings) of every product
    products = [(v, p, spec) for v in FACTORS for p, spec in PRODUCTS[v].items()]
    for v, p, spec in products:
        if spec["source"] == "ilamb" and p not in il.PRODUCTS[v]:
            raise ValueError(f"prep.obs.{v}_products: {p!r} has source ilamb but is not an ILAMB {v} product "
                             f"({list(il.PRODUCTS[v])})")
        if spec["source"] == "native" and spec["freq"] not in NATIVE_FREQS:
            raise ValueError(f"prep.obs.{v}_products: {p!r} has freq {spec['freq']!r}; native files must be "
                             f"one of {NATIVE_FREQS}")

    # ------------------------------------------------------------------
    # Steps 1-8, once per product: annual means on the target grid
    # ------------------------------------------------------------------
    ann = {v: {} for v in FACTORS}  # {variable: {product: (year, lat, lon)}}
    native_res = {}  # {(variable, product): native resolution}
    for v, p, spec in products:
        label = f"{v}/{p}"
        print(f"\n=== {label} ===")
        from_ilamb = spec["source"] == "ilamb"

        # 1-2. Monthly (or yearly) native field over TIME_SLICE
        try:
            if from_ilamb:
                da = il.load_native(v, p, TIME_SLICE)
            else:
                da = lo.load_obs(spec["dataset"], spec["variable"], TIME_SLICE, version=spec["version"],
                                 freq=spec["freq"])
        except FileNotFoundError as err:
            print(f"{err}; skipping {label}")
            continue
        # Composites -> monthly means. Half-monthly (native GIMMS LAI4g): the equal-weight mean of each month's
        # valid halves. 8-day (HiQ-LAI): the equal-weight mean of the valid composites that start in the month
        if not from_ilamb and spec["freq"] == "half-month":
            da = temporal.half_monthly_to_monthly(da)
        elif not from_ilamb and spec["freq"] == "8-day":
            da = temporal.eight_day_to_monthly(da)

        # 3. ET, P and Rn in W/m2, keyed on the file's units attr (kg m-2 s-1, W m-2, MJ m-2 day-1, or a
        # water depth per hour, day, month or year such as the monthly totals of PML and the mm/day of GPCP)
        da = da.assign_attrs(units="m2/m2") if v == "lai" else units.flux_to_wm2(da)

        # 4. Annual means over complete years: ET, P and Rn need all 12 months, a missing LAI month
        # counts as 0. A yearly total (native files only) is already the annual mean
        if not from_ilamb and spec["freq"] == "yearly":
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

        # 11b. Annual evaporation ratio ET/P (always saved; binned with bin_variable: et_pr), with the annual P of the
        # combination's own P product, the one in its AI, over the same shared years
        data = {"et": fields["et"], **clim, "ai": ai}
        data["et_pr"] = binning.evaporation_ratio(fields["et"], fields["pr"], label=f"{cid}/et_pr")

        # The combination is a source with one member
        combos.append(xr.Dataset(data).expand_dims(member=[cid]).assign_coords(
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
        "family": FAMILY, "dataset": "observational products (ILAMB and native files)",
        "time_period": temporal.period_str(inputs.year.values), "grid": grid, "prep_id": cfg.PREP_ID,
        # Products in at least one saved combination
        "native_res": "; ".join(
            f"{v}/{p}: {res}" for (v, p), res in native_res.items() if p in inputs[f"{v}_product"].values
        ),
        "na_thresh": cfg.NA_THRESH, "min_annual_et": cfg.MIN_ANNUAL_ET,
        "min_years": MIN_YEARS,
        "land_mask": "Natural Earth land, without Greenland/Iceland",
    }
    fout = cfg.inputs_path(FAMILY)
    fout.parent.mkdir(parents=True, exist_ok=True)
    inputs.to_netcdf(fout)
    print(f"{fout}")

    # Diagnostics: climatology maps and LAI/AI histograms
    fig_dir = cfg.output_dir(cfg.FIG_ROOT, FAMILY, "clim", cfg.PREP_ID)
    plotting.save_clim_diagnostics(inputs, fig_dir, FAMILY, cfg.PREP_ID, title="obs")


if __name__ == "__main__":
    main()
