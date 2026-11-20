"""
prep_lsmobs.py
==============
Binning inputs of the land-model and reanalysis ET products ("lsmobs"): steps
1-11 of docs/binning-workflow.md, written out in `main()`. Steps 1-8 run once
per field, steps 9-11 once per (ET, LAI, P, Rn) combination. Every combination
is one source. make_edges.py and bin_all.py then use the saved inputs.

Products
--------
ET_PRODUCTS (prep.lsmobs.et_products of config/pipeline.yml)

LAI, P and Rn: SHARED_PRODUCTS (prep.lsmobs.<v>_products), each set as the
products of prep_obs.py (ILAMB or native files) and combined with every ET
product, or None. A null setting with an ET product that has no such field of
its own raises an error before anything is loaded.

Native land mask: Natural Earth land, as for obs, without Greenland and Iceland.
A field without data in TIME_SLICE is skipped.

Each combination uses the complete years shared by its four fields inside
TIME_SLICE, and is skipped if there are fewer than MIN_YEARS.

Outputs
-------
<prep_id> : PREP_ID of the prep settings (etunc.config; config/runs.csv)

PROC_ROOT/<prep_id>/lsmobs/inputs/lsmobs.inputs.<prep_id>.nc
    et (member, year, lat, lon), NaN outside a combination's years; lai, pr, rn, ai (member, lat, lon);
    and et_pr (member, year, lat, lon), ET/P with the combination's P;
    `member` labelled with the ET product if its LAI, P and Rn are its own, else "<et>-<lai>-<pr>-<rn>"
    (an own field named by its ET product), with coords `source` (= member), `et_product`, `lai_product`,
    `pr_product`, `rn_product`, `n_years` and `time_period`
FIG_ROOT/<prep_id>/lsmobs/clim/lsmobs.<var>.map.<prep_id>.png        var in et, et_pr, lai, pr, rn, ai
    mean, standard deviation and range of the climatology over the combinations
FIG_ROOT/<prep_id>/lsmobs/clim/lsmobs.<var>.map_members.<prep_id>.png    var in lai, pr, rn
    the climatology of every member, on one color scale (more than one member only)
FIG_ROOT/<prep_id>/lsmobs/clim/lsmobs.{lai,ai}_clim.hist.<prep_id>.png
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
import etunc.load.era5 as le
import etunc.load.ilamb as il
import etunc.load.obs as lo

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS. Will affect performance.", category=UserWarning)

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

FAMILY = "lsmobs"

TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES
MIN_YEARS = cfg.MIN_YEARS  # minimum complete years shared by the four fields of a combination

FACTORS = ("et", "lai", "pr", "rn")  # order of the fields in a combination

# ET products: prep.lsmobs.et_products of config/pipeline.yml, {label: {dataset, version, variable, freq, source}}
ET_PRODUCTS = cfg.LSMOBS_ET_PRODUCTS
# LAI, P and Rn products (as prep.obs), or None: each ET product's own field (OWN_FIELDS)
SHARED_PRODUCTS = {
    "lai": cfg.LSMOBS_LAI_PRODUCTS,
    "pr": cfg.LSMOBS_PR_PRODUCTS,
    "rn": cfg.LSMOBS_RN_PRODUCTS,
}
# Native variables of each dataset's own P and Rn (combined in `load_own`)
# Each product uses the same LAI product as obs
OWN_FIELDS = {
    "era5-land": {"pr": ("tp",), "rn": ("ssr", "str")},
    "gldas": {"pr": ("Rainf_f_tavg",), "rn": ("Swnet_tavg", "Lwnet_tavg")},
    "era5": {"pr": ("mtpr",), "rn": ("msnswrf", "msnlwrf")},
}


# ------------------------------------------------------------------
# Steps 1-3: monthly native fields in W/m2 (LAI in m2/m2)
# ------------------------------------------------------------------

def load_et(spec: dict) -> xr.DataArray:
    """ET of an lsmobs ET product over TIME_SLICE, in W/m2 (positive upward)."""
    dataset, var = spec["dataset"], spec["variable"]
    if dataset == "era5-land":
        # Mean daily evaporation [m of water/day, positive downward] -> W/m2 (positive upward)
        return units.convert_units(var, le.load_era5_land(var, TIME_SLICE))
    if dataset == "era5":
        # Mean rate [kg m-2 s-1, positive downward]; convert_units also flips the sign of mer
        return units.convert_units(var, le.load_native([var], TIME_SLICE)[var])
    # Native files (GLDAS: kg m-2 s-1), keyed on the units attr
    return units.flux_to_wm2(lo.load_obs(dataset, var, TIME_SLICE, version=spec["version"], freq=spec["freq"]))


def load_own(v: str, spec: dict) -> xr.DataArray:
    """The own LAI, P or Rn (`v`) of the dataset of the ET product `spec`, over TIME_SLICE (OWN_FIELDS)."""
    dataset = spec["dataset"]
    names = OWN_FIELDS[dataset][v]
    if dataset == "era5-land":
        f = {n: le.load_era5_land(n, TIME_SLICE) for n in names}
        if v == "lai":
            # Grid-box LAI: the LAI of the high and low vegetation, weighted by their fractional cover
            cvh, cvl = le.load_era5_land_invariant("cvh"), le.load_era5_land_invariant("cvl")
            lai = f["lai_hv"] * cvh.assign_coords(lat=f["lai_hv"].lat, lon=f["lai_hv"].lon) \
                + f["lai_lv"] * cvl.assign_coords(lat=f["lai_lv"].lat, lon=f["lai_lv"].lon)
            return lai.rename("lai").assign_attrs(units="m2/m2")
        if v == "pr":
            return units.convert_units("tp", f["tp"])  # m of water/day -> W/m2
        return units.net_radiation_era5_land(f["ssr"], f["str"])
    if dataset == "gldas":
        f = {n: lo.load_obs("gldas", n, TIME_SLICE, version=spec["version"], freq=spec["freq"]) for n in names}
        if v == "pr":
            return units.flux_to_wm2(f["Rainf_f_tavg"])  # kg m-2 s-1
        return units.net_radiation_gldas(f["Swnet_tavg"], f["Lwnet_tavg"])
    # era5
    f = le.load_native(names, TIME_SLICE)
    if v == "pr":
        return units.convert_units("mtpr", f["mtpr"])
    return units.net_radiation_era5(f["msnswrf"], f["msnlwrf"])


def load_shared(v: str, label: str, spec: dict) -> xr.DataArray:
    """A LAI, P or Rn product of SHARED_PRODUCTS over TIME_SLICE, read as in prep_obs.py."""
    if spec["source"] == "ilamb":
        da = il.load_native(v, label, TIME_SLICE)
    else:
        da = lo.load_obs(spec["dataset"], spec["variable"], TIME_SLICE, version=spec["version"], freq=spec["freq"])
        if spec["freq"] == "half-month":  # GIMMS LAI4g: mean of each month's valid halves
            da = temporal.half_monthly_to_monthly(da)
    return da.assign_attrs(units="m2/m2") if v == "lai" else units.flux_to_wm2(da)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.register_run()
    cfg.save_settings("prep")  # PROC_ROOT/FIG_ROOT/<prep_id>/settings.<prep_id>.json
    cfg.summarize_settings(globals())
    grid = rg.grid_tag(TARGET_RES)

    # Every own field asked for must exist, and every shared ILAMB product, before anything is loaded
    for v, shared in SHARED_PRODUCTS.items():
        if shared is None:
            for p, spec in ET_PRODUCTS.items():
                if v not in OWN_FIELDS.get(spec["dataset"], {}):
                    raise ValueError(f"prep.lsmobs.{v}_products is null, but {p} (dataset {spec['dataset']}) has "
                                     f"no {v} of its own: set prep.lsmobs.{v}_products")
        else:
            for p, spec in shared.items():
                if spec["source"] == "ilamb" and p not in il.PRODUCTS[v]:
                    raise ValueError(f"prep.lsmobs.{v}_products: {p!r} has source ilamb but is not an ILAMB {v} "
                                     f"product ({list(il.PRODUCTS[v])})")
                if p in ET_PRODUCTS:
                    raise ValueError(f"prep.lsmobs.{v}_products: {p!r} is also an ET product label")

    # (variable, label, settings, kind) of every field: the ET products; for LAI, P and Rn, the shared products,
    # or each ET product's own field, labelled with the ET product
    fields = [("et", p, spec, "et") for p, spec in ET_PRODUCTS.items()]
    for v, shared in SHARED_PRODUCTS.items():
        if shared is None:
            fields += [(v, p, spec, "own") for p, spec in ET_PRODUCTS.items()]
        else:
            fields += [(v, p, spec, "shared") for p, spec in shared.items()]

    # ------------------------------------------------------------------
    # Steps 1-8, once per field: annual means on the target grid
    # ------------------------------------------------------------------
    ann = {v: {} for v in FACTORS}  # {variable: {label: (year, lat, lon)}}
    native_res = {}  # {(variable, label): native resolution}
    for v, p, spec, kind in fields:
        label = f"{v}/{p}" + (" (own)" if kind == "own" else "")
        print(f"\n=== {label} ===")

        # 1-3. Monthly native field over TIME_SLICE; ET, P and Rn in W/m2
        try:
            if kind == "et":
                da = load_et(spec)
            elif kind == "own":
                da = load_own(v, spec)
            else:
                da = load_shared(v, p, spec)
        except FileNotFoundError as err:
            print(f"{err}; skipping {label}")
            continue

        # 4. Annual means over complete years: ET, P and Rn need all 12 months, a missing LAI month
        # counts as 0. A yearly total (native files only) is already the annual mean
        if kind == "shared" and spec["source"] == "native" and spec["freq"] == "yearly":
            a = temporal.yearly_to_annual(da).load()
        else:
            a = temporal.aggregate(da, "year", require_all_months=(v != "lai"), label=label)
        if a.sizes["year"] == 0:
            print(f"{label}: no complete year in {TIME_SLICE}, skipping")
            continue

        # 5. Native land mask, without Greenland/Iceland: Natural Earth land, as for obs
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
    # Steps 9-11, once per (et, lai, pr, rn) combination
    # ------------------------------------------------------------------
    print("\n=== Combinations ===")
    combos = []
    n_all = 0
    own = all(SHARED_PRODUCTS[v] is None for v in ("lai", "pr", "rn"))  # member named by its ET product alone
    for et_label in ann["et"]:
        # LAI, P and Rn of this ET product: its own field, or every shared product
        candidates = [[et_label] if SHARED_PRODUCTS[v] is None else list(ann[v]) for v in ("lai", "pr", "rn")]
        for combo in itertools.product([et_label], *candidates):
            n_all += 1
            cid = et_label if own else "-".join(combo)
            if missing := [v for v, p in zip(FACTORS, combo) if p not in ann[v]]:
                print(f"{cid}: no {missing} (skipped above), skipping")
                continue
            data = {v: ann[v][p] for v, p in zip(FACTORS, combo)}

            # 9. Complete years shared by the four fields
            years = temporal.shared_years(*data.values())
            if len(years) < MIN_YEARS:
                print(f"{cid}: {len(years)} shared years < MIN_YEARS = {MIN_YEARS}, skipping")
                continue
            data = {v: da.sel(year=years) for v, da in data.items()}

            # 10. Climatological LAI, P and Rn over those years
            clim = {k: temporal.aggregate(data[k], "clim", label=f"{cid}/{k}") for k in ("lai", "pr", "rn")}

            # 11. Aridity index from the climatological Rn and P
            ai = edges.compute_aridity_index(clim["pr"], clim["rn"])

            # 11b. Annual evaporation ratio ET/P (always saved; binned with bin_variable: et_pr), with the annual
            # P of the combination, the one in its AI, over the same shared years
            out = {"et": data["et"], **clim, "ai": ai}
            out["et_pr"] = binning.evaporation_ratio(data["et"], data["pr"], label=f"{cid}/et_pr")

            # The combination is a source with one member
            combos.append(xr.Dataset(out).expand_dims(member=[cid]).assign_coords(
                source=("member", [cid]),
                **{f"{v}_product": ("member", [p]) for v, p in zip(FACTORS, combo)},
                n_years=("member", [len(years)]),
                time_period=("member", [temporal.period_str(years)]),
            ))
    if not combos:
        raise RuntimeError(f"No combination has at least {MIN_YEARS} shared years.")
    print(f"{len(combos)} of {n_all} combinations have >= {MIN_YEARS} shared years")

    # ET is NaN in the years outside a combination's shared years
    inputs = xr.concat(combos, dim="member", join="outer", combine_attrs="drop_conflicts")
    inputs.attrs = {
        "family": FAMILY,
        "dataset": "land-model and reanalysis ET products, with their own LAI, P and Rn or shared products",
        "own_fields": "; ".join(f"{v}: {'own' if s is None else 'shared'}" for v, s in SHARED_PRODUCTS.items()),
        "time_period": temporal.period_str(inputs.year.values), "grid": grid, "prep_id": cfg.PREP_ID,
        # Fields in at least one saved combination
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
    plotting.save_clim_diagnostics(inputs, fig_dir, FAMILY, cfg.PREP_ID, title="lsmobs")


if __name__ == "__main__":
    main()
