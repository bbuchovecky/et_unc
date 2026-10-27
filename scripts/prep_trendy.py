"""
prep_trendy.py
==============
Binning inputs of the TRENDY (GCB2025) DGVMs in one experiment: steps 1-11 of
docs/binning-workflow.md, written out in `main()` for one model at a time.
Every model is one source with one member. make_edges.py and bin_all.py then
use the saved inputs. Run once per EXPERIMENT; S2 and S3 are separate families
("trendy_s2", "trendy_s3"), so a model never counts twice in the pooled edges.

The TRENDY models are offline land models driven by the same CRU-JRA v3
climate forcing. ET is evapotrans and LAI is lai, from each model; P is the
CRU-JRA precipitation forcing itself, the same field for every model, so
the models' own pr outputs (missing in some, 12x too small in ED) are not
used. The native land mask is Natural Earth land (as for the obs), since few
models provide a land fraction and some fill the ocean with 0, without
Greenland and Iceland; annual LAI missing on land is set to 0.

Rn is the ERA5 net surface radiation (msnswrf + msnlwrf, as in prep_era5.py),
the same field for every model: the models output at most the downward
shortwave (rsds), not the upward terms of the net radiation. Its native land
mask is ERA5 LANDFRAC > LF_THRESH, as in prep_era5.py. So the AI is the same
for every model and the models differ only in ET and LAI. TRENDY is not in
cfg.FAMILIES, so make_edges.py and the bin scripts do not read the inputs.

Models: every model directory with VARIABLES for EXPERIMENT (ISAM has no lai),
minus OMIT_MODELS.

Outputs
-------
<family> : trendy_s2 or trendy_s3 (EXPERIMENT)
<grid>   : grid tag of TARGET_RES, e.g. "1deg"
<period> : YYYYMM-YYYYMM of TIME_SLICE

PROC_ROOT/<family>/inputs/<period>/<family>.inputs.<grid>.<period>.nc
    et (member, year, lat, lon); lai, pr, rn, ai (member, lat, lon); with BIN_VARIABLE = "et_pr",
    also et_pr (member, year, lat, lon); `member` and `source` = model name
FIG_ROOT/<family>/clim/<period>/<family>.<var>.<grid>.map.<period>.png        var in et, lai, pr, rn, ai (and et_pr)
    mean, standard deviation and range of the climatology over all models
FIG_ROOT/<family>/clim/<period>/<family>.<var>.<grid>.map_members.<period>.png     var in lai, pr, rn
    the climatology of every model, on one color scale
FIG_ROOT/<family>/clim/<period>/<family>.{lai,ai}_clim.<grid>.hist.<period>.png
    histograms of climatological LAI and AI, one line per model
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
import etunc.load.trendy as lt

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS. Will affect performance.", category=UserWarning)

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

EXPERIMENT = "S3"  # "S2": climate + CO2; "S3": climate + CO2 + land use change
FAMILY = f"trendy_{EXPERIMENT.lower()}"

TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES
BIN_VARIABLE = cfg.BIN_VARIABLE  # "et_pr" also saves the annual evaporation ratio ET/P (binned by bin_all.py)

# Model variables (P is the CRU-JRA forcing, Rn is ERA5's)
VARIABLES = ["evapotrans", "lai"]
# GDEX "meanflux" parameter codes of the ERA5 net radiation: net shortwave and net longwave
ERA5_RN_CODES = ["msnswrf", "msnlwrf"]

# None = every model directory with all VARIABLES, minus OMIT_MODELS
MODELS: list[str] | None = None
OMIT_MODELS: dict[str, str] = {
    "CARDAMOM": "data-assimilation system covering 2003-2024 only",
    # Gridcell LAI far above observed values (90th percentile 8-11, max 20-41 m2/m2), likely a sum
    # over PFTs not weighted by their cover
    "LPJml": "implausible LAI",
    "LPJwsl": "implausible LAI",
    "LPX-Bern": "implausible LAI",
    "VISIT": "implausible LAI",
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.summarize_settings(globals())
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)

    # Models with every variable, minus OMIT_MODELS
    print(f"=== Models ({EXPERIMENT}) ===")
    models = []
    for model in MODELS or lt.models():
        if model in OMIT_MODELS:
            print(f"{model:10}: omitted ({OMIT_MODELS[model]})")
        elif missing := lt.missing_variables(model, EXPERIMENT, VARIABLES):
            print(f"{model:10}: no {missing} in {EXPERIMENT}, skipping")
        else:
            models.append(model)
            print(f"{model:10}: ok")

    # P: the CRU-JRA precipitation forcing, the same for every model, so steps 1-8 run once, on its own grid
    print("\n=== CRU-JRA precipitation (forcing of every model) ===")
    # 1. Monthly totals [mm/month] over TIME_SLICE
    pr = lt.load_forcing("pre", TIME_SLICE)
    # 3. mm/month -> kg m-2 s-1 -> W/m2
    pr = units.convert_units("pr", units.accumulation_to_flux(pr))
    # 4. Annual means over complete years (all 12 months needed)
    pr = temporal.aggregate(pr, "year", label="forcing/pr")
    # 5. Native land mask: Natural Earth land, without Greenland/Iceland
    pr_grid = rg.bounded_src_grid(pr)
    pr = rg.mask_to_land(pr, rg.landne_mask(pr_grid), "forcing/pr")
    # 7. Conservative regridding onto the target grid (NaN where more than NA_THRESH is missing)
    regridder = rg.bounded_conservative_regridder(pr_grid, TARGET_RES)
    pr = rg.on_grid(rg.regrid_with_na_thresh(pr, regridder, cfg.NA_THRESH), TARGET_RES, "forcing/pr")
    # 8. Domain without Antarctica
    pr = pr.sel(lat=cfg.LAT_BNDS)

    # Rn: the ERA5 net surface radiation, the same for every model, so steps 1-8 run once, on the ERA5 grid
    print("\n=== ERA5 net radiation (the same for every model) ===")
    # 1-2. Monthly mean net fluxes [W/m2, positive downward] over TIME_SLICE, and the ERA5 land fraction
    era5 = le.load_native(ERA5_RN_CODES, TIME_SLICE)
    landfrac, era5_grid = le.load_land_fraction()
    # 3. Rn = net shortwave + net longwave [W/m2]
    rn = units.net_radiation_era5(era5["msnswrf"], era5["msnlwrf"])
    # 4. Annual means over complete years (all 12 months needed)
    rn = temporal.aggregate(rn, "year", label="era5/rn")
    # 5. Native land mask: ERA5 LANDFRAC > LF_THRESH (as in prep_era5.py), without Greenland/Iceland
    rn = rg.mask_to_land(rn, rg.landfrac_mask(landfrac, cfg.LF_THRESH), "era5/rn")
    # 7. Conservative regridding onto the target grid (NaN where more than NA_THRESH is missing)
    regridder = rg.bounded_conservative_regridder(era5_grid, TARGET_RES)
    rn = rg.on_grid(rg.regrid_with_na_thresh(rn, regridder, cfg.NA_THRESH), TARGET_RES, "era5/rn")
    # 8. Domain without Antarctica
    rn = rn.sel(lat=cfg.LAT_BNDS)

    per_model = []
    native_res = {}  # {model: native resolution}, of the models that are saved
    for model in models:
        print(f"\n=== {model} ===")
        # 1. Monthly native fields over TIME_SLICE. A model whose files are inconsistent is skipped
        try:
            raw = lt.load_native(model, EXPERIMENT, TIME_SLICE, VARIABLES)
        except ValueError as err:
            print(f"{model}: {err}, skipping")
            continue
        src_grid = rg.bounded_src_grid(raw["evapotrans"])

        # 3. ET in W/m2 (P is the forcing's and Rn ERA5's, above)
        monthly = {
            "et": units.convert_units("et", raw["evapotrans"]),  # kg m-2 s-1 x L
            "lai": raw["lai"].assign_attrs(units="m2/m2"),
        }

        # 4. Annual means over complete years: ET needs all 12 months, a missing LAI month counts as 0
        ann = {
            k: temporal.aggregate(da, "year", require_all_months=(k != "lai"), label=f"{model}/{k}")
            for k, da in monthly.items()
        }

        # 5. Native land mask: Natural Earth land, without Greenland/Iceland
        land = rg.landne_mask(src_grid)
        ann = {k: rg.mask_to_land(da, land, f"{model}/{k}") for k, da in ann.items()}

        # LAI missing on land (cells outside the model's own land mask, unvegetated cells) is set to 0;
        # masking again after the fill sets the ocean back to NaN
        n_nan = int(ann["lai"].isnull().sum())
        ann["lai"] = rg.mask_to_land(ann["lai"].fillna(0), land, f"{model}/lai")
        print(f"{model}/lai: {n_nan - int(ann['lai'].isnull().sum())} land gridcell-years with missing LAI set to 0")

        # 6. ET outlier filter (on land, before regridding, so an outlier does not enter its target cell)
        ann["et"] = binning.drop_et_outliers(ann["et"], cfg.MIN_ANNUAL_ET, label=f"{model}/et")

        # 7. Conservative regridding onto the target grid (NaN where more than NA_THRESH is missing)
        regridder = rg.bounded_conservative_regridder(src_grid, TARGET_RES)
        ann = {
            k: rg.on_grid(rg.regrid_with_na_thresh(da, regridder, cfg.NA_THRESH), TARGET_RES, f"{model}/{k}")
            for k, da in ann.items()
        }

        # 8. Domain without Antarctica
        ann = {k: da.sel(lat=cfg.LAT_BNDS) for k, da in ann.items()}

        # 9. Complete years shared by ET, LAI, the forcing P and the ERA5 Rn
        years = temporal.shared_years(*ann.values(), pr, rn)
        if len(years) < cfg.MIN_YEARS:
            print(f"{model}: only {len(years)} shared years, fewer than MIN_YEARS = {cfg.MIN_YEARS}, skipping")
            continue
        ann = {k: da.sel(year=years) for k, da in ann.items()}
        ann["pr"] = pr.sel(year=years)
        ann["rn"] = rn.sel(year=years)

        # 10. Climatological LAI, P and Rn over those years
        clim = {k: temporal.aggregate(ann[k], "clim", label=f"{model}/{k}") for k in ("lai", "pr", "rn")}

        # 11. Aridity index from the climatological Rn and P
        ai = edges.compute_aridity_index(clim["pr"], clim["rn"])

        # 11b. Annual evaporation ratio ET/P (BIN_VARIABLE = "et_pr" only), with the annual forcing P
        data = {"et": ann["et"], **clim, "ai": ai}
        if BIN_VARIABLE == "et_pr":
            data["et_pr"] = binning.evaporation_ratio(ann["et"], ann["pr"], label=f"{model}/et_pr")

        # The model is the source of its one member
        native_res[model] = rg.native_res(src_grid)
        per_model.append(xr.Dataset(data).expand_dims(member=[model]).assign_coords(source=("member", [model])))

    if not per_model:
        raise RuntimeError("No model could be loaded.")
    inputs = xr.concat(per_model, dim="member", join="outer", combine_attrs="drop_conflicts")
    inputs.attrs = {
        "family": FAMILY, "dataset": f"TRENDY (GCB2025) {EXPERIMENT}", "experiment": EXPERIMENT,
        "time_period": temporal.period_str(inputs.year.values), "grid": grid,
        "native_res": "; ".join(f"{m}: {res}" for m, res in native_res.items()),
        "na_thresh": cfg.NA_THRESH, "lf_thresh": cfg.LF_THRESH, "min_annual_et": cfg.MIN_ANNUAL_ET,
        "land_mask": "models: Natural Earth land; ERA5 Rn: LANDFRAC > lf_thresh; both without Greenland/Iceland",
        "pr_source": "CRU-JRA v3.02 precipitation forcing (the same for every model)",
        "rn_source": "ERA5 net surface radiation msnswrf + msnlwrf (the same for every model)",
        "omitted_models": "; ".join(f"{m}: {why}" for m, why in OMIT_MODELS.items()),
    }
    fout = cfg.inputs_path(FAMILY, grid, period)
    fout.parent.mkdir(parents=True, exist_ok=True)
    inputs.to_netcdf(fout)
    print(f"\n{inputs.sizes['member']} models, {inputs.attrs['time_period']}\n{fout}")

    # Diagnostics: climatology maps and LAI/AI histograms
    fig_dir = cfg.output_dir(cfg.FIG_ROOT, FAMILY, "clim", period)
    plotting.save_clim_diagnostics(inputs, fig_dir, FAMILY, grid, period, title=f"TRENDY {EXPERIMENT}")


if __name__ == "__main__":
    main()
