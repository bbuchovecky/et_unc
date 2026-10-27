"""
prep_ippe.py
============
Binning inputs of the CESM2 offline (CLM5) perturbed-parameter ensemble (IHIST
PPE, "ippe"): steps 1-11 of docs/binning-workflow.md, written out in `main()`.
Every member (one parameter perturbation, the same as in the FHIST PPE) is one
source. make_edges.py and bin_all.py then use the saved inputs.

The IHIST PPE runs CLM offline, forced by one coupler history (CPLHIST) for
every member. So P and the downwelling radiation are the same in all members,
and the AI differs between members only through the land's own Rn (albedo,
surface temperature). There is no atmosphere output, so every variable is from
"lnd": ET is EFLX_LH_TOT (total latent heat flux, W/m2), LAI is TLAI, P is
RAIN + SNOW, and Rn = FSA - FIRA, the land's net radiation (the FHIST PPE uses
the CAM gridcell mean FSNS - FLNS instead, which differs in coastal cells).
The native land mask is the LANDFRAC > LF_THRESH of the FHIST PPE grid file
(the same f19 grid and land fraction), without Greenland and Iceland.

Outputs
-------
<grid>   : grid tag of TARGET_RES, e.g. "1deg"
<period> : YYYYMM-YYYYMM of TIME_SLICE

PROC_ROOT/ippe/inputs/<period>/ippe.inputs.<grid>.<period>.nc
    et (member, year, lat, lon); lai, pr, rn, ai (member, lat, lon); with BIN_VARIABLE = "et_pr",
    also et_pr (member, year, lat, lon); `member` labelled by parameter
    name (e.g. "2.dleaf.max"), `source` = member
FIG_ROOT/ippe/clim/<period>/ippe.<var>.<grid>.map.<period>.png        var in et, lai, pr, rn, ai (and et_pr)
    member mean, standard deviation and range of the climatology
FIG_ROOT/ippe/clim/<period>/ippe.<var>.<grid>.map_members.<period>.png    var in lai, pr, rn
    the climatology of every member, on one color scale (more than one member only)
FIG_ROOT/ippe/clim/<period>/ippe.{lai,ai}_clim.<grid>.hist.<period>.png
    histograms of climatological LAI and AI, one line per member
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
import etunc.load.cesm as lc

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS. Will affect performance.", category=UserWarning)

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

FAMILY = "ippe"

TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES
BIN_VARIABLE = cfg.BIN_VARIABLE  # "et_pr" also saves the annual evaporation ratio ET/P (binned by bin_all.py)

FREQUENCY = "month_1"
STREAM = "h0"
# {CESM variable: component}; offline CLM, so all "lnd"
VARIABLES = {"EFLX_LH_TOT": "lnd", "TLAI": "lnd", "RAIN": "lnd", "SNOW": "lnd", "FSA": "lnd", "FIRA": "lnd"}

# Members (0-28): None = every member not in DROP_OUTLIERS, or an explicit list (the same as the FHIST PPE)
MEMBERS: list[int] | None = None
DROP_OUTLIERS = (13, 28)  # unreasonable LAI and ET in the FHIST PPE


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.summarize_settings(globals())
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)

    # 1-2. Monthly native fields over TIME_SLICE, and the native land fraction
    print("=== 1. Load IHIST PPE ===")
    raw = lc.load_native(
        "ippe", MEMBERS, TIME_SLICE, variables=VARIABLES, frequency=FREQUENCY, stream=STREAM,
        drop_outliers=DROP_OUTLIERS,
    )
    landfrac, src_grid = lc.load_land_fraction("ippe")

    # 3. ET, P and Rn in W/m2
    monthly = {
        "et": raw["EFLX_LH_TOT"].assign_attrs(units="W/m2"),     # latent heat flux, already W/m2
        "lai": raw["TLAI"].assign_attrs(units="m2/m2"),
        # rain + snow (the atmospheric forcing, after CLM's rain/snow repartitioning), mm/s x L
        "pr": (units.convert_units("RAIN", raw["RAIN"]) + units.convert_units("SNOW", raw["SNOW"])).assign_attrs(units="W/m2"),
        "rn": units.net_radiation_clm(raw["FSA"], raw["FIRA"]),  # land-only net radiation
    }

    # 4. Annual means over complete years: ET, P and Rn need all 12 months, a missing LAI month counts as 0
    print("\n=== 4. Annual means ===")
    ann = {k: temporal.aggregate(da, "year", require_all_months=(k != "lai"), label=k) for k, da in monthly.items()}

    # 5. Native land mask: LANDFRAC > LF_THRESH, without Greenland/Iceland
    land = rg.landfrac_mask(landfrac, cfg.LF_THRESH)
    ann = {k: rg.mask_to_land(da, land, k) for k, da in ann.items()}

    # 6. ET outlier filter (on land, before regridding, so an outlier does not enter its target cell)
    ann["et"] = binning.drop_et_outliers(ann["et"], cfg.MIN_ANNUAL_ET)

    # 7. Conservative regridding onto the target grid (NaN where more than NA_THRESH is missing)
    regridder = rg.bounded_conservative_regridder(src_grid, TARGET_RES)
    ann = {k: rg.on_grid(rg.regrid_with_na_thresh(da, regridder, cfg.NA_THRESH), TARGET_RES, k) for k, da in ann.items()}

    # 8. Domain without Antarctica
    ann = {k: da.sel(lat=cfg.LAT_BNDS) for k, da in ann.items()}

    # 9. Complete years shared by ET, LAI, P and Rn
    years = temporal.shared_years(*ann.values())
    if len(years) < cfg.MIN_YEARS:
        raise RuntimeError(f"Only {len(years)} shared years, fewer than MIN_YEARS = {cfg.MIN_YEARS}")
    ann = {k: da.sel(year=years) for k, da in ann.items()}

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

    # Every member is one source, labelled by its parameter name
    inputs = xr.Dataset(data)
    names = inputs["member_id"].values.tolist()
    inputs = inputs.drop_vars("member_id").assign_coords(member=names, source=("member", names))
    inputs.attrs = {
        "family": FAMILY, "dataset": "CESM2 offline (CLM5) perturbed-parameter ensemble",
        "time_period": temporal.period_str(years), "grid": grid, "native_res": rg.native_res(src_grid),
        "na_thresh": cfg.NA_THRESH, "lf_thresh": cfg.LF_THRESH, "min_annual_et": cfg.MIN_ANNUAL_ET,
        "land_mask": "LANDFRAC > lf_thresh, without Greenland/Iceland",
    }
    fout = cfg.inputs_path(FAMILY, grid, period)
    fout.parent.mkdir(parents=True, exist_ok=True)
    inputs.to_netcdf(fout)
    print(f"\n{len(names)} members, {inputs.attrs['time_period']}\n{fout}")

    # Diagnostics: climatology maps and LAI/AI histograms
    fig_dir = cfg.output_dir(cfg.FIG_ROOT, FAMILY, "clim", period)
    plotting.save_clim_diagnostics(inputs, fig_dir, FAMILY, grid, period, title="IHIST PPE")


if __name__ == "__main__":
    main()
