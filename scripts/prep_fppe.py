"""
prep_fppe.py
============
Binning inputs of the CESM2 FHIST perturbed-parameter ensemble (FHIST PPE,
"fppe"): steps 1-11 of docs/binning-workflow.md, written out in `main()`.
Every member (one parameter perturbation) is one source. make_edges.py and
bin_all.py then use the saved inputs.

ET is EFLX_LH_TOT (total latent heat flux, W/m2) and LAI is TLAI from "lnd";
P is PRECT_calculated and Rn = FSNS - FLNS from "atm". Only members with every
variable are used. The native land mask is the grid file's LANDFRAC >
LF_THRESH, without Greenland and Iceland.

Outputs
-------
<grid>   : grid tag of TARGET_RES, e.g. "1deg"
<period> : YYYYMM-YYYYMM of TIME_SLICE

PROC_ROOT/fppe/fppe.inputs.<grid>.<period>.nc
    et (member, year, lat, lon); lai, pr, rn, ai (member, lat, lon); `member` labelled by parameter
    name (e.g. "2.dleaf.max"), `source` = member
FIG_ROOT/fppe/clim/fppe.<var>.<grid>.map.<period>.png        var in et, lai, pr, rn, ai
    member mean, standard deviation and range of the climatology
FIG_ROOT/fppe/clim/fppe.{lai,ai}_clim.<grid>.hist.<period>.png
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

FAMILY = "fppe"
FIG_ROOT = cfg.FIG_ROOT / FAMILY

TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES

FREQUENCY = "month_1"
STREAM = "h0"
# {CESM variable: component}
VARIABLES = {"EFLX_LH_TOT": "lnd", "TLAI": "lnd", "PRECT_calculated": "atm", "FSNS": "atm", "FLNS": "atm"}

# Members (0-28): None = every member not in DROP_OUTLIERS, or an explicit list
MEMBERS: list[int] | None = None
DROP_OUTLIERS = (13, 28)  # unreasonable LAI and ET


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.summarize_settings(globals())
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)

    # 1-2. Monthly native fields over TIME_SLICE, and the native land fraction
    print("=== 1. Load FHIST PPE ===")
    raw = lc.load_native(
        "fppe", MEMBERS, TIME_SLICE, variables=VARIABLES, frequency=FREQUENCY, stream=STREAM,
        drop_outliers=DROP_OUTLIERS,
    )
    landfrac, src_grid = lc.load_land_fraction("fppe")

    # 3. ET, P and Rn in W/m2
    monthly = {
        "et": raw["EFLX_LH_TOT"].assign_attrs(units="W/m2"),     # latent heat flux, already W/m2
        "lai": raw["TLAI"].assign_attrs(units="m2/m2"),
        "pr": units.convert_units("PRECT_calculated", raw["PRECT_calculated"]),  # m/s x L x water density
        "rn": units.net_radiation_cesm(raw["FSNS"], raw["FLNS"]),
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

    # Every member is one source, labelled by its parameter name
    inputs = xr.Dataset({"et": ann["et"], **clim, "ai": ai})
    names = inputs["member_id"].values.tolist()
    inputs = inputs.drop_vars("member_id").assign_coords(member=names, source=("member", names))
    inputs.attrs = {
        "family": FAMILY, "dataset": "CESM2 FHIST perturbed-parameter ensemble",
        "time_period": temporal.period_str(years), "grid": grid, "native_res": rg.native_res(src_grid),
        "na_thresh": cfg.NA_THRESH, "lf_thresh": cfg.LF_THRESH, "min_annual_et": cfg.MIN_ANNUAL_ET,
        "land_mask": "LANDFRAC > lf_thresh, without Greenland/Iceland",
    }
    fout = cfg.inputs_path(FAMILY, grid, period)
    fout.parent.mkdir(parents=True, exist_ok=True)
    inputs.to_netcdf(fout)
    print(f"\n{len(names)} members, {inputs.attrs['time_period']}\n{fout}")

    # Diagnostics: climatology maps and LAI/AI histograms
    plotting.save_clim_diagnostics(inputs, FIG_ROOT / "clim", FAMILY, grid, period, title="FHIST PPE")


if __name__ == "__main__":
    main()
