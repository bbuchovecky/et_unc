"""
prep_cmip.py
============
Binning inputs of the CMIP6 historical simulations: steps 1-11 of
docs/binning-workflow.md, written out in `main()` for one model at a time.
Every model is one source; its members (one by default) are members of that
source. make_edges.py and bin_all.py then use the saved inputs.

Models: every model in CATALOG with all VARIABLES for at least one member and
an sftlf file in FX_CATALOG, minus OMIT_SOURCE_IDS. Members per model from
MEMBER_IDS / DEFAULT_MEMBERS ("top" = one member per model). ET is evspsbl,
LAI is lai, P is pr and Rn = rsds - rsus + rlds - rlus. The native land mask
is the model's sftlf > LF_THRESH, without Greenland and Iceland. A model whose
sftlf is not on its data grid is skipped (MIROC-ES2H).

Outputs
-------
<grid>   : grid tag of TARGET_RES, e.g. "1deg"
<period> : YYYYMM-YYYYMM of TIME_SLICE

PROC_ROOT/cmip/cmip.inputs.<grid>.<period>.nc
    et (member, year, lat, lon); lai, pr, rn, ai (member, lat, lon); `member` labelled
    "<source_id>.<member_id>", with coords `source` (= source_id) and `member_id`
FIG_ROOT/cmip/clim/cmip.<var>.<grid>.map.<period>.png        var in et, lai, pr, rn, ai
    mean, standard deviation and range of the climatology over all members
FIG_ROOT/cmip/clim/cmip.{lai,ai}_clim.<grid>.hist.<period>.png
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
import etunc.load.cmip as cmip
from etunc.load.cmip import CMIPESGFLoader

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS. Will affect performance.", category=UserWarning)

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

FAMILY = "cmip"
CATALOG = cfg.CMIP_CATALOG        # evspsbl, lai, pr, radiative fluxes (on purgeable scratch)
FX_CATALOG = cfg.CMIP_FX_CATALOG  # one sftlf file per model, from any experiment
FIG_ROOT = cfg.FIG_ROOT / FAMILY

TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES

EXPERIMENT_ID = "historical"
VARIABLES = ["evspsbl", "lai", "pr", "rsds", "rsus", "rlds", "rlus"]

# None = every model with all VARIABLES and sftlf, minus OMIT_SOURCE_IDS
SOURCE_IDS: list[str] | None = None
OMIT_SOURCE_IDS: list[str] = []

# Members per model: a list of member_ids, or
#   "top"   : first member sorted by r/i/p/f (one member per model)
#   "max_r" : the largest group of members sharing i/p/f (initial-condition ensemble)
#   "all"   : every member with all VARIABLES
# Models not listed use DEFAULT_MEMBERS.
DEFAULT_MEMBERS: str | list[str] = "top"
MEMBER_IDS: dict[str, str | list[str]] = {
    # "CESM2": ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"],
    # "CanESM5": "max_r",
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.summarize_settings(globals())
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)
    loader = CMIPESGFLoader(CATALOG)

    # Models with every variable and an sftlf file, and their members
    print("=== Models and members ===")
    avail = cmip.available_members(loader.catalog, VARIABLES, EXPERIMENT_ID)
    sftlf = cmip.sftlf_files(FX_CATALOG)
    members = {}
    for sid in sorted(set(SOURCE_IDS or avail) - set(OMIT_SOURCE_IDS)):
        if sid not in avail:
            print(f"{sid:16}: not all of {VARIABLES} available, skipping")
        elif sid not in sftlf:
            print(f"{sid:16}: no sftlf in {FX_CATALOG.name}, skipping")
        elif mids := cmip.select_members(sid, avail[sid], MEMBER_IDS, DEFAULT_MEMBERS):
            members[sid] = mids
            print(f"{sid:16}: {len(mids)} of {len(avail[sid])} members {mids}")

    per_model = []
    for sid, mids in members.items():
        print(f"\n=== {sid} ===")
        # 1-2. Monthly native fields over TIME_SLICE, and the native land fraction. sftlf may come
        # from another experiment, so a model whose sftlf is on another grid is skipped
        try:
            raw = cmip.load_native(loader, sid, mids, TIME_SLICE, variables=VARIABLES, experiment_id=EXPERIMENT_ID)
            landfrac, src_grid = cmip.load_land_fraction(sftlf[sid])
            rg.check_same_grid(raw["evspsbl"], landfrac, f"{sid} sftlf")
        except ValueError as err:
            print(f"{sid}: {err}, skipping")
            continue

        # 3. ET, P and Rn in W/m2
        monthly = {
            "et": units.convert_units("evspsbl", raw["evspsbl"]),   # kg m-2 s-1 x L
            "lai": raw["lai"].assign_attrs(units="m2/m2"),
            "pr": units.convert_units("pr", raw["pr"]),             # kg m-2 s-1 x L
            "rn": units.net_radiation_cmip(raw["rsds"], raw["rsus"], raw["rlds"], raw["rlus"]),
        }

        # 4. Annual means over complete years: ET, P and Rn need all 12 months, a missing LAI month counts as 0
        ann = {
            k: temporal.aggregate(da, "year", require_all_months=(k != "lai"), label=f"{sid}/{k}")
            for k, da in monthly.items()
        }

        # 5. Native land mask: sftlf > LF_THRESH, without Greenland/Iceland
        land = rg.landfrac_mask(landfrac, cfg.LF_THRESH)
        ann = {k: rg.mask_to_land(da, land, f"{sid}/{k}") for k, da in ann.items()}

        # 6. ET outlier filter (on land, before regridding, so an outlier does not enter its target cell)
        ann["et"] = binning.drop_et_outliers(ann["et"], cfg.MIN_ANNUAL_ET, label=f"{sid}/et")

        # 7. Conservative regridding onto the target grid (NaN where more than NA_THRESH is missing)
        regridder = rg.bounded_conservative_regridder(src_grid, TARGET_RES)
        ann = {
            k: rg.on_grid(rg.regrid_with_na_thresh(da, regridder, cfg.NA_THRESH), TARGET_RES, f"{sid}/{k}")
            for k, da in ann.items()
        }

        # 8. Domain without Antarctica
        ann = {k: da.sel(lat=cfg.LAT_BNDS) for k, da in ann.items()}

        # 9. Complete years shared by ET, LAI, P and Rn
        years = temporal.shared_years(*ann.values())
        if len(years) < cfg.MIN_YEARS:
            print(f"{sid}: only {len(years)} shared years, fewer than MIN_YEARS = {cfg.MIN_YEARS}, skipping")
            continue
        ann = {k: da.sel(year=years) for k, da in ann.items()}

        # 10. Climatological LAI, P and Rn over those years
        clim = {k: temporal.aggregate(ann[k], "clim", label=f"{sid}/{k}") for k in ("lai", "pr", "rn")}

        # 11. Aridity index from the climatological Rn and P
        ai = edges.compute_aridity_index(clim["pr"], clim["rn"])

        # The model is the source of its members
        inputs = xr.Dataset({"et": ann["et"], **clim, "ai": ai})
        per_model.append(inputs.assign_coords(
            member=[f"{sid}.{m}" for m in mids], source=("member", [sid] * len(mids)),
        ))

    if not per_model:
        raise RuntimeError("No model could be loaded.")
    # A model missing a year of TIME_SLICE gets NaN ET in that year
    inputs = xr.concat(per_model, dim="member", join="outer", combine_attrs="drop_conflicts")
    n_models = len(dict.fromkeys(inputs["source"].values.tolist()))
    inputs.attrs = {
        "family": FAMILY, "dataset": f"CMIP6 {EXPERIMENT_ID}",
        "time_period": temporal.period_str(inputs.year.values), "grid": grid,
        "na_thresh": cfg.NA_THRESH, "lf_thresh": cfg.LF_THRESH, "min_annual_et": cfg.MIN_ANNUAL_ET,
        "land_mask": "sftlf > lf_thresh, without Greenland/Iceland",
    }
    fout = cfg.inputs_path(FAMILY, grid, period)
    fout.parent.mkdir(parents=True, exist_ok=True)
    inputs.to_netcdf(fout)
    print(f"\n{n_models} models, {inputs.sizes['member']} members, {inputs.attrs['time_period']}\n{fout}")

    # Diagnostics: climatology maps and LAI/AI histograms
    plotting.save_clim_diagnostics(inputs, FIG_ROOT / "clim", FAMILY, grid, period, title="CMIP6")


if __name__ == "__main__":
    main()
