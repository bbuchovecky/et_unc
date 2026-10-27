"""
compare_rn.py
=============
Surface radiation of ERA5, CERES and the CRU-JRA v3 forcing of the TRENDY
models, to choose the Rn of prep_trendy.py (the TRENDY models do not output
the upward terms of the net radiation). Analysis, not part of the pipeline.

Sources
-------
- SW down: ERA5 msdwswrf, CERES rsds, and CRU-JRA: the median rsds of
  RSDS_MODELS (crujra3/ holds no shortwave file; each model's rsds is the
  shortwave forcing it received)
- LW down: ERA5 msdwlwrf, CERES rlds, CRU-JRA dlwrf (`lt.load_forcing`)
- Rn:      ERA5 msnswrf + msnlwrf (as in prep_era5.py), CERES rns (as in
  prep_obs.py). CRU-JRA has no net radiation.
CERES is EBAF ed4.2 from the ILAMB data (`il.load_native`).

Processing
----------
Each field: annual means over complete years (step 4), Natural Earth land
without Greenland/Iceland (`rg.landne_mask`, the same for every field, so all
compare the same cells), conservative regridding to TARGET_RES with NA_THRESH
(step 7), LAT_BNDS (step 8) and the climatology over TIME_SLICE (step 10).
Every pair of sources is compared on the gridcells valid in both, land-area
weighted (`rg.land_area`), globally and in BANDS: mean of each, bias, RMSE and
pattern correlation of the climatologies, and the correlation of their annual
land-mean series over their shared years.

Outputs
-------
<grid>   : grid tag of TARGET_RES, e.g. "1deg"
<period> : YYYYMM-YYYYMM of TIME_SLICE

PROC_ROOT/trendy/radiation/<period>/trendy.compare_rn.<period>.csv
    statistics of every pair of sources, per variable and band
PROC_ROOT/trendy/radiation/<period>/trendy.compare_rn.<grid>.<period>.nc
    climatology of each variable and source on the target grid
FIG_ROOT/trendy/radiation/<period>/trendy.compare_rn.<grid>.<period>.png
    difference maps of every pair of sources and zonal land means, per variable
"""

from __future__ import annotations

import itertools
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import etunc.config as cfg
import etunc.grid as rg
import etunc.temporal as temporal
import etunc.load.era5 as le
import etunc.load.ilamb as il
import etunc.load.trendy as lt

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS. Will affect performance.", category=UserWarning)

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

EXPERIMENT = "S3"  # "S2": climate + CO2; "S3": climate + CO2 + land use change
FAMILY = f"trendy_{EXPERIMENT.lower()}"

TIME_SLICE = cfg.TIME_SLICE
TARGET_RES = cfg.TARGET_RES

# Models whose rsds is the forcing shortwave. Left out: SDGVM (~12x too small), VISIT (~3x too large),
# LPJwsl and LPX-Bern (~31% low), VISIT-UT (6.5% high); DLEM, JULES-ES and LPJ-GUESS have no rsds
RSDS_MODELS = ["CABLE-POP", "CARDAMOM", "CLASSIC", "CLM", "CLM-FATES", "ED", "ELM", "ELM-FATES",
               "IBIS", "iMAPLE", "ISAM", "JSBACH", "LPJml", "ORCHIDEE", "TEM"]
CERES_PRODUCT = "CERESed4.2"

# Latitude bands of the statistics: {name: (south, north)}
BANDS = {
    "Global": (-58, 90),
    "NH high (50-90N)": (50, 90),
    "NH mid (23-50N)": (23, 50),
    "Tropics (23S-23N)": (-23, 23),
    "SH extratropics (58-23S)": (-58, -23),
}
DIFF_LIM = 40  # W/m2, color limit of the difference maps


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def annual_on_target(monthly: xr.DataArray, label: str) -> xr.DataArray:
    """Steps 4, 5, 7 and 8 with the Natural Earth land mask: annual means on the target grid."""
    ann = temporal.aggregate(monthly, "year", label=label)
    grid = rg.bounded_src_grid(ann)
    ann = rg.mask_to_land(ann, rg.landne_mask(grid), label)
    regridder = rg.bounded_conservative_regridder(grid, TARGET_RES)
    ann = rg.on_grid(rg.regrid_with_na_thresh(ann, regridder, cfg.NA_THRESH), TARGET_RES, label)
    return ann.sel(lat=cfg.LAT_BNDS)


def pair_stats(a: xr.DataArray, b: xr.DataArray, area: xr.DataArray) -> dict[str, float]:
    """Land-area weighted means, bias, RMSE and pattern correlation of a vs b, on cells valid in both."""
    w = area.where(a.notnull() & b.notnull(), 0)
    mean_a, mean_b = float((a * w).sum() / w.sum()), float((b * w).sum() / w.sum())
    rmse = float(np.sqrt(((a - b) ** 2 * w).sum() / w.sum()))
    da, db = (a - mean_a).fillna(0), (b - mean_b).fillna(0)
    r = float((da * db * w).sum() / np.sqrt((da**2 * w).sum() * (db**2 * w).sum()))
    return {"mean_a": mean_a, "mean_b": mean_b, "bias_a_minus_b": mean_a - mean_b,
            "rel_bias_pct": 100 * (mean_a - mean_b) / mean_b, "rmse": rmse, "pattern_r": r}


def land_mean(ann: xr.DataArray, area: xr.DataArray) -> xr.DataArray:
    """Land-area weighted mean over the valid cells of each year."""
    w = area.where(ann.notnull(), 0)
    return (ann.fillna(0) * w).sum(["lat", "lon"]) / w.sum(["lat", "lon"])


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    cfg.summarize()
    cfg.summarize_settings(globals())
    period = temporal.format_time_period(TIME_SLICE)
    grid = rg.grid_tag(TARGET_RES)
    area = rg.land_area(rg.target_grid(TARGET_RES)).sel(lat=cfg.LAT_BNDS)

    # ERA5 downward and net fluxes (positive downward)
    print("=== ERA5 ===")
    era5 = le.load_native(["msdwswrf", "msdwlwrf", "msnswrf", "msnlwrf"], TIME_SLICE)

    # CRU-JRA shortwave: the median of the models' rsds. Their spread around it shows they share one field
    print("\n=== TRENDY rsds ===")
    rsds = {m: annual_on_target(lt.load_native(m, EXPERIMENT, TIME_SLICE, ["rsds"])["rsds"], f"{m}/rsds")
            for m in RSDS_MODELS}
    rsds = xr.concat(list(rsds.values()), dim=pd.Index(RSDS_MODELS, name="model", dtype=object), join="outer")
    cru_sw = rsds.median("model")
    dev = (rsds.mean("year") - cru_sw.mean("year")) ** 2
    print("RMS deviation of each model's climatological rsds from the median [W/m2]:")
    for m in RSDS_MODELS:
        print(f"  {m:10} {float(np.sqrt(dev.sel(model=m).mean())):.2f}")

    print("\n=== Annual means on the target grid ===")
    ann = {
        "sw_down": {
            "ERA5": annual_on_target(era5["msdwswrf"], "era5/msdwswrf"),
            "CERES": annual_on_target(il.load_native("rsds", CERES_PRODUCT, TIME_SLICE), "ceres/rsds"),
            "CRU-JRA": cru_sw,
        },
        "lw_down": {
            "ERA5": annual_on_target(era5["msdwlwrf"], "era5/msdwlwrf"),
            "CERES": annual_on_target(il.load_native("rlds", CERES_PRODUCT, TIME_SLICE), "ceres/rlds"),
            "CRU-JRA": annual_on_target(lt.load_forcing("dlwrf", TIME_SLICE), "crujra/dlwrf"),
        },
        "rn": {
            "ERA5": annual_on_target(era5["msnswrf"] + era5["msnlwrf"], "era5/rn"),
            "CERES": annual_on_target(il.load_native("rn", CERES_PRODUCT, TIME_SLICE), "ceres/rns"),
        },
    }
    clim = {v: {s: temporal.aggregate(da, "clim", label=f"{s}/{v}") for s, da in d.items()}
            for v, d in ann.items()}

    # Statistics of every pair of sources: climatologies by band, and the interannual correlation
    rows = []
    for var, d in clim.items():
        for a, b in itertools.combinations(d, 2):
            years = temporal.shared_years(ann[var][a], ann[var][b])
            iav_r = float(xr.corr(land_mean(ann[var][a].sel(year=years), area),
                                  land_mean(ann[var][b].sel(year=years), area)))
            for band, (south, north) in BANDS.items():
                sel = dict(lat=slice(south, north))
                rows.append({"variable": var, "a": a, "b": b, "band": band,
                             **pair_stats(d[a].sel(**sel), d[b].sel(**sel), area.sel(**sel)),
                             "iav_r_land_mean": iav_r if band == "Global" else np.nan})
    stats = pd.DataFrame(rows)
    proc_dir = cfg.output_dir(cfg.PROC_ROOT, FAMILY, "radiation", period)
    proc_dir.mkdir(parents=True, exist_ok=True)
    fout = proc_dir / f"{FAMILY}.compare_rn.{period}.csv"
    stats.to_csv(fout, index=False, float_format="%.3f")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(f"\n{stats.round(2).to_string(index=False)}\n{fout}")

    # Climatologies, e.g. sw_down_era5, lw_down_crujra
    ds = xr.Dataset({f"{v}_{s.replace('-', '').lower()}": da for v, d in clim.items() for s, da in d.items()})
    ds.attrs = {"time_period": period, "grid": grid, "units": "W/m2",
                "crujra_sw": f"median rsds of the TRENDY {EXPERIMENT} models {', '.join(RSDS_MODELS)}"}
    fout = proc_dir / f"{FAMILY}.compare_rn.{grid}.{period}.nc"
    ds.to_netcdf(fout)
    print(fout)

    # Figure: per variable, the difference map of every pair and the zonal land means
    colors = {"ERA5": "k", "CERES": "tab:orange", "CRU-JRA": "tab:blue"}
    titles = {"sw_down": "SW down", "lw_down": "LW down", "rn": "Rn"}
    fig, axs = plt.subplots(len(clim), 4, figsize=(20, 3.7 * len(clim)), constrained_layout=True)
    for i, (var, d) in enumerate(clim.items()):
        pairs = list(itertools.combinations(d, 2))
        for j, (a, b) in enumerate(pairs):
            (d[a] - d[b]).plot(ax=axs[i, j], cmap="RdBu_r", vmin=-DIFF_LIM, vmax=DIFF_LIM, add_labels=False,
                               cbar_kwargs={"label": "W/m2"})
            axs[i, j].set_title(f"{titles[var]}: {a} - {b}")
        for j in range(len(pairs), 3):
            axs[i, j].axis("off")
        for s, da in d.items():
            da.mean("lon").plot(ax=axs[i, 3], color=colors[s], label=s)
        axs[i, 3].set(title=f"{titles[var]}: zonal land mean", ylabel="W/m2")
        axs[i, 3].legend()
    fig.suptitle(f"Surface radiation, land climatology {period}, {grid}")
    fig_dir = cfg.output_dir(cfg.FIG_ROOT, FAMILY, "radiation", period)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fout = fig_dir / f"{FAMILY}.compare_rn.{grid}.{period}.png"
    fig.savefig(fout, dpi=110)
    print(fout)


if __name__ == "__main__":
    main()
