# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for quantifying evapotranspiration (ET) uncertainty. ET is binned in a 2-D space of climatological leaf area index (LAI, y-axis) and aridity index (AI = Rn / L·P, x-axis), so CMIP6 models, CESM ensembles (FHIST PPE "fppe", GOGA2, LENS2) and ILAMB observational products can be compared bin by bin.

Runs on NCAR's glade (Derecho/Casper). Data and outputs live outside the repo:
- Inputs: `/glade/campaign/univ/uwas0155/` (ILAMB obs, CMIP6 catalogs, regridded CMIP6). This is shared campaign storage, so don't write there unless asked.
- Outputs: `/glade/work/bbuchovecky/et_unc/{proc,fig}` (binned stats `proc/<dataset>/qbin/`, bin edges `proc/qbin_edges/`, figures `fig/`).

## Commands

The Python env is `/glade/work/bbuchovecky/miniforge3/envs/data-sci-py312/bin/python`. It has xarray, xesmf, regionmask, cartopy, ILAMB, and the user's own `xclimate` package.

```bash
PY=/glade/work/bbuchovecky/miniforge3/envs/data-sci-py312/bin/python
$PY -m pytest test_binned_et.py                      # full suite (~70 s, no external data needed)
$PY -m pytest test_binned_et.py::test_bin_stats_mask # single test
$PY ilamb_binned_et.py                               # obs pipeline (log: ilamb_binned_et.log)
$PY cmip_binned_et.py                                # CMIP6 pipeline
qsub regrid_cmip_esgf.pbs                            # batch regrid of CMIP6 to 1° (Casper, account UWAS0155)
```

The driver scripts take no CLI arguments. They are configured by module-level constants at the top of each file (`TIME_SLICE`, `N_XBINS`/`N_YBINS`, `SOURCE_IDS`, `MEMBER_IDS`, `MIN_YEARS`, `TARGET_GRID`, …). `download_ilamb.py` runs at import time and has no `__main__` guard.

## Architecture

**`binned_et.py`** is the dataset-agnostic core library (imported as `be`). Its module docstring documents the pipeline:
1. Load monthly fields plus a land mask on one shared lat/lon grid, with water fluxes converted to energy fluxes in W/m².
2. `prepare_inputs` returns `{"et": annual-mean ET (year,lat,lon), "lai": climatological LAI, "ai": climatological AI}`.
3. `pooled_bin_edges` builds quantile edges pooled across every dataset that should share bins.
4. `bin_stats` (one dataset, optionally per member via `member_dim`) or `bin_stats_by_source` (dict of models, concatenated along `source_id`) computes the bin statistics.
5. Post-processing and plotting: `drop_zero_width_bins`, `frac_valid`, `test_significance`, `ensemble_spread`, `plot_*`.

**Driver scripts** apply that pipeline to one data family each and save NetCDF and PNG outputs. Their docstrings list every output path and filename pattern.
- `ilamb_binned_et.py`: every (ET, LAI, pr, rns) ILAMB product combination on a 0.5° grid. Each combination uses only the complete years shared by all four products. Produces "pooled_obs" and per-"combo" edges.
- `cmip_binned_et.py`: CMIP6 historical on a common 1° grid (xESMF bilinear), per member. Produces "pooled_cmip" and per-"model" edges. Member selection is `"top"`, `"max_r"`, `"all"` or an explicit list.
- Each driver keeps its own small copies of `format_grid`, `complete_years`, `annual_mean` and `land_mask`. They differ deliberately: for ILAMB, a missing LAI month counts as 0, while ET, pr and rns need all 12 months. Check both drivers before unifying them.

**`load_cmip_esgf.CMIPESGFLoader`** reads a catalog CSV produced by the external `cmip-intake-esgf-fetch` tool. It handles model, member and variable availability and loading. `be.load_cmip` wraps it, then puts every variable on the `areacella` grid and applies the `sftlf` land mask (Greenland and Iceland excluded). `regrid_cmip_esgf.py` uses the same loader to write regridded files.

**Notebooks** are exploratory and analysis work. `binned_stats.ipynb` is the original version in which the binning functions were defined inline; `binned_et.py` was extracted from it. `open_bin_stats` / `_ensure_bin_coords` still read the notebook-era files. `agu-abstract.ipynb` and `compare-ilamb.ipynb` read the saved `qbin` outputs. `cmip_trends.py` and `obs_trends.py` are empty placeholders.

## Conventions that matter

- **Binning semantics:** every (year, gridcell) annual-mean ET sample is binned by that gridcell's climatological LAI and AI. `bin_stats` broadcasts the (lat, lon) LAI and AI fields over `year` (and `member`).
- **Grid alignment:** inputs must share coordinates exactly. `bin_stats` uses `xr.align(join="exact")` and `check_same_grid` raises an error instead of reindexing silently. Regrid or `reindex_like(..., method="nearest", tolerance=1e-3)` before binning.
- **Duplicate quantile edges** (from many LAI = 0 values) are kept on purpose. `searchsorted(side="right")` puts tied values in the *last* duplicate bin, which leaves zero-width empty bins that `drop_zero_width_bins` removes. Out-of-range values are clipped into the edge bins.
- **Output layout:** `bin_stats` returns dims `([member|source_id|combo,] stats, y_bin, x_bin)`, with `stats` ordered as `be.STATS = ("mean", "var_pop", "var_samp", "count", "count_pos")`. The edges are stored in attrs and the per-bin lower/upper/center coords.
- **Units:** there are two conversion paths. `convert_units` keys on the variable name (CMIP, CESM, ERA5), and `latent_heat_to_wm2` keys on the `units` attr (ILAMB). Both use L = 2.45e6 J/kg.
- **Net radiation sign conventions** differ by source. Use the matching `net_radiation_cmip`, `net_radiation_cesm` or `net_radiation_era5`.
- **Domain:** `LAT_BNDS = slice(-58, 90)` excludes Antarctica, and the land fraction threshold is `LF_THRESH = 0.5`.
- **Tests:** tests cover `binned_et.py` with synthetic data. The loaders (`load_cmip`, `load_cesm_*`, `load_ilamb_obs`) and `compute_cell_area` need glade data or ILAMB and are not tested.
