# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for quantifying evapotranspiration (ET) uncertainty. ET is binned in a 2-D space of climatological leaf area index (LAI, y-axis) and aridity index (AI = Rn / L·P, x-axis), so CMIP6 models, CESM ensembles (FHIST PPE "fppe", GOGA2, LENS2) and ILAMB observational products can be compared bin by bin.

Runs on NCAR's glade (Derecho/Casper). Data and outputs live outside the repo:
- Inputs: `/glade/campaign/univ/uwas0155/` (ILAMB obs, CMIP6 catalogs, regridded CMIP6). This is shared campaign storage, so don't write there unless asked.
- Outputs: `/glade/work/bbuchovecky/et_unc/{proc,fig}` (binned stats `proc/<dataset>/qbin/`, bin edges `proc/qbin_edges/`, figures `fig/`).

## Commands

The Python env is `etunc` (`/glade/work/bbuchovecky/miniforge3/envs/etunc/bin/python`), defined in `envs/etunc.yml` (exact versions in `envs/etunc.lock.yml`) with the repo installed editable (`pip install --no-deps -e .`). It has xarray, xesmf, regionmask, cartopy and dask-jobqueue, but not ILAMB: `etunc.legacy.compute_cell_area` and the notebooks that call `ilamblib` still need `data-sci-py312` until the refactor ports `CellAreas`. The repo is self-contained: don't add dependencies on the user's other packages (e.g. xclimate) or on modules from their other projects.

```bash
PY=/glade/work/bbuchovecky/miniforge3/envs/etunc/bin/python
$PY -m pytest                                        # full suite (~7 min, no external data needed)
$PY -m pytest test_binned_et.py                      # binning tests only (~70 s)
$PY -m pytest test_binned_et.py::test_bin_stats_mask # single test
$PY ilamb_binned_et.py                               # obs pipeline (log: ilamb_binned_et.log)
$PY cmip_binned_et.py                                # CMIP6 pipeline
qsub regrid_cmip_esgf.pbs                            # batch regrid of CMIP6 to 1° (Casper, account UWAS0155)
```

The driver scripts take no CLI arguments. They are configured by module-level constants at the top of each file (`TIME_SLICE`, `N_XBINS`/`N_YBINS`, `SOURCE_IDS`, `MEMBER_IDS`, `MIN_YEARS`, `TARGET_GRID`, …). `download_ilamb.py` runs at import time and has no `__main__` guard.

## Architecture

The dataset-agnostic core library is the `etunc` package (split out of the former `binned_et.py`): `config` (paths and domain constants), `units` (unit conversions, net radiation), `temporal` (annual means, aggregation, period strings), `binning` (inputs, edges, bin statistics, post-processing), `plotting` (maps and bin heatmaps), `grid` (below; also `check_same_grid`/`equal_coords` and `mask_greenland`) and `legacy` (notebook-only loaders and helpers). Scripts import them as `import etunc.binning as binning` etc., with `etunc.grid` as `rg`. The `etunc.binning` docstring documents the pipeline:
1. Load monthly fields plus a land mask on one shared lat/lon grid, with water fluxes converted to energy fluxes in W/m².
2. `prepare_inputs` returns `{"et": annual-mean ET (year,lat,lon), "lai": climatological LAI, "ai": climatological AI}`.
3. `pooled_bin_edges` builds quantile edges pooled across every dataset that should share bins.
4. `bin_stats` (one dataset, optionally per member via `member_dim`) or `bin_stats_by_source` (dict of models, concatenated along `source_id`) computes the bin statistics.
5. Post-processing and plotting: `drop_zero_width_bins`, `frac_valid`, `test_significance`, `ensemble_spread`; figures in `etunc.plotting`.

**Driver scripts** apply that pipeline to one data family each and save NetCDF and PNG outputs. Their docstrings list every output path and filename pattern.
- `ilamb_binned_et.py`: every (ET, LAI, pr, rns) ILAMB product combination on a 0.5° grid. Each combination uses only the complete years shared by all four products. Produces "pooled_obs" and per-"combo" edges.
- `cmip_binned_et.py`: CMIP6 historical for every model in `cmip_evap.csv` with sftlf in `cmip6_fx_glade.csv` (catalogs in `/glade/derecho/scratch/bbuchovecky/cmip_intake_esgf_fetch/catalogs/`). Annual means are masked with native sftlf, then conservatively regridded to the common 1° grid (`na_thres=0.5`), per member. All models are binned over one common area mask. Produces "pooled_cmip" and per-"model" edges. Member selection is `"top"`, `"max_r"`, `"all"` or an explicit list.
- Each driver keeps its own small copies of `complete_years`, `annual_mean` and `land_mask` (plus `format_grid` in ILAMB). They differ deliberately: for ILAMB, a missing LAI month counts as 0, while ET, pr and rns need all 12 months. Check both drivers before unifying them.

**`etunc.load.cmip.CMIPESGFLoader`** reads a catalog CSV produced by the external `cmip-intake-esgf-fetch` tool. It handles model, member and variable availability and loading. `etunc.legacy.load_cmip` (notebook-only) wraps it, then puts every variable on the `areacella` grid and applies the `sftlf` land mask (Greenland and Iceland excluded). `regrid_cmip_esgf.py` uses the same loader to write regridded files.

**`etunc/grid.py`** (imported as `rg`) holds the shared regridding utilities. `rg.target_grid(res)` is the only source of the two common grids, 0.5° and 1° (`rg.RESOLUTIONS`). Both are global, lon in [-180, 180], lat ascending, with `lat_b`/`lon_b` edges. Never build a target grid by hand. The module also provides the xESMF regridders onto those grids: `conservative_regridder` (cached, for regular 1-D sources), `bounded_conservative_regridder` (any 1-D source such as Gaussian grids, with edges from CF bounds or midpoints), `bilinear_regridder` (periodic) and `make_regridder` (any method or source grid), plus `approx_resolution`. `regrid_obs.py`, `regrid_cmip_esgf.py` and both binning drivers use it, and `test_regrid.py` checks that their grids are identical.

**`etunc/load/obs.py`** (imported as `lo`) loads per-year gridded obs products (GLEAM v4.3, PML-V2.2 in `/glade/campaign/univ/uwas0155/obs/`) into one lazy DataArray with `load_obs(dataset, var, time_slice, version=, freq=)` (or `load_gleam`/`load_pml`). Each product is an `ObsDataset` entry (path template, versions, frequencies, coord names), so a new product is added with `register_dataset`. Monthly and yearly files hold totals (mm/month, mm/year); `accumulation_to_flux` converts them to kg m-2 s-1. Tested in `test_load_obs.py`.

**`etunc/load/cesm.py`** (imported as `lc`) loads the CESM2 ensembles from the tseries archives on glade: `load_fhist_ppe` (FHIST PPE, members in parallel threads), `load_goga2`, `load_cesm2le`, plus `load_grid("fppe"|"goga"|"lens")` and `ppe_member_name`. Callers give each variable's `gcomp` ("lnd"/"atm") and `stream` explicitly; there is no variable lookup table. `load_cesm_grid` / `load_cesm_variable` in the same module wrap it. **`etunc/load/era5.py`** (`le`) loads ERA5 monthly means from GDEX and the ERA5 grid file. **`etunc/dask_cluster.py`** (`dc`) starts and stops a PBS dask cluster on Casper or Derecho for the notebooks. All three were ported from the user's former `xclimate` package.

**Notebooks** are exploratory and analysis work. `binned_stats.ipynb` is the original version in which the binning functions were defined inline; the former `binned_et.py` was extracted from it. `legacy.open_bin_stats` / `binning.ensure_bin_coords` still read the notebook-era files. `agu-abstract.ipynb` and `compare-ilamb.ipynb` read the saved `qbin` outputs.

## Conventions that matter

- **Binning semantics:** every (year, gridcell) annual-mean ET sample is binned by that gridcell's climatological LAI and AI. `bin_stats` broadcasts the (lat, lon) LAI and AI fields over `year` (and `member`).
- **Grid alignment:** inputs must share coordinates exactly. `bin_stats` uses `xr.align(join="exact")` and `check_same_grid` raises an error instead of reindexing silently. Regrid or `reindex_like(..., method="nearest", tolerance=1e-3)` before binning.
- **Duplicate quantile edges** (from many LAI = 0 values) are kept on purpose. `searchsorted(side="right")` puts tied values in the *last* duplicate bin, which leaves zero-width empty bins that `drop_zero_width_bins` removes. Out-of-range values are clipped into the edge bins.
- **Output layout:** `bin_stats` returns dims `([member|source_id|combo,] stats, y_bin, x_bin)`, with `stats` ordered as `binning.STATS = ("mean", "var_pop", "var_samp", "count", "count_pos")`. The edges are stored in attrs and the per-bin lower/upper/center coords.
- **Units:** there are two conversion paths. `convert_units` keys on the variable name (CMIP, CESM, ERA5), and `latent_heat_to_wm2` keys on the `units` attr (ILAMB). Both use L = 2.45e6 J/kg.
- **Net radiation sign conventions** differ by source. Use the matching `net_radiation_cmip`, `net_radiation_cesm` or `net_radiation_era5`.
- **Domain:** `LAT_BNDS = slice(-58, 90)` excludes Antarctica, and the land fraction threshold is `LF_THRESH = 0.5`.
- **Tests:** all tests use synthetic data. `test_binned_et.py` covers `units`, `temporal`, `binning`, `plotting`, the grid checks and `legacy`, `test_regrid.py` covers `etunc/grid.py`, `test_load_obs.py` and `test_regrid_obs.py` cover the obs loader and regridder, and `test_load_ilamb.py` and `test_load_cmip.py` pin the driver loaders (`ib.load_product`, the monthly ILAMB loaders, CMIP member selection, `cbe.load_model`). Not tested, because they need glade data, PBS or ILAMB: `load_cmip`, `load_cesm_*`, `load_ilamb_obs`, `etunc/load/cesm.py`, `etunc/load/era5.py`, `etunc/dask_cluster.py` and `compute_cell_area`.

## Refactoring guidelines
- Prefer incremental changes over large rewrites
- Document any behavior changes in commit messages
- Add comments to improve readability and interpretability. Keep the comments concise and avoid being excessively verbose.

## Hard Rules
- Do NOT create stub functions
- Do NOT modify any files outside of /glade/u/home/bbuchovecky/projects/et_unc/. The only exceptions are for the refactor in `refactor.md`:
  - creating and installing into the `etunc` env at `/glade/work/bbuchovecky/miniforge3/envs/etunc`
  - installing the `etunc` Jupyter kernel (`~/.local/share/jupyter/kernels/etunc`)
- Always run tests before reporting a task is complete
