# et_unc

Research code for quantifying evapotranspiration (ET) uncertainty. ET is binned in
a 2-D space of climatological leaf area index (LAI) and aridity index (AI = Rn / Lv×P),
so that observational products (ILAMB, PML-V2.2, GLEAM v4.3, SiTHv2), ERA5, CMIP6 models,
CESM2 perturbed parameter ensembles, and the CESM2 Large Ensemble can be compared bin by bin.

The code runs on NCAR's HPC system (Casper/Derecho). Inputs and outputs live outside
the repo:

| | Path |
| --- | --- |
| Inputs (shared, read-only) | `/glade/campaign/univ/uwas0155/` (ILAMB and gridded obs, and CMIP6) |
| CMIP6 catalogs | `/glade/derecho/scratch/$USER/cmip_intake_esgf_fetch/` (purgeable scratch; see `etunc/config.py`) |
| Outputs | `/glade/work/$USER/et_unc/proc` and `fig/`, as `<root>/<family>/<kind>/<period>/[lai<N>_ai<N>/]` (`cfg.output_dir`) |

All of these paths are set in `etunc/config.py`.

## Data products

Each data product in the binning pipeline is organized into families (e.g., gridded observations, CMIP6, ERA5, CESM2 FPPE). Within each family, there can exist members (e.g., observational data products, CMIP6 models, CESM2 ensemble members, TRENDY models). Each member then can have multiple sources (e.g., CMIP6 ensemble members with different r-i-p-f- IDs). For now, a single CMIP6 model-member combination is used.

| Data product | Abbreviation | Reference |
| -------------| ------------ | --------- |
| Gridded observational products | `obs` |  |
| ERA5 | `era5` |  |
| CMIP6 Historical Experiment | `cmip` |  |
| CESM2 Historical PPE, coupled | `fppe` |  |
| CESM2 Historical PPE, offline | `ippe` |  |
| CESM2 Large Ensemble | `lens` |  |
| TRENDY, Global Carbon Budget 2025 | `trendy` |  |
| PLUMBER2 flux towers | `plumber` |  |

### Observations
The following product choices and corresponding time periods were made:
- `199501-201412` uses GEWEX.SRB for net surface radiation, GIMMS_LAI4g for LAI, and GPCPv2.3 for precipitation.
- `200101-201412` uses CERESed4.2 for net surface radiation, GIMMS_LAI4g for LAI, and GPCPv2.3 for precipitation.

### TRENDY
The LSMs and DVGMs participating in TRENDY do not provide upwelling surface radiation, so ERA5 net surface radiation is used for the aridity calculation. A full comparison of CERESed4.2, ERA5, and the CRU-JRA3 (downwelling-only) is done with `scripts/compare_rn.py`.

## Layout

| Path | What it holds |
| --- | --- |
| `etunc/` | The library (`import etunc.binning as binning`, ...). No module runs anything on import. |
| `etunc/config.py` | Paths, domain and pipeline constants (`TIME_SLICE`, `LAT_BNDS`, `LF_THRESH`, `NA_THRESH`, `MIN_ANNUAL_ET`, `FAMILIES`, L = 2.45e6 J/kg, ...), and the paths of the files passed between the pipeline scripts |
| `etunc/units.py` | Unit conversions to W/m², net radiation for each data source |
| `etunc/temporal.py` | Annual means and climatologies (`aggregate`: complete years, missing-month rules, missing-year report), shared years, period strings |
| `etunc/grid.py` | Target grids (`target_grid`), xESMF regridders (`regrid_with_na_thresh`), grid checks and snapping, native land masks (`landfrac_mask`, `landne_mask`, `mask_to_land`), the common area mask (`valid_area`, `common_area`), cell areas (`cell_area`) and Natural Earth land areas (`land_area`) |
| `etunc/edges.py` | Aridity index, hierarchical weights and weighted quantile edges, saved edges |
| `etunc/binning.py` | ET outlier filter, bin statistics, post-processing |
| `etunc/regression.py` | OLS regression of ET on climatological LAI and AI, and the partition of R² between them |
| `etunc/plotting.py` | Maps, LAI/AI histograms, edge plots, bin heatmaps |
| `etunc/load/` | Loaders: `ilamb`, `obs` (PML/GLEAM/SiTH), `cmip`, `cesm`, `era5`, `trendy` (TRENDY DGVMs and their CRU-JRA precipitation forcing). Each `load_native` returns raw monthly fields on the native grid in native units; the model loaders also have `load_land_fraction` |
| `etunc/legacy/` | Unmaintained, untested code kept for the notebooks and `scripts/legacy_regrid/`: the former annual loaders, replaced functions, notebook-era helpers |
| `etunc/dask_cluster.py` | Start and stop a PBS dask cluster for notebooks |
| `docs/binning-workflow.md` | The binning pipeline step by step, with every assumption and choice |
| `scripts/` | Runnable pipelines (`prep_*.py` → `make_edges.py` → `bin_*.py`; the analysis `regress_lai_aridity.py`, outside the pipeline, after them), configured by the constants at the top of each file |
| `scripts/download/` | Download scripts for the obs products (and `trendy/` for TRENDY) |
| `notebooks/` | Exploratory and analysis notebooks |
| `tests/` | pytest suite (synthetic data, no glade data needed) |
| `envs/` | Conda environments outside the pixi manifest: `openet.yml`, `lftp.yml` |
| `pixi.lock` | Exact versions of the `etunc` env (manifest in `pyproject.toml` under `[tool.pixi.*]`) |

## Setup

The `etunc` env is managed by [pixi](https://pixi.sh). One-time setup, so the
env (~3 GB) and the package cache go to `/glade/work` instead of home:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi config set --global detached-environments /glade/work/$USER/pixi/envs
export PIXI_CACHE_DIR=/glade/work/$USER/pixi/cache   # can also put in ~/.bashrc
```

Then, from the repo:

```bash
pixi install       # creates the env from pixi.lock, with etunc installed editable
pixi run kernel    # registers the `etunc` Jupyter kernel for the notebooks
```

The package is installed editable, so changes under `etunc/` take effect
without reinstalling. Add or change a dependency with `pixi add <pkg>` (or edit
`[tool.pixi.dependencies]`, then `pixi install`); either way, commit the updated
`pixi.lock`.

## Running

The binning pipeline (described step by step in `docs/binning-workflow.md`) has three stages:

```bash
# Stage 1
pixi run prep-obs         # steps 1-11: binning inputs of every obs product combination
pixi run prep-era5        # steps 1-11: binning inputs of ERA5 (ET, P, Rn) with obs LAI (GIMMS_LAI4g)
pixi run prep-cmip        # steps 1-11: binning inputs of the CMIP6 historical models
pixi run prep-lens        # steps 1-11: binning inputs of the CESM2 Large Ensemble historical members
pixi run prep-fppe        # steps 1-11: binning inputs of the CESM2 FHIST perturbed-parameter ensemble
pixi run prep-ippe        # steps 1-11: binning inputs of the CESM2 offline (CLM5) perturbed-parameter ensemble
qsub -N prep_obs -v FAMILY=obs scripts/prep.pbs  # any prep-* as a Casper job (FAMILY: obs, era5, cmip, lens, fppe, ippe)

# Stage 2
pixi run make-edges       # steps 12-13: common area mask and pooled LAI/AI edges (needs every prep-*)

# Stage 3
pixi run bin-all          # step 14: binned ET of every family with the pooled_all and pooled_<family> edges
pixi run bin-all-clim     # step 14 with climatological ET: one sample per gridcell instead of per gridcell-year
pixi run bin-area         # step 14 for land area: total Natural Earth land area in each bin, per member and source
pixi run bin-plumber      # step 14 for the PLUMBER2 flux towers: tower ET binned by the obs LAI/AI at the nearest gridcell
pixi run bin-plumber-clim # step 14 for the PLUMBER2 flux towers with climatological ET: one sample per tower (needs bin-plumber)

# Analyses of the step 14 outputs, not part of the pipeline
pixi run regress-lai-aridity   # OLS of ET on LAI and AI: variance explained, with the bins' eta2
pixi run make-mask             # common valid-data mask of the obs products
pixi run diagnose-obs          # coverage of every obs ET, LAI, P and Rn product (docs/obs-availability.md)
qsub scripts/diagnose_obs.pbs  # diagnose-obs as a Casper job (too large for a login node)
pixi run compare-rn            # ERA5, CERES and CRU-JRA3 (TRENDY forcing) surface radiation, for the Rn of prep_trendy.py
qsub scripts/compare_rn.pbs    # compare-rn as a Casper job

pixi run test                  # all tests (~6 min)
```

Each task is `scripts/run_logged.sh scripts/<name>.py`, which logs the output,
the git commit, `etunc/config.py` and the script's settings to `WORK_ROOT/logs`.
The scripts take no arguments: edit the settings at the top of each script
(products, models, members, ...) or the shared ones in `etunc/config.py` (time
period, grid, thresholds, number of bins). Each script's docstring lists every
output file it writes. Every family runs the same steps, written out in the
script's `main()`; only the loader (step 1), the unit conversion (step 3) and
the land mask (step 5) differ.

## To reproduce all results

After the data are downloaded (`scripts/download/`), `scripts/run_all.sh` runs
every step in order (prep-\*, make-edges, bin-\*), each through `scripts/run_logged.sh`,
and stops at the first failure. It needs a large node (~128 GB), so run it as one job:

```bash
qsub scripts/run_all.pbs
```

`BIN_VARIABLE`, `AREA_WEIGHTED`, `N_XBINS` and `N_YBINS` of `etunc/config.py` can be set by
environment variables (`ETUNC_BIN_VARIABLE`, ...), `SKIP_PREP=1` starts at make-edges, and
`SWEEP=1` runs every combination (et/et_pr, unweighted/area-weighted, 10/15/20 bins) in one go:

```bash
qsub -v ETUNC_BIN_VARIABLE=et_pr,ETUNC_AREA_WEIGHTED=1,ETUNC_N_XBINS=20,ETUNC_N_YBINS=20 scripts/run_all.pbs
qsub -v SWEEP=1 scripts/run_all.pbs
```

It runs only the pipeline: not the analyses (`regress-lai-aridity`) or the diagnostics (`make-mask`, `obs-availability`, `diagnose-obs`).

## Config
`etunc/config.py` contains the domain constants, such as `LF_THRESH` and `NA_THRESH`.
Binning settings (e.g., `N_XBINS`, `N_YBINS`) exist there but they also can changed at
the top of each run script. `BIN_VARIABLE`, `AREA_WEIGHTED`, `N_XBINS` and `N_YBINS` can
also be set by environment variables (`ETUNC_BIN_VARIABLE`, ...). For consistent analysis,
all run script should have the same binning settings.

## TODO
- Allow data families to use their own valid area for binning, instead of using the common area mask.
- Perform sensitivity test to resolution, comparing the bin stats between native and 1° regridded resolution resolution.
