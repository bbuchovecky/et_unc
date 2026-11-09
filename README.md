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
| CMIP6 catalogs | `/glade/derecho/scratch/$USER/cmip_intake_esgf_fetch/` (purgeable scratch) |
| Outputs | `/glade/work/$USER/et_unc/proc` and `fig/`, as `<root>/<run or prep ID>/<family>/<kind>/` (`cfg.output_dir`) |

All of these paths are set in `config/paths.yml`, which is not in git: copy
`config/paths.example.yml` to `config/paths.yml` and edit it for your machine.

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

### PLUMBER2
The PLUMBER2 flux tower data are not included in the core binning pipeline. Instead, `scripts/bin_plumber.py` and `scripts/bin_plumber_clim.py` handle their processing and binning.

## Layout

| Path | What it holds |
| --- | --- |
| `etunc/` | The library (`import etunc.binning as binning`, ...). No module runs anything on import. |
| `etunc/config.py` | Reads `config/` (settings of the run, paths) into constants (`TIME_SLICE`, `LAT_BNDS`, `LF_THRESH`, `NA_THRESH`, `MIN_ANNUAL_ET`, `FAMILIES`, ...; L = 2.45e6 J/kg), the run IDs and register (`RUN_ID`, `PREP_ID`, `register_run`), and the paths of the files passed between the pipeline scripts |
| `etunc/units.py` | Unit conversions to W/m², net radiation for each data source |
| `etunc/temporal.py` | Annual means and climatologies (`aggregate`: complete years, missing-month rules, missing-year report), shared years, period strings |
| `etunc/grid.py` | Target grids (`target_grid`), xESMF regridders (`regrid_with_na_thresh`), grid checks and snapping, native land masks (`landfrac_mask`, `landne_mask`, `mask_to_land`), the common area mask (`valid_area`, `common_area`), cell areas (`cell_area`) and Natural Earth land areas (`land_area`) |
| `etunc/variants.py` | Groups of variants (CMIP6 models of one base model, versions of one obs ET product), so each counts once in the means, spreads and weights across a family (`source_groups`, `group_means`) |
| `etunc/edges.py` | Aridity index, hierarchical weights and weighted quantile edges, saved edges |
| `etunc/binning.py` | ET outlier filter, bin statistics, post-processing |
| `etunc/regression.py` | OLS regression of ET on climatological LAI and AI, and the partition of R² between them |
| `etunc/decomposition.py` | State, function and interaction decomposition of the area-mean ET of a dataset relative to an obs reference, from counterfactual means on the LAI-AI bins |
| `etunc/plotting.py` | Maps, LAI/AI histograms, edge plots, bin heatmaps |
| `etunc/load/` | Loaders: `ilamb`, `obs` (PML/GLEAM/SiTH), `cmip`, `cesm`, `era5`, `trendy` (TRENDY DGVMs and their CRU-JRA precipitation forcing). Each `load_native` returns raw monthly fields on the native grid in native units; the model loaders also have `load_land_fraction` |
| `etunc/legacy/` | Unmaintained, untested code kept for the notebooks and `scripts/legacy_regrid/`: the former annual loaders, replaced functions, notebook-era helpers |
| `etunc/dask_cluster.py` | Start and stop a PBS dask cluster for notebooks |
| `config/` | Settings of the pipeline and of each run, paths, and the register of run IDs (see Config below) |
| `docs/binning-workflow.md` | The binning pipeline step by step, with every assumption and choice |
| `scripts/` | Runnable pipelines (`prep_*.py` → `make_edges.py` → `bin_*.py`; the analyses `regress_lai_aridity.py` and `decompose_state_function.py` (each bin count: `run_decomp.sh`), outside the pipeline, after them), configured by the constants at the top of each file |
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
qsub -N prep_obs -v FAMILY=obs scripts/prep.pbs  # any prep-* as a Casper job (FAMILY: obs, era5, cmip, lens, fppe, ippe, trendy)

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
pixi run decompose-state-function  # state, function and interaction terms of the area-mean ET against the obs (an area-weighted run: RUN_ID in the script, default that of config/runs/aw_bins15.yml)
pixi run run-decomp                # scripts/run_decomp.sh: decompose-state-function for each run of RUNS (default config/runs/aw_bins{10,15,20}.yml)
pixi run make-mask             # common valid-data mask of the obs products
pixi run diagnose-obs          # coverage of every obs ET, LAI, P and Rn product (docs/obs-availability.md)
qsub scripts/diagnose_obs.pbs  # diagnose-obs as a Casper job (too large for a login node)
pixi run compare-rn            # ERA5, CERES and CRU-JRA3 (TRENDY forcing) surface radiation, for the Rn of prep_trendy.py
qsub scripts/compare_rn.pbs    # compare-rn as a Casper job

pixi run test                  # all tests (~6 min)
```

Each task is `scripts/run_logged.sh scripts/<name>.py`, which logs the output,
the git commit, the run's IDs and settings and the script's settings to `WORK_ROOT/logs`.
The scripts take no arguments: edit the settings at the top of each script
(products, models, members, ...), or choose a run (below) for the shared ones (time
period, grid, thresholds, number of bins, weighting). Each script's docstring lists every
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

`RUNS` lists the runs to make (run files, or `default`), `SKIP_PREP=1` starts at make-edges,
and the prep scripts run once for the runs that share their prep settings:

```bash
qsub -v RUNS=config/runs/aw_bins20.yml scripts/run_all.pbs
qsub -v RUNS="default config/runs/bins10.yml config/runs/aw_bins15.yml" scripts/run_all.pbs
pixi run run-all-sweep    # the defaults and every file of config/runs/
```

It runs only the pipeline: not the analyses (`regress-lai-aridity`, `decompose-state-function`, `run-decomp`) or the diagnostics (`make-mask`, `obs-availability`, `diagnose-obs`).

## Config
All settings shared by the scripts are in `config/` (read by `etunc/config.py`):

| File | What it holds |
| --- | --- |
| `config/pipeline.yml` | The default settings: `prep` (period, grid, thresholds of steps 1-11, and each family's products, models and members), `binning` (families, binned variable, weighting, bin counts of steps 12-14) and `plotting` |
| `config/runs/<name>.yml` | One run's changes to the defaults, only the keys that differ, selected with `ETUNC_RUN=config/runs/<name>.yml` |
| `config/paths.yml` | Every input and output path (not in git; copy `config/paths.example.yml`, or set `ETUNC_PATHS`) |
| `config/runs.csv` | Every run ID: its prep ID, run file, time, git commit and branch, and settings |

A run is identified by hashes of its settings: `RUN_ID` (`r` + 7 hex digits, of `prep` and
`binning`) names the folder and tags the files of everything from make-edges on, and `PREP_ID`
(`p` + 7 hex digits, of `prep` only) those of the prep inputs, which runs that differ only in
`binning` share. The same settings always give the same IDs. Every script adds its run to
`config/runs.csv`, so the settings behind any output folder or file can be looked up there:

```bash
ETUNC_RUN=config/runs/aw_bins20.yml pixi run bin-all-clim   # one script for one run
pixi run python -c "import etunc.config as c; c.summarize()"  # the IDs, paths and settings of the defaults
```

## TODO
- Allow data families to use their own valid area for binning, instead of using the common area mask.
- Perform sensitivity test to resolution, comparing the bin stats between native and 1° regridded resolution resolution.
