# et_unc

Research code for quantifying evapotranspiration (ET) uncertainty. Annual-mean
ET is binned in a 2-D space of climatological leaf area index (LAI) and aridity
index (AI = Rn / Lv×P), so that observational products (ILAMB, PML-V2.2,
GLEAM v4.3, SiTHv2), ERA5, CMIP6 models and CESM2 ensembles can be compared bin by
bin.

The code runs on NCAR's glade (Casper/Derecho). Inputs and outputs live outside
the repo:

| | Path |
| --- | --- |
| Inputs (shared, read-only) | `/glade/campaign/univ/uwas0155/` (ILAMB and gridded obs, regridded obs and CMIP6) |
| CMIP6 catalogs | `/glade/derecho/scratch/$USER/cmip_intake_esgf_fetch/` (purgeable scratch; see `etunc/config.py`) |
| Outputs | `/glade/work/$USER/et_unc/{proc,fig}` |

All of these paths are set in `etunc/config.py`.

## Layout

| Path | What it holds |
| --- | --- |
| `etunc/` | The library (`import etunc.binning as binning`, ...). No module runs anything on import. |
| `etunc/config.py` | Paths, domain and pipeline constants (`TIME_SLICE`, `LAT_BNDS`, `LF_THRESH`, `NA_THRESH`, `MIN_ANNUAL_ET`, `FAMILIES`, L = 2.45e6 J/kg, ...), and the paths of the files passed between the pipeline scripts |
| `etunc/units.py` | Unit conversions to W/m², net radiation for each data source |
| `etunc/temporal.py` | Annual means and climatologies (`aggregate`: complete years, missing-month rules, missing-year report), shared years, period strings |
| `etunc/grid.py` | Target grids (`target_grid`), xESMF regridders (`regrid_with_na_thresh`), grid checks and snapping, native land masks (`landfrac_mask`, `landne_mask`, `mask_to_land`), the common area mask (`valid_area`, `common_area`), cell areas |
| `etunc/edges.py` | Aridity index, hierarchical weights and weighted quantile edges, saved edges |
| `etunc/binning.py` | ET outlier filter, bin statistics, post-processing |
| `etunc/regression.py` | OLS regression of ET on climatological LAI and AI, and the partition of R² between them |
| `etunc/plotting.py` | Maps, LAI/AI histograms, edge plots, bin heatmaps |
| `etunc/load/` | Loaders: `ilamb`, `obs` (PML/GLEAM/SiTH), `cmip`, `cesm`, `era5`. Each `load_native` returns raw monthly fields on the native grid in native units; the model loaders also have `load_land_fraction` |
| `etunc/legacy/` | Unmaintained, untested code kept for the notebooks and `scripts/legacy_regrid/`: the former annual loaders, replaced functions, notebook-era helpers |
| `etunc/dask_cluster.py` | Start and stop a PBS dask cluster for notebooks |
| `docs/binning-workflow.md` | The binning pipeline step by step, with every assumption and choice |
| `scripts/` | Runnable pipelines (`prep_*.py` → `make_edges.py` → `bin_all.py` or `bin_all_clim.py`; `regress_lai_aridity.py` after them), configured by the constants at the top of each file |
| `scripts/download/` | Download scripts for the obs products |
| `notebooks/` | Exploratory and analysis notebooks |
| `tests/` | pytest suite (synthetic data, no glade data needed) |
| `envs/` | Conda environments outside the pixi manifest: `openet.yml`, `lftp.yml` |
| `pixi.lock` | Exact versions of the `etunc` env (manifest in `pyproject.toml` under `[tool.pixi.*]`) |

## Setup

The `etunc` env is managed by [pixi](https://pixi.sh). One-time setup, so the
env (~3 GB) and the package cache go to /glade/work instead of home:

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
`pixi.lock`. `notebooks/check-disalexi.ipynb` uses the separate conda env
`openet` (`envs/openet.yml`).

## Running

The binning pipeline (described step by step in `docs/binning-workflow.md`) has three stages:

```bash
pixi run prep-obs         # steps 1-11: binning inputs of every obs product combination
pixi run prep-era5        # steps 1-11: binning inputs of ERA5 (ET, P, Rn) with obs LAI (GIMMS_LAI4g)
pixi run prep-cmip        # steps 1-11: binning inputs of the CMIP6 historical models
pixi run prep-lens        # steps 1-11: binning inputs of the CESM2 Large Ensemble historical members
pixi run prep-fppe        # steps 1-11: binning inputs of the CESM2 FHIST perturbed-parameter ensemble
pixi run prep-ippe        # steps 1-11: binning inputs of the CESM2 offline (CLM5) perturbed-parameter ensemble
qsub -N prep_obs -v FAMILY=obs scripts/prep.pbs  # any prep-* as a Casper job (FAMILY: obs, era5, cmip, lens, fppe, ippe)
pixi run make-edges       # steps 12-13: common area mask and pooled LAI/AI edges (needs every prep-*)
pixi run bin-all          # step 14: binned ET of every family with the pooled_all and pooled_<family> edges
pixi run bin-all-clim     # step 14 with climatological ET: one sample per gridcell instead of per gridcell-year
pixi run bin-plumber      # step 14 for the PLUMBER2 flux towers: tower ET binned by the obs LAI/AI at the nearest gridcell
pixi run bin-plumber-clim # step 14 for the PLUMBER2 flux towers with climatological ET: one sample per tower (needs bin-plumber)
pixi run make-mask        # common valid-data mask of the obs products
pixi run obs-availability # data availability of the obs ET products
qsub scripts/legacy_regrid/regrid_obs.pbs        # deprecated: save GLEAM/PML/SiTH regridded to 0.5° and 1°
qsub scripts/legacy_regrid/regrid_cmip.pbs       # deprecated: save CMIP6 fields regridded to 1°

pixi run test             # all tests (~6 min)
```

Each task is `scripts/run_logged.sh scripts/<name>.py`, which logs the output,
the git commit, `etunc/config.py` and the script's settings to `WORK_ROOT/logs`.
The scripts take no arguments: edit the settings at the top of each script
(products, models, members, ...) or the shared ones in `etunc/config.py` (time
period, grid, thresholds, number of bins). Each script's docstring lists every
output file it writes. Every family runs the same steps, written out in the
script's `main()`; only the loader (step 1), the unit conversion (step 3) and
the land mask (step 5) differ.

## To reproduce results

```bash
qsub -N prep_obs -v FAMILY=obs scripts/prep.pbs
qsub -N prep_cmip -v FAMILY=cmip scripts/prep.pbs
qsub -N prep_fppe -v FAMILY=ippe scripts/prep.pbs
qsub -N prep_ippe -v FAMILY=ippe scripts/prep.pbs
qsub -N prep_lens -v FAMILY=lens scripts/prep.pbs
pixi run make-edges && pixi run bin-all && pixi run bin-all-clim && pixi run bin-plumber && pixi run bin-plumber-clim
```

## Config
`etunc/config.py` contains the domain constants, such as `LF_THRESH` and `NA_THRESH`.
Binning settings (e.g., `N_XBINS`, `N_YBINS`) exist there but they also can changed at
the top of each run script. For consistent analysis, all run script should have the
same binning settings.

## TODO
- Allow data families to use their own valid area for binning, instead of using the common area mask.
- Perform sensitivity test to resolution, comparing the bin stats between native and 1° regridded resolution resolution.
