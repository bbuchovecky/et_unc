# et_unc

Research code for quantifying evapotranspiration (ET) uncertainty. Annual-mean
ET is binned in a 2-D space of climatological leaf area index (LAI) and aridity
index (AI = Rn / Lv×P), so that observational products (ILAMB, ERA5, PML-V2.2,
GLEAM v4.3, SiTHv2), CMIP6 models and CESM2 ensembles can be compared bin by
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
| `etunc/config.py` | Paths and domain constants (`LAT_BNDS`, `LF_THRESH`, L = 2.45e6 J/kg, ...) |
| `etunc/units.py` | Unit conversions to W/m², net radiation for each data source |
| `etunc/temporal.py` | Annual means (complete years, missing-month rules), aggregation, period strings |
| `etunc/grid.py` | Target grids (`target_grid`), xESMF regridders, `regrid_annual` (the one regridding rule of every loader), grid checks and snapping, land masks, cell areas |
| `etunc/binning.py` | The binning pipeline: inputs, quantile edges, bin statistics, post-processing |
| `etunc/plotting.py` | Maps, edge and histogram plots, bin heatmaps |
| `etunc/load/` | Loaders: `ilamb`, `obs` (PML/GLEAM/SiTH), `cmip`, `cesm`, `era5`. Each has `load_native` (annual means on the native grid, land-masked) and `load_annual` (= `grid.regrid_annual` of it) |
| `etunc/legacy.py` | Older loaders and helpers that only the notebooks still use |
| `etunc/dask_cluster.py` | Start and stop a PBS dask cluster for notebooks |
| `scripts/` | Runnable pipelines, configured by the constants at the top of each file |
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

```bash
pixi run python scripts/bin_obs.py               # bin ET of every obs product combination (0.5°)
pixi run python scripts/bin_cmip.py              # bin ET of CMIP6 historical models (1°)
pixi run python scripts/bin_lens.py              # bin ET of the CESM2 Large Ensemble historical members (1°)
pixi run python scripts/bin_fppe.py              # bin ET of the CESM2 FHIST perturbed-parameter ensemble members (1°)
pixi run python scripts/make_mask.py             # common valid-data mask of the obs products
pixi run python scripts/obs_et_availability.py   # data availability of the obs ET products
pixi run bin-obs                                 # same as scripts/run_logged.sh scripts/bin_obs.py
                                                 # (also bin-cmip, bin-lens, bin-fppe, make-mask, obs-availability)
qsub scripts/legacy_regrid/regrid_obs.pbs        # deprecated: save GLEAM/PML/SiTH regridded to 0.5° and 1°
qsub scripts/legacy_regrid/regrid_cmip.pbs       # deprecated: save CMIP6 fields regridded to 1°

pixi run test                                    # all tests (~7 min)
```

The scripts take no arguments: edit the settings at the top of each script (time
period, products or models, number of bins, ...). Each script's docstring lists
its steps and every output file it writes. The pipelines regrid their inputs in
memory, so the two regrid jobs are not needed before running them. Every data
family goes onto the target grid by the same rule: annual means over complete
years on the native grid, masked to land there, then conservatively regridded
(`etunc.grid.regrid_annual`, `NA_THRESH`).

## Config
`etunc/config.py` contains the domain constants, such as `LF_THRESH` and `NA_THRESH`. Binning settings (e.g., `N_XBINS`, `N_YBINS`) exist there but they also can changed at the top of each run script. For consistent analysis, all run script should have the same binning settings.
