# et_unc

Research code for quantifying evapotranspiration (ET) uncertainty. Annual-mean
ET is binned in a 2-D space of climatological leaf area index (LAI) and aridity
index (AI = Rn / L·P), so that observational products (ILAMB, PML-V2.2,
GLEAM v4.3, SiTHv2), CMIP6 models and CESM2 ensembles can be compared bin by
bin.

The code runs on NCAR's glade (Casper/Derecho). Inputs and outputs live outside
the repo:

| | Path |
| --- | --- |
| Inputs (shared, read-only) | `/glade/campaign/univ/uwas0155/` (ILAMB and gridded obs, regridded obs and CMIP6) |
| CMIP6 catalogs | `/glade/derecho/scratch/bbuchovecky/cmip_intake_esgf_fetch/` (purgeable scratch; see `etunc/config.py`) |
| Outputs | `/glade/work/bbuchovecky/et_unc/{proc,fig}` |

All of these paths are set in `etunc/config.py`.

## Layout

| Path | What it holds |
| --- | --- |
| `etunc/` | The library (`import etunc.binning as binning`, ...). No module runs anything on import. |
| `etunc/config.py` | Paths and domain constants (`LAT_BNDS`, `LF_THRESH`, L = 2.45e6 J/kg, ...) |
| `etunc/units.py` | Unit conversions to W/m², net radiation for each data source |
| `etunc/temporal.py` | Annual means (complete years, missing-month rules), aggregation, period strings |
| `etunc/grid.py` | Target grids (`target_grid`), xESMF regridders, grid checks and snapping, land masks, cell areas |
| `etunc/binning.py` | The binning pipeline: inputs, quantile edges, bin statistics, post-processing |
| `etunc/plotting.py` | Maps, edge and histogram plots, bin heatmaps |
| `etunc/load/` | Loaders: `ilamb`, `obs` (PML/GLEAM/SiTH and their regridded files), `cmip`, `cesm`, `era5` |
| `etunc/legacy.py` | Older loaders and helpers that only the notebooks still use |
| `etunc/dask_cluster.py` | Start and stop a PBS dask cluster for notebooks |
| `scripts/` | Runnable pipelines, configured by the constants at the top of each file |
| `scripts/download/` | Download scripts for the obs products |
| `notebooks/` | Exploratory and analysis notebooks |
| `tests/` | pytest suite (synthetic data, no glade data needed) |
| `envs/` | Conda environments: `etunc.yml` (+ exact `etunc.lock.yml`), `openet.yml`, `lftp.yml` |

## Setup

```bash
mamba env create -f envs/etunc.yml          # or envs/etunc.lock.yml for the exact versions
mamba run -n etunc pip install --no-deps -e .
mamba run -n etunc python -m ipykernel install --user --name etunc   # Jupyter kernel for the notebooks
```

The package is installed editable, so changes under `etunc/` take effect
without reinstalling. `notebooks/check-disalexi.ipynb` uses the separate
`openet` env (`envs/openet.yml`).

## Running

```bash
PY=/glade/work/bbuchovecky/miniforge3/envs/etunc/bin/python

$PY scripts/bin_obs.py               # bin ET of every obs product combination (0.5°)
$PY scripts/bin_cmip.py              # bin ET of CMIP6 historical models (1°)
$PY scripts/make_mask.py             # common valid-data mask of the obs products
$PY scripts/obs_et_availability.py   # data availability of the obs ET products
qsub scripts/regrid_obs.pbs          # regrid GLEAM/PML/SiTH to 0.5° and 1° (Casper)
qsub scripts/regrid_cmip.pbs         # regrid CMIP6 fields to 1° (Casper)

$PY -m pytest                        # all tests (~7 min)
```

The scripts take no arguments: edit the settings at the top of each script (time
period, products or models, number of bins, ...). Each script's docstring lists
its steps and every output file it writes.
