# et_unc

Research code for quantifying evapotranspiration (ET) uncertainty. ET is binned in a 2-D space of
climatological leaf area index (LAI) and aridity index (AI = Rn / L·P), so that observations, reanalysis,
CMIP6 models, CESM2 ensembles and TRENDY land models can be compared bin by bin.

The code runs on NCAR's Casper/Derecho. Full details of every step and assumption are in
[docs/binning-workflow.md](docs/binning-workflow.md).

## Data

Each data family is prepared separately and then binned on the same edges:

| Family | Data | Source |
| --- | --- | --- |
| `obs` | ET: ILAMB products, PML-V2.2, GLEAM v4.3, SiTHv2. LAI: GIMMS_LAI4g. P: GPCPv2.3. Rn: CERESed4.2 | `scripts/download/` |
| `era5` | ERA5 ET, P, Rn with obs LAI | NCAR GDEX |
| `cmip` | CMIP6 historical, one member per model | external `cmip-intake-esgf-fetch` catalogs |
| `lens` | CESM2 Large Ensemble | NCAR GDEX |
| `fppe` / `ippe` | CESM2 perturbed-parameter ensemble, coupled / offline CLM5 | campaign storage |
| `trendy_s3` | TRENDY (GCB2025) DGVMs; P from CRU-JRA forcing, Rn from ERA5 | `scripts/download/trendy/` |
| `plumber` | PLUMBER2 flux towers, binned by the obs LAI/AI at the nearest gridcell | — |

The products, models and members of each family are set in `config/pipeline.yml` (`prep.<family>`).

## Pipeline

```text
prep_<family>.py   steps 1-11   monthly native fields -> W/m2 -> annual means -> land mask -> ET outlier filter
                                -> conservative regrid (1°) -> shared complete years -> climatological LAI, P, Rn -> AI
make_edges.py      steps 12-13  common valid area of all families; LAI/AI quantile bin edges
                                (pooled_all: every family weighted equally; pooled_<family>)
bin_*.py           step 14      bin statistics of ET in each LAI-AI bin (bin_all: annual ET; bin_all_clim:
                                climatological ET; bin_area: land area; bin_plumber[_clim]: flux towers)
```

After step 14, three analyses are run separately: `regress_lai_aridity.py` (OLS of ET on LAI and AI),
`decompose_state_function.py` (state/function/interaction decomposition of area-mean ET against obs) and
`gridded_state_function_decomp.py` (the same decomposition mapped back to the lat/lon grid).

## Setup

```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi config set --global detached-environments /glade/work/$USER/pixi/envs   # keep the ~3 GB env out of home
export PIXI_CACHE_DIR=/glade/work/$USER/pixi/cache

cp config/paths.example.yml config/paths.yml   # edit the input/output paths for your machine
pixi install                                   # env from pixi.lock, with etunc installed editable
pixi run test                                  # synthetic-data tests (~6 min)
```

The PBS scripts (`scripts/*.pbs`) hard-code the project account and repo path; edit them for your
allocation.

## Reproduce

```bash
qsub scripts/run_all.pbs                                      # every prep, make-edges and bin step, default settings (~128 GB node)
qsub -v RUNS="default config/runs/aw_bins15.yml" scripts/run_all.pbs   # one or more runs (prep runs once per PREP_ID)
pixi run run-all-sweep                                        # defaults + every file in config/runs/

pixi run regress-lai-aridity                                  # analysis: needs bin-all and bin-all-clim
pixi run run-decomp                                           # analysis: needs the area-weighted runs config/runs/aw_bins{10,15,20}.yml
```

Each step is also a pixi task (`prep-obs`, `make-edges`, `bin-all`, ...; see `[tool.pixi.tasks]` in
`pyproject.toml`). `qsub -v FAMILY=obs scripts/prep.pbs` runs one prep script as a job. `SKIP_PREP=1`
makes `run_all.pbs` start at `make-edges`. Each task's log, with the git commit and settings, is saved in
`<work_root>/logs`.

## Runs and outputs

- `config/pipeline.yml` holds the default settings: period (2001–2014), grid, thresholds, products and
  members (`prep`), and the families, bin counts, binned variable and area weighting (`binning`).
- A file in `config/runs/<name>.yml` overrides only the keys that differ. Select it with
  `ETUNC_RUN=config/runs/<name>.yml`.
- Every run is identified by hashes of its settings. `PREP_ID` (`p…`) hashes `prep`, and `RUN_ID` (`r…`)
  hashes `prep` + `binning`. Each script records its run in `config/runs.csv`. The defaults are
  `p2abea0a` / `r183a58a`.
- Outputs go to `<work_root>/{proc,fig}/<id>/<family>/<kind>/`, with file names ending in `.<id>.nc`.
  Prep inputs are stored under the `PREP_ID`; everything from `make-edges` on is under the `RUN_ID`.

Print the IDs, paths and settings of the current run with
`pixi run python -c "import etunc.config as c; c.summarize()"`.

## Layout

| Path | Contents |
| --- | --- |
| `etunc/` | Library: loaders (`load/`), units, temporal aggregation, grids and regridding, edges, binning, regression, decomposition, plotting |
| `scripts/` | Pipeline and analysis scripts, configured by constants at the top of each file; `download/` |
| `config/` | Settings, run files, run register, paths template |
| `docs/` | Pipeline details (`binning-workflow.md`), obs coverage (`obs-availability.md`) |
| `tests/` | pytest suite on synthetic data |
| `notebooks/` | Exploratory and figure notebooks (`etunc` kernel: `pixi run kernel`) |
