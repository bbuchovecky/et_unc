# Refactor et_unc into an `etunc` package plus `scripts/`

## Context
The repo grew script by script, so shared logic now sits inside driver scripts, and scripts import each other.
`mask.py`, `obs_et_availability.py` and `obs_binned_et.py` import `ilamb_binned_et` (`obs_binned_et` even calls its
`run()`); `test_regrid.py` imports four scripts; `test_regrid_obs.py` tests functions of the `regrid_obs` script;
the `regrid-nan-thres` notebook uses `regrid_obs`; and `cmip-regrid-res` imports `cmip_binned_et` as a module
(about 20 helpers and constants) and re-implements `load_model` and `target_grid`. `binned_et.py` (1,336 lines) mixes
constants, units, time aggregation, binning, plotting and three families of loaders. A newcomer can't tell library
code from runnable code, or where a function should go.

Duplication found (all of it gets one home):

| Code | Copies now |
|---|---|
| `complete_years`, `annual_mean`, `land_mask`, `hatch_bins` | `ilamb_binned_et.py` and `cmip_binned_et.py`, byte-identical. The ET-vs-LAI rule is the `require_all_months` argument, so the CLAUDE.md note that they "differ deliberately" is out of date |
| `period_str` | both binning drivers (only the docstrings differ) |
| `valid_area`, `common_area` | both binning drivers; CMIP adds `.all("member")` |
| `save_map`, `MAP_KWARGS`, `EDGE_ATTRS`, `N_XBINS`/`N_YBINS`, `SIGNIF_ALPHA`/`SIGNIF_N_MIN`, output roots | both binning drivers |
| `check_coords` / `equal_coords` | `binned_et.py`, `load_cmip_esgf.py`, `CMIPESGFLoader._check_coords`. **Not** identical: diff them before merging |
| `to_yyyymm` | `binned_et.py`, `regrid_cmip_esgf.py` (identical) |
| Monthly ILAMB load (open → `format_grid` → fill → regrid → snap to grid) | `ib.load_product`, `mask.load_ilamb`, `obs_et_availability.load_ilamb` |
| Gridded obs load (`load_obs` → units → annual → snap to grid) | `mask.load_gridded`, `obs_et_availability.load_gridded`, `obs_binned_et.load_gridded` |
| `on_grid` (check, then snap to the target grid) | `mask.py`, `obs_et_availability.py` |
| `regridded_dataset` | `obs_et_availability.py` (superseded by `load_obs(res=...)`) |
| ILAMB binning steps 2–4 (`ib.run`) | `ilamb_binned_et.py`, called from `obs_binned_et.py` |
| Native CMIP load + conservative regrid (`load_model` in two halves); `target_grid` for other spacings | `cmip_binned_et.py`, re-implemented in `cmip-regrid-res.ipynb` |
| Facet heatmap of bin means (edge ticks or quantile ticks) | `plot_model_bin_means`, `plot_combo_bin_means` |
| Facet map grid; mask-agreement map | `obs_et_availability.facets` / `plot_map_mask_agreement`, `mask.plot_product_masks` / `plot_mask_agreement` |
| Cell areas | `ilamblib.CellAreas`, called from `be.compute_cell_area`, `obs-et.ipynb`, `binned_stats.ipynb` |

These are *not* duplicates and keep separate names: `format_grid` (ILAMB driver only); and `regrid_to_target`, which
in `ilamb_binned_et` is bilinear and in `regrid_cmip_esgf` is conservative with the EC-Earth lat/lon fix.

Intended outcome: one obvious module per concept, no script imports another script, nothing needs ILAMB/MPI, and
behavior is unchanged (pinned by the tests). The code runs in its own mamba env, `etunc`,
instead of the shared `data-sci-py312`. Pixi is out of scope; it's a later, separate step, and pixi can import
`envs/etunc.yml`.

Out of scope, but flagged: the CMIP catalogs and about 2/3 of the file paths in `cmip_evap.csv` are in purgeable
`/glade/derecho/scratch/bbuchovecky/cmip_intake_esgf_fetch/`. `config.py` centralizes these paths, so a later move
to campaign storage is a one-line change.

## Dedicated environment `etunc`
`envs/etunc.yml`: channels `conda-forge` and `nodefaults`, created at `/glade/work/bbuchovecky/miniforge3/envs/etunc`.
- **conda-forge packages:** `python=3.12`, `numpy`, `pandas`, `scipy` (`test_significance`), `xarray`, `dask`, `distributed`, `dask-jobqueue` (needed by `dask_cluster`), `netcdf4`, `xesmf` (brings `esmpy`/`esmf`), `regionmask`, `cartopy`, `matplotlib`, `pytest`, `ipykernel`, `pip`.
- **Pinned versions:** `numpy`, `pandas`, `scipy`, `xarray`, `dask`, `xesmf`, `esmpy`, `regionmask` and `cartopy` are pinned to the versions now in `data-sci-py312` (read with `conda list` at implementation time; e.g. xarray 2026.1.0, numpy 2.3.5, xesmf 0.9.2, esmpy 8.9.1, regionmask 0.13.0). Then any difference after the refactor comes from the code, not from version changes. The pins can be loosened once it passes.
  As built, `envs/etunc.yml` also pins packages that change results without being imported directly: xarray's optional accelerators `bottleneck`, `numbagg` (+`numba`) and `flox`; `cftime`, `netcdf4`, `libnetcdf`, `hdf5`; `shapely`, `geopandas`, `pyproj`, `proj`; `libopenblas`; and `matplotlib`, `freetype` (PNG rendering). `esmf`, `hdf5` and `libnetcdf` are pinned to `nompi_*` builds, in the yml and in the lock.
- **pip:** none besides `etunc` itself (the repo no longer depends on xclimate), installed after the env is created with `mamba run -n etunc pip install --no-deps -e .`.
- **Lock file:** `mamba env export -n etunc --no-builds > envs/etunc.lock.yml`, committed alongside the yml.
- **Jupyter kernel:** `python -m ipykernel install --user --name etunc`.
- **Left out on purpose:**
  - ILAMB and mpi4py: their only use was `ilamblib.CellAreas`, which is ported to `grid.cell_area`.
  - `ee`/`geemap`: `check-disalexi.ipynb` keeps the existing `openet` env.

  Every other notebook, `binned_stats` included, runs in `etunc`. Each notebook's first cell gets a one-line note
  naming its env.
- `data-sci-py312` is not modified.

## Target layout
```
et_unc/
  pyproject.toml        setuptools metadata for `etunc` (deps listed for documentation; installed with --no-deps);
                        package find limited to `etunc*`; [tool.pytest.ini_options] testpaths = ["tests"]
  envs/etunc.yml        dedicated env (+ etunc.lock.yml); existing lftp.yml / openet.yml stay
  README.md             layout table, how to install and run
  etunc/
    config.py           paths and domain constants (see below)
    grid.py             target grids and regridders (all of regrid.py), with target_grid(res) for any spacing
                        that divides 180 (RESOLUTIONS/grid_tag stay the standard 0.5°/1° output grids);
                        format_grid; check_coords, equal_coords, check_same_grid; on_grid(da, res); land_mask,
                        mask_greenland; cell_area (numpy port of ilamblib.CellAreas); regrid_with_na_thres
                        (regrid_obs.regrid + NA_THRES); valid-month product masks (product_mask, common_mask from
                        mask.py)
    units.py            convert_units, latent_heat_to_wm2, accumulation_to_flux, mm/day handling (ib.to_wm2),
                        net_radiation_cmip/cesm/era5
    temporal.py         complete_years, annual_mean(require_all_months), yearly_to_annual, compute_annual_mean,
                        aggregate, format_time_period, period_str, to_yyyymm, on_month_axis
    binning.py          compute_aridity_index, prepare_inputs, build_edges, pooled_bin_edges, finite_flat
                        (was _finite_flat), bin_stats, bin_stats_by_source, pool_members, shared_years,
                        valid_area/common_area(..., member_dim=None), drop_zero_width_bins, frac_count, frac_valid,
                        test_significance, ensemble_spread, STATS, EDGE_ATTRS, N_XBINS/N_YBINS,
                        SIGNIF_ALPHA/SIGNIF_N_MIN
    plotting.py         finish (was _finish), map_ax (was _map_ax), add_gridlines, facets, quick_map, save_map,
                        MAP_KWARGS, plot_input_*, plot_edges*, plot_bin_field/facets/summary,
                        ONE plot_bin_means(bs, dim, hatch=None), hatch_bins, plot_mask_agreement.
                        No module calls matplotlib.use; only scripts set Agg
    load/
      obs.py            load_obs.py as is (ObsDataset registry), importing accumulation_to_flux from units; plus
                        output_path and regrid_file from regrid_obs.py, so the reader (load_obs(res=)) and the
                        writer share one path layout
      ilamb.py          ILAMB_DATA_ROOT, PRODUCTS, FILL_THRESH; load_ilamb(variable, product, time_slice, res)
                        returns monthly data on the grid (regrids each month); load_ilamb_annual = old
                        ib.load_product, which takes annual means *before* regridding, so it must not be built on
                        load_ilamb (pinned by test_load_ilamb.py); load_gridded_annual (the three load_gridded
                        copies)
      cmip.py           CMIPESGFLoader (uses grid.check_coords); available_members, select_members, sftlf_files,
                        member_tag, load_land_fraction; EC-Earth lat/lon fix and conservative regrid_to_target
                        (from regrid_cmip_esgf); load_model split into
                          load_native(loader, sid, members, sftlf_path, variables, experiment_id, time_slice)
                            -> (ann, native_mask, src_grid)
                          regrid_annual(ann, src_grid, res, mask, na_thres)
                          load_model = regrid_annual(*load_native(...)) + member_id coord + summary print
      cesm.py           load_cesm.py as is (load_fhist_ppe, load_goga2, load_cesm2le, load_grid,
                        ppe_member_name), plus load_cesm_grid, load_cesm_variable from binned_et
      era5.py           load_era5.py as is (load_era5, load_era5_grid)
    dask_cluster.py     dask_cluster.py as is (create_dask_cluster, close_dask_cluster)
    legacy.py           notebook-only code, not used by any script: load_ilamb_obs, compute_cell_area (now via
                        grid.cell_area), load_cmip (used by 5 notebooks; its sys.path.insert(__file__) hack is
                        dropped, so call this out in the commit), filter_all_variables_available,
                        open_bin_stats/_ensure_bin_coords (notebook-era files), data_dict_nybtes, align_dicts
  scripts/              settings at the top + script-specific steps + main(); no reusable code
    bin_obs.py (was ilamb_binned_et.py + obs_binned_et.py)   bin_cmip.py (was cmip_binned_et.py)
    make_mask.py (was mask.py)               obs_et_availability.py
    regrid_obs.py + .pbs                     regrid_cmip.py + .pbs (was regrid_cmip_esgf)
    download/  download_ilamb.py (add __main__ guard), download_*.sh
  notebooks/            all *.ipynb
  tests/                test_binning.py (was test_binned_et.py), test_grid.py (was test_regrid.py),
                        test_load_obs.py, test_regrid_obs.py, test_load_ilamb.py, test_load_cmip.py (step 0),
                        test_temporal.py (new), data/ (tiny fixtures)
  logs/                 *.log and PBS *.o<jobid> (gitignored; none are tracked)
```

`scripts/bin_obs.py` settings: `ILAMB_PRODUCTS` (replaces `RUN_PRODUCTS`), and `GRIDDED_ET_PRODUCTS` plus
`MIN_ANNUAL_ET` from `obs_binned_et` (`GRIDDED_ET_PRODUCTS = {}` gives the ILAMB-only run). `main()` is the old
`main` and `run` written out in one place: load ILAMB → optionally load gridded ET over the years shared by the
(lai, pr, rns) products → maps → combos → area mask → edges → binning → plots. `concat_combos` and `concat_models`
stay in their scripts, since each is used once.

`config.py` contents:
- **Inputs:** `CAMPAIGN_ROOT` (`/glade/campaign/univ/uwas0155`), `OBS_ROOT`, `ILAMB_ROOT`, the regridded-file roots `OBS_REGRID_ROOT` and `CMIP_REGRID_ROOT` (written by `regrid_obs`/`regrid_cmip` and read by `load_obs(res=)`), and the CMIP catalogs: `CMIP_CATALOG_ROOT`, `CMIP_CATALOG`, `CMIP_FX_CATALOG`, `ESGF_CACHE_CATALOG` (the last is used by `regrid_cmip`). A comment notes the scratch purge risk.
- **Outputs:** `WORK_ROOT` (`/glade/work/bbuchovecky/et_unc`), plus `PROC_ROOT`, `FIG_ROOT`, `BIN_EDGES_ROOT`.
- **Constants:** `LAT_BNDS`, `LF_THRESH`, `LATENT_HEAT_VAPORIZATION`, `LIQ_WATER_DENSITY`, `DPI`, `PROJECTION`.

Conventions for every module:
- An opening docstring that says what belongs in it.
- `import etunc.grid as grid` style imports.
- Drivers keep `main()` written out step by step; that's the readable recipe. Only the helpers are deduplicated.
- No behavior changes. Any unavoidable change is called out in its commit message.

## Git workflow
- On `main`: commit this plan and CLAUDE.md, run `pytest` and record the result, then tag `pre-refactor` and push the tag. If the refactored code looks wrong, restart from that tag.
- Work on branch `refactor/package`, pushed to origin as a backup. Freeze feature work on `main` until the merge. If `main` must change, merge it into the branch.
- **Every commit is green:** `pytest` passes and every script still imports. A commit that moves or merges code updates the tests of that code in the same commit (there is no separate "fix the tests" step). The full suite takes ~7 min, so while working run only the tests of the module being moved, and run the full suite before each commit.
- One commit per move or dedup, not one per step. Steps 3–5 contain many of them.
- Merge with `git merge --no-ff refactor/package` (not squash), so the per-commit history and its behavior-change notes survive for `git bisect` and `git log --follow`.

## Steps (on branch `refactor/package`)
0. **Safety (done on `main`).** Commit 06aa43d added the tests below. They pin the current behavior of code that the refactor merges or splits, using synthetic data against the old scripts. Later commits re-point their imports without changing the expected values:
   - `test_load_ilamb.py`, the ILAMB loaders that become `load_ilamb`/`load_ilamb_annual`. Uses a fake 1° product (lon 0–360, lat descending, a missing month, a ~1e37 fill value, mm/day units, a partial final year):
     - `ib.load_product` keeps complete years only, takes the annual mean *before* the bilinear regrid, and the test shows the other order gives different values; a missing LAI month counts as 0, while pr/ET years with a missing month are NaN.
     - Fill values become NaN, mm/day becomes W/m2, and the output has the exact 0.5° coordinates within `LAT_BNDS`. A 0.5° product off by round-off is snapped onto the grid; one off by more than the tolerance raises.
     - The monthly loaders `mask.load_ilamb` (native units, partial years kept) and `obs_et_availability.load_ilamb` (W/m2, reindexed to `MONTHS`) regrid each month.
   - `test_load_cmip.py`, member selection: `CMIPESGFLoader.sort_member_ids` (numeric r/i/p/f order, invalid IDs raise), `group_member_ids_by_ipf`, `available_members` (intersection over `VARIABLES`, experiment filter), `select_members` in every mode (top, all, max_r and its tie-break, explicit list with missing members dropped, unknown mode raises), `sftlf_files` (duplicates raise), `member_tag`.
   - `test_load_cmip.py`, `load_model` with a stub loader that returns `load_data`'s structure on a synthetic 2.25° native grid, checking:
     - the native sftlf mask (Greenland excluded), with % and fraction sftlf files giving identical output;
     - complete years only; a missing LAI month counts as 0, while a missing ET month makes that year NaN;
     - `NA_THRES` (a target cell with 75% land is kept, one with 25% is NaN), the target `mask`, unit conversions, net radiation, and dims, coordinates and names;
     - a sftlf file on another grid raises.

   Recorded result in `data-sci-py312` on 2026-10-08: 150 passed in 7 min 05 s. The empty `cmip_trends.py` and `obs_trends.py` were already deleted in 55dfd8a.
1. **Environment and skeleton.**
   - Write `envs/etunc.yml` and create the `etunc` env (see "Dedicated environment").
   - Create `etunc/__init__.py`, `pyproject.toml` and `config.py`, then install the package editable into `etunc`. Add `*.egg-info/` and `.pytest_cache/` to `.gitignore`.
   - Run the *old* test suite in `etunc` to confirm the env reproduces `data-sci-py312` results before any code moves.
   - Commit `envs/etunc.yml` and `envs/etunc.lock.yml`.
   - Update the `#PBS` scripts and CLAUDE.md commands to `PY=/glade/work/bbuchovecky/miniforge3/envs/etunc/bin/python`.
2. **Move the modules that are already self-contained**, using `git mv` to keep history: `regrid.py`→`etunc/grid.py`, `load_obs.py`→`etunc/load/obs.py`, `load_cmip_esgf.py`→`etunc/load/cmip.py`, `load_cesm.py`→`etunc/load/cesm.py`, `load_era5.py`→`etunc/load/era5.py`, `dask_cluster.py`→`etunc/dask_cluster.py`. In the same commit, fix their imports in every script, test and module that uses them.
3. **Split `binned_et.py`** into `config`, `units`, `temporal`, `binning`, `plotting`, `load/cesm` (appended to the moved `load_cesm.py`) and `legacy`, with the public renames (`finish`, `map_ax`, `finite_flat`). As done: every helper used across modules became public (`_edge_ticks` → `set_edge_ticks`, since `plot_bin_field` has an `edge_ticks` argument; `_ensure_bin_coords`, `_as_field_dict`, `_edges_and_attrs` lose their underscore), and `ensure_bin_coords` stays in `binning`, which uses it, rather than `legacy`. Package modules import each other with `from etunc.x import name`, since module aliases like `units`/`grid`/`legacy` collide with existing local names. `data_dict_nybtes`/`align_dicts` move to `legacy` in the step 4 commit that reconciles `check_coords`/`equal_coords`, because `align_dicts` uses `load/cmip.py`'s own `equal_coords`. Add `grid.cell_area` and route `legacy.compute_cell_area` through it, with tests (the sum equals 4πR² on a global grid; bounds and midpoint edges give ILAMB's formula on a coarse grid). Delete `binned_et.py`. Re-point the `be.` imports in the scripts and `test_binned_et.py` in the same commit(s).
4. **Deduplicate the driver helpers into the package** (table above), one commit per row. Each commit adds synthetic tests for the helper it moves into the package, re-points the imports in `test_load_ilamb.py`/`test_load_cmip.py` without changing their expected values.
   As done (11 rows, `fd085e7`..`ea944c0`): each commit also deletes the drivers' copies and calls the package, so the drivers exercise the package code at once (step 5 is left with moving the files). Deviations: `N_XBINS`/`N_YBINS`/`SIGNIF_*` stay in the scripts as per-pipeline settings; the "three `load_gridded` copies" were three different functions, so only their shared pieces moved (`temporal.yearly_to_annual`, `grid.on_grid`, `units.accumulation_to_flux`) and `obs_binned_et.load_gridded` stays a script function; `shared_years` went to `temporal`; `common_mask` is a one-line expression and stays in `mask.main`; `ib.to_wm2` became `units.flux_to_wm2`. The plotting merges (`save_map`, `plot_bin_means`, `plot_mask_agreement`) were checked with a pixel-exact before/after render of every affected figure from synthetic data, and by eye.
   - Rows: grid/time helpers, `load/ilamb.py`, CMIP member selection, the `load_native`/`regrid_annual`/`load_model` split, `valid_area`/`common_area`, the masks from `mask.py`, `regrid_with_na_thres`, `output_path`/`regrid_file`, the general `target_grid`, `facets`, `plot_mask_agreement`.
   - New tests in those commits: `format_grid`, `complete_years`, `annual_mean` with both settings, `yearly_to_annual`, `on_grid`, `product_mask`/`common_mask`, `to_yyyymm`, `period_str`; `target_grid(0.25)` and `target_grid(2.0)` shapes and edges, while `grid_tag` still rejects them; `regrid_annual` on a synthetic native grid (`na_thres` masking, target-grid coordinates) and `regrid_annual(*load_native(...))` equal to `load_model`; `valid_area`/`common_area` with and without `member_dim`.
   - Module constants that become arguments (`MEMBER_IDS`, `DEFAULT_MEMBERS`, `VARIABLES`, `NA_THRES`, …) are passed explicitly by the tests instead of monkeypatched.
   - `plot_bin_means` (merging `plot_combo_bin_means` and `plot_model_bin_means`) and `save_map` are not true duplicates: different path patterns, titles, masking and tick logic. Give each its own commit, and check the affected figures by eye after it.
   - The EC-Earth i/j branch of `regrid_cmip_esgf._format_lat_lon` is not reached by the EC-Earth files on disk (all 1-D lat/lon), so it gets a synthetic unit test when it moves to `load/cmip.py`.
   - For `check_coords`/`equal_coords`, keep one version only where it changes neither caller; otherwise keep both, with names that say how they differ.
5. **Move and slim the scripts.** `git mv` them into `scripts/` with the new names and replace their helper copies with package imports. Merge `obs_binned_et.py` into `bin_obs.py`, then delete it. `obs_et_availability.py` drops `regridded_dataset` and uses `load_obs(res=...)`. Replace `test_scripts_use_target_grids` (scripts are no longer importable, and the grids now have one source) in the same commit.
   As done: output roots, catalogs and regrid roots in the scripts come from `etunc.config` (same values); `obs_et_availability.py` has its own `TARGET_RES = 0.5` instead of reading `ilamb_binned_et`'s; `bin_obs.main` is the old `ilamb_binned_et.main` + `run` and `obs_binned_et.main` in one function, and its default settings reproduce the old `obs_binned_et` run (`GRIDDED_ET_PRODUCTS = {}` reproduces `ilamb_binned_et`); the `.pbs` files `cd` to the repo and run `scripts/...`; `download_ilamb.py` has a `__main__` guard; `etunc.load.obs` takes `OBS_ROOT`/`REGRID_ROOT` from `etunc.config`. `test_scripts_use_target_grids` became a check that `load_obs`'s resolution tags match `grid.RESOLUTIONS`.
6. **Move the tests** into `tests/` (`git mv`, pure move) and set `testpaths`. `test_regrid_obs.py` gets `output_path`/`regrid_file` from `etunc.load.obs` (this already happened in the step 4 commit that moved them).
   As done: `test_binned_et.py` → `tests/test_binning.py`, `test_regrid.py` → `tests/test_grid.py`, the others keep their names; only their "run with" lines changed. No `tests/data/` was needed (the ILAMB `CellAreas` reference arrays were dropped with step 1b).
7. **Notebooks.** Move them to `notebooks/` and edit only their import cells (and the cells that held copies of package code). Set every kernel to `etunc` except `check-disalexi` (stays on `openet`).
   - `cmip-regrid-res`: `cbe.*`/`be.*`/`rg.*` → package. Its own `target_grid`, `conservative_regridder`, `load_native` and `regrid_annual` → `grid.target_grid`, `grid.bounded_conservative_regridder`, `load.cmip.load_native`, `load.cmip.regrid_annual`. Settings such as `TIME_SLICE`/`VARIABLES` are copied from `bin_cmip.py` into its settings cell. Drop the workaround that restored inline figures after `matplotlib.use("Agg")`.
   - `check-cmip-esgf`, `check-cmip-grids`, `cmip-et`, `load-cmip`: `load_cmip_esgf` → `etunc.load.cmip` (`data_dict_nybtes`/`align_dicts` → `etunc.legacy`), `regrid` → `grid`, `be.*` → `units`/`grid`/`config`/`legacy` (`load_cmip`).
   - `agu-abstract`: `load_cesm` → `etunc.load.cesm`.
   - `compare-ilamb`: `be.latent_heat_to_wm2` → `units`.
   - `obs-et`: `ilamblib.CellAreas` → `grid.cell_area`, `load_obs` → `etunc.load.obs`, `load_era5` → `etunc.load.era5`.
   - `regrid-nan-thres`: `lo`/`rg`/`ro` → `load.obs`, `grid` (`regrid_with_na_thres`, `NA_THRES`, `RESOLUTIONS`).
   - `binned_stats`: `ilamblib.CellAreas` → `grid.cell_area`; `dask_cluster`, `load_cesm` and `load_cmip_esgf` → package; `be._bin_stats_flat`/`build_edges` → `binning`; `be.load_ilamb_obs`/`filter_all_variables_available`/`load_cmip` → `legacy`.

   A scan found only `ROOT / "x.csv"`-style paths in the notebooks, which don't depend on the working directory. Re-grep for bare `open(`/`savefig(` filenames before moving.
   As done: every notebook gets a first markdown cell naming its env; only code cells that referenced moved names changed (cell outputs untouched), and every import statement and every `etunc` attribute the notebooks use was checked to exist in `etunc`. `cmip-regrid-res` keeps a local copy of `concat_models` (it stays in `scripts/bin_cmip.py`), and its settings cell copies `bin_cmip.py`'s values. `check-disalexi` keeps its `python3` kernelspec, since no `openet` Jupyter kernel is registered; its env cell names `openet`. Not done yet: re-running `cmip-regrid-res` on real data to compare its figures (heavy: run it as a batch job).
8. **Docs.** Rewrite `README.md` with the layout table and a quickstart. Update `CLAUDE.md`: layout, commands (`scripts/…`, `pytest tests/`), remove the out-of-date "differ deliberately" note, replace "`rg.target_grid` is the only source of the two common grids" (it now builds any spacing, and the standard grids are `RESOLUTIONS`), drop ILAMB from the env description, note the scratch-catalog risk, and remove the refactor exceptions from the hard rules. Update the scripts' docstrings with their new names.
   As done: README.md has the paths, a layout table, setup and run commands; CLAUDE.md has the layout pointer, the notebooks' location and env, the scratch-catalog risk, and the hard rules without the refactor exceptions (most of its other updates landed with steps 2–6).

## Verification
All checks run in the `etunc` env. There is no output-regression baseline: the tests pin behavior, and if the
refactored code looks wrong, restart from the `pre-refactor` tag.
- Env parity: the old test suite passes in `etunc` before any refactoring (step 1).
- Every commit: `pytest` passes.
- `pytest tests/` passes: every existing test plus the new ones. `test_load_ilamb.py` and `test_load_cmip.py` pass with the same expected values they had against the old scripts.
- Every library module imports in a fresh interpreter, ILAMB isn't needed anywhere, and MPI never starts: `python -c "import etunc.grid, etunc.units, etunc.temporal, etunc.binning, etunc.plotting, etunc.load.obs, etunc.load.ilamb, etunc.load.cmip, etunc.legacy"`.
- Before merging, run the new scripts once and look over their outputs and figures, in particular the figures affected by the `plot_bin_means`/`save_map` merges. Re-run `cmip-regrid-res` top to bottom and compare its figures with the committed outputs.
- `grep -rE "import (binned_et|ilamb_binned_et|cmip_binned_et|obs_binned_et|load_obs|load_cmip_esgf|regrid_obs|regrid)\b|from (load_cmip_esgf|regrid|ILAMB) |ilamblib"` over `etunc scripts tests notebooks` finds nothing.
- Each moved notebook's import cell runs without errors.
- `qsub` dry check: the `.pbs` files point at `scripts/…` paths, which exist.
