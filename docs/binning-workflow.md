## Binning Workflow
Every data family (obs, ERA5, CMIP6, LENS2, FHIST PPE, IHIST PPE) runs the same numbered steps.
They are written out in `main()` of each script. The scripts differ only in
step 1 (which loader), step 3 (which unit conversion) and step 5 (which land mask).

```
prep_{obs,era5,cmip,lens,fppe,ippe}.py                 steps 1-11   -> PROC_ROOT/<prep_id>/<family>/inputs/<family>.inputs.<prep_id>.nc
make_edges.py                                           steps 12-13  -> common area mask, pooled LAI/AI edges
bin_all.py                                              step 14      -> binned ET of every family
bin_all_clim.py                                         step 14      -> binned climatological ET of every family
bin_area.py                                             step 14      -> land area in each bin, for every family
bin_plumber.py                                          step 14      -> binned ET of the PLUMBER2 flux towers
bin_plumber_clim.py                                     step 14      -> binned climatological ET of the PLUMBER2 flux towers
```
`prep_trendy.py` (TRENDY DGVMs, one family per experiment: `trendy_s2`, `trendy_s3`) runs steps 1-11 too (see [TRENDY](#trendy)), and `run_all.sh` runs it for each experiment of `prep.trendy.experiments` (default S3, the family `trendy_s3`), which is in `FAMILIES`, so it enters the common area mask and the pooled_all edges (`trendy_s2` does not).
Run them in that order. `scripts/run_all.sh` (`pixi run run-all`, or `qsub scripts/run_all.pbs` as one Casper job) runs all of them, for each run of `RUNS` (below).
`regress_lai_aridity.py`, `decompose_state_function.py` and `gridded_state_function_decomp.py` are analyses of the step 14 outputs, not part of the pipeline (see [analyses outside the pipeline](#analyses-outside-the-pipeline)).

**Settings and runs.** Every setting shared by the scripts is in `config/pipeline.yml`, in three sections: `prep` (steps 1-11: `time_slice`, `target_res`, `min_years`, `min_annual_et`, `lat_bnds`, `na_thresh`, `lf_thresh`, and each family's products, models and members in `prep.obs`, `prep.era5`, `prep.cmip`, `prep.lens`, `prep.fppe`, `prep.ippe` and `prep.trendy`), `binning` (steps 12-14 and the analyses: `families`, `bin_variable`, `area_weighted`, `n_xbins`, `n_ybins`, `signif_alpha`, `signif_n_min`) and `plotting` (`dpi`, `hist_bins`). `etunc/config.py` reads it and exposes each setting as a constant (`prep.time_slice` -> `cfg.TIME_SLICE`, `binning.n_xbins` -> `cfg.N_XBINS`, ...). A run changes some of them in its own file, `config/runs/<name>.yml`, holding only the keys that differ (a misspelt key is an error; a mapping inside a family, such as `prep.cmip.member_ids` or `prep.trendy.omit_models`, is replaced whole), selected with the environment variable `ETUNC_RUN=config/runs/<name>.yml`; without it, a run has the defaults. The paths are in `config/paths.yml` (not in git; a copy of `config/paths.example.yml`, or the file named by `ETUNC_PATHS`). The product, model and member choices are part of `prep`, so of the PREP_ID: changing one family's gives every family a new PREP_ID. Each prep script assigns them to its own constants at its top (`ILAMB_PRODUCTS = cfg.OBS_ILAMB_PRODUCTS`, ...), next to the settings particular to it (variables, file streams, ...).
- Each run has two IDs, hashes of its settings (`etunc.config`): `PREP_ID` = "p" + 7 hex digits of the `prep` settings, and `RUN_ID` = "r" + 7 hex digits of the `prep` and `binning` settings. `plotting` and the paths are not hashed. The same settings always give the same IDs, so rerunning a run overwrites its own outputs, and runs that differ only in `binning` share their prep inputs.
- Every script registers its run in `config/runs.csv` (`cfg.register_run`, next to `cfg.summarize`): run and prep IDs, run file, time, git commit and branch, the key settings and the full hashed settings as JSON. Every script prints the IDs, paths and settings, and its own settings, into its log (`scripts/run_logged.sh`, whose log names carry the run ID).
- `run_all.sh` takes `RUNS` = run files separated by spaces (`default` for the defaults; e.g. `qsub -v RUNS=config/runs/aw_bins20.yml scripts/run_all.pbs`), runs the prep scripts once per `PREP_ID` (none with `SKIP_PREP=1`) and the bin scripts for every run. `pixi run run-all-sweep` runs the defaults and every file of `config/runs/` (10, 15 and 20 bins, unweighted and area-weighted).
- `binning.bin_variable` chooses what step 14 bins: `et`, annual mean ET, or `et_pr`, the annual evaporation ratio ET/P (step 11b, which the prep scripts always save). Outputs are named `<family>.<bin_variable>.qbin_*` and `<family>.<bin_variable>_clim.qbin_*`.
- `binning.area_weighted` makes `bin_all.py` and `bin_all_clim.py` weight each sample by its gridcell's land area (`rg.land_area`, one field for every family) in the bin statistics.

**Output layout.** Every output, under `PROC_ROOT` (netCDF) and `FIG_ROOT` (figures) alike, is in `<root>/<id>/<family>/<kind>/` (`cfg.output_dir(root, family, kind, id)`), and its file name ends in `.<id>.<ext>`. `<id>` is the `PREP_ID` for the outputs of the prep settings alone (`inputs` of prep_*.py and of the PLUMBER2 towers (`bin_plumber.py`), `clim` diagnostics of the prep scripts and of the PLUMBER2 towers, `coverage` of `diagnose_obs.py`, see [obs-availability.md](obs-availability.md), `radiation` of `compare_rn.py`, `mask` of `make_mask.py`), and the `RUN_ID` for everything from `make_edges.py` on: `mask` (common area mask), `edges` (LAI/AI edges and their figures, family `all`), `qbin` (binned statistics), `regress`, `decomp` (`decompose_state_function.py`, family `all`). File names carry no grid, period or bin counts: the settings of an ID are in `config/runs.csv` (and the netCDF attrs keep `time_period`, `grid` and the ID). The helpers are `cfg.inputs_path(family[, prep_id])`, `cfg.area_mask_path([run_id])` and `edges.edges_path(kind, variable[, run_id])`.

## binning
- Annual-mean (gridcell, year) ET is binned by that gridcell's climatological LAI and aridity.
- Inputs *must* share coordinates exactly.
- Duplicate quantile edges are kept: tied values are put in the *last* duplicate bin, leaving zero-width empty bins that are then removed. Out-of-range values are clipped into the edge bins.
- By default every (year, gridcell) sample counts once (no area weighting), so a gridcell's weight in its bin scales with its number of valid ET years.
- With `AREA_WEIGHTED`, each sample is weighted by its gridcell's land area (`binning.bin_stats(weights=)`): the bin `mean`, `var_pop` and `var_samp` are weighted, `sum_w` is the bin's summed land area and `n_eff = (Σw)²/Σw²` its Kish effective sample size, which sets `var_samp` = `var_pop`·n_eff/(n_eff−1) and the t-test of `binning.test_significance`. `count` stays the raw number of samples. Unweighted, `sum_w` = `n_eff` = `count`. Members are pooled (`binning.pool_members`) by `sum_w`, and η² (`binning.between_bin_fraction`) weights bins by `sum_w`.

## domain
- `prep.time_slice` (`cfg.TIME_SLICE`, default 2001-01 to 2014-12) for every family (`config/pipeline.yml`)
- Land: `LANDFRAC > LF_THRESH = 0.5` (models, ERA5) or Natural Earth land (obs products without a land fraction)
- Exclude Antarctica, Greenland, and Iceland
	- Trim `LAT_BNDS = slice(-58, 90)`
	- Exclude AR6 region 0 (`grid.greenland_mask`)

## pipeline
`prep_<family>.py`, once per product (obs, ERA5) or model (CMIP6) or ensemble (LENS2, FHIST PPE, IHIST PPE):
1. Load monthly native-resolution variables (`load/*.load_native`, `lo.load_obs`) [native grid]
2. The time period is selected by the loader (`TIME_SLICE`) [native grid]
3. Convert units of ET, P, Rn to W/m2 (`units.*`, written out in the script) [native grid]
	- $L_v = 2.45 \times 10^6$ J/kg for every water flux
	- Net radiation with the sign convention of each source (`units.net_radiation_*`)
	- IHIST PPE (offline CLM, no atmosphere output): P = RAIN + SNOW and Rn = FSA - FIRA, the land's net radiation (`units.net_radiation_clm`); the FHIST PPE uses the CAM gridcell mean FSNS - FLNS
4. Compute the annual mean (`temporal.aggregate(da, "year")`) [native grid]
	- Only keep complete calendar years: all 12 months in the time axis, and 12 valid months for ET, P, Rn
	- Missing LAI months count as zero
	- The number of gridcell-years affected is printed
5. Mask to land and remove AR6 region 0 (`grid.mask_to_land`) [native grid]
	- Use Natural Earth for observational products (ILAMB, PML, GLEAM, SiTH): `grid.landne_mask`
	- Use land fraction > `LF_THRESH` for model output and ERA5, since they have a land fraction variable: `grid.landfrac_mask`
	- ERA5 family: ERA5 ET, P and Rn use the ERA5 land fraction, and its obs LAI (GIMMS_LAI4g) uses Natural Earth
	- CMIP6 only: annual LAI missing on land is set to 0 (some models leave LAI undefined where there is no vegetation)
6. ET outlier filter: annual ET < `MIN_ANNUAL_ET = -1 W/m2` is NaN (`binning.drop_et_outliers`) [native grid]
	- On land, before regridding, so an outlier does not enter its target cell
	- The log reports how many fall inside `LAT_BNDS`
7. **Conservatively regrid to target resolution with `NA_THRESH = 0.5`** (`grid.bounded_conservative_regridder`, `grid.regrid_with_na_thresh`, `grid.on_grid`) [target grid]
	- A target cell is the mean of its valid (land) source cells, and NaN where more than `NA_THRESH` of its area is missing
8. Exclude Antarctica by reducing domain to `LAT_BNDS` [target grid]

`prep_<family>.py`, once per obs combination, ERA5 source or model (all members of a model or ensemble share their years):

9. Select only shared complete years from all four products, at least `MIN_YEARS = 3` (`temporal.shared_years`) [target grid]
10. Compute the climatological LAI, Rn, and P over those years (`temporal.aggregate(da, "clim")`) [target grid]
	- Report any missing years, as these will propagate to the aridity calculation
11. Compute aridity from the climatological Rn and P (`edges.compute_aridity_index`) [target grid]
11b. `BIN_VARIABLE = "et_pr"` only: the annual evaporation ratio ET/P (`binning.evaporation_ratio`) [target grid]
	- Annual ET over the annual P of the same year, product and source as the P of the AI (for obs, each combination's own P product), over the shared years of step 9
	- NaN where P <= 0; no other filter (ET has already been filtered in step 6)
	- Saved as `et_pr (member, year, lat, lon)` next to `et`, which is always saved

`make_edges.py`:

12. Apply the common area mask to ensure each data product uses the same gridcells [target grid]
	- Each family's own valid area (`grid.valid_area`: ET in any year, LAI and a finite AI, in every member), then their intersection (`grid.common_area`), saved to disk
	- Report how many gridcells and how much land area is lost compared with each family's own mask
13. Compute climatological LAI and aridity quantiles pooled across all data products [target grid]
	- Saved to disk so that the binning script loads the same quantile edges for every family
	- `pooled_all`: hierarchical weights (`edges.hierarchical_weights`) reduce the contribution of members of the same model, of variants of one model or obs product, and of FPPE, IPPE and LENS2:
	   w = 1/N_families × 1/N_groups(family) × 1/N_sources(group) × 1/N_members(source) × 1/N_values(member).
	   A source is an obs combination, a CMIP6 model, or a PPE/LENS member. A group of variants (`variants.source_groups`) is the CMIP6 models of one base model (`variants.group_model_variants`, e.g. CESM2, CESM2-WACCM, ...), or the obs combinations whose ET products are versions of one product (`variants.OBS_ET_PRODUCT_GROUPS`: GLEAMv4.3a/b, PMLv2.2a_MODIS/c; GLEAMv3.3a is its own product) with the same LAI, P and Rn; in other families each source is its own group.
	- `pooled_<family>`: one family, its groups of variants weighted equally
	- Weighted quantiles with `np.quantile(..., weights=, method="inverted_cdf")`: edges fall on data values, the first and last edges are the pooled minimum and maximum

`bin_all.py`:

14. Bin ET (or ET/P, `BIN_VARIABLE`) by climatological LAI and aridity, for every family in `BIN_FAMILIES`, with the `pooled_all` and `pooled_<family>` edges (`binning.bin_stats`, per member; with `AREA_WEIGHTED`, each sample weighted by its gridcell's land area) [target grid]

**Means and spreads across a family** (the clim maps of the prep scripts, the bin summary figures of `bin_all*.py`, the obs spread across ET products, the pooled families and variance partition of `decompose_state_function.py`, and the notebooks) count each group of variants once, as the edges do: `variants.group_means` averages the members of each source, then the sources of each group (NaN skipped at each level), and the mean and std are taken across the groups. The per-source statistics (`sources_pooled`) and the ranges (max − min) are across every source and member.

`bin_all_clim.py`: step 14 with climatological ET. Each member's annual mean ET is averaged over `year` with the step 10 rule (`temporal.aggregate(et, "clim")`: every year weighs equally, a gridcell's NaN years are skipped), then binned as in `bin_all.py`. Each gridcell is one sample, so every gridcell weighs equally in its bin whatever its number of valid ET years, and the within-bin variance is spatial only. Outputs are named `<family>.et_clim.qbin_*`. With `BIN_VARIABLE = "et_pr"`, the annual ET/P is averaged the same way (the mean of the annual ratios), and outputs are named `<family>.et_pr_clim.qbin_*`.

`bin_area.py`: step 14 for land area: the total land area in each bin. Each gridcell of the common area is one sample, as in `bin_all_clim.py`, binned by its climatological LAI and AI of the member, with its land area as the value (`rg.land_area`: the target-grid cell area times its fractional Natural Earth land cover, regionmask `mask_3D_frac_approx`). It is one field for every family, since the obs products have no land fraction of their own and native area variables no longer apply after regridding, so the binned area differs between families and members only through their LAI and AI. A bin's total is the `mean` x `count` of `binning.bin_stats`. Each member's total is the land area of the common area mask. Sources average their members' areas. Outputs: `PROC_ROOT/<run_id>/<family>/qbin/<family>.land_area.qbin_*` (`land_area` [km2], `land_area_frac`, `n_cells`).

`bin_plumber.py`: step 14 for the PLUMBER2 flux towers (`cfg.PLUMBER_ROOT`). Only each tower's longest run of consecutive complete years of Qle in `TIME_SLICE` (after the ET outlier filter) is kept, and towers whose run is shorter than `MIN_CONSECUTIVE_YEARS` are left out. Each year of the run is one annual mean ET sample, binned by the climatological obs LAI and AI (from the obs inputs) at the gridcell nearest to the tower, once per (LAI, P, Rn) product set, with the `pooled_all` and `pooled_obs` edges. Towers are not masked by the common area mask; towers whose nearest gridcell has no obs data are not binned. It also plots each tower's record-mean ET against the climatological LAI and AI of `SCATTER_PRODUCTS` (GIMMS_LAI4g, GPCPv2.3, GEWEX.SRB Rn) over the years of the longest tower record, computed from the products with steps 1-11. With `BIN_VARIABLE = "et_pr"`, each tower-year's ET is divided by that year's P at the nearest gridcell, from the P product of each product set (the P of its AI); annual P is computed over `TIME_SLICE` with steps 1-8, since the obs inputs end with the obs climatology years. The tower scatter stays ET only. The tower inputs (with ET/P always, as step 11b) and the scatter climatology depend on the prep settings only, so they are saved under the `PREP_ID` (`cfg.inputs_path("plumber")`, shared by every run with that `PREP_ID` and read by `bin_plumber_clim.py`); the binned outputs are saved under the `RUN_ID`, with the number of towers in the common area mask in attr `n_towers_in_common_area`.

`bin_plumber_clim.py`: `bin_plumber.py` with climatological ET, as `bin_all_clim.py`. It reads the PLUMBER2 inputs of `bin_plumber.py`, so the towers and years follow the same consecutive-year requirement (`min_consecutive_years` in the inputs attrs). Each tower's ET is averaged over its run (`temporal.aggregate(et, "clim")`), so each tower is one sample and weighs equally in its bin. Outputs are named `plumber.et_clim.qbin_*`.

**TODO:** For testing the sensitivity to resolution, keep model output at native resolution by skipping step 7 (needs the common area mask on each native grid).

## analyses outside the pipeline
Run by hand after step 14; `scripts/run_all.sh` does not run them.

`regress_lai_aridity.py`: how much of the ET variance in the heatmaps climatological LAI and AI explain. Per member, on exactly the samples of `bin_all.py` (annual) and `bin_all_clim.py` (clim), on the common area and unweighted: OLS fits ET ~ LAI, ET ~ AI and ET ~ LAI + AI (`regression.lai_ai_regression`), the partition of R² into unique LAI, unique AI and shared parts (LAI and AI are correlated), and for comparison the fraction of variance explained by the bin means (η², `binning.between_bin_fraction`) with the `pooled_all` and `pooled_<family>` edges. Climatological predictors cannot explain interannual variability, so the annual R² is at most the between-gridcell share of the variance. Outputs: `PROC_ROOT/<run_id>/<family>/regress/<family>.et.regress_lai_ai.<run_id>.nc` and `FIG_ROOT/<run_id>/all/regress/all.et.regress_lai_ai.<run_id>.png`.

`decompose_state_function.py` (`pixi run decompose-state-function`): how much of a dataset's area-mean ET difference from the observations comes from where its land sits in LAI-AI space (state), how much from its ET at a given LAI and AI (function) and how much from the two together (interaction). It reads the land-area weighted climatological ET binned with the `pooled_all` edges (`bin_all_clim.py` in a run with `binning.area_weighted: true` and `bin_variable: et`, e.g. the run of `config/runs/aw_bins15.yml`; the script raises otherwise); its `RUN_ID` constant (or `ETUNC_RUN_ID`) is required and fixes the inputs and settings, which are those registered under it in `config/runs.csv` (`cfg.run_settings`), not the current config, and `scripts/run_decomp.sh` passes each run file's `RUN_ID` as `ETUNC_RUN_ID`. For each member, f(b) is the bin mean ET (`mean`) and p(b) = `sum_w`(b) / Σ_b `sum_w` the fraction of the common area's land in bin b (never `count`, which ignores the cell areas), so E = Σ_b p f is the area-mean ET over the common area mask. For a dataset m and an obs reference r, over the common support S (the bins where both have ET), four counterfactual domain means: E_rr = Σ_S p_r f_r (observed state and function), E_rm = Σ_S p_r f_m (m's ET response on the observed state), E_mr = Σ_S p_m f_r (the observed response on m's state) and E_mm = Σ_S p_m f_m. The terms are FUNCTION = E_rm − E_rr, STATE = E_mr − E_rr and INTERACTION = E_mm − E_rm − E_mr + E_rr (> 0: the state and function biases reinforce; < 0: they compensate), which sum to E_mm − E_rr. The full-domain means E_m and E_r and the occupancy outside S (OFF_SUPPORT_m/_r, never filled) are saved too; the off-support ET is E_m − E_mm and E_r − E_rr. Only these scalars are saved. FUNCTION uses the observed state and STATE the observed response, so both inherit the reference's uncertainty. The references are each obs source and `obs_pooled` (one weight per group of variants, each source of a group an equal share, `OBS_POOL_WEIGHTING = "variant"`; the effective weight of each product is reported); a family pooled (`<family>_pooled`, `FAMILY_POOL_WEIGHTING = "variant"`) averages the members of a source, then weighs each group of variants once (e.g. the CMIP6 models of one base model share a weight), so E_pool = mean of the groups' E. Comparisons: every member and pooled family against every obs reference (cfg.FAMILIES without obs, and `TRENDY_FAMILIES`, trendy_s2 skipped if absent); the obs sources against `obs_pooled` (the observational spread, and which terms of each dataset lie within it); the range of each term across the obs references, overall and by product axis (`OBS_PRODUCT_AXES`, an axis with one product skipped); the variance (ddof=0) of E_mm − E_rr across the groups of variants of each family (the sources of a group averaged first) against `obs_pooled`, split into Var(STATE), Var(FUNCTION), Var(INTERACTION) and twice their covariances (equal to Var(E_mm) when E_rr is the same for every group, i.e. a complete support; Var(E_mm) and Var(E_rr) are saved alongside), with the fppe/ippe components within the lens values flagged; the state shift against `obs_pooled` (`decomposition.state_shift`): the land in other bins than obs, LAND_SHIFT = Σ_b |p_m − p_r| / 2, next to STATE split into STATE_GAINS = Σ_S max(dp, 0) f_r and STATE_LOSSES = Σ_S min(dp, 0) f_r (since Σ dp = 0, a large shift gives a small STATE when land moves between bins of similar observed ET, or both up and down in it); and, once several bin counts have been run, the shift of the variance fractions with the bin count (> 10 points flagged). `scripts/run_decomp.sh` (`pixi run run-decomp`) runs the script for each run file of `RUNS` (default `config/runs/aw_bins{10,15,20}.yml`); comparison 6 reads every registered run (`config/runs.csv`) that differs from the current one only in the bin counts, so the last run's summary covers them all. The obs product combinations are read from the `<axis>_product` coords of the obs inputs, so adding LAI, P or Rn products to `prep_obs.py` needs no change to the script: rerun the pipeline, then it. Caveats: an accounting identity on a shared partition of a shared domain, not a causal attribution (LAI and AI respond to ET in coupled runs); FUNCTION includes every control not on the two axes; the obs state uncertainty is sampled only along the product axes that vary, and each obs source's LAI, P and Rn climatology is over its own years; TRENDY's P and Rn are prescribed (CRU-JRA, ERA5), so its AI is shared by every model and its state differs only in LAI. Outputs: `PROC_ROOT/<run_id>/all/decomp/all.et_clim.decomp_{vs_obs,obs_axis_range,variance}.qbin_pooled_all.<run_id>.nc` (and the `vs_obs` CSV), `PROC_ROOT/<run_id>/all/decomp/all.et_clim.decomp_variance.qbin_pooled_all.all_bins.<run_id>.nc` and `FIG_ROOT/<run_id>/all/decomp/` (and `decomp_state_shift` nc, CSV and one figure per family).

`gridded_state_function_decomp.py` (`pixi run gridded-state-function-decomp`): the same decomposition, mapped from the bins back to the lat/lon grid, to show where on land STATE, FUNCTION and INTERACTION come from. It reads the inputs of `decompose_state_function.py` (same `RUN_ID`, checks and families) and each family's binning inputs of the run's `PREP_ID`, from which each member's gridcell bins (`binning.bin_index`, the assignment rule of `bin_stats`) and climatological ET e. For a dataset m and an obs reference r, each gridcell i takes the bin means at its own bins in m and in r, b_m(i) and b_r(i) (`decomposition.decompose_gridded`): E_xy(i) = f_y(b_x(i)) on the common support (0 off it, with the weight off it in OFF_SUPPORT_m/_r), STATE(i) = f_r(b_m) − f_r(b_r) (the observed ET change from the gridcell's move in LAI-AI space), FUNCTION(i) = f_m(b_r) − f_r(b_r), INTERACTION(i) = E_mm − E_rm − E_mr + E_rr and WITHIN(i) = (e_m − E_mm) − (e_r − E_rr) (the difference of the within-bin anomalies), so DIFF = e_m − e_r = STATE + FUNCTION + INTERACTION + WITHIN at every gridcell. `obs_pooled` and `<family>_pooled` have no single LAI-AI map: their state is the mixture of their members' bin maps with the pool weights (and e their weighted mean ET), and since p_pool = Σ_k w_k p_k the land-area mean of every map is still the scalar of `decompose_state_function.py` (WITHIN: 0 on a complete support). The script checks that the gridcell bins reproduce each member's occupancy and that every map averages to its scalar, and raises otherwise. A gridcell's STATE is not a unique "share" of the area-mean STATE: the attribution follows the gridcells, which carry their LAI and AI from one bin to another. Outputs: `PROC_ROOT/<run_id>/all/decomp/all.et_clim.decomp_gridded.qbin_pooled_all.<family>.<run_id>.nc` (every member and the pooled family, or each obs source, against each obs source and `obs_pooled`: (dataset, obs_reference, lat, lon), float32) and, against `obs_pooled`, `FIG_ROOT/<run_id>/all/decomp/all.et_clim.decomp_gridded.qbin_pooled_all.<family>.{pooled,group_std,members_<term>}.<run_id>.png` (the pooled family's DIFF, STATE, FUNCTION, INTERACTION and WITHIN; their std across the groups of variants; every member's map of each term).


## TRENDY
`prep_trendy.py` prepares the TRENDY (GCB2025) DGVMs of each experiment of `prep.trendy.experiments` ("S2" climate + CO2, "S3" climate + CO2 + land use change; default [S3]) as the family `trendy_s2` or `trendy_s3`, one member (= source) per model. It differs from the other prep scripts in:
- Step 1 (`load/trendy.py`): the files are not consistent with each other, so every time axis is rebuilt as a monthly series ending in Dec 2024 (the end of every GCB2025 run), checked against the decoded time where it can be decoded; lat/lon names, orders and lon ranges are normalized.
- P is the CRU-JRA v3 precipitation forcing of every model (6-hourly totals summed to months), not the models' own pr (missing in some models, 12x too small in ED). It runs steps 1-8 once on its own grid and is shared by every model.
- Step 3: ET is `evapotrans` (kg m-2 s-1).
- Rn is the ERA5 net surface radiation (`msnswrf + msnlwrf`, native land mask ERA5 LANDFRAC > `LF_THRESH`, as in `prep_era5.py`), since the models output at most the downward shortwave (`rsds`), not the upward terms of the net radiation. Like P, it runs steps 1-8 once and is shared by every model and experiment, so the AI is the same for every model. A `trendy_*` family in `FAMILIES` must have its experiment in `prep.trendy.experiments` (checked by `etunc.config`).
  `compare_rn.py` (`qsub scripts/compare_rn.pbs`) compares the surface radiation of ERA5, CERES and the CRU-JRA forcing (downward shortwave, downward longwave, Rn) on which this choice rests; its required `PREP_ID` fixes the inputs (the prep settings registered under it in `config/runs.csv`, `cfg.prep_settings`); outputs in `PROC_ROOT/<prep_id>/trendy_<experiment>/radiation/` and `FIG_ROOT/<prep_id>/trendy_<experiment>/radiation/`.
- Step 5: Natural Earth land (`grid.landne_mask`), since few models provide a land fraction and some fill the ocean with 0; annual LAI missing on land is set to 0, as for CMIP6.
- Models: every model with `evapotrans` and `lai` (ISAM has no lai), minus `prep.trendy.omit_models`: CARDAMOM (2003-2024 only) and LPJml, LPJwsl, LPX-Bern and VISIT (implausible LAI, 90th percentile 8-11 m2/m2), and ED (ET = 0 in high northern latitudes). CLASSIC S2 is read from its `correctedS2/` upload.

## code
Core modules and their respective functions.
```
load/*.py:
load_native(...)                         # step 1: raw monthly fields in native units
obs.load_obs(dataset, var, time_slice)   # step 1 for the gridded obs products
trendy.load_forcing(variable, time_slice)  # step 1 for the TRENDY forcing ("pre": P in mm/month, "dlwrf", "fd")
load_land_fraction()                     # land fraction and native grid (era5, cmip, cesm)

units.py:
convert_units()                          # keyed on the variable name (CMIP, CESM, ERA5)
flux_to_wm2() / latent_heat_to_wm2()     # keyed on the units attr (ILAMB)
accumulation_to_flux()                   # totals per month/year (gridded obs)
net_radiation_*()

temporal.py:
complete_years(da)                       # years with all 12 months in the time axis
yearly_to_annual()                       # yearly totals are annual means
aggregate(da, how, require_all_months)   # how = "year" (step 4) or "clim" (step 10); prints
                                         # missing months/years and saves them in attrs
shared_years(*anns)                      # sorted years shared across anns

grid.py:
greenland_mask(grid)                     # False in AR6 region 0
landfrac_mask(landfrac, lf_thresh)       # landfrac > lf_thresh, without Greenland/Iceland
landne_mask(grid)                        # Natural Earth land, without Greenland/Iceland
mask_to_land(da, land, label)            # step 5, after checking that both are on one grid
target_grid(res)                         # global grid at the specified spacing
bounded_src_grid(ds)                     # grid with cell edges for xESMF
bounded_conservative_regridder(ds, res)  # xESMF regridder for any monotonic 1D grid
regrid_with_na_thresh(da, regridder)     # apply a conservative regridder skipping NaN
on_grid(da, res, label)                  # exact target coords
valid_area(inputs)                       # gridcells where inputs are valid in every member
common_area(valid_areas)                 # intersection of all valid areas, with the loss report

variants.py:
source_groups()                          # group of variants of each member/source (cmip: base model; obs: ET product versions)
group_means()                            # nested means: members of a source, then sources of a group

edges.py:
compute_aridity_index()
hierarchical_weights()                   # family, group of variants, source, member
pooled_bin_edges()
edges_path() / load_bin_edges()          # PROC_ROOT/<run_id>/all/edges/

binning.py:
drop_et_outliers()                       # step 6
bin_stats()                              # step 14
pool_members()
pool_by_source()                         # members of each source pooled
drop_zero_width_bins()
frac_valid()
test_significance()
between_bin_fraction()                   # eta2: variance explained by the bin means
et_product_spread()                      # obs: std of the bin mean across ET products (versions averaged first)
ensemble_spread()                        # std across members/sources, or across groups of variants with `family`

regression.py:
ols()                                    # OLS with an intercept: coefficients and R2
lai_ai_regression()                      # per member, ET ~ LAI, ~ AI, ~ LAI + AI, and the R2 partition

decomposition.py:
occupancy_and_mean()                     # p = sum_w / total, f = mean of a land-area weighted bin_stats output
pool(), members_to_sources()             # datasets pooled with given weights (E_pool = weighted mean of E)
source_weights(), product_weights()      # pooled-dataset weights (per source or per group of variants), and the weight of each product
counterfactual_mean()                    # sum over the common support of p_x f_y
decompose()                              # E_rr, E_rm, E_mr, E_mm, FUNCTION/STATE/INTERACTION and off-support mass
state_shift()                            # land in other bins than obs; STATE from the bins gained and lost
product_axis_range()                     # range of a quantity across obs references, by product axis
variance_partition()                     # Var(E_mm - E_rr) across datasets split into the terms
```
Code that the pipeline no longer uses was removed (`etunc/legacy/` and `scripts/legacy_regrid/`, commit 8b26f0b); it is in the git history.
