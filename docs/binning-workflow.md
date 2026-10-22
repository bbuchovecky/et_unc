## Binning Workflow
Every data family (obs, ERA5, CMIP6, LENS2, FHIST PPE, IHIST PPE) runs the same numbered steps.
They are written out in `main()` of each script. The scripts differ only in
step 1 (which loader), step 3 (which unit conversion) and step 5 (which land mask).

```
prep_{obs,era5,cmip,lens,fppe,ippe}.py                 steps 1-11   -> PROC_ROOT/<family>/<family>.inputs.<grid>.<period>.nc
make_edges.py                                           steps 12-13  -> common area mask, pooled LAI/AI edges
bin_all.py                                              step 14      -> binned ET of every family
bin_all_clim.py                                         step 14      -> binned climatological ET of every family
bin_area.py                                             step 14      -> land area in each bin, for every family
bin_plumber.py                                          step 14      -> binned ET of the PLUMBER2 flux towers
bin_plumber_clim.py                                     step 14      -> binned climatological ET of the PLUMBER2 flux towers
regress_lai_aridity.py                                  after 14     -> ET variance explained by LAI and AI (OLS)
```
Run them in that order. `scripts/run_all.sh` (`pixi run run-all`, or `qsub scripts/run_all.pbs` as one Casper job) runs all of them except `regress_lai_aridity.py`.
`BIN_VARIABLE` (default `cfg.BIN_VARIABLE = "et"`, set at the top of each prep and bin script) chooses what step 14 bins: `"et"`, annual mean ET, or `"et_pr"`, the annual evaporation ratio ET/P (step 11b). Outputs are named `<family>.<BIN_VARIABLE>.qbin_*` and `<family>.<BIN_VARIABLE>_clim.qbin_*`. The common area mask and the edges (steps 12-13) do not depend on it, so ET and ET/P are binned in the same bins.
`AREA_WEIGHTED` (default `cfg.AREA_WEIGHTED = False`) makes `bin_all.py` and `bin_all_clim.py` weight each sample by its gridcell's land area (`rg.land_area`, one field for every family) in the bin statistics. Outputs then carry `.area_weighted` after the variable (e.g. `<family>.et.area_weighted.qbin_*`, `<family>.et_clim.area_weighted.qbin_*`). The common area mask and the edges do not depend on it, so weighted and unweighted runs use the same bins.
The bin counts (`N_YBINS` for LAI, `N_XBINS` for AI in `etunc/config.py`) are in the file names, so runs with different counts do not overwrite each other: edges are named `<kind>.<lai|ai>_clim.<n>_quantiles.*`, and the outputs of step 14 and `regress_lai_aridity.py` carry `lai<N_YBINS>_ai<N_XBINS>` (e.g. `<family>.et.qbin_pooled_all.lai15_ai15.1deg.199501-201412.nc`). The common area mask does not depend on them.
Every script prints `etunc/config.py` and its own settings into its log (`scripts/run_logged.sh`).

## binning
- Annual-mean (gridcell, year) ET is binned by that gridcell's climatological LAI and aridity.
- Inputs *must* share coordinates exactly.
- Duplicate quantile edges are kept: tied values are put in the *last* duplicate bin, leaving zero-width empty bins that are then removed. Out-of-range values are clipped into the edge bins.
- By default every (year, gridcell) sample counts once (no area weighting), so a gridcell's weight in its bin scales with its number of valid ET years.
- With `AREA_WEIGHTED`, each sample is weighted by its gridcell's land area (`binning.bin_stats(weights=)`): the bin `mean`, `var_pop` and `var_samp` are weighted, `sum_w` is the bin's summed land area and `n_eff = (Σw)²/Σw²` its Kish effective sample size, which sets `var_samp` = `var_pop`·n_eff/(n_eff−1) and the t-test of `binning.test_significance`. `count` stays the raw number of samples. Unweighted, `sum_w` = `n_eff` = `count`. Members are pooled (`binning.pool_members`) by `sum_w`, and η² (`binning.between_bin_fraction`) weights bins by `sum_w`.

## domain
- `TIME_SLICE = slice("1995-01", "2014-12")` for every family (`etunc/config.py`)
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
	- `pooled_all`: hierarchical weights (`edges.hierarchical_weights`) reduce the contribution of members of the same model and of FPPE, IPPE and LENS2:
	   w = 1/N_families × 1/N_sources(family) × 1/N_members(source) × 1/N_values(member).
	   A source is an obs combination, a CMIP6 model, or a PPE/LENS member.
	- `pooled_<family>`: one family, its sources weighted equally
	- Weighted quantiles with `np.quantile(..., weights=, method="inverted_cdf")`: edges fall on data values, the first and last edges are the pooled minimum and maximum

`bin_all.py`:

14. Bin ET (or ET/P, `BIN_VARIABLE`) by climatological LAI and aridity, for every family in `BIN_FAMILIES`, with the `pooled_all` and `pooled_<family>` edges (`binning.bin_stats`, per member; with `AREA_WEIGHTED`, each sample weighted by its gridcell's land area) [target grid]

`bin_all_clim.py`: step 14 with climatological ET. Each member's annual mean ET is averaged over `year` with the step 10 rule (`temporal.aggregate(et, "clim")`: every year weighs equally, a gridcell's NaN years are skipped), then binned as in `bin_all.py`. Each gridcell is one sample, so every gridcell weighs equally in its bin whatever its number of valid ET years, and the within-bin variance is spatial only. Outputs are named `<family>.et_clim.qbin_*`. With `BIN_VARIABLE = "et_pr"`, the annual ET/P is averaged the same way (the mean of the annual ratios), and outputs are named `<family>.et_pr_clim.qbin_*`.

`bin_area.py`: step 14 for land area: the total land area in each bin. Each gridcell of the common area is one sample, as in `bin_all_clim.py`, binned by its climatological LAI and AI of the member, with its land area as the value (`rg.land_area`: the target-grid cell area times its fractional Natural Earth land cover, regionmask `mask_3D_frac_approx`). It is one field for every family, since the obs products have no land fraction of their own and native area variables no longer apply after regridding, so the binned area differs between families and members only through their LAI and AI. A bin's total is the `mean` x `count` of `binning.bin_stats`. Each member's total is the land area of the common area mask. Sources average their members' areas. Outputs: `PROC_ROOT/<family>/qbin/<family>.land_area.qbin_*` (`land_area` [km2], `land_area_frac`, `n_cells`).

`bin_plumber.py`: step 14 for the PLUMBER2 flux towers (`cfg.PLUMBER_ROOT`). Only each tower's longest run of consecutive complete years of Qle in `TIME_SLICE` (after the ET outlier filter) is kept, and towers whose run is shorter than `MIN_CONSECUTIVE_YEARS` are left out. Each year of the run is one annual mean ET sample, binned by the climatological obs LAI and AI (from the obs inputs) at the gridcell nearest to the tower, once per (LAI, P, Rn) product set, with the `pooled_all` and `pooled_obs` edges. Towers are not masked by the common area mask; towers whose nearest gridcell has no obs data are not binned. It also plots each tower's record-mean ET against the climatological LAI and AI of `SCATTER_PRODUCTS` (GIMMS_LAI4g, GPCPv2.3, GEWEX.SRB Rn) over the years of the longest tower record, computed from the products with steps 1-11. With `BIN_VARIABLE = "et_pr"`, each tower-year's ET is divided by that year's P at the nearest gridcell, from the P product of each product set (the P of its AI); annual P is computed over `TIME_SLICE` with steps 1-8, since the obs inputs end with the obs climatology years. The tower scatter stays ET only.

`bin_plumber_clim.py`: `bin_plumber.py` with climatological ET, as `bin_all_clim.py`. It reads the PLUMBER2 inputs of `bin_plumber.py`, so the towers and years follow the same consecutive-year requirement (`min_consecutive_years` in the inputs attrs). Each tower's ET is averaged over its run (`temporal.aggregate(et, "clim")`), so each tower is one sample and weighs equally in its bin. Outputs are named `plumber.et_clim.qbin_*`.

`regress_lai_aridity.py`: how much of the ET variance in the heatmaps climatological LAI and AI explain. Per member, on exactly the samples of `bin_all.py` (annual) and `bin_all_clim.py` (clim), on the common area and unweighted: OLS fits ET ~ LAI, ET ~ AI and ET ~ LAI + AI (`regression.lai_ai_regression`), the partition of R² into unique LAI, unique AI and shared parts (LAI and AI are correlated), and for comparison the fraction of variance explained by the bin means (η², `binning.between_bin_fraction`) with the `pooled_all` and `pooled_<family>` edges. Climatological predictors cannot explain interannual variability, so the annual R² is at most the between-gridcell share of the variance. Outputs: `PROC_ROOT/<family>/regress/<family>.et.regress_lai_ai.*.nc` and `FIG_ROOT/all/all.et.regress_lai_ai.*.png`.

**TODO:** For testing the sensitivity to resolution, keep model output at native resolution by skipping step 7 (needs the common area mask on each native grid).

## code
Core modules and their respective functions.
```
load/*.py:
load_native(...)                         # step 1: raw monthly fields in native units
obs.load_obs(dataset, var, time_slice)   # step 1 for the gridded obs products
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

edges.py:
compute_aridity_index()
hierarchical_weights()
pooled_bin_edges()
edges_path() / load_bin_edges()

binning.py:
drop_et_outliers()                       # step 6
bin_stats()                              # step 14
pool_members()
pool_by_source()                         # members of each source pooled
drop_zero_width_bins()
frac_valid()
test_significance()
between_bin_fraction()                   # eta2: variance explained by the bin means
et_product_spread()                      # obs: std of the bin mean across ET products

regression.py:
ols()                                    # OLS with an intercept: coefficients and R2
lai_ai_regression()                      # per member, ET ~ LAI, ~ AI, ~ LAI + AI, and the R2 partition
```
Code that the pipeline no longer uses is in `etunc/legacy/` (unmaintained, untested).
