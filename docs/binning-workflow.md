## Binning Workflow
Every data family (obs, CMIP6, LENS2, FHIST PPE) runs the same numbered steps.
They are written out in `main()` of each script. The scripts differ only in
step 1 (which loader), step 3 (which unit conversion) and step 5 (which land mask).

```
prep_obs.py, prep_cmip.py, prep_lens.py, prep_fppe.py   steps 1-11   -> PROC_ROOT/<family>/<family>.inputs.<grid>.<period>.nc
make_edges.py                                           steps 12-13  -> common area mask, pooled LAI/AI edges
bin_all.py                                              step 14      -> binned ET of every family
```
Run them in that order.
Every script prints `etunc/config.py` and its own settings into its log (`scripts/run_logged.sh`).

## binning
- Annual-mean (gridcell, year) ET is binned by that gridcell's climatological LAI and aridity.
- Inputs *must* share coordinates exactly.
- Duplicate quantile edges are kept: tied values are put in the *last* duplicate bin, leaving zero-width empty bins that are then removed. Out-of-range values are clipped into the edge bins.
- Every (year, gridcell) sample counts once (no area weighting), so a gridcell's weight in its bin scales with its number of valid ET years.

## domain
- `TIME_SLICE = slice("1995-01", "2014-12")` for every family (`etunc/config.py`)
- Land: `LANDFRAC > LF_THRESH = 0.5` (models, ERA5) or Natural Earth land (obs products without a land fraction)
- Exclude Antarctica, Greenland, and Iceland
	- Trim `LAT_BNDS = slice(-58, 90)`
	- Exclude AR6 region 0 (`grid.greenland_mask`)

## pipeline
`prep_<family>.py`, once per product (obs) or model (CMIP6) or ensemble (LENS2, FHIST PPE):
1. Load monthly native-resolution variables (`load/*.load_native`, `lo.load_obs`) [native grid]
2. The time period is selected by the loader (`TIME_SLICE`) [native grid]
3. Convert units of ET, P, Rn to W/m2 (`units.*`, written out in the script) [native grid]
	- $L_v = 2.45 \times 10^6$ J/kg for every water flux
	- Net radiation with the sign convention of each source (`units.net_radiation_*`)
4. Compute the annual mean (`temporal.aggregate(da, "year")`) [native grid]
	- Only keep complete calendar years: all 12 months in the time axis, and 12 valid months for ET, P, Rn
	- Missing LAI months count as zero
	- The number of gridcell-years affected is printed
5. Mask to land and remove AR6 region 0 (`grid.mask_to_land`) [native grid]
	- Use Natural Earth for observational products (ILAMB, PML, GLEAM, SiTH): `grid.landne_mask`
	- Use land fraction > `LF_THRESH` for model output and ERA5, since they have a land fraction variable: `grid.landfrac_mask`
	- CMIP6 only: annual LAI missing on land is set to 0 (some models leave LAI undefined where there is no vegetation)
6. ET outlier filter: annual ET < `MIN_ANNUAL_ET = -1 W/m2` is NaN (`binning.drop_et_outliers`) [native grid]
	- On land, before regridding, so an outlier does not enter its target cell
	- The log reports how many fall inside `LAT_BNDS`
7. **Conservatively regrid to target resolution with `NA_THRESH = 0.5`** (`grid.bounded_conservative_regridder`, `grid.regrid_with_na_thresh`, `grid.on_grid`) [target grid]
	- A target cell is the mean of its valid (land) source cells, and NaN where more than `NA_THRESH` of its area is missing
8. Exclude Antarctica by reducing domain to `LAT_BNDS` [target grid]

`prep_<family>.py`, once per obs combination or model (all members of a model or ensemble share their years):

9. Select only shared complete years from all four products, at least `MIN_YEARS = 3` (`temporal.shared_years`) [target grid]
10. Compute the climatological LAI, Rn, and P over those years (`temporal.aggregate(da, "clim")`) [target grid]
	- Report any missing years, as these will propagate to the aridity calculation
11. Compute aridity from the climatological Rn and P (`edges.compute_aridity_index`) [target grid]

`make_edges.py`:

12. Apply the common area mask to ensure each data product uses the same gridcells [target grid]
	- Each family's own valid area (`grid.valid_area`: ET in any year, LAI and a finite AI, in every member), then their intersection (`grid.common_area`), saved to disk
	- Report how many gridcells and how much land area is lost compared with each family's own mask
13. Compute climatological LAI and aridity quantiles pooled across all data products [target grid]
	- Saved to disk so that the binning script loads the same quantile edges for every family
	- `pooled_all`: hierarchical weights (`edges.hierarchical_weights`) reduce the contribution of members of the same model and of FPPE and LENS2:
	   w = 1/N_families × 1/N_sources(family) × 1/N_members(source) × 1/N_values(member).
	   A source is an obs combination, a CMIP6 model, or a PPE/LENS member.
	- `pooled_<family>`: one family, its sources weighted equally
	- Weighted quantiles with `np.quantile(..., weights=, method="inverted_cdf")`: edges fall on data values, the first and last edges are the pooled minimum and maximum

`bin_all.py`:

14. Bin ET by climatological LAI and aridity, for every family in `BIN_FAMILIES`, with the `pooled_all` and `pooled_<family>` edges (`binning.bin_stats`, per member) [target grid]

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
drop_zero_width_bins()
frac_valid()
test_significance()
```
Code that the pipeline no longer uses is in `etunc/legacy/` (unmaintained, untested).
