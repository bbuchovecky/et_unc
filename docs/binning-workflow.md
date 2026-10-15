## Binning Workflow
## binning
- Annual-mean (gridcell, year) ET is binned by that gridcell's climatological LAI and aridity.
- Inputs *must* share coordinates exactly.
- Duplicate quantile edges are kept: tied values are put in *last* duplicate bin, leaving zero-width empty bins that are then removed. Out-of-range values are clipped into the edge bins.
## domain
- `TIME_SLICE = slice("1995-01", "2014-12")`
- `LANDFRAC > 0.5`
- Exclude Antarctica, Greenland, and Iceland
	- Trim `LAT_BNDS = slice(-58, 90)`
	- Exclude AR6 region 0
## pipeline
1. Load monthly native resolution variables (native grid)
2. Select one fixed `TIME_SLICE = slice("1995-01", "2014-12")` (native grid)
3. Convert units of ET, P, Rn to W/m2 (native grid)
4. Compute the annual mean (native grid)
	1. Only keep complete calendar years with 12 valid months for ET, P, Rn
	2. Missing LAI months count as zero
5. Mask to land and remove AR6 region 0 (native grid)
	1. Use Natural Earth for observational products (ILAMB, PML, GLEAM, SiTH)
	2. Use land fraction > `LF_THRESH` for model output, since they have a land fraction variable
6. Compute the climatological LAI, Rn, and P (native grid)
	1. Report any missing years, as these will propagate to the aridity calculation
7. **Conservatively regrid to target resolution with `NA_THRESH = 0.5` (target grid)**
8. Select only shared complete years from all four products (target grid)
9. Compute aridity from the climatological Rn and P (target grid)
10. Exclude Antarctica by reducing domain to `LAT_BNDS` (target grid)
11. Apply the common area mask to ensure each data product uses the same gridcells (target grid)
	1. Do this earlier when pooling for shared quantile edges and save to disk
	2. Report how many gridcells and how much land area is lost compared with each family's own mask
12. Compute climatological LAI and aridity quantiles pooled across all data products (target grid)
	1. Do this earlier and save to disk so that each binning pipeline script loads the same quantile edges
	2. TODO: Reduce contribution from same CMIP6 model members, FPPE, and LENS2. Compare averaging across model families
13. Bin ET by climatological LAI and aridity (target grid)

^ For testing the sensitivity to resolution, keep model output at native resolution by skipping step 7. 

## code
Non-exhaustive list of core modules and their respective functions.
```
load/*.py:
load_native(var, freq, time_slice, *)    # core loader for each data product
load_land_fraction()                     # loads the land fraction (seperate variable or grid file)

units.py: --> same as before
convert_units()
accumulation_to_flux()
net_radiation_*()

temporal.py:
complete_years(da)                       # return years with all 12 months in time axis
yearly_to_annual()                       # convert yearly totals to annual
aggregate(da, how, require_all_months)   # temporal aggregation of monthly field,
                                         # print missing months/years and save in attrs
                                         # how = ["year", "clim"]
shared_years(*anns)                      # return sorted years shared across anns

grid.py:
greenland_mask(grid)                     # create mask excluding AR6 region 0
landfrac_mask(landfrac, lf_thresh)       # create land mask where landfrac > lf_thresh
landne_mask(grid)                        # create land mask from Natural Earth

target_grid(res)                         # create global grid at the specified spacing
bounded_src_grid(ds)                     # create grid with cell edges for xESMF
bounded_conservative_regridder(ds, res)  # create an xESMF regridder for any monotonic 1D grid
regrid_with_na_thresh(da, src_grid, res) # apply conservative regridder skipping NaN

valid_area(inputs, member_dim)           # gridcells where inputs are valid
common_area(*valid_areas)                # intersection of all valid areas

edges.py:
compute_aridity_index()
build_bin_edges()
pooled_bin_edges()
load_bin_edges()

binning.py:
bin_stats()
frac_count()
frac_valid()
test_significance()
```