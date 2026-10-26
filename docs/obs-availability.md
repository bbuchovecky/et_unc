# Observational availability

Every observational product of each variable runs through the same spatial and temporal availability diagnostic, `scripts/diagnose_obs.py` (`pixi run diagnose-obs`). The variables are net radiation (`rn`), precipitation (`pr`), evapotranspiration (`et`) and leaf area index (`lai`). The script creates the following plots for each variable, with one panel or line per product:
- Gridcells and land area covered, on the native grids and on the pipeline's target grid
- Maps of coverage on the native grids (valid months, complete years, seasonal gaps)
- Timeseries of the valid fraction of the land area in each latitude row
- Timeseries of global and regional area-weighted valid variable
- Climatological maps and factorial differences between products
- Histograms of valid data with vertical lines for specified quantiles

These diagnostics were first made in `notebooks/check-et-obs.ipynb` (ET) and `notebooks/check-clara-a3.ipynb` (Rn).

## Products

- ILAMB: every product of `il.PRODUCTS` for each variable (`ILAMB_PRODUCTS`)
- Gridded products from their native monthly files (`GRIDDED_PRODUCTS`): PML-V2.2 (a-MODIS, c), GLEAM v4.3 (a, b) and SiTHv2 ET, and CLARA-A3 Rn (`SRB`)

A product without files, or without a complete calendar year in `TIME_SLICE`, is skipped.

## Definitions

- **Domain and land**: as in `prep_obs.py` (step 5), Natural Earth land by cell center without Greenland and Iceland (`rg.landne_mask`), within `LAT_BNDS`. On the native grids each cell weighs its area (`rg.cell_area`). On the target grid each cell weighs its fractional Natural Earth land area (`rg.land_area`), so coastal cells count only their land part.
- **Units**: ET, P and Rn in W/m2 as in step 3 of `prep_obs.py`, LAI in m2/m2.
- **Valid month**: the product has a value at a land cell in that month.
- **Valid year**: the rule of step 4. ET, P and Rn need all 12 months valid. LAI needs at least one, since a missing LAI month counts as 0 (`REQUIRE_ALL_MONTHS`). Only the complete calendar years of the product's record inside `TIME_SLICE` count.
- **Seasonal gap**: a calendar month that is valid at a gridcell in fewer than `SEASONAL_GAP_FRAC` (0.5) of the years.
- **Target grid**: the annual means over `TIME_SLICE` are computed as in steps 4-8 of `prep_obs.py` (annual means, land mask, ET outlier filter for ET, conservative regridding with `NA_THRESH`, `LAT_BNDS`). The climatology is computed as in step 10 (mean of the valid years). A target gridcell is covered if it has a climatology.
- **Timeseries**: each product's full record (`RECORD_SLICE`), with `TIME_SLICE` shaded. A regional mean of a month covers only that month's valid land, so seasonal gaps (e.g. winter at high latitudes) bias it.
- **Differences**: for every pair of products, the mean over their shared valid years of the difference of their annual means on the target grid, so both cover the same years in each cell.
- **Histograms**: land-area weighted histograms of the valid monthly values on the native grids over `TIME_SLICE`, on the fixed bins of `HIST_BINS`. Values outside the bins are counted in the end bins. The quantiles (`QUANTILES`) are interpolated linearly within the bins, so they are accurate to one bin width.

Each product is read once and reduced as it is read (dask), so a full 0.1° record (about 10 GB) is never held in memory. A full run takes about 15 min on a Casper node (8 cores, 128 GB), most of it reading the 0.1° ET products.

## Outputs

`<period>` is the `YYYYMM-YYYYMM` of `TIME_SLICE`, `<grid>` the target grid (e.g. `1deg`), `<var>` one of `et`, `lai`, `pr`, `rn`.

`PROC_ROOT/obs/coverage/<period>/obs.<var>.coverage.<period>.csv`: one row per product, with its record, native resolution, valid years, land area [10⁶ km²] covered on the native grid (land, any valid month, at least one valid year, every year valid) and on the target grid (globally and in each of `REGIONS`, in 10⁶ km² and as a percentage of the region's land), and the quantiles of the valid monthly values.

`FIG_ROOT/obs/coverage/<period>/obs.<var>.<diag>.<period>.png`:

| Plot | `<diag>` | Content |
|---|---|---|
| Land area covered | `land_area` | land area with any valid month and with a valid year (native), and with a climatology (target) |
| | `map_n_products.<grid>` | number of products with a climatology at each target gridcell |
| Native coverage maps | `map_valid_months` | fraction of the months in `TIME_SLICE` that are valid |
| | `map_valid_years` | fraction of the years that are valid |
| | `map_seasonal_gaps` | number of calendar months that are seasonal gaps |
| Valid land by latitude | `zonal_time` | valid fraction of the land area of each latitude row, by month over the record |
| | `zonal_calendar_month` | the same by calendar month, averaged over the years in `TIME_SLICE` |
| Regional timeseries | `ts_valid_land` | valid fraction of each region's land area |
| | `ts_mean` | area-weighted mean over the valid land, monthly and annual (12 valid months) |
| Climatologies | `map_clim.<grid>` | climatology of each product, with its global land-area weighted mean |
| | `map_diff.<grid>` | differences of every pair of products (row minus column) |
| Histograms | `hist` | density of the valid monthly values, with vertical lines at the quantiles |

In the line plots each product keeps one color (fixed order, at most 8 per panel). ET has more products than that, so its ILAMB and gridded products are in separate columns.
