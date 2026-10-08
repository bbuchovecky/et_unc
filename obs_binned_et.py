"""
obs_binned_et.py
================
Bin evapotranspiration (ET) in a 2-D space of climatological leaf area index
(LAI, y-axis) and aridity index (AI = Rn / L*P, x-axis), as in
`ilamb_binned_et.py`, for every ET product: the ILAMB ET products plus
PML-V2.2, GLEAM v4.3 and SiTHv2. LAI, precipitation and net radiation come
from ILAMB (MODIS, GPCPv2.3, CERESed4.2).

Processing
----------
- ILAMB products are loaded with `ilamb_binned_et.load_product`.
- PML, GLEAM and SiTHv2 are read from the 0.5 deg files written by
  `regrid_obs.py` (conservative regridding) and converted from mm/month or
  mm/year to W/m2. Only the years shared by the LAI, pr and rns products are
  loaded, since combinations use shared years only.
  - Monthly products (`freq="monthly"`): complete years only, and a year is
    valid only with all 12 months, as for the ILAMB ET products.
  - Yearly products (`freq="yearly"`): the yearly total is the annual mean.
  SiTHv2 is used yearly for now; set its `freq` to "monthly" in
  GRIDDED_ET_PRODUCTS once the monthly files are regridded.
  - Gridcell-years with annual mean ET < MIN_ANNUAL_ET (-1 W/m2) are NaN
    (GLEAM v4.3b has extreme negative values at high latitudes).
- Maps, combinations, the common area mask, bin edges, binning and plots are
  `ilamb_binned_et.run`; see `ilamb_binned_et.py` for the steps and outputs.
  Outputs use the same paths and "obs." prefix as `ilamb_binned_et.py`.
"""

from __future__ import annotations

import itertools

import xarray as xr

import etunc.config as config
import etunc.units as units
import etunc.grid as rg
import ilamb_binned_et as ib
import etunc.load.obs as lo


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

# ILAMB products, by variable (keys of ib.PRODUCTS[variable])
ILAMB_PRODUCTS = {
    "et":  list(ib.PRODUCTS["et"]),  # every ILAMB ET product
    "lai": ["MODIS"],
    "pr":  ["GPCPv2.3"],
    "rns": ["CERESed4.2"],
}

# Gridded ET products: {label: (load_obs dataset, version, variable, freq)}
GRIDDED_ET_PRODUCTS = {
    "PMLv2.2a_MODIS": ("pml", "V2.2a-MODIS", "ET", "monthly"),
    # "PMLv2.2b":       ("pml", "V2.2b", "ET", "monthly"),
    "PMLv2.2c":       ("pml", "V2.2c", "ET", "monthly"),
    "GLEAMv4.3a":     ("gleam", "v4.3a", "E", "monthly"),
    "GLEAMv4.3b":     ("gleam", "v4.3b", "E", "monthly"),
    "SiTHv2":         ("sith", "v2", "ET", "yearly"),  # "monthly" once re-downloaded
}
GRIDDED_RES = "0.5"  # load_obs `res` of the regridded files; must match ib.TARGET_RES
# Gridcell-years of gridded products with annual mean ET below this [W/m2] are
# NaN; GLEAM v4.3b has extreme negative monthly E at high latitudes in winter
MIN_ANNUAL_ET = -1.0


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def yearly_to_annual(da: xr.DataArray) -> xr.DataArray:
    """Yearly field (one time step per year) with `time` replaced by `year`."""
    years = da.time.dt.year.values
    if len(set(years)) != len(years):
        raise ValueError(f"{da.name}: more than one time step in a year")
    return da.assign_coords(year=("time", years)).swap_dims(time="year").drop_vars("time")


def load_gridded(label: str, years: tuple[int, int]) -> xr.DataArray:
    """Annual mean ET [W/m2] (year, lat, lon) of a gridded product within `years` (first, last)."""
    dataset, version, var, freq = GRIDDED_ET_PRODUCTS[label]
    da = lo.load_obs(
        dataset, var, slice(str(years[0]), str(years[1])), version=version, freq=freq, res=GRIDDED_RES,
    ).load()
    da = units.latent_heat_to_wm2(lo.accumulation_to_flux(da))

    if freq == "monthly":
        da = da.sel(time=da.time.dt.year.isin(ib.complete_years(da)))
        ann = ib.annual_mean(da, require_all_months=True)
    elif freq == "yearly":
        ann = yearly_to_annual(da)
    else:
        raise ValueError(f"{label}: unsupported freq {freq!r}")

    outlier = ann < MIN_ANNUAL_ET
    if outlier.any():
        print(f"et/{label}: {int(outlier.sum())} gridcell-years with annual ET < {MIN_ANNUAL_ET} W/m2 set to NaN "
              f"(min {float(ann.min()):0.3g})")
        with xr.set_options(keep_attrs=True):
            ann = ann.where(~outlier)

    # Same grid checks and LAT_BNDS selection as ib.load_product
    rg.check_same_grid(ann, ib.TARGET_GRID, f"et/{label}")
    ann = ann.assign_coords(lat=ib.TARGET_GRID.lat, lon=ib.TARGET_GRID.lon)
    ann = ann.sel(lat=config.LAT_BNDS).rename("et")

    print(
        f"et  {label:12}: {ann.dims} {ann.shape} {ib.period_str(ann.year.values)} "
        f"[{float(ann.min()):0.3g}, {float(ann.max()):0.3g}] {ann.attrs['units']}"
    )
    return ann


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    if float(GRIDDED_RES) != ib.TARGET_RES:
        raise ValueError(f"GRIDDED_RES {GRIDDED_RES} does not match ib.TARGET_RES {ib.TARGET_RES}")
    mask = ib.land_mask(ib.TARGET_GRID).sel(lat=config.LAT_BNDS)

    print("=== Load ILAMB products ===")
    ann = {v: {p: ib.load_product(v, p) for p in products} for v, products in ILAMB_PRODUCTS.items()}

    # Gridded ET only over the years that some (lai, pr, rns) product set shares
    years = set().union(*(
        ib.shared_years(*das) for das in itertools.product(*(ann[v].values() for v in ib.FACTORS[1:]))
    ))
    if not years:
        raise RuntimeError("No (lai, pr, rns) product set shares any year.")
    span = (min(years), max(years))
    print(f"\n=== Load gridded ET products ({span[0]}-{span[1]}) ===")
    for label in GRIDDED_ET_PRODUCTS:
        ann["et"][label] = load_gridded(label, span)

    ib.run(ann, mask)


if __name__ == "__main__":
    main()
