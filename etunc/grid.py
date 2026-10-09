"""
etunc.grid
==========
Target grids and regridders shared by every script that puts a dataset on a
common grid, plus grid helpers: coordinate checks and snapping (`on_grid`),
land masks, and cell areas.

Target grids
------------
There are two standard grids (`RESOLUTIONS`, named by `grid_tag`), used for
every saved output. `target_grid` also builds the same kind of grid at any
spacing that divides 180 (e.g. for resolution-sensitivity tests). All are
global and regular, with lon in [-180, 180], lat ascending, and cell edges at
multiples of the spacing from -90 and -180:

    "0.5deg": 360 x 720, centers lat -89.75 ... 89.75, lon -179.75 ... 179.75
    "1deg":   180 x 360, centers lat -89.5 ... 89.5,   lon -179.5 ... 179.5

Always get a grid from `target_grid` (by spacing or tag) instead of building
one by hand, so that every regridded product has identical lat/lon values and
passes `check_same_grid` (below). The grids carry their cell edges (`lat_b`,
`lon_b`) for conservative regridding.

Regridders
----------
All of them build an xESMF regridder onto `target_grid(res)`.

conservative_regridder : first-order conservative from a regular 1-D lat/lon
    grid with explicit cell edges (`source_grid`), cached per source grid.
    Used for the 0.1 deg obs products.
bounded_conservative_regridder : first-order conservative from any 1-D
    lat/lon grid (e.g. Gaussian), with cell edges from CF bounds or midpoints
    (`bounded_source_grid`). Used for CMIP6 annual means.
bilinear_regridder : bilinear, periodic in lon. Used for ILAMB 1 deg products.
make_regridder : any xESMF method from an arbitrary source grid (1-D or 2-D
    lat/lon, edges inferred by xESMF if absent).

All but `make_regridder` set target cells outside the source domain to NaN
(`unmapped_to_nan=True`).
"""
from __future__ import annotations

import warnings

from typing import Iterable

import numpy as np
import regionmask as regmask
import xarray as xr
import xesmf as xe

from etunc.config import EARTH_RADIUS, LAT_BNDS, LF_THRESH


RESOLUTIONS = {"0.5deg": 0.5, "1deg": 1.0}  # grid tag (output directory name) -> spacing [deg]

LAT_ATTRS = {"units": "degrees_north", "standard_name": "latitude"}
LON_ATTRS = {"units": "degrees_east", "standard_name": "longitude"}

_CONSERVATIVE: dict[tuple, xe.Regridder] = {}

NA_THRES = 0.5  # default: target cells with more than this fraction of missing source area are NaN


# ------------------------------------------------------------------
# Grids
# ------------------------------------------------------------------

def _spacing(res: float | str) -> float:
    """Grid spacing [deg] of a supported resolution, given as spacing (0.5) or tag ("0.5deg")."""
    spacing = RESOLUTIONS.get(res, res) if isinstance(res, str) else res
    if spacing not in RESOLUTIONS.values():
        raise ValueError(f"unsupported resolution {res!r}; use one of {RESOLUTIONS}")
    return float(spacing)


def grid_tag(res: float | str) -> str:
    """Tag of a supported resolution, e.g. 1.0 -> "1deg"."""
    spacing = _spacing(res)
    return next(tag for tag, r in RESOLUTIONS.items() if r == spacing)


def _any_spacing(res: float | str) -> float:
    """Grid spacing [deg] given as a tag of RESOLUTIONS or as any positive number that divides 180."""
    if isinstance(res, str):
        if res not in RESOLUTIONS:
            raise ValueError(f"unsupported resolution {res!r}; use a tag of {RESOLUTIONS} or a spacing")
        return float(RESOLUTIONS[res])
    spacing = float(res)
    n = 180 / spacing if spacing > 0 else 0.5
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"unsupported resolution {res!r}; the spacing must divide 180")
    return spacing


def target_grid(res: float | str) -> xr.Dataset:
    """
    Global grid at spacing `res` with cell edges: a standard tag ("0.5deg",
    "1deg") or any spacing that divides 180 (e.g. 0.25, 2.0).
    """
    res = _any_spacing(res)
    nlat, nlon = round(180 / res), round(360 / res)
    # Centers stop half a cell short of 90/180, so float round-off in arange cannot add a
    # cell; the edges come from linspace so they land exactly on -90/90 and -180/180
    return xr.Dataset(coords={
        "lat": ("lat", np.arange(-90 + res / 2, 90, res), LAT_ATTRS),
        "lon": ("lon", np.arange(-180 + res / 2, 180, res), LON_ATTRS),
        "lat_b": ("lat_b", np.linspace(-90, 90, nlat + 1)),
        "lon_b": ("lon_b", np.linspace(-180, 180, nlon + 1)),
    })


def cell_bounds(centers: np.ndarray, name: str) -> np.ndarray:
    """Cell edges of an ascending, regularly spaced 1-D coordinate."""
    diffs = np.diff(centers)
    step = float(np.median(diffs))
    if not np.allclose(diffs, step, rtol=0, atol=1e-6 * max(1.0, abs(step))):
        raise ValueError(f"{name} is not regularly spaced (spacings {np.unique(diffs)})")
    return np.append(centers - step / 2, centers[-1] + step / 2)


def source_grid(da: xr.DataArray) -> xr.Dataset:
    """Grid of `da` (1-D ascending, regularly spaced lat/lon) with explicit cell edges for xESMF."""
    lat, lon = da["lat"].values, da["lon"].values
    return xr.Dataset(coords={
        "lat": ("lat", lat),
        "lon": ("lon", lon),
        "lat_b": ("lat_b", np.clip(cell_bounds(lat, "lat"), -90, 90)),
        "lon_b": ("lon_b", cell_bounds(lon, "lon")),
    })


def cell_edges(centers: np.ndarray, bnds: np.ndarray | None = None, name: str = "") -> np.ndarray:
    """
    Cell edges (n + 1) of a monotonic 1-D coordinate with any spacing: from CF
    bounds `bnds` (n, 2) if given, else midpoints between centers with the
    outer edges half a cell beyond the outer centers.
    """
    if bnds is not None:
        edges = np.append(bnds[:, 0], bnds[-1, 1])
    else:
        mid = (centers[1:] + centers[:-1]) / 2
        edges = np.concatenate([[2 * centers[0] - mid[0]], mid, [2 * centers[-1] - mid[-1]]])
    diffs = np.diff(edges)
    lo, hi = np.minimum(edges[:-1], edges[1:]), np.maximum(edges[:-1], edges[1:])
    if not (np.all(diffs > 0) or np.all(diffs < 0)) or not np.all((centers >= lo) & (centers <= hi)):
        raise ValueError(f"{name} cell edges are not monotonic or do not enclose the centers")
    return edges


def bounded_source_grid(ds: xr.Dataset | xr.DataArray) -> xr.Dataset:
    """
    Grid of `ds` (1-D lat/lon, any spacing) with cell edges for xESMF, taken
    from the CF bounds variables (`lat_bnds`/`lon_bnds`, or the coord's
    "bounds" attr) when `ds` has them and from midpoints otherwise.
    Latitude edges are clipped to [-90, 90].
    """
    def edges(name: str) -> np.ndarray:
        bnds_name = ds[name].attrs.get("bounds", f"{name}_bnds")
        bnds = ds[bnds_name].values if isinstance(ds, xr.Dataset) and bnds_name in ds else None
        return cell_edges(ds[name].values, bnds, name)

    return xr.Dataset(coords={
        "lat": ("lat", ds["lat"].values),
        "lon": ("lon", ds["lon"].values),
        "lat_b": ("lat_b", np.clip(edges("lat"), -90, 90)),
        "lon_b": ("lon_b", edges("lon")),
    })


# ------------------------------------------------------------------
# Regridders
# ------------------------------------------------------------------

def make_regridder(src: xr.Dataset | xr.DataArray, res: float | str, method: str, **kwargs) -> xe.Regridder:
    """xESMF regridder with `method` from the lat/lon grid of `src` onto `target_grid(res)`."""
    return xe.Regridder(src, target_grid(res), method, **kwargs)


def conservative_regridder(da: xr.DataArray, res: float | str) -> xe.Regridder:
    """Conservative regridder from the regular grid of `da` onto `target_grid(res)`, cached per grid."""
    # Keyed on the exact coordinate values, so the yearly files of one product reuse a
    # single weight matrix (building one from a 0.1 deg grid is slow)
    key = (_any_spacing(res), da["lat"].values.tobytes(), da["lon"].values.tobytes())
    if key not in _CONSERVATIVE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _CONSERVATIVE[key] = make_regridder(source_grid(da), res, "conservative", unmapped_to_nan=True)
    return _CONSERVATIVE[key]


def bounded_conservative_regridder(ds: xr.Dataset | xr.DataArray, res: float | str) -> xe.Regridder:
    """Conservative regridder from the 1-D grid of `ds` (edges from `bounded_source_grid`) onto `target_grid(res)`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_regridder(bounded_source_grid(ds), res, "conservative", unmapped_to_nan=True)


def bilinear_regridder(da: xr.DataArray | xr.Dataset, res: float | str) -> xe.Regridder:
    """Bilinear regridder (periodic in lon) from the lat/lon grid of `da` onto `target_grid(res)`."""
    src = xr.Dataset(coords={"lat": da["lat"], "lon": da["lon"]})
    return make_regridder(src, res, "bilinear", periodic=True, unmapped_to_nan=True)


def regrid_with_na_thres(da: xr.DataArray, regridder: xe.Regridder, na_thres: float = NA_THRES) -> xr.DataArray:
    """
    Apply a conservative `regridder` skipping NaN: each target cell is the
    area-weighted mean of its valid source cells, and NaN where more than
    `na_thres` of its area is missing. Keeps the dtype and attrs (minus
    "grid_mapping") and sets standard lat/lon attrs.
    """
    # skipna renormalizes by the valid source area, so a coastal target cell is the mean of
    # its land source cells rather than diluted by ocean NaN; na_thres sets how much missing
    # area is tolerated, and so how far the regridded land extends along the coasts
    out = regridder(da, skipna=True, na_thres=na_thres, keep_attrs=True)
    out = out.astype(da.dtype)  # xESMF returns float64 even for float32 input
    out.attrs.pop("grid_mapping", None)  # PML's "crs" variable is not carried over
    out["lat"].attrs = dict(LAT_ATTRS)
    out["lon"].attrs = dict(LON_ATTRS)
    return out


# ------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------

def coord_name(ds: xr.DataArray | xr.Dataset, candidates: list[str], standard_names: list[str]) -> str | None:
    """First of `candidates` among the coords of `ds`, else the first coord with one of `standard_names`."""
    for name in candidates:
        if name in ds.coords:
            return name
    for name, var in ds.coords.items():
        if str(var.attrs.get("standard_name", "")) in standard_names:
            return name
    return None


def approx_resolution(ds: xr.DataArray | xr.Dataset) -> tuple[float | None, float | None]:
    """Median (lat, lon) spacing [deg] of the unique coordinate values; None if it cannot be found."""
    lat_name = coord_name(ds, ["lat", "latitude", "nav_lat"], ["latitude"])
    lon_name = coord_name(ds, ["lon", "longitude", "nav_lon"], ["longitude"])

    def spacing(name: str | None) -> float | None:
        if name is None or name not in ds.coords:
            return None
        arr = np.asarray(ds[name].values)
        if arr.ndim == 0:
            return None
        vals = np.unique(arr[np.isfinite(arr)])
        if vals.size < 2:
            return None
        diffs = np.diff(np.sort(vals))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return None
        return float(np.nanmedian(diffs))

    return spacing(lat_name), spacing(lon_name)


# ------------------------------------------------------------------
# Coordinate checks and land masks
# ------------------------------------------------------------------

def check_coords(da: xr.DataArray, coords: Iterable[str]) -> bool:
    """Check that da has non-empty coords."""
    for coord in coords:
        if coord not in da.coords:
            return False
        if da[coord].ndim == 0:
            continue  # scalar coord counts as present
        if len(da[coord]) == 0:
            return False
    return True


def equal_coords(
    a: xr.DataArray | xr.Dataset,
    b: xr.DataArray | xr.Dataset,
    coords: Iterable[str],
    atol: float = 1e-3,
) -> bool:
    """Check that a and b have the same coords (numeric coords within atol)."""
    if not check_coords(a, coords) or not check_coords(b, coords):
        return False
    for crd in coords:
        if a[crd].shape != b[crd].shape:
            return False
        # Non-numeric coords (datetime64, cftime `time`) are only compared by shape
        if np.issubdtype(a[crd].dtype, np.number):
            if not np.allclose(a[crd], b[crd], atol=atol):
                return False
    return True


def check_same_grid(
    da: xr.DataArray,
    ref: xr.DataArray | xr.Dataset,
    label: str,
    atol: float = 1e-3,
) -> None:
    """Raise if `da` does not share its lat/lon grid with `ref`, rather than silently reindex."""
    if da.sizes.get("lat") != ref.sizes.get("lat") or da.sizes.get("lon") != ref.sizes.get("lon"):
        raise ValueError(
            f"{label}: grid shape lat={da.sizes.get('lat')}, lon={da.sizes.get('lon')} does not match "
            f"the reference grid lat={ref.sizes.get('lat')}, lon={ref.sizes.get('lon')}."
        )
    if not equal_coords(da, ref, ("lat", "lon"), atol=atol):
        raise ValueError(f"{label}: lat/lon values differ from the reference grid by more than {atol}.")


def on_grid(da: xr.DataArray, res: float | str, label: str) -> xr.DataArray:
    """
    `da` with the exact lat/lon of `target_grid(res)`, cut to LAT_BNDS, after
    checking that it is on that grid (`check_same_grid`). Products whose
    coordinates differ only by float round-off then line up cell for cell.
    """
    grid = target_grid(res)
    check_same_grid(da, grid, label)
    # Overwrite with the exact grid values: `bin_stats` aligns with join="exact"
    da = da.assign_coords(lat=grid.lat, lon=grid.lon)
    return da.sel(lat=LAT_BNDS)


def format_grid(da: xr.DataArray) -> xr.DataArray:
    """Longitude in [-180, 180] and latitude ascending."""
    return da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon").sortby("lat")


def mask_greenland(landfrac: xr.DataArray, lf_thresh: float = LF_THRESH) -> xr.DataArray:
    """Land mask: True where landfrac > lf_thresh, excluding Greenland/Iceland (AR6 region 0)."""
    mask = regmask.defined_regions.ar6.land.mask(landfrac.lon, landfrac.lat)
    # Strictly greater: a cell at exactly lf_thresh is not land. NaN landfrac is not land either
    return xr.where((mask == 0) & (landfrac > lf_thresh), False, landfrac > lf_thresh)


def land_mask(grid: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """
    Natural Earth land mask without Greenland/Iceland. (Same Natural Earth
    mask as `etunc.legacy.compute_cell_area`, which keeps Greenland/Iceland.)
    """
    # Binary by cell center: a coastal cell is land only if its center is on land, with no
    # fractional coverage. This mask sets the land area binned by bin_obs and bin_cmip
    land = regmask.defined_regions.natural_earth_v5_1_2.land_50.mask(grid.lon, grid.lat)
    return mask_greenland(xr.where(land.notnull(), 1.0, 0.0))


def product_mask(da: xr.DataArray, land: xr.DataArray, min_frac: float, label: str) -> xr.Dataset:
    """
    Valid-month counts of one product's monthly field `da` (time, lat, lon) and
    its mask: `land` gridcells valid in at least `min_frac` of the product's
    valid months (months with any valid land gridcell). Months without valid
    land data add 0 to every gridcell's count, so they only need to be left
    out of the denominator. Raises if no month has valid land data.
    """
    # True where the product has data on a land gridcell, per month
    valid = da.notnull() & land
    # The product's valid months: months with data on at least one land
    # gridcell. This is the denominator of the valid fraction.
    n_months = int(valid.any(("lat", "lon")).sum())
    if n_months == 0:
        raise ValueError(f"{label}: no valid land data")
    # Number of months with data at each gridcell (0 over the ocean)
    n_valid = valid.sum("time")
    return xr.Dataset({
        "n_product_months": n_months,
        "n_valid_months": n_valid.astype("i2"),
        "frac_valid_months": (n_valid / n_months).where(land).astype("f4"),  # NaN over the ocean
        # Compare counts rather than fractions, so 1.0 means exactly "every month"
        "product_mask": land & (n_valid >= min_frac * n_months),
    })


# ------------------------------------------------------------------
# Cell areas
# ------------------------------------------------------------------

def cell_area(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_bnds: np.ndarray | None = None,
    lon_bnds: np.ndarray | None = None,
) -> np.ndarray:
    """
    Cell areas [m2] of a 1-D lat/lon grid, shape (lat, lon). Port of ILAMB's
    `ilamblib.CellAreas`, so the package does not need ILAMB (whose import
    initializes MPI).

    With `lat_bnds`/`lon_bnds` (shape (n, 2)) the areas are exact. Otherwise
    the cell edges are the midpoints between centers, extrapolated by half a
    cell at both ends and clipped to [-90, 90] and [-180, 180]; longitudes in
    [0, 360] are shifted by -180 first (which leaves the widths unchanged).
    """
    # Kept line for line as in ILAMB so the areas stay bit-identical; do not simplify
    if lat_bnds is not None and lon_bnds is not None:
        return EARTH_RADIUS**2 * np.outer(
            (
                np.sin(lat_bnds[:, 1] * np.pi / 180.0)
                - np.sin(lat_bnds[:, 0] * np.pi / 180.0)
            ),
            (lon_bnds[:, 1] - lon_bnds[:, 0]) * np.pi / 180.0,
        )

    x = np.zeros(lon.size + 1)
    x[1:-1] = 0.5 * (lon[1:] + lon[:-1])
    x[0] = lon[0] - 0.5 * (lon[1] - lon[0])
    x[-1] = lon[-1] + 0.5 * (lon[-1] - lon[-2])
    if x.max() > 181:
        x -= 180
    x = x.clip(-180, 180)
    x *= np.pi / 180.0

    y = np.zeros(lat.size + 1)
    y[1:-1] = 0.5 * (lat[1:] + lat[:-1])
    y[0] = lat[0] - 0.5 * (lat[1] - lat[0])
    y[-1] = lat[-1] + 0.5 * (lat[-1] - lat[-2])
    y = y.clip(-90, 90)
    y *= np.pi / 180.0

    dx = EARTH_RADIUS * (x[1:] - x[:-1])
    dy = EARTH_RADIUS * (np.sin(y[1:]) - np.sin(y[:-1]))
    areas = np.outer(dx, dy).T  # (lon, lat) -> (lat, lon)

    return areas

