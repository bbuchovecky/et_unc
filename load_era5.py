"""
load_era5.py
============
Load ERA5 monthly means from the NCAR GDEX archive (ds633.0 / ds633.5) and the
ERA5 grid file (LANDFRAC, ...).

Example
-------
>>> import load_era5 as le
>>> grid = le.load_era5_grid()
>>> rns = le.load_era5("msnswrf", "1995-01", "2014-12", kind="meanflux")
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import xarray as xr


GDEX_ROOTS = {
    # (years < SPLIT_YEAR, years >= SPLIT_YEAR)
    "sfc": (Path("/gdex/data/d633005/e5p.moda.an.sfc"), Path("/gdex/data/d633001/e5.moda.an.sfc")),
    "meanflux": Path("/gdex/data/d633001/e5.moda.fc.sfc.meanflux"),
    "accum": Path("/gdex/data/d633001/e5.moda.fc.sfc.accumu"),
}
SPLIT_YEAR = 1979

ERA5_GRID_PATH = Path("/glade/work/bbuchovecky/data/era5/e5.invariant.grid.nc")


def _root(kind: str, year: int) -> Path:
    paths = GDEX_ROOTS[kind]
    if isinstance(paths, tuple):
        return paths[0] if year < SPLIT_YEAR else paths[1]
    return paths


def load_era5(
    var_code: str,
    start: str,
    end: str,
    kind: str | None = None,
    chunks: dict | None = None,
) -> xr.DataArray | xr.Dataset:
    """
    Load an ERA5 monthly-mean variable for the years of [start, end].

    Parameters
    ----------
    var_code : str
        Parameter code as it appears at the end of the filename, before the
        grid tag (e.g. "msnswrf", or "128_167_2t" for 2 m temperature).
    start, end : str
        "YYYY-MM" strings. Whole years are loaded; the months are not trimmed.
    kind : str, optional
        Which GDEX_ROOTS stream to read ("sfc", "meanflux" or "accum"). If
        None, the first stream with a matching file is used, in that order.
    chunks : dict, optional
        Dask chunks. Defaults to {"time": 1}, one month per chunk.

    Returns
    -------
    xr.DataArray
        Lazily loaded and concatenated along time, with dims renamed to
        lat/lon. If the files hold several variables and none is named
        "VAR_<VAR_CODE>", the whole Dataset is returned with a warning.

    Raises
    ------
    FileNotFoundError
        If any year has no matching file, since that more often means a wrong
        `var_code` or `kind` than a real gap.
    """
    if chunks is None:
        chunks = {"time": 1}

    files = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        for k in [kind] if kind is not None else GDEX_ROOTS:
            subdir = _root(k, year) / f"{year}"
            matches = sorted(subdir.glob(f"*_{var_code}.*.nc"))
            if matches:
                break
        if not matches:
            raise FileNotFoundError(
                f"No file matching '*_{var_code}.*.nc' under {subdir}. "
                f"Verify var_code against actual filenames there, and "
                f"confirm {year} is assigned to the correct root."
            )
        files.extend(matches)

    if not files:
        raise FileNotFoundError(f"No files found for '{var_code}' in range {start} to {end}.")

    ds = xr.open_mfdataset(files, combine="by_coords", parallel=True)
    ds = ds.chunk(chunks).rename({"latitude": "lat", "longitude": "lon"})

    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        return ds[data_vars[0]]
    try:
        return ds["VAR_" + var_code.upper()]
    except KeyError:
        sys.stderr.write(
            f"WARNING: {len(data_vars)} data variables found for '{var_code}': "
            f"{data_vars}. Returning the full Dataset; select explicitly.\n"
        )
        return ds


@lru_cache(maxsize=1)
def load_era5_grid() -> xr.Dataset:
    """ERA5 grid dataset (LANDFRAC, ...) on the 0.25 deg lat/lon grid."""
    return xr.open_dataset(ERA5_GRID_PATH)
