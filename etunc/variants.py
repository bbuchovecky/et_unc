"""
etunc.variants
==============
Groups of closely related sources ("variants"), so that each group counts once
in the means, standard deviations and weights taken across the sources of a
family, however many variants it has:

- cmip: the CMIP6 models of one base model (`group_model_variants`),
  e.g. CESM2 and CESM2-WACCM, or UKESM1-0-LL and HadGEM3-GC31-LL;
- obs: the product combinations whose ET products are versions of one product
  (OBS_ET_PRODUCT_GROUPS), with the same LAI, P and Rn products;
- every other family: each source is its own group.

- `group_model_variants`: the CMIP6 source_ids of each base model.
- `source_groups`: the group of each member or source.
- `group_means`: nested means, the members of each source averaged, then the
  sources of each group, so a mean or std across the result's `group` dim
  counts each group once.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import xarray as xr


# Versions of one obs ET product, merged into one group: {ET product: group}. Products not
# listed are their own group (GLEAMv3.3a is a separate product from GLEAM v4)
OBS_ET_PRODUCT_GROUPS = {
    "GLEAMv4.3a": "GLEAMv4.3",
    "GLEAMv4.3b": "GLEAMv4.3",
    "PMLv2.2a_MODIS": "PMLv2.2",
    "PMLv2.2b": "PMLv2.2",
    "PMLv2.2c": "PMLv2.2",
}


# Base models whose source_id prefix differs from the rest of their modeling center's models,
# or is not the model's name
BASE_MODEL_ALIASES = {
    "EC": "EC-Earth",    # EC-Earth3, EC-Earth3-Veg, ...
    "NorCPM": "NorESM",  # NCC: NorCPM1 is built on NorESM1
    "UKESM": "HadGEM",   # Met Office: UKESM1 is built on HadGEM3-GC3.1
}


def group_model_variants(source_ids: Iterable[str]) -> dict[str, list[str]]:
    """
    {base model: sorted source_ids of its variants}, grouping the models of
    one modeling center by source_id. The base model is the source_id up to its
    first "-", without trailing version digits, then BASE_MODEL_ALIASES, e.g.
    CESM2, CESM2-WACCM -> "CESM"; MIROC6, MIROC-ES2L -> "MIROC";
    MPI-ESM1-2-LR, MPI-ESM-1-2-HAM -> "MPI"; UKESM1-0-LL, HadGEM3-GC31-LL -> "HadGEM".
    """
    groups: dict[str, list[str]] = {}
    for sid in sorted(set(source_ids)):
        base = re.sub(r"\d+$", "", sid.split("-")[0])
        groups.setdefault(BASE_MODEL_ALIASES.get(base, base), []).append(sid)
    return dict(sorted(groups.items()))


def source_groups(family: str, da: xr.DataArray | xr.Dataset, dim: str) -> np.ndarray:
    """
    Group label of each entry of `da` along `dim` (the members, with a `source` coord, or the
    sources). For obs, the label is the combination with the ET product replaced by its group,
    "<ET group>-<lai>-<pr>-<rn>", from the `<v>_product` coords.
    """
    sources = [str(s) for s in da["source"].values]
    if da["source"].dims != (dim,):
        raise ValueError(f"`source` must be along {dim!r}, not {da['source'].dims}")
    if family == "cmip":
        group_of = {sid: g for g, sids in group_model_variants(sources).items() for sid in sids}
        return np.array([group_of[s] for s in sources])
    if family == "obs":
        et = [OBS_ET_PRODUCT_GROUPS.get(str(p), str(p)) for p in da["et_product"].values]
        others = [[str(p) for p in da[f"{v}_product"].values] for v in ("lai", "pr", "rn")]
        return np.array(["-".join(c) for c in zip(et, *others)])
    return np.array(sources)


def group_means(da: xr.DataArray | xr.Dataset, family: str, dim: str) -> xr.DataArray | xr.Dataset:
    """
    `da` averaged within each group of variants (`source_groups`), along a new `group` dim
    (sorted labels) that replaces `dim`. Nested means, NaN skipped at each level: with `dim`
    the members, the members of each source are averaged first, then the sources of each
    group, so a gridcell or bin missing in one variant takes the group's other variants
    only. A mean or std across `group` then weighs every group once.
    """
    groups = source_groups(family, da, dim)
    if dim != "source":
        # Members of each source (one group per source, so the source's group is its members')
        group_of = dict(zip((str(s) for s in da["source"].values), groups))
        da = da.groupby("source").mean(keep_attrs=True)
        groups = np.array([group_of[str(s)] for s in da["source"].values])
    return da.assign_coords(group=("source", groups)).groupby("group").mean(keep_attrs=True)
