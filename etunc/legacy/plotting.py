"""
etunc.legacy.plotting
=====================
Unmaintained and untested. Plotting functions of `etunc.plotting` that the
binning pipeline no longer uses (notebook diagnostics of binning inputs and
bin edges). The pipeline uses `etunc.plotting.plot_clim_hist` and
`plot_edges_compare` instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from etunc.binning import edges_and_attrs
from etunc.legacy.binning import as_field_dict, finite_flat
import cartopy.crs as ccrs

from etunc.config import LAT_BNDS, PROJECTION
from etunc.plotting import _spatial_mean_over_other_dims, finish, map_ax


def plot_input_maps(
    inputs: Mapping[str, xr.DataArray],
    title: str = "",
    fout: str | Path | None = None,
    lat_bnds: slice = LAT_BNDS,
    cmaps: Mapping[str, str] | None = None,
):
    """
    One map per binning input (e.g. the dict from `prepare_inputs`); non-spatial
    dims are averaged. Useful to check masking, units and grids before binning.
    """
    cmaps = {"et": "YlGnBu", "lai": "Greens", "ai": "BrBG_r", **(cmaps or {})}
    n = len(inputs)
    fig, axs = plt.subplots(
        n, 1, figsize=(8, 3 * n), layout="constrained", squeeze=False,
        subplot_kw={"projection": PROJECTION},
    )
    for ax, (key, da) in zip(axs[:, 0], inputs.items()):
        _spatial_mean_over_other_dims(da).plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(), robust=True, cmap=cmaps.get(key, "viridis"),
            cbar_kwargs={"label": f"{key} [{da.attrs.get('units', '?')}]"},
        )
        ax.set_title(key)
        map_ax(ax, lat_bnds)
    fig.suptitle(title)
    return finish(fig, fout)


def plot_input_hist(
    fields: xr.DataArray | Mapping[str, xr.DataArray],
    edges: xr.DataArray | np.ndarray | None = None,
    *,
    bins: int | np.ndarray = 100,
    value_range: tuple[float, float] | None = None,
    log: bool = False,
    ax=None,
    title: str = "",
    fout: str | Path | None = None,
):
    """
    Density histograms of each field's finite values, with bin edges as
    vertical lines. Use to compare e.g. LAI distributions across models
    against the pooled edges.
    """
    fields = as_field_dict(fields)
    flats = {k: finite_flat(v) for k, v in fields.items()}
    if value_range is None:
        # Cut the top 0.5%, so the long upper tail (e.g. AI in deserts) does not squash the rest
        pooled = np.concatenate(list(flats.values()))
        value_range = tuple(np.quantile(pooled, [0, 0.995]))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    else:
        fig = ax.figure
    for label, flat in flats.items():
        ax.hist(flat, bins=bins, range=value_range, density=True, histtype="step", lw=1, label=label)
    if edges is not None:
        for e in edges_and_attrs(edges)[0]:
            if value_range[0] <= e <= value_range[1]:
                ax.axvline(e, color="k", lw=0.6, ls="--", alpha=0.6)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel("density")
    ax.set_title(title)
    if len(flats) <= 12:
        ax.legend(fontsize=7, ncols=2)
    return finish(fig, fout)


def plot_edges(edges: xr.DataArray, fout: str | Path | None = None, title: str = ""):
    """Bin edges vs edge index; bottom panel omits the last (maximum) edge."""
    fig, axs = plt.subplots(2, 1, layout="constrained")
    edges.plot(ax=axs[0], marker="o")
    edges[:-1].plot(ax=axs[1], marker="o")
    for ax in axs:
        ax.set_xlim(0, len(edges))
    axs[0].set_title(title)
    return finish(fig, fout)
