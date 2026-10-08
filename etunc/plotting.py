"""
etunc.plotting
==============
Figures: maps of input fields, histograms and bin edges, and heatmaps of
bin statistics. Every function returns its figure, and saves and closes it
when given `fout`. No module calls matplotlib.use; scripts set the backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs

from etunc.binning import as_field_dict, edges_and_attrs, ensemble_spread, ensure_bin_coords, finite_flat
from etunc.config import DPI, LAT_BNDS, PROJECTION


def finish(fig, fout: str | Path | None):
    """Save and close `fig` if `fout` is given; otherwise leave it open for display."""
    if fout is not None:
        fout = Path(fout)
        fout.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fout, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return fig


def add_gridlines(ax, lat_bnds: slice = LAT_BNDS):
    x_gls = np.arange(-180, 181, 30)
    y_gls = np.arange(-90, 91, 30)
    y_gls = y_gls[(y_gls >= lat_bnds.start) & (y_gls <= lat_bnds.stop)]
    ax.gridlines(
        draw_labels=False,
        xlocs=x_gls,
        ylocs=y_gls,
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )


def map_ax(ax, lat_bnds: slice):
    ax.coastlines(color="k", lw=0.8)
    ax.set_extent((-180, 180, lat_bnds.start, lat_bnds.stop), crs=PROJECTION)
    add_gridlines(ax, lat_bnds)


def _spatial_mean_over_other_dims(da: xr.DataArray) -> xr.DataArray:
    other = [d for d in da.dims if d not in ("lat", "lon")]
    return da.mean(other) if other else da


# ------------------------------------------------------------------
# Maps and input diagnostics
# ------------------------------------------------------------------

def quick_map(
    da: xr.DataArray,
    fout: str | Path | None = None,
    title: str = "",
    robust: bool = True,
    vmin: float | None = 0,
    lat_bnds: slice = LAT_BNDS,
    **kwargs,
):
    """Map of a (lat, lon) field; non-spatial dims are averaged. Saved and closed if `fout` is given."""
    fig, ax = plt.subplots(figsize=(8, 3), layout="constrained", subplot_kw={"projection": PROJECTION})
    kwargs.setdefault("cmap", "viridis")
    _spatial_mean_over_other_dims(da).plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(), vmin=vmin, robust=robust, **kwargs
    )
    ax.set_title(title)
    map_ax(ax, lat_bnds)
    return finish(fig, fout)


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


def plot_edges_compare(
    edges: Mapping[str, xr.DataArray],
    *,
    drop_last: bool = True,
    ax=None,
    title: str = "",
    fout: str | Path | None = None,
):
    """Overlay several sets of bin edges, e.g. {"FPPE": ..., "CMIP6": ...} or LAI aggregations."""
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = ax.figure
    markers = "o^sPdvX*"
    for i, (label, e) in enumerate(edges.items()):
        (e[:-1] if drop_last else e).plot(ax=ax, marker=markers[i % len(markers)], alpha=0.75, label=label)
    ax.legend()
    ax.set_title(title)
    return finish(fig, fout)


# ------------------------------------------------------------------
# Bin statistics
# ------------------------------------------------------------------

def set_edge_ticks(ax, da: xr.DataArray, fmt: str):
    for ax_name, dim, set_ticks in (("x", "x_bin", ax.set_xticks), ("y", "y_bin", ax.set_yticks)):
        lower, upper = f"{ax_name}_bin_lower", f"{ax_name}_bin_upper"
        if lower not in da.coords:
            continue
        idx = da[dim].values
        pos = np.append(idx - 0.5, idx[-1] + 0.5)
        vals = np.append(da[lower].values, da[upper].values[-1])
        set_ticks(pos, [fmt.format(v) for v in vals], fontsize=7, rotation=90 if ax_name == "x" else 0)


def plot_bin_field(
    da: xr.DataArray,
    *,
    ax=None,
    hatch: xr.DataArray | None = None,
    edge_ticks: bool = True,
    fmt: str = "{:.2g}",
    title: str = "",
    fout: str | Path | None = None,
    **kwargs,
):
    """
    Heatmap of a 2-D (y_bin, x_bin) field, e.g. bs.sel(stats="mean"), a
    `frac_valid` or `ensemble_spread` result, or a ratio of spreads.

    hatch : boolean (y_bin, x_bin) field; True cells are hatched
        (e.g. ~significant, or a ratio range)
    edge_ticks : label bin boundaries with the edge values
    **kwargs : passed to DataArray.plot.pcolormesh
    """
    da = ensure_bin_coords(da)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    else:
        fig = ax.figure
    da.plot.pcolormesh(ax=ax, x="x_bin", y="y_bin", **kwargs)

    if hatch is not None:
        hatch = ensure_bin_coords(hatch).transpose("y_bin", "x_bin")
        yb, xb = hatch["y_bin"].values, hatch["x_bin"].values
        for j, i in zip(*np.nonzero(hatch.fillna(False).astype(bool).values)):
            ax.add_patch(Rectangle(
                (xb[i] - 0.5, yb[j] - 0.5), 1, 1, fill=False, hatch="///", lw=0, edgecolor="0.3",
            ))

    if edge_ticks:
        set_edge_ticks(ax, da, fmt)
    ax.set_xlabel(f"{da.attrs.get('x_variable', 'x')} $\\rightarrow$")
    ax.set_ylabel(f"{da.attrs.get('y_variable', 'y')} $\\rightarrow$")
    ax.set_title(title)
    return finish(fig, fout)


def plot_bin_facets(
    bs: xr.DataArray,
    dim: str,
    *,
    stat: str = "mean",
    signif: xr.DataArray | None = None,
    col_wrap: int = 6,
    fout: str | Path | None = None,
    **kwargs,
):
    """One panel per member/source along `dim` of statistic `stat`, optionally masked by `signif`."""
    da = bs.sel(stats=stat)
    if signif is not None:
        da = da.where(signif)
    kwargs.setdefault("cmap", "BrBG")
    fg = da.plot.pcolormesh(x="x_bin", y="y_bin", col=dim, col_wrap=col_wrap, **kwargs)
    finish(fg.fig, fout)
    return fg


def plot_bin_summary(
    bs: xr.DataArray,
    dim: str | None = None,
    *,
    title: str = "",
    fout: str | Path | None = None,
):
    """
    Three-panel diagnostic of a bin-stats result: mean ET (averaged over `dim`),
    total sample count (log scale), and spread (std across `dim` if given,
    otherwise the within-bin sample std).
    """
    bs = ensure_bin_coords(bs)
    mean = bs.sel(stats="mean")
    count = bs.sel(stats="count")
    if dim is not None:
        mean = mean.mean(dim)
        count = count.sum(dim)
        spread = ensemble_spread(bs, dim)
        spread_label = f"std of bin mean across {dim}"
    else:
        spread = np.sqrt(bs.sel(stats="var_samp"))
        spread_label = "within-bin std"
    units = bs.attrs.get("units", "?")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    plot_bin_field(mean, ax=axs[0], cmap="YlGnBu", cbar_kwargs={"label": f"mean [{units}]"})
    plot_bin_field(
        count.where(count > 0), ax=axs[1], cmap="magma_r",
        norm=mcolors.LogNorm(), cbar_kwargs={"label": "sample count"},
    )
    plot_bin_field(spread, ax=axs[2], cmap="viridis", cbar_kwargs={"label": f"{spread_label} [{units}]"})
    for ax in axs:
        ax.set_title("")
    fig.suptitle(title)
    return finish(fig, fout)
