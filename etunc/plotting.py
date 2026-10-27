"""
etunc.plotting
==============
Figures: maps of input fields, histograms and bin edges, and heatmaps of
bin statistics. Every function returns its figure, and saves and closes it
when given `fout`. No module calls matplotlib.use; scripts set the backend.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs

from etunc.binning import edges_and_attrs, ensemble_spread, ensure_bin_coords
from etunc.config import DPI, HIST_BINS, LAT_BNDS, PROJECTION


# quick_map style of each variable's climatology map (obs and CMIP6 names)
MAP_KWARGS = {
    "et":        {"cmap": "YlGnBu"},
    "et_pr":     {"cmap": "YlGnBu"},
    "evspsbl":   {"cmap": "YlGnBu"},
    "lai":       {"cmap": "Greens"},
    "pr":        {"cmap": "Blues"},
    "rn":        {"cmap": "YlOrRd", "vmin": None},  # Rn can be negative; quick_map's default vmin is 0
    "ai":        {"cmap": "BrBG_r"},
    "land_area": {"cmap": "RdPu"},
    "land_area_fraction": {"cmap": "RdPu"},
}


def finish(fig, fout: str | Path | None):
    """Save and close `fig` if `fout` is given; otherwise leave it open for display."""
    if fout is not None:
        fout = Path(fout)
        fout.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fout, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return fig


def add_gridlines(ax, lat_bnds: slice = LAT_BNDS):
    """Dashed 30 deg gridlines, latitudes limited to `lat_bnds`."""
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
    """Coastlines, global extent cut to `lat_bnds`, and gridlines on a map axis."""
    ax.coastlines(color="k", lw=0.8)
    ax.set_extent((-180, 180, lat_bnds.start, lat_bnds.stop), crs=PROJECTION)
    add_gridlines(ax, lat_bnds)


def _spatial_mean_over_other_dims(da: xr.DataArray) -> xr.DataArray:
    other = [d for d in da.dims if d not in ("lat", "lon")]
    return da.mean(other) if other else da


def facets(
    n: int, *, ncols: int = 4, maps: bool = False, panel_size: tuple[float, float] = (4.2, 3.0), **kwargs,
):
    """
    Figure with `n` panels in rows of `ncols` (unused panels removed), map
    panels if `maps`; returns (fig, list of the n axes).
    """
    ncols = min(n, ncols)
    nrows = math.ceil(n / ncols)
    subplot_kw = {"projection": PROJECTION} if maps else None
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), squeeze=False,
        layout="constrained", subplot_kw=subplot_kw, **kwargs,
    )
    for ax in axs.flat[n:]:
        ax.remove()
    return fig, list(axs.flat[:n])


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


def save_map(da: xr.DataArray, variable: str, fout: str | Path, title: str = ""):
    """
    `quick_map` of a climatology of `variable` with its MAP_KWARGS style and a
    "<variable> [<units>]" colorbar label, saved to `fout` (printed).
    """
    fig = quick_map(
        da, fout, title=title,
        cbar_kwargs={"label": f"{variable} [{da.attrs.get('units', '?')}]"}, **MAP_KWARGS[variable],
    )
    print(fout)
    return fig


def save_ensemble_map(
    da: xr.DataArray,
    variable: str,
    fout: str | Path | None = None,
    dim: str = "member",
    title: str = "",
    lat_bnds: slice = LAT_BNDS,
):
    """
    Three stacked maps of a climatology of `variable` with an ensemble dim
    `dim`: the mean along `dim` (MAP_KWARGS style), the standard deviation
    (ddof=0, as `binning.ensemble_spread`) and the range (max - min), each
    gridcell over the members with a value there; a single map (`save_map`)
    if there is only one member. Saved (path printed) if `fout` is given.
    """
    if da.sizes[dim] == 1:
        return save_map(da.squeeze(dim, drop=True), variable, fout, title=title)
    units = da.attrs.get("units", "?")
    panels = {
        "ensemble mean": (da.mean(dim), {"vmin": 0, **MAP_KWARGS[variable]}),
        "ensemble standard deviation": (da.std(dim), {"vmin": 0, "cmap": "magma_r"}),
        "ensemble range (max - min)": (da.max(dim) - da.min(dim), {"vmin": 0, "cmap": "magma_r"}),
    }
    fig, axs = plt.subplots(3, 1, figsize=(8, 9), layout="constrained", subplot_kw={"projection": PROJECTION})
    for ax, (label, (field, kwargs)) in zip(axs, panels.items()):
        field.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(), robust=True,
            cbar_kwargs={"label": f"{variable} [{units}]"}, **kwargs,
        )
        ax.set_title(label)
        map_ax(ax, lat_bnds)
    fig.suptitle(f"{title}, {da.sizes[dim]} members" if title else f"{da.sizes[dim]} members")
    finish(fig, fout)
    if fout is not None:
        print(fout)
    return fig


def save_member_maps(
    da: xr.DataArray,
    variable: str,
    fout: str | Path | None = None,
    dim: str = "member",
    ncols: int = 5,
    title: str = "",
    lat_bnds: slice = LAT_BNDS,
):
    """
    One map per member (along `dim`) of a climatology of `variable`, in rows
    of `ncols`, titled with the member label. Every panel shares one color
    scale (MAP_KWARGS style) from the 2nd to 98th percentile over all
    members, with vmin = 0 unless the style sets it, so members compare
    directly. Saved (path printed) if `fout` is given.
    """
    kwargs = {"vmin": 0, **MAP_KWARGS[variable]}
    lo, hi = np.nanpercentile(da.values, [2, 98])
    if kwargs["vmin"] is None:
        kwargs["vmin"] = lo
    kwargs["vmax"] = hi
    n = da.sizes[dim]
    fig, axs = facets(n, ncols=ncols, maps=True, panel_size=(3.2, 1.6))
    for i, ax in enumerate(axs):
        pm = da.isel({dim: i}).plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, add_labels=False, **kwargs,
        )
        ax.set_title(str(da[dim].values[i]), fontsize=8)
        map_ax(ax, lat_bnds)
    # One horizontal colorbar under all panels; values outside [vmin, vmax] are saturated
    fig.colorbar(pm, ax=axs, location="bottom", shrink=0.5, aspect=40, extend="max" if kwargs["vmin"] == 0 else "both",
                 label=f"{variable} [{da.attrs.get('units', '?')}]")
    fig.suptitle(f"{title}, {n} members" if title else f"{n} members")
    finish(fig, fout)
    if fout is not None:
        print(fout)
    return fig


def plot_mask_agreement(
    masks: xr.DataArray,
    land: xr.DataArray,
    title: str = "",
    fout: str | Path | None = None,
    *,
    label: str = "products with the gridcell in their mask",
    highlight_common: bool = False,
):
    """
    Map of the number of products whose mask (`masks`, boolean (product, lat,
    lon)) includes each `land` gridcell, one discrete color per count. The
    title gives the size of the common mask (all products). With
    `highlight_common`, the common mask is orange instead of the darkest purple.
    """
    n = masks.sizes["product"]
    # 0..n products per land gridcell; NaN (blank) over the ocean
    agree = masks.sum("product").where(land)
    if highlight_common:
        # Shades of purple for 0..n-1, and orange for n (all products agree) so it stands out
        cmap = mcolors.ListedColormap([*plt.get_cmap("Purples")(np.linspace(0.05, 0.75, n)), "#e66101"])
    else:
        cmap = plt.get_cmap("Purples", n + 1)
    # Bin edges at half-integers, so each integer count gets its own color
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n + 1.5), cmap.N)
    fig, ax = plt.subplots(figsize=(9, 3.6), layout="constrained", subplot_kw={"projection": PROJECTION})
    pm = agree.plot.pcolormesh(ax=ax, transform=PROJECTION, cmap=cmap, norm=norm, add_colorbar=False)
    map_ax(ax, LAT_BNDS)
    fig.colorbar(pm, ax=ax, ticks=np.arange(n + 1), label=label)
    ax.set_title(f"{title}\ncommon mask (all {n} products): {int((agree == n).sum())} of {int(land.sum())} land cells")
    return finish(fig, fout)


def plot_clim_hist(
    fields: Mapping[str, xr.DataArray | tuple[np.ndarray, np.ndarray]],
    bins: np.ndarray,
    *,
    pooled: tuple[np.ndarray, np.ndarray] | None = None,
    edges: xr.DataArray | np.ndarray | None = None,
    xlabel: str = "",
    ax=None,
    title: str = "",
    fout: str | Path | None = None,
):
    """
    Density histograms of climatological LAI or AI on fixed `bins`
    (`config.HIST_BINS`), one line per field of `fields` ({label: DataArray}
    of equally weighted values, e.g. one per source, or {label: (values,
    weights)}), so histograms of different figures share their bin widths.
    Values outside `bins` are counted in the end bins; the largest
    fraction outside is printed and noted on the figure. `pooled` = (values,
    weights) adds the weighted distribution of all values (e.g. from
    `edges.hierarchical_weights`) as a thick black line, and `edges` the bin
    edges as vertical lines.
    """
    def density(x, w=None):
        h, _ = np.histogram(np.clip(x, bins[0], bins[-1]), bins, weights=w)
        return h / h.sum() / np.diff(bins)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    else:
        fig = ax.figure
    many = len(fields) > 12  # thin grey lines without a legend
    outside = {}
    for label, field in fields.items():
        if isinstance(field, tuple):
            x, w = field
        else:
            x = np.asarray(field.values, dtype=float).ravel()
            x, w = x[np.isfinite(x)], None
        outside[label] = np.average((x < bins[0]) | (x > bins[-1]), weights=w)
        style = {"color": "0.5", "lw": 0.6, "alpha": 0.6} if many else {"lw": 1.2, "label": label}
        ax.stairs(density(x, w), bins, **style)
    if pooled is not None:
        ax.stairs(density(*pooled), bins, color="k", lw=2, label="pooled (weighted)")
    if edges is not None:
        for e in edges_and_attrs(edges)[0]:
            ax.axvline(min(max(e, bins[0]), bins[-1]), color="tab:red", lw=0.7, ls="--")

    worst = max(outside, key=outside.get)
    note = f"outside [{bins[0]:g}, {bins[-1]:g}], counted in the end bins: up to {100 * outside[worst]:.2f}% ({worst})"
    print(f"{title or xlabel}: values {note}")
    ax.text(0.99, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    if not many or pooled is not None:
        ax.legend(fontsize=7, ncols=2, loc="center right")
    return finish(fig, fout)


def save_clim_diagnostics(
    inputs: xr.Dataset, fig_dir: Path, family: str, grid: str, period: str, title: str = "",
):
    """
    Diagnostics of the binning inputs of one family (written by the prep
    scripts): maps of the climatological ET, LAI, P, Rn and AI (mean,
    standard deviation and range over members; one map for one member),
    maps of the climatological LAI, P and Rn of every member (with more than
    one member), and histograms of climatological LAI and AI on
    `config.HIST_BINS`, one line per source. Saved as
    fig_dir/<family>.<var>.<grid>.{map,map_members,hist}.<period>.png.
    """
    title = f"{title}, {inputs.attrs.get('time_period', period)}"
    clim = {"et": inputs["et"].mean("year", keep_attrs=True), **{v: inputs[v] for v in ("lai", "pr", "rn", "ai")}}
    # The annual ET/P, if the prep script saved it (BIN_VARIABLE = "et_pr")
    if "et_pr" in inputs:
        clim["et_pr"] = inputs["et_pr"].mean("year", keep_attrs=True)
    for var, da in clim.items():
        save_ensemble_map(da, var, fig_dir / f"{family}.{var}.{grid}.map.{period}.png", title=title)
    if inputs.sizes["member"] > 1:
        for var in ("lai", "pr", "rn"):
            save_member_maps(inputs[var], var, fig_dir / f"{family}.{var}.{grid}.map_members.{period}.png", title=title)
    sources = list(dict.fromkeys(inputs["source"].values.tolist()))
    for var in ("lai", "ai"):
        fields = {s: inputs[var].where(inputs["source"] == s) for s in sources}
        fout = fig_dir / f"{family}.{var}_clim.{grid}.hist.{period}.png"
        plot_clim_hist(fields, HIST_BINS[var], xlabel=f"climatological {var}", title=f"{title}, {len(sources)} sources",
                       fout=fout)
        print(fout)


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
    """Label the bin boundaries of a (y_bin, x_bin) heatmap with the bin edge values."""
    for ax_name, dim, set_ticks in (("x", "x_bin", ax.set_xticks), ("y", "y_bin", ax.set_yticks)):
        lower, upper = f"{ax_name}_bin_lower", f"{ax_name}_bin_upper"
        if lower not in da.coords:
            continue
        idx = da[dim].values
        # Ticks at index +- 0.5 assume contiguous bin indices. Zero-width LAI = 0 bins are
        # dropped from the start, which keeps the rest contiguous; a gap mid-axis would
        # shift these ticks off the pcolormesh cell boundaries (also in `hatch_bins`)
        pos = np.append(idx - 0.5, idx[-1] + 0.5)
        vals = np.append(da[lower].values, da[upper].values[-1])
        set_ticks(pos, [fmt.format(v) for v in vals], fontsize=7, rotation=90 if ax_name == "x" else 0)


def hatch_bins(ax, hatch: xr.DataArray):
    """Hatch the cells of a (y_bin, x_bin) heatmap where `hatch` is True."""
    hatch = hatch.transpose("y_bin", "x_bin")
    yb, xb = hatch["y_bin"].values, hatch["x_bin"].values
    for j, i in zip(*np.nonzero(hatch.fillna(False).astype(bool).values)):
        ax.add_patch(Rectangle(
            (xb[i] - 0.5, yb[j] - 0.5), 1, 1, fill=False, hatch="///", lw=0, edgecolor="0.3",
        ))


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
        hatch_bins(ax, ensure_bin_coords(hatch))

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


def plot_bin_means(
    bs: xr.DataArray,
    dim: str,
    *,
    title: str = "",
    fout: str | Path | None = None,
    col_wrap: int = 6,
    size: float = 3,
    hatch: xr.DataArray | None = None,
    cbar_shrink: float | None = None,
    cmap: str = "YlGnBu",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    One heatmap of the bin mean of `bs` per entry along `dim` (e.g. "combo",
    "source_id") on a shared colour scale. Panels share axes, so ticks show
    the bin edge values when all panels share edges, and quantile levels when
    each has its own edges. Panel titles are the `dim` label, plus its
    `time_period` when that is a coord along `dim`.

    hatch : boolean (dim, y_bin, x_bin) field; True bins are hatched
        (e.g. ~`binning.test_significance`)
    cbar_shrink : colorbar length as a fraction of the grid height; by default
        about two panels tall, since the colorbar otherwise spans every row
    """
    col_wrap = min(col_wrap, bs.sizes[dim])
    if cbar_shrink is None:
        nrows = -(-bs.sizes[dim] // col_wrap)
        cbar_shrink = min(1.0, 2 / nrows)
    fg = plot_bin_facets(
        bs, dim=dim, col_wrap=col_wrap, size=size, cmap=cmap, robust=True, vmin=vmin, vmax=vmax,
        cbar_kwargs={
            "label": f"bin mean {bs.name} [{bs.attrs.get('units', '?')}]",
            "shrink": cbar_shrink, "aspect": 20 * cbar_shrink,
        },
    )
    shared_edges = dim not in bs["y_bin_lower"].dims and dim not in bs["x_bin_lower"].dims
    with_period = "time_period" in bs.coords and bs["time_period"].dims == (dim,)
    for ax, name_dict in zip(fg.axs.flat, fg.name_dicts.flat):
        if name_dict is None:
            continue
        label = name_dict[dim]
        if with_period:
            label = f"{label}\n{bs.sel(**name_dict).time_period.item()}"
        ax.set_title(label, fontsize=8)
        if hatch is not None:
            hatch_bins(ax, hatch.sel(**name_dict))
    # FacetGrid axes are shared, so ticks set on the first panel apply to every panel
    if shared_edges:
        set_edge_ticks(fg.axs.flat[0], bs, "{:.2g}")
    else:
        for d, set_ticks in (("x_bin", fg.axs.flat[0].set_xticks), ("y_bin", fg.axs.flat[0].set_yticks)):
            n = bs.sizes[d]
            set_ticks(np.arange(n + 1) - 0.5, [f"Q{q:.0f}" for q in np.linspace(0, 100, n + 1)])
    for ax in fg.axs.flat:
        ax.tick_params(axis="both", labelsize=7)
        ax.tick_params(axis="x", labelrotation=90)
    fg.set_axis_labels("ai $\\rightarrow$", "lai $\\rightarrow$")
    fg.fig.suptitle(title, y=1.02)
    return finish(fg.fig, fout)


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
        # Unweighted across `dim`: each member/source counts equally, whatever its sample count
        # (unlike `binning.pool_members`)
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


def plot_et_product_spread(spread: xr.Dataset, title: str = "", fout: str | Path | None = None, label: str = "et"):
    """
    One heatmap per (lai, pr, rn) product set of `binning.et_product_spread`; hatched bins lack an ET product.
    `label` names the binned variable in the colorbar label.
    """
    groups = spread.lai_pr_rn.values
    fig, axs = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 4.5), squeeze=False, layout="constrained")
    vmax = float(spread["et_std"].quantile(0.98))
    for ax, g in zip(axs.ravel(), groups):
        s = spread.sel(lai_pr_rn=g)
        plot_bin_field(
            s["et_std"], ax=ax, hatch=s["n_products"] < s["n_et_products"], cmap="viridis", vmin=0, vmax=vmax,
            extend="max", cbar_kwargs={"label": f"std of bin mean {label} [{spread.attrs.get('units', '?')}]"},
            title=f"{g}\n{int(s.n_et_products)} ET products",
        )
        ax.set_xlabel("ai $\\rightarrow$")
        ax.set_ylabel("lai $\\rightarrow$")
    fig.suptitle(title)
    return finish(fig, fout)
