"""
Tests for etunc/edges.py: hierarchical weights, weighted quantile edges and
the saved edges. Run from the project root with:

    python -m pytest tests/test_edges.py

Synthetic data only; takes a few seconds.
"""

import numpy as np
import pytest
import xarray as xr

import etunc.edges as bin_edges


def family(values_per_member, sources):
    """(member, lat, lon=1) field with one row of values per member and a `source` coord."""
    n = max(len(v) for v in values_per_member)
    data = np.full((len(values_per_member), n, 1), np.nan)
    for i, v in enumerate(values_per_member):
        data[i, : len(v), 0] = v
    return xr.DataArray(data, dims=("member", "lat", "lon"), coords={"source": ("member", sources)})


def test_hierarchical_weights_sum_per_level():
    """
    Each family weighs 1/N_families whatever its size, each source an equal share of its family, each member
    an equal share of its source, and each finite value an equal share of its member; NaN values are dropped.
    """
    fields = {
        "A": family([[1.0, 2.0, 3.0, 4.0]], ["a1"]),
        "B": family([[5.0, 6.0], [7.0, np.nan, 8.0], [9.0]], ["m1", "m1", "m2"]),
    }
    values, weights = bin_edges.hierarchical_weights(fields)
    np.testing.assert_array_equal(values, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    np.testing.assert_allclose(weights.sum(), 1.0)
    np.testing.assert_allclose(weights[:4], 0.5 / 4)            # family A: one source, one member
    np.testing.assert_allclose(weights[4:6], 0.5 / 2 / 2 / 2)   # B, source m1 (2 members), 2 values
    np.testing.assert_allclose(weights[6:8], 0.5 / 2 / 2 / 2)   # B, source m1, other member (NaN dropped)
    np.testing.assert_allclose(weights[8], 0.5 / 2)             # B, source m2: one member, one value


def test_pooled_bin_edges_equal_weights_is_inverted_cdf():
    """With equal weights the edges are numpy's inverted_cdf quantiles: data values, from the min to the max."""
    values = np.random.default_rng(0).gamma(2.0, 1.0, 101)
    edges = bin_edges.pooled_bin_edges(values, np.ones(values.size), 4, name="lai", attrs={"units": "m2/m2"})
    np.testing.assert_allclose(edges, np.quantile(values, np.linspace(0, 1, 5), method="inverted_cdf"))
    assert edges[0] == values.min() and edges[-1] == values.max()
    assert np.isin(edges.values, values).all()
    assert edges.dims == ("edge",) and edges.name == "lai"
    assert edges.attrs["strategy"] == "quantile" and edges.attrs["n_bins"] == 4
    assert edges.attrs["units"] == "m2/m2"


def test_pooled_bin_edges_weights_balance_families():
    """
    A small family weighs as much as a large one: with hierarchical weights a 10-value family of 1s holds half
    the weight against 1000 values of 2s, so the 20% and 40% edges are 1; without weights every edge above the
    minimum is 2. (5 bins, so no edge falls exactly on the 50% boundary.)
    """
    fields = {"small": family([np.ones(10)], ["s"]), "large": family([np.full(1000, 2.0)], ["l"])}
    values, weights = bin_edges.hierarchical_weights(fields)
    weighted = bin_edges.pooled_bin_edges(values, weights, 5, name="ai")
    unweighted = bin_edges.pooled_bin_edges(values, np.ones(values.size), 5, name="ai")
    np.testing.assert_array_equal(weighted, [1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(unweighted, [1, 2, 2, 2, 2, 2])


def test_pooled_bin_edges_keeps_duplicate_edges_and_warns_on_zeros():
    """Many zeros give duplicate edges at 0 (kept, not collapsed) and a warning."""
    values = np.concatenate([np.zeros(60), np.linspace(1, 2, 40)])
    with pytest.warns(UserWarning, match="equal to 0"):
        edges = bin_edges.pooled_bin_edges(values, np.ones(values.size), 4, name="lai")
    assert len(edges) == 5 and edges[0] == edges[1] == edges[2] == 0


def test_saved_edges_roundtrip(tmp_path, monkeypatch):
    """
    Edges written to edges_path (in the folder of the period and both bin counts, named by the variable's own
    count) load back unchanged; missing edges raise FileNotFoundError.
    """
    monkeypatch.setattr(bin_edges, "BIN_EDGES_ROOT", tmp_path)
    edges = bin_edges.pooled_bin_edges(np.arange(10.0), np.ones(10), 3, name="lai")
    path = bin_edges.edges_path("pooled_all", "lai", 3, 4, "1deg", "199501-201412")
    folder = tmp_path / "199501-201412" / "lai3_ai4"
    assert path == folder / "pooled_all.lai_clim.3_quantiles.1deg.199501-201412.nc"
    assert bin_edges.edges_path("pooled_all", "ai", 3, 4, "1deg", "199501-201412") == \
        folder / "pooled_all.ai_clim.4_quantiles.1deg.199501-201412.nc"
    path.parent.mkdir(parents=True)
    edges.to_netcdf(path)
    xr.testing.assert_identical(bin_edges.load_bin_edges("pooled_all", "lai", 3, 4, "1deg", "199501-201412"), edges)
    with pytest.raises(FileNotFoundError, match="make_edges.py"):
        bin_edges.load_bin_edges("pooled_cmip", "lai", 3, 4, "1deg", "199501-201412")
