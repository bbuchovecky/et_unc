"""
Tests for etunc/variants.py: the groups of variants of each family and the
nested group means. Run from the project root with:

    python -m pytest tests/test_variants.py

Synthetic data only; takes a few seconds.
"""

import numpy as np
import xarray as xr

import etunc.variants as variants


def obs_sources(et_products, lai_products=None):
    """(source, x) values 0, 1, ... with the obs `<v>_product` coords along `source`."""
    n = len(et_products)
    lai_products = lai_products or ["L"] * n
    return xr.DataArray(
        np.arange(n, dtype=float).reshape(n, 1), dims=("source", "x"),
        coords={
            "source": [f"{e}-{l}-P-R" for e, l in zip(et_products, lai_products)],
            "et_product": ("source", et_products), "lai_product": ("source", lai_products),
            "pr_product": ("source", ["P"] * n), "rn_product": ("source", ["R"] * n),
        },
    )


def test_group_model_variants():
    """Variants of one model share a base model: source_id up to the first "-", without version digits, then the aliases."""
    sids = [
        "CESM2-WACCM", "CESM2", "MIROC6", "MIROC-ES2L", "MPI-ESM1-2-LR", "MPI-ESM-1-2-HAM",
        "UKESM1-0-LL", "HadGEM3-GC31-LL", "NorCPM1", "NorESM2-LM", "E3SM-1-0", "EC-Earth3-Veg", "CESM2",
    ]
    assert variants.group_model_variants(sids) == {
        "CESM": ["CESM2", "CESM2-WACCM"],
        "E3SM": ["E3SM-1-0"],
        "EC-Earth": ["EC-Earth3-Veg"],
        "HadGEM": ["HadGEM3-GC31-LL", "UKESM1-0-LL"],
        "MIROC": ["MIROC-ES2L", "MIROC6"],
        "MPI": ["MPI-ESM-1-2-HAM", "MPI-ESM1-2-LR"],
        "NorESM": ["NorCPM1", "NorESM2-LM"],
    }


def test_source_groups_by_family():
    """
    cmip: the base model of each source; obs: the combination with the ET product's group (GLEAMv4.3a/b and
    PMLv2.2a_MODIS/c merged, GLEAMv3.3a apart); any other family: each source.
    """
    cmip = xr.DataArray(np.zeros(3), dims="member", coords={"source": ("member", ["CESM2", "CESM2-WACCM", "MIROC6"])})
    np.testing.assert_array_equal(variants.source_groups("cmip", cmip, "member"), ["CESM", "CESM", "MIROC"])
    np.testing.assert_array_equal(variants.source_groups("lens", cmip, "member"), ["CESM2", "CESM2-WACCM", "MIROC6"])

    obs = obs_sources(["GLEAMv4.3a", "GLEAMv4.3b", "GLEAMv3.3a", "PMLv2.2a_MODIS", "PMLv2.2c", "GLEAMv4.3a"],
                      ["L", "L", "L", "L", "L", "L2"])
    np.testing.assert_array_equal(variants.source_groups("obs", obs, "source"), [
        "GLEAMv4.3-L-P-R", "GLEAMv4.3-L-P-R", "GLEAMv3.3a-L-P-R", "PMLv2.2-L-P-R", "PMLv2.2-L-P-R",
        "GLEAMv4.3-L2-P-R",
    ])


def test_group_means_nested():
    """
    Members are averaged within their source, then sources within their group, NaN skipped at each level:
    CESM2 has members 1 and 3 (mean 2) and CESM2-WACCM 6, so CESM is 4, not the member mean 10/3. Where
    CESM2-WACCM is NaN, CESM is CESM2's mean alone.
    """
    da = xr.DataArray(
        [[1.0, 1.0], [3.0, 3.0], [6.0, np.nan], [10.0, 10.0]], dims=("member", "x"),
        coords={"source": ("member", ["CESM2", "CESM2", "CESM2-WACCM", "MIROC6"])},
    )
    g = variants.group_means(da, "cmip", "member")
    assert list(g["group"].values) == ["CESM", "MIROC"]
    np.testing.assert_allclose(g.sel(group="CESM"), [4.0, 2.0])
    np.testing.assert_allclose(g.sel(group="MIROC"), [10.0, 10.0])
    # Mean and std across groups weigh CESM once
    np.testing.assert_allclose(g.mean("group")[0], 7.0)
    np.testing.assert_allclose(g.std("group")[0], 3.0)


def test_group_means_obs_along_source():
    """Along `source`, the obs sources of one ET product group are averaged; other families keep each source."""
    obs = obs_sources(["GLEAMv4.3a", "GLEAMv4.3b", "SiTHv2"])  # values 0, 1, 2
    g = variants.group_means(obs, "obs", "source")
    assert list(g["group"].values) == ["GLEAMv4.3-L-P-R", "SiTHv2-L-P-R"]
    np.testing.assert_allclose(g[:, 0], [0.5, 2.0])
    same = variants.group_means(obs, "lens", "source")
    np.testing.assert_allclose(same[:, 0], [0.0, 1.0, 2.0])
