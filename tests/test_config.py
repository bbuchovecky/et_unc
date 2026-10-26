"""Environment overrides of the binning settings in etunc.config (ETUNC_*)."""

import importlib

import pytest

import etunc.config as cfg

OVERRIDES = ("ETUNC_BIN_VARIABLE", "ETUNC_AREA_WEIGHTED", "ETUNC_N_XBINS", "ETUNC_N_YBINS")


@pytest.fixture
def reload_config(monkeypatch):
    """Reload etunc.config under the test's environment, and with no overrides afterwards."""
    for name in OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    yield lambda: importlib.reload(cfg)
    monkeypatch.undo()
    importlib.reload(cfg)


def test_defaults(reload_config):
    c = reload_config()
    assert (c.BIN_VARIABLE, c.AREA_WEIGHTED, c.N_XBINS, c.N_YBINS) == ("et", False, 15, 15)


def test_overrides(reload_config, monkeypatch):
    monkeypatch.setenv("ETUNC_BIN_VARIABLE", "et_pr")
    monkeypatch.setenv("ETUNC_AREA_WEIGHTED", "1")
    monkeypatch.setenv("ETUNC_N_XBINS", "20")
    monkeypatch.setenv("ETUNC_N_YBINS", "10")
    c = reload_config()
    assert (c.BIN_VARIABLE, c.AREA_WEIGHTED, c.N_XBINS, c.N_YBINS) == ("et_pr", True, 20, 10)


@pytest.mark.parametrize("value, expected", [("1", True), ("true", True), ("True", True),
                                             ("0", False), ("false", False), ("FALSE", False)])
def test_area_weighted_values(reload_config, monkeypatch, value, expected):
    monkeypatch.setenv("ETUNC_AREA_WEIGHTED", value)
    assert reload_config().AREA_WEIGHTED is expected


@pytest.mark.parametrize("name, value", [("ETUNC_BIN_VARIABLE", "pr"), ("ETUNC_AREA_WEIGHTED", "yes"),
                                         ("ETUNC_N_XBINS", "ten")])
def test_invalid_values(reload_config, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        reload_config()
