"""
Tests for etunc/config.py: the settings of config/pipeline.yml and of a run file, their validation, the
run IDs, the run register (config/runs.csv) and the paths file. Run from the project root with:

    python -m pytest tests/test_config.py
"""

import copy
import csv
import importlib
import json

import pytest
import yaml

import etunc.config as cfg


def write_yaml(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def ids(settings):
    """(PREP_ID, RUN_ID) of `settings`."""
    return cfg.settings_id(settings, ("prep",), "p"), cfg.settings_id(settings, ("prep", "binning"), "r")


@pytest.fixture
def reload_config(monkeypatch):
    """Reload etunc.config under the test's environment, and as before afterwards."""
    yield lambda: importlib.reload(cfg)
    monkeypatch.undo()
    importlib.reload(cfg)


def test_defaults():
    """With no run file, the constants are the values of config/pipeline.yml."""
    d = cfg.read_yaml(cfg.PIPELINE_FILE)
    assert cfg.RUN_FILE is None
    assert cfg.TIME_SLICE == slice(*d["prep"]["time_slice"])
    assert cfg.LAT_BNDS == slice(*(float(v) for v in d["prep"]["lat_bnds"]))
    assert cfg.FAMILIES == tuple(d["binning"]["families"])
    assert (cfg.BIN_VARIABLE, cfg.AREA_WEIGHTED, cfg.N_XBINS, cfg.N_YBINS) == tuple(
        d["binning"][k] for k in ("bin_variable", "area_weighted", "n_xbins", "n_ybins"))
    assert len(cfg.HIST_BINS["lai"]) == d["plotting"]["hist_bins"]["lai"][2]
    # Products, models and members of each family
    assert cfg.OBS_ILAMB_PRODUCTS == d["prep"]["obs"]["ilamb_products"]
    assert cfg.OBS_GRIDDED_ET_PRODUCTS == d["prep"]["obs"]["gridded_et_products"]
    assert cfg.ERA5_LAI_PRODUCT == d["prep"]["era5"]["lai_product"]
    assert (cfg.CMIP_SOURCE_IDS, cfg.CMIP_OMIT_SOURCE_IDS, cfg.CMIP_DEFAULT_MEMBERS, cfg.CMIP_MEMBER_IDS) == tuple(
        d["prep"]["cmip"][k] for k in ("source_ids", "omit_source_ids", "default_members", "member_ids"))
    assert cfg.LENS_MEMBERS == d["prep"]["lens"]["members"]
    assert (cfg.FPPE_MEMBERS, cfg.FPPE_DROP_OUTLIERS) == (d["prep"]["fppe"]["members"], d["prep"]["fppe"]["drop_outliers"])
    assert (cfg.IPPE_MEMBERS, cfg.IPPE_DROP_OUTLIERS) == (d["prep"]["ippe"]["members"], d["prep"]["ippe"]["drop_outliers"])
    assert (cfg.TRENDY_EXPERIMENTS, cfg.TRENDY_MODELS, cfg.TRENDY_OMIT_MODELS) == tuple(
        d["prep"]["trendy"][k] for k in ("experiments", "models", "omit_models"))
    assert cfg.RUN_ID[0] == "r" and cfg.PREP_ID[0] == "p" and len(cfg.RUN_ID) == len(cfg.PREP_ID) == 8
    assert (cfg.PREP_ID, cfg.RUN_ID) == ids(cfg.SETTINGS)


def test_run_file_changes_only_its_keys(tmp_path):
    run = write_yaml(tmp_path, "run.yml", {"binning": {"n_xbins": 20, "area_weighted": True}})
    s, d = cfg.load_settings(run), cfg.load_settings()
    assert s["binning"]["n_xbins"] == 20 and s["binning"]["area_weighted"] is True
    assert s["binning"]["n_ybins"] == d["binning"]["n_ybins"]
    assert s["prep"] == d["prep"] and s["plotting"] == d["plotting"]


def test_run_file_family_settings(tmp_path):
    """
    A run file changes single keys of a family (prep.cmip.default_members), and replaces a mapping inside a
    family whole: member_ids takes new model names, omit_models keeps only the models it gives.
    """
    run = write_yaml(tmp_path, "run.yml", {"prep": {
        "cmip": {"default_members": "max_r", "member_ids": {"CESM2": ["r1i1p1f1", "r2i1p1f1"]}},
        "trendy": {"omit_models": {"CARDAMOM": "2003-2024 only"}},
    }})
    s, d = cfg.load_settings(run), cfg.load_settings()
    assert s["prep"]["cmip"]["default_members"] == "max_r"
    assert s["prep"]["cmip"]["member_ids"] == {"CESM2": ["r1i1p1f1", "r2i1p1f1"]}
    assert s["prep"]["cmip"]["omit_source_ids"] == d["prep"]["cmip"]["omit_source_ids"]
    assert s["prep"]["trendy"]["omit_models"] == {"CARDAMOM": "2003-2024 only"}
    assert s["prep"]["obs"] == d["prep"]["obs"]


@pytest.mark.parametrize("changes", [
    {"binning": {"n_bins": 20}},           # misspelt setting
    {"prepp": {"min_years": 3}},           # misspelt section
    {"binning": 3},                        # a section that is not a mapping
])
def test_unknown_settings_raise(tmp_path, changes):
    with pytest.raises((KeyError, ValueError)):
        cfg.load_settings(write_yaml(tmp_path, "run.yml", changes))


@pytest.mark.parametrize("changes", [
    {"binning": {"bin_variable": "pr"}},
    {"binning": {"area_weighted": 1}},
    {"binning": {"n_xbins": "ten"}},
    {"binning": {"n_ybins": 0}},
    {"prep": {"time_slice": ["2001-01"]}},
    {"prep": {"target_res": True}},
    {"binning": {"families": []}},
    {"prep": {"obs": {"ilamb_products": {"et": None, "lai": ["GIMMS_LAI4g"]}}}},  # pr and rn missing
    {"prep": {"obs": {"gridded_et_products": {"SiTHv2": {"dataset": "sith", "version": "v2"}}}}},
    {"prep": {"era5": {"lai_product": None}}},
    {"prep": {"cmip": {"default_members": "first"}}},
    {"prep": {"cmip": {"member_ids": {"CESM2": 3}}}},
    {"prep": {"fppe": {"members": ["0", "1"]}}},
    {"prep": {"ippe": {"drop_outliers": [29]}}},
    {"prep": {"trendy": {"experiments": ["S4"]}}},
    {"prep": {"trendy": {"experiments": []}}},
    {"prep": {"trendy": {"experiments": ["S2"]}}},  # binning.families has trendy_s3, no longer prepared
])
def test_invalid_values_raise(tmp_path, changes):
    with pytest.raises(ValueError):
        cfg.load_settings(write_yaml(tmp_path, "run.yml", changes))


def test_ids_follow_the_hashed_settings(tmp_path):
    """
    A binning setting changes RUN_ID only, a prep setting (a family's products too) both, a plotting setting
    neither; equal values
    of another type (1 and 1.0) give the same IDs.
    """
    d = cfg.load_settings()
    assert ids(d) == ids(cfg.load_settings())
    binning = copy.deepcopy(d)
    binning["binning"]["n_xbins"] += 5
    prep = copy.deepcopy(d)
    prep["prep"]["min_years"] += 1
    products = copy.deepcopy(d)
    products["prep"]["cmip"]["default_members"] = "all"
    plotting = copy.deepcopy(d)
    plotting["plotting"]["dpi"] += 100
    assert ids(binning)[0] == ids(d)[0] and ids(binning)[1] != ids(d)[1]
    assert ids(prep)[0] != ids(d)[0] and ids(prep)[1] != ids(d)[1]
    assert ids(products)[0] != ids(d)[0] and ids(products)[1] != ids(d)[1]
    assert ids(plotting) == ids(d)
    as_int = write_yaml(tmp_path, "int.yml", {"prep": {"target_res": int(d["prep"]["target_res"])}})
    assert ids(cfg.load_settings(as_int)) == ids(d)


def test_shipped_run_files():
    """Every file of config/runs loads, has its own RUN_ID, and shares the default PREP_ID (binning only)."""
    run_files = sorted((cfg.CONFIG_ROOT / "runs").glob("*.yml"))
    assert run_files
    run_ids = {ids(cfg.load_settings(f))[1] for f in run_files} | {cfg.RUN_ID}
    assert len(run_ids) == len(run_files) + 1
    assert {ids(cfg.load_settings(f))[0] for f in run_files} == {cfg.PREP_ID}


def test_register_run(tmp_path):
    """A run is added once, with its IDs and settings; the same ID with other settings raises."""
    csv_path = tmp_path / "runs.csv"
    assert cfg.register_run(csv_path) is True
    assert cfg.register_run(csv_path) is False
    rows = cfg.read_runs(csv_path)
    assert len(rows) == 1
    row = rows[0]
    assert (row["run_id"], row["prep_id"], row["run_file"]) == (cfg.RUN_ID, cfg.PREP_ID, "default")
    assert json.loads(row["settings_json"]) == {s: cfg.SETTINGS[s] for s in ("prep", "binning")}
    assert row["n_xbins"] == str(cfg.N_XBINS) and row["git_commit"]
    # The same run ID with other settings
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cfg.RUNS_COLUMNS)
        writer.writeheader()
        writer.writerow({**row, "settings_json": "{}"})
    with pytest.raises(ValueError, match="other settings"):
        cfg.register_run(csv_path)


def test_run_and_paths_from_environment(reload_config, monkeypatch, tmp_path):
    """ETUNC_RUN selects the run file and ETUNC_PATHS the paths file, from which the output roots derive."""
    run = write_yaml(tmp_path, "aw.yml", {"binning": {"area_weighted": True}})
    paths = cfg.read_yaml(cfg.CONFIG_ROOT / "paths.example.yml")
    paths["work_root"] = str(tmp_path / "work")
    default_ids = (cfg.PREP_ID, cfg.RUN_ID)
    monkeypatch.setenv("ETUNC_RUN", str(run))
    monkeypatch.setenv("ETUNC_PATHS", str(write_yaml(tmp_path, "paths.yml", paths)))
    c = reload_config()
    assert c.AREA_WEIGHTED is True and c.RUN_FILE == run.resolve()
    assert c.PREP_ID == default_ids[0] and c.RUN_ID != default_ids[1]
    assert c.PROC_ROOT == tmp_path / "work" / "proc" and c.FIG_ROOT == tmp_path / "work" / "fig"


def test_missing_files_raise(reload_config, monkeypatch, tmp_path):
    monkeypatch.setenv("ETUNC_PATHS", str(tmp_path / "none.yml"))
    with pytest.raises(FileNotFoundError, match="paths.example.yml"):
        reload_config()
    monkeypatch.setenv("ETUNC_PATHS", str(cfg.CONFIG_ROOT / "paths.example.yml"))
    monkeypatch.setenv("ETUNC_RUN", str(tmp_path / "none.yml"))
    with pytest.raises(FileNotFoundError, match="no such run file"):
        reload_config()
