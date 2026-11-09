"""
etunc.config
============
Settings, paths and run IDs, read from the files in config/ (each explained
in its header):

- config/pipeline.yml: the default settings of the pipeline, in three
  sections: `prep` (steps 1-11, including each family's products, models and
  members in `prep.<family>`), `binning` (steps 12-14 and the analyses) and
  `plotting`.
- config/runs/<name>.yml: one run's changes to the defaults (only the keys
  that differ), selected with the environment variable ETUNC_RUN; without it,
  the defaults.
- config/paths.yml (not in git; a copy of config/paths.example.yml), or the
  file named by ETUNC_PATHS: every input and output path.
- config/runs.csv: every run ID, its run file, git commit and settings,
  added by `register_run` when a script runs.

Every setting is exposed as a module constant (prep.time_slice -> TIME_SLICE,
binning.n_xbins -> N_XBINS, ...), and every path too (OBS_ROOT, PROC_ROOT, ...).

Run IDs are hashes of the settings, so the same settings always give the same ID:
- PREP_ID = "p" + hash of `prep`: the folder and file tag of the prep outputs
  (`inputs_path`), shared by every run with the same prep settings;
- RUN_ID = "r" + hash of `prep` and `binning`: the folder and file tag of
  everything from make_edges.py on (`area_mask_path`, edges, bins, analyses).
`plotting` and the paths are not hashed. Outputs are in
`output_dir(root, family, kind, id)` = <root>/<id>/<family>/<kind>/, and every
file name ends in .<id>.<ext>.

`summarize` prints the run, paths and settings to the log, and
`summarize_settings` a script's own settings.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import cartopy.crs as ccrs
import numpy as np
import yaml


PROJ_ROOT = Path(__file__).resolve().parents[1]  # the repo
CONFIG_ROOT = PROJ_ROOT / "config"
PIPELINE_FILE = CONFIG_ROOT / "pipeline.yml"
RUNS_CSV = CONFIG_ROOT / "runs.csv"


# ------------------------------------------------------------------
# Settings: config/pipeline.yml, changed by a run file (ETUNC_RUN)
# ------------------------------------------------------------------

def read_yaml(path: str | Path) -> dict:
    """The mapping in a YAML file ({} if the file is empty)."""
    with open(path) as f:
        out = yaml.safe_load(f)
    if out is None:
        return {}
    if not isinstance(out, dict):
        raise ValueError(f"{path}: expected a mapping of settings, not {type(out).__name__}")
    return out


def merge_settings(defaults: dict, changes: dict, where: str = "", depth: int = 0) -> dict:
    """
    `defaults` with the values of `changes`, key by key in a section (e.g. `prep`) and in a
    mapping of a section (e.g. `prep.cmip`). A key of `changes` that is not in `defaults` raises
    a KeyError, so a misspelt setting never passes silently. Deeper mappings (e.g.
    `prep.cmip.member_ids`) are values, replaced whole, so their keys are free.
    """
    out = dict(defaults)
    for key, value in changes.items():
        name = f"{where}{key}"
        if key not in defaults:
            raise KeyError(f"unknown setting {name!r}; the settings are those of {PIPELINE_FILE}")
        if isinstance(defaults[key], dict) and depth < 2:
            if not isinstance(value, dict):
                raise ValueError(f"setting {name!r} must be a mapping, not {value!r}")
            out[key] = merge_settings(defaults[key], value, f"{name}.", depth + 1)
        else:
            out[key] = value
    return out


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"setting {name!r} must be a number, not {value!r}")
    return float(value)


def _int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"setting {name!r} must be an integer, not {value!r}")
    return value


def _pair(value, name: str, kind) -> list:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"setting {name!r} must be a list of two values, not {value!r}")
    return [kind(v, name) for v in value]


def _string(value, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"setting {name!r} must be a string, not {value!r}")
    return value


def _list(value, name: str, kind, allow_none: bool = False) -> list | None:
    """A list of values of `kind` (or None, if `allow_none`)."""
    if value is None and allow_none:
        return None
    if not isinstance(value, list):
        raise ValueError(f"setting {name!r} must be a list{' or null' if allow_none else ''}, not {value!r}")
    return [kind(v, name) for v in value]


def _mapping(value, name: str, keys=None) -> dict:
    """A mapping with string keys (exactly `keys`, if given)."""
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError(f"setting {name!r} must be a mapping with string keys, not {value!r}")
    if keys is not None and set(value) != set(keys):
        raise ValueError(f"setting {name!r} must have exactly the keys {list(keys)}, not {list(value)}")
    return value


def _cmip_members(value, name: str) -> str | list[str]:
    """CMIP members of a model: "top", "max_r", "all" or a list of member_ids."""
    if isinstance(value, str):
        if value not in CMIP_MEMBER_CHOICES:
            raise ValueError(f"setting {name!r} must be one of {CMIP_MEMBER_CHOICES} or a list, not {value!r}")
        return value
    return _list(value, name, _string)


def _ppe_member(value, name: str) -> int:
    """A member of the CESM2 PPEs (0-N_PPE_MEMBERS)."""
    value = _int(value, name)
    if not 0 <= value <= N_PPE_MEMBERS:
        raise ValueError(f"setting {name!r}: PPE members are 0-{N_PPE_MEMBERS}, not {value}")
    return value


ILAMB_VARIABLES = ("et", "lai", "pr", "rn")
GRIDDED_PRODUCT_KEYS = ("dataset", "version", "variable", "freq")  # lo.load_obs arguments
CMIP_MEMBER_CHOICES = ("top", "max_r", "all")  # cmip.select_members
N_PPE_MEMBERS = 28
TRENDY_EXPERIMENTS_ALL = ("S2", "S3")  # as etunc.load.trendy.EXPERIMENTS (which imports this module)


def validate_products(prep: dict) -> dict:
    """The product, model and member settings of each family (prep.obs, prep.era5, ...), checked."""
    obs, cmip, trendy = prep["obs"], prep["cmip"], prep["trendy"]
    ilamb = _mapping(obs["ilamb_products"], "prep.obs.ilamb_products", ILAMB_VARIABLES)
    gridded = _mapping(obs["gridded_et_products"], "prep.obs.gridded_et_products")
    member_ids = _mapping(cmip["member_ids"], "prep.cmip.member_ids")
    omit_models = _mapping(trendy["omit_models"], "prep.trendy.omit_models")
    experiments = _list(trendy["experiments"], "prep.trendy.experiments", _string)
    if not experiments or not set(experiments) <= set(TRENDY_EXPERIMENTS_ALL):
        raise ValueError(f"setting 'prep.trendy.experiments' must be a non-empty list of {TRENDY_EXPERIMENTS_ALL}, "
                         f"not {experiments!r}")
    gridded_out = {}
    for label, g in gridded.items():
        name = f"prep.obs.gridded_et_products.{label}"
        g = _mapping(g, name, GRIDDED_PRODUCT_KEYS)
        gridded_out[label] = {k: _string(g[k], f"{name}.{k}") for k in GRIDDED_PRODUCT_KEYS}
    out = {
        "obs": {
            "ilamb_products": {v: _list(ilamb[v], f"prep.obs.ilamb_products.{v}", _string, allow_none=True)
                               for v in ILAMB_VARIABLES},
            "gridded_et_products": gridded_out,
        },
        "era5": {"lai_product": _string(prep["era5"]["lai_product"], "prep.era5.lai_product")},
        "cmip": {
            "source_ids": _list(cmip["source_ids"], "prep.cmip.source_ids", _string, allow_none=True),
            "omit_source_ids": _list(cmip["omit_source_ids"], "prep.cmip.omit_source_ids", _string),
            "default_members": _cmip_members(cmip["default_members"], "prep.cmip.default_members"),
            "member_ids": {sid: _cmip_members(m, f"prep.cmip.member_ids.{sid}") for sid, m in member_ids.items()},
        },
        "lens": {"members": _list(prep["lens"]["members"], "prep.lens.members", _string, allow_none=True)},
        "trendy": {
            "experiments": experiments,
            "models": _list(trendy["models"], "prep.trendy.models", _string, allow_none=True),
            "omit_models": {m: _string(why, f"prep.trendy.omit_models.{m}") for m, why in omit_models.items()},
        },
    }
    for ppe in ("fppe", "ippe"):
        out[ppe] = {
            "members": _list(prep[ppe]["members"], f"prep.{ppe}.members", _ppe_member, allow_none=True),
            "drop_outliers": _list(prep[ppe]["drop_outliers"], f"prep.{ppe}.drop_outliers", _ppe_member),
        }
    return out


def validate_settings(settings: dict) -> dict:
    """
    The settings with every value checked and given one type (e.g. 1 -> 1.0 for a number), so
    that equal settings always hash to the same ID. Raises ValueError for an invalid value.
    """
    prep, binning, plotting = settings["prep"], settings["binning"], settings["plotting"]

    if binning["bin_variable"] not in ("et", "et_pr"):
        raise ValueError(f"setting 'binning.bin_variable' must be 'et' or 'et_pr', not {binning['bin_variable']!r}")
    if not isinstance(binning["area_weighted"], bool):
        raise ValueError(f"setting 'binning.area_weighted' must be true or false, not {binning['area_weighted']!r}")
    families = binning["families"]
    if not isinstance(families, list) or not families or not all(isinstance(f, str) for f in families):
        raise ValueError(f"setting 'binning.families' must be a list of family names, not {families!r}")
    out = {
        "prep": {
            "time_slice": _pair(prep["time_slice"], "prep.time_slice", _string),
            "target_res": _number(prep["target_res"], "prep.target_res"),
            "min_years": _int(prep["min_years"], "prep.min_years"),
            "min_annual_et": _number(prep["min_annual_et"], "prep.min_annual_et"),
            "lat_bnds": _pair(prep["lat_bnds"], "prep.lat_bnds", _number),
            "na_thresh": _number(prep["na_thresh"], "prep.na_thresh"),
            "lf_thresh": _number(prep["lf_thresh"], "prep.lf_thresh"),
            **validate_products(prep),
        },
        "binning": {
            "families": list(families),
            "bin_variable": binning["bin_variable"],
            "area_weighted": binning["area_weighted"],
            "n_xbins": _int(binning["n_xbins"], "binning.n_xbins"),
            "n_ybins": _int(binning["n_ybins"], "binning.n_ybins"),
            "signif_alpha": _number(binning["signif_alpha"], "binning.signif_alpha"),
            "signif_n_min": _int(binning["signif_n_min"], "binning.signif_n_min"),
        },
        "plotting": {
            "dpi": _int(plotting["dpi"], "plotting.dpi"),
            "hist_bins": {v: [_number(a, f"plotting.hist_bins.{v}") for a in args]
                          for v, args in plotting["hist_bins"].items()},
        },
    }
    if out["binning"]["n_xbins"] < 1 or out["binning"]["n_ybins"] < 1:
        raise ValueError("settings 'binning.n_xbins' and 'binning.n_ybins' must be at least 1")
    # Every TRENDY family binned must be prepared: trendy_<experiment> with its experiment in prep.trendy
    prepared = {f"trendy_{e.lower()}" for e in out["prep"]["trendy"]["experiments"]}
    if missing := [f for f in families if f.startswith("trendy_") and f not in prepared]:
        raise ValueError(f"setting 'binning.families' has {missing}, whose experiments are not in "
                         f"'prep.trendy.experiments' {out['prep']['trendy']['experiments']}")
    return out


def load_settings(run_file: str | Path | None = None, pipeline_file: str | Path = PIPELINE_FILE) -> dict:
    """The default settings of `pipeline_file`, changed by those of `run_file` (if given), validated."""
    settings = read_yaml(pipeline_file)
    if run_file is not None:
        settings = merge_settings(settings, read_yaml(run_file))
    return validate_settings(settings)


def settings_id(settings: dict, sections: tuple[str, ...], prefix: str) -> str:
    """`prefix` + the first 7 hex digits of the SHA-256 of the `sections` of `settings` (as sorted JSON)."""
    text = json.dumps({s: settings[s] for s in sections}, sort_keys=True, separators=(",", ":"))
    return prefix + hashlib.sha256(text.encode()).hexdigest()[:7]


def _run_file() -> Path | None:
    """The run file named by ETUNC_RUN (relative paths from the repo), or None for the defaults."""
    value = os.environ.get("ETUNC_RUN")
    if not value:
        return None
    path = Path(value)
    path = path if path.is_absolute() or path.exists() else PROJ_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"ETUNC_RUN={value}: no such run file")
    return path.resolve()


RUN_FILE = _run_file()
SETTINGS = load_settings(RUN_FILE)
PREP_ID = settings_id(SETTINGS, ("prep",), "p")
RUN_ID = settings_id(SETTINGS, ("prep", "binning"), "r")

# Steps 1-11 (docs/binning-workflow.md)
TIME_SLICE = slice(*SETTINGS["prep"]["time_slice"])
TARGET_RES = SETTINGS["prep"]["target_res"]
MIN_YEARS = SETTINGS["prep"]["min_years"]
MIN_ANNUAL_ET = SETTINGS["prep"]["min_annual_et"]
LAT_BNDS = slice(*SETTINGS["prep"]["lat_bnds"])
NA_THRESH = SETTINGS["prep"]["na_thresh"]
LF_THRESH = SETTINGS["prep"]["lf_thresh"]

# Products, models and members of each family (prep.<family>), read by prep_<family>.py
OBS_ILAMB_PRODUCTS = SETTINGS["prep"]["obs"]["ilamb_products"]            # {variable: products or None (all)}
OBS_GRIDDED_ET_PRODUCTS = SETTINGS["prep"]["obs"]["gridded_et_products"]  # {label: lo.load_obs arguments}
ERA5_LAI_PRODUCT = SETTINGS["prep"]["era5"]["lai_product"]
CMIP_SOURCE_IDS = SETTINGS["prep"]["cmip"]["source_ids"]
CMIP_OMIT_SOURCE_IDS = SETTINGS["prep"]["cmip"]["omit_source_ids"]
CMIP_DEFAULT_MEMBERS = SETTINGS["prep"]["cmip"]["default_members"]
CMIP_MEMBER_IDS = SETTINGS["prep"]["cmip"]["member_ids"]
LENS_MEMBERS = SETTINGS["prep"]["lens"]["members"]
FPPE_MEMBERS = SETTINGS["prep"]["fppe"]["members"]
FPPE_DROP_OUTLIERS = SETTINGS["prep"]["fppe"]["drop_outliers"]
IPPE_MEMBERS = SETTINGS["prep"]["ippe"]["members"]
IPPE_DROP_OUTLIERS = SETTINGS["prep"]["ippe"]["drop_outliers"]
TRENDY_EXPERIMENTS = SETTINGS["prep"]["trendy"]["experiments"]  # each the family trendy_<experiment>
TRENDY_MODELS = SETTINGS["prep"]["trendy"]["models"]
TRENDY_OMIT_MODELS = SETTINGS["prep"]["trendy"]["omit_models"]  # {model: reason}

# Steps 12-14 and the analyses
FAMILIES = tuple(SETTINGS["binning"]["families"])
BIN_VARIABLE = SETTINGS["binning"]["bin_variable"]
AREA_WEIGHTED = SETTINGS["binning"]["area_weighted"]
N_XBINS = SETTINGS["binning"]["n_xbins"]  # aridity index
N_YBINS = SETTINGS["binning"]["n_ybins"]  # LAI
SIGNIF_ALPHA = SETTINGS["binning"]["signif_alpha"]
SIGNIF_N_MIN = SETTINGS["binning"]["signif_n_min"]

# Figures
DPI = SETTINGS["plotting"]["dpi"]
HIST_BINS = {v: np.linspace(start, stop, int(num)) for v, (start, stop, num) in SETTINGS["plotting"]["hist_bins"].items()}
PROJECTION = ccrs.PlateCarree()

# Physical constants. One Lv (near 20 C) for every water flux, ET and precipitation alike, with no
# sublimation or fusion term. It cancels in ET/P but sets the W/m2 scale of ET and AI
LATENT_HEAT_VAPORIZATION = 2.45e6  # J/kg
LIQ_WATER_DENSITY = 1e3            # kg/m3
EARTH_RADIUS = 6.371e6             # m, as in ILAMB, so cell areas match ILAMB's


# ------------------------------------------------------------------
# Paths: config/paths.yml (or ETUNC_PATHS)
# ------------------------------------------------------------------

PATHS_FILE = Path(os.environ.get("ETUNC_PATHS") or CONFIG_ROOT / "paths.yml")
if not PATHS_FILE.exists():
    raise FileNotFoundError(
        f"No paths file {PATHS_FILE}: copy {CONFIG_ROOT / 'paths.example.yml'} to {CONFIG_ROOT / 'paths.yml'} "
        "and edit it for this machine (or set ETUNC_PATHS)")
PATHS = read_yaml(PATHS_FILE)

# Inputs
CAMPAIGN_ROOT = Path(PATHS["campaign_root"])
OBS_ROOT = Path(PATHS["obs_root"])
ILAMB_ROOT = Path(PATHS["ilamb_root"])
OBS_REGRID_ROOT = Path(PATHS["obs_regrid_root"])  # DEPRECATED: regridded files of the legacy scripts
PLUMBER_ROOT = Path(PATHS["plumber_root"])
CMIP_FETCH_ROOT = Path(PATHS["cmip_fetch_root"])
CMIP_CATALOG_ROOT = CMIP_FETCH_ROOT / "catalogs"
CMIP_CATALOG = CMIP_CATALOG_ROOT / "cmip6_bin.csv"  # evspsbl, lai, pr, radiative fluxes
CMIP_FX_CATALOG = CMIP_CATALOG_ROOT / "cmip6_fx_glade.csv"  # one sftlf file per model
TRENDY_ROOT = Path(PATHS["trendy_root"])
FHIST_PPE_ROOT = Path(PATHS["fhist_ppe_root"])
IHIST_PPE_ROOT = Path(PATHS["ihist_ppe_root"])
LENS2_ROOT = Path(PATHS["lens2_root"])
CESM_GRID_PATHS = {k: Path(v) for k, v in PATHS["cesm_grid_paths"].items()}
ERA5_GDEX_ROOTS = {k: tuple(Path(p) for p in v) if isinstance(v, list) else Path(v)
                   for k, v in PATHS["era5_gdex_roots"].items()}
ERA5_GRID_PATH = Path(PATHS["era5_grid_path"])

# Outputs
WORK_ROOT = Path(PATHS["work_root"])
PROC_ROOT = WORK_ROOT / "proc"
FIG_ROOT = WORK_ROOT / "fig"
LOG_ROOT = WORK_ROOT / "logs"


# ------------------------------------------------------------------
# Output layout: <root>/<id>/<family>/<kind>/<file>.<id>.<ext>, the same under PROC_ROOT and FIG_ROOT
# ------------------------------------------------------------------
# id: PREP_ID for the prep outputs (kind inputs, clim, coverage, radiation, mask of make_mask.py), RUN_ID
# for everything from make_edges.py on (kind mask, edges, qbin, regress, decomp). The settings of an id
# are in config/runs.csv, so file names carry no grid, period or bin counts

def output_dir(root: Path, family: str, kind: str, run_id: str) -> Path:
    """Folder of one kind of output of a run or prep ID: root/<id>/<family>/<kind>."""
    return Path(root) / run_id / family / kind


def inputs_path(family: str, prep_id: str | None = None) -> Path:
    """Binning inputs of one family, written by scripts/prep_<family>.py (default: this run's PREP_ID)."""
    prep_id = prep_id or PREP_ID
    return output_dir(PROC_ROOT, family, "inputs", prep_id) / f"{family}.inputs.{prep_id}.nc"


def area_mask_path(run_id: str | None = None) -> Path:
    """Common area mask of all families, written by scripts/make_edges.py (default: this RUN_ID)."""
    run_id = run_id or RUN_ID
    return output_dir(PROC_ROOT, "all", "mask", run_id) / f"all.area_mask.{run_id}.nc"


# ------------------------------------------------------------------
# Run register: config/runs.csv
# ------------------------------------------------------------------

RUNS_COLUMNS = (
    "run_id", "prep_id", "run_file", "created_utc", "git_commit", "git_branch",
    "time_slice", "target_res", "families", "bin_variable", "area_weighted", "n_ybins", "n_xbins",
    "settings_json",
)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(PROJ_ROOT), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _hashed_json(settings: dict) -> str:
    """The hashed sections of `settings` as sorted JSON (the RUN_ID content)."""
    return json.dumps({s: settings[s] for s in ("prep", "binning")}, sort_keys=True, separators=(",", ":"))


def read_runs(csv_path: str | Path = RUNS_CSV) -> list[dict]:
    """The rows of the run register ([] if it does not exist yet)."""
    if not Path(csv_path).exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def register_run(csv_path: str | Path = RUNS_CSV) -> bool:
    """
    Add this run (RUN_ID, PREP_ID, run file, time, git commit and branch, key settings and all
    hashed settings) to the run register if it is not there yet. Returns True if a row was added.
    Raises if RUN_ID is registered with other settings (a hash collision).
    """
    hashed = _hashed_json(SETTINGS)
    for row in read_runs(csv_path):
        if row["run_id"] == RUN_ID:
            if row["settings_json"] != hashed:
                raise ValueError(f"run ID {RUN_ID} is in {csv_path} with other settings; change a setting "
                                 "to get another ID")
            return False
    p, b = SETTINGS["prep"], SETTINGS["binning"]
    row = {
        "run_id": RUN_ID, "prep_id": PREP_ID,
        "run_file": str(RUN_FILE.relative_to(PROJ_ROOT)) if RUN_FILE and RUN_FILE.is_relative_to(PROJ_ROOT)
        else str(RUN_FILE or "default"),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git("rev-parse", "--short", "HEAD"), "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "time_slice": "-".join(p["time_slice"]), "target_res": p["target_res"], "families": " ".join(b["families"]),
        "bin_variable": b["bin_variable"], "area_weighted": b["area_weighted"],
        "n_ybins": b["n_ybins"], "n_xbins": b["n_xbins"], "settings_json": hashed,
    }
    new = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    print(f"registered run {RUN_ID} (prep {PREP_ID}) in {csv_path}")
    return True


# ------------------------------------------------------------------
# Logs
# ------------------------------------------------------------------

def summarize():
    """Print the run, the paths and the settings (the start of every script's log)."""
    print("#### config ####")
    print(f"\n{'RUN_ID':18} = {RUN_ID}")
    print(f"{'PREP_ID':18} = {PREP_ID}")
    print(f"{'RUN_FILE':18} = {RUN_FILE or 'none (defaults)'}")
    print(f"{'PIPELINE_FILE':18} = {PIPELINE_FILE}")
    print(f"{'PATHS_FILE':18} = {PATHS_FILE}")
    print("\nPaths:")
    print("------")
    for name in ("CAMPAIGN_ROOT", "OBS_ROOT", "ILAMB_ROOT", "OBS_REGRID_ROOT", "PLUMBER_ROOT", "CMIP_CATALOG",
                 "CMIP_FX_CATALOG", "TRENDY_ROOT", "FHIST_PPE_ROOT", "IHIST_PPE_ROOT", "LENS2_ROOT", "ERA5_GRID_PATH",
                 "WORK_ROOT", "PROC_ROOT", "FIG_ROOT"):
        print(f"{name:18} = {globals()[name]}")
    print("\nSettings:")
    print("---------")
    for section, values in SETTINGS.items():
        for key, value in values.items():
            print(f"{section + '.' + key:24} = {value}")
    print(f"\n{'LATENT_HEAT_VAPORIZATION':24} = {LATENT_HEAT_VAPORIZATION} J/kg")
    print(f"{'LIQ_WATER_DENSITY':24} = {LIQ_WATER_DENSITY} kg/m3")
    print(f"{'EARTH_RADIUS':24} = {EARTH_RADIUS} m")
    print("\n#### config ####\n")


def summarize_settings(namespace: dict):
    """
    Print the settings of a script: every UPPERCASE module-level name in
    `namespace` (call with `globals()`) that holds a plain value (number,
    string, path, slice, or a container of them), so the log of each run
    records them. Larger objects such as grids are skipped.
    """
    plain = (bool, int, float, str, Path, slice, tuple, list, dict, type(None))
    print("\n#### script settings ####\n")
    for name, value in namespace.items():
        if name.isupper() and isinstance(value, plain):
            print(f"{name:18} = {value!r}")
    print("\n#### script settings ####\n")
