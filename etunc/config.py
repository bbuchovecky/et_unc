"""
etunc.config
============
Settings, paths and run IDs, read from the files in config/ (each explained
in its header):

- config/pipeline.yml: the default settings of the pipeline, in three
  sections: `prep` (steps 1-11), `binning` (steps 12-14 and the analyses) and
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


def merge_settings(defaults: dict, changes: dict, where: str = "") -> dict:
    """
    `defaults` with the values of `changes`, section by section. A key of `changes` that is
    not in `defaults` raises a KeyError, so a misspelt setting never passes silently.
    """
    out = dict(defaults)
    for key, value in changes.items():
        name = f"{where}{key}"
        if key not in defaults:
            raise KeyError(f"unknown setting {name!r}; the settings are those of {PIPELINE_FILE}")
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"setting {name!r} must be a mapping, not {value!r}")
            out[key] = merge_settings(defaults[key], value, f"{name}.")
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


def validate_settings(settings: dict) -> dict:
    """
    The settings with every value checked and given one type (e.g. 1 -> 1.0 for a number), so
    that equal settings always hash to the same ID. Raises ValueError for an invalid value.
    """
    prep, binning, plotting = settings["prep"], settings["binning"], settings["plotting"]

    def string(v, name):
        if not isinstance(v, str):
            raise ValueError(f"setting {name!r} must be a string, not {v!r}")
        return v

    if binning["bin_variable"] not in ("et", "et_pr"):
        raise ValueError(f"setting 'binning.bin_variable' must be 'et' or 'et_pr', not {binning['bin_variable']!r}")
    if not isinstance(binning["area_weighted"], bool):
        raise ValueError(f"setting 'binning.area_weighted' must be true or false, not {binning['area_weighted']!r}")
    families = binning["families"]
    if not isinstance(families, list) or not families or not all(isinstance(f, str) for f in families):
        raise ValueError(f"setting 'binning.families' must be a list of family names, not {families!r}")
    out = {
        "prep": {
            "time_slice": _pair(prep["time_slice"], "prep.time_slice", string),
            "target_res": _number(prep["target_res"], "prep.target_res"),
            "min_years": _int(prep["min_years"], "prep.min_years"),
            "min_annual_et": _number(prep["min_annual_et"], "prep.min_annual_et"),
            "lat_bnds": _pair(prep["lat_bnds"], "prep.lat_bnds", _number),
            "na_thresh": _number(prep["na_thresh"], "prep.na_thresh"),
            "lf_thresh": _number(prep["lf_thresh"], "prep.lf_thresh"),
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
