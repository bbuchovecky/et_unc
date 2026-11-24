"""Download GLOBMAP LAI Version 3 (1981-2023, ~8 km, half-monthly to 2000, 8-day from 2001) from Zenodo.
Dataset: Liu, Liu and Chen (2024), GLOBMAP global Leaf Area Index since 1981, doi:10.5281/zenodo.12698637
(algorithm: Liu et al. (2012), J. Geophys. Res., 117, G04003, doi:10.1029/2012JG002084)

The file list (name, URL, MD5) is read from JSON_REF, a saved copy of the Zenodo record
(https://zenodo.org/api/records/12698637), so the files downloaded are those of this exact version.
The record holds each period twice, as HDF and as GeoTIFF; only the FORMAT copy is downloaded:
7 RAR archives of 3-10 years each, ~3.6 GB (~32 GB unpacked; one 0.0727 deg int16 GeoTIFF per
composite, LAI x 100, valid range 0-1000).

Local layout:  OUTPUT_ROOT/<file name>          (the flat layout of the Zenodo record)
               OUTPUT_ROOT/<rar name without .rar>/...   (the contents of each archive)
OUTPUT_ROOT is SCRATCH_ROOT/globmap-lai-v3 (config/paths.yml); move it to OBS_ROOT by hand once complete.
pooch writes each file to a temporary name and moves it into place only once its MD5 matches the record;
a file that exists is checked against its MD5 and skipped if it matches (downloaded again otherwise),
so re-running is safe and retries only the missing files. An interrupted file restarts from the beginning.
pooch cannot read RAR, so each archive is extracted with unar (/usr/bin/unar on glade, not in the pixi env)
once downloaded, and again on a re-run if its folder is missing. unar writes into a temporary folder
that is moved into place only once it succeeds, so an interrupted extraction leaves no partial folder.
The archives are kept, since pooch needs them to tell that a file is complete; delete them once the
data are moved.

No login needed. Run:  pixi run python scripts/download/download_globmap.py
"""

import json
import shutil
import subprocess
from pathlib import Path

import pooch

from etunc.config import SCRATCH_ROOT

# Saved Zenodo record, relative to this script so the script runs from any directory
JSON_REF = Path(__file__).parent / "references/globmap_zenodo.json"
OUTPUT_ROOT = SCRATCH_ROOT / "globmap-lai-v3"
# "TIFF" (read by rioxarray, like GIMMS LAI4g and HiQ-LAI) or "HDF": the same data in two formats
FORMAT = "TIFF"


def read_record(json_ref):
    """The record's version and its files as {name: (url, checksum)}, checksum as "md5:<hex>"."""
    record = json.loads(Path(json_ref).read_text())
    files = {
        name: (entry["links"]["content"], entry["checksum"])
        for name, entry in record["files"]["entries"].items()
    }
    return record["metadata"]["version"], files


def unrar(fname, action, pup):
    """pooch processor: extract a .rar archive into a folder named after it (<archive without .rar>/).

    Skipped when the archive was already there (action "fetch") and its folder exists. Each archive
    holds one top-level folder with that name (e.g. 2021-2023_TIFF/), which is moved into place.
    """
    archive = Path(fname)
    out = archive.with_suffix("")
    if action == "fetch" and out.is_dir():
        return str(out)
    part = archive.with_suffix(".part")
    shutil.rmtree(part, ignore_errors=True)  # left by an interrupted extraction
    shutil.rmtree(out, ignore_errors=True)   # extracted from the archive this download replaced
    # unar exits with an error if any member fails, which leaves `out` missing for the re-run
    subprocess.run(["unar", "-quiet", "-force-overwrite", "-no-directory",
                    "-output-directory", str(part), str(archive)], check=True)
    (part / out.name).rename(out)
    part.rmdir()
    return str(out)


def main():
    if shutil.which("unar") is None:
        raise SystemExit("unar not found: it extracts the RAR archives (/usr/bin/unar on glade)")
    version, files = read_record(JSON_REF)
    files = {name: v for name, v in files.items() if name.endswith(f"_{FORMAT}.rar")}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"GLOBMAP LAI {version} ({FORMAT}): {len(files)} files -> {OUTPUT_ROOT}")

    failed = []
    for name, (url, checksum) in files.items():
        try:
            # Downloads unless the file exists with the right checksum, which pooch verifies
            pooch.retrieve(url, known_hash=checksum, fname=name, path=OUTPUT_ROOT,
                           processor=unrar, progressbar=True)
        except Exception as e:
            failed.append(name)
            print(f"Failed: {name}: {e}")

    print(f"done: {len(files) - len(failed)}, failed: {len(failed)}")
    if failed:
        print("Re-run to retry the failed files.")


if __name__ == "__main__":
    main()
