"""Download HiQ-LAI Version 1 (2000-2022, 5 km, 8-day) from Zenodo.
Dataset: Yan et al. (2023), A High-Quality Reprocessed MODIS Leaf Area Index Dataset (HiQ-LAI),
doi:10.5281/zenodo.8296768

The file list (name, URL, MD5) is read from JSON_REF, a saved copy of the Zenodo record
(https://zenodo.org/api/records/8296768), so the files downloaded are those of this exact version.
5 zip archives of 2-6 years each, ~38 GB (up to 9.5 GB per file).

Local layout:  OUTPUT_ROOT/<file name>          (the flat layout of the Zenodo record)
               OUTPUT_ROOT/<zip name without .zip>/...   (the contents of each zip)
OUTPUT_ROOT is SCRATCH_ROOT/hiq-lai-v1 (config/paths.yml); move it to OBS_ROOT by hand once complete.
pooch writes each file to a temporary name and moves it into place only once its MD5 matches the record;
a file that exists is checked against its MD5 and skipped if it matches (downloaded again otherwise),
so re-running is safe and retries only the missing files. An interrupted file restarts from the beginning.
Each zip is extracted once downloaded, and again on a re-run if any of its members is missing. The zips
are kept, since pooch needs them to tell that a file is complete; delete them once the data are moved.
A file truncated by an interrupted extraction is not detected: delete that zip's folder and re-run.

No login needed. Run:  pixi run python scripts/download/download_hiq_lai.py
"""

import json
from pathlib import Path

import pooch

from etunc.config import SCRATCH_ROOT

# Saved Zenodo record, relative to this script so the script runs from any directory
JSON_REF = Path(__file__).parent / "references/hiq_lai_zenodo.json"
OUTPUT_ROOT = SCRATCH_ROOT / "hiq-lai-v1"


def read_record(json_ref):
    """The record's version and its files as {name: (url, checksum)}, checksum as "md5:<hex>"."""
    record = json.loads(Path(json_ref).read_text())
    files = {
        name: (entry["links"]["content"], entry["checksum"])
        for name, entry in record["files"]["entries"].items()
    }
    return record["metadata"]["version"], files


def main():
    version, files = read_record(JSON_REF)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"HiQ-LAI {version}: {len(files)} files -> {OUTPUT_ROOT}")

    failed = []
    for name, (url, checksum) in files.items():
        # Extract each zip into a folder named after it
        unzip = pooch.Unzip(extract_dir=Path(name).stem) if name.endswith(".zip") else None
        try:
            # Downloads unless the file exists with the right checksum, which pooch verifies
            pooch.retrieve(url, known_hash=checksum, fname=name, path=OUTPUT_ROOT,
                           processor=unzip, progressbar=True)
        except Exception as e:
            failed.append(name)
            print(f"Failed: {name}: {e}")

    print(f"done: {len(files) - len(failed)}, failed: {len(failed)}")
    if failed:
        print("Re-run to retry the failed files.")


if __name__ == "__main__":
    main()
