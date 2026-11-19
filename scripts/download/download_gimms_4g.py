"""Download GIMMS LAI4g V1.2 (1982-2020, 1/12 deg, half-monthly) from Zenodo.
Dataset: Cao et al. (2023), Spatiotemporally consistent global dataset of the GIMMS Leaf Area Index
(GIMMS LAI4g) from 1982 to 2020, doi:10.5281/zenodo.8281930

The file list (name, URL, MD5) is read from JSON_REF, a saved copy of the Zenodo record
(https://zenodo.org/api/records/8281930), so the files downloaded are those of this exact version.
9 files, ~6.9 GB: the AVHRR-MODIS consolidated record (1982-2020) and the AVHRR-only record
(1982-2015), each as zip archives of about a decade, plus the README (PDF).

Local layout:  OUTPUT_ROOT/<file name>          (the flat layout of the Zenodo record)
               OUTPUT_ROOT/<zip name without .zip>/...   (the contents of each zip)
OUTPUT_ROOT is SCRATCH_ROOT/gimms-lai4g-v1.2 (config/paths.yml); move it to OBS_ROOT by hand once complete.
pooch writes each file to a temporary name and moves it into place only once its MD5 matches the record;
a file that exists is checked against its MD5 and skipped if it matches (downloaded again otherwise),
so re-running is safe and retries only the missing files. An interrupted file restarts from the beginning.
Each zip is extracted once downloaded, and again on a re-run if any of its members is missing. The zips
are kept, since pooch needs them to tell that a file is complete; delete them once the data are moved.
A file truncated by an interrupted extraction is not detected: delete that zip's folder and re-run.

No login needed. Run:  pixi run python scripts/download/download_gimms_4g.py
"""

import json
from pathlib import Path

import pooch

from etunc.config import SCRATCH_ROOT

# Saved Zenodo record, relative to this script so the script runs from any directory
JSON_REF = Path(__file__).parent / "references/gimms_lai4g_zenodo.json"
OUTPUT_ROOT = SCRATCH_ROOT / "gimms-lai4g-v1.2"


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
    print(f"GIMMS LAI4g {version}: {len(files)} files -> {OUTPUT_ROOT}")

    failed = []
    for name, (url, checksum) in files.items():
        # Extract each zip into a folder named after it (the README PDF is kept as is)
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
