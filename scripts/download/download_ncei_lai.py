"""Download the NOAA NCEI LAI climate data record (obs4MIPs, 0.05 deg monthly means) from ESGF.
    https://www.ncei.noaa.gov/products/climate-data-records/leaf-area-index-and-fapar

obs4MIPs publishes the record as one dataset per sensor (SENSORS):
  avhrr: obs4MIPs.NOAA-NCEI.NOAA-NCEI-LAI-AVHRR-5-0.mon.lai.gn  1981-06 to 2013-12, 33 files, 4.9 GiB
  viirs: obs4MIPs.NOAA-NCEI.NOAA-NCEI-LAI-VIIRS-1-0.mon.lai.gn  2014-01 to 2024-12, 11 files, 1.7 GiB
(an earlier combined publication, NOAA-NCEI-LAI-5-0 v20250731, stops in 2018 and is not used).
The file list (URL, size, SHA256) of the latest version comes from the ESGF search API of the ORNL ESGF 1.5
bridge (INDEX_NODE), since the European index nodes do not list these datasets. A query can take ~1 min.

Local layout:  OUTPUT_ROOT/<folder>/data/obs4MIPs/NOAA-NCEI/<source_id>/mon/lai/gn/v<version>/
                   lai_mon_<source_id>_BE_gn_<YYYYMM>-<YYYYMM>.nc   (one file per year)
with <folder> avhrr-lai-v5 or viirs-lai-v1: the ESGF directory layout, which etunc.load.obs ("avhrr",
"viirs", and both joined by load_ncei_lai) reads under OBS_ROOT. OUTPUT_ROOT is SCRATCH_ROOT
(config/paths.yml); move each <folder> to OBS_ROOT by hand once complete.

Each file goes to <name>.part, is resumed after an interruption, and is renamed once its size and SHA256
match the index. A file that exists is skipped if its size matches (its SHA256 was checked when it was
downloaded), so re-running is safe and retries only the missing files.

No login needed. Run:  pixi run python scripts/download/download_ncei_lai.py
"""

import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

from etunc.config import SCRATCH_ROOT

OUTPUT_ROOT = SCRATCH_ROOT
INDEX_NODE = "https://esgf-node.ornl.gov/esgf-1-5-bridge/"
# sensor: (obs4MIPs source_id, local folder)
SENSORS = {
    "avhrr": ("NOAA-NCEI-LAI-AVHRR-5-0", "avhrr-lai-v5"),
    "viirs": ("NOAA-NCEI-LAI-VIIRS-1-0", "viirs-lai-v1"),
}

PARALLEL = 2     # files downloaded at once
MAX_TRIES = 5    # attempts per file, each resuming the last
RETRY_WAIT = 30  # s between attempts
TIMEOUT = 300    # s for a search query, and without data before a download attempt fails


def log(msg):
    print(f"{datetime.now():%F %T}  {msg}", flush=True)


def list_files(source_id):
    """Files of the latest version of a sensor's dataset, as dicts with url, size, sha256 and path (the
    publish path under the data node's root, e.g. obs4MIPs/NOAA-NCEI/<source_id>/mon/lai/gn/v<version>/...)."""
    params = {
        "type": "File", "project": "obs4MIPs", "institution_id": "NOAA-NCEI", "source_id": source_id,
        "frequency": "mon", "variable_id": "lai", "grid_label": "gn", "latest": "true",
        "format": "application/solr+json", "limit": 1000,
    }
    r = requests.get(INDEX_NODE, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    files = {}
    for doc in r.json()["response"]["docs"]:
        if doc.get("retracted"):
            continue
        assert doc["checksum_type"] == ["SHA256"], doc["checksum_type"]
        # Each url is "<url>|<mime type>|<service>"; HTTPServer is the plain download
        url = next(u.split("|")[0] for u in doc["url"] if u.endswith("|HTTPServer"))
        path = doc["publish_path"][0].split("/", 1)[1]  # drop the data node's root ("user_pub_work")
        # The same file may be listed by several data nodes: keep one
        files[doc["instance_id"]] = {"url": url, "size": doc["size"], "sha256": doc["checksum"][0], "path": path}
    return sorted(files.values(), key=lambda f: f["path"])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def download(file, path):
    """Download file (a dict of list_files) to path through <path>.part, resuming it; raise after MAX_TRIES."""
    part = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_TRIES + 1):
        try:
            done = part.stat().st_size if part.exists() else 0
            if done >= file["size"] and sha256(part) != file["sha256"]:  # complete but wrong: start over
                part.unlink()
                done = 0
            if done < file["size"]:
                with requests.get(file["url"], headers={"Range": f"bytes={done}-"} if done else {},
                                  stream=True, timeout=TIMEOUT) as r:
                    r.raise_for_status()
                    if r.status_code == 200:  # the range was ignored: the whole file
                        done = 0
                    with open(part, "ab" if done else "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
            if part.stat().st_size != file["size"]:
                raise OSError(f"got {part.stat().st_size} of {file['size']} bytes")
            if sha256(part) != file["sha256"]:
                part.unlink()
                raise OSError("SHA256 does not match the index")
            part.rename(path)
            return
        except Exception as e:
            log(f"WARNING: {path.name} attempt {attempt}/{MAX_TRIES}: {e}")
            if attempt == MAX_TRIES:
                raise
            time.sleep(RETRY_WAIT)


def main():
    # (file, local path) of every file of every sensor
    files = []
    for sensor, (source_id, folder) in SENSORS.items():
        log(f"{sensor}: querying {INDEX_NODE} for {source_id}")
        listed = list_files(source_id)
        versions = sorted({f["path"].split("/")[-2] for f in listed})
        log(f"{sensor}: {len(listed)} files, {sum(f['size'] for f in listed) / 2**30:.1f} GiB, version "
            f"{', '.join(versions)} -> {OUTPUT_ROOT / folder}")
        files += [(f, OUTPUT_ROOT / folder / "data" / f["path"]) for f in listed]
    todo = [(f, p) for f, p in files if not (p.exists() and p.stat().st_size == f["size"])]
    log(f"{len(files)} files listed, {len(todo)} to download ({sum(f['size'] for f, _ in todo) / 2**30:.1f} GiB)")

    failed = []
    with ThreadPoolExecutor(PARALLEL) as executor:
        futures = {executor.submit(download, f, p): p for f, p in todo}
        for future in as_completed(futures):
            try:
                future.result()
                log(f"done {futures[future].name}")
            except Exception as e:
                log(f"FAILED: {futures[future].name}: {e}")
                failed.append(futures[future].name)

    log(f"done: {len(files) - len(failed)} of {len(files)} files complete")
    if failed:
        sys.exit(f"FAILED: {' '.join(failed)}; re-run to retry.")


if __name__ == "__main__":
    main()
