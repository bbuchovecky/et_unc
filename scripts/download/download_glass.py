"""Download a GLASS product (Global LAnd Surface Satellite, Liang et al.) from the HKU archive.
Archive: https://glass.hku.hk/archive/ (nginx directory listings, no login)

The archive is laid out as <variable>/<sensor>/[<resolution>/]<year>/[<doy>/]<file>, with a depth that
differs by product, e.g.
    LAI/MODIS/0.05D/2001/GLASS01B01.V61.A2001001.2025248.hdf   (V6.1, 8-day, 2000-2025)
    LAI/MODIS/0.25D, 0.5D, 0.1D                                (the same record aggregated)
    LAI/MODIS/250M/2001/001/GLASS01D01.V60.A2001001.h00v08.2022012.hdf   (V6.0, sinusoidal tiles)
    LAI/AVHRR/1982/GLASS01B02.V40.A1982001.2019353.hdf         (V4.0, 0.05 deg, 8-day, 1981-2018)
    ET/MODIS/0.05D/2001/..., ET/AVHRR/0.5D/..., RN/, FAPAR/, GPP/, ...
so PRODUCT names any folder of the archive and every folder below it is walked (year folders outside
YEARS are skipped). Each data file has a quick-look .jpg and an XML sidecar (<file>.xml) that gives
its size and its POSIX `cksum` CRC; PATTERNS select the files to download by their path in PRODUCT.

The file list (path, URL, size, cksum, listing date) is saved to JSON_REF, one JSON per product in
references/, so the files downloaded are those of this exact listing. It is built once, by walking the
listings and reading each data file's sidecar (a HEAD request for the size of files without one), and
read back on every later run. The archive is updated in place (V6.0 -> V6.1 of the MODIS LAI in 2025),
so a file that no longer matches the saved list fails its check; set REFRESH = True to list the
product again (the git diff of JSON_REF shows what changed). A run whose PRODUCT, YEARS or PATTERNS
differ from those saved in JSON_REF stops and asks for REFRESH.

Local layout:  OUTPUT_ROOT/<path in PRODUCT>   (the layout of the archive, e.g. <year>/<file>)
OUTPUT_ROOT is SCRATCH_ROOT/glass/<PRODUCT> (config/paths.yml); move it to OBS_ROOT by hand once complete.
Each file is written to <file>.part and moved into place only once its size (and cksum, from the
sidecar) match the list, so a file that exists is complete: it is skipped, and re-running is safe
and retries only the missing files. An interrupted file restarts from the beginning.

The 0.05 deg MODIS LAI (2000-2025) is ~2.5 GB. Run:  pixi run python scripts/download/download_glass.py
"""

import fnmatch
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from etunc.config import SCRATCH_ROOT

ARCHIVE = "https://glass.hku.hk/archive/"
# Folder of the archive to download (see the docstring for the layout)
PRODUCT = "LAI/MODIS/0.05D"
# Years to download (year folders outside it are not walked), or None for every year
YEARS = None
# Files to download, as fnmatch patterns of the path in PRODUCT ("*" also matches "/"):
# the HDF data and their XML sidecars, not the .jpg quick looks
PATTERNS = ["*.hdf", "*.hdf.xml"]
# True: list the archive again and overwrite JSON_REF; False: download the files of JSON_REF
REFRESH = False
# Concurrent requests to the archive (~0.5 s per request from glade, so listing one at a time is slow)
N_WORKERS = 4
TIMEOUT = 120  # seconds per request
# Attempts per listing, sidecar or HEAD request: the archive drops one now and then (e.g. a TLS
# handshake timeout), which would otherwise stop a listing of ~1000 requests
RETRIES = 5

# Saved file list, relative to this script so the script runs from any directory
JSON_REF = Path(__file__).parent / "references" / f"glass_{PRODUCT.replace('/', '_')}.json"
OUTPUT_ROOT = SCRATCH_ROOT / "glass" / PRODUCT

# Entries of an nginx listing: <a href="name">name</a>   dd-Mon-yyyy hh:mm   size
LISTING_ENTRY = re.compile(r'<a href="([^"?]+)">[^<]*</a>\s+(\d{2}-\w{3}-\d{4} \d{2}:\d{2})')


def fetch(url, method="GET"):
    """Body and headers of a small request, retried RETRIES times with a growing wait.

    Network errors and server errors (5xx) are retried; other HTTP errors (e.g. 404) are raised.
    """
    for attempt in range(1, RETRIES + 1):
        try:
            with urlopen(Request(url, method=method), timeout=TIMEOUT) as r:
                return r.read(), r.headers
        except (URLError, TimeoutError, ConnectionError) as e:
            if (isinstance(e, HTTPError) and e.code < 500) or attempt == RETRIES:
                raise
            wait = 10 * attempt
            print(f"Retry {attempt}/{RETRIES - 1} in {wait} s: {url}: {e}")
            time.sleep(wait)


def list_folder(url):
    """Sub-folders and files of a listing page, each as {name: listing date (yyyy-mm-dd hh:mm)}."""
    page = fetch(url)[0].decode()
    folders, files = {}, {}
    for name, date in LISTING_ENTRY.findall(page):
        date = datetime.strptime(date, "%d-%b-%Y %H:%M").strftime("%Y-%m-%d %H:%M")
        if name.endswith("/"):
            folders[name.rstrip("/")] = date
        else:
            files[name] = date
    return folders, files


def walk(product, years):
    """Every file below the product folder as {path in product: listing date}.

    A folder named by a 4-digit year outside `years` is skipped (not listed).
    """
    found = {}
    todo = [""]
    while todo:
        rel = todo.pop()
        folders, files = list_folder(ARCHIVE + product + "/" + rel)
        found.update({rel + name: date for name, date in files.items()})
        for name in folders:
            if years is not None and re.fullmatch(r"\d{4}", name) and int(name) not in years:
                continue
            todo.append(rel + name + "/")
    return found


def read_sidecar(url):
    """Size (bytes) and POSIX cksum of a data file from its XML sidecar, and the sidecar's own size."""
    body = fetch(url)[0]
    container = ElementTree.fromstring(body).find(".//DataFileContainer")
    if container.findtext("ChecksumType") != "CKSUM":
        raise ValueError(f"{url}: checksum type {container.findtext('ChecksumType')}, not CKSUM")
    return int(container.findtext("FileSize")), int(container.findtext("Checksum")), len(body)


def remote_size(url):
    """Size (bytes) of a file from a HEAD request."""
    return int(fetch(url, method="HEAD")[1]["Content-Length"])


def describe(rel, listed):
    """List entries of a selected file: {path: {url, size, cksum, listed}}, with its sidecar's entry.

    Size and cksum come from the file's XML sidecar when it has one, else the size from a HEAD
    request and no cksum.
    """
    url = ARCHIVE + PRODUCT + "/" + rel
    sidecar = rel + ".xml"
    if sidecar in listed:
        size, cksum, sidecar_size = read_sidecar(url + ".xml")
        entries = {rel: {"url": url, "size": size, "cksum": cksum, "listed": listed[rel]}}
        # The sidecar's entry, kept if PATTERNS select it
        entries[sidecar] = {"url": url + ".xml", "size": sidecar_size, "cksum": None,
                            "listed": listed[sidecar]}
        return entries
    return {rel: {"url": url, "size": remote_size(url), "cksum": None, "listed": listed[rel]}}


def build_reference():
    """Walk the archive and describe every file of PATTERNS: the saved form of JSON_REF."""
    listed = walk(PRODUCT, YEARS)
    selected = sorted(rel for rel in listed if any(fnmatch.fnmatch(rel, p) for p in PATTERNS))
    # Sidecars are described with their data file, so only the others are requested on their own
    to_describe = [rel for rel in selected if not (rel.endswith(".xml") and rel[:-4] in listed)]
    print(f"Listing {len(selected)} files of {PRODUCT} ({len(to_describe)} requests)")

    files = {}
    with ThreadPoolExecutor(N_WORKERS) as pool:
        for entries in pool.map(lambda rel: describe(rel, listed), to_describe):
            files.update({rel: e for rel, e in entries.items() if rel in selected})
    missing = set(selected) - set(files)
    if missing:  # a sidecar selected without its data file
        raise RuntimeError(f"Not described: {sorted(missing)[:5]} ...; add its data file to PATTERNS")

    return {
        "archive": ARCHIVE,
        "product": PRODUCT,
        "years": None if YEARS is None else sorted(YEARS),
        "patterns": PATTERNS,
        "listed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_files": len(files),
        "total_bytes": sum(e["size"] for e in files.values()),
        "files": dict(sorted(files.items())),
    }


def load_reference():
    """JSON_REF, listed again if REFRESH or missing; stops if its settings differ from this script's."""
    if REFRESH or not JSON_REF.exists():
        reference = build_reference()
        JSON_REF.parent.mkdir(parents=True, exist_ok=True)
        JSON_REF.write_text(json.dumps(reference, indent=2) + "\n")
        print(f"Saved the file list to {JSON_REF}")
        return reference
    reference = json.loads(JSON_REF.read_text())
    settings = {"archive": ARCHIVE, "product": PRODUCT, "patterns": PATTERNS,
                "years": None if YEARS is None else sorted(YEARS)}
    differ = {k: (reference[k], v) for k, v in settings.items() if reference[k] != v}
    if differ:
        raise SystemExit(f"{JSON_REF.name} was listed with other settings (saved, current): {differ}. "
                         "Set REFRESH = True to list the archive again.")
    return reference


def posix_cksum(path):
    """CRC of a file as printed by the POSIX `cksum` command (the checksum of the GLASS sidecars)."""
    out = subprocess.run(["cksum", str(path)], capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def download(entry, dest):
    """Download one file to dest through dest.part, checking its size and cksum against the list."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(entry["url"], timeout=TIMEOUT) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    size = part.stat().st_size
    if size != entry["size"]:
        raise ValueError(f"size {size}, listed {entry['size']}")
    if entry["cksum"] is not None:
        cksum = posix_cksum(part)
        if cksum != entry["cksum"]:
            raise ValueError(f"cksum {cksum}, listed {entry['cksum']}")
    part.rename(dest)


def main():
    reference = load_reference()
    files = reference["files"]
    todo = {rel: e for rel, e in files.items() if not (OUTPUT_ROOT / rel).exists()}
    print(f"GLASS {PRODUCT} (listed {reference['listed_at']}): {len(files)} files, "
          f"{reference['total_bytes'] / 1e9:.2f} GB -> {OUTPUT_ROOT}")
    print(f"{len(files) - len(todo)} already downloaded, {len(todo)} to download")

    failed = []
    with ThreadPoolExecutor(N_WORKERS) as pool:
        futures = {pool.submit(download, e, OUTPUT_ROOT / rel): rel for rel, e in todo.items()}
        for i, future in enumerate(as_completed(futures), 1):
            rel = futures[future]
            try:
                future.result()
            except Exception as e:
                failed.append(rel)
                print(f"Failed: {rel}: {e}")
            if i % 100 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}")

    print(f"done: {len(todo) - len(failed)}, failed: {len(failed)}")
    if failed:
        print("Re-run to retry the failed files. Files that fail their size or cksum on every run "
              "have changed on the archive: set REFRESH = True to list it again.")


if __name__ == "__main__":
    main()
