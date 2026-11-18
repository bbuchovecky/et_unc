"""Download TRENDY v14 (GCB 2025) DGVM outputs and their forcing from the GCB S3 bucket (Wasabi).

The bucket cannot be listed, so the files are taken from trendy_files.sh (next to this script), the list of
every GCB 2025 file (one `wget "<url>"` line each), with these rules:
  - outputs (land/output/): <MODEL>/<EXPT>/<MODEL>_<S2|S3>_<var>[_PFT<nn>].nc[.gz|.tar.gz] of OUTPUT_VARIABLES
    and EXPERIMENTS; a variable matches exactly ("gpp" does not match "gpppft"; CLM's _PFT<nn> files are kept);
  - forcing (land/input/): every file with a word of INPUT_VARIABLES in its name, delimited by "." or "_"
    ("pre" does not match "pres"); the forcing is shared by every experiment.

Local layout:  OUTPUT_ROOT/<MODEL>/<EXPT>/*.nc   (outputs; EXPT as on the server, e.g. S2, S3, correctedS2,
                                                  since some models reuse file names across folders)
               OUTPUT_ROOT/<folder>/*.nc         (forcing: crujra3/, Radiation/, ...)
OUTPUT_ROOT is SCRATCH_ROOT/trendy (config/paths.yml); move it to TRENDY_ROOT by hand once complete.

.nc.gz files (SDGVM, VISIT, CRU-JRA, Radiation) are gunzipped and .nc.tar.gz files (iMAPLE, one .nc each)
extracted, then the archive is deleted. Each download goes to <name>.part, is resumed after an interruption,
and is renamed once its size matches the server's; a decompressed .nc is written as <name>.nc.part and
renamed once complete. So a .nc decompressed from an archive is complete once it exists, and an existing
plain .nc is checked against the server's size (a HEAD request; it is resumed if shorter): re-running is safe
and retries only the missing or incomplete files.

No login needed. Run:  pixi run python scripts/download/download_trendy.py
Long runs: run inside tmux, or nohup ... > download_trendy.log 2>&1 &
"""

import gzip
import re
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

from etunc.config import SCRATCH_ROOT

URL_LIST = Path(__file__).resolve().parent / "trendy_files.sh"
OUTPUT_ROOT = SCRATCH_ROOT / "trendy"

# DGVM outputs (land/output/)
OUTPUT_VARIABLES = ["evapotrans", "tran", "evspsblsoi", "evspsblveg", "lai", "pr", "rsds", "landCoverFrac"]
EXPERIMENTS = ["S2", "S3"]  # S2: climate + CO2, S3: + land use change
# Forcing (land/input/): CRU-JRA precipitation and downward longwave (crujra3/), and the diffuse
# fraction of the downward shortwave (Radiation/)
INPUT_VARIABLES = ["pre", "dlwrf", "fd"]

PARALLEL = 4     # files downloaded at once
MAX_TRIES = 5    # attempts per file, each resuming the last
RETRY_WAIT = 30  # s between attempts
TIMEOUT = 120    # s without data before an attempt fails


def log(msg):
    print(f"{datetime.now():%F %T}  {msg}", flush=True)


def is_output(rel, variables):
    """Whether rel (a path under land/output/) is an output of `variables` in EXPERIMENTS."""
    return re.search(rf"_({'|'.join(EXPERIMENTS)})_({'|'.join(variables)})(_PFT\d+)?\.[^/]*$", rel) is not None


def is_input(rel, variables):
    """Whether rel (a path under land/input/) has a word of `variables` in its file name."""
    return re.search(rf"(^|/)([^/]*[._])?({'|'.join(variables)})[._][^/]*$", rel) is not None


def select(urls):
    """(url, path under OUTPUT_ROOT) of the files to download, in the order of URL_LIST."""
    files = []
    for url in urls:
        source, _, rel = url.partition("/land/")[2].partition("/")  # ocean files give source ""
        if (source == "output" and is_output(rel, OUTPUT_VARIABLES)
                or source == "input" and is_input(rel, INPUT_VARIABLES)):
            files.append((url, rel))
    return files


def download(url, path):
    """Download url to path through <path>.part, resuming it; return the bytes transferred.
    An earlier copy at path is kept as is if its size is the server's."""
    part = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not part.exists():
        path.rename(part)  # an earlier copy, possibly incomplete: resume or confirm it
    transferred = 0
    for attempt in range(1, MAX_TRIES + 1):
        try:
            # Size on the server, from a HEAD request: this server answers a range at or past the
            # end of the file with the whole file, so a complete copy must be recognized first
            head = requests.head(url, timeout=TIMEOUT)
            head.raise_for_status()
            total = int(head.headers["Content-Length"])
            done = part.stat().st_size if part.exists() else 0
            if done > total:  # the server's file changed: start over
                part.unlink()
                done = 0
            if done < total:
                with requests.get(url, headers={"Range": f"bytes={done}-"} if done else {},
                                  stream=True, timeout=TIMEOUT) as r:
                    r.raise_for_status()
                    if r.status_code == 200:  # the range was ignored: the whole file
                        done = 0
                    with open(part, "ab" if done else "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                            transferred += len(chunk)
            if part.stat().st_size != total:
                raise OSError(f"got {part.stat().st_size} of {total} bytes")
            part.rename(path)
            return transferred
        except Exception as e:
            log(f"WARNING: {path.name} attempt {attempt}/{MAX_TRIES}: {e}")
            if attempt == MAX_TRIES:
                raise
            time.sleep(RETRY_WAIT)


def decompress(archive, nc):
    """Decompress a .nc.gz, or the .nc named nc.name in a .nc.tar.gz, to nc through <nc>.part."""
    part = nc.with_name(nc.name + ".part")
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r|gz") as tar:  # streamed: the archive is read once
            for member in tar:
                # Only the member's content is read, to a path chosen here, never extracted as named
                if member.isfile() and Path(member.name).name == nc.name:
                    with tar.extractfile(member) as src, open(part, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 24)
                    break
            else:
                raise ValueError(f"{archive.name} has no {nc.name}")
    else:
        with gzip.open(archive, "rb") as src, open(part, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
    part.rename(nc)


def get(url, rel):
    """Download url to OUTPUT_ROOT/rel and decompress it to its .nc; return the bytes transferred."""
    archive = OUTPUT_ROOT / rel
    nc = OUTPUT_ROOT / re.sub(r"(\.tar)?\.gz$", "", rel)
    if archive != nc and nc.exists() and not archive.exists():
        return 0  # decompressed in an earlier run (an archive left next to its .nc means it was interrupted)
    n = download(url, archive)
    if archive != nc:
        decompress(archive, nc)
        archive.unlink()
    return n


def main():
    urls = re.findall(r'^wget "([^"]+)"', URL_LIST.read_text(), re.M)
    files = select(urls)
    log(f"{len(files)} of the {len(urls)} files of {URL_LIST.name} -> {OUTPUT_ROOT}")
    for v in OUTPUT_VARIABLES:
        log(f"  output {v}: {sum('/land/output/' in u and is_output(r, [v]) for u, r in files)} files")
    for v in INPUT_VARIABLES:
        log(f"  input {v}: {sum('/land/input/' in u and is_input(r, [v]) for u, r in files)} files")

    failed, n_bytes = [], 0
    with ThreadPoolExecutor(PARALLEL) as executor:
        futures = {executor.submit(get, url, rel): rel for url, rel in files}
        for future in as_completed(futures):
            try:
                n = future.result()
                n_bytes += n
                if n:
                    log(f"done {futures[future]} ({n / 1e9:.2f} GB)")
            except Exception as e:
                log(f"FAILED: {futures[future]}: {e}")
                failed.append(futures[future])

    log(f"done: {len(files) - len(failed)} of {len(files)} files complete, {n_bytes / 1e9:.1f} GB downloaded, "
        f"in {OUTPUT_ROOT}")
    if failed:
        sys.exit(f"FAILED: {' '.join(failed)}; re-run to retry.")


if __name__ == "__main__":
    main()
