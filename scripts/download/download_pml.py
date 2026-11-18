"""Download PML-V2.2 global ET/GPP (0.1 deg) from the TPDC FTP server.
    https://data.tpdc.ac.cn/en/data/1eec058e-74fd-452d-9255-dae4f1340783

Remote layout:  PML/<version>/<resolution>/<variable>/PML-<version>_<variable>_<year>.nc
Local layout:   OUTPUT_ROOT/<version>/<resolution>/<variable>/...
OUTPUT_ROOT is SCRATCH_ROOT/pml-v2.2 (config/paths.yml); move it to OBS_ROOT by hand once complete.
Full dataset: 1820 files, ~43 GB.

The server caps each connection at ~30 KB/s and each account at 10 concurrent connections (across
both hosts), so files are fetched PARALLEL at a time, alternating hosts (~8 x 30 KB/s: the full
dataset takes ~2 days). A file whose local size matches the remote size is skipped. Others are
downloaded to <name>.part, resumed from its size after an interruption (each new attempt on the other
host), and renamed once complete, so re-running is safe and retries only the missing files.

Needs the TPDC FTP account of the dataset's download request under "pml" in config/auth.yml
(gitignored; template config/auth.example.yml).
Run:  pixi run python scripts/download/download_pml.py
Long runs: nohup ... > download_pml.log 2>&1 &
"""

import ftplib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from etunc.config import SCRATCH_ROOT, auth

OUTPUT_ROOT = SCRATCH_ROOT / "pml-v2.2"
LOGIN = auth("pml")  # {"user": ..., "password": ...}
FTP_HOSTS = ["ftp2.tpdc.ac.cn", "ftp3.tpdc.ac.cn"]
FTP_PORT = 6201
REMOTE_ROOT = "PML"
DOCS = ["README.txt"]  # in REMOTE_ROOT

VERSIONS = ["V2.2a-MODIS", "V2.2a-VIIRS", "V2.2b", "V2.2c"]
RESOLUTIONS = ["8-day", "half-month", "monthly", "yearly"]
VARIABLES = ["ET", "Ec", "Es", "Ei", "E", "Ew", "PET", "GPP"]
# V2.2a-* have 8-day files and V2.2b/c half-month ones: the other combinations do not exist
SKIP = {("V2.2a-MODIS", "half-month"), ("V2.2a-VIIRS", "half-month"), ("V2.2b", "8-day"), ("V2.2c", "8-day")}

PARALLEL = 8     # files transferred at once, one FTP session each; must stay below the server's limit of 10
MAX_TRIES = 12   # attempts per file, alternating hosts, each resuming the last
RETRY_WAIT = 60  # s between attempts
TIMEOUT = 120    # s without data before an attempt fails


def log(msg):
    print(f"{datetime.now():%F %T}  {msg}", flush=True)


def connect(host):
    """Open and log in an FTP session (passive mode) on host."""
    ftp = ftplib.FTP(timeout=TIMEOUT)
    ftp.connect(host, FTP_PORT)
    ftp.login(LOGIN["user"], LOGIN["password"])
    ftp.voidcmd("TYPE I")  # binary mode, which SIZE needs on many servers
    return ftp


def close(ftp):
    try:
        ftp.quit()
    except Exception:
        ftp.close()


def list_dir(ftp, d):
    """(file name, size) of each .nc file of remote directory d, from the LIST output."""
    lines = []
    ftp.retrlines(f"LIST {REMOTE_ROOT}/{d}/", lines.append)
    # Unix-style lines: permissions, links, owner, group, size, month, day, time or year, name
    return [(p[8], int(p[4])) for p in (line.split(None, 8) for line in lines)
            if len(p) == 9 and p[8].endswith(".nc")]


def fetch(path, size, idx):
    """Download PML/<path> (size bytes) to OUTPUT_ROOT/<path> through a .part file, resuming it; attempt
    i is on host (idx + i) mod 2, so the parallel downloads are spread over both hosts. Raise after MAX_TRIES."""
    local = OUTPUT_ROOT / path
    part = local.with_name(local.name + ".part")
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and not part.exists():
        local.rename(part)  # an earlier copy whose size is not the server's: resume it
    for attempt in range(MAX_TRIES):
        host = FTP_HOSTS[(idx + attempt) % len(FTP_HOSTS)]
        try:
            done = part.stat().st_size if part.exists() else 0
            if done > size:  # the remote file changed: start over
                part.unlink()
                done = 0
            ftp = connect(host)
            try:
                with open(part, "ab") as f:
                    ftp.retrbinary(f"RETR {REMOTE_ROOT}/{path}", f.write, blocksize=1 << 16, rest=done or None)
            finally:
                close(ftp)
            if part.stat().st_size != size:
                raise OSError(f"got {part.stat().st_size} of {size} bytes")
            part.rename(local)
            return
        except Exception as e:
            log(f"WARNING: {path} failed on {host} (attempt {attempt + 1}/{MAX_TRIES}): {e}")
            if attempt + 1 == MAX_TRIES:
                raise
            time.sleep(RETRY_WAIT)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # List every remote file as (path under PML/, size); keep the missing or incomplete ones
    files, unlisted, failed = [], [], []
    ftp = connect(FTP_HOSTS[0])
    try:
        files += [(name, ftp.size(f"{REMOTE_ROOT}/{name}")) for name in DOCS]
        for version in VERSIONS:
            for res in RESOLUTIONS:
                if (version, res) in SKIP:
                    continue
                for var in VARIABLES:
                    d = f"{version}/{res}/{var}"
                    try:
                        files += [(f"{d}/{name}", size) for name, size in list_dir(ftp, d)]
                    except ftplib.all_errors as e:
                        log(f"WARNING: cannot list {d}: {e}")
                        unlisted.append(d)
    finally:
        close(ftp)
    todo = [(path, size) for path, size in files
            if not ((OUTPUT_ROOT / path).exists() and (OUTPUT_ROOT / path).stat().st_size == size)]
    log(f"{len(files)} files listed, {len(todo)} to download ({sum(s for _, s in todo) / 1e9:.1f} GB), "
        f"{PARALLEL} connections -> {OUTPUT_ROOT}")

    with ThreadPoolExecutor(PARALLEL) as executor:
        futures = {executor.submit(fetch, path, size, idx): path for idx, (path, size) in enumerate(todo)}
        for future in as_completed(futures):
            try:
                future.result()
                log(f"done {futures[future]}")
            except Exception as e:
                log(f"FAILED: {futures[future]}: {e}")
                failed.append(futures[future])

    log(f"done: {len(files) - len(failed)} of {len(files)} files complete "
        f"({sum(s for _, s in files) / 1e9:.1f} GB listed) in {OUTPUT_ROOT}")
    if unlisted or failed:
        sys.exit(f"FAILED: {' '.join(unlisted + failed)}; re-run to retry.")


if __name__ == "__main__":
    main()
