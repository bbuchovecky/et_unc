"""Download SiTH v2 global ET components and soil moisture (1982-2022) from the TPDC FTP server.

Remote layout:  Monthly/<variable>.SiTHv2.A1982_2022.M.nc
                Yearly/<variable>.SiTHv2.A1982_2022.Y.nc
                Daily/<year>/<variable>.SiTHv2.A<year>.nc
Local layout:   OUTPUT_ROOT/<same as remote>
OUTPUT_ROOT is SCRATCH_ROOT/sith-v2 (config/paths.yml); move it to OBS_ROOT by hand once complete.
Each variable is one large file: Monthly is 8.6 GB in all, Yearly 1.1 GB and Daily 175 GB.
SM alone is 40% of Monthly and 60% of Daily.

The server gives ~50 KB/s per connection and allows 5 connections per IP to each host; beyond that it
refuses with "421 Too many connections (5) from this IP". The limit is shared with every other TPDC
download from the same machine (e.g. download_pml.py), so lower SEGMENTS when running those at the
same time. The variables are dealt to the two hosts, which download at the same time, one file at a
time each, every file in SEGMENTS byte ranges over SEGMENTS FTP sessions (~150 KB/s per host by
default). Each range goes to <name>.part<k>of<SEGMENTS>, resumed from its size after an interruption;
the ranges are joined and renamed to <name> once all are complete. A file whose local size matches
the remote size is skipped, so re-running is safe and retries only the missing files. A file that
fails on one host is retried on the other.

Needs the TPDC FTP account of the dataset's download request under "sith" in config/auth.yml
(gitignored; template config/auth.example.yml).
Run:  pixi run python scripts/download/download_sith.py
Long runs: run inside tmux, or nohup ... > download_sith.log 2>&1 &
"""

import ftplib
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from etunc.config import SCRATCH_ROOT, auth

OUTPUT_ROOT = SCRATCH_ROOT / "sith-v2"
LOGIN = auth("sith")  # {"user": ..., "password": ...}
FTP_HOSTS = ["ftp2.tpdc.ac.cn", "ftp3.tpdc.ac.cn"]
FTP_PORT = 6201

RESOLUTIONS = ["Monthly"]          # also "Yearly" (1.1 GB) and "Daily" (175 GB, opt-in)
YEARS = range(1982, 2023)          # Daily only
VARIABLES = ["ET", "Ei", "En", "Es", "Tr"]  # also SM

SEGMENTS = 3     # FTP sessions per file (= per host); must stay below the server's limit of 5 per IP per host
MAX_TRIES = 30   # attempts per byte range, each resuming the last
RETRY_WAIT = 30  # s between attempts
TIMEOUT = 120    # s without data before an attempt fails


def log(msg):
    print(f"{datetime.now():%F %T}  {msg}", flush=True)


def connect(host):
    """Open and log in an FTP session (passive mode, binary) on host."""
    ftp = ftplib.FTP(timeout=TIMEOUT)
    ftp.connect(host, FTP_PORT)
    ftp.login(LOGIN["user"], LOGIN["password"])
    ftp.voidcmd("TYPE I")
    return ftp


def close(ftp):
    try:
        ftp.quit()
    except Exception:
        ftp.close()


def list_dir(ftp, d):
    """(file name, size) of each file of remote directory d, from the LIST output."""
    lines = []
    ftp.retrlines(f"LIST {d}/", lines.append)
    # Unix-style lines: permissions, links, owner, group, size, month, day, time or year, name
    return [(p[8], int(p[4])) for p in (line.split(None, 8) for line in lines)
            if len(p) == 9 and p[0].startswith("-")]


def fetch_range(host, path, start, end, part):
    """Download bytes [start, end) of remote path to the file part, resuming it; raise after MAX_TRIES."""
    for attempt in range(1, MAX_TRIES + 1):
        done = part.stat().st_size if part.exists() else 0
        if done > end - start:  # the remote file changed: start over
            part.unlink()
            done = 0
        if done == end - start:
            return
        try:
            ftp = connect(host)
            try:
                # RETR from start + done (REST), read up to end, then drop the connection
                with ftp.transfercmd(f"RETR {path}", rest=start + done) as conn, open(part, "ab") as f:
                    remaining = end - start - done
                    while remaining > 0 and (chunk := conn.recv(min(1 << 16, remaining))):
                        f.write(chunk)
                        remaining -= len(chunk)
            finally:
                ftp.close()  # not quit(): the RETR may be cut short, so its end reply is not read
            if part.stat().st_size != end - start:
                raise OSError(f"got {part.stat().st_size} of {end - start} bytes")
            return
        except Exception as e:
            log(f"WARNING: {part.name} on {host} (attempt {attempt}/{MAX_TRIES}): {e}")
            if attempt == MAX_TRIES:
                raise
            time.sleep(RETRY_WAIT)


def fetch(host, path, size):
    """Download remote path (size bytes) to OUTPUT_ROOT/path in SEGMENTS byte ranges in parallel."""
    local = OUTPUT_ROOT / path
    local.parent.mkdir(parents=True, exist_ok=True)
    bounds = [size * k // SEGMENTS for k in range(SEGMENTS + 1)]
    parts = [local.with_name(f"{local.name}.part{k + 1}of{SEGMENTS}") for k in range(SEGMENTS)]
    log(f"{path} ({size / 1e9:.2f} GB) on {host}")
    with ThreadPoolExecutor(SEGMENTS) as executor:
        futures = [executor.submit(fetch_range, host, path, bounds[k], bounds[k + 1], parts[k])
                   for k in range(SEGMENTS)]
        for future in futures:
            future.result()  # raises if a range failed (its .part file is kept for the next try)
    # Join the ranges, then rename, so <name> exists only when complete
    joined = local.with_name(local.name + ".part")
    with open(joined, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 24)
    if joined.stat().st_size != size:
        raise OSError(f"{path}: joined {joined.stat().st_size} of {size} bytes")
    joined.rename(local)
    for p in parts:
        p.unlink()
    log(f"done {path}")


def fetch_all(host, files):
    """Download files [(path, size), ...] one at a time on host; return the (path, size) that failed."""
    failed = []
    for path, size in files:
        try:
            fetch(host, path, size)
        except Exception as e:
            log(f"WARNING: {path} failed on {host}: {e}")
            failed.append((path, size))
    return failed


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Deal the variables out to the hosts round-robin
    host_of = {var: FTP_HOSTS[i % len(FTP_HOSTS)] for i, var in enumerate(VARIABLES)}
    dirs = [f"Daily/{year}" for year in YEARS] if "Daily" in RESOLUTIONS else []
    dirs = [res for res in RESOLUTIONS if res != "Daily"] + dirs

    n_files, unlisted, failed = 0, [], []
    for d in dirs:
        # Remote files of the chosen variables; the ".SiTHv2." suffix keeps "E" from matching "Ei", "Es", ...
        ftp = connect(FTP_HOSTS[0])
        try:
            listing = list_dir(ftp, d)
        except ftplib.all_errors as e:
            log(f"WARNING: cannot list {d}: {e}")
            unlisted.append(d)
            continue
        finally:
            close(ftp)
        todo = {host: [] for host in FTP_HOSTS}
        for name, size in listing:
            var = name.split(".SiTHv2.")[0]
            if var in host_of:
                n_files += 1
                local = OUTPUT_ROOT / d / name
                if not (local.exists() and local.stat().st_size == size):
                    todo[host_of[var]].append((f"{d}/{name}", size))
        n_todo = sum(len(f) for f in todo.values())
        log(f"{d}: {n_todo} files to download ({sum(s for f in todo.values() for _, s in f) / 1e9:.1f} GB)")

        # One worker per host at the same time, each on its own share of the variables
        with ThreadPoolExecutor(len(FTP_HOSTS)) as executor:
            failed_on = dict(zip(FTP_HOSTS, executor.map(fetch_all, todo, todo.values())))
        # Retry each failed file on the next host, one host at a time so no host exceeds SEGMENTS sessions
        for i, host in enumerate(FTP_HOSTS):
            other = FTP_HOSTS[(i + 1) % len(FTP_HOSTS)]
            if failed_on[host]:
                log(f"retrying {len(failed_on[host])} files of {host} on {other}")
                failed += [path for path, _ in fetch_all(other, failed_on[host])]

    log(f"done: {n_files - len(failed)} of {n_files} files complete in {OUTPUT_ROOT}")
    if unlisted or failed:
        sys.exit(f"FAILED: {' '.join(unlisted + failed)}; re-run to retry.")


if __name__ == "__main__":
    main()
