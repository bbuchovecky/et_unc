"""Download GLEAM v4.3a and v4.3b monthly and yearly data (0.1 deg) from the GLEAM SFTP server.
    https://www.gleam.eu

Remote layout:  /data/<version>/<resolution>/<variable>/<variable>_<year>_GLEAM_<version>_<MO|YR>.nc
Local layout:   OUTPUT_ROOT/<version>/<resolution>/<variable>/...
OUTPUT_ROOT is SCRATCH_ROOT/gleam-v4.3 (config/paths.yml); move it to OBS_ROOT by hand once complete.
v4.3a covers 1980-2025 and v4.3b 2003-2025. Monthly files are ~90 MB per variable-year and yearly
~6 MB, so all 15 variables come to roughly 100 GB.

A file whose local size matches the remote size is skipped. Others are downloaded to <name>.part,
resumed from its size after an interruption, and renamed once complete, so re-running is safe and
retries only the missing files.

Needs the GLEAM SFTP login (sent on registration at gleam.eu) under "gleam" in config/auth.yml
(gitignored; template config/auth.example.yml).
Run:  pixi run python scripts/download/download_gleam.py
Long runs: run inside tmux, or nohup ... > download_gleam.log 2>&1 &
"""

import stat
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import paramiko

from etunc.config import SCRATCH_ROOT, auth

OUTPUT_ROOT = SCRATCH_ROOT / "gleam-v4.3"
LOGIN = auth("gleam")  # {"user": ..., "password": ...}
HOST = "aether.ugent.be"
PORT = 2225
REMOTE_ROOT = "/data"
DOCS = ["README_GLEAM4.3.pdf", "Log_file_GLEAM4.3.pdf"]  # in REMOTE_ROOT

VERSIONS = ["v4.3a", "v4.3b"]
RESOLUTIONS = ["monthly", "yearly"]
VARIABLES = ["E", "Et", "Ec", "Es", "Ei", "Eb", "Ew", "Ep"]  # also: Ep_aero Ep_rad H S SMrz SMs

PARALLEL = 4     # files transferred at once, each over its own SFTP connection
MAX_TRIES = 10   # attempts per file, each resuming the last
RETRY_WAIT = 30  # s between attempts
TIMEOUT = 120    # s without data before an attempt fails


def log(msg):
    print(f"{datetime.now():%F %T}  {msg}", flush=True)


def connect():
    """Open an SSH connection and its SFTP session; close the SSH client when done."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # accept the server's host key without a known_hosts entry
    ssh.connect(HOST, PORT, username=LOGIN["user"], password=LOGIN["password"], timeout=TIMEOUT,
                look_for_keys=False, allow_agent=False)
    sftp = ssh.open_sftp()
    sftp.get_channel().settimeout(TIMEOUT)
    return ssh, sftp


def fetch(remote, size, local):
    """Download remote (size bytes) to local through <local>.part, resuming it; raise after MAX_TRIES."""
    part = local.with_name(local.name + ".part")
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and not part.exists():
        local.rename(part)  # an earlier copy whose size is not the server's: resume it
    for attempt in range(1, MAX_TRIES + 1):
        try:
            done = part.stat().st_size if part.exists() else 0
            if done > size:  # the remote file changed: start over
                part.unlink()
                done = 0
            ssh, sftp = connect()
            try:
                with sftp.open(remote, "rb") as src, open(part, "ab") as dst:
                    src.seek(done)
                    src.prefetch(size, max_concurrent_requests=64)  # pipelined reads, much faster than one at a time
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
            finally:
                ssh.close()
            if part.stat().st_size != size:
                raise OSError(f"got {part.stat().st_size} of {size} bytes")
            part.rename(local)
            return
        except Exception as e:
            log(f"WARNING: {remote} attempt {attempt}/{MAX_TRIES}: {e}")
            if attempt == MAX_TRIES:
                raise
            time.sleep(RETRY_WAIT)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # List every remote file as (remote path, size, local path); keep the missing or incomplete ones
    files, unlisted, failed = [], [], []
    ssh, sftp = connect()
    try:
        for name in DOCS:
            files.append((f"{REMOTE_ROOT}/{name}", sftp.stat(f"{REMOTE_ROOT}/{name}").st_size, OUTPUT_ROOT / name))
        for version in VERSIONS:
            for res in RESOLUTIONS:
                for var in VARIABLES:
                    d = f"{version}/{res}/{var}"
                    try:
                        entries = sftp.listdir_attr(f"{REMOTE_ROOT}/{d}")
                    except OSError as e:
                        log(f"WARNING: cannot list {d}: {e}")
                        unlisted.append(d)
                        continue
                    files += [(f"{REMOTE_ROOT}/{d}/{a.filename}", a.st_size, OUTPUT_ROOT / d / a.filename)
                              for a in entries if stat.S_ISREG(a.st_mode)]
    finally:
        ssh.close()
    todo = [f for f in files if not (f[2].exists() and f[2].stat().st_size == f[1])]
    log(f"{len(files)} files listed, {len(todo)} to download ({sum(f[1] for f in todo) / 1e9:.1f} GB), "
        f"{PARALLEL} at a time -> {OUTPUT_ROOT}")

    with ThreadPoolExecutor(PARALLEL) as executor:
        futures = {executor.submit(fetch, *f): f for f in todo}
        for future in as_completed(futures):
            remote = futures[future][0]
            try:
                future.result()
                log(f"done {remote}")
            except Exception as e:
                log(f"FAILED: {remote}: {e}")
                failed.append(remote)

    log(f"done: {len(files) - len(failed)} of {len(files)} files complete "
        f"({sum(f[1] for f in files) / 1e9:.1f} GB listed) in {OUTPUT_ROOT}")
    if unlisted or failed:
        sys.exit(f"FAILED: {' '.join(unlisted + failed)}; re-run to retry.")


if __name__ == "__main__":
    main()
