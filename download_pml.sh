#!/usr/bin/env bash
# Download PML-V2.2 global ET/GPP (0.1 deg) from the TPDC FTP server.
#   https://data.tpdc.ac.cn/en/data/1eec058e-74fd-452d-9255-dae4f1340783
#
# Remote layout:  PML/<version>/<resolution>/<variable>/PML-<version>_<variable>_<year>.nc
# Local layout:   $OUTPUT_ROOT/<version>/<resolution>/<variable>/...
# Full dataset: 1820 files, ~43 GB.
#
# The server caps each connection at ~30 KB/s and each account at 10 concurrent
# connections (across both hosts), so files are fetched PARALLEL at a time,
# alternating hosts (~8 x 30 KB/s: the full dataset takes ~2 days). Re-running
# skips files whose local size matches the remote size and resumes partial ones.
#
# Subset with space-separated env vars, e.g.
#   VERSIONS="V2.2c" RESOLUTIONS="monthly" VARIABLES="ET PET" ./download_pml.sh
# Long runs: nohup ./download_pml.sh > download_pml.log 2>&1 &

set -uo pipefail

FTP_HOSTS=(ftp2.tpdc.ac.cn ftp3.tpdc.ac.cn)
FTP_PORT=6201
FTP_USER=download_84312140
FTP_PASS=44795442
REMOTE_ROOT=PML
OUTPUT_ROOT=/glade/derecho/scratch/bbuchovecky/pml-v2.2
PARALLEL=${PARALLEL:-8}  # must stay below the server's limit of 10

# V2.2a-* use "8-day", V2.2b/c use "half-month"; invalid combinations are skipped
VERSIONS=${VERSIONS:-"V2.2a-MODIS V2.2a-VIIRS V2.2b V2.2c"}
RESOLUTIONS=${RESOLUTIONS:-"8-day half-month monthly yearly"}
RESOLUTIONS=${RESOLUTIONS:-"8-day half-month monthly yearly"}
VARIABLES=${VARIABLES:-"ET Ec Es Ei E Ew PET GPP"}

export FTP_PORT FTP_USER FTP_PASS REMOTE_ROOT OUTPUT_ROOT
export FTP_HOSTS_STR="${FTP_HOSTS[*]}"

# Download one file ($1 = path under PML/, $2 = job index), resuming partial
# files; start on host (index mod nhosts) and alternate hosts over 6 attempts.
fetch() {
    local path=$1 idx=$2 hosts i host
    read -ra hosts <<< "$FTP_HOSTS_STR"
    mkdir -p "$OUTPUT_ROOT/$(dirname "$path")"
    for (( i = 0; i < 6; i++ )); do
        (( i > 0 )) && sleep 60
        host=${hosts[$(( (idx + i) % ${#hosts[@]} ))]}
        wget -c -nv --tries=5 --waitretry=30 --timeout=120 \
            --ftp-user="$FTP_USER" --ftp-password="$FTP_PASS" \
            -P "$OUTPUT_ROOT/$(dirname "$path")" "ftp://${host}:${FTP_PORT}/${REMOTE_ROOT}/${path}" && return 0
        echo "WARNING: $path failed on $host" >&2
    done
    echo "FAILED: $path" >&2
    return 1
}
export -f fetch

# List a remote directory as "<size> <path>" lines for each .nc file
list_dir() {
    curl -s --retry 3 -m 120 -u "$FTP_USER:$FTP_PASS" \
        "ftp://${FTP_HOSTS[0]}:${FTP_PORT}/${REMOTE_ROOT}/$1/" \
        | awk -v d="$1" '$NF ~ /\.nc$/ {print $5, d "/" $NF}'
}

mkdir -p "$OUTPUT_ROOT"
fetch README.txt 0

# Build the list of files that are missing or incomplete locally
todo=$(mktemp)
trap 'rm -f "$todo"' EXIT
n_total=0
for version in $VERSIONS; do
    for res in $RESOLUTIONS; do
        case "$version/$res" in
            V2.2a-*/half-month | V2.2b/8-day | V2.2c/8-day) continue ;;
        esac
        for var in $VARIABLES; do
            while read -r size path; do
                (( n_total++ ))
                local_size=$(stat -c %s "$OUTPUT_ROOT/$path" 2>/dev/null || echo -1)
                [[ $local_size == "$size" ]] || echo "$size $path" >> "$todo"
            done < <(list_dir "$version/$res/$var")
        done
    done
done

n_todo=$(wc -l < "$todo")
gb_todo=$(awk '{s += $1} END {printf "%.1f", s/1e9}' "$todo")
echo "$(date '+%F %T')  $n_total files listed, $n_todo to download ($gb_todo GB), $PARALLEL connections"

# Fetch in parallel; the line number picks the starting host
awk '{print $2, NR}' "$todo" | xargs -P "$PARALLEL" -n 2 bash -c 'fetch "$0" "$1"'
status=$?

echo "$(date '+%F %T')  done: $(find "$OUTPUT_ROOT" -name '*.nc' | wc -l) .nc files," \
     "$(du -sh "$OUTPUT_ROOT" | cut -f1) in $OUTPUT_ROOT"
(( status == 0 )) || { echo "Some files failed (see FAILED lines); re-run to retry." >&2; exit 1; }
