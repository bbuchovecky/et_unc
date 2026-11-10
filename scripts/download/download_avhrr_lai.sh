#!/usr/bin/env bash
# Download the NOAA-NCEI AVHRR LAI v5 monthly (obs4MIPs) from ESGF with esgpull.
#   https://esgf.github.io/esgf-download/
#   dataset: obs4MIPs.NOAA-NCEI.NOAA-NCEI-LAI-AVHRR-5-0.mon.lai.gn (v20260110)
#
# Remote:       https://esgf-node.ornl.gov/thredds/fileServer/user_pub_work/<local layout below>
# Local layout: $OUTPUT_ROOT/data/obs4MIPs/NOAA-NCEI/NOAA-NCEI-LAI-AVHRR-5-0/mon/lai/gn/v<version>/
#                   lai_mon_NOAA-NCEI-LAI-AVHRR-5-0_BE_gn_<YYYYMM>-<YYYYMM>.nc
# 33 yearly files (1981-06 to 2013-12, ~160 MB each), 4.9 GiB in all.
#
# ESGF's command for this dataset is
#   esgpull add master_id:'"obs4MIPs.NOAA-NCEI.NOAA-NCEI-LAI-AVHRR-5-0.mon.lai.gn"' --track
#   esgpull update <query id>; esgpull download --disable-ssl
# but it does not work as is with esgpull 0.10.0:
#   - the default index node (esgf-node.ipsl.upmc.fr) is down and the other European indexes do
#     not list the dataset, so the query goes to the ORNL ESGF 1.5 bridge (INDEX_NODE);
#   - the bridge rejects master_id, so the query uses the facets that make up the master_id.
# SSL verification is kept on (ORNL's certificate is valid); DISABLE_SSL=1 turns it off.
#
# OUTPUT_ROOT is its own esgpull install (database, config, logs, data/). Its install registry is
# kept in $OUTPUT_ROOT/.installs (ESGPULL_INSTALLS_PATH), so ~/.esgpull and any other install are
# left alone. The query is tagged "avhrr-lai"; esgpull skips the query if it was already added,
# skips downloaded files and verifies each file's SHA256, so re-running is safe.
#
# esgpull is not in the etunc env: the script uses `esgpull` if it is on PATH, otherwise runs a
# pinned version through `pixi exec` (a cached env, nothing installed). Override with ESGPULL.
#
# Settings are env vars, e.g.
#   OUTPUT_ROOT=/glade/derecho/scratch/$USER/avhrr-lai ./download_avhrr_lai.sh
#   ESGPULL=/path/to/env/bin/esgpull DISABLE_SSL=1 ./download_avhrr_lai.sh
# Long runs: run inside tmux, or nohup ./download_avhrr_lai.sh > download_avhrr_lai.log 2>&1 &

set -uo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/glade/derecho/scratch/bbuchovecky/avhrr-lai-v5}
INDEX_NODE=${INDEX_NODE:-esgf-node.ornl.gov/esgf-1-5-bridge}
TAG=avhrr-lai
FACETS=(
    project:obs4MIPs
    institution_id:NOAA-NCEI
    source_id:NOAA-NCEI-LAI-AVHRR-5-0
    frequency:mon
    variable_id:lai
    grid_label:gn
)
ATTEMPTS=${ATTEMPTS:-3}  # download passes; each pass after the first re-queues failed files
DISABLE_SSL=${DISABLE_SSL:-0}

# esgpull command (a string, split into words when run)
if [[ -z ${ESGPULL:-} ]]; then
    if command -v esgpull > /dev/null; then
        ESGPULL=esgpull
    else
        ESGPULL="${PIXI:-$HOME/.pixi/bin/pixi} exec --spec esgpull==0.10.0 esgpull"
    fi
fi
esgpull() { $ESGPULL "$@"; }

# Keep this install's registry next to the data, so the install below is the current one
export ESGPULL_INSTALLS_PATH=$OUTPUT_ROOT/.installs

# One-time setup: create the install and point it at the ORNL index
if [[ ! -f $ESGPULL_INSTALLS_PATH/installs.json ]]; then
    mkdir -p "$OUTPUT_ROOT" || exit 1
    esgpull self install "$OUTPUT_ROOT" --name "$TAG" || exit 1
fi
esgpull config api.index_node "$INDEX_NODE" > /dev/null || exit 1

# Add the query (skipped if it exists), then fetch its file list from the index into the queue
esgpull add --tag "$TAG" "${FACETS[@]}" --track || exit 1
esgpull update --tag "$TAG" --yes || exit 1

download_opts=(--tag "$TAG")
(( DISABLE_SSL )) && download_opts+=(--disable-ssl)
for (( i = 1; i <= ATTEMPTS; i++ )); do
    echo "$(date '+%F %T')  download pass $i of $ATTEMPTS"
    (( i > 1 )) && esgpull retry error cancelled
    esgpull download "${download_opts[@]}"
done

# Report: the files of the query that are not downloaded
esgpull status --all
not_done=$(esgpull show --tag "$TAG" --files --json | python3 -I -c '
import json, sys
for query in json.load(sys.stdin).values():
    for f in query["files"]:
        if f["status"] != "Done":
            print(f["status"], f["filename"])')
if [[ -n $not_done ]]; then
    echo "Not downloaded:" >&2
    echo "$not_done" >&2
    exit 1
fi
echo "All files in $OUTPUT_ROOT/data"
