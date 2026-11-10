#!/usr/bin/env bash
# Download the NOAA NCEI LAI climate data record (obs4MIPs, 0.05 deg monthly means) from ESGF with esgpull.
#   https://esgf.github.io/esgf-download/
#   https://www.ncei.noaa.gov/products/climate-data-records/leaf-area-index-and-fapar
#
# obs4MIPs publishes the record as one dataset per sensor; SENSOR picks one:
#   avhrr: obs4MIPs.NOAA-NCEI.NOAA-NCEI-LAI-AVHRR-5-0.mon.lai.gn  1981-06 to 2013-12, 33 files, 4.9 GiB
#   viirs: obs4MIPs.NOAA-NCEI.NOAA-NCEI-LAI-VIIRS-1-0.mon.lai.gn  2014-01 to 2024-12, 11 files, 1.7 GiB
# (an earlier combined publication, NOAA-NCEI-LAI-5-0 v20250731, stops in 2018 and is not used).
#
# Remote:       https://esgf-node.ornl.gov/thredds/fileServer/user_pub_work/<local layout below>
# Local layout: $OUTPUT_ROOT/data/obs4MIPs/NOAA-NCEI/<source_id>/mon/lai/gn/v<version>/
#                   lai_mon_<source_id>_BE_gn_<YYYYMM>-<YYYYMM>.nc   (one file per year)
# etunc.load.obs ("avhrr", "viirs", and both joined by load_ncei_lai) reads this layout from
# OBS_ROOT/avhrr-lai-v5 and OBS_ROOT/viirs-lai-v1.
#
# ESGF's command for the AVHRR dataset is
#   esgpull add master_id:'"obs4MIPs.NOAA-NCEI.NOAA-NCEI-LAI-AVHRR-5-0.mon.lai.gn"' --track
#   esgpull update <query id>; esgpull download --disable-ssl
# but it does not work as is with esgpull 0.10.0:
#   - the default index node (esgf-node.ipsl.upmc.fr) is down and the other European indexes do
#     not list these datasets, so the query goes to the ORNL ESGF 1.5 bridge (INDEX_NODE);
#   - the bridge rejects master_id, so the query uses the facets that make up the master_id.
# SSL verification is kept on (ORNL's certificate is valid); DISABLE_SSL=1 turns it off.
#
# OUTPUT_ROOT is its own esgpull install (database, config, logs, data/). Its install registry is
# kept in $OUTPUT_ROOT/.installs (ESGPULL_INSTALLS_PATH), so ~/.esgpull and any other install are
# left alone, and it is rewritten on every run, so OUTPUT_ROOT can be moved (e.g. from scratch to
# campaign) and the script re-run there. The query is tagged with the sensor; esgpull skips the query
# if it was already added, skips downloaded files and verifies each file's SHA256, so re-running is
# safe.
#
# esgpull is not in the etunc env: the script uses `esgpull` if it is on PATH, otherwise runs a
# pinned version through `pixi exec` (a cached env, nothing installed). Override with ESGPULL.
#
# Settings are env vars, e.g.
#   SENSOR=viirs ./download_ncei_lai.sh
#   SENSOR=viirs OUTPUT_ROOT=/glade/campaign/univ/uwas0155/obs/viirs-lai-v1 ./download_ncei_lai.sh
#   ESGPULL=/path/to/env/bin/esgpull DISABLE_SSL=1 ./download_ncei_lai.sh
# Long runs: run inside tmux, or nohup ./download_ncei_lai.sh > download_ncei_lai.log 2>&1 &

set -uo pipefail

SENSOR=${SENSOR:-avhrr}
case $SENSOR in
    avhrr) SOURCE_ID=NOAA-NCEI-LAI-AVHRR-5-0; DIR_NAME=avhrr-lai-v5 ;;
    viirs) SOURCE_ID=NOAA-NCEI-LAI-VIIRS-1-0; DIR_NAME=viirs-lai-v1 ;;
    *) echo "Unknown SENSOR=$SENSOR (avhrr or viirs)" >&2; exit 2 ;;
esac
OUTPUT_ROOT=${OUTPUT_ROOT:-/glade/derecho/scratch/bbuchovecky/$DIR_NAME}
INDEX_NODE=${INDEX_NODE:-esgf-node.ornl.gov/esgf-1-5-bridge}
TAG=$SENSOR-lai
FACETS=(
    project:obs4MIPs
    institution_id:NOAA-NCEI
    source_id:$SOURCE_ID
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

# Keep this install's registry next to the data, so the install below is the current one. The
# registry holds the absolute path of the install, so it is rebuilt each run (`self install` on an
# existing install reuses its database and files)
export ESGPULL_INSTALLS_PATH=$OUTPUT_ROOT/.installs
rm -f "$ESGPULL_INSTALLS_PATH/installs.json"
mkdir -p "$OUTPUT_ROOT" || exit 1
esgpull self install "$OUTPUT_ROOT" --name "$TAG" || exit 1
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
