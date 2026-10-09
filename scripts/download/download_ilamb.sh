#!/usr/bin/env bash
# Download all ILAMB datasets using the `ilamb-fetch` utility.
#   https://www.ilamb.org/doc/ilamb_fetch.html
#
#   ilamb-fetch --local_root <destination-path> --remote_root https://www.ilamb.org/DATA -y
#
# Remote layout:  $REMOTE_ROOT/SHA1SUM_<collection> lists every file as <collection>/<variable>/<product>/<file>.nc
# Local layout:   $OUTPUT_ROOT/<collection>/<variable>/<product>/... (= etunc.config.ILAMB_ROOT/ILAMB-Data)
#
# ilamb-fetch downloads the remote checksum file, hashes every local file and downloads only the
# files that are missing or whose SHA1 differs, so re-running is safe (but hashing the existing
# files takes a while). -y accepts its prompts; run ilamb-fetch without it to review the file list
# before downloading.
#
# Requires the ILAMB Python package, which can be installed via the conda-forge channel.
#   https://www.ilamb.org/doc/install.html
# Note: The ILAMB package does have an MPI dependency, so it is not in the etunc env.
#
# Override the settings with env vars, e.g.
#   ILAMB_FETCH=/glade/work/bbuchovecky/miniforge3/envs/data-sci-py312/bin/ilamb-fetch ./download_ilamb.sh
#   OUTPUT_ROOT=/glade/derecho/scratch/$USER/ilamb ./download_ilamb.sh
# Long runs:
#   tmux new -s ilamb
#   ./download_ilamb.sh > download_ilamb.log 2>&1 &

set -uo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-https://www.ilamb.org/DATA}
OUTPUT_ROOT=${OUTPUT_ROOT:-/glade/campaign/univ/uwas0155/obs/ilamb}
COLLECTION=${COLLECTION:-ILAMB-Data}  # or ABoVE-Data, NGEEA-Data
ILAMB_FETCH=${ILAMB_FETCH:-ilamb-fetch}

command -v "$ILAMB_FETCH" > /dev/null || {
    echo "ilamb-fetch not found. Install ILAMB (e.g. conda install -c conda-forge ilamb)" \
         "and set ILAMB_FETCH=<env>/bin/ilamb-fetch." >&2
    exit 1
}

mkdir -p "$OUTPUT_ROOT" || exit 1

echo "Fetching $COLLECTION from $REMOTE_ROOT into $OUTPUT_ROOT"
"$ILAMB_FETCH" \
    --local_root "$OUTPUT_ROOT" \
    --remote_root "$REMOTE_ROOT" \
    --collection "$COLLECTION" \
    -y
