#!/usr/bin/env bash
# Download GLEAM v4.3a and v4.3b monthly and yearly data (0.1 deg) from the GLEAM SFTP server.
#   https://www.gleam.eu
#
# Remote layout:  /data/<version>/<resolution>/<variable>/<variable>_<year>_GLEAM_<version>_<MO|YR>.nc
# Local layout:   $OUTPUT_ROOT/<version>/<resolution>/<variable>/...
# v4.3a covers 1980-2025 and v4.3b 2003-2025. Monthly files are ~90 MB per variable-year and
# yearly ~6 MB, so all 15 variables come to roughly 100 GB.
#
# Each directory is mirrored with lftp, which skips files that are already complete locally and
# resumes partial ones, so re-running is safe.
#
# Subset with space-separated env vars, e.g.
#   VERSIONS="v4.3a" RESOLUTIONS="monthly" VARIABLES="E Et Ep" ./download_gleam.sh
# Long runs: run inside tmux, or nohup ./download_gleam.sh > download_gleam.log 2>&1 &

set -uo pipefail

SFTP_HOST=aether.ugent.be
SFTP_PORT=2225
SFTP_USER=gleamuser
SFTP_PASS='GLEAM4#h-cel_111'
REMOTE_ROOT=/data
OUTPUT_ROOT=/glade/derecho/scratch/bbuchovecky/gleam-v4.3
PARALLEL=${PARALLEL:-4}  # files transferred at once within each mirror
LFTP=${LFTP:-lftp}

VERSIONS=${VERSIONS:-"v4.3a v4.3b"}
RESOLUTIONS=${RESOLUTIONS:-"monthly yearly"}
# VARIABLES=${VARIABLES:-"E Et Ec Es Ei Eb Ew Ep Ep_aero Ep_rad H S SMrz SMs"}
VARIABLES=${VARIABLES:-"E Et Ec Es Ei Eb Ew Ep"}

command -v "$LFTP" > /dev/null || {
    echo "lftp not found. Install it (e.g. conda create -n lftp -c conda-forge lftp)" \
         "and set LFTP=<env>/bin/lftp." >&2
    exit 1
}

# Run lftp commands ($1) in one SFTP session; the exit status is that of the last command.
# The credentials are quoted because lftp treats "#" as the start of a comment.
run_lftp() {
    "$LFTP" -c "
        set sftp:auto-confirm yes;
        set net:timeout 120;
        set net:max-retries 10;
        set net:reconnect-interval-base 30;
        open -u \"$SFTP_USER,$SFTP_PASS\" -p $SFTP_PORT sftp://$SFTP_HOST;
        $1"
}

mkdir -p "$OUTPUT_ROOT"
run_lftp "get -c -O $OUTPUT_ROOT $REMOTE_ROOT/README_GLEAM4.3.pdf $REMOTE_ROOT/Log_file_GLEAM4.3.pdf"

failed=()
for version in $VERSIONS; do
    for res in $RESOLUTIONS; do
        for var in $VARIABLES; do
            dir=$version/$res/$var
            echo "$(date '+%F %T')  $dir"
            run_lftp "mirror --continue --no-perms --parallel=$PARALLEL --verbose=1 \
                      $REMOTE_ROOT/$dir $OUTPUT_ROOT/$dir" || failed+=("$dir")
        done
    done
done

echo "$(date '+%F %T')  done: $(find "$OUTPUT_ROOT" -name '*.nc' | wc -l) .nc files," \
     "$(du -sh "$OUTPUT_ROOT" | cut -f1) in $OUTPUT_ROOT"
(( ${#failed[@]} == 0 )) || { echo "FAILED: ${failed[*]}; re-run to retry." >&2; exit 1; }
