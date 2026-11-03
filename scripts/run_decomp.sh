#!/usr/bin/env bash
# Run the climatological state, function and interaction decomposition of the
# area-mean ET (scripts/decompose_state_function.py) for each bin count. Each run
# goes through scripts/run_logged.sh, so it has its own log (with git header) in
# WORK_ROOT/logs or $LOG_DIR. Stops at the first run that fails.
#
# Usage: scripts/run_decomp.sh          (or: pixi run run-decomp)
#
# Settings, as environment variables:
#   N_BINS    e.g. "10 15 20"    bin counts, N_XBINS = N_YBINS (default: those of run_all.sh's SWEEP)
# e.g. N_BINS="15 20" scripts/run_decomp.sh
#
# Needs, for every bin count, the land-area weighted climatological ET binned with
# the pooled_all edges (bin_all_clim.py with ETUNC_AREA_WEIGHTED=1), e.g. from
# SWEEP=1 scripts/run_all.sh. The decomposition pins its own BIN_VARIABLE (et) and
# AREA_WEIGHTED (1), so only the bin counts are passed.
#
# Each run's comparison 6 (bin-count sensitivity) reads the outputs of every bin
# count already run, so the summary of the last run covers them all.

set -euo pipefail

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$repo"

read -r -a n_bins <<< "${N_BINS:-10 15 20}"
script=scripts/decompose_state_function.py
[[ -f $script ]] || { echo "No such script: $script" >&2; exit 2; }

# Run the decomposition through run_logged.sh, with the ETUNC_* settings it inherits
n_run=0
run_step() {
    local s=$1 t0=$SECONDS
    n_run=$((n_run + 1))
    echo -e "\n========= [$n_run] $s ($(date '+%Y-%m-%d %H:%M:%S'))"
    env | grep '^ETUNC_' | sort | sed 's/^/          /' || true
    scripts/run_logged.sh "$s"
    echo "========= $s done in $(( (SECONDS - t0) / 60 )) min"
}

start=$SECONDS
for n in "${n_bins[@]}"; do
    export ETUNC_N_XBINS=$n ETUNC_N_YBINS=$n
    run_step "$script"
done

echo -e "\n========= all $n_run runs (bin counts: ${n_bins[*]}) done in $(( (SECONDS - start) / 60 )) min"
