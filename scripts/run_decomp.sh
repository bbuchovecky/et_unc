#!/usr/bin/env bash
# Run the climatological state, function and interaction decomposition of the
# area-mean ET (scripts/decompose_state_function.py) for several runs, e.g. one per
# bin count. Each run goes through scripts/run_logged.sh, so it has its own log (with
# git header) in WORK_ROOT/logs or $LOG_DIR. Stops at the first run that fails.
#
# Usage: scripts/run_decomp.sh          (or: pixi run run-decomp)
#
# Settings, as environment variables:
#   RUNS    run files (config/runs/<name>.yml), separated by spaces
#           (default: config/runs/aw_bins10.yml, aw_bins15.yml and aw_bins20.yml)
# e.g. RUNS="config/runs/aw_bins15.yml config/runs/aw_bins20.yml" scripts/run_decomp.sh
#
# Every run must have binning.area_weighted: true and bin_variable: et, and its
# pooled_all clim ET bins already made (bin_all_clim.py in that run, e.g. from
# RUNS=... scripts/run_all.sh).
#
# Each run's comparison 6 (bin-count sensitivity) reads the decompositions of every
# registered run (config/runs.csv) that differs from it only in the bin counts, so the
# summary of the last run covers them all.

set -euo pipefail

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$repo"

read -r -a runs <<< "${RUNS:-config/runs/aw_bins10.yml config/runs/aw_bins15.yml config/runs/aw_bins20.yml}"
script=scripts/decompose_state_function.py
[[ -f $script ]] || { echo "No such script: $script" >&2; exit 2; }
for r in "${runs[@]}"; do
    [[ -f $r ]] || { echo "No such run file: $r" >&2; exit 2; }
done

# Run the decomposition through run_logged.sh, with the ETUNC_RUN it inherits
n_run=0
run_step() {
    local s=$1 t0=$SECONDS
    n_run=$((n_run + 1))
    echo -e "\n========= [$n_run] $s ($(date '+%Y-%m-%d %H:%M:%S')), run $ETUNC_RUN"
    scripts/run_logged.sh "$s"
    echo "========= $s done in $(( (SECONDS - t0) / 60 )) min"
}

start=$SECONDS
for r in "${runs[@]}"; do
    export ETUNC_RUN=$r
    run_step "$script"
done

echo -e "\n========= all $n_run runs (${runs[*]}) done in $(( (SECONDS - start) / 60 )) min"
