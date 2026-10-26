#!/usr/bin/env bash
# Run the whole binning pipeline of docs/binning-workflow.md, after the data are
# downloaded (scripts/download/). Each script runs through scripts/run_logged.sh,
# so each step has its own log (with git header) in WORK_ROOT/logs or $LOG_DIR.
# Stops at the first step that fails.
#
# Usage: scripts/run_all.sh          (or: pixi run run-all, qsub scripts/run_all.pbs)
#
# Settings, as environment variables (unset: the defaults of etunc/config.py):
#   ETUNC_BIN_VARIABLE    et | et_pr            variable binned in step 14
#   ETUNC_AREA_WEIGHTED   1 | 0                 land-area weighted bin statistics
#   ETUNC_N_XBINS         e.g. 10               number of AI bins
#   ETUNC_N_YBINS         e.g. 10               number of LAI bins
#   SKIP_PREP=1           start at make_edges.py, with the inputs already saved by the prep scripts
#   SWEEP=1               every combination of SWEEP_* below (the ETUNC_* settings are ignored): the prep
#                         scripts once with et_pr (they save et too), make_edges.py and bin_area.py once per
#                         bin count, bin_plumber*.py once per bin count and variable, bin_all*.py for each
#                         combination
# e.g. ETUNC_BIN_VARIABLE=et_pr ETUNC_N_XBINS=20 ETUNC_N_YBINS=20 scripts/run_all.sh
#      SWEEP=1 SKIP_PREP=1 scripts/run_all.sh
#
# Not run here:
#   - scripts/download/                       the data downloads
#   - regress_lai_aridity.py,                 analyses of the step 14 outputs, not part
#                                             of the pipeline
#   - make_mask.py, obs_et_availability.py    diagnostics, no pipeline step reads them

set -euo pipefail

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$repo"

# Combinations run with SWEEP=1 (N_XBINS = N_YBINS)
SWEEP_BIN_VARIABLES=(et et_pr)
SWEEP_AREA_WEIGHTED=(0 1)
SWEEP_N_BINS=(10 15 20)

# Steps 1-11: inputs of each family
prep_steps=(
    scripts/prep_obs.py
    scripts/prep_era5.py
    scripts/prep_cmip.py
    scripts/prep_lens.py
    scripts/prep_fppe.py
    scripts/prep_ippe.py
)
# Steps 12-14: common area mask and pooled edges, then binning (bin_plumber_clim.py
# reads bin_plumber.py's inputs)
bin_steps=(
    scripts/make_edges.py
    scripts/bin_all.py
    scripts/bin_all_clim.py
    scripts/bin_area.py
    scripts/bin_plumber.py
    scripts/bin_plumber_clim.py
)

# Check every script exists before starting, so a typo fails now and not hours in
for s in "${prep_steps[@]}" "${bin_steps[@]}"; do
    [[ -f $s ]] || { echo "No such script: $s" >&2; exit 2; }
done

# Run one script through run_logged.sh, with the ETUNC_* settings it inherits
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
if [[ ${SWEEP:-0} != 1 ]]; then
    # One combination, the ETUNC_* settings (or the defaults)
    if [[ ${SKIP_PREP:-0} != 1 ]]; then
        for s in "${prep_steps[@]}"; do run_step "$s"; done
    fi
    for s in "${bin_steps[@]}"; do run_step "$s"; done
else
    # Every combination. et_pr makes the prep scripts save ET/P in addition to ET
    if [[ ${SKIP_PREP:-0} != 1 ]]; then
        for s in "${prep_steps[@]}"; do ETUNC_BIN_VARIABLE=et_pr run_step "$s"; done
    fi
    for n in "${SWEEP_N_BINS[@]}"; do
        export ETUNC_N_XBINS=$n ETUNC_N_YBINS=$n
        unset ETUNC_BIN_VARIABLE ETUNC_AREA_WEIGHTED
        # Neither depends on the variable or the weighting
        run_step scripts/make_edges.py
        run_step scripts/bin_area.py
        for v in "${SWEEP_BIN_VARIABLES[@]}"; do
            export ETUNC_BIN_VARIABLE=$v
            unset ETUNC_AREA_WEIGHTED
            # Towers are not area weighted
            run_step scripts/bin_plumber.py
            run_step scripts/bin_plumber_clim.py
            for w in "${SWEEP_AREA_WEIGHTED[@]}"; do
                export ETUNC_AREA_WEIGHTED=$w
                run_step scripts/bin_all.py
                run_step scripts/bin_all_clim.py
            done
        done
    done
fi

echo -e "\n========= all $n_run steps done in $(( (SECONDS - start) / 60 )) min"
