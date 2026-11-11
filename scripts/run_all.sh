#!/usr/bin/env bash
# Run the whole binning pipeline of docs/binning-workflow.md, after the data are
# downloaded (scripts/download/). Each script runs through scripts/run_logged.sh,
# so each step has its own log (with git header) in WORK_ROOT/logs or $LOG_DIR.
# Stops at the first step that fails.
#
# Usage: scripts/run_all.sh          (or: pixi run run-all, qsub scripts/run_all.pbs)
#
# Settings, as environment variables:
#   RUNS="a.yml b.yml"    run files (config/runs/<name>.yml, paths from the repo), run one after the other;
#                         "default" is the defaults of config/pipeline.yml; unset or empty: "default"
#   SKIP_PREP=1           start at make_edges.py, with the inputs of each run's PREP_ID already saved
# Each run's outputs are in WORK_ROOT/{proc,fig}/<RUN_ID>/, its prep inputs in WORK_ROOT/proc/<PREP_ID>/.
# The prep scripts run once per PREP_ID in this invocation: runs that differ only in their binning
# settings share the prep inputs.
# e.g. RUNS=config/runs/bins20.yml scripts/run_all.sh
#      RUNS="default config/runs/bins20.yml config/runs/aw_bins15.yml" scripts/run_all.sh
#      SKIP_PREP=1 RUNS=config/runs/bins10.yml scripts/run_all.sh
#
# Not run here:
#   - scripts/download/                          the data downloads
#   - regress_lai_aridity.py,                    analyses of the step 14 outputs, not part
#     decompose_state_function.py                of the pipeline (run_decomp.sh runs the latter
#                                                for several runs)
#   - make_mask.py, diagnose_obs.py,             diagnostics, no pipeline step reads them
#     compare_rn.py

set -euo pipefail

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$repo"

# Python of the repo's pixi env, to read each run's IDs (as in run_logged.sh)
PIXI=${PIXI:-$(command -v pixi || echo "$HOME/.pixi/bin/pixi")}
PY=("$PIXI" run --manifest-path "$repo/pyproject.toml" python)

# Runs: the run files of RUNS ("default": the defaults)
read -r -a runs <<< "${RUNS:-default}"
[[ ${#runs[@]} -gt 0 ]] || runs=(default)

# Steps 1-11: inputs of each family
prep_steps=(
    scripts/prep_obs.py
    scripts/prep_era5.py
    scripts/prep_cmip.py
    scripts/prep_lens.py
    scripts/prep_fppe.py
    scripts/prep_ippe.py
    scripts/prep_trendy.py
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
for r in "${runs[@]}"; do
    [[ $r == default || -f $r ]] || { echo "No such run file: $r" >&2; exit 2; }
done

# Run one script through run_logged.sh, with the ETUNC_RUN it inherits
n_run=0
run_step() {
    local s=$1 t0=$SECONDS
    n_run=$((n_run + 1))
    echo -e "\n========= [$n_run] $s ($(date '+%Y-%m-%d %H:%M:%S')), run ${ETUNC_RUN:-defaults}"
    scripts/run_logged.sh "$s"
    echo "========= $s done in $(( (SECONDS - t0) / 60 )) min"
}

start=$SECONDS
declare -A prepped  # PREP_IDs whose prep scripts ran in this invocation
for r in "${runs[@]}"; do
    if [[ $r == default ]]; then export ETUNC_RUN=""; else export ETUNC_RUN=$r; fi
    read -r prep_id run_id < <("${PY[@]}" -c "import etunc.config as c; print(c.PREP_ID, c.RUN_ID)")
    [[ -n ${prep_id:-} ]] || { echo "Could not read the IDs of run $r" >&2; exit 1; }
    echo -e "\n######### run $r: RUN_ID $run_id, PREP_ID $prep_id"
    if [[ ${SKIP_PREP:-0} == 1 ]]; then
        echo "SKIP_PREP=1: the inputs of $prep_id are read as saved"
    elif [[ -n ${prepped[$prep_id]:-} ]]; then
        echo "prep $prep_id already run in this invocation"
    else
        for s in "${prep_steps[@]}"; do run_step "$s"; done
        prepped[$prep_id]=1
    fi
    for s in "${bin_steps[@]}"; do run_step "$s"; done
done

echo -e "\n========= all $n_run steps of ${#runs[@]} runs done in $(( (SECONDS - start) / 60 )) min"
