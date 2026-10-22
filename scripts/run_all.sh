#!/usr/bin/env bash
# Run the whole binning pipeline of docs/binning-workflow.md, after the data are
# downloaded (scripts/download/). Each script runs through scripts/run_logged.sh,
# so each step has its own log (with git header) in WORK_ROOT/logs or $LOG_DIR.
# Stops at the first step that fails.
#
# Usage: scripts/run_all.sh          (or: pixi run run-all, qsub scripts/run_all.pbs)
#
# Not run here:
#   - scripts/download/                       the data downloads
#   - regress_lai_aridity.py,                 analyses of the step 14 outputs, not part
#                                             of the pipeline
#   - make_mask.py, obs_et_availability.py    diagnostics, no pipeline step reads them

set -euo pipefail

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$repo"

steps=(
    # Steps 1-11: inputs of each family
    scripts/prep_obs.py
    scripts/prep_era5.py
    scripts/prep_cmip.py
    scripts/prep_lens.py
    scripts/prep_fppe.py
    scripts/prep_ippe.py
    # Steps 12-13: common area mask and pooled edges
    scripts/make_edges.py
    # Step 14: binning (bin_plumber_clim.py reads bin_plumber.py's inputs)
    scripts/bin_all.py
    scripts/bin_all_clim.py
    scripts/bin_area.py
    scripts/bin_plumber.py
    scripts/bin_plumber_clim.py
)

# Check every script exists before starting, so a typo fails now and not hours in
for s in "${steps[@]}"; do
    [[ -f $s ]] || { echo "No such script: $s" >&2; exit 2; }
done

start=$SECONDS
for i in "${!steps[@]}"; do
    s=${steps[$i]}
    t0=$SECONDS
    echo -e "\n========= [$((i + 1))/${#steps[@]}] $s ($(date '+%Y-%m-%d %H:%M:%S'))"
    scripts/run_logged.sh "$s"
    echo "========= $s done in $(( (SECONDS - t0) / 60 )) min"
done

echo -e "\n========= all ${#steps[@]} steps done in $(( (SECONDS - start) / 60 )) min"
