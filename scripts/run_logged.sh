#!/usr/bin/env bash
# Run a pipeline script with its output (stdout and stderr) printed to the
# terminal and saved to a timestamped log, headed by the git state of the repo.
#
# Usage: scripts/run_logged.sh scripts/bin_obs.py
#
# Logs go to WORK_ROOT/logs (WORK_ROOT from etunc.config), or to $LOG_DIR if set:
#   <LOG_DIR>/<script name>.<YYYYmmdd-HHMMSS>.log

set -euo pipefail  # pipefail: the exit status is the script's, not tee's

PY=/glade/work/bbuchovecky/miniforge3/envs/etunc/bin/python

if [[ $# -ne 1 || ! -f $1 ]]; then
    echo "Usage: $0 <script.py>" >&2
    exit 2
fi
script=$1
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)

LOG_DIR=${LOG_DIR:-$("$PY" -c "import etunc.config as c; print(c.WORK_ROOT / 'logs')")}
mkdir -p "$LOG_DIR"
log=$LOG_DIR/$(basename "$script" .py).$(date +%Y%m%d-%H%M%S).log

# Header: what was run, where, and from which code
{
    echo "######### run_logged.sh"
    echo "script  : $script"
    echo "start   : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "host    : $(hostname)"
    echo "python  : $PY"
    echo "commit  : $(git -C "$repo" rev-parse HEAD) ($(git -C "$repo" rev-parse --abbrev-ref HEAD))"
    echo "#########"
    echo
} > "$log"

echo "Logging to $log"
start=$SECONDS
status=0
# -u: unbuffered, so lines reach the terminal and log as they are printed
"$PY" -u "$script" 2>&1 | tee -a "$log" || status=$?

echo -e "\n######### exit status $status after $(( (SECONDS - start) / 60 )) min, $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$log"
exit $status
