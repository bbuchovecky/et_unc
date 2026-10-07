#!/usr/bin/env bash
# Download SiTH v2 global ET components and soil moisture (1982-2022) from the TPDC FTP server.
#
# Remote layout:  Monthly/<variable>.SiTHv2.A1982_2022.M.nc
#                 Yearly/<variable>.SiTHv2.A1982_2022.Y.nc
#                 Daily/<year>/<variable>.SiTHv2.A<year>.nc
# Local layout:   $OUTPUT_ROOT/<same as remote>
# Each variable is one large file: Monthly is 8.6 GB in all, Yearly 1.1 GB and Daily 175 GB.
# SM alone is 40% of Monthly and 60% of Daily.
#
# The server gives ~50 KB/s per connection and allows 5 connections per IP to each host; beyond
# that it refuses with "421 Too many connections (5) from this IP". The limit is shared with every
# other TPDC download from the same machine (e.g. download_pml.sh), so lower CONN_PER_HOST when
# running those at the same time. The variables are split between the two hosts, and each host
# gets one lftp mirror capped at CONN_PER_HOST connections: its control connection plus SEGMENTS
# pieces of each of PARALLEL files downloading in parallel (2 x 3 data connections, ~300 KB/s
# by default). lftp skips files that are already complete locally and resumes partial ones, so
# re-running is safe. If a mirror fails on one host it is retried on the other.
#
# Subset with space-separated env vars, e.g.
#   RESOLUTIONS="Daily" YEARS="2001 2002" VARIABLES="ET Tr" ./download_sith.sh
# Long runs: run inside tmux, or nohup ./download_sith.sh > download_sith.log 2>&1 &

set -uo pipefail

FTP_HOSTS=(ftp2.tpdc.ac.cn ftp3.tpdc.ac.cn)
FTP_PORT=6201
FTP_USER=download_55713989
FTP_PASS=48344443
OUTPUT_ROOT=/glade/derecho/scratch/bbuchovecky/sith-v2
CONN_PER_HOST=${CONN_PER_HOST:-4}            # must stay below the server's limit of 5 per IP per host
PARALLEL=${PARALLEL:-1}                      # files transferred at once on each host
SEGMENTS=${SEGMENTS:-$(( CONN_PER_HOST - 1 ))}  # connections per file; PARALLEL x SEGMENTS < CONN_PER_HOST
LFTP=${LFTP:-/glade/work/bbuchovecky/miniforge3/envs/lftp/bin/lftp}

# RESOLUTIONS=${RESOLUTIONS:-"Monthly Yearly"}  # Daily is opt-in (175 GB)
RESOLUTIONS=${RESOLUTIONS:-"Monthly"}  # Daily is opt-in (175 GB)
YEARS=${YEARS:-$(seq -s ' ' 1982 2022)}       # Daily only
# VARIABLES=${VARIABLES:-"ET Ei En Es Tr SM"}
VARIABLES=${VARIABLES:-"ET Ei En Es Tr"}

command -v "$LFTP" > /dev/null || {
    echo "lftp not found. Install it (e.g. conda create -n lftp -c conda-forge lftp)" \
         "and set LFTP=<env>/bin/lftp." >&2
    exit 1
}

# Run lftp commands ($2) in one FTP session on host $1, opening at most CONN_PER_HOST connections
# (extra pget pieces wait for a free one); the exit status is that of the last command.
run_lftp() {
    "$LFTP" -c "
        set net:connection-limit $CONN_PER_HOST;
        set net:timeout 120;
        set net:max-retries 30;
        set net:reconnect-interval-base 30;
        set net:reconnect-interval-max 300;
        open -u \"$FTP_USER,$FTP_PASS\" -p $FTP_PORT ftp://$1;
        $2"
}

# Mirror directory $2 from host $1, only the variables in $3. The ".SiTHv2." suffix keeps "E*"
# from matching "Ei", "Es", ...
mirror_vars() {
    local include="" var
    for var in $3; do
        include+=" -I '$var.SiTHv2.*'"
    done
    run_lftp "$1" "mirror --continue --no-perms --parallel=$PARALLEL --use-pget-n=$SEGMENTS \
                   --verbose=1 $include $2 $OUTPUT_ROOT/$2"
}

# Deal the variables out to the hosts round-robin
nhosts=${#FTP_HOSTS[@]}
host_vars=()
i=0
for var in $VARIABLES; do
    h=$(( i++ % nhosts ))
    host_vars[h]="${host_vars[h]:+${host_vars[h]} }$var"
done

dirs=()
for res in $RESOLUTIONS; do
    if [[ $res == Daily ]]; then
        for year in $YEARS; do dirs+=("Daily/$year"); done
    else
        dirs+=("$res")
    fi
done

mkdir -p "$OUTPUT_ROOT"
failed=()
for dir in "${dirs[@]}"; do
    echo "$(date '+%F %T')  $dir"
    # One mirror per host at the same time, each on its own share of the variables
    pids=()
    for h in "${!host_vars[@]}"; do
        mirror_vars "${FTP_HOSTS[h]}" "$dir" "${host_vars[h]}" &
        pids[h]=$!
    done
    failed_h=()
    for h in "${!pids[@]}"; do
        wait "${pids[h]}" || failed_h+=("$h")
    done
    # Retry each failed share on the next host, one at a time so no host exceeds CONN_PER_HOST
    for h in ${failed_h[@]+"${failed_h[@]}"}; do
        other=${FTP_HOSTS[(h + 1) % nhosts]}
        echo "WARNING: $dir (${host_vars[h]}) failed on ${FTP_HOSTS[h]}; retrying on $other" >&2
        mirror_vars "$other" "$dir" "${host_vars[h]}" || failed+=("$dir (${host_vars[h]})")
    done
done

echo "$(date '+%F %T')  done: $(find "$OUTPUT_ROOT" -name '*.nc' | wc -l) .nc files," \
     "$(du -sh "$OUTPUT_ROOT" | cut -f1) in $OUTPUT_ROOT"
(( ${#failed[@]} == 0 )) || { echo "FAILED: ${failed[*]}; re-run to retry." >&2; exit 1; }
