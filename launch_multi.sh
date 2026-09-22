#!/bin/bash
# launch_multi.sh <pod> <gpu> <count> <ttl> <job.sh> : a multi-GPU pod, synced, job started detached
set -eo pipefail
cd "$(dirname "$0")"
N=$1; G=$2; C=$3; T=$4; J=$5
python3 pod.py --name "$N" up --ttl "$T" --gpu "$G" --gpus "$C" 2>&1 | tail -1
python3 pod.py --name "$N" sync 2>&1 | tail -1
python3 pod.py --name "$N" exec "bash setup.sh > setup.log 2>&1; nohup bash jobs/$J > job.log 2>&1 < /dev/null & sleep 3; pgrep -af '^bash jobs' | grep -v pgrep | wc -l"
echo "LAUNCHED $N $J"
