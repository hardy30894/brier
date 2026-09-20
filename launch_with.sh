#!/bin/bash
# launch_with.sh <pod> <gpu> <ttl> <job.sh> <path to push> : up, sync, push a local directory, run the job
set -eo pipefail
cd "$(dirname "$0")"
N=$1; G=$2; T=$3; J=$4; P=$5
python3 pod.py --name "$N" up --ttl "$T" --gpu "$G" 2>&1 | tail -1
python3 pod.py --name "$N" sync 2>&1 | tail -1
python3 pod.py --name "$N" push "$P" 2>&1 | grep -v "extended header" | tail -1
python3 pod.py --name "$N" exec "bash setup.sh > setup.log 2>&1; pip install -q scikit-learn > /dev/null 2>&1; nohup bash jobs/$J > job.log 2>&1 < /dev/null & sleep 3; pgrep -af '^bash jobs' | grep -v pgrep | wc -l"
echo "LAUNCHED $N $J"
