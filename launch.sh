#!/bin/bash
# launch.sh <pod name> <gpu type> <ttl minutes> <job script under jobs/> : bring a pod up, sync, start the job detached
set -eo pipefail
cd "$(dirname "$0")"
N=$1; G=$2; T=$3; J=$4
python3 pod.py --name "$N" up --ttl "$T" --gpu "$G" 2>&1 | tail -1
python3 pod.py --name "$N" sync 2>&1 | tail -1
python3 pod.py --name "$N" exec "bash setup.sh > setup.log 2>&1; pip install -q scikit-learn > /dev/null 2>&1; nohup bash jobs/$J > job.log 2>&1 < /dev/null & sleep 3; pgrep -af 'bash jobs' | grep -v pgrep | wc -l"
echo "LAUNCHED $N $J"
