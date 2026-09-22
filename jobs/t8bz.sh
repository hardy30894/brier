set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/typed
python3 -m brier calibrate --model Qwen/Qwen3-8B --task typed --split eval --n 400 --tag zeroshot-qwen8b --out results/v2/typed
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/typed/DONE_t8bz
