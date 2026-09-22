set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/typed
python3 rl.py --model Qwen/Qwen3-0.6B --task typed --steps 600 --batch 8 --seed 2 --out adapters/typed-out-2
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter adapters/typed-out-2/Qwen3-0.6B-brier-rl-typed-s600 --task typed --split eval --n 400 --tag outcome-qwen0.6b-seed2 --out results/v2/typed
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/typed/DONE_typed-out-2
