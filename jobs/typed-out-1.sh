set -x; cd /workspace/brier; mkdir -p results/v2/typed
python3 rl.py --model Qwen/Qwen3-0.6B --task typed --steps 600 --batch 8 --seed 1 --out adapters/typed-out-1
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter adapters/typed-out-1/Qwen3-0.6B-brier-rl-typed-s600 --task typed --split eval --n 400 --tag outcome-qwen0.6b-seed1 --out results/v2/typed
touch results/v2/typed/DONE_typed-out-1
