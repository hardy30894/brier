set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/typed

python3 train.py --model Qwen/Qwen3-0.6B --tasks typed --examples 1200 --steps 1200 --accum 4 --batch 1 --split train --seed 2 --out adapters/t06-2
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter adapters/t06-2/Qwen3-0.6B-brier-r16-s1200-train --task typed --split eval --n 400 --tag ft-qwen0.6b-seed2 --out results/v2/typed
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/typed/DONE_t06-2
