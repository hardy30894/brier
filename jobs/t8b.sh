set -x; cd /workspace/brier; mkdir -p results/v2/typed
python3 train.py --model Qwen/Qwen3-8B --tasks typed --examples 1200 --steps 600 --accum 4 --batch 1 --split train --seed 1 --out adapters/t8b
python3 -m brier calibrate --model Qwen/Qwen3-8B --adapter adapters/t8b/Qwen3-8B-brier-r16-s600-train --task typed --split eval --n 400 --tag ft-qwen8b-seed1 --out results/v2/typed
touch results/v2/typed/DONE_t8b
