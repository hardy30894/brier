set -x; cd /workspace/brier; mkdir -p results/v2/typed
python3 train.py --model answerdotai/ModernBERT-large --tasks typed --examples 1200 --steps 1200 --warmup 120 --accum 4 --batch 1 --split train --seed 2 --pointer --full --lr 3e-5 --head-lr 1e-4 --out adapters/tenc-2
A=$(ls -d adapters/tenc-2/*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task typed --split eval --n 400 --tag ft-encoder-seed2 --out results/v2/typed
touch results/v2/typed/DONE_tenc-2
