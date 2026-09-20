set -x; cd /workspace/brier; mkdir -p results/v2/typed
A=$(ls -d adapters/release/ModernBERT*/ | head -1)
python3 rl.py --model answerdotai/ModernBERT-large --init $A --task typed --pointer --full --lr 1e-5 --head-lr 3e-5 --steps 600 --batch 8 --seed 3 --out adapters/warm-typed-3
B=$(ls -d adapters/warm-typed-3/*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $B --task typed --split eval --n 400 --tag warm-outcome-encoder-seed3 --out results/v2/typed
touch results/v2/typed/DONE_warm-typed-3
