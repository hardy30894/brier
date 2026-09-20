set -x; cd /workspace/brier; mkdir -p results/v2/outcome
python3 rl.py --model answerdotai/ModernBERT-large --task toolcall --pointer --full --lr 3e-5 --head-lr 1e-4 --steps 600 --batch 8 --seed 1 --out adapters/enc-out-toolcall-1
A=$(ls -d adapters/enc-out-toolcall-1/*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task toolcall --split eval --n 300 --tag enc-outcome-toolcall-seed1 --out results/v2/outcome
python3 train.py --model answerdotai/ModernBERT-large --tasks toolcall --examples 1200 --steps 600 --warmup 60 --accum 4 --batch 2 --split train --uncertain 0.3 --seed 1 --pointer --full --lr 3e-5 --head-lr 1e-4 --out adapters/enc-sup-toolcall-1
B=$(ls -d adapters/enc-sup-toolcall-1/*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $B --task toolcall --split eval --n 300 --tag enc-supervised-toolcall-seed1 --out results/v2/outcome
touch results/v2/outcome/DONE_enc-out-toolcall-1
