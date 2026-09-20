set -x; cd /workspace/brier; mkdir -p results/v2/outcome
python3 rl.py --model answerdotai/ModernBERT-large --task tickets --pointer --full --lr 3e-5 --head-lr 1e-4 --steps 600 --batch 8 --seed 2 --out adapters/enc-out-tickets-2
A=$(ls -d adapters/enc-out-tickets-2/*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task tickets --split eval --n 300 --tag enc-outcome-tickets-seed2 --out results/v2/outcome
python3 train.py --model answerdotai/ModernBERT-large --tasks tickets --examples 1200 --steps 600 --warmup 60 --accum 4 --batch 2 --split train --uncertain 0.3 --seed 2 --pointer --full --lr 3e-5 --head-lr 1e-4 --out adapters/enc-sup-tickets-2
B=$(ls -d adapters/enc-sup-tickets-2/*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $B --task tickets --split eval --n 300 --tag enc-supervised-tickets-seed2 --out results/v2/outcome
touch results/v2/outcome/DONE_enc-out-tickets-2
