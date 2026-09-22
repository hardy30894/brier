set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/outcome
A=$(ls -d adapters/release/ModernBERT*/ | head -1)
python3 rl.py --model answerdotai/ModernBERT-large --init $A --task tickets --pointer --full --lr 1e-5 --head-lr 3e-5 --steps 600 --batch 8 --seed 1 --out adapters/warm-tickets-1
B=$(ls -d adapters/warm-tickets-1/*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $B --task tickets --split eval --n 300 --tag warm-outcome-tickets-seed1 --out results/v2/outcome
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task tickets --split eval --n 300 --tag warm-before-tickets-seed1 --out results/v2/outcome
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/outcome/DONE_warm-tickets-1
