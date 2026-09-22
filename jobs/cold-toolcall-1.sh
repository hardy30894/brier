set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/outcome
A=$(ls -d adapters/corpus-general/ModernBERT*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task toolcall --split eval --n 300 --tag cold-before-toolcall-seed1 --out results/v2/outcome
python3 rl.py --model answerdotai/ModernBERT-large --init $A --task toolcall --pointer --full --lr 1e-5 --head-lr 3e-5 --steps 600 --batch 8 --seed 1 --out adapters/cold-toolcall-1
B=$(ls -d adapters/cold-toolcall-1/*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $B --task toolcall --split eval --n 300 --tag cold-outcome-toolcall-seed1 --out results/v2/outcome
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/outcome/DONE_cold-toolcall-1
