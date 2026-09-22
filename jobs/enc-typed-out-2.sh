set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/typed
python3 rl.py --model answerdotai/ModernBERT-large --task typed --pointer --full --lr 3e-5 --head-lr 1e-4 --steps 600 --batch 8 --seed 2 --out adapters/enc-typed-out-2
A=$(ls -d adapters/enc-typed-out-2/*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task typed --split eval --n 400 --tag outcome-encoder-seed2 --out results/v2/typed
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/typed/DONE_enc-typed-out-2
