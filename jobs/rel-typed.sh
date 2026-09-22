set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/release
python3 train.py --model answerdotai/ModernBERT-large --tasks typed --examples 1200 --steps 1200 --warmup 120 --accum 4 --batch 1 --split train --seed 1 --pointer --full --lr 3e-5 --head-lr 1e-4 --out adapters/release/enc-typed
python3 train.py --model Qwen/Qwen3-0.6B --tasks typed --examples 1200 --steps 1200 --accum 4 --batch 1 --split train --seed 1 --pointer --out adapters/release/ptr-typed
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/release/DONE_rel-typed
