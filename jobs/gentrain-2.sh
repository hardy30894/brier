set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/gen
G=$(nvidia-smi -L | wc -l)
STEPS=3000
ACCUM=$(( 32 / (G * 4) )); [ "$ACCUM" -lt 1 ] && ACCUM=1
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --master_port $((20000 + RANDOM % 20000)) --nproc_per_node $G train.py --model answerdotai/ModernBERT-large --tasks schema --examples 96000 --steps $STEPS --warmup 300 --accum $ACCUM --batch 4 --split train --seed 2 --pointer --full --lr 2e-5 --head-lr 1e-4 --out adapters/gentrain-2
A=$(ls -d adapters/gentrain-2/ModernBERT*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task typed --split eval --n 400 --tag gen-only-seed2-typed --out results/v2/gen
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task hub_trec,hub_anli,hub_commonsense_qa,hub_financial_news,hub_paws,hub_boolq,hub_sms_spam,hub_tweet_irony --split eval --n 300 --tag gen-only-seed2-held --out results/v2/gen
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task schema --split eval --n 400 --tag gen-only-seed2-schema --out results/v2/gen
mkdir -p results/logs && cp job.log results/logs/gentrain-2.log 2>/dev/null || true
touch results/v2/gen/DONE_gentrain-2
