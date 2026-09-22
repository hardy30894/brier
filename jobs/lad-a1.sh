set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/ladder
G=$(nvidia-smi -L | wc -l)
STEPS=$(( 96000 / (4 * G) ))
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node $G train.py --model answerdotai/ModernBERT-large --tasks hub_mixed --examples 16500 --steps $STEPS --warmup 300 --accum 1 --batch 4 --split train --seed 1 --pointer --full --lr 2e-5 --head-lr 1e-4 --out adapters/lad-a1
A=$(ls -d adapters/lad-a1/ModernBERT*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task typed --split eval --n 400 --tag ladder-a-pool16500-seed1-typed --out results/v2/ladder
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task hub_trec,hub_anli,hub_commonsense_qa,hub_financial_news,hub_paws,hub_boolq,hub_sms_spam,hub_tweet_irony --split eval --n 300 --tag ladder-a-pool16500-seed1-held --out results/v2/ladder
mkdir -p results/logs && cp job.log results/logs/lad-a1.log 2>/dev/null || true
touch results/v2/ladder/DONE_lad-a1
