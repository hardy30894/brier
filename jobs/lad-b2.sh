set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/ladder
G=$(nvidia-smi -L | wc -l)
# Every run must take the same number of optimizer steps at the same effective batch, whatever
# hardware it landed on: otherwise a pod that got 8 cards and one that got 4 differ in training
# dynamics as well as in pool size, and the ladder measures two things at once.
STEPS=3000
ACCUM=$(( 32 / (G * 4) )); [ "$ACCUM" -lt 1 ] && ACCUM=1
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node $G train.py --model answerdotai/ModernBERT-large --tasks hub_mixed --examples 101490 --steps $STEPS --warmup 300 --accum $ACCUM --batch 4 --split train --seed 2 --pointer --full --lr 2e-5 --head-lr 1e-4 --out adapters/lad-b2
A=$(ls -d adapters/lad-b2/ModernBERT*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task typed --split eval --n 400 --tag ladder-b-pool101490-seed2-typed --out results/v2/ladder
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task hub_trec,hub_anli,hub_commonsense_qa,hub_financial_news,hub_paws,hub_boolq,hub_sms_spam,hub_tweet_irony --split eval --n 300 --tag ladder-b-pool101490-seed2-held --out results/v2/ladder
mkdir -p results/logs && cp job.log results/logs/lad-b2.log 2>/dev/null || true
touch results/v2/ladder/DONE_lad-b2
