set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/corpus results/v2/typed
python3 train.py --model answerdotai/ModernBERT-large --tasks hub_mixed --examples 16500 --steps 6000 --warmup 600 --split train --seed 1 --pointer --full --lr 2e-5 --head-lr 1e-4 --out adapters/gen-1
A=$(ls -d adapters/gen-1/ModernBERT*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task typed --split eval --n 400 --tag general63-zeroshot-seed1 --out results/v2/typed
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task hub_trec,hub_anli,hub_commonsense_qa,hub_financial_news,hub_paws,hub_boolq,hub_sms_spam,hub_tweet_irony --split eval --n 300 --tag general63-heldout-seed1 --out results/v2/corpus
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task frustration,duplicates,tickets,banking --split eval --n 300 --tag general63-ours-seed1 --out results/v2/corpus
python3 rl.py --model answerdotai/ModernBERT-large --init $A --task typed --pointer --full --lr 1e-5 --head-lr 3e-5 --steps 600 --batch 8 --seed 1 --out adapters/gen-out-1
B=$(ls -d adapters/gen-out-1/*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $B --task typed --split eval --n 400 --tag general63-outcome-seed1 --out results/v2/typed
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/corpus/DONE_gen-1
