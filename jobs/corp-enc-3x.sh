set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/corpus
python3 train.py --model answerdotai/ModernBERT-large --tasks hub_mixed --examples 16500 --steps 6000 --warmup 600 --split train --seed 1 --pointer --full --lr 2e-5 --head-lr 1e-4 --out adapters/corp-enc-3x
A=$(ls -d adapters/corp-enc-3x/ModernBERT*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task hub_trec,hub_anli,hub_commonsense_qa,hub_financial_news,hub_paws,hub_boolq,hub_sms_spam,hub_tweet_irony --split eval --n 300 --tag corp-enc-3x-heldout --out results/v2/corpus
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task typed --split eval --n 400 --tag corp-enc-3x-typed-zeroshot --out results/v2/corpus
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task frustration,duplicates,tickets,banking --split eval --n 300 --tag corp-enc-3x-ours --out results/v2/corpus
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/corpus/DONE_corp-enc-3x
