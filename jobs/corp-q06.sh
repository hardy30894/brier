set -x; cd /workspace/brier; mkdir -p results/v2/corpus
python3 train.py --model Qwen/Qwen3-0.6B --tasks hub_mixed --examples 16500 --steps 3000 --split train --seed 1 --pointer --out adapters/corp-q06
A=$(ls -d adapters/corp-q06/Qwen3-0.6B*/ | head -1)
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter $A --task hub_trec,hub_anli,hub_commonsense_qa,hub_financial_news,hub_paws,hub_boolq,hub_sms_spam,hub_tweet_irony --split eval --n 300 --tag corp-q06-heldout --out results/v2/corpus
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter $A --task typed --split eval --n 400 --tag corp-q06-typed-zeroshot --out results/v2/corpus
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter $A --task frustration,duplicates,tickets,banking --split eval --n 300 --tag corp-q06-ours --out results/v2/corpus
touch results/v2/corpus/DONE_corp-q06
