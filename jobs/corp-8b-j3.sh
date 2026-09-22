set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/corpus results/v2/typed
python3 train.py --model Qwen/Qwen3-8B --tasks hub_mixed --examples 16500 --steps 4200 --accum 4 --batch 1 --split train --seed 1 --pointer --max-candidates 16 --out adapters/corp-8b-j3
A=$(ls -d adapters/corp-8b-j3/Qwen3-8B*/ | head -1)
python3 -m brier calibrate --model Qwen/Qwen3-8B --adapter $A --task typed --split eval --n 400 --tag corp-8b-joint-zeroshot --out results/v2/typed
python3 -m brier calibrate --model Qwen/Qwen3-8B --adapter $A --task typed --split eval --n 400 --pointer-grade --tag corp-8b-joint-zeroshot-ptr --out results/v2/typed
python3 -m brier calibrate --model Qwen/Qwen3-8B --adapter $A --task hub_trec,hub_anli,hub_commonsense_qa,hub_financial_news,hub_paws,hub_boolq,hub_sms_spam,hub_tweet_irony --split eval --n 300 --tag corp-8b-joint-heldout --out results/v2/corpus
python3 -m brier calibrate --model Qwen/Qwen3-8B --adapter $A --task frustration,duplicates,tickets,banking --split eval --n 300 --tag corp-8b-joint-ours --out results/v2/corpus
mkdir -p results/logs && cp job.log results/logs/corp-8b-j3.log 2>/dev/null || true
touch results/v2/corpus/DONE_corp-8b-j3
