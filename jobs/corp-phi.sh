set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/corpus results/v2/typed
# base zero-shot already measured (0.606)
python3 train.py --model microsoft/Phi-4-mini-instruct --tasks hub_mixed --examples 16500 --steps 3000 --accum 4 --batch 1 --split train --seed 1 --pointer --pointer-only --out adapters/corp-phi
A=$(ls -d adapters/corp-phi/Phi-4-mini*/ | head -1)
python3 -m brier calibrate --model microsoft/Phi-4-mini-instruct --adapter $A --task typed --split eval --n 400 --pointer-grade --tag corp-phi-zeroshot --out results/v2/typed
python3 -m brier calibrate --model microsoft/Phi-4-mini-instruct --adapter $A --task hub_trec,hub_anli,hub_commonsense_qa,hub_financial_news,hub_paws,hub_boolq,hub_sms_spam,hub_tweet_irony --split eval --n 300 --pointer-grade --tag corp-phi-heldout --out results/v2/corpus
python3 -m brier calibrate --model microsoft/Phi-4-mini-instruct --adapter $A --task frustration,duplicates,tickets,banking --split eval --n 300 --pointer-grade --tag corp-phi-ours --out results/v2/corpus
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/corpus/DONE_corp-phi
