set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/corpus
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --task hub_trec,hub_anli,hub_commonsense_qa,hub_financial_news,hub_paws,hub_boolq,hub_sms_spam,hub_tweet_irony --split eval --n 300 --tag base-qwen0.6b-heldout --out results/v2/corpus
python3 -m brier calibrate --model microsoft/Phi-4-mini-instruct --task hub_trec,hub_anli,hub_commonsense_qa,hub_financial_news,hub_paws,hub_boolq,hub_sms_spam,hub_tweet_irony --split eval --n 300 --tag base-phi4mini-heldout --out results/v2/corpus
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/corpus/DONE_corp-base
