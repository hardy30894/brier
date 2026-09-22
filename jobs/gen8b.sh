set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/general
M=Qwen/Qwen3-8B
A=$(ls -d adapters/t8b-ptr/Qwen3-8B*/ | head -1)
HELD="hub_trec,hub_anli,hub_commonsense_qa,hub_financial_news,hub_paws,hub_boolq,hub_sms_spam,hub_tweet_irony"
OURS="frustration,duplicates,tickets,banking,mnli,imdb,emotion,snli,sst5,yelp,dbpedia,civil,msmarco"
python3 -m brier calibrate --model $M --task $HELD --split eval --n 300 --tag base8b-held --out results/v2/general
python3 -m brier calibrate --model $M --adapter $A --task $HELD --split eval --n 300 --tag tuned8b-held --out results/v2/general
python3 -m brier calibrate --model $M --task $OURS --split eval --n 300 --tag base8b-ours --out results/v2/general
python3 -m brier calibrate --model $M --adapter $A --task $OURS --split eval --n 300 --tag tuned8b-ours --out results/v2/general
mkdir -p results/logs && cp job.log results/logs/gen8b.log 2>/dev/null || true
touch results/v2/general/DONE_gen8b
