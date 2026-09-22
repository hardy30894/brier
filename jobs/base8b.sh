set -exo pipefail; cd /workspace/brier; mkdir -p results/baselines
python3 baselines.py --model Qwen/Qwen3-8B --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp --n 300 --out results/baselines
touch results/baselines/DONE_base8b
