set -x; cd /workspace/brier; mkdir -p results/baselines
python3 baselines.py --model Qwen/Qwen3-0.6B --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp --n 300 --out results/baselines
python3 baselines.py --model microsoft/Phi-4-mini-instruct --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp --n 300 --out results/baselines
touch results/baselines/DONE_base
