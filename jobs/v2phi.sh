set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/seeds
M=microsoft/Phi-4-mini-instruct
python3 -m brier calibrate --model $M --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp,dbpedia,civil --split eval --n 300 --tag phi-untrained --out results/v2/seeds
python3 train.py --model $M --tasks tickets,gridworld,toolcall,passages,matching,ag_news,dbpedia,sst5,yelp,snli,qqp,civil,msmarco,massive,banking --examples 600 --steps 2000 --split train --uncertain 0.3 --batch 1 --accum 4 --seed 100 --out adapters/v2phi
python3 -m brier calibrate --model $M --adapter adapters/v2phi/Phi-4-mini-instruct-brier-r16-s2000-u0.3-train --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp,dbpedia,civil --split eval --n 300 --tag phi-mixed --out results/v2/seeds
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/seeds/DONE_v2phi
