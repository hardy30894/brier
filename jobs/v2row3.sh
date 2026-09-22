set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/seeds
python3 train.py --model Qwen/Qwen3-0.6B --tasks tickets,gridworld,toolcall,passages,matching,ag_news,dbpedia,sst5,yelp,snli,qqp,civil,msmarco,massive,banking --examples 600 --steps 2000 --split train --uncertain 0.3 --batch 2 --accum 2 --seed 3 --out adapters/v2row3
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter adapters/v2row3/Qwen3-0.6B-brier-r16-s2000-u0.3-train --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp,dbpedia,civil --split eval --n 300 --tag row-seed3 --out results/v2/seeds
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/seeds/DONE_v2row3
