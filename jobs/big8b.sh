set -exo pipefail; cd /workspace/brier; mkdir -p results/8b
M=Qwen/Qwen3-8B
python3 bench.py --model $M --tasks tickets --n 50 --methods scored,json 2>&1 | tail -20 > results/8b/bench_json_vs_scored.txt
python3 -m brier calibrate --model $M --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp --split eval --n 300 --tag 8b-untrained --out results/8b
python3 train.py --model $M --tasks tickets,gridworld,toolcall,passages,matching,ag_news,dbpedia,sst5,yelp,snli,qqp,civil,msmarco,massive,banking --examples 600 --steps 2000 --split train --uncertain 0.3 --batch 1 --accum 4 --seed 100 --out adapters/8b
python3 -m brier calibrate --model $M --adapter adapters/8b/Qwen3-8B-brier-r16-s2000-u0.3-train --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp --split eval --n 300 --tag 8b-mixed --out results/8b
touch results/8b/DONE_8b
