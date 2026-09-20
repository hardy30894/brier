set -x; cd /workspace/brier; mkdir -p results/v2/release
python3 train.py --model Qwen/Qwen3-8B --tasks tickets,gridworld,toolcall,passages,matching,ag_news,dbpedia,sst5,yelp,snli,qqp,civil,msmarco,massive,banking --examples 600 --steps 2000 --split train --uncertain 0.3 --batch 1 --accum 4 --seed 100 --out adapters/release
python3 -m brier calibrate --model Qwen/Qwen3-8B --adapter adapters/release/Qwen3-8B-brier-r16-s2000-u0.3-train --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp,dbpedia,civil --split eval --n 300 --tag release-8b --out results/v2/release
touch results/v2/release/DONE_rel-8b
