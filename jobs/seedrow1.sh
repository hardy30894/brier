set -exo pipefail; cd /workspace/brier; mkdir -p results/seeds
python3 train.py --model Qwen/Qwen3-0.6B --tasks tickets,gridworld,toolcall,passages,matching,ag_news,dbpedia,sst5,yelp,snli,qqp,civil,msmarco,massive,banking --examples 600 --steps 2000 --split train --uncertain 0.3 --batch 2 --accum 2 --seed 1 --out adapters/seedrow1
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter adapters/seedrow1/Qwen3-0.6B-brier-r16-s2000-u0.3-train --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp --split eval --n 300 --tag row-seed1 --out results/seeds
touch results/seeds/DONE_seedrow1
for E in 7 8 9; do python3 -m brier calibrate --model Qwen/Qwen3-0.6B --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp --split eval --n 300 --seed $E --tag untrained-eval$E --out results/seeds; done
touch results/seeds/DONE_untrained
