set -x; cd /workspace/brier; mkdir -p results/seeds
python3 train.py --model Qwen/Qwen3-0.6B --tasks tickets,gridworld,toolcall,passages,matching,ag_news,dbpedia,sst5,yelp,snli,qqp,civil,msmarco,massive,banking --examples 1500 --steps 3000 --split train --uncertain 0.3 --seed 2 --pointer --out adapters/seedptr2
A=adapters/seedptr2/Qwen3-0.6B-brier-r16-s3000-u0.3-train-pointer
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter $A --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp --split eval --n 300 --tag ptr-seed2 --out results/seeds
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter $A --task tickets,sst5,yelp,frustration --split eval --n 300 --tag ptr-seed2 --pointer-grade --out results/seeds
touch results/seeds/DONE_seedptr2
