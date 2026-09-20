set -x; cd /workspace/brier; mkdir -p results/v2/release
python3 train.py --model answerdotai/ModernBERT-large --tasks tickets,gridworld,toolcall,passages,matching,ag_news,dbpedia,sst5,yelp,snli,qqp,civil,msmarco,massive,banking --examples 1500 --steps 4000 --warmup 400 --split train --uncertain 0.3 --seed 3 --pointer --full --lr 3e-5 --head-lr 1e-4 --out adapters/release
A=$(ls -d adapters/release/ModernBERT*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp,dbpedia,civil --split eval --n 300 --tag release-encoder --out results/v2/release
touch results/v2/release/DONE_rel-enc
