set -x; cd /workspace/brier; mkdir -p results/v2/encoder
python3 train.py --model answerdotai/ModernBERT-large --tasks tickets,gridworld,toolcall,passages,matching,ag_news,dbpedia,sst5,yelp,snli,qqp,civil,msmarco,massive,banking --examples 1500 --steps 4000 --warmup 400 --split train --uncertain 0.3 --seed 3 --pointer --full --lr 3e-5 --head-lr 1e-4 --out adapters/enc9
A=$(ls -d adapters/enc9/*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp,dbpedia,civil --split eval --n 300 --tag enc-large-full-w400-h1e4-seed3 --out results/v2/encoder
touch results/v2/encoder/DONE_enc9
