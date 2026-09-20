set -x; cd /workspace/brier; mkdir -p results/v2/encoder
python3 train.py --model answerdotai/ModernBERT-base --tasks tickets,gridworld,toolcall,passages,matching,ag_news,dbpedia,sst5,yelp,snli,qqp,civil,msmarco,massive,banking --examples 1500 --steps 3000 --split train --uncertain 0.3 --seed 1 --pointer --full --lr 3e-5 --out adapters/encbase
A=$(ls -d adapters/encbase/*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-base --adapter $A --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp,dbpedia,civil --split eval --n 300 --tag enc-base-full-seed1 --out results/v2/encoder
touch results/v2/encoder/DONE_encbase
