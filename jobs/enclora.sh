set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/encoder
python3 train.py --model answerdotai/ModernBERT-large --tasks tickets,gridworld,toolcall,passages,matching,ag_news,dbpedia,sst5,yelp,snli,qqp,civil,msmarco,massive,banking --examples 1500 --steps 3000 --split train --uncertain 0.3 --seed 1 --pointer  --lr 1e-4 --out adapters/enclora
A=$(ls -d adapters/enclora/*/ | head -1)
python3 -m brier calibrate --model answerdotai/ModernBERT-large --adapter $A --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp,dbpedia,civil --split eval --n 300 --tag enc-large-lora-seed1 --out results/v2/encoder
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/encoder/DONE_enclora
