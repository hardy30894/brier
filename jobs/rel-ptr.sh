set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/release
python3 train.py --model Qwen/Qwen3-0.6B --tasks tickets,gridworld,toolcall,passages,matching,ag_news,dbpedia,sst5,yelp,snli,qqp,civil,msmarco,massive,banking --examples 1500 --steps 3000 --split train --uncertain 0.3 --seed 1 --pointer --grade-boost 3 --out adapters/release
A=$(ls -d adapters/release/Qwen3-0.6B*pointer*/ | head -1)
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter $A --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp,dbpedia,civil --split eval --n 300 --tag release-pointer --out results/v2/release
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/release/DONE_rel-ptr
