set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/grade
python3 train.py --model Qwen/Qwen3-0.6B --tasks tickets,gridworld,toolcall,passages,matching,ag_news,dbpedia,sst5,yelp,snli,qqp,civil,msmarco,massive,banking --examples 1500 --steps 3000 --split train --uncertain 0.3 --seed 1 --pointer --joint --grade-boost 3 --out adapters/v2joint1
A=adapters/v2joint1/Qwen3-0.6B-brier-r16-s3000-u0.3-train-pointer-gb3-joint
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter $A --task tickets,sst5,yelp,frustration --split eval --n 300 --tag grade-joint-gb3 --pointer-grade --out results/v2/grade
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter $A --task tickets,frustration,duplicates,banking,msmarco,mnli,imdb,emotion,snli,sst5,yelp,dbpedia,civil --split eval --n 300 --tag grade-joint-gb3 --out results/v2/grade
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/grade/DONE_v2joint1
