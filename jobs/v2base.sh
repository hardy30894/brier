set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/baselines
python3 baselines.py --model Qwen/Qwen3-0.6B --task msmarco,imdb,dbpedia,civil --n 300 --out results/v2/baselines
python3 baselines.py --model microsoft/Phi-4-mini-instruct --task msmarco,imdb,dbpedia,civil --n 300 --out results/v2/baselines
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/baselines/DONE_v2base
