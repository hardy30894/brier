set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/typed
python3 train.py --model microsoft/Phi-4-mini-instruct --tasks typed --examples 1200 --steps 600 --accum 4 --batch 1 --split train --seed 1 --out adapters/tphi
python3 -m brier calibrate --model microsoft/Phi-4-mini-instruct --adapter adapters/tphi/Phi-4-mini-instruct-brier-r16-s600-train --task typed --split eval --n 400 --tag ft-phi4mini-seed1 --out results/v2/typed
mkdir -p results/logs && cp job.log results/logs/$(basename $0 .sh).log 2>/dev/null || true
touch results/v2/typed/DONE_tphi
