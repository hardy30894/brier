set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/typed
M=microsoft/Phi-4-mini-instruct
A=$(ls -d adapters/corp-phi-j/Phi-4-mini*/ | head -1)
python3 rl.py --model $M --init $A --task typed --pointer --lr 5e-6 --head-lr 2e-5 --steps 600 --batch 8 --seed 3 --out adapters/phi-out-3
B=$(ls -d adapters/phi-out-3/*/ | head -1)
python3 -m brier calibrate --model $M --adapter $B --task typed --split eval --n 400 --pointer-grade --tag phi-outcome-seed3 --out results/v2/typed
mkdir -p results/logs && cp job.log results/logs/phi-out-3.log 2>/dev/null || true
touch results/v2/typed/DONE_phi-out-3
