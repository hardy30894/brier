set -exo pipefail; cd /workspace/brier; mkdir -p results/v2/typed
M=microsoft/Phi-4-mini-instruct
A=$(ls -d adapters/corp-phi-j/Phi-4-mini*/ | head -1)
python3 -m brier calibrate --model $M --adapter $A --task typed --split eval --n 400 --pointer-grade --tag phi-general-ptr --out results/v2/typed
touch results/v2/typed/DONE_phi-base-ptr
