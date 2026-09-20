set -x; cd /workspace/brier; mkdir -p results/v2/outcome
python3 rl.py --model Qwen/Qwen3-0.6B --task toolcall --steps 600 --batch 8 --seed 3 --out adapters/rl-toolcall-3
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter adapters/rl-toolcall-3/Qwen3-0.6B-brier-rl-toolcall-s600 --task toolcall --split eval --n 300 --tag outcome-toolcall-seed3 --out results/v2/outcome
python3 train.py --model Qwen/Qwen3-0.6B --tasks toolcall --examples 1200 --steps 600 --accum 4 --batch 2 --split train --uncertain 0.3 --seed 3 --out adapters/sup-toolcall-3
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter adapters/sup-toolcall-3/Qwen3-0.6B-brier-r16-s600-u0.3-train --task toolcall --split eval --n 300 --tag supervised-toolcall-seed3 --out results/v2/outcome

touch results/v2/outcome/DONE_out-toolcall-3
