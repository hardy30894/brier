set -x; cd /workspace/brier; mkdir -p results/v2/outcome
python3 rl.py --model Qwen/Qwen3-0.6B --task tickets --steps 600 --batch 8 --seed 2 --out adapters/rl-tickets-2
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter adapters/rl-tickets-2/Qwen3-0.6B-brier-rl-tickets-s600 --task tickets --split eval --n 300 --tag outcome-tickets-seed2 --out results/v2/outcome
python3 train.py --model Qwen/Qwen3-0.6B --tasks tickets --examples 1200 --steps 600 --accum 4 --batch 2 --split train --uncertain 0.3 --seed 2 --out adapters/sup-tickets-2
python3 -m brier calibrate --model Qwen/Qwen3-0.6B --adapter adapters/sup-tickets-2/Qwen3-0.6B-brier-r16-s600-u0.3-train --task tickets --split eval --n 300 --tag supervised-tickets-seed2 --out results/v2/outcome

touch results/v2/outcome/DONE_out-tickets-2
