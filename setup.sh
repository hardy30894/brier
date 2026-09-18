#!/usr/bin/env bash
# One-shot setup + smoke run on a fresh GPU pod (RunPod PyTorch template or any CUDA box).
set -euo pipefail
cd "$(dirname "$0")"
pip install -q "transformers>=4.51,<4.57" accelerate numpy huggingface_hub
python3 -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B')"
python3 bench.py --n 10
