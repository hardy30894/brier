# Releasing Brier

Five steps, in order. Nothing here runs a model.

1. **Pull the weights.** Every release candidate is trained on a pod and pulled with
   `python3 pod.py --name <pod> pull --adapters`, which lands under `adapters/`. A LoRA
   release directory holds `adapter_config.json`, `adapter_model.safetensors`, `train_args.json`,
   and for pointer adapters `pointer.pt` and `pointer.json`; an encoder release directory
   holds `backbone.pt`, `pointer.pt`, `pointer.json`, `train_args.json`.

2. **Write the held-out numbers.** A JSON file per model, a list of rows with `task`,
   `question`, `accuracy`, `ece` (raw), optional `brier` and `note`, taken from the
   `cal_*.json` files under `results/` for that exact adapter. The card prints this table
   verbatim; nothing that is not in `results/` goes on a card.

3. **Dry-run the cards.**

   ```sh
   python3 release/publish.py --dry-run --org brier-lab \
     --model qwen3-0.6b-pointer-mixed --dir adapters/<pointer adapter dir> \
     --backbone Qwen/Qwen3-0.6B --kind pointer --data mixed \
     --recipe "python3 train.py --model Qwen/Qwen3-0.6B --tasks <mixed list> --examples 1500 --steps 3000 --split train --uncertain 0.3 --seed 1 --pointer --grade-boost 3" \
     --numbers release/numbers/qwen3-0.6b-pointer-mixed.json
   ```

   Read `release/cards/<model>/README.md`. Repeat for every model. Cards must name the
   backbone, the data, the exact recipe, the limitations, and nothing about any other
   vendor's product.

4. **Publish.** Put `HF_TOKEN=...` (a write token for the org) in `.env` or the environment,
   drop `--dry-run`, run the same commands. The script creates the repository if needed and
   uploads the directory with the card as `README.md`. `release/stage/` is a scratch copy
   and is not committed.

5. **Package.** Bump `version` in `pyproject.toml`, then

   ```sh
   python3 -m build
   python3 -m twine upload dist/*
   ```

   Tag the commit `v<version>`. Announce with the head-to-head table from `README.md`.

Release set for v0.1.0: `modernbert-large-decisions-mixed` (encoder, mixed corpus, the
seed with the best held-out mean), `qwen3-0.6b-pointer-mixed` (joint pointer adapter),
`qwen3-8b-decisions-mixed` (row adapter), `modernbert-large-decisions-typed` and
`qwen3-0.6b-pointer-typed` (fine-tuned on the public benchmark's training split).
