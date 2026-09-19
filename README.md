# GR00T Baseline

Note that a lot of the main parts of this project are made private due to research restrictions.

Two isolated Python environments share `data/` at the repo root. Do not mix their venvs.

```
GR00Tbaseline/
  data/raw/          # HA-Lab downloads and other raw teleop
  data/processed/    # GR00T LeRobot v2 / SONIC datasets (used by both sides)
  preprocess/        # conversion + validation (lightweight; this Mac is fine)
  train/             # Isaac-GR00T clone + GPU venv (Linux + NVIDIA CUDA)
```

## preprocess (this machine)

Lightweight conversion only — no PyTorch, no CUDA.

```bash
cd preprocess
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[halab]"

gr00t-download-halab --task pick_tennis_ball_place_black_basket
gr00t-convert-halab --overwrite
gr00t-validate --dataset-root ../data/processed/pick_tennis_ball_place_black_basket
```

`data/` is gitignored (~11 GB). On a fresh clone, restore the three baseline tasks with:

```bash
cd preprocess
./setup.sh
./redownload_halab.sh --num-tasks 3 --convert
# more tasks: ./redownload_halab.sh -n 8 --convert
# GPU box:    ./redownload_halab.sh -n 3 --convert --transcode-h264
```

CLI defaults resolve to `../data/...` from the repo root, so you can run them from any cwd after the venv is active. Full pipeline: [preprocess/README.md](preprocess/README.md).

## train (GPU machine)

Isaac-GR00T needs Linux and an NVIDIA GPU. Copy or share this repo (including `data/processed/`) onto that box, then:

```bash
cd train
./setup.sh
# Hugging Face login is required (gated nvidia/Cosmos-Reason2-2B)
./finetune_sonic.sh
./serve_gr00t.sh
```

`train/.venv` is created inside the cloned Isaac-GR00T tree by `uv sync`. Details: [train/README.md](train/README.md).
