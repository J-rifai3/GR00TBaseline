# Train — Isaac-GR00T (GPU machine only)

This folder is a **separate** environment from `../preprocess/`. It clones [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) and uses `uv` to create a Python 3.12 + CUDA venv inside `Isaac-GR00T/.venv`.

**Requirements:** Linux, NVIDIA GPU (fine-tune: ~40 GB+ VRAM recommended; inference: 16 GB+), `git`, `git-lfs`, `ffmpeg` 4–7, and a Hugging Face account with access to gated [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B).

Do not run `setup.sh` on macOS. Do not `pip install` Isaac-GR00T into the preprocess venv.

## Get the data onto this machine

Processed datasets live at the **repo root**: `../data/processed/<task>/`.

If this GPU box does not share a disk with the preprocess machine:

```bash
# from the Mac / preprocess machine
rsync -avz data/processed/ user@gpu-box:/path/to/GR00Tbaseline/data/processed/
```

Expected HA-Lab SONIC tasks (already converted on the preprocess side):

- `pick_tennis_ball_place_black_basket`
- `pick_puppy_doll_place_black_basket`
- `pick_tennis_racket_place_black_chair`

## Setup

```bash
cd train
./setup.sh
```

That clones `Isaac-GR00T/` (gitignored) and runs `uv sync --python 3.12`. Then authenticate:

```bash
cd Isaac-GR00T
uv run huggingface-cli login
```

Without HF access, loading `nvidia/GR00T-N1.7-3B` fails because it pulls the gated Cosmos-Reason2 backbone.

## Fine-tune UNITREE_G1_SONIC

```bash
cd train
./finetune_sonic.sh
```

Override with env vars:

| Variable | Default |
| --- | --- |
| `NUM_GPUS` | `1` |
| `MAX_STEPS` | `2000` |
| `GLOBAL_BATCH_SIZE` | `32` |
| `OUTPUT_DIR` | `train/checkpoints/g1_sonic` |
| `BASE_MODEL` | `nvidia/GR00T-N1.7-3B` |
| `DATASET_PATH` | colon-separated processed tasks that exist on disk |

## Serve a checkpoint

```bash
cd train
./serve_gr00t.sh
# or: MODEL_PATH=nvidia/GR00T-N1.7-3B ./serve_gr00t.sh   # base model, no finetune
```

The policy server listens on `0.0.0.0:5555` by default. Point a SONIC / robot client at that host.

Official whole-body collect → finetune → deploy docs: [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl).
