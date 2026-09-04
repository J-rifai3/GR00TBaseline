# GR00T Baseline — Teleoperation Preprocessing

Note that this a lot of the main parts of this project are made private due to research restrictions!

Preprocessing pipeline for converting teleoperation demonstrations into **GR00T LeRobot v2** format for fine-tuning with [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T).

This repo handles data conversion and validation. Training runs in a separate Isaac-GR00T environment.

## Quick start

```bash
# Create a virtualenv and install preprocessing deps
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Generate synthetic raw episodes to smoke-test the pipeline
python scripts/generate_synthetic_raw.py

# Inspect raw data and get a draft modality.json
gr00t-inspect --raw-root data/raw

# Convert to GR00T LeRobot v2
gr00t-convert \
  --raw-root data/raw \
  --output-root data/processed/demo \
  --modality-config configs/modality_template.json

# Validate before training
gr00t-validate --dataset-root data/processed/demo

# Optional: train/val split
gr00t-split --dataset-root data/processed/demo --val-ratio 0.1
```

## Raw teleop format (default)

Each episode is a directory named `episode_XXXXXX`:

```
data/raw/
  episode_000000/
    metadata.json    # {"task": "pick cube", "fps": 30}
    states.npy       # float32 [T, state_dim]
    actions.npy      # float32 [T, action_dim]
    front.mp4
    wrist.mp4
```

If your teleop stack uses a different layout (LeRobot v3, HDF5, ROS bags), adapt `src/gr00t_baseline/raw_io.py` or add a converter under `src/gr00t_baseline/converters/`.

## Output format (GR00T LeRobot v2)

```
data/processed/my_dataset/
  meta/
    info.json
    episodes.jsonl
    tasks.jsonl
    modality.json       # required by GR00T
  data/chunk-000/
    episode_000000.parquet
  videos/chunk-000/
    observation.images.front/
      episode_000000.mp4
    observation.images.wrist/
      episode_000000.mp4
```

Edit `configs/modality_template.json` to match your robot's state/action dimensions and camera names. Keys must align with `configs/embodiment_config.example.py` when you fine-tune.

## Fine-tuning (Isaac-GR00T)

1. Clone Isaac-GR00T and set up its Python 3.12 + GPU environment.
2. Copy your processed dataset and register a modality config (see `configs/embodiment_config.example.py`).
3. Compute stats and fine-tune:

```bash
# Inside Isaac-GR00T checkout
python gr00t/data/stats.py --dataset-path /path/to/data/processed/my_dataset
python gr00t/experiment/launch_finetune.py \
  --dataset-path /path/to/data/processed/my_dataset \
  --embodiment-tag NEW_EMBODIMENT
```

See the [GR00T data format docs](https://nvidia-isaac-gr00t.mintlify.app/concepts/data-format) for full schema details.

## Project layout

| Path | Purpose |
|------|---------|
| `configs/modality_template.json` | State/action/video slice definitions |
| `configs/embodiment_config.example.py` | Isaac-GR00T training config template |
| `scripts/` | CLI wrappers |
| `src/gr00t_baseline/` | Core conversion + validation library |

## Notes

- Videos should be H.264 MP4 for GR00T's `torchcodec` decoder.
- If you collected data with LeRobot v3, use Isaac-GR00T's `scripts/lerobot_conversion/convert_v3_to_v2.py` first, then add `meta/modality.json`.
- Preprocessing deps are intentionally lightweight — no PyTorch or GR00T training stack required here.
