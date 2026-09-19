# Preprocess — Teleoperation → GR00T LeRobot v2

Conversion and validation only. Training lives in `../train/` with a separate venv.

Shared data is at the **repo root**, not inside this folder:

- `../data/raw/` — HA-Lab downloads and generic `episode_*` folders
- `../data/processed/` — GR00T LeRobot v2 / SONIC datasets

## Setup

```bash
cd preprocess
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[halab]"
# optional: pip install -e ".[halab,dev]"  # pytest, ruff
```

Or: `./setup.sh`

`../data/` is gitignored. To re-download the three baseline tasks (see `../train/datasets.txt`):

```bash
./redownload_halab.sh                         # first 3 tasks, raw only
./redownload_halab.sh --num-tasks 1           # just one
./redownload_halab.sh -n 8 --convert          # first 8 (3 baseline, then HF list)
./redownload_halab.sh --convert --transcode-h264 --validate
./redownload_halab.sh --task pick_tennis_ball_place_black_basket
```

## Quick start (synthetic)

```bash
python scripts/generate_synthetic_raw.py
gr00t-inspect
gr00t-convert --output-root ../data/processed/demo --modality-config configs/modality_template.json
gr00t-validate --dataset-root ../data/processed/demo
gr00t-split --dataset-root ../data/processed/demo --val-ratio 0.1
```

## HA-Lab dataset (G1 + SONIC)

[sehoonha/ha-lab-dataset](https://huggingface.co/datasets/sehoonha/ha-lab-dataset) is already LeRobot v2.1 per task (`data/`, `meta/`, `videos/`), but it is **not** GR00T/SONIC-ready: there is no `meta/modality.json`, no concatenated `action` column, and no `annotation.human.task_description` column.

Each HA-Lab parquet row has:

| Column | Dim | Role |
| --- | --- | --- |
| `observation.state` | 76 | body_q (29) + body_dq (29) + base quat wxyz (4) + Inspire hands (12) + camera (2) |
| `action.latent_state` | 64 | SONIC motion token |
| `action.measured_state` | 65 | body_q + body_dq + quat + angular velocity |
| `action.hand_action` | 12 | Inspire finger commands (6 left + 6 right, typically `[0, 1]`) |
| `observation.images.left` | video | head/ego camera, 376×672 @ 30 FPS |

The converter rebuilds the `UNITREE_G1_SONIC` vectors:

- **state (46-D):** G1 body (29) + left/right hands padded 6→7 + projected gravity (3)
- **action (78-D):** motion token (64) + left/right hand joints padded 6→7
- **video:** `observation.images.left` copied to `observation.images.ego_view`

Inspire hands are 6-DoF; SONIC's Dex3 slot is 7-DoF, so the last hand dimension is zero-padded.

### Convert one task

```bash
gr00t-download-halab --list-tasks
gr00t-download-halab --task pick_tennis_ball_place_black_basket
gr00t-inspect-halab
gr00t-convert-halab --overwrite
gr00t-validate --dataset-root ../data/processed/pick_tennis_ball_place_black_basket
```

Use `--max-episodes 1` for a smoke test. A full task is roughly 1–2 GB of video.

HA-Lab videos are stored as `mp4v`. For GR00T's `torchcodec` decoder, re-encode while converting:

```bash
gr00t-convert-halab --overwrite --transcode-h264
```

Requires `ffmpeg` on PATH.

Fine-tuning is **not** done in this venv. Use `../train/` on a Linux+NVIDIA machine with `--embodiment-tag UNITREE_G1_SONIC`.

## Raw teleop format (generic converter)

Each episode is a directory named `episode_XXXXXX`:

```
../data/raw/
  episode_000000/
    metadata.json    # {"task": "pick cube", "fps": 30}
    states.npy       # float32 [T, state_dim]
    actions.npy      # float32 [T, action_dim]
    front.mp4
    wrist.mp4
```

If your teleop stack uses a different layout (LeRobot v3, HDF5, ROS bags), adapt `src/gr00t_baseline/raw_io.py`.

## Output format (GR00T LeRobot v2)

```
../data/processed/my_dataset/
  meta/
    info.json
    episodes.jsonl
    tasks.jsonl
    modality.json       # required by GR00T
  data/chunk-000/
    episode_000000.parquet
  videos/chunk-000/
    observation.images.ego_view/
      episode_000000.mp4
```

Edit `configs/modality_template.json` to match your robot. Keys must align with `configs/embodiment_config.example.py` (generic) or `configs/embodiment_config_g1_sonic.example.py` (HA-Lab / SONIC) when you fine-tune.

## Layout

| Path | Purpose |
|------|---------|
| `configs/modality_template.json` | Generic state/action/video slice definitions |
| `configs/halab_g1_sonic_modality.json` | UNITREE_G1_SONIC slices for HA-Lab conversion |
| `configs/embodiment_config.example.py` | Isaac-GR00T training config template (generic) |
| `configs/embodiment_config_g1_sonic.example.py` | Isaac-GR00T config matching HA-Lab SONIC keys |
| `scripts/` | CLI wrappers |
| `src/gr00t_baseline/` | Core conversion + validation library |

## Notes

- Videos should be H.264 MP4 for GR00T's `torchcodec` decoder.
- If you collected data with LeRobot v3, use Isaac-GR00T's `scripts/lerobot_conversion/convert_v3_to_v2.py` first, then add `meta/modality.json`.
- Preprocessing deps are intentionally lightweight — no PyTorch or GR00T training stack in this venv.
