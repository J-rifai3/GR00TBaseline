"""HA-Lab teleop layout (Unitree G1 + Inspire Dynamixel hands).

HA-Lab tasks on Hugging Face are already LeRobot v2.1 folders, but they are
not GR00T/SONIC-ready: there is no ``meta/modality.json``, no concatenated
``action`` column, and no ``annotation.human.task_description`` column.

Empirically, ``observation.state`` (76-D) is packed as:

    [0:29]   G1 body joint positions (body_q)
    [29:58]  G1 body joint velocities (body_dq)
    [58:62]  base quaternion wxyz
    [62:74]  Inspire hand state, 6 left + 6 right, typically in [0, 1]
    [74:76]  head/camera joints (2)

``action.measured_state`` (65-D) is [body_q(29), body_dq(29), quat(4), ang_vel(3)].
``action.latent_state`` (64-D) is the SONIC motion token.
``action.hand_action`` (12-D) is the Inspire finger command (6+6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HF_REPO_ID = "sehoonha/ha-lab-dataset"
HF_TASK_PREFIX = "halab"

# Default first task for the one-task baseline.
DEFAULT_TASK = "pick_tennis_ball_place_black_basket"

G1_BODY_JOINT_NAMES = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

INSPIRE_FINGER_NAMES = [
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_pitch",
    "thumb_yaw",
]

# SONIC / UNITREE_G1_SONIC expects 7-DoF Dex3 hands. Inspire is 6-DoF; pad last dim.
SONIC_HAND_DIM = 7
INSPIRE_HAND_DIM = 6
MOTION_TOKEN_DIM = 64
G1_BODY_DIM = 29
SOURCE_STATE_DIM = 76
MEASURED_STATE_DIM = 65

BODY_Q_SLICE = slice(0, 29)
BODY_DQ_SLICE = slice(29, 58)
BASE_QUAT_SLICE = slice(58, 62)
LEFT_HAND_STATE_SLICE = slice(62, 68)
RIGHT_HAND_STATE_SLICE = slice(68, 74)
CAMERA_STATE_SLICE = slice(74, 76)

SOURCE_VIDEO_KEY = "observation.images.left"
TARGET_VIDEO_KEY = "observation.images.ego_view"
ANNOTATION_COLUMN = "annotation.human.task_description"

# Concatenated GR00T/SONIC vectors written by the converter.
SONIC_STATE_SLICES = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "right_arm": (22, 29),
    "left_hand": (29, 36),
    "right_hand": (36, 43),
    "projected_gravity": (43, 46),
}
SONIC_ACTION_SLICES = {
    "motion_token": (0, 64),
    "left_hand_joints": (64, 71),
    "right_hand_joints": (71, 78),
}

SONIC_STATE_DIM = 46
SONIC_ACTION_DIM = 78


@dataclass
class HalabSourceLayout:
    """Resolved on-disk layout for one HA-Lab task folder."""

    task_root: Path
    data_dir: Path
    meta_dir: Path
    videos_dir: Path
    source_video_key: str
    annotation_dir: Path | None


def discover_halab_task(task_root: Path) -> HalabSourceLayout:
    task_root = task_root.resolve()
    data_dir = _find_chunk_dir(task_root / "data")
    videos_dir = task_root / "videos"
    video_chunk = _find_chunk_dir(videos_dir) if videos_dir.exists() else None
    source_video_key = SOURCE_VIDEO_KEY
    if video_chunk is not None:
        camera_dirs = sorted(p for p in video_chunk.iterdir() if p.is_dir())
        if camera_dirs:
            source_video_key = camera_dirs[0].name

    annotation_dir = task_root / "preprocess" / "annotation"
    if not annotation_dir.exists():
        annotation_dir = None

    return HalabSourceLayout(
        task_root=task_root,
        data_dir=data_dir,
        meta_dir=task_root / "meta",
        videos_dir=video_chunk if video_chunk is not None else videos_dir,
        source_video_key=source_video_key,
        annotation_dir=annotation_dir,
    )


def _find_chunk_dir(parent: Path) -> Path:
    if not parent.exists():
        raise FileNotFoundError(f"Missing HA-Lab directory: {parent}")
    chunks = sorted(p for p in parent.iterdir() if p.is_dir() and p.name.startswith("chunk-"))
    if chunks:
        return chunks[0]
    return parent


def load_info(task_root: Path) -> dict[str, Any]:
    path = task_root / "meta" / "info.json"
    with path.open() as f:
        return json.load(f)


def load_task_prompt(task_root: Path) -> str:
    tasks_path = task_root / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        with tasks_path.open() as f:
            first = f.readline().strip()
        if first:
            return str(json.loads(first).get("task") or _task_prompt_from_dirname(task_root))
    return _task_prompt_from_dirname(task_root)


def _task_prompt_from_dirname(task_root: Path) -> str:
    return task_root.name.replace("_", " ")


def list_episode_parquets(layout: HalabSourceLayout) -> list[Path]:
    files = sorted(layout.data_dir.glob("episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode_*.parquet files under {layout.data_dir}")
    return files


def stack_float_column(series: pd.Series, expected_dim: int | None = None) -> np.ndarray:
    values = np.stack([np.asarray(v, dtype=np.float32) for v in series])
    if expected_dim is not None and values.shape[1] != expected_dim:
        raise ValueError(
            f"Column dim {values.shape[1]} != expected {expected_dim} "
            f"(series length {len(series)})"
        )
    return values


def pad_hand_to_sonic(hand6: np.ndarray) -> np.ndarray:
    """Pad Inspire 6-DoF fingers to SONIC's 7-DoF Dex3 slot."""
    if hand6.ndim != 2:
        raise ValueError(f"hand array must be [T, D], got {hand6.shape}")
    out = np.zeros((hand6.shape[0], SONIC_HAND_DIM), dtype=np.float32)
    n = min(hand6.shape[1], SONIC_HAND_DIM)
    out[:, :n] = hand6[:, :n]
    return out


def projected_gravity_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """Gravity expressed in the robot base frame.

    Matches the usual Isaac / SONIC ``projected_gravity`` convention:
    rotate world gravity ``[0, 0, -1]`` by the inverse of a wxyz quaternion.
    """
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"quat must be [T, 4] wxyz, got {quat.shape}")
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    # v = [0, 0, -1],  q^{-1} v q
    gx = 2.0 * (x * z - w * y)
    gy = 2.0 * (y * z + w * x)
    gz = -(w * w - x * x - y * y + z * z)
    return np.stack([gx, gy, gz], axis=1).astype(np.float32)


def build_sonic_state(
    source_state: np.ndarray,
    measured_state: np.ndarray | None = None,
) -> np.ndarray:
    """Rebuild UNITREE_G1_SONIC observation.state (46-D)."""
    if source_state.shape[1] != SOURCE_STATE_DIM:
        raise ValueError(
            f"observation.state dim {source_state.shape[1]} != {SOURCE_STATE_DIM}"
        )
    body_q = source_state[:, BODY_Q_SLICE]
    if measured_state is not None and measured_state.shape[1] >= G1_BODY_DIM:
        # Prefer the named measured body_q when present; it is the same 29-D
        # motor order and is slightly cleaner than the packed state copy.
        body_q = measured_state[:, :G1_BODY_DIM]
    left_hand = pad_hand_to_sonic(source_state[:, LEFT_HAND_STATE_SLICE])
    right_hand = pad_hand_to_sonic(source_state[:, RIGHT_HAND_STATE_SLICE])
    gravity = projected_gravity_from_quat_wxyz(source_state[:, BASE_QUAT_SLICE])
    return np.concatenate([body_q, left_hand, right_hand, gravity], axis=1).astype(np.float32)


def build_sonic_action(
    latent: np.ndarray,
    hand_action: np.ndarray | None,
    source_state: np.ndarray,
) -> np.ndarray:
    """Rebuild UNITREE_G1_SONIC action (78-D): token + left/right hands."""
    if latent.shape[1] != MOTION_TOKEN_DIM:
        raise ValueError(f"latent dim {latent.shape[1]} != {MOTION_TOKEN_DIM}")
    if hand_action is not None and hand_action.shape[1] >= 2 * INSPIRE_HAND_DIM:
        left = pad_hand_to_sonic(hand_action[:, :INSPIRE_HAND_DIM])
        right = pad_hand_to_sonic(hand_action[:, INSPIRE_HAND_DIM : 2 * INSPIRE_HAND_DIM])
    else:
        left = pad_hand_to_sonic(source_state[:, LEFT_HAND_STATE_SLICE])
        right = pad_hand_to_sonic(source_state[:, RIGHT_HAND_STATE_SLICE])
    return np.concatenate([latent, left, right], axis=1).astype(np.float32)


def sonic_state_names() -> list[str]:
    names = list(G1_BODY_JOINT_NAMES)
    for side in ("left", "right"):
        names.extend(f"{side}_{n}" for n in INSPIRE_FINGER_NAMES)
        names.append(f"{side}_hand_pad")
    names.extend(["gravity_x", "gravity_y", "gravity_z"])
    return names


def sonic_action_names() -> list[str]:
    names = [f"token_{i}" for i in range(MOTION_TOKEN_DIM)]
    for side in ("left", "right"):
        names.extend(f"{side}_{n}" for n in INSPIRE_FINGER_NAMES)
        names.append(f"{side}_hand_pad")
    return names


def default_sonic_modality_dict() -> dict[str, Any]:
    return {
        "state": {
            key: {"start": start, "end": end} for key, (start, end) in SONIC_STATE_SLICES.items()
        },
        "action": {
            key: {"start": start, "end": end, "absolute": True}
            for key, (start, end) in SONIC_ACTION_SLICES.items()
        },
        "video": {
            "ego_view": {"original_key": TARGET_VIDEO_KEY},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }
