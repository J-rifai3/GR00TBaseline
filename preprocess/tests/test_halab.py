"""Tests for HA-Lab -> GR00T/SONIC conversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gr00t_baseline.halab import (
    SONIC_ACTION_DIM,
    SONIC_STATE_DIM,
    build_sonic_action,
    build_sonic_state,
    pad_hand_to_sonic,
    projected_gravity_from_quat_wxyz,
)
from gr00t_baseline.halab_convert import convert_halab_task
from gr00t_baseline.schema import ModalitySchema
from gr00t_baseline.validator import validate_dataset


def test_pad_hand_to_sonic() -> None:
    hand = np.arange(12, dtype=np.float32).reshape(2, 6)
    padded = pad_hand_to_sonic(hand)
    assert padded.shape == (2, 7)
    np.testing.assert_array_equal(padded[:, :6], hand)
    np.testing.assert_array_equal(padded[:, 6], 0)


def test_projected_gravity_identity_quat() -> None:
    identity = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    gravity = projected_gravity_from_quat_wxyz(identity)
    np.testing.assert_allclose(gravity, [[0.0, 0.0, -1.0]], atol=1e-6)


def _identity_source_state(n: int = 8) -> np.ndarray:
    state = np.zeros((n, 76), dtype=np.float32)
    state[:, 0:29] = np.linspace(0.0, 0.2, 29, dtype=np.float32)
    state[:, 58] = 1.0  # quat w
    state[:, 62:68] = 0.25
    state[:, 68:74] = 0.75
    return state


def test_build_sonic_vectors() -> None:
    state = _identity_source_state()
    latent = np.full((8, 64), 0.0625, dtype=np.float32)
    hand = np.concatenate(
        [np.ones((8, 6), dtype=np.float32), np.zeros((8, 6), dtype=np.float32)],
        axis=1,
    )
    sonic_state = build_sonic_state(state)
    sonic_action = build_sonic_action(latent, hand, state)
    assert sonic_state.shape == (8, SONIC_STATE_DIM)
    assert sonic_action.shape == (8, SONIC_ACTION_DIM)
    np.testing.assert_allclose(sonic_state[0, :29], state[0, :29])
    np.testing.assert_allclose(sonic_state[0, 29:35], 0.25)
    np.testing.assert_allclose(sonic_state[0, 36:42], 0.75)
    np.testing.assert_allclose(sonic_state[0, 43:46], [0.0, 0.0, -1.0], atol=1e-5)
    np.testing.assert_allclose(sonic_action[0, :64], 0.0625)
    np.testing.assert_allclose(sonic_action[0, 64:70], 1.0)
    np.testing.assert_allclose(sonic_action[0, 71:77], 0.0)


def _write_fake_halab_task(root: Path, *, with_video: bool = False) -> None:
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "robot_type": "g1_inspire_dynamixel",
                "fps": 30.0,
                "total_episodes": 1,
                "features": {
                    "observation.images.left": {
                        "dtype": "video",
                        "shape": [376, 672, 3],
                    }
                },
            }
        )
        + "\n"
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps(
            {
                "task_index": 0,
                "task": "Pick a tennis ball on top of the shelf and place it into the black basket",
            }
        )
        + "\n"
    )
    n = 12
    state = _identity_source_state(n)
    latent = np.zeros((n, 64), dtype=np.float32)
    measured = np.zeros((n, 65), dtype=np.float32)
    measured[:, :29] = state[:, :29]
    measured[:, 58] = 1.0
    hand = np.ones((n, 12), dtype=np.float32)
    df = pd.DataFrame(
        {
            "timestamp": np.arange(n, dtype=np.float32) / 30.0,
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.zeros(n, dtype=np.int64),
            "index": np.arange(n, dtype=np.int64),
            "task_index": np.zeros(n, dtype=np.int64),
            "observation.state": [row.tolist() for row in state],
            "action.latent_state": [row.tolist() for row in latent],
            "action.measured_state": [row.tolist() for row in measured],
            "action.hand_action": [row.tolist() for row in hand],
        }
    )
    df.to_parquet(data / "episode_000000.parquet", index=False)
    if with_video:
        import cv2

        video_dir = root / "videos" / "chunk-000" / "observation.images.left"
        video_dir.mkdir(parents=True)
        writer = cv2.VideoWriter(
            str(video_dir / "episode_000000.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (16, 16),
        )
        for _ in range(n):
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
        writer.release()


def test_convert_halab_task(tmp_path: Path) -> None:
    source = tmp_path / "halab_task"
    output = tmp_path / "processed"
    _write_fake_halab_task(source, with_video=True)

    result = convert_halab_task(source, output, overwrite=True, require_videos=True)
    assert result.num_episodes == 1
    assert result.num_frames == 12
    assert (output / "meta" / "modality.json").exists()
    assert (output / "data" / "chunk-000" / "episode_000000.parquet").exists()
    assert (
        output / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000000.mp4"
    ).exists()

    modality = ModalitySchema.load(output / "meta" / "modality.json")
    assert modality.state_dim == SONIC_STATE_DIM
    assert modality.action_dim == SONIC_ACTION_DIM
    assert "ego_view" in modality.video

    df = pd.read_parquet(output / "data" / "chunk-000" / "episode_000000.parquet")
    assert "action" in df.columns
    assert "annotation.human.task_description" in df.columns
    assert len(df["observation.state"].iloc[0]) == SONIC_STATE_DIM
    assert len(df["action"].iloc[0]) == SONIC_ACTION_DIM

    report = validate_dataset(output, frame_tolerance=2)
    assert report.ok, [i.message for i in report.issues]


def test_convert_halab_missing_video_can_be_skipped(tmp_path: Path) -> None:
    source = tmp_path / "halab_task"
    output = tmp_path / "processed"
    _write_fake_halab_task(source, with_video=False)
    with pytest.raises(FileNotFoundError):
        convert_halab_task(source, output, overwrite=True, require_videos=True)
    convert_halab_task(source, output, overwrite=True, require_videos=False)
    assert (output / "data" / "chunk-000" / "episode_000000.parquet").exists()
