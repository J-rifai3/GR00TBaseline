"""Write GR00T LeRobot v2 episodes (parquet + mp4 + metadata)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import jsonlines
import numpy as np
import pandas as pd

from gr00t_baseline.paths import (
    dataset_data_dir,
    dataset_meta_dir,
    dataset_videos_dir,
    episode_name,
    parquet_path,
    video_dir_for_camera,
    video_path,
)
from gr00t_baseline.raw_io import RawEpisode
from gr00t_baseline.schema import DEFAULT_FPS, DEFAULT_ROBOT_TYPE, ModalitySchema


class EpisodeWriter:
    """Convert one raw episode into LeRobot v2 parquet + video files."""

    def __init__(
        self,
        dataset_root: Path,
        modality: ModalitySchema,
        chunk: str = "chunk-000",
        global_index_offset: int = 0,
    ):
        self.dataset_root = dataset_root
        self.modality = modality
        self.chunk = chunk
        self.global_index_offset = global_index_offset

    def write_episode(
        self,
        episode: RawEpisode,
        task_index: int,
        annotation_column: str = "annotation.human.task_description",
    ) -> tuple[int, int]:
        """Write parquet and videos. Returns (episode_length, next_global_index)."""
        length = episode.length
        if length == 0:
            raise ValueError(f"Episode {episode.episode_index} is empty")

        timestamps = np.arange(length, dtype=np.float32) / episode.fps
        global_indices = np.arange(
            self.global_index_offset,
            self.global_index_offset + length,
            dtype=np.int64,
        )
        episode_indices = np.full(length, episode.episode_index, dtype=np.int64)
        rewards = np.zeros(length, dtype=np.float32)
        dones = np.zeros(length, dtype=bool)
        dones[-1] = True

        df = pd.DataFrame(
            {
                "observation.state": [row.tolist() for row in episode.states],
                "action": [row.tolist() for row in episode.actions],
                "timestamp": timestamps,
                "episode_index": episode_indices,
                "index": global_indices,
                "next.reward": rewards,
                "next.done": dones,
                annotation_column: np.full(length, task_index, dtype=np.int64),
                "task_index": np.full(length, task_index, dtype=np.int64),
            }
        )

        out_parquet = parquet_path(self.dataset_root, episode.episode_index, self.chunk)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_parquet, index=False)

        for short_key, original_key in self.modality.video.items():
            src = episode.videos.get(short_key)
            if src is None:
                raise FileNotFoundError(
                    f"Episode {episode.episode_index} missing video for camera '{short_key}'"
                )
            dst = video_path(self.dataset_root, original_key, episode.episode_index, self.chunk)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        return length, self.global_index_offset + length


def write_dataset_info(
    dataset_root: Path,
    *,
    fps: float = DEFAULT_FPS,
    robot_type: str = DEFAULT_ROBOT_TYPE,
    total_episodes: int,
    total_frames: int,
    chunk: str = "chunk-000",
) -> None:
    meta_dir = dataset_meta_dir(dataset_root)
    meta_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 0,  # updated by converter after tasks are known
        "total_videos": total_episodes * 1,  # placeholder; converter may override
        "total_chunks": 1,
        "chunks_size": total_episodes,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": f"data/{chunk}/episode_{{episode_index:06d}}.parquet",
        "video_path": f"videos/{chunk}/{{video_key}}/episode_{{episode_index:06d}}.mp4",
    }
    with (meta_dir / "info.json").open("w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")


def append_episode_record(
    dataset_root: Path,
    episode_index: int,
    task: str,
    length: int,
) -> None:
    meta_dir = dataset_meta_dir(dataset_root)
    meta_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = meta_dir / "episodes.jsonl"
    with jsonlines.open(episodes_path, mode="a") as writer:
        writer.write(
            {
                "episode_index": episode_index,
                "tasks": [task],
                "length": length,
            }
        )


def register_task(dataset_root: Path, task: str, task_registry: dict[str, int]) -> int:
    if task not in task_registry:
        task_registry[task] = len(task_registry)
    return task_registry[task]


def write_tasks_file(dataset_root: Path, task_registry: dict[str, int]) -> None:
    meta_dir = dataset_meta_dir(dataset_root)
    meta_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = meta_dir / "tasks.jsonl"
    with jsonlines.open(tasks_path, mode="w") as writer:
        for task, task_index in sorted(task_registry.items(), key=lambda x: x[1]):
            writer.write({"task_index": task_index, "task": task})


def ensure_dataset_dirs(dataset_root: Path, modality: ModalitySchema, chunk: str = "chunk-000") -> None:
    dataset_data_dir(dataset_root, chunk).mkdir(parents=True, exist_ok=True)
    dataset_videos_dir(dataset_root, chunk).mkdir(parents=True, exist_ok=True)
    for original_key in modality.video.values():
        video_dir_for_camera(dataset_root, original_key, chunk).mkdir(parents=True, exist_ok=True)


def transcode_video_to_h264(src: Path, dst: Path, fps: float) -> None:
    """Re-encode video as H.264 MP4 (GR00T expects torchcodec-compatible MP4)."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {src}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)

    cap.release()
    writer.release()
