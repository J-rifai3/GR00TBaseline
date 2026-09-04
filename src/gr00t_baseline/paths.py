"""Path conventions for GR00T LeRobot v2 datasets."""

from __future__ import annotations

from pathlib import Path


def episode_name(episode_index: int) -> str:
    return f"episode_{episode_index:06d}"


def chunk_dir(name: str = "chunk-000") -> str:
    return name


def dataset_meta_dir(dataset_root: Path) -> Path:
    return dataset_root / "meta"


def dataset_data_dir(dataset_root: Path, chunk: str = "chunk-000") -> Path:
    return dataset_root / "data" / chunk


def dataset_videos_dir(dataset_root: Path, chunk: str = "chunk-000") -> Path:
    return dataset_root / "videos" / chunk


def parquet_path(dataset_root: Path, episode_index: int, chunk: str = "chunk-000") -> Path:
    return dataset_data_dir(dataset_root, chunk) / f"{episode_name(episode_index)}.parquet"


def video_dir_for_camera(
    dataset_root: Path,
    original_video_key: str,
    chunk: str = "chunk-000",
) -> Path:
    """LeRobot stores videos under observation.images.<camera_name>."""
    return dataset_videos_dir(dataset_root, chunk) / original_video_key


def video_path(
    dataset_root: Path,
    original_video_key: str,
    episode_index: int,
    chunk: str = "chunk-000",
) -> Path:
    return video_dir_for_camera(dataset_root, original_video_key, chunk) / (
        f"{episode_name(episode_index)}.mp4"
    )
