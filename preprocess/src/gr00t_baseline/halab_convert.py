"""Convert one HA-Lab task folder into a GR00T LeRobot v2 / SONIC dataset."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from gr00t_baseline.episode_writer import (
    append_episode_record,
    ensure_dataset_dirs,
    write_tasks_file,
)
from gr00t_baseline.halab import (
    ANNOTATION_COLUMN,
    SONIC_ACTION_DIM,
    SONIC_STATE_DIM,
    SOURCE_VIDEO_KEY,
    TARGET_VIDEO_KEY,
    HalabSourceLayout,
    build_sonic_action,
    build_sonic_state,
    default_sonic_modality_dict,
    discover_halab_task,
    list_episode_parquets,
    load_info,
    load_task_prompt,
    sonic_action_names,
    sonic_state_names,
    stack_float_column,
)
from gr00t_baseline.paths import dataset_meta_dir, parquet_path, video_path
from gr00t_baseline.schema import ModalitySchema


@dataclass
class HalabConversionResult:
    dataset_root: Path
    num_episodes: int
    num_frames: int
    skipped_missing_video: int
    task: str
    transcoded_videos: int


def convert_halab_task(
    source_root: Path,
    output_root: Path,
    *,
    modality: ModalitySchema | None = None,
    overwrite: bool = False,
    max_episodes: int | None = None,
    symlink_videos: bool = False,
    require_videos: bool = True,
    transcode_h264: bool = False,
    chunk: str = "chunk-000",
) -> HalabConversionResult:
    layout = discover_halab_task(source_root)
    source_info = load_info(layout.task_root)
    task_prompt = load_task_prompt(layout.task_root)
    modality = modality or ModalitySchema.from_dict(default_sonic_modality_dict())

    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"Output dataset already exists: {output_root}. Pass overwrite=True to replace."
            )

    output_root.mkdir(parents=True)
    modality.save(dataset_meta_dir(output_root) / "modality.json")
    ensure_dataset_dirs(output_root, modality, chunk)

    parquet_files = list_episode_parquets(layout)
    if max_episodes is not None:
        parquet_files = parquet_files[: max(0, max_episodes)]

    fps = float(source_info.get("fps", 30.0))
    robot_type = str(source_info.get("robot_type", "g1_inspire_dynamixel"))
    task_registry = {task_prompt: 0}
    global_index = 0
    total_frames = 0
    skipped_missing_video = 0
    converted_episodes = 0
    transcoded_videos = 0

    for parquet_file in tqdm(parquet_files, desc="Converting HA-Lab episodes"):
        episode_index = _episode_index_from_name(parquet_file.name)
        df = pd.read_parquet(parquet_file)
        sonic_state, sonic_action = _episode_vectors(df)

        if len(df) == 0:
            raise ValueError(f"{parquet_file.name} is empty")
        if sonic_state.shape != (len(df), SONIC_STATE_DIM):
            raise ValueError(
                f"{parquet_file.name}: rebuilt state {sonic_state.shape} "
                f"!= ({len(df)}, {SONIC_STATE_DIM})"
            )
        if sonic_action.shape != (len(df), SONIC_ACTION_DIM):
            raise ValueError(
                f"{parquet_file.name}: rebuilt action {sonic_action.shape} "
                f"!= ({len(df)}, {SONIC_ACTION_DIM})"
            )

        src_video = _source_video_path(layout, episode_index)
        if src_video is None:
            skipped_missing_video += 1
            if require_videos:
                raise FileNotFoundError(
                    f"Missing video for episode {episode_index:06d} under {layout.videos_dir}"
                )
        else:
            dst_video = video_path(output_root, TARGET_VIDEO_KEY, episode_index, chunk)
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            if symlink_videos:
                if transcode_h264:
                    raise ValueError("Cannot combine symlink_videos with transcode_h264")
                if dst_video.exists() or dst_video.is_symlink():
                    dst_video.unlink()
                dst_video.symlink_to(src_video.resolve())
            elif transcode_h264:
                _transcode_h264(src_video, dst_video)
                transcoded_videos += 1
            else:
                shutil.copy2(src_video, dst_video)

        timestamps = _timestamps(df, fps)
        dones = np.zeros(len(df), dtype=bool)
        dones[-1] = True
        out = pd.DataFrame(
            {
                "observation.state": [row.tolist() for row in sonic_state],
                "action": [row.tolist() for row in sonic_action],
                "timestamp": timestamps.astype(np.float32),
                "episode_index": np.full(len(df), episode_index, dtype=np.int64),
                "index": np.arange(global_index, global_index + len(df), dtype=np.int64),
                "next.reward": np.zeros(len(df), dtype=np.float32),
                "next.done": dones,
                ANNOTATION_COLUMN: np.zeros(len(df), dtype=np.int64),
                "task_index": np.zeros(len(df), dtype=np.int64),
            }
        )
        out_parquet = parquet_path(output_root, episode_index, chunk)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_parquet, index=False)

        append_episode_record(output_root, episode_index, task_prompt, len(df))
        global_index += len(df)
        total_frames += len(df)
        converted_episodes += 1

    if converted_episodes == 0:
        raise ValueError(f"No episodes converted from {layout.task_root}")

    write_tasks_file(output_root, task_registry)
    _write_info(
        output_root,
        source_info=source_info,
        modality=modality,
        fps=fps,
        robot_type=robot_type,
        total_episodes=converted_episodes,
        total_frames=total_frames,
        chunk=chunk,
    )

    return HalabConversionResult(
        dataset_root=output_root,
        num_episodes=converted_episodes,
        num_frames=total_frames,
        skipped_missing_video=skipped_missing_video,
        task=task_prompt,
        transcoded_videos=transcoded_videos,
    )


def _transcode_h264(src: Path, dst: Path) -> None:
    """Re-encode as H.264 for GR00T's torchcodec decoder."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg or omit --transcode-h264 "
            "to copy the source MP4 as-is."
        )
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(dst),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}:\n{result.stderr[-2000:]}")


def _episode_vectors(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "observation.state" not in df.columns or "action.latent_state" not in df.columns:
        raise ValueError(
            "HA-Lab parquet must contain observation.state and action.latent_state; "
            f"got {list(df.columns)}"
        )
    source_state = stack_float_column(df["observation.state"], expected_dim=76)
    latent = stack_float_column(df["action.latent_state"], expected_dim=64)
    measured = None
    if "action.measured_state" in df.columns:
        measured = stack_float_column(df["action.measured_state"])
    hand_action = None
    if "action.hand_action" in df.columns:
        hand_action = stack_float_column(df["action.hand_action"])
    return (
        build_sonic_state(source_state, measured),
        build_sonic_action(latent, hand_action, source_state),
    )


def _timestamps(df: pd.DataFrame, fps: float) -> np.ndarray:
    if "timestamp" in df.columns:
        values = np.asarray(df["timestamp"], dtype=np.float32)
        if np.isfinite(values).all() and values[-1] >= values[0]:
            return values
    return (np.arange(len(df), dtype=np.float32) / float(fps))


def _episode_index_from_name(name: str) -> int:
    stem = Path(name).stem  # episode_000000
    try:
        return int(stem.split("_")[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot parse episode index from {name}") from exc


def _source_video_path(layout: HalabSourceLayout, episode_index: int) -> Path | None:
    filename = f"episode_{episode_index:06d}.mp4"
    candidates = [
        layout.videos_dir / layout.source_video_key / filename,
        layout.videos_dir / SOURCE_VIDEO_KEY / filename,
        layout.videos_dir / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _write_info(
    output_root: Path,
    *,
    source_info: dict,
    modality: ModalitySchema,
    fps: float,
    robot_type: str,
    total_episodes: int,
    total_frames: int,
    chunk: str,
) -> None:
    video_shape = [376, 672, 3]
    source_features = source_info.get("features") or {}
    source_video = source_features.get(SOURCE_VIDEO_KEY) or source_features.get(
        "observation.images.left"
    )
    if isinstance(source_video, dict) and source_video.get("shape"):
        video_shape = list(source_video["shape"])

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * len(modality.video),
        "total_chunks": 1,
        "chunks_size": total_episodes,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": f"data/{chunk}/episode_{{episode_index:06d}}.parquet",
        "video_path": f"videos/{chunk}/{{video_key}}/episode_{{episode_index:06d}}.mp4",
        "features": {
            TARGET_VIDEO_KEY: {
                "dtype": "video",
                "shape": video_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [SONIC_STATE_DIM],
                "names": sonic_state_names(),
            },
            "action": {
                "dtype": "float32",
                "shape": [SONIC_ACTION_DIM],
                "names": sonic_action_names(),
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "next.reward": {"dtype": "float32", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
            ANNOTATION_COLUMN: {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    path = dataset_meta_dir(output_root) / "info.json"
    with path.open("w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
