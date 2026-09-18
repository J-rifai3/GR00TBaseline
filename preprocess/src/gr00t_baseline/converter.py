"""Convert raw teleop directories to GR00T LeRobot v2 datasets."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from gr00t_baseline.episode_writer import (
    EpisodeWriter,
    append_episode_record,
    ensure_dataset_dirs,
    register_task,
    write_dataset_info,
    write_tasks_file,
)
from gr00t_baseline.paths import dataset_meta_dir
from gr00t_baseline.raw_io import RawEpisodeLoader
from gr00t_baseline.schema import ModalitySchema


@dataclass
class ConversionResult:
    dataset_root: Path
    num_episodes: int
    num_frames: int
    num_tasks: int


def convert_raw_to_lerobot(
    raw_root: Path,
    output_root: Path,
    modality: ModalitySchema,
    *,
    chunk: str = "chunk-000",
    overwrite: bool = False,
    robot_type: str = "custom_teleop",
) -> ConversionResult:
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"Output dataset already exists: {output_root}. Pass overwrite=True to replace."
            )

    output_root.mkdir(parents=True)
    modality.save(dataset_meta_dir(output_root) / "modality.json")

    loader = RawEpisodeLoader(raw_root)
    episodes = loader.load_all()
    if not episodes:
        raise ValueError(f"No episode_* directories found under {raw_root}")

    ensure_dataset_dirs(output_root, modality, chunk)

    # Clear episodes.jsonl if re-creating inside existing tree
    episodes_jsonl = dataset_meta_dir(output_root) / "episodes.jsonl"
    if episodes_jsonl.exists():
        episodes_jsonl.unlink()

    task_registry: dict[str, int] = {}
    global_index = 0
    total_frames = 0
    fps_values = []

    for episode in tqdm(episodes, desc="Converting episodes"):
        _validate_episode_dims(episode, modality)
        task_index = register_task(output_root, episode.task, task_registry)

        writer = EpisodeWriter(
            dataset_root=output_root,
            modality=modality,
            chunk=chunk,
            global_index_offset=global_index,
        )
        length, global_index = writer.write_episode(episode, task_index=task_index)
        append_episode_record(output_root, episode.episode_index, episode.task, length)
        total_frames += length
        fps_values.append(episode.fps)

    write_tasks_file(output_root, task_registry)
    mean_fps = sum(fps_values) / len(fps_values)
    write_dataset_info(
        output_root,
        fps=mean_fps,
        robot_type=robot_type,
        total_episodes=len(episodes),
        total_frames=total_frames,
        chunk=chunk,
    )

    # Patch task count in info.json
    info_path = dataset_meta_dir(output_root) / "info.json"
    import json

    with info_path.open() as f:
        info = json.load(f)
    info["total_tasks"] = len(task_registry)
    info["total_videos"] = len(episodes) * len(modality.video)
    with info_path.open("w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")

    return ConversionResult(
        dataset_root=output_root,
        num_episodes=len(episodes),
        num_frames=total_frames,
        num_tasks=len(task_registry),
    )


def _validate_episode_dims(episode, modality: ModalitySchema) -> None:
    expected_state = modality.state_dim
    expected_action = modality.action_dim
    if episode.states.shape[1] != expected_state:
        raise ValueError(
            f"Episode {episode.episode_index}: state dim {episode.states.shape[1]} "
            f"!= modality.json state dim {expected_state}"
        )
    if episode.actions.shape[1] != expected_action:
        raise ValueError(
            f"Episode {episode.episode_index}: action dim {episode.actions.shape[1]} "
            f"!= modality.json action dim {expected_action}"
        )
    missing_cameras = set(modality.video.keys()) - set(episode.videos.keys())
    if missing_cameras:
        raise ValueError(
            f"Episode {episode.episode_index} missing cameras: {sorted(missing_cameras)}"
        )
