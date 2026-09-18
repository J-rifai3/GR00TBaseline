"""CLI entry points for preprocessing scripts."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import jsonlines
import tyro

from gr00t_baseline.converter import convert_raw_to_lerobot
from gr00t_baseline.halab import DEFAULT_TASK, default_sonic_modality_dict, discover_halab_task
from gr00t_baseline.halab_convert import convert_halab_task
from gr00t_baseline.halab_download import download_halab_task, list_halab_tasks
from gr00t_baseline.paths import configs_dir, data_root
from gr00t_baseline.raw_io import RawEpisodeLoader
from gr00t_baseline.schema import ModalitySchema
from gr00t_baseline.validator import validate_dataset

_DATA = data_root()
_CONFIGS = configs_dir()


@dataclass
class InspectArgs:
    raw_root: Path = _DATA / "raw"
    """Directory containing episode_* folders."""

    suggest_modality: bool = True
    """Print a draft modality.json based on the first episode."""


@dataclass
class ConvertArgs:
    raw_root: Path = _DATA / "raw"
    """Input directory with episode_* folders."""

    output_root: Path = _DATA / "processed" / "my_dataset"
    """Output GR00T LeRobot v2 dataset directory."""

    modality_config: Path = _CONFIGS / "modality_template.json"
    """Path to meta/modality.json template."""

    overwrite: bool = False
    """Replace output directory if it exists."""

    robot_type: str = "custom_teleop"
    """Robot identifier stored in meta/info.json."""


@dataclass
class ValidateArgs:
    dataset_root: Path = _DATA / "processed" / "my_dataset"
    """Dataset root containing meta/, data/, videos/."""

    frame_tolerance: int = 2
    """Allowed video vs parquet frame count difference."""


@dataclass
class DownloadHalabArgs:
    task: str = DEFAULT_TASK
    """HA-Lab task folder name under halab/ on Hugging Face."""

    output_root: Path = _DATA / "raw"
    """Local directory that will contain halab/<task>/."""

    list_tasks: bool = False
    """Print available HA-Lab task names and exit."""

    include_videos: bool = True
    """Download MP4s (needed for GR00T)."""

    include_annotations: bool = True
    """Download preprocess/annotation parquet (optional language extras)."""

    include_aligned_extras: bool = False
    """Download Boxer/BEV extras. Not required for GR00T/SONIC."""

    include_raw_streams: bool = False
    """Download raw tar/jsonl streams. Large; not required for GR00T/SONIC."""


@dataclass
class InspectHalabArgs:
    source_root: Path = _DATA / "raw" / "halab" / DEFAULT_TASK
    """Local HA-Lab task directory (contains data/, meta/, videos/)."""


@dataclass
class ConvertHalabArgs:
    source_root: Path = _DATA / "raw" / "halab" / DEFAULT_TASK
    """Local HA-Lab task directory."""

    output_root: Path = _DATA / "processed" / DEFAULT_TASK
    """Output GR00T LeRobot v2 dataset directory."""

    modality_config: Path | None = None
    """Optional modality.json. Defaults to the bundled UNITREE_G1_SONIC slices."""

    overwrite: bool = False
    """Replace output directory if it exists."""

    max_episodes: int | None = None
    """Convert only the first N episodes (useful for a smoke test)."""

    symlink_videos: bool = False
    """Symlink MP4s instead of copying them."""

    require_videos: bool = True
    """Fail if an episode is missing its camera MP4."""

    transcode_h264: bool = False
    """Re-encode videos as H.264 (recommended for GR00T torchcodec). Requires ffmpeg."""


@dataclass
class SplitArgs:
    dataset_root: Path = _DATA / "processed" / "my_dataset"
    """Dataset to split."""

    val_ratio: float = 0.1
    """Fraction of episodes reserved for validation."""

    seed: int = 42
    """Random seed for reproducible splits."""

    output_dir: Path | None = None
    """Optional: copy split subsets to separate dataset roots."""


def inspect_main() -> None:
    """Inspect raw teleoperation episode directories."""
    args = tyro.cli(InspectArgs)

    loader = RawEpisodeLoader(args.raw_root)
    episode_dirs = loader.list_episode_dirs()
    if not episode_dirs:
        print(f"No episode_* directories found under {args.raw_root}")
        return

    print(f"Found {len(episode_dirs)} episode(s) under {args.raw_root}\n")

    all_summaries = []
    for ep_dir in episode_dirs:
        episode = loader.load(ep_dir)
        summary = episode.summary()
        all_summaries.append(summary)
        print(json.dumps(summary, indent=2))
        print()

    if args.suggest_modality and all_summaries:
        first = loader.load(episode_dirs[0])
        state_dim = first.states.shape[1]
        action_dim = first.actions.shape[1]
        cameras = sorted(first.videos.keys())

        draft = {
            "state": {"joints": {"start": 0, "end": state_dim}},
            "action": {"joints": {"start": 0, "end": action_dim}},
            "video": {
                cam: {"original_key": f"observation.images.{cam}"} for cam in cameras
            },
            "annotation": {
                "human.task_description": {"original_key": "task_index"},
            },
        }
        print("Suggested modality.json (edit keys/slices for your robot):")
        print(json.dumps(draft, indent=2))


def download_halab_main() -> None:
    """Download one HA-Lab task from Hugging Face."""
    args = tyro.cli(DownloadHalabArgs)
    if args.list_tasks:
        for name in list_halab_tasks():
            print(name)
        return

    task_root = download_halab_task(
        task=args.task,
        output_root=args.output_root,
        include_videos=args.include_videos,
        include_annotations=args.include_annotations,
        include_aligned_extras=args.include_aligned_extras,
        include_raw_streams=args.include_raw_streams,
    )
    print(f"Downloaded HA-Lab task '{args.task}' -> {task_root}")
    print("Next:")
    print(f"  gr00t-inspect-halab --source-root {task_root}")
    print(
        f"  gr00t-convert-halab --source-root {task_root} "
        f"--output-root {_DATA / 'processed' / args.task}"
    )


def inspect_halab_main() -> None:
    """Print HA-Lab task metadata and a sample parquet schema."""
    args = tyro.cli(InspectHalabArgs)
    import pandas as pd

    from gr00t_baseline.halab import list_episode_parquets, load_info, load_task_prompt

    layout = discover_halab_task(args.source_root)
    info = load_info(layout.task_root)
    parquets = list_episode_parquets(layout)
    df = pd.read_parquet(parquets[0])
    first_state = df["observation.state"].iloc[0]
    print(json.dumps(
        {
            "task_root": str(layout.task_root),
            "task_prompt": load_task_prompt(layout.task_root),
            "robot_type": info.get("robot_type"),
            "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"),
            "episodes_on_disk": len(parquets),
            "source_video_key": layout.source_video_key,
            "parquet_columns": list(df.columns),
            "observation.state_dim": len(first_state),
            "action.latent_state_dim": len(df["action.latent_state"].iloc[0])
            if "action.latent_state" in df.columns
            else None,
            "action.hand_action_dim": len(df["action.hand_action"].iloc[0])
            if "action.hand_action" in df.columns
            else None,
        },
        indent=2,
    ))
    print("\nSuggested GR00T/SONIC modality.json:")
    print(json.dumps(default_sonic_modality_dict(), indent=2))


def convert_halab_main() -> None:
    """Convert a local HA-Lab task into GR00T LeRobot v2 / SONIC format."""
    args = tyro.cli(ConvertHalabArgs)
    if args.modality_config is None:
        modality = ModalitySchema.from_dict(default_sonic_modality_dict())
    else:
        modality = ModalitySchema.load(args.modality_config)

    result = convert_halab_task(
        source_root=args.source_root,
        output_root=args.output_root,
        modality=modality,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
        symlink_videos=args.symlink_videos,
        require_videos=args.require_videos,
        transcode_h264=args.transcode_h264,
    )
    print(
        f"Converted {result.num_episodes} episodes ({result.num_frames} frames) "
        f"for task '{result.task}' -> {result.dataset_root}"
    )
    if result.skipped_missing_video:
        print(f"Warning: {result.skipped_missing_video} episode(s) had no video")
    print(f"Next: gr00t-validate --dataset-root {result.dataset_root}")


def convert_main() -> None:
    """Convert raw teleop episodes to GR00T LeRobot v2 format."""
    args = tyro.cli(ConvertArgs)
    modality = ModalitySchema.load(args.modality_config)

    result = convert_raw_to_lerobot(
        raw_root=args.raw_root,
        output_root=args.output_root,
        modality=modality,
        overwrite=args.overwrite,
        robot_type=args.robot_type,
    )

    print(
        f"Converted {result.num_episodes} episodes "
        f"({result.num_frames} frames, {result.num_tasks} tasks) -> {result.dataset_root}"
    )


def validate_main() -> None:
    """Validate a GR00T LeRobot v2 dataset."""
    args = tyro.cli(ValidateArgs)
    report = validate_dataset(args.dataset_root, frame_tolerance=args.frame_tolerance)

    if not report.issues:
        print(f"OK: {args.dataset_root} passed validation.")
        return

    for issue in report.issues:
        prefix = "ERROR" if issue.level == "error" else "WARN"
        print(f"{prefix}: {issue.message}")

    if not report.ok:
        raise SystemExit(1)


def split_main() -> None:
    """Split episodes into train/val subsets by writing meta/splits.json."""
    args = tyro.cli(SplitArgs)

    episodes_path = args.dataset_root / "meta" / "episodes.jsonl"
    with jsonlines.open(episodes_path) as reader:
        episodes = list(reader)

    indices = [ep["episode_index"] for ep in episodes]
    rng = random.Random(args.seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * args.val_ratio)) if len(shuffled) > 1 else 0
    val_set = set(shuffled[:n_val])
    train_set = [i for i in indices if i not in val_set]
    val_list = sorted(val_set)

    splits = {
        "train": train_set,
        "val": val_list,
    }
    splits_path = args.dataset_root / "meta" / "splits.json"
    with splits_path.open("w") as f:
        json.dump(splits, f, indent=2)
        f.write("\n")

    print(f"Wrote {splits_path}")
    print(f"  train: {len(train_set)} episodes")
    print(f"  val:   {len(val_list)} episodes")

    if args.output_dir:
        _materialize_split(args.dataset_root, args.output_dir / "train", train_set)
        _materialize_split(args.dataset_root, args.output_dir / "val", val_list)
        print(f"Copied split datasets to {args.output_dir}")


def _materialize_split(src_root: Path, dst_root: Path, episode_indices: list[int]) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)

    shutil.copytree(src_root / "meta", dst_root / "meta")
    for ep_idx in episode_indices:
        pq = src_root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        if pq.exists():
            dst = dst_root / pq.relative_to(src_root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pq, dst)

        for video_root in (src_root / "videos" / "chunk-000").glob("observation.images.*"):
            src_vid = video_root / f"episode_{ep_idx:06d}.mp4"
            if src_vid.exists():
                dst = dst_root / src_vid.relative_to(src_root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_vid, dst)
