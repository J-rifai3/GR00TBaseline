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
from gr00t_baseline.raw_io import RawEpisodeLoader
from gr00t_baseline.schema import ModalitySchema
from gr00t_baseline.validator import validate_dataset


@dataclass
class InspectArgs:
    raw_root: Path = Path("data/raw")
    """Directory containing episode_* folders."""

    suggest_modality: bool = True
    """Print a draft modality.json based on the first episode."""


@dataclass
class ConvertArgs:
    raw_root: Path = Path("data/raw")
    """Input directory with episode_* folders."""

    output_root: Path = Path("data/processed/my_dataset")
    """Output GR00T LeRobot v2 dataset directory."""

    modality_config: Path = Path("configs/modality_template.json")
    """Path to meta/modality.json template."""

    overwrite: bool = False
    """Replace output directory if it exists."""

    robot_type: str = "custom_teleop"
    """Robot identifier stored in meta/info.json."""


@dataclass
class ValidateArgs:
    dataset_root: Path = Path("data/processed/my_dataset")
    """Dataset root containing meta/, data/, videos/."""

    frame_tolerance: int = 2
    """Allowed video vs parquet frame count difference."""


@dataclass
class SplitArgs:
    dataset_root: Path = Path("data/processed/my_dataset")
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
