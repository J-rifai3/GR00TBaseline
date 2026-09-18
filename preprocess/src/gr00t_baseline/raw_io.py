"""Load raw teleoperation episodes from disk.

Expected layout (customize RawEpisodeLoader if your format differs):

    raw/
      episode_000000/
        metadata.json   # {"task": "pick cube", "fps": 30}
        states.npy      # float32 array [T, state_dim]
        actions.npy     # float32 array [T, action_dim]
        front.mp4
        wrist.mp4
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EPISODE_DIR_PATTERN = re.compile(r"episode_(\d+)$")


@dataclass
class RawEpisode:
    episode_index: int
    task: str
    fps: float
    states: np.ndarray  # [T, state_dim]
    actions: np.ndarray  # [T, action_dim]
    videos: dict[str, Path] = field(default_factory=dict)  # camera_name -> mp4 path
    source_dir: Path | None = None

    @property
    def length(self) -> int:
        return int(self.states.shape[0])

    def summary(self) -> dict:
        return {
            "episode_index": self.episode_index,
            "task": self.task,
            "fps": self.fps,
            "length": self.length,
            "state_shape": list(self.states.shape),
            "action_shape": list(self.actions.shape),
            "cameras": sorted(self.videos.keys()),
            "source_dir": str(self.source_dir) if self.source_dir else None,
        }


class RawEpisodeLoader:
    """Discover and load episode folders from a raw teleop directory."""

    def __init__(
        self,
        raw_root: Path,
        state_file: str = "states.npy",
        action_file: str = "actions.npy",
        metadata_file: str = "metadata.json",
        video_extensions: tuple[str, ...] = (".mp4", ".MP4"),
    ):
        self.raw_root = raw_root
        self.state_file = state_file
        self.action_file = action_file
        self.metadata_file = metadata_file
        self.video_extensions = video_extensions

    def list_episode_dirs(self) -> list[Path]:
        if not self.raw_root.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_root}")

        dirs = []
        for path in sorted(self.raw_root.iterdir()):
            if path.is_dir() and EPISODE_DIR_PATTERN.match(path.name):
                dirs.append(path)
        return dirs

    def load(self, episode_dir: Path) -> RawEpisode:
        match = EPISODE_DIR_PATTERN.match(episode_dir.name)
        if not match:
            raise ValueError(f"Not an episode directory: {episode_dir}")

        episode_index = int(match.group(1))
        metadata_path = episode_dir / self.metadata_file
        states_path = episode_dir / self.state_file
        actions_path = episode_dir / self.action_file

        for required in (metadata_path, states_path, actions_path):
            if not required.exists():
                raise FileNotFoundError(f"Missing required file: {required}")

        with metadata_path.open() as f:
            metadata = json.load(f)

        states = np.load(states_path).astype(np.float32)
        actions = np.load(actions_path).astype(np.float32)

        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError(
                f"{episode_dir.name}: states/actions must be 2D [T, dim], "
                f"got {states.shape=} {actions.shape=}"
            )
        if states.shape[0] != actions.shape[0]:
            raise ValueError(
                f"{episode_dir.name}: state/action length mismatch "
                f"({states.shape[0]} vs {actions.shape[0]})"
            )

        videos: dict[str, Path] = {}
        for path in sorted(episode_dir.iterdir()):
            if path.suffix in self.video_extensions and path.is_file():
                videos[path.stem] = path

        return RawEpisode(
            episode_index=episode_index,
            task=str(metadata.get("task", "unknown task")),
            fps=float(metadata.get("fps", 30.0)),
            states=states,
            actions=actions,
            videos=videos,
            source_dir=episode_dir,
        )

    def load_all(self) -> list[RawEpisode]:
        return [self.load(d) for d in self.list_episode_dirs()]
