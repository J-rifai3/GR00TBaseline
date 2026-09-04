"""GR00T LeRobot v2 schema helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModalitySlice:
    """Index range for a named slice inside a concatenated state/action vector."""

    start: int
    end: int

    @property
    def dim(self) -> int:
        return self.end - self.start


@dataclass
class ModalitySchema:
    """Parsed meta/modality.json."""

    state: dict[str, ModalitySlice] = field(default_factory=dict)
    action: dict[str, ModalitySlice] = field(default_factory=dict)
    video: dict[str, str] = field(default_factory=dict)  # short_key -> original_key
    annotation: dict[str, str] = field(default_factory=dict)  # ann_key -> original_key

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModalitySchema:
        def parse_slices(section: dict[str, Any]) -> dict[str, ModalitySlice]:
            return {
                key: ModalitySlice(start=v["start"], end=v["end"])
                for key, v in section.items()
            }

        video = {
            key: v["original_key"]
            for key, v in data.get("video", {}).items()
        }
        annotation = {
            key: v.get("original_key", key)
            for key, v in data.get("annotation", {}).items()
        }
        return cls(
            state=parse_slices(data.get("state", {})),
            action=parse_slices(data.get("action", {})),
            video=video,
            annotation=annotation,
        )

    @classmethod
    def load(cls, path: Path) -> ModalitySchema:
        with path.open() as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": {k: {"start": v.start, "end": v.end} for k, v in self.state.items()},
            "action": {k: {"start": v.start, "end": v.end} for k, v in self.action.items()},
            "video": {k: {"original_key": v} for k, v in self.video.items()},
            "annotation": {k: {"original_key": v} for k, v in self.annotation.items()},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    @property
    def state_dim(self) -> int:
        if not self.state:
            return 0
        return max(s.end for s in self.state.values())

    @property
    def action_dim(self) -> int:
        if not self.action:
            return 0
        return max(s.end for s in self.action.values())


# Required parquet columns for GR00T LeRobot v2
REQUIRED_PARQUET_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "episode_index",
    "index",
    "next.reward",
    "next.done",
)

# Default dataset metadata written to meta/info.json
DEFAULT_FPS = 30
DEFAULT_ROBOT_TYPE = "custom_teleop"
