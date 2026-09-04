"""Validate GR00T LeRobot v2 datasets before fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import jsonlines
import pandas as pd

from gr00t_baseline.paths import dataset_meta_dir, parquet_path, video_path
from gr00t_baseline.schema import REQUIRED_PARQUET_COLUMNS, ModalitySchema


@dataclass
class ValidationIssue:
    level: str  # "error" | "warning"
    message: str


@dataclass
class ValidationReport:
    dataset_root: Path
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    def error(self, message: str) -> None:
        self.issues.append(ValidationIssue("error", message))

    def warn(self, message: str) -> None:
        self.issues.append(ValidationIssue("warning", message))


def validate_dataset(dataset_root: Path, *, frame_tolerance: int = 2) -> ValidationReport:
    report = ValidationReport(dataset_root=dataset_root)

    modality_path = dataset_meta_dir(dataset_root) / "modality.json"
    if not modality_path.exists():
        report.error(f"Missing required file: {modality_path}")
        return report

    modality = ModalitySchema.load(modality_path)

    for meta_file in ("info.json", "episodes.jsonl", "tasks.jsonl"):
        path = dataset_meta_dir(dataset_root) / meta_file
        if not path.exists():
            report.error(f"Missing meta file: {path}")

    episodes_path = dataset_meta_dir(dataset_root) / "episodes.jsonl"
    if not episodes_path.exists():
        return report

    with jsonlines.open(episodes_path) as reader:
        episodes = list(reader)

    if not episodes:
        report.error("episodes.jsonl is empty")
        return report

    for ep in episodes:
        ep_idx = ep["episode_index"]
        length = ep["length"]
        pq = parquet_path(dataset_root, ep_idx)
        if not pq.exists():
            report.error(f"Missing parquet for episode {ep_idx}: {pq}")
            continue

        df = pd.read_parquet(pq)
        missing_cols = set(REQUIRED_PARQUET_COLUMNS) - set(df.columns)
        if missing_cols:
            report.error(f"Episode {ep_idx} parquet missing columns: {sorted(missing_cols)}")

        if len(df) != length:
            report.error(
                f"Episode {ep_idx}: episodes.jsonl length={length} but parquet rows={len(df)}"
            )

        if "observation.state" in df.columns:
            state_lens = df["observation.state"].apply(len)
            if not (state_lens == modality.state_dim).all():
                report.error(
                    f"Episode {ep_idx}: observation.state dim mismatch "
                    f"(expected {modality.state_dim})"
                )

        if "action" in df.columns:
            action_lens = df["action"].apply(len)
            if not (action_lens == modality.action_dim).all():
                report.error(
                    f"Episode {ep_idx}: action dim mismatch (expected {modality.action_dim})"
                )

        for short_key, original_key in modality.video.items():
            vid = video_path(dataset_root, original_key, ep_idx)
            if not vid.exists():
                report.error(f"Episode {ep_idx} missing video '{short_key}': {vid}")
                continue

            cap = cv2.VideoCapture(str(vid))
            if not cap.isOpened():
                report.error(f"Episode {ep_idx}: cannot open video {vid}")
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if abs(frame_count - length) > frame_tolerance:
                report.warn(
                    f"Episode {ep_idx} camera '{short_key}': "
                    f"video frames ({frame_count}) != parquet rows ({length}) "
                    f"[tolerance={frame_tolerance}]"
                )

    return report
