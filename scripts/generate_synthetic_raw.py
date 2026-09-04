#!/usr/bin/env python3
"""Generate synthetic raw teleop episodes for pipeline smoke-testing."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import tyro


def main(
    output_dir: Path = Path("data/raw"),
    num_episodes: int = 3,
    episode_length: int = 60,
    state_dim: int = 6,
    action_dim: int = 6,
    fps: float = 30.0,
    task: str = "pick and place the cube",
    cameras: tuple[str, ...] = ("front", "wrist"),
    width: int = 64,
    height: int = 64,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in range(num_episodes):
        ep_dir = output_dir / f"episode_{ep_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        t = np.linspace(0, 2 * np.pi, episode_length, dtype=np.float32)
        states = np.stack(
            [0.1 * np.sin(t + i * 0.3) for i in range(state_dim)],
            axis=1,
        ).astype(np.float32)
        actions = np.roll(states, shift=-1, axis=0)
        actions[-1] = actions[-2]

        np.save(ep_dir / "states.npy", states)
        np.save(ep_dir / "actions.npy", actions)

        with (ep_dir / "metadata.json").open("w") as f:
            json.dump({"task": task, "fps": fps}, f, indent=2)
            f.write("\n")

        for cam_idx, cam in enumerate(cameras):
            video_path = ep_dir / f"{cam}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            for frame_idx in range(episode_length):
                color = (
                    int(127 + 127 * np.sin(frame_idx * 0.1 + cam_idx)),
                    int(127 + 127 * np.sin(frame_idx * 0.1 + ep_idx)),
                    80,
                )
                frame = np.full((height, width, 3), color, dtype=np.uint8)
                writer.write(frame)
            writer.release()

        print(f"Wrote {ep_dir} ({episode_length} steps, cameras={list(cameras)})")

    print(f"\nSynthetic raw data ready at {output_dir}")
    print("Next steps:")
    print("  gr00t-inspect --raw-root data/raw")
    print("  gr00t-convert --raw-root data/raw --output-root data/processed/demo")
    print("  gr00t-validate --dataset-root data/processed/demo")


if __name__ == "__main__":
    tyro.cli(main)
