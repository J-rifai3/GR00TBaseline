"""Download one HA-Lab task from Hugging Face."""

from __future__ import annotations

from pathlib import Path

from gr00t_baseline.halab import DEFAULT_TASK, HF_REPO_ID, HF_TASK_PREFIX


def download_halab_task(
    task: str = DEFAULT_TASK,
    output_root: Path = Path("data/raw"),
    *,
    include_videos: bool = True,
    include_annotations: bool = True,
    include_aligned_extras: bool = False,
    include_raw_streams: bool = False,
    repo_id: str = HF_REPO_ID,
) -> Path:
    """Download a single task folder. Returns the local task root.

    Raw streams are huge (per-episode tar/jsonl) and are not required for GR00T.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download HA-Lab data. "
            "Install with: pip install -e '.[halab]'"
        ) from exc

    task_prefix = f"{HF_TASK_PREFIX}/{task}"
    allow_patterns = [
        f"{task_prefix}/data/**",
        f"{task_prefix}/meta/**",
    ]
    if include_videos:
        allow_patterns.append(f"{task_prefix}/videos/**")
    if include_annotations:
        allow_patterns.append(f"{task_prefix}/preprocess/annotation/**")
    if include_aligned_extras:
        allow_patterns.append(f"{task_prefix}/aligned_extras/**")
    if include_raw_streams:
        allow_patterns.append(f"{task_prefix}/raw_streams/**")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=str(output_root),
    )
    task_root = output_root / HF_TASK_PREFIX / task
    if not (task_root / "meta" / "info.json").exists():
        raise FileNotFoundError(
            f"Download finished but {task_root}/meta/info.json is missing. "
            f"Check that task '{task}' exists in {repo_id}."
        )
    return task_root


def list_halab_tasks(repo_id: str = HF_REPO_ID) -> list[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to list HA-Lab tasks. "
            "Install with: pip install -e '.[halab]'"
        ) from exc

    api = HfApi()
    try:
        items = api.list_repo_tree(
            repo_id, repo_type="dataset", path_in_repo=HF_TASK_PREFIX, recursive=False
        )
    except TypeError:
        items = api.list_repo_tree(repo_id, repo_type="dataset", recursive=False)
    names = []
    for item in items:
        path = getattr(item, "path", "") or ""
        item_type = getattr(item, "type", None) or type(item).__name__
        is_dir = item_type in {"directory", "RepoFolder"} or path.count("/") == 1
        if HF_TASK_PREFIX not in path:
            continue
        name = path[len(HF_TASK_PREFIX) :].lstrip("/")
        if "/" in name or not name or name.endswith(".parquet"):
            continue
        if is_dir or item_type == "directory":
            names.append(name)
    return sorted(set(names))
