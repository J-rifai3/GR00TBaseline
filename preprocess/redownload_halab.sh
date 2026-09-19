#!/usr/bin/env bash
# Re-download the gitignored HA-Lab tasks from Hugging Face.
#
#   ./redownload_halab.sh
#   ./redownload_halab.sh --convert
#   ./redownload_halab.sh --convert --transcode-h264
#   ./redownload_halab.sh --task pick_tennis_ball_place_black_basket
#
# Writes raw data to ../data/raw/halab/<task>/ (and processed/ if --convert).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
TASK_LIST="$REPO_ROOT/train/datasets.txt"
VENV="$ROOT/.venv"

CONVERT=0
TRANSCODE=0
VALIDATE=0
TASKS=()

usage() {
  cat <<'EOF'
Re-download HA-Lab tasks that are gitignored under data/.

Usage: ./redownload_halab.sh [options]

  --convert           After download, convert to GR00T LeRobot v2 in data/processed/
  --transcode-h264    With --convert, re-encode videos as H.264 (needs ffmpeg)
  --validate          After convert, run gr00t-validate on each task
  --task NAME         Download only this task (repeatable). Default: train/datasets.txt
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --convert) CONVERT=1; shift ;;
    --transcode-h264) TRANSCODE=1; shift ;;
    --validate) VALIDATE=1; shift ;;
    --task)
      [[ $# -ge 2 ]] || { echo "--task requires a name"; exit 1; }
      TASKS+=("$2")
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ "$TRANSCODE" -eq 1 && "$CONVERT" -eq 0 ]]; then
  echo "--transcode-h264 requires --convert"
  exit 1
fi
if [[ "$VALIDATE" -eq 1 && "$CONVERT" -eq 0 ]]; then
  echo "--validate requires --convert"
  exit 1
fi

if [[ ${#TASKS[@]} -eq 0 ]]; then
  if [[ ! -f "$TASK_LIST" ]]; then
    echo "No task list at $TASK_LIST and no --task given."
    exit 1
  fi
  while IFS= read -r task || [[ -n "$task" ]]; do
    [[ -z "$task" || "$task" == \#* ]] && continue
    TASKS+=("$task")
  done < "$TASK_LIST"
fi

if [[ ! -x "$VENV/bin/gr00t-download-halab" ]]; then
  echo "Preprocess venv is missing. From preprocess/: ./setup.sh"
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "Downloading ${#TASKS[@]} HA-Lab task(s) into $REPO_ROOT/data/raw/halab/"
for task in "${TASKS[@]}"; do
  echo
  echo "=== download $task ==="
  gr00t-download-halab --task "$task" --output-root "$REPO_ROOT/data/raw"

  if [[ "$CONVERT" -eq 1 ]]; then
    echo "=== convert $task ==="
    CONVERT_ARGS=(
      --source-root "$REPO_ROOT/data/raw/halab/$task"
      --output-root "$REPO_ROOT/data/processed/$task"
      --overwrite
    )
    if [[ "$TRANSCODE" -eq 1 ]]; then
      CONVERT_ARGS+=(--transcode-h264)
    fi
    gr00t-convert-halab "${CONVERT_ARGS[@]}"
  fi

  if [[ "$VALIDATE" -eq 1 ]]; then
    echo "=== validate $task ==="
    gr00t-validate --dataset-root "$REPO_ROOT/data/processed/$task"
  fi
done

echo
echo "Done. Raw: $REPO_ROOT/data/raw/halab/"
if [[ "$CONVERT" -eq 1 ]]; then
  echo "Processed: $REPO_ROOT/data/processed/"
fi
