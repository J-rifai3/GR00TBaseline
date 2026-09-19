#!/usr/bin/env bash
# Re-download gitignored HA-Lab tasks from Hugging Face.
#
#   ./redownload_halab.sh                      # first 3 tasks (train/datasets.txt)
#   ./redownload_halab.sh --num-tasks 1
#   ./redownload_halab.sh -n 8 --convert
#   ./redownload_halab.sh --task some_task     # exact name(s) only
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
NUM_TASKS=3
NUM_TASKS_SET=0
EXPLICIT_TASKS=()

usage() {
  cat <<'EOF'
Re-download HA-Lab tasks that are gitignored under data/.

Usage: ./redownload_halab.sh [options]

  -n, --num-tasks N   How many tasks to take (default: 3). Baseline names in
                      train/datasets.txt come first; the rest are filled from
                      Hugging Face (alphabetical).
  --convert           After download, convert to GR00T LeRobot v2 in data/processed/
  --transcode-h264    With --convert, re-encode videos as H.264 (needs ffmpeg)
  --validate          After convert, run gr00t-validate on each task
  --task NAME         Use this task name (repeatable). Ignores --num-tasks
                      unless you also want to cap the explicit list with -n.
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --convert) CONVERT=1; shift ;;
    --transcode-h264) TRANSCODE=1; shift ;;
    --validate) VALIDATE=1; shift ;;
    -n|--num-tasks)
      [[ $# -ge 2 ]] || { echo "--num-tasks requires an integer"; exit 1; }
      NUM_TASKS="$2"
      NUM_TASKS_SET=1
      shift 2
      ;;
    --task)
      [[ $# -ge 2 ]] || { echo "--task requires a name"; exit 1; }
      EXPLICIT_TASKS+=("$2")
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if ! [[ "$NUM_TASKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--num-tasks must be a positive integer, got: $NUM_TASKS"
  exit 1
fi

if [[ "$TRANSCODE" -eq 1 && "$CONVERT" -eq 0 ]]; then
  echo "--transcode-h264 requires --convert"
  exit 1
fi
if [[ "$VALIDATE" -eq 1 && "$CONVERT" -eq 0 ]]; then
  echo "--validate requires --convert"
  exit 1
fi

if [[ ! -x "$VENV/bin/gr00t-download-halab" ]]; then
  echo "Preprocess venv is missing. From preprocess/: ./setup.sh"
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

read_baseline_tasks() {
  local tasks=()
  if [[ -f "$TASK_LIST" ]]; then
    while IFS= read -r task || [[ -n "$task" ]]; do
      [[ -z "$task" || "$task" == \#* ]] && continue
      tasks+=("$task")
    done < "$TASK_LIST"
  fi
  printf '%s\n' "${tasks[@]}"
}

TASKS=()
if [[ ${#EXPLICIT_TASKS[@]} -gt 0 ]]; then
  TASKS=("${EXPLICIT_TASKS[@]}")
  if [[ "$NUM_TASKS_SET" -eq 1 && ${#TASKS[@]} -gt "$NUM_TASKS" ]]; then
    TASKS=("${TASKS[@]:0:$NUM_TASKS}")
  fi
else
  declare -A seen=()
  while IFS= read -r task; do
    [[ -z "$task" || -n "${seen[$task]:-}" ]] && continue
    seen["$task"]=1
    TASKS+=("$task")
  done < <(read_baseline_tasks)

  if [[ ${#TASKS[@]} -lt "$NUM_TASKS" ]]; then
    echo "Listing HA-Lab tasks on Hugging Face to fill --num-tasks $NUM_TASKS..."
    while IFS= read -r task; do
      [[ -z "$task" || -n "${seen[$task]:-}" ]] && continue
      seen["$task"]=1
      TASKS+=("$task")
      [[ ${#TASKS[@]} -ge "$NUM_TASKS" ]] && break
    done < <(gr00t-download-halab --list-tasks)
  fi

  if [[ ${#TASKS[@]} -gt "$NUM_TASKS" ]]; then
    TASKS=("${TASKS[@]:0:$NUM_TASKS}")
  fi
fi

if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "No tasks to download."
  exit 1
fi

if [[ ${#TASKS[@]} -lt "$NUM_TASKS" && ${#EXPLICIT_TASKS[@]} -eq 0 ]]; then
  echo "Only found ${#TASKS[@]} task(s); requested $NUM_TASKS."
fi

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
