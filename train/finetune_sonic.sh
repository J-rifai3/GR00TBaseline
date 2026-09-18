#!/usr/bin/env bash
# Fine-tune nvidia/GR00T-N1.7-3B on processed HA-Lab SONIC tasks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
ISAAC="$ROOT/Isaac-GR00T"
DATA="$REPO_ROOT/data/processed"
CONFIG="$ROOT/configs/embodiment_config_g1_sonic.example.py"
TASK_LIST="$ROOT/datasets.txt"

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "Fine-tuning needs Linux + NVIDIA CUDA. Run this on the GPU machine."
  exit 1
fi

if [[ ! -d "$ISAAC/.git" ]]; then
  echo "Isaac-GR00T is not cloned. Run $ROOT/setup.sh first."
  exit 1
fi

if [[ -z "${DATASET_PATH:-}" ]]; then
  DATASET_PATH=""
  while IFS= read -r task || [[ -n "$task" ]]; do
    [[ -z "$task" || "$task" == \#* ]] && continue
    if [[ -d "$DATA/$task" ]]; then
      if [[ -n "$DATASET_PATH" ]]; then
        DATASET_PATH="${DATASET_PATH}:${DATA}/${task}"
      else
        DATASET_PATH="${DATA}/${task}"
      fi
    else
      echo "Skipping missing dataset: $DATA/$task"
    fi
  done < "$TASK_LIST"
fi

if [[ -z "$DATASET_PATH" ]]; then
  echo "No processed datasets found under $DATA"
  echo "Copy ../data/processed from the preprocess machine, or set DATASET_PATH."
  exit 1
fi

NUM_GPUS="${NUM_GPUS:-1}"
MAX_STEPS="${MAX_STEPS:-2000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/checkpoints/g1_sonic}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"

mkdir -p "$OUTPUT_DIR"

echo "dataset-path: $DATASET_PATH"
echo "output-dir:   $OUTPUT_DIR"
echo "gpus:         $NUM_GPUS"

cd "$ISAAC"

LAUNCH=(
  gr00t/experiment/launch_finetune.py
  --base-model-path "$BASE_MODEL"
  --dataset-path "$DATASET_PATH"
  --embodiment-tag UNITREE_G1_SONIC
  --modality-config-path "$CONFIG"
  --num-gpus "$NUM_GPUS"
  --output-dir "$OUTPUT_DIR"
  --max-steps "$MAX_STEPS"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --dataloader-num-workers "$DATALOADER_WORKERS"
)

if [[ "$NUM_GPUS" -gt 1 ]]; then
  uv run torchrun --nproc_per_node="$NUM_GPUS" --master_port="${MASTER_PORT:-29500}" "${LAUNCH[@]}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python "${LAUNCH[@]}"
fi
