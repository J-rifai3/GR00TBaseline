#!/usr/bin/env bash
# Serve a GR00T policy over ZMQ (default: latest SONIC finetune checkpoint).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ISAAC="$ROOT/Isaac-GR00T"

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "Policy serving needs Linux + NVIDIA CUDA. Run this on the GPU machine."
  exit 1
fi

if [[ ! -d "$ISAAC/.git" ]]; then
  echo "Isaac-GR00T is not cloned. Run $ROOT/setup.sh first."
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-$ROOT/checkpoints/g1_sonic}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5555}"
DEVICE="${DEVICE:-cuda:0}"

if [[ ! -e "$MODEL_PATH" && "$MODEL_PATH" != nvidia/* ]]; then
  echo "Checkpoint not found: $MODEL_PATH"
  echo "Set MODEL_PATH to a finetune output dir, or MODEL_PATH=nvidia/GR00T-N1.7-3B"
  exit 1
fi

cd "$ISAAC"
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "$MODEL_PATH" \
  --embodiment-tag UNITREE_G1_SONIC \
  --device "$DEVICE" \
  --host "$HOST" \
  --port "$PORT"
