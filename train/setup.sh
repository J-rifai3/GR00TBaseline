#!/usr/bin/env bash
# Clone Isaac-GR00T and create its uv venv. Linux + NVIDIA CUDA only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ISAAC="$ROOT/Isaac-GR00T"

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "Isaac-GR00T needs Linux + NVIDIA CUDA. Run this script on the GPU machine."
  echo "This Mac should only use ../preprocess/ for conversion."
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required"
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is required (sudo apt install git-lfs && git lfs install)"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [[ ! -d "$ISAAC/.git" ]]; then
  echo "Cloning NVIDIA/Isaac-GR00T into $ISAAC"
  git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T "$ISAAC"
else
  echo "Using existing clone at $ISAAC"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Warning: ffmpeg not found. torchcodec needs FFmpeg 4-7. Install with: sudo apt-get install -y ffmpeg"
fi

cd "$ISAAC"
uv sync --python 3.12
uv run python -c "import gr00t; print('GR00T installed successfully')"

echo
echo "Next:"
echo "  cd $ISAAC && uv run huggingface-cli login"
echo "  (request access to https://huggingface.co/nvidia/Cosmos-Reason2-2B first)"
echo "  then from $ROOT: ./finetune_sonic.sh"
