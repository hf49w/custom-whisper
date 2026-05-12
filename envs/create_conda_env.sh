#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-custom-whisper-mm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[INFO] conda env already exists: $ENV_NAME"
else
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"

conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio "pytorch-cuda=$CUDA_VERSION"
conda install -y -c conda-forge ffmpeg

python -m pip install --upgrade pip
python -m pip install \
  "transformers>=4.40,<5" \
  "huggingface_hub>=0.23" \
  "numpy>=1.24,<2.0" \
  "numba>=0.58" \
  "tqdm>=4.66" \
  "pillow>=10.0" \
  "tiktoken>=0.6" \
  "regex>=2024.0.0" \
  "more-itertools>=10.0"

python - <<'PY'
import torch
import transformers
import PIL
import numba
import tiktoken
print("[DONE] torch", torch.__version__)
print("[DONE] cuda_available", torch.cuda.is_available())
print("[DONE] transformers", transformers.__version__)
print("[DONE] pillow", PIL.__version__)
print("[DONE] numba", numba.__version__)
print("[DONE] tiktoken", tiktoken.__version__)
PY
