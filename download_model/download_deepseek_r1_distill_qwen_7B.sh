#!/bin/bash

# Download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B from Hugging Face into the TraceGuard model directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACEGUARD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ID="${REPO_ID:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
MODEL_DIR="${MODEL_DIR:-$TRACEGUARD_ROOT/model/DeepSeek-R1-Distill-Qwen-7B}"
REVISION="${REVISION:-main}"
export REPO_ID MODEL_DIR REVISION

mkdir -p "$MODEL_DIR"

echo "=========================================="
echo "Downloading Hugging Face model"
echo "Repository: $REPO_ID"
echo "Revision: $REVISION"
echo "Target: $MODEL_DIR"
echo "=========================================="

if command -v huggingface-cli >/dev/null 2>&1; then
    hf_token_args=()
    if [ -n "${HF_TOKEN:-}" ]; then
        hf_token_args=(--token "$HF_TOKEN")
    fi

    huggingface-cli download "$REPO_ID" \
        --revision "$REVISION" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False \
        "${hf_token_args[@]}"
elif python3 -c "import huggingface_hub" >/dev/null 2>&1; then
    python3 - <<PY_DOWNLOAD
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["REPO_ID"],
    revision=os.environ["REVISION"],
    local_dir=os.environ["MODEL_DIR"],
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN") or None,
)
PY_DOWNLOAD
else
    echo "Error: install huggingface_hub first: pip install huggingface_hub"
    exit 1
fi

echo "Download completed: $MODEL_DIR"
