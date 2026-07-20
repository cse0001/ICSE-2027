#!/bin/bash

# Run functionality evaluations for qwen3.5-27B on MBPP-cpp and HumanEval-cpp.
# Each dataset is evaluated with origin and TraceGuard settings in serial order.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUNCTIONALITY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HARNESS_DIR="${HARNESS_DIR:-/root/bigcode-evaluation-harness}"
RESULT_ROOT="$FUNCTIONALITY_DIR/result/qwen3.5-27B"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-27B-cwe-detection}"
N_SAMPLES="${N_SAMPLES:-5}"
MAX_TOKENS="${MAX_TOKENS:-30000}"
CWE_CHECK_INTERVAL="${CWE_CHECK_INTERVAL:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1200}"
MAX_RETRIES="${MAX_RETRIES:-2}"
RETRY_BACKOFF="${RETRY_BACKOFF:-1.5}"
MBPP_LIMIT="${MBPP_LIMIT:-397}"
HUMANEVAL_LIMIT="${HUMANEVAL_LIMIT:-164}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

check_service() {
    local health_url="${BASE_URL%/v1}/health"
    echo "Checking local API service at $health_url..."
    if ! curl -s "$health_url" > /dev/null; then
        echo "Error: local API service is not running at $health_url"
        exit 1
    fi
}

run_case() {
    local dataset_label="$1"
    local data_path="$2"
    local limit="$3"
    local method="$4"
    local use_traceguard="$5"

    local case_dir="$RESULT_ROOT/$method/$dataset_label"
    local gen_output="$case_dir/generations_${dataset_label}_${method}.json"
    local raw_output="$case_dir/raw_generations_${dataset_label}_${method}.json"
    local eval_output="$case_dir/${dataset_label}_eval_${method}.json"
    local log_file="$case_dir/log_${dataset_label}_${method}.txt"

    mkdir -p "$case_dir"

    echo "=========================================="
    echo "Model: qwen3.5-27B"
    echo "Dataset: $dataset_label"
    echo "Method: $method"
    echo "Data: $data_path"
    echo "Output: $case_dir"
    echo "=========================================="

    {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Generation Phase ==="
        python3 gen_multiple_cpp_from_api.py \
            --base-url "$BASE_URL" \
            --model "$MODEL_NAME" \
            --use-traceguard "$use_traceguard" \
            --cwe-check-interval "$CWE_CHECK_INTERVAL" \
            --dataset-path "$data_path" \
            --n-samples "$N_SAMPLES" \
            --limit "$limit" \
            --max-tokens "$MAX_TOKENS" \
            --timeout "$REQUEST_TIMEOUT" \
            --max-retries "$MAX_RETRIES" \
            --retry-backoff "$RETRY_BACKOFF" \
            --output-path "$gen_output" \
            --save-raw-completions-path "$raw_output" \
            --save-every-problem \
            --resume

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Evaluation Phase ==="
        accelerate launch main.py \
            --model "local-api-qwen3.5-27B" \
            --tasks multiple-cpp \
            --load_data_path "$data_path" \
            --load_generations_path "$gen_output" \
            --allow_code_execution \
            --n_samples "$N_SAMPLES" \
            --limit "$limit" \
            --metric_output_path "$eval_output"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation completed successfully"
    } > "$log_file" 2>&1

    if [ -f "$eval_output" ]; then
        python3 - <<PY_SUMMARY
import json
from pathlib import Path
path = Path("$eval_output")
data = json.loads(path.read_text())
metrics = data.get("multiple-cpp", {})
pass1 = metrics.get("pass@1")
pass5 = metrics.get("pass@5")
print(f"Completed $dataset_label/$method: pass@1={pass1 * 100:.2f}%" if pass1 is not None else "Completed $dataset_label/$method")
if pass5 is not None:
    print(f"Completed $dataset_label/$method: pass@5={pass5 * 100:.2f}%")
PY_SUMMARY
    fi
}

cd "$HARNESS_DIR"
mkdir -p "$RESULT_ROOT"

MBPP_DATA_PATH="${MBPP_DATA_PATH:-$HARNESS_DIR/multi_pl_e_mbpp_cpp.jsonl}"
HUMANEVAL_DATA_PATH="${HUMANEVAL_DATA_PATH:-$HARNESS_DIR/multi_pl_e_humaneval_cpp.jsonl}"

check_service

echo "Running functionality evaluations for qwen3.5-27B"
echo "Harness: $HARNESS_DIR"
echo "Base URL: $BASE_URL"
echo "Model: $MODEL_NAME"
echo "Result root: $RESULT_ROOT"

run_case "mbpp" "$MBPP_DATA_PATH" "$MBPP_LIMIT" "origin" "false"
run_case "humaneval" "$HUMANEVAL_DATA_PATH" "$HUMANEVAL_LIMIT" "origin" "false"
run_case "mbpp" "$MBPP_DATA_PATH" "$MBPP_LIMIT" "traceguard" "true"
run_case "humaneval" "$HUMANEVAL_DATA_PATH" "$HUMANEVAL_LIMIT" "traceguard" "true"

echo "All functionality evaluations for qwen3.5-27B completed."
echo "Results saved under: $RESULT_ROOT"
