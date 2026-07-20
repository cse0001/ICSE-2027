#!/bin/bash

# Run effectiveness evaluations for deepseek-14B across origin, CodeGuard, TraceGuard, RCI, and SOSecure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACEGUARD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PURPLELLAMA_DIR="$TRACEGUARD_ROOT/secbenchmark/PurpleLlama"
DATASETS="$PURPLELLAMA_DIR/CybersecurityBenchmarks/datasets"
RESULT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/result/deepseek-14B"

MAX_TOKENS="${MAX_TOKENS:-30000}"
RCI_ROUNDS="${RCI_ROUNDS:-1}"
CWE_INJECTION_MODE="${CWE_INJECTION_MODE:-delayed}"

ORIGIN_API_URL="${ORIGIN_API_URL:-http://localhost:8000/v1}"
CODEGUARD_API_URL="${CODEGUARD_API_URL:-http://localhost:8000/v1}"
TRACEGUARD_API_URL="${TRACEGUARD_API_URL:-http://localhost:8000/v1}"
RCI_API_URL="${RCI_API_URL:-http://localhost:8000/v1}"
SOSECURE_API_URL="${SOSECURE_API_URL:-http://localhost:8000/v1}"

ORIGIN_MODEL="${ORIGIN_MODEL:-deepseek-14B}"
CODEGUARD_MODEL="${CODEGUARD_MODEL:-deepseek-codeguard-block-14B}"
TRACEGUARD_MODEL="${TRACEGUARD_MODEL:-deepseek-traceguard-block-14B}"
RCI_MODEL="${RCI_MODEL:-deepseek-rci-14B}"
SOSECURE_MODEL="${SOSECURE_MODEL:-deepseek-sosecure-14B}"

ORIGIN_LLM="LOCALTRACEGUARD::${ORIGIN_MODEL}::${ORIGIN_API_URL}::false"
CODEGUARD_LLM="LOCALTRACEGUARD::${CODEGUARD_MODEL}::${CODEGUARD_API_URL}::false"
TRACEGUARD_LLM="LOCALTRACEGUARD::${TRACEGUARD_MODEL}::${TRACEGUARD_API_URL}::true::${CWE_INJECTION_MODE}"
RCI_LLM="LOCALTRACEGUARD::${RCI_MODEL}::${RCI_API_URL}::false"
SOSECURE_LLM="LOCALTRACEGUARD::${SOSECURE_MODEL}::${SOSECURE_API_URL}::false"

check_service() {
    local name="$1"
    local api_url="$2"
    local health_url="${api_url%/v1}/health"

    echo "Checking ${name} service at ${health_url}..."
    if ! curl -s "$health_url" > /dev/null; then
        echo "Error: ${name} service is not running at ${health_url}"
        exit 1
    fi
}

run_benchmark() {
    local label="$1"
    local prompt_path="$2"
    local response_path="$3"
    local stat_path="$4"
    local llm_under_test="$5"
    shift 5

    mkdir -p "$(dirname "$response_path")"
    echo "=========================================="
    echo "$label"
    echo "Prompt: $prompt_path"
    echo "Responses: $response_path"
    echo "Stats: $stat_path"
    echo "=========================================="

    local cmd=(
        python3 -m CybersecurityBenchmarks.benchmark.run
        --benchmark=instruct
        --prompt-path="$prompt_path"
        --response-path="$response_path"
        --stat-path="$stat_path"
        --llm-under-test="$llm_under_test"
        --max-tokens="$MAX_TOKENS"
    )
    cmd+=("$@")
    "${cmd[@]}"
}

cd "$PURPLELLAMA_DIR"
mkdir -p "$RESULT_ROOT"

echo "Running effectiveness evaluations for deepseek-14B"
echo "Dataset path: $DATASETS"
echo "Result root: $RESULT_ROOT"

check_service "origin" "$ORIGIN_API_URL"
run_benchmark "Origin - C Instruct" "$DATASETS/instruct/Cinstruct.json" "$RESULT_ROOT/origin/Cinstruct_responses.json" "$RESULT_ROOT/origin/Cinstruct_stat.json" "$ORIGIN_LLM"
run_benchmark "Origin - C++ Instruct" "$DATASETS/instruct/CPPinstruct.json" "$RESULT_ROOT/origin/CPPinstruct_responses.json" "$RESULT_ROOT/origin/CPPinstruct_stat.json" "$ORIGIN_LLM"

check_service "CodeGuard" "$CODEGUARD_API_URL"
run_benchmark "CodeGuard - C/C++ Instruct" "$DATASETS/instruct/Instruction_C_CPP_Def.json" "$RESULT_ROOT/codeguard/C_CPPinstruct_responses.json" "$RESULT_ROOT/codeguard/C_CPPinstruct_stat.json" "$CODEGUARD_LLM"

check_service "TraceGuard" "$TRACEGUARD_API_URL"
run_benchmark "TraceGuard - C/C++ Instruct" "$DATASETS/instruct/instruct.json" "$RESULT_ROOT/traceguard/C_CPPinstruct_responses.json" "$RESULT_ROOT/traceguard/C_CPPinstruct_stat.json" "$TRACEGUARD_LLM"

check_service "RCI" "$RCI_API_URL"
run_benchmark "RCI - C Instruct" "$DATASETS/instruct/Cinstruct.json" "$RESULT_ROOT/rci/Cinstruct_responses.json" "$RESULT_ROOT/rci/Cinstruct_stat.json" "$RCI_LLM" --enable-rci --rci-rounds="$RCI_ROUNDS"
run_benchmark "RCI - C++ Instruct" "$DATASETS/instruct/CPPinstruct.json" "$RESULT_ROOT/rci/CPPinstruct_responses.json" "$RESULT_ROOT/rci/CPPinstruct_stat.json" "$RCI_LLM" --enable-rci --rci-rounds="$RCI_ROUNDS"

check_service "SOSecure" "$SOSECURE_API_URL"
run_benchmark "SOSecure - C Instruct" "$DATASETS/instruct/SOSecure_Cinstruct.json" "$RESULT_ROOT/sosecure/Cinstruct_responses.json" "$RESULT_ROOT/sosecure/Cinstruct_stat.json" "$SOSECURE_LLM"
run_benchmark "SOSecure - C++ Instruct" "$DATASETS/instruct/SOSecure_CPPinstruct.json" "$RESULT_ROOT/sosecure/CPPinstruct_responses.json" "$RESULT_ROOT/sosecure/CPPinstruct_stat.json" "$SOSECURE_LLM"

echo "All effectiveness evaluations for deepseek-14B completed."
echo "Results saved under: $RESULT_ROOT"
