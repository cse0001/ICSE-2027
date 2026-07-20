#!/bin/bash

# TraceGuard LLM API server startup script

# Load environment variables if a .env file exists.
if [ -f .env ]; then
    echo "Loading .env configuration file..."
    export $(grep -v '^#' .env | xargs)
fi

# Set default values.
export MODEL_PATH=${MODEL_PATH:-"../../autodl-fs/DeepSeek-7B"} # Qwen3.5-27B
export DEVICE_ID=${DEVICE_ID:-"0"} 
export TORCH_DTYPE=${TORCH_DTYPE:-"float16"}
export API_HOST=${API_HOST:-"0.0.0.0"}
export API_PORT=${API_PORT:-"8000"}
export MAX_NEW_TOKENS_DEFAULT=${MAX_NEW_TOKENS_DEFAULT:-"12000"}

echo "=========================================="
echo "Starting TraceGuard LLM API server"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Device ID: $DEVICE_ID"
echo "Data type: $TORCH_DTYPE"
echo "Listening address: $API_HOST:$API_PORT"
echo "Default max_new_tokens: $MAX_NEW_TOKENS_DEFAULT"
echo "=========================================="

# Check Python dependencies.
echo "Checking dependencies..."
python3 -c "import fastapi, uvicorn, torch, transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: missing required Python dependencies"
    echo "Please install: pip install fastapi uvicorn torch transformers"
    exit 1
fi

# Start the server.
echo "Starting server..."
python3 llm_api_server_cwe.py --host "$API_HOST" --port "$API_PORT"

# If the server exits unexpectedly.
echo "Server stopped"
