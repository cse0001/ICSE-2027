# TraceGuard Artifact

This repository contains the artifact for evaluating TraceGuard, a reasoning-stage security enhancement approach for LRM code generation. The artifact is organized for anonymous review and includes scripts for model download, local API deployment, code-generation security evaluation, and functionality evaluation.

## Repository Layout

```text
TraceGuard/
├── download_model/              # Hugging Face model download scripts
├── model/                       # Downloaded model weights
├── server/                      # OpenAI-compatible local TraceGuard API server
├── effectiveness/               # Code-generation security evaluation scripts
│   ├── script/
│   └── result/                  # Generated outputs
├── functionality/               # MBPP/HumanEval functionality evaluation scripts
│   ├── script/
│   └── result/                  # Generated outputs
├── secbenchmark/                # Security benchmark code and datasets
├── bigcode-evaluation-harness/  # Functionality evaluation harness
├── requirements-trans.txt
├── requirements-cyber.txt
└── requirements-bigcode.txt
```

Generated outputs and downloaded model weights are intentionally excluded from version control.

## 1. Prepare Model Download Utilities

Install the Hugging Face download utility before fetching model weights. The command below can be run in the base environment or in the `trans` environment after it is created.

```bash
cd /root/TraceGuard
python -m pip install huggingface_hub
```

## 2. Download Models

Download model weights into `/root/TraceGuard/model/`:

```bash
cd /root/TraceGuard/download_model

./download_deepseek_r1_distill_qwen_7B.sh
./download_deepseek_r1_distill_qwen_14B.sh
./download_deepseek_r1_distill_qwen_32B.sh
./download_qwen3_5_27B.sh
```

If authentication is required by the hosting service, set `HF_TOKEN` before running a download script:

```bash
HF_TOKEN=<token> ./download_qwen3_5_27B.sh
```

The scripts download the following Hugging Face repositories:

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
- `Qwen/Qwen3.5-27B`

## 3. Prepare Conda Environments

The artifact uses three separate conda environments. Create the environments and install the corresponding requirement files from the repository root.

```bash
cd /root/TraceGuard

conda create -n trans python=3.10 -y
conda run -n trans python -m pip install -r requirements-trans.txt

conda create -n cyber python=3.10 -y
conda run -n cyber python -m pip install -r requirements-cyber.txt

conda create -n bigcode python=3.10 -y
conda run -n bigcode python -m pip install -r requirements-bigcode.txt
```

Environment roles:

- `trans`: local model serving and TraceGuard reasoning-stage logic.
- `cyber`: code-generation security evaluation scripts under `effectiveness/` and `secbenchmark/`.
- `bigcode`: MBPP-cpp and HumanEval-cpp functionality evaluation.

## 4. Start the Local API Server

The local server must be started from the `server/` directory. This working directory is required because the server uses relative paths for model files and TraceGuard rule files.

```bash
conda activate trans
cd /root/TraceGuard/server
```

Start the server for the model you want to evaluate. For example:

```bash
MODEL_PATH=../model/DeepSeek-R1-Distill-Qwen-7B API_PORT=8000 DEVICE_ID=0 ./start_api_server.sh
```

For the other models, change `MODEL_PATH` accordingly:

```text
../model/DeepSeek-R1-Distill-Qwen-7B
../model/DeepSeek-R1-Distill-Qwen-14B
../model/DeepSeek-R1-Distill-Qwen-32B
../model/Qwen3.5-27B
```

By default, the evaluation scripts expect the local API at:

```text
http://localhost:8000/v1
```

Keep the server running while executing the evaluation scripts in another terminal.

## 5. Effectiveness Evaluation

The effectiveness evaluation measures code-generation security across four models and five methods:

- `origin`: base model without an enhancement method.
- `codeguard`: CodeGuard-style prompt enhancement.
- `traceguard`: TraceGuard reasoning-stage safety intervention.
- `rci`: iterative critique-and-rewrite refinement.
- `sosecure`: SOSecure-style prompting.

Activate the `cyber` environment and run one script per model:

```bash
conda activate cyber
cd /root/TraceGuard/effectiveness/script

./run_deepseek_7B.sh
./run_deepseek_14B.sh
./run_deepseek_32B.sh
./run_qwen3_5_27B.sh
```

Each script writes results to:

```text
/root/TraceGuard/effectiveness/result/<model>/<method>/
```

Run the script corresponding to the model currently served by the local API. To evaluate another model, stop the server, restart it with the new `MODEL_PATH`, and run the matching script.

## 6. Functionality Evaluation

The functionality evaluation compares the original model and TraceGuard on two standard code-generation datasets:

- `mbpp`: MBPP-cpp, 397 tasks by default.
- `humaneval`: HumanEval-cpp, 164 tasks by default.

Each task uses `N_SAMPLES=5` by default. The scripts run serially under the same local API server at port `8000`.

Activate the `bigcode` environment and run one script per model:

```bash
conda activate bigcode
cd /root/TraceGuard/functionality/script

./run_deepseek_7B.sh
./run_deepseek_14B.sh
./run_deepseek_32B.sh
./run_qwen3_5_27B.sh
```

Outputs are written to:

```text
/root/TraceGuard/functionality/result/<model>/<method>/<dataset>/
```

The functionality scripts use the local BigCode-style harness and call the local API through `gen_multiple_cpp_from_api.py`. The dataset loader attempts to download from Hugging Face first and falls back to local JSONL files after repeated failures.

## 7. Useful Runtime Overrides

Common server overrides:

```bash
MODEL_PATH=../model/Qwen3.5-27B
API_PORT=8000
DEVICE_ID=0
TORCH_DTYPE=float16
MAX_NEW_TOKENS_DEFAULT=12000
```

Common effectiveness overrides:

```bash
MAX_TOKENS=30000
RCI_ROUNDS=1
CWE_INJECTION_MODE=delayed
```

Common functionality overrides:

```bash
BASE_URL=http://127.0.0.1:8000/v1
N_SAMPLES=5
MAX_TOKENS=30000
MBPP_LIMIT=397
HUMANEVAL_LIMIT=164
CWE_CHECK_INTERVAL=1
```

## 8. Expected Workflow

A typical review workflow is:

1. Install the model download utility.
2. Download the required model weights.
3. Create the three conda environments.
4. Start the local server from `/root/TraceGuard/server` with the `trans` environment.
5. Run effectiveness scripts with the `cyber` environment.
6. Run functionality scripts with the `bigcode` environment.
7. Inspect generated JSON result files under `effectiveness/result/` and `functionality/result/`.

## Notes

- The artifact is configured for local execution with an OpenAI-compatible API endpoint.
- The evaluation result directories are ignored by git to keep the repository lightweight.
- Large model weights are stored under `model/`, which is also ignored by git.
- If the hosting service is unavailable during functionality evaluation, the dataset loader retries Hugging Face downloads and then falls back to local dataset files when provided.
