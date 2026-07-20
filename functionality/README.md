# Functionality Evaluation

This directory contains scripts for running functionality evaluations with MBPP-cpp and HumanEval-cpp.

## Layout

- `script/`: executable evaluation scripts.
- `result/`: generated outputs and metrics. This directory is ignored by git.

## Scripts

Each script evaluates one model serially on two datasets and two methods:

- datasets: `mbpp`, `humaneval`
- methods: `origin`, `traceguard`

Scripts:

- `script/run_deepseek_7B.sh`
- `script/run_deepseek_14B.sh`
- `script/run_deepseek_32B.sh`
- `script/run_qwen3_5_27B.sh`

## Usage

Start the local OpenAI-compatible API on port `8000`, then run:

```bash
cd /root/TraceGuard/functionality/script
./run_deepseek_7B.sh
```

Results are written to:

```text
/root/TraceGuard/functionality/result/<model>/<method>/<dataset>/
```

## Configuration

Common environment overrides:

- `BASE_URL`: defaults to `http://127.0.0.1:8000/v1`.
- `MODEL_NAME`: model id sent to the API.
- `HARNESS_DIR`: defaults to `/root/bigcode-evaluation-harness`.
- `N_SAMPLES`: defaults to `5`.
- `MAX_TOKENS`: defaults to `30000`.
- `MBPP_LIMIT`: defaults to `397`.
- `HUMANEVAL_LIMIT`: defaults to `164`.
- `CWE_CHECK_INTERVAL`: defaults to `1`.
