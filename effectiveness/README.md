# Effectiveness Evaluation

This directory contains the scripts used to run the effectiveness evaluation across four model families and five evaluation settings.

## Directory Layout

- `script/`: executable benchmark scripts.
- `result/`: generated benchmark outputs. This directory is ignored by git.

## Scripts

Each script evaluates one model across the same five settings: `origin`, `codeguard`, `traceguard`, `rci`, and `sosecure`.

- `script/run_deepseek_7B.sh`
- `script/run_deepseek_14B.sh`
- `script/run_deepseek_32B.sh`
- `script/run_qwen3_5_27B.sh`

## Output Layout

Results are written under:

```text
effectiveness/result/<model>/<tool>/
```

For example:

```text
effectiveness/result/deepseek-7B/origin/
effectiveness/result/deepseek-7B/codeguard/
effectiveness/result/deepseek-7B/traceguard/
effectiveness/result/deepseek-7B/rci/
effectiveness/result/deepseek-7B/sosecure/
```

## Usage

Start the local API service before running a script. By default, all settings use:

```text
http://localhost:8000/v1
```

Run one model evaluation with:

```bash
cd /root/TraceGuard/effectiveness/script
./run_deepseek_7B.sh
```

## Configuration

The scripts support environment-variable overrides for model names, API URLs, and benchmark limits. Common variables include:

- `MAX_TOKENS`
- `MAX_TIME` for the qwen script
- `RCI_ROUNDS`
- `CWE_INJECTION_MODE`
- `ORIGIN_API_URL`
- `CODEGUARD_API_URL`
- `TRACEGUARD_API_URL`
- `RCI_API_URL`
- `SOSECURE_API_URL`

Example:

```bash
TRACEGUARD_API_URL=http://localhost:8000/v1 CWE_INJECTION_MODE=delayed ./run_deepseek_14B.sh
```
