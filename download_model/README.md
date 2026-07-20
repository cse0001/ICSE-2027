# Model Download Scripts

This directory contains scripts for downloading the models used by TraceGuard evaluations from Hugging Face.

## Target Directory

All scripts download model files into:

```text
/root/TraceGuard/model/
```

## Scripts

- `download_deepseek_r1_distill_qwen_7B.sh`: downloads `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`.
- `download_deepseek_r1_distill_qwen_14B.sh`: downloads `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`.
- `download_deepseek_r1_distill_qwen_32B.sh`: downloads `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`.
- `download_qwen3_5_27B.sh`: downloads `Qwen/Qwen3.5-27B`.

## Usage

```bash
cd /root/TraceGuard/download_model
./download_deepseek_r1_distill_qwen_7B.sh
```

If the repository requires authentication or your environment is rate-limited, set `HF_TOKEN` before running a script:

```bash
HF_TOKEN=<your-token> ./download_qwen3_5_27B.sh
```

Optional overrides:

- `MODEL_DIR`: custom target directory.
- `REPO_ID`: custom Hugging Face repository ID.
- `REVISION`: branch, tag, or commit; defaults to `main`.
