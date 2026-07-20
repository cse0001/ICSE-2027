from datasets import Dataset
import json
import os

arrow_path = os.path.expanduser(
    "~/.cache/huggingface/datasets/nuprl___multi_pl-e/humaneval-cpp/0.0.0/28441b6024e71d4a1c1c0f6bf171c935cd5a43f2/multi_pl-e-test.arrow"
)

out_path = "multi_pl_e_humaneval_cpp.jsonl"

ds = Dataset.from_file(arrow_path)

with open(out_path, "w", encoding="utf-8") as f:
    for item in ds:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"saved {len(ds)} rows to {out_path}")