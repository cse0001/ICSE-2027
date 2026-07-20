"""MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation
https://arxiv.org/abs/2107.03374

MultiPL-E is a dataset for evaluating large language models for code generation that supports 18 programming languages.
It takes the OpenAI "HumanEval" and the MBPP Python benchmarks and uses little compilers to translate them to other languages.

Homepage: https://nuprl.github.io/MultiPL-E/
"""

import json
import os
import re
import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from time import time

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.multiple_metrics.evaluation import \
    evaluate_problem
from bigcode_eval.tasks.custom_metrics.multiple_metrics.single_experiment_pass_k import \
    for_file

_CITATION = """
@article{cassano2022scalable,
  author={Cassano, Federico and Gouwar, John and Nguyen, Daniel and Nguyen, Sydney and Phipps-Costin, Luna and Pinckney, Donald and Yee, Ming-Ho and Zi, Yangtian and Anderson, Carolyn Jane and Feldman, Molly Q and Guha, Arjun and Greenberg, Michael and Jangda, Abhinav},
  journal={IEEE Transactions on Software Engineering}, 
  title={MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation}, 
  year={2023},
  volume={49},
  number={7},
  pages={3675-3691},
  doi={10.1109/TSE.2023.3267446}
}
"""

LANGUAGES = [
    "py",
    "sh",
    "clj",
    "cpp",
    "cs",
    "d",
    "dart",
    "elixir",
    "go",
    "hs",
    "java",
    "js",
    "jl",
    "lua",
    "ml",
    "pl",
    "php",
    "r",
    "rkt",
    "rb",
    "rs",
    "scala",
    "swift",
    "ts",
]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {f"multiple-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class MultiPLE(GeneralMultiPLE):
        def __init__(self, prompt="prompt", load_data_path=None):
            super().__init__(language, prompt=prompt, load_data_path=load_data_path)

    return MultiPLE


class GeneralMultiPLE(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "nuprl/MultiPL-E"
    DATASET_NAME = None
    DATASET_REVISION = "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2"

    def __init__(self, language, prompt="prompt", load_data_path=None):
        self.language = language
        self.prompt_key = prompt
        self.DATASET_NAME = f"humaneval-{language}"
        local_records = self._load_local_records(load_data_path)
        if local_records:
            self.dataset = {"test": local_records}
            self.eval_docs = local_records
            print(
                f"Loaded local MultiPL-E data from {load_data_path}: "
                f"{len(self.eval_docs)} problems for language={self.language}"
            )
            stop_words = self.eval_docs[0]["stop_tokens"] + ["<file_sep>"]
            super().__init__(
                stop_words=stop_words,
                requires_execution=True,
            )
            return
        # we need the dataset to get stop words for each language
        self.dataset = load_dataset(
            GeneralMultiPLE.DATASET_PATH,
            self.DATASET_NAME,
            revision=self.DATASET_REVISION)
        self.eval_docs = self._build_eval_docs(load_data_path)
        stop_words = self.eval_docs[0]["stop_tokens"] + ["<file_sep>"]
        super().__init__(
            stop_words=stop_words,
            requires_execution=True,
        )

    def _load_local_records(self, load_data_path):
        if not load_data_path:
            return None
        if not os.path.exists(load_data_path):
            raise FileNotFoundError(f"Local data file does not exist: {load_data_path}")

        if load_data_path.endswith(".jsonl"):
            records = []
            with open(load_data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            return records

        with open(load_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if isinstance(data.get("data"), list):
                return data["data"]
            if isinstance(data.get("records"), list):
                return data["records"]
        raise ValueError(
            f"Unsupported local data format for MultiPL-E: {load_data_path}"
        )

    def _build_eval_docs(self, load_data_path):
        hf_docs = list(self.dataset["test"])
        local_records = self._load_local_records(load_data_path)
        if not local_records:
            return hf_docs

        id_keys = ("name", "task_id", "problem_id", "id")
        local_by_name = {}
        for rec in local_records:
            if not isinstance(rec, dict):
                continue
            rec_language = rec.get("language")
            if rec_language and str(rec_language) != self.language:
                continue
            rec_name = None
            for key in id_keys:
                if rec.get(key) is not None:
                    rec_name = str(rec[key])
                    break
            if rec_name is None:
                continue
            local_by_name[rec_name] = rec

        merged_docs = []
        replaced = 0
        for doc in hf_docs:
            name = str(doc.get("name", ""))
            local_doc = local_by_name.get(name)
            if not local_doc:
                merged_docs.append(doc)
                continue

            merged = dict(doc)
            # Main fields used in generation/eval.
            if isinstance(local_doc.get("prompt"), str):
                merged["prompt"] = local_doc["prompt"]
            if isinstance(local_doc.get("tests"), str):
                merged["tests"] = local_doc["tests"]
            if isinstance(local_doc.get("stop_tokens"), list) and local_doc["stop_tokens"]:
                merged["stop_tokens"] = local_doc["stop_tokens"]
            merged_docs.append(merged)
            replaced += 1

        print(
            f"Loaded local MultiPL-E overrides from {load_data_path}: "
            f"matched {replaced}/{len(hf_docs)} problems for language={self.language}"
        )
        return merged_docs

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.eval_docs

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return doc["prompt"].strip()

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["tests"]

    @staticmethod
    def remove_last_block(string, stop_words):
        # Remove the last block of the code containing stop_words for HumanEval
        string_list = re.split("(%s)" % "|".join(stop_words), string)
        # last string should be ""
        return "".join(string_list[:-2])


    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        completion = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(completion, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        # get prompts and problem names
        prompts_names = [
            {"prompt": self.get_prompt(doc), "name": doc["name"], "lang": doc["language"]}
            for i, doc in enumerate(self.get_dataset())
            if i < len(generations)
        ]
        # Use an isolated temp dir per evaluation run to avoid mixing
        # stale *.results.json files from previous runs.
        temp_dir = tempfile.mkdtemp(prefix="multiple_eval_")
        list_files = []
        for (prompt_name, generation, reference) in zip(
            prompts_names, generations, references
        ):
            problem = {
                "name": prompt_name["name"],
                "language": prompt_name["lang"],
                "prompt": prompt_name["prompt"],
                "completions": generation,
                "tests": reference,
            }
            # each problem is save in a json file
            temp_file_name = os.path.join(temp_dir, f"{prompt_name['name']}.json")
            list_files.append(temp_file_name)
            with open(temp_file_name, "wt") as f:
                json.dump(problem, f)
        print(
            f"Saved {len(list_files)} problems in {temp_dir} for evaluation, each problem has {len(generations[0])} completions"
        )

        # execute the problems to evaluate them
        max_workers = cpu_count() - 1 if cpu_count() > 1 else 1
        for file in tqdm(list_files):
            evaluate_problem(temp_dir, file, max_workers)

        # compute pass@k scores
        result_array = np.array([for_file(p) for p in Path(temp_dir).glob("*.results.json")])
        result = result_array.mean(axis=0)
        name = (
            temp_dir.split("/")[-1]
            if temp_dir.split("/")[-1] != ""
            else temp_dir.split("/")[-2]
        )
        results = {
            f"pass@{k}": v
            for k, v in zip([1, 5, 100], result)
            if k <= len(generations[0])
        }
        return results
