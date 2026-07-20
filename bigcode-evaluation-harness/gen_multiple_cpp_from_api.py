#!/usr/bin/env python3
import argparse
import json
import os
import re
import socket
import time
from typing import List, Optional, Tuple
from urllib import error, request
from requests import exceptions as requests_exceptions

from datasets import load_dataset
from tqdm import tqdm


DEFAULT_DATASET_PATH = "nuprl/MultiPL-E"
DEFAULT_DATASET_NAME = "humaneval-cpp"
DEFAULT_DATASET_REVISION = "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2"


DEFAULT_RCI_CRITIQUE_PROMPT = (
    "You are a security reviewer for generated code. "
    "Review the current answer and identify concrete security weaknesses, risky APIs, "
    "missing validations, and safer alternatives. Be concise and specific."
)

DEFAULT_RCI_REWRITE_PROMPT = (
    "You are a secure coding assistant. Rewrite the answer using the critique. "
    "Keep functional intent, but prefer safer APIs, strict input validation, "
    "proper error handling, and principle of least privilege. Return only the improved answer."
)


def load_local_generation_dataset(dataset_path: str):
    """Load a local MultiPL-E JSON or JSONL file."""
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        if dataset_path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        else:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and isinstance(data.get("data"), list):
                records = data["data"]
            elif isinstance(data, dict) and isinstance(data.get("records"), list):
                records = data["records"]
            else:
                raise ValueError(f"Unsupported local dataset format: {dataset_path}")
    print(f"Loaded local dataset from {dataset_path}: {len(records)} rows")
    return records


def infer_hf_dataset_name(dataset_path: str, dataset_name: str) -> str:
    """Infer the Hugging Face MultiPL-E config when a local fallback file is provided."""
    if not os.path.isfile(dataset_path):
        return dataset_name
    basename = os.path.basename(dataset_path).lower()
    if "mbpp" in basename:
        return "mbpp-cpp"
    if "humaneval" in basename:
        return "humaneval-cpp"
    return dataset_name


def load_generation_dataset(
    dataset_path: str,
    dataset_name: str,
    dataset_revision: str,
    hf_retry_count: int = 5,
    hf_retry_backoff: float = 2.0,
):
    """Load a Hugging Face dataset first, then fall back to a local JSON/JSONL file."""
    hf_dataset_path = dataset_path if not os.path.isfile(dataset_path) else DEFAULT_DATASET_PATH
    hf_dataset_name = infer_hf_dataset_name(dataset_path, dataset_name)
    local_fallback_path = dataset_path if os.path.isfile(dataset_path) else None
    last_error = None

    for attempt in range(1, hf_retry_count + 1):
        try:
            print(
                f"Loading dataset from Hugging Face: path={hf_dataset_path}, "
                f"name={hf_dataset_name}, revision={dataset_revision} "
                f"(attempt {attempt}/{hf_retry_count})"
            )
            ds = load_dataset(
                hf_dataset_path,
                hf_dataset_name,
                revision=dataset_revision,
            )["test"]
            print(f"Loaded Hugging Face dataset: {len(ds)} rows")
            return ds
        except (
            TimeoutError,
            socket.timeout,
            requests_exceptions.Timeout,
            requests_exceptions.ConnectionError,
            error.URLError,
        ) as exc:
            last_error = exc
            print(f"Hugging Face dataset load timed out or failed: {exc}")
        except Exception as exc:
            last_error = exc
            print(f"Hugging Face dataset load failed: {exc}")

        if attempt < hf_retry_count:
            sleep_s = hf_retry_backoff * (2 ** (attempt - 1))
            print(f"Retrying Hugging Face dataset load in {sleep_s:.1f}s ...")
            time.sleep(sleep_s)

    if local_fallback_path:
        print(
            "Hugging Face dataset load failed after "
            f"{hf_retry_count} attempts; falling back to local file: {local_fallback_path}"
        )
        return load_local_generation_dataset(local_fallback_path)

    raise RuntimeError(
        "Failed to load dataset from Hugging Face and no local fallback file was provided"
    ) from last_error


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value}. Use true/false."
    )


def cut_by_stop(text: str, stop_tokens: List[str]) -> str:
    end = len(text)
    for token in stop_tokens:
        pos = text.find(token)
        if pos != -1:
            end = min(end, pos)
    return text[:end]


def extract_content_in_code_blocks(text: str) -> List[str]:
    """Benchmark-style extraction: find content between triple backticks."""
    return re.findall(r"```(.*?)```", text, re.DOTALL)


def _strip_code_block_language_header(block: str) -> str:
    """Remove code fence language header like cpp/c++ from block body."""
    normalized = block.lstrip("\n")
    lines = normalized.splitlines()
    if not lines:
        return normalized
    first_line = lines[0].strip().lower()
    if first_line in {"cpp", "c++", "cc", "cxx", "c"}:
        return "\n".join(lines[1:])
    return normalized


def _extract_function_name_from_prompt(prompt: str) -> Optional[str]:
    """Parse the function name from the prompt's last signature line.

    MultiPL-E prompts end with `<ret_type> <name>(args) {`.
    """
    lines = prompt.rstrip().splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped.endswith("{"):
            continue
        # Skip lines that look like control flow rather than function defs.
        head = stripped.split("(", 1)[0].strip()
        if head in {"if", "for", "while", "switch", "do", "else"}:
            continue
        match = re.search(r"([A-Za-z_]\w*)\s*\(", stripped)
        if match:
            return match.group(1)
    return None


def _select_best_code_block(blocks: List[str], func_name: Optional[str]) -> str:
    """Pick the block most likely to hold the target function body."""
    if not blocks:
        return ""
    if func_name:
        with_func = [b for b in blocks if re.search(rf"\b{re.escape(func_name)}\s*\(", b)]
        if with_func:
            return with_func[-1]
    # No func match: prefer the longest block (likely the full solution).
    return max(blocks, key=len)


def _extract_function_body(code: str, func_name: str) -> Optional[str]:
    """Return the content between the matching braces of `func_name(...) { ... }`.

    Uses brace counting to correctly handle nested braces. Returns ``None`` when
    no such function definition is found.
    """
    pattern = re.compile(rf"\b{re.escape(func_name)}\s*\([^;{{}}]*\)\s*\{{")
    matches = list(pattern.finditer(code))
    if not matches:
        return None
    match = matches[-1]
    start = match.end()
    depth = 1
    i = start
    in_line_comment = False
    in_block_comment = False
    in_string = False
    in_char = False
    while i < len(code):
        ch = code[i]
        nxt = code[i + 1] if i + 1 < len(code) else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
        elif in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 1
        elif in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif in_char:
            if ch == "\\":
                i += 1
            elif ch == "'":
                in_char = False
        else:
            if ch == "/" and nxt == "/":
                in_line_comment = True
                i += 1
            elif ch == "/" and nxt == "*":
                in_block_comment = True
                i += 1
            elif ch == '"':
                in_string = True
            elif ch == "'":
                in_char = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return code[start:i]
        i += 1
    # Unmatched: return everything we have so the caller can still try to use it.
    return code[start:]


def extract_final_answer(text: str) -> str:
    """Extract final answer and remove explicit reasoning sections when possible."""
    text = text.replace("\r\n", "\n")

    think_end_matches = list(re.finditer(r"</think\s*>", text, flags=re.IGNORECASE))
    if think_end_matches:
        suffix = text[think_end_matches[-1].end() :].strip()
        if suffix:
            return suffix

    code_blocks = re.findall(
        r"```(?:cpp|c\+\+|cc|cxx)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if code_blocks:
        return code_blocks[-1].strip()

    # If there is an unclosed think block, try to recover from first code-like marker.
    if re.search(r"<think\b", text, flags=re.IGNORECASE):
        markers = [r"#include", r"\busing\s+namespace\b", r"\b(?:bool|int|long|float|double|char|std::vector)\b"]
        best_pos = len(text)
        for marker in markers:
            match = re.search(marker, text, flags=re.IGNORECASE)
            if match:
                best_pos = min(best_pos, match.start())
        if best_pos < len(text):
            return text[best_pos:].strip()

    return text.strip()


def _strip_auxiliary_artifacts(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    text = re.sub(
        r"\[SECURITY INTERRUPT:.*?(?:\n\n|\Z)",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\[Common Security Reminder\].*?(?:\n\n|\Z)",
        "",
        text,
        flags=re.DOTALL,
    )

    if "```" in text:
        code_blocks = re.findall(
            r"```(?:cpp|c\+\+)?\s*(.*?)```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if code_blocks:
            text = "\n\n".join(code_blocks)
        text = text.replace("```cpp", "").replace("```c++", "").replace("```", "")
    return text


_PLACEHOLDER_BODY = "// no parseable code in model output\nreturn {};"


def sanitize_cpp_completion(raw_completion: str, prompt: str) -> str:
    """Extract a clean function body usable as MultiPL-E completion.

    Pipeline:
      1. Drop reasoning by extracting text after ``</think>`` (when present).
      2. Pull fenced ``` blocks; pick the one containing the target function.
      3. Strip the code fence language header (``cpp`` / ``c++`` line).
      4. Brace-match ``<func_name>(...) { ... }`` to keep ONLY the body.
      5. Prepend ``using namespace std;`` so bare std names compile.
      6. Drop any trailing ``main()`` / ``auto candidate = ...`` test snippets.
      7. Fallbacks (no fence / no body found) avoid emitting natural language.
    """
    func_name = _extract_function_name_from_prompt(prompt)

    text = extract_final_answer(raw_completion)
    text = _strip_auxiliary_artifacts(text)

    fenced = extract_content_in_code_blocks(text)
    used_fenced_block = False
    if fenced:
        block = _select_best_code_block(fenced, func_name)
        block = _strip_code_block_language_header(block)
        if block.strip():
            text = block
            used_fenced_block = True

    # Drop trailing test harness if the model echoed one.
    for marker in ("\nint main(", "\nauto candidate =", "\n// Driver", "\n// Test"):
        pos = text.find(marker)
        if pos != -1:
            text = text[:pos]

    body: Optional[str] = None
    if func_name:
        body = _extract_function_body(text, func_name)

    if body is not None and body.strip():
        cleaned_body = body.strip("\n")
        # Drop a stray leading blank line but keep meaningful indentation.
        cleaned_body = cleaned_body.rstrip()
        # Prepend `using namespace std;` so bodies using bare std names compile.
        out = "\nusing namespace std;\n" + cleaned_body + "\n"
        return out

    # No function body could be recovered. If we used a fenced block, return its
    # contents (best effort). Otherwise treat as no parseable code.
    if used_fenced_block:
        cleaned = text.strip("\n").rstrip()
        # Ensure we do not double-close the function — the test prepends `}`.
        while cleaned.endswith("}"):
            cleaned = cleaned[:-1].rstrip()
        if not cleaned:
            return "\n" + _PLACEHOLDER_BODY + "\n"
        return "\nusing namespace std;\n" + cleaned + "\n"

    return "\n" + _PLACEHOLDER_BODY + "\n"


def call_chat_completions(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    use_traceguard: bool,
    cwe_check_interval: int,
    timeout: int,
    system_prompt: str,
    max_retries: int,
    retry_backoff: float,
) -> Tuple[str, str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
        "use_traceguard": use_traceguard,
        "cwe_check_interval": cwe_check_interval,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            break
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            last_error = RuntimeError(
                f"HTTP {exc.code} calling {endpoint}. response={detail}"
            )
            # Retry only common transient errors.
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= max_retries:
                raise last_error from exc
        except error.URLError as exc:
            last_error = RuntimeError(f"Failed to call {endpoint}: {exc}")
            if attempt >= max_retries:
                raise last_error from exc
        except (TimeoutError, socket.timeout) as exc:
            last_error = RuntimeError(f"Request timeout calling {endpoint}: {exc}")
            if attempt >= max_retries:
                raise last_error from exc

        sleep_s = retry_backoff * (2 ** attempt)
        print(
            f"[retry {attempt + 1}/{max_retries}] request failed, "
            f"sleeping {sleep_s:.1f}s ..."
        )
        time.sleep(sleep_s)
    else:
        if last_error is not None:
            raise last_error

    obj = json.loads(body)
    choices = obj.get("choices", [])
    if not choices:
        raise RuntimeError(f"No choices in response: {body}")
    choice = choices[0]
    message = choice.get("message", {})
    content = message.get("content", "")
    if content is None:
        return "", str(choice.get("finish_reason", ""))
    return content, str(choice.get("finish_reason", ""))


def run_rci_loop(
    *,
    query_fn,
    user_prompt: str,
    initial_response: str,
    rounds: int,
    critic_prompt: str,
    rewrite_prompt: str,
) -> Tuple[str, List[dict]]:
    """Iterative Critique -> Rewrite refinement over an initial answer.

    Mirrors the TraceGuard PurpleLlama RCI logic: each round runs one critique
    query and one rewrite query against the same LLM. Returns the final answer
    and a trace of every round.
    """
    if rounds <= 0:
        return initial_response, []

    current_response = initial_response
    trace: List[dict] = []
    for round_idx in range(rounds):
        critique_input = (
            f"{critic_prompt}\n\n"
            f"[Original User Prompt]\n{user_prompt}\n\n"
            f"[Current Answer]\n{current_response}\n\n"
            "Output a short critique with concrete security fixes."
        )
        critique, _ = query_fn(critique_input)

        rewrite_input = (
            f"{rewrite_prompt}\n\n"
            f"[Original User Prompt]\n{user_prompt}\n\n"
            f"[Current Answer]\n{current_response}\n\n"
            f"[Critique]\n{critique}\n\n"
            "Return the improved final answer."
        )
        improved, _ = query_fn(rewrite_input)

        trace.append(
            {
                "round": round_idx + 1,
                "critique": critique,
                "rewritten_response": improved,
            }
        )
        current_response = improved

    return current_response, trace


def _print_truncation_summary(length_truncated_count: int, total_calls: int) -> None:
    if total_calls <= 0:
        return
    ratio = 100.0 * length_truncated_count / total_calls
    print(
        f"API finish_reason=length: {length_truncated_count}/{total_calls} "
        f"({ratio:.2f}%)."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate MultiPL-E humaneval-cpp outputs from OpenAI-compatible local API."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8001/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-cwe-detection",
        help="Model id passed to /v1/chat/completions.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of completions per problem.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=164,
        help="Number of problems to generate.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retry count for transient API errors (e.g. 502).",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.5,
        help="Base backoff seconds; actual sleep is backoff * 2^attempt.",
    )
    parser.add_argument(
        "--use-traceguard",
        type=str2bool,
        default=False,
        help="Pass use_traceguard=true/false to your API.",
    )
    parser.add_argument(
        "--cwe-check-interval",
        type=int,
        default=5,
        help="Pass cwe_check_interval to your API.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="generations_multiple_cpp.json",
        help="Output generations JSON path for bigcode-evaluation-harness.",
    )
    parser.add_argument(
        "--save-raw-completions-path",
        type=str,
        default=None,
        help="Optional path to save raw API completions before postprocess.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
    )
    parser.add_argument(
        "--dataset-revision",
        type=str,
        default=DEFAULT_DATASET_REVISION,
    )
    parser.add_argument(
        "--hf-dataset-retries",
        type=int,
        default=5,
        help="Number of Hugging Face dataset load attempts before local fallback.",
    )
    parser.add_argument(
        "--hf-dataset-retry-backoff",
        type=float,
        default=2.0,
        help="Base backoff seconds between Hugging Face dataset load attempts.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a code completion model. Output only C++ code, no explanations, no markdown.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file if present.",
    )
    parser.add_argument(
        "--save-every-problem",
        action="store_true",
        help="Persist output json after each problem for crash safety.",
    )
    parser.add_argument(
        "--enable-rci",
        type=str2bool,
        default=False,
        help="Enable iterative RCI refinement (critique + rewrite) on each completion.",
    )
    parser.add_argument(
        "--rci-rounds",
        type=int,
        default=1,
        help="Number of RCI refinement rounds when --enable-rci is true.",
    )
    parser.add_argument(
        "--rci-critic-prompt",
        type=str,
        default=DEFAULT_RCI_CRITIQUE_PROMPT,
        help="Prompt used for the RCI critique step.",
    )
    parser.add_argument(
        "--rci-rewrite-prompt",
        type=str,
        default=DEFAULT_RCI_REWRITE_PROMPT,
        help="Prompt used for the RCI rewrite step.",
    )
    args = parser.parse_args()

    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    ds = load_generation_dataset(
        args.dataset_path,
        args.dataset_name,
        args.dataset_revision,
        hf_retry_count=args.hf_dataset_retries,
        hf_retry_backoff=args.hf_dataset_retry_backoff,
    )

    limit = min(args.limit, len(ds))
    all_generations = []
    all_raw_completions = []
    if args.resume:
        try:
            with open(args.output_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                all_generations = loaded
                print(f"Resume enabled: loaded {len(all_generations)} problems from {args.output_path}")
        except FileNotFoundError:
            pass
        if args.save_raw_completions_path:
            try:
                with open(args.save_raw_completions_path, "r", encoding="utf-8") as f:
                    loaded_raw = json.load(f)
                if isinstance(loaded_raw, list):
                    all_raw_completions = loaded_raw
                    print(
                        "Resume enabled: loaded "
                        f"{len(all_raw_completions)} raw problems from {args.save_raw_completions_path}"
                    )
            except FileNotFoundError:
                pass

    print(
        f"Generating {limit} problems, n_samples={args.n_samples}, "
        f"use_traceguard={args.use_traceguard}, cwe_check_interval={args.cwe_check_interval}"
    )

    start_idx = len(all_generations)
    if start_idx > 0:
        print(f"Resuming from problem index {start_idx}")
    if args.save_raw_completions_path and len(all_raw_completions) < start_idx:
        missing = start_idx - len(all_raw_completions)
        all_raw_completions.extend([[] for _ in range(missing)])
    total_api_calls = 0
    length_truncated_count = 0

    def rci_query(p: str) -> Tuple[str, str]:
        return call_chat_completions(
            endpoint=endpoint,
            model=args.model,
            prompt=p,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            use_traceguard=args.use_traceguard,
            cwe_check_interval=args.cwe_check_interval,
            timeout=args.timeout,
            system_prompt=args.system_prompt,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
        )

    if args.enable_rci:
        print(f"RCI enabled: rounds={args.rci_rounds} (each round = 1 critique + 1 rewrite call)")

    for i, doc in enumerate(tqdm(ds, total=limit)):
        if i >= limit:
            break
        if i < start_idx:
            continue

        prompt = doc["prompt"].strip()
        stop_tokens = list(doc["stop_tokens"]) + ["<file_sep>"]
        one_problem_generations = []
        one_problem_raw = []

        for sample_idx in range(args.n_samples):
            user_prompt = f"Complete the following C++ code:\n\n{prompt}"
            completion, finish_reason = call_chat_completions(
                endpoint=endpoint,
                model=args.model,
                prompt=user_prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                use_traceguard=args.use_traceguard,
                cwe_check_interval=args.cwe_check_interval,
                timeout=args.timeout,
                system_prompt=args.system_prompt,
                max_retries=args.max_retries,
                retry_backoff=args.retry_backoff,
            )
            raw_completion = completion
            total_api_calls += 1
            if finish_reason == "length":
                length_truncated_count += 1

            initial_completion = completion
            rci_trace: List[dict] = []
            if args.enable_rci:
                completion, rci_trace = run_rci_loop(
                    query_fn=rci_query,
                    user_prompt=user_prompt,
                    initial_response=completion,
                    rounds=args.rci_rounds,
                    critic_prompt=args.rci_critic_prompt,
                    rewrite_prompt=args.rci_rewrite_prompt,
                )
                total_api_calls += 2 * args.rci_rounds
                raw_completion = completion

            completion = cut_by_stop(completion, stop_tokens)
            completion = sanitize_cpp_completion(completion, prompt)

            # bigcode-evaluation-harness multiple task expects prompt+completion
            one_problem_generations.append(prompt + completion)
            if args.save_raw_completions_path:
                raw_record = {
                    "problem_idx": i,
                    "sample_idx": sample_idx,
                    "finish_reason": finish_reason,
                    "raw_completion": raw_completion,
                }
                if args.enable_rci:
                    raw_record["rci_enabled"] = True
                    raw_record["rci_rounds"] = args.rci_rounds
                    raw_record["initial_completion"] = initial_completion
                    raw_record["rci_trace"] = rci_trace
                one_problem_raw.append(raw_record)

        all_generations.append(one_problem_generations)
        if args.save_raw_completions_path:
            all_raw_completions.append(one_problem_raw)
        if args.save_every_problem:
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(all_generations, f, ensure_ascii=False)
            if args.save_raw_completions_path:
                with open(args.save_raw_completions_path, "w", encoding="utf-8") as f:
                    json.dump(all_raw_completions, f, ensure_ascii=False)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(all_generations, f, ensure_ascii=False)
    if args.save_raw_completions_path:
        with open(args.save_raw_completions_path, "w", encoding="utf-8") as f:
            json.dump(all_raw_completions, f, ensure_ascii=False)
        print(f"Saved raw completions to {args.save_raw_completions_path}")
    _print_truncation_summary(length_truncated_count, total_api_calls)

    print(
        f"Saved generations to {args.output_path}. "
        f"shape=({len(all_generations)}, {args.n_samples})"
    )


if __name__ == "__main__":
    main()
