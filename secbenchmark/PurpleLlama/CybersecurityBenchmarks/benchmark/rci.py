from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .llm import LLM


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


@dataclass
class RCIConfig:
    enabled: bool = False
    rounds: int = 1
    critic_prompt: str = DEFAULT_RCI_CRITIQUE_PROMPT
    rewrite_prompt: str = DEFAULT_RCI_REWRITE_PROMPT


def run_rci_loop(
    llm: LLM,
    user_prompt: str,
    initial_response: str,
    config: RCIConfig,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Run iterative Critique -> Rewrite refinement over an initial answer.
    Returns (final_response, trace).
    """
    if not config.enabled or config.rounds <= 0:
        return initial_response, []

    current_response = initial_response
    trace: List[Dict[str, str]] = []

    for round_idx in range(config.rounds):
        critique_input = (
            f"{config.critic_prompt}\n\n"
            f"[Original User Prompt]\n{user_prompt}\n\n"
            f"[Current Answer]\n{current_response}\n\n"
            "Output a short critique with concrete security fixes."
        )
        critique = llm.query_with_retries(critique_input)

        rewrite_input = (
            f"{config.rewrite_prompt}\n\n"
            f"[Original User Prompt]\n{user_prompt}\n\n"
            f"[Current Answer]\n{current_response}\n\n"
            f"[Critique]\n{critique}\n\n"
            "Return the improved final answer."
        )
        improved = llm.query_with_retries(rewrite_input)

        trace.append(
            {
                "round": str(round_idx + 1),
                "critique": critique,
                "rewritten_response": improved,
            }
        )
        current_response = improved

    return current_response, trace
