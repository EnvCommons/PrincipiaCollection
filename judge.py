"""LLM-based equivalence judging for Principia mathematical answers."""

from __future__ import annotations

import re

import openai

JUDGE_PROMPT_TEMPLATE = """\
### Question: {problem}

### Ground Truth Answer: {ground_truth}

### Candidate: {candidate}

### Guidelines: For the above question, please verify if the candidate is equivalent with the ground truth answer or not.

DO NOT ATTEMPT TO SOLVE the question by yourself; instead focus on checking if the two candidates are equivalent.

If the two candidates are equivalent, output "Final Judgment: Yes <End of Judgment>". If not, output "Final Judgment: No <End of Judgment>". Most importantly, DO NOT MAKE a judgment first. Instead, first reason about whether the candidates are equivalent or not based on the specified rules above (read through all of them, not only one), and then output the final judgment.

### Reasoning:
"""


def _parse_judgment(response_text: str) -> bool:
    """Parse 'Final Judgment: Yes/No <End of Judgment>' from LLM response."""
    match = re.search(r"Final Judgment:\s*(Yes|No)", response_text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "yes"
    return False


async def judge_equivalence(
    problem: str,
    ground_truth: str,
    candidate: str,
    client: openai.AsyncClient,
    model: str = "gpt-5-mini",
) -> bool:
    """Single LLM call to judge if candidate is equivalent to ground truth."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        problem=problem,
        ground_truth=ground_truth,
        candidate=candidate,
    )
    response = await client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
    )
    response_text = ""
    for item in response.output:
        if hasattr(item, "text") and item.text:
            response_text += item.text
        elif hasattr(item, "content"):
            response_text += str(item.content)
    return _parse_judgment(response_text)
