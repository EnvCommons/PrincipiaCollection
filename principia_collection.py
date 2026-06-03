"""
PrincipiaCollection Environment

Large-scale synthetic STEM training problems for RL.
248K mathematical_object problems (LLM-judged) + 306K numerical problems (math-verify).

Dataset: HuggingFace facebook/principia-collection
"""

from __future__ import annotations

import os

import openai
import pandas as pd
from math_verify import parse, verify
from pydantic import BaseModel

from openreward.environments import Environment, JSONObject, Server, TextBlock, ToolOutput, tool

from judge import judge_equivalence

# Module-level data loading
if os.path.exists("/orwd_data"):
    mo_data = pd.read_parquet("/orwd_data/principia_collection_mathematical_object.parquet").to_dict(orient="records")
    num_data = pd.read_parquet("/orwd_data/principia_collection_numerical.parquet").to_dict(orient="records")
else:
    mo_data = pd.read_parquet("principia_collection_mathematical_object.parquet").to_dict(orient="records")
    num_data = pd.read_parquet("principia_collection_numerical.parquet").to_dict(orient="records")

# Extract answers and clean task specs
MO_ANSWERS: dict[str, str] = {}
for i, task in enumerate(mo_data):
    task_id = f"mo_{i}"
    MO_ANSWERS[task_id] = task.pop("answer")
    task["id"] = task_id
    task["split"] = "train"

NUM_ANSWERS: dict[str, str] = {}
for i, task in enumerate(num_data):
    task_id = f"num_{i}"
    NUM_ANSWERS[task_id] = task.pop("answer")
    task["id"] = task_id
    task["split"] = "train_numerical"

ALL_ANSWERS = {**MO_ANSWERS, **NUM_ANSWERS}


def _parse_answer(answer: str) -> list:
    """Parse a mathematical answer string, with LaTeX fallback."""
    parsed = parse(answer)
    # Handle LaTeX expressions that need $ delimiters for math-verify
    if not parsed:
        parsed = parse(f"$ {answer} $")
    return parsed


def _verify_math_answer(ground_truth: str, candidate: str) -> bool:
    """Check if candidate is mathematically equivalent to ground truth using math-verify."""
    try:
        return verify(_parse_answer(ground_truth), _parse_answer(candidate))
    except Exception:
        return False


class SubmitParams(BaseModel, extra="forbid"):
    answer: str


class PrincipiaCollection(Environment):
    """
    Training environment for STEM mathematical derivation.
    Two splits: 'train' (mathematical objects, LLM-judged) and 'train_numerical' (exact match).
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}):
        super().__init__(task_spec)
        self.task_id = task_spec["id"]
        self.problem = task_spec["problem_statement"]
        self.topic = task_spec.get("topic", "")
        self.answer_type = task_spec.get("answer_type", "")
        self.task_split = task_spec.get("split", "train")
        self.ground_truth = ALL_ANSWERS[self.task_id]

        self.is_numerical = self.task_split == "train_numerical"

        if not self.is_numerical:
            api_key = secrets.get("openai_api_key")
            if not api_key:
                raise ValueError(
                    "OpenAI API key required in secrets for LLM judging. "
                    "Pass via secrets={'openai_api_key': 'your-key'}"
                )
            self.client = openai.AsyncClient(api_key=api_key, max_retries=6)
        else:
            self.client = None

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "train_numerical"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "train":
            return mo_data
        if split == "train_numerical":
            return num_data
        return []

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(type="text", text=self.problem)]

    @tool
    async def submit(self, params: SubmitParams) -> ToolOutput:
        """Submit your answer for grading. This will end the episode."""
        if self.is_numerical:
            is_correct = _verify_math_answer(self.ground_truth, params.answer)
        else:
            is_correct = await judge_equivalence(
                problem=self.problem,
                ground_truth=self.ground_truth,
                candidate=params.answer,
                client=self.client,
            )

        reward = 1.0 if is_correct else 0.0

        return ToolOutput(
            blocks=[TextBlock(type="text", text=f"Reward: {reward}")],
            metadata={"correct": is_correct, "answer_type": self.answer_type},
            reward=reward,
            finished=True,
        )
