"""
PrincipiaCollection Environment

Large-scale synthetic STEM training problems for RL.
248K mathematical_object problems (LLM-judged) + 306K numerical problems (math-verify).

Dataset: HuggingFace facebook/principia-collection
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import openai
import pyarrow.parquet as pq
from math_verify import parse, verify
from pydantic import BaseModel

from openreward.environments import Environment, JSONObject, Server, TextBlock, ToolOutput, tool

from judge import judge_equivalence

# Paths only — no parquet is read at import, so workers boot without paying the
# ~30s cost of materializing the 554K-row dataset (which timed out create_session
# under concurrent cold starts). Data is loaded lazily and cached per worker.
_DATA_DIR = "/orwd_data" if os.path.exists("/orwd_data") else "."
_SPLIT_FILES = {
    "train": os.path.join(_DATA_DIR, "principia_collection_mathematical_object.parquet"),
    "train_numerical": os.path.join(_DATA_DIR, "principia_collection_numerical.parquet"),
}
_SPLIT_ID_PREFIX = {"train": "mo", "train_numerical": "num"}
_PREFIX_SPLIT = {prefix: split for split, prefix in _SPLIT_ID_PREFIX.items()}


@functools.lru_cache(maxsize=None)
def _split_table(split: str):
    """Full Arrow table for a split, read once and cached per worker. Used by the
    task-enumeration endpoints, not the session hot path."""
    path = _SPLIT_FILES.get(split)
    if path is None:
        raise KeyError(f"Unknown split: {split!r}")
    return pq.read_table(path)


@functools.lru_cache(maxsize=None)
def _answer_column(split: str):
    """The ``answer`` column for a split, read once and cached per worker. This is
    the only dataset access on the session hot path (``__init__``)."""
    path = _SPLIT_FILES.get(split)
    if path is None:
        raise KeyError(f"Unknown split: {split!r}")
    return pq.read_table(path, columns=["answer"]).column("answer")


def _split_for_id(task_id: str) -> str:
    prefix, _, _ = task_id.rpartition("_")
    split = _PREFIX_SPLIT.get(prefix)
    if split is None:
        raise KeyError(f"Unrecognized task id: {task_id!r}")
    return split


def _index_for_id(task_id: str) -> int:
    _, _, idx = task_id.rpartition("_")
    return int(idx)


def _ground_truth_for(task_id: str) -> str:
    """Resolve a task's ground-truth answer without materializing the dataset."""
    return _answer_column(_split_for_id(task_id))[_index_for_id(task_id)].as_py()


def _public_task_spec(split: str, index: int, row: dict) -> JSONObject:
    """Build the public task spec for a row: strip the answer, stamp id/split."""
    row.pop("answer", None)  # never expose the ground truth in the task spec
    row["id"] = f"{_SPLIT_ID_PREFIX[split]}_{index}"
    row["split"] = split
    return row


def __getattr__(name: str):
    """Build ``ALL_ANSWERS`` lazily (used by tests) so the full {id: answer} map
    is never materialized at import."""
    if name == "ALL_ANSWERS":
        all_answers: dict[str, str] = {}
        for split, prefix in _SPLIT_ID_PREFIX.items():
            for i, ans in enumerate(_answer_column(split).to_pylist()):
                all_answers[f"{prefix}_{i}"] = ans
        globals()["ALL_ANSWERS"] = all_answers
        return all_answers
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
        # Public specs have the answer stripped, so resolve it from the dataset;
        # an inline answer (e.g. hand-built test specs) is honored if present.
        if "answer" in task_spec:
            self.ground_truth = task_spec["answer"]
        else:
            self.ground_truth = _ground_truth_for(self.task_id)

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
    async def num_tasks(cls, split: str) -> int:
        """Row count straight from parquet metadata — no data is materialized."""
        path = _SPLIT_FILES.get(split)
        if path is None:
            return 0
        return pq.ParquetFile(path).metadata.num_rows

    @classmethod
    async def get_task(cls, split: str, index: int) -> JSONObject:
        """Single task spec served from the cached Arrow table (no full list)."""
        row = _split_table(split).slice(index, 1).to_pylist()[0]
        return _public_task_spec(split, index, row)

    @classmethod
    async def get_task_range(
        cls, split: str, start: Optional[int] = None, stop: Optional[int] = None
    ) -> list[JSONObject]:
        """Range of task specs from the cached Arrow table (slice, then convert)."""
        if split not in _SPLIT_FILES:
            return []
        table = _split_table(split)
        total = table.num_rows
        if start is None:
            start = 0
        if stop is None:
            stop = total
        if start < 0:
            start = max(total + start, 0)
        if stop < 0:
            stop = max(total + stop, 0)
        start = min(start, total)
        stop = min(stop, total)
        rows = table.slice(start, max(stop - start, 0)).to_pylist()
        return [_public_task_spec(split, start + i, row) for i, row in enumerate(rows)]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """Full task list. Kept for compatibility (tests, bulk export); no longer
        on the session-creation hot path and never run at import."""
        if split not in _SPLIT_FILES:
            return []
        rows = _split_table(split).to_pylist()
        return [_public_task_spec(split, i, row) for i, row in enumerate(rows)]

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
