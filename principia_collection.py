"""
PrincipiaCollection Environment

Large-scale synthetic STEM training problems for RL.
248K mathematical_object problems (LLM-judged) + 306K numerical problems (math-verify).

Dataset: HuggingFace facebook/principia-collection
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import json
import logging
import math
import multiprocessing
import os
import re
from concurrent.futures.process import BrokenProcessPool
from typing import Optional

import openai
import pyarrow as pa
import pyarrow.parquet as pq
import sympy
from math_verify import parse, verify
from pydantic import BaseModel

from openreward.environments import Environment, JSONObject, Server, TextBlock, ToolOutput, terminal, tool

from judge import judge_equivalence

logger = logging.getLogger(__name__)

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
# Parquet rows left out of each split, listed by source row. The numerical
# split leaves out references that state no valid value exists ("No feasible
# integer N satisfies all three constraints."): the numerical grader can't
# credit a correct reply to them. build_excluded_rows.py regenerates the list.
_EXCLUDED_ROWS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "excluded_rows.json")


@functools.lru_cache(maxsize=None)
def _kept_rows(split: str) -> pa.Array:
    """Source parquet row of each task index in a split, skipping excluded rows."""
    path = _SPLIT_FILES.get(split)
    if path is None:
        raise KeyError(f"Unknown split: {split!r}")
    with open(_EXCLUDED_ROWS_FILE) as f:
        excluded = set(json.load(f).get(split, []))
    num_rows = pq.ParquetFile(path).metadata.num_rows
    return pa.array([row for row in range(num_rows) if row not in excluded], type=pa.int64())


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
    split = _split_for_id(task_id)
    row = _kept_rows(split)[_index_for_id(task_id)].as_py()
    return _answer_column(split)[row].as_py()


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
            for i, ans in enumerate(_answer_column(split).take(_kept_rows(split)).to_pylist()):
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


# LaTeX spacing commands (\, \; \: \!). math-verify strips a trailing unit
# such as \text{ns} only when it ends the expression, so spacing around it (or
# around a boxed answer) makes the unit parse as a symbol.
_LATEX_SPACING = re.compile(r"\\[,;:!]")
_DISPLAY_MATH = re.compile(r"(\$\$|\\\[)(.*?)(\$\$|\\\])", re.DOTALL)
# Markdown bold (**15/23**), but not a power such as 2**10.
_MARKDOWN_BOLD = re.compile(r"(?<![\w)])\*\*(?=\S)(.+?)(?<=\S)\*\*(?![\w(])", re.DOTALL)
# A reference that is one decimal number (decimal point required, optional
# exponent), optionally followed by a unit without digits: "0.0011", "57.0 mK",
# "9.1e-6". Integers, fractions and compound expressions don't match.
_DECIMAL_REFERENCE = re.compile(r"\s*(-?(\d*)\.(\d+)(?:[eE][-+]?\d+)?)(?:\s+[^\d\s]\D*)?", re.DOTALL)
_E_NOTATION = re.compile(r"(\d)[eE]([-+]?\d+)\b")
# A reference rounded to fewer significant figures than this is compared
# exactly: its rounding interval (over 5% of the value) is too wide to credit.
_MIN_ROUNDED_SIG_FIGS = 2


def _normalize_reply(text: str) -> str:
    """Remove formatting that hides the answer from math-verify: LaTeX spacing,
    markdown bold and code marks, and whitespace or sentence punctuation at the
    end of a display-math block."""
    text = _MARKDOWN_BOLD.sub(r"\1", _LATEX_SPACING.sub("", text)).replace("`", "")
    return _DISPLAY_MATH.sub(
        lambda m: m.group(1) + m.group(2).strip().rstrip(".,;").rstrip() + m.group(3), text
    )


def _numeric_values(parsed: list) -> list[float]:
    """Real numbers among math-verify parses (the right side of `x = value`)."""
    values = []
    for expr in parsed:
        if isinstance(expr, sympy.Eq):
            expr = expr.rhs
        if isinstance(expr, sympy.Expr) and expr.is_number and expr.is_real:
            values.append(float(expr.evalf()))
    return values


def _matches_rounded_reference(ground_truth: str, reply: str, parsed: list) -> bool:
    """Whether the reply's answer rounds to a decimal reference.

    A decimal reference is a rounded value: "0.0011" stands for any value in
    [0.00105, 0.00115], so a more precise answer such as 0.00112 is correct.
    The reference's significant figures set the tolerance (half a unit in its
    last digit), so it is relative, unlike math-verify's fixed 6 decimal places.
    Applies only to references matching _DECIMAL_REFERENCE with at least
    _MIN_ROUNDED_SIG_FIGS significant figures.
    """
    match = _DECIMAL_REFERENCE.fullmatch(ground_truth)
    if match is None:
        return False
    sig_figs = len((match.group(2) + match.group(3)).lstrip("0"))
    reference = float(match.group(1))
    if sig_figs < _MIN_ROUNDED_SIG_FIGS or reference == 0:
        return False
    # Half a unit in the reference's last significant digit, with slack for float error.
    half_unit = 0.5 * 10.0 ** (math.floor(math.log10(abs(reference))) - sig_figs + 1) * (1 + 1e-9)
    values = _numeric_values(parsed)
    if _E_NOTATION.search(reply):
        # math-verify reads "9.1e-6" as 9.1; rewrite it as 9.1*10^(-6).
        values += _numeric_values(_parse_answer(_E_NOTATION.sub(r"\1*10^(\2)", reply)))
    return any(abs(value - reference) <= half_unit for value in values)


def _verify_math_answer(ground_truth: str, candidate: str) -> bool:
    """Check if candidate is mathematically equivalent to ground truth using
    math-verify, or rounds to a decimal ground truth.

    math-verify bounds each parse and comparison with signal.alarm, which only
    works in a process's main thread: in any other thread it raises instead of
    grading. Call this on a main thread (e.g. in a _verify_pool() worker).
    """
    reply = _normalize_reply(candidate)
    parsed = _parse_answer(reply)
    gold = _parse_answer(_normalize_reply(ground_truth))
    return verify(gold, parsed) or _matches_rounded_reference(ground_truth, reply, parsed)


# math-verify runs sympy, which is CPU-bound and can run for minutes on
# pathological expressions (e.g. \boxed{9^{9^{9}}}). The env-server runs every
# session on a single asyncio event loop, so verification must not run on it.
# It runs in worker processes instead of threads: each task runs on the worker's
# main thread, where math-verify's own per-operation timeouts (signal.alarm)
# interrupt runaway sympy, and the loop never shares a GIL with it.
# _VERIFY_TIMEOUT_S is an outer backstop that also covers time queued behind
# other verifications; it is longer than math-verify's worst case for a
# typical answer (a few parses and comparisons, at most 5 s each).
_VERIFY_TIMEOUT_S = 60.0
_VERIFY_WORKERS = 2
_verify_pool_instance: Optional[concurrent.futures.ProcessPoolExecutor] = None


def _verify_pool() -> concurrent.futures.ProcessPoolExecutor:
    global _verify_pool_instance
    if _verify_pool_instance is None:
        # spawn: forking a process that already runs an event loop and threads
        # is unsafe, and the workers need nothing from the parent's memory.
        _verify_pool_instance = concurrent.futures.ProcessPoolExecutor(
            max_workers=_VERIFY_WORKERS,
            mp_context=multiprocessing.get_context("spawn"),
        )
    return _verify_pool_instance


async def _verify_math_answer_async(ground_truth: str, candidate: str) -> Optional[bool]:
    """Grade off the event loop. Returns None if the grader itself failed, so the
    caller can report the submission as ungraded rather than incorrect."""
    global _verify_pool_instance
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_verify_pool(), _verify_math_answer, ground_truth, candidate),
            timeout=_VERIFY_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        # Runaway sympy that outlived math-verify's own timeouts: treat as
        # incorrect. The worker stays busy until its alarm fires.
        logger.warning("math-verify timed out after %.0fs; scoring as incorrect", _VERIFY_TIMEOUT_S)
        return False
    except BrokenProcessPool:
        # A worker died (e.g. killed for memory). Replace the pool so later
        # submissions are graded.
        logger.exception("math-verify worker pool broke; replacing it")
        _verify_pool_instance = None
        return None
    except Exception:
        logger.exception("math-verify raised")
        return None


# Reward for a submission made after the task has already been graded. Negative
# so repeat submissions are actively discouraged, not merely left unscored.
REPEAT_SUBMISSION_PENALTY = -0.1


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

        # Graded submissions this session. @terminal already hides this tool from
        # the model, so the harness normally invokes it once at the end of the
        # rollout -- but Environment._call_tool dispatches by name and does not
        # exclude terminal tools, so a direct second call would re-grade and pay
        # out again. Defence in depth.
        self.submitted = 0

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
        """Task count from parquet metadata and the exclusion list — no data is materialized."""
        if split not in _SPLIT_FILES:
            return 0
        # Reading the parquet footer is blocking I/O; keep it off the event loop.
        return await asyncio.to_thread(lambda: len(_kept_rows(split)))

    @classmethod
    async def get_task(cls, split: str, index: int) -> JSONObject:
        """Single task spec served from the cached Arrow table (no full list)."""
        # _split_table() materializes the full (554K-row) table on first touch —
        # a multi-second, blocking read. get_task runs on the create hot path, so
        # offload it to a worker thread to avoid stalling the loop for other
        # sessions on the pod.
        row = await asyncio.to_thread(
            lambda: _split_table(split).slice(_kept_rows(split)[index].as_py(), 1).to_pylist()[0]
        )
        return _public_task_spec(split, index, row)

    @classmethod
    async def get_task_range(
        cls, split: str, start: Optional[int] = None, stop: Optional[int] = None
    ) -> list[JSONObject]:
        """Range of task specs from the cached Arrow table (slice, then convert)."""
        if split not in _SPLIT_FILES:
            return []

        def _materialize() -> list[JSONObject]:
            # Full-table materialization + slice->pylist is blocking; run off-loop.
            kept = _kept_rows(split)
            total = len(kept)
            lo = 0 if start is None else start
            hi = total if stop is None else stop
            if lo < 0:
                lo = max(total + lo, 0)
            if hi < 0:
                hi = max(total + hi, 0)
            lo = min(lo, total)
            hi = min(hi, total)
            rows = _split_table(split).take(kept.slice(lo, max(hi - lo, 0))).to_pylist()
            return [_public_task_spec(split, lo + i, row) for i, row in enumerate(rows)]

        return await asyncio.to_thread(_materialize)

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """Full task list. Kept for compatibility (tests, bulk export); no longer
        on the session-creation hot path and never run at import."""
        if split not in _SPLIT_FILES:
            return []
        rows = _split_table(split).take(_kept_rows(split)).to_pylist()
        return [_public_task_spec(split, i, row) for i, row in enumerate(rows)]

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(type="text", text=self.problem)]

    @terminal
    @tool
    async def submit(self, params: SubmitParams) -> ToolOutput:
        """Grade the assistant's final message against the ground truth.

        Terminal tool: hidden from the model, which replies with its answer as
        an ordinary message rather than calling a tool. The harness routes that
        message text here. Numerical answers are checked with math-verify
        (parses LaTeX, \boxed{...}, and other common formats out of prose);
        non-numerical answers go to an LLM equivalence judge. Since this is
        the environment's only tool, the model is given no tools at all.
        """
        if self.submitted > 0:
            return ToolOutput(
                blocks=[TextBlock(type="text", text="An answer has already been submitted for this task. "
                                       "This episode is over: it is not re-graded, and repeat "
                                       "submissions are penalised (reward -0.1).")],
                metadata={"already_submitted": True, "submission_count": self.submitted},
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        grader_error = False
        if self.is_numerical:
            verdict = await _verify_math_answer_async(self.ground_truth, params.answer)
            grader_error = verdict is None
            is_correct = bool(verdict)
        else:
            is_correct = await judge_equivalence(
                problem=self.problem,
                ground_truth=self.ground_truth,
                candidate=params.answer,
                client=self.client,
            )

        reward = 1.0 if is_correct else 0.0

        self.submitted += 1

        metadata = {"correct": is_correct, "answer_type": self.answer_type}
        if grader_error:
            # Keep a grader fault distinguishable from a wrong answer.
            metadata["grader_error"] = True
        return ToolOutput(
            blocks=[TextBlock(type="text", text=f"Reward: {reward}")],
            metadata=metadata,
            reward=reward,
            finished=True,
        )
