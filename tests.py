"""
Tests for PrincipiaCollection environment.

Tests both grading paths:
- train (mathematical_object): LLM judge (requires OPENAI_API_KEY)
- train_numerical: math-verify equivalence (no LLM needed)
"""

import asyncio
import os
import threading
import time
from concurrent.futures.process import BrokenProcessPool

import pytest
from openreward.environments.types import JSONObject

import principia_collection
from principia_collection import (
    PrincipiaCollection,
    SubmitParams,
    ALL_ANSWERS,
    _verify_math_answer,
)

MO_TASKS = PrincipiaCollection.list_tasks("train")
NUM_TASKS = PrincipiaCollection.list_tasks("train_numerical")
SECRETS = {"openai_api_key": os.environ.get("OPENAI_API_KEY", "")}


# --- math-verify unit tests (no LLM needed) ---

def test_verify_integer():
    assert _verify_math_answer("42", "42")
    assert _verify_math_answer("42", "  42  ")
    assert not _verify_math_answer("42", "43")


def test_verify_float():
    assert _verify_math_answer("3.14", "3.14")
    assert _verify_math_answer("3.14", "3.140000")
    assert not _verify_math_answer("3.14", "3.15")


def test_verify_fraction():
    assert _verify_math_answer("1/2", "0.5")
    assert _verify_math_answer("\\frac{1}{3}", "1/3")
    assert _verify_math_answer("\\dfrac{3}{28}", "3/28")


def test_verify_latex_fraction():
    assert _verify_math_answer("$\\frac{1}{4}$", "0.25")
    assert _verify_math_answer("\\frac{4}{15}", "4/15")


# References are rounded decimals: an answer that rounds to the reference at the
# reference's significant figures is correct; integers, fractions and references
# with a single significant figure stay exact.
@pytest.mark.parametrize("reference,answer,ok", [
    ("0.0011", "\\boxed{0.00112}", True),
    ("0.0011", "\\[\n\\Delta\\theta \\approx 0.001117^\\circ\n\\]", True),
    ("0.0011", "\\boxed{0.00116}", False),
    ("0.0011", "\\boxed{0.0012}", False),
    ("0.0011", "\\boxed{0.001}", False),
    ("0.93", "**Answer:** 0.92968", True),
    ("0.93", "**Answer:** 0.9249", False),
    ("57.0 mK", "57.03 mK", True),
    ("57.0 mK", "57.1 mK", False),
    ("9.1e-6", "\\[\n\\boxed{9.108\\times10^{-6}}\n\\]", True),
    ("9.1e-6", "As a decimal number: **9.108e-6**", True),
    ("9.1e-6", "\\boxed{9.2\\times10^{-6}}", False),
    ("9.1e-6", "**9.108e-5**", False),
    ("0.6", "\\boxed{0.62}", False),
    ("2", "\\boxed{2.4}", False),
    ("15/23", "\\boxed{0.652}", False),
])
def test_verify_rounded_decimal_reference(reference: str, answer: str, ok: bool):
    assert _verify_math_answer(reference, answer) is ok


# Formatting around a correct final answer must not hide it from the grader.
@pytest.mark.parametrize("reference,answer,ok", [
    ("\\displaystyle \\frac{10}{3}\\ \\text{m}",
     "\\[\n\\boxed{\\;\\overline{PQ}=\\dfrac{10}{3}\\text{ metres}\\;}\n\\]", True),
    ("\\frac{5}{4}\\,\\text{ns}", "The lifetime is:\n\n\\[\n\\frac{5}{4}\\,\\text{ns}\n\\]", True),
    ("2/11", "Their ratio is\n\n\\[\n\\frac{2}{11}.\n\\]", True),
    ("15/23", "The volume fraction is:\n\n**15/23**", True),
    ("15/23", "The volume fraction is:\n\n**15/2**", False),
    ("1/7 points", "**Answer: `1/7 points`**", True),
    ("1/7 points", "**Answer: `1/8 points`**", False),
    ("\\frac{24\\,910\\,430\\,999}{312\\,500\\,000}", "\\boxed{\\frac{24\\,910\\,430\\,999}{312\\,500\\,000}}", True),
    ("\\frac{24\\,910\\,430\\,999}{312\\,500\\,000}", "\\boxed{0}", False),
])
def test_verify_formatted_reply(reference: str, answer: str, ok: bool):
    assert _verify_math_answer(reference, answer) is ok


# --- Numerical split: gold and xfail ---

@pytest.mark.asyncio
@pytest.mark.parametrize("task", NUM_TASKS[:5])
async def test_numerical_gold(task: JSONObject):
    """Submit the ground truth numeric answer — should get reward=1.0."""
    env = PrincipiaCollection(task_spec=task, secrets=SECRETS)
    gold_answer = ALL_ANSWERS[task["id"]]
    result = await env.submit(SubmitParams(answer=gold_answer))
    assert result.reward == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("task", NUM_TASKS[:5])
async def test_numerical_xfail(task: JSONObject):
    """Submit a clearly wrong numeric answer — should get reward=0.0."""
    env = PrincipiaCollection(task_spec=task, secrets=SECRETS)
    result = await env.submit(SubmitParams(answer="999999999.12345"))
    assert result.reward == 0.0


# --- Numerical grading through the async submit path ---

def _numerical_spec(answer: str) -> JSONObject:
    return {
        "id": "num_0",
        "problem_statement": "Compute epsilon.",
        "answer": answer,
        "answer_type": "Decimal w/o unit",
        "split": "train_numerical",
    }


def _reply(boxed: str) -> str:
    """A final message in the shape models write: working, then a boxed value."""
    return (
        "Dividing by $0.08\\,\\pi^{4}$ gives $\\displaystyle \\varepsilon=\\frac{4}{\\pi}$.\n\n"
        "Numerically,\n$$\\varepsilon=\\frac{4}{\\pi}\\approx 1.2732395\\ldots$$\n\n"
        f"Rounded to three decimal places,\n\n$$\\boxed{{{boxed}}}$$"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reply,expected", [
    (_reply("1.273"), 1.0),
    ("\\boxed{1.273}", 1.0),
    ("1.273", 1.0),
    (_reply("1.274"), 0.0),
    (_reply("1.2732"), 1.0),
    ("\\boxed{-1.273}", 0.0),
])
async def test_numerical_submit_grades_prose_reply(reply: str, expected: float):
    env = PrincipiaCollection(task_spec=_numerical_spec("1.273"))
    result = await env.submit(SubmitParams(answer=reply))
    assert result.reward == expected
    assert "grader_error" not in result.metadata


def test_numerical_submit_from_non_main_thread():
    """Grading must not depend on which thread runs the event loop."""
    out = {}

    def run():
        env = PrincipiaCollection(task_spec=_numerical_spec("1.273"))
        out["result"] = asyncio.run(env.submit(SubmitParams(answer=_reply("1.273"))))

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=120)
    assert out["result"].reward == 1.0


@pytest.mark.asyncio
async def test_pathological_answer_is_bounded_and_does_not_block_loop():
    gaps = []
    stop = asyncio.Event()

    async def heartbeat():
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(0.05)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    hb = asyncio.create_task(heartbeat())
    env = PrincipiaCollection(task_spec=_numerical_spec("1.273"))
    t0 = time.monotonic()
    result = await env.submit(SubmitParams(answer="\\boxed{9^{9^{9}}}"))
    elapsed = time.monotonic() - t0
    stop.set()
    await hb
    assert result.reward == 0.0
    assert elapsed < principia_collection._VERIFY_TIMEOUT_S + 5
    assert max(gaps) < 1.0


class _BrokenPool:
    def submit(self, *args, **kwargs):
        raise BrokenProcessPool("worker died")


@pytest.mark.asyncio
async def test_grader_fault_is_flagged_not_silently_incorrect(monkeypatch):
    monkeypatch.setattr(principia_collection, "_verify_pool", lambda: _BrokenPool())
    env = PrincipiaCollection(task_spec=_numerical_spec("1.273"))
    result = await env.submit(SubmitParams(answer="\\boxed{1.273}"))
    assert result.reward == 0.0
    assert result.finished
    assert result.metadata.get("grader_error") is True
    assert "1.273" not in str(result.metadata) + result.blocks[0].text


def _clear_split_caches():
    for fn in (principia_collection._kept_rows, principia_collection._split_table,
               principia_collection._answer_column):
        fn.cache_clear()
    principia_collection.__dict__.pop("ALL_ANSWERS", None)


@pytest.fixture
def five_row_numerical_split(tmp_path, monkeypatch):
    """A 5-row numerical split whose rows 1 and 4 are excluded."""
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq

    answers = ["10", "No such N exists", "30", "40", "none"]
    path = tmp_path / "numerical.parquet"
    pq.write_table(pa.table({
        "topic": ["t"] * 5,
        "problem_statement": [f"problem {row}" for row in range(5)],
        "answer": answers,
        "answer_type": ["Integer w/o unit"] * 5,
    }), path)
    excluded = tmp_path / "excluded_rows.json"
    excluded.write_text(json.dumps({"train_numerical": [1, 4]}))
    monkeypatch.setattr(principia_collection, "_EXCLUDED_ROWS_FILE", str(excluded))
    monkeypatch.setitem(principia_collection._SPLIT_FILES, "train_numerical", str(path))
    _clear_split_caches()
    yield
    _clear_split_caches()


@pytest.mark.asyncio
async def test_excluded_rows_are_not_served(five_row_numerical_split):
    split = "train_numerical"
    assert await PrincipiaCollection.num_tasks(split) == 3

    served = ["problem 0", "problem 2", "problem 3"]
    listed = PrincipiaCollection.list_tasks(split)
    assert [t["problem_statement"] for t in listed] == served
    assert [t["id"] for t in listed] == ["num_0", "num_1", "num_2"]
    ranged = await PrincipiaCollection.get_task_range(split)
    assert [t["problem_statement"] for t in ranged] == served
    ranged = await PrincipiaCollection.get_task_range(split, 1, 3)
    assert [t["problem_statement"] for t in ranged] == served[1:]

    task = await PrincipiaCollection.get_task(split, 1)
    assert (task["id"], task["problem_statement"]) == ("num_1", "problem 2")
    assert "answer" not in task
    # The ground truth follows the served row, not the source row with the same index.
    env = PrincipiaCollection(task_spec=task)
    assert env.ground_truth == "30"
    assert (await env.submit(SubmitParams(answer="\\boxed{30}"))).reward == 1.0
    assert principia_collection.ALL_ANSWERS["num_2"] == "40"


def test_shipped_exclusions():
    import json
    from build_excluded_rows import states_no_value

    with open(principia_collection._EXCLUDED_ROWS_FILE) as f:
        rows = json.load(f)
    assert set(rows) == {"train_numerical"}
    assert 62174 in rows["train_numerical"]
    assert rows["train_numerical"] == sorted(set(rows["train_numerical"]))

    for reference in [
        "No feasible integer N satisfies all three constraints.",
        "none",
        "\\text{No such a exists}",
        "Target cannot be achieved within the given time and speed limits.",
        "The population never reaches 4.0×10⁸ cells; it will decline to extinction.",
    ]:
        assert states_no_value(reference), reference
    for reference in [
        "42", "\\frac{22}{15} \\text{ central charge units}", "70 vehicles per hour",
        "0.625 bits per symbol", "\\infty", "n+1 directions", "12\u202f%",
    ]:
        assert not states_no_value(reference), reference

    # Why these rows can't be graded numerically: a quoted parameter is credited.
    reference = "No real critical pressure; the ratio never reaches 1.5, so the material remains insulating."
    assert _verify_math_answer(reference, "\\boxed{1.5}")


# --- Mathematical object split: gold and xfail (requires LLM) ---

@pytest.mark.asyncio
@pytest.mark.parametrize("task", MO_TASKS[:3])
async def test_mo_gold(task: JSONObject):
    """Submit the ground truth math answer — should get reward=1.0."""
    env = PrincipiaCollection(task_spec=task, secrets=SECRETS)
    gold_answer = ALL_ANSWERS[task["id"]]
    result = await env.submit(SubmitParams(answer=gold_answer))
    assert result.reward == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("task", MO_TASKS[:3])
async def test_mo_xfail(task: JSONObject):
    """Submit a clearly wrong math answer — should get reward=0.0."""
    env = PrincipiaCollection(task_spec=task, secrets=SECRETS)
    result = await env.submit(SubmitParams(answer="banana"))
    assert result.reward == 0.0
