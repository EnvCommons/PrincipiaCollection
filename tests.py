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
