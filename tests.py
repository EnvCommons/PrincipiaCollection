"""
Tests for PrincipiaCollection environment.

Tests both grading paths:
- train (mathematical_object): LLM judge (requires OPENAI_API_KEY)
- train_numerical: math-verify equivalence (no LLM needed)
"""

import os

import pytest
from openreward.environments.types import JSONObject

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
