"""Sanity tests for the benchmark grader primitives.

If these ever break, every benchmark result afterward becomes suspect, so we
keep them quick and cover the edge cases explicitly.
"""

from __future__ import annotations

from benchmarks.tasks import (
    TASKS,
    both,
    contains,
    contains_all,
    number_close,
    tasks_by_category,
    used_tool,
)
from agentmesh.utils.types import Step


def test_task_count_is_50():
    assert len(TASKS) == 50


def test_task_ids_are_unique():
    ids = [t.id for t in TASKS]
    assert len(ids) == len(set(ids))


def test_categories_are_expected():
    cats = set(tasks_by_category())
    assert cats == {"math", "search", "composite", "stateful", "adversarial"}


def test_contains_is_case_insensitive():
    g = contains("Paris")
    assert g("the capital is paris.", []) is True
    assert g("The Capital Is PARIS.", []) is True
    assert g("no match here", []) is False


def test_contains_all_requires_all_needles():
    g = contains_all("a", "b")
    assert g("a then b", []) is True
    assert g("only a", []) is False


def test_number_close_tolerance():
    g = number_close(100.0, tol=0.5)
    assert g("the answer is 100", []) is True
    assert g("approximately 100.3 units", []) is True
    assert g("approximately 99.7 units", []) is True
    assert g("got 105", []) is False
    assert g("no number here at all", []) is False


def test_number_close_picks_first_valid_number():
    g = number_close(42)
    # multiple numbers in the string — any match within tol counts
    assert g("had 3 attempts; final 42", []) is True


def test_used_tool_accepts_flat_and_namespaced():
    steps = [
        Step(kind="tool_call", summary="x", payload={"tool": "demo__calculator"}),
    ]
    assert used_tool("calculator")("", steps) is True
    assert used_tool("demo__calculator")("", steps) is True
    assert used_tool("search")("", steps) is False


def test_both_combinator():
    g = both(contains("hello"), number_close(5))
    assert g("hello 5", []) is True
    assert g("hello", []) is False
    assert g("5 only", []) is False


def test_every_task_has_required_fields():
    for t in TASKS:
        assert t.id and t.category and t.prompt
        assert callable(t.grader)
