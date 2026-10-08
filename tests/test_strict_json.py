"""Tests for strict JSON serialization (no NaN tokens, ever)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from portfolio_optimization_lab.pipeline import strict_json_dumps

REPO_ROOT = Path(__file__).resolve().parents[1]


def _raise_on_constant(token: str):
    raise AssertionError(f"non-finite JSON constant encountered: {token}")


def test_strict_json_dumps_converts_nonfinite_to_null_and_round_trips():
    payload = {
        "a": float("nan"),
        "b": [1.0, float("inf"), float("-inf")],
        "c": {"d": None, "e": "text", "f": [[float("nan")]]},
        "g": 2,
    }
    text = strict_json_dumps(payload)
    # strict parser: parse_constant fires on NaN/Infinity tokens; there must be none
    parsed = json.loads(text, parse_constant=_raise_on_constant)
    assert parsed["a"] is None
    assert parsed["b"] == [1.0, None, None]
    assert parsed["c"]["f"] == [[None]]
    assert parsed["g"] == 2
    assert "NaN" not in text and "Infinity" not in text


def test_allow_nan_false_guard_catches_raw_nan():
    """The guarantee mechanism itself: json.dumps(allow_nan=False) raises."""
    with pytest.raises(ValueError):
        json.dumps({"x": float("nan")}, allow_nan=False)


def test_committed_optimization_results_strict_parses():
    """The shipped artifact must contain no NaN/Infinity tokens."""
    path = REPO_ROOT / "reports" / "optimization_results.json"
    text = path.read_text(encoding="utf-8")
    parsed = json.loads(text, parse_constant=_raise_on_constant)
    assert parsed["project"] == "portfolio-optimization-lab"
    # infeasible frontier targets are emitted as null, not NaN
    infeasible = [row for row in parsed["frontier"] if row["volatility"] is None]
    assert all("volatility" in row for row in parsed["frontier"])
    assert isinstance(infeasible, list)
