"""Spec for the official ARC-AGI 2-attempt metric.

These are the rules the rest of the harness trusts; if any of these change,
you are changing the contract with Kaggle, not just refactoring.
"""

from __future__ import annotations

import pytest

from src.eval.metric import NUM_ATTEMPTS, score_attempts


GRID_A = ((0, 1), (2, 3))
GRID_B = ((9, 9), (9, 9))
GRID_C = ((0, 0), (0, 0))
GRID_DIFF_SHAPE = ((0, 1, 2),)


def test_score_is_1_when_attempt_1_matches() -> None:
    assert score_attempts((GRID_A, GRID_B), GRID_A) == 1


def test_score_is_1_when_attempt_2_matches() -> None:
    assert score_attempts((GRID_B, GRID_A), GRID_A) == 1


def test_score_is_1_when_both_attempts_match() -> None:
    # Allowed (just wasteful): two identical correct attempts still score 1.
    assert score_attempts((GRID_A, GRID_A), GRID_A) == 1


def test_score_is_0_when_neither_attempt_matches() -> None:
    assert score_attempts((GRID_B, GRID_C), GRID_A) == 0


def test_score_is_0_on_shape_mismatch() -> None:
    # Wrong-shape predictions are not an error — they simply don't equal truth.
    assert score_attempts((GRID_DIFF_SHAPE, GRID_DIFF_SHAPE), GRID_A) == 0


def test_score_is_0_on_off_by_one_pixel() -> None:
    near_miss = ((0, 1), (2, 4))  # last cell differs by 1
    assert score_attempts((near_miss, GRID_B), GRID_A) == 0


def test_score_requires_exactly_two_attempts() -> None:
    with pytest.raises(ValueError):
        score_attempts((GRID_A,), GRID_A)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        score_attempts((GRID_A, GRID_A, GRID_A), GRID_A)  # type: ignore[arg-type]


def test_num_attempts_is_two() -> None:
    # Belt-and-braces: if someone ever bumps this, tests will scream.
    assert NUM_ATTEMPTS == 2
