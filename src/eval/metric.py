"""Official ARC-AGI 2-attempt scoring metric.

Per-test-input: submit exactly two predicted output grids. Score is 1 iff either
attempt exactly equals the ground-truth grid (cell-for-cell), else 0. Task-level
and split-level scores are simple means over test outputs.

This module is intentionally tiny and dependency-free. It is the most
load-bearing piece of the eval harness; treat the tests in tests/test_metric.py
as the spec.
"""

from __future__ import annotations

from .data import Grid

NUM_ATTEMPTS = 2


def score_attempts(attempts: tuple[Grid, Grid], truth: Grid) -> int:
    """Return 1 if either attempt exactly matches truth, else 0.

    Raises ValueError if `attempts` does not contain exactly NUM_ATTEMPTS grids.
    Shape mismatches do not raise; they simply fail the equality check and score 0.
    """
    if len(attempts) != NUM_ATTEMPTS:
        raise ValueError(
            f"expected exactly {NUM_ATTEMPTS} attempts per test input, got {len(attempts)}"
        )
    a1, a2 = attempts
    return 1 if (a1 == truth or a2 == truth) else 0
