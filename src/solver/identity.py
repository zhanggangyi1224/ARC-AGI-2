"""Trivial solver: predicts the test input unchanged, twice.

Exists only to exercise the harness end-to-end before a real solver lands.
Expected score on public_eval: ~0 (a handful of identity-rule tasks may slip
through, which is itself a useful sanity check on the metric).
"""

from __future__ import annotations

from src.eval.data import Grid, Task


class IdentitySolver:
    name = "identity"

    def predict(self, task: Task) -> list[tuple[Grid, Grid]]:
        return [(pair.input, pair.input) for pair in task.test]
