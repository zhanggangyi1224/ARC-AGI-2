"""Solver interface.

A Solver consumes a Task and returns exactly 2 candidate output grids for each
of the task's test inputs, in order. The harness owns scoring; the solver only
produces predictions.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.eval.data import Grid, Task


@runtime_checkable
class Solver(Protocol):
    name: str

    def predict(self, task: Task) -> list[tuple[Grid, Grid]]:
        """Return a list of (attempt_1, attempt_2) tuples, one per task.test."""
        ...
