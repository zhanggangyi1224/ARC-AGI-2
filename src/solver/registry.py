"""Solver name → factory mapping.

Keep this tiny. New solvers register by importing here and adding an entry.
"""

from __future__ import annotations

from typing import Callable

from src.solver.base import Solver
from src.solver.identity import IdentitySolver

SOLVERS: dict[str, Callable[[], Solver]] = {
    IdentitySolver.name: IdentitySolver,
}


def get_solver(name: str) -> Solver:
    if name not in SOLVERS:
        raise ValueError(
            f"unknown solver {name!r}; available: {sorted(SOLVERS)}"
        )
    return SOLVERS[name]()
