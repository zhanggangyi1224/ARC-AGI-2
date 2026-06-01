"""Dataset loading for ARC-AGI-2 public splits.

A Task is a single puzzle: a list of demo (train) input/output pairs plus a list
of test input/output pairs. Grids are tuples of tuples of ints so they are
hashable and cheap to equality-compare in the metric.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Pair:
    input: Grid
    output: Grid | None  # None when ground truth is hidden (e.g. Kaggle test set)


@dataclass(frozen=True)
class Task:
    task_id: str
    train: tuple[Pair, ...]
    test: tuple[Pair, ...]


# Project-relative split paths. Keep these in one place so the runner, report,
# and tests all agree on where data lives.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "arc-agi-2"

SPLIT_DIRS: dict[str, Path] = {
    "public_train": DATA_ROOT / "training",
    "public_eval": DATA_ROOT / "evaluation",
}


def _to_grid(raw: list[list[int]]) -> Grid:
    return tuple(tuple(int(v) for v in row) for row in raw)


def _to_pair(raw: dict) -> Pair:
    out_raw = raw.get("output")
    return Pair(
        input=_to_grid(raw["input"]),
        output=_to_grid(out_raw) if out_raw is not None else None,
    )


def load_task(path: Path) -> Task:
    """Load a single task JSON file."""
    with open(path) as f:
        raw = json.load(f)
    return Task(
        task_id=path.stem,
        train=tuple(_to_pair(p) for p in raw["train"]),
        test=tuple(_to_pair(p) for p in raw["test"]),
    )


def iter_split(split: str, limit: int | None = None) -> Iterator[Task]:
    """Yield Tasks from a named split in deterministic (sorted) order."""
    if split not in SPLIT_DIRS:
        raise ValueError(
            f"unknown split {split!r}; expected one of {sorted(SPLIT_DIRS)}"
        )
    split_dir = SPLIT_DIRS[split]
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"split directory missing: {split_dir}. "
            f"Run the data fetch step (see README) to populate it."
        )
    paths = sorted(split_dir.glob("*.json"))
    if limit is not None:
        paths = paths[:limit]
    for p in paths:
        yield load_task(p)
