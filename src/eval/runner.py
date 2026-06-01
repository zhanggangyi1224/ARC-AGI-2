"""Local eval harness CLI.

    python -m src.eval.runner --split public_eval --solver identity [--limit N]

Loads tasks → runs the named solver → scores with the 2-attempt metric →
writes `runs/<timestamp>/{summary,per_task,config}.json` and prints the score.

This is the trustworthy local ruler. The Kaggle leaderboard is *confirmation*,
not exploration — check changes here first.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.eval.data import REPO_ROOT, Task, iter_split
from src.eval.metric import NUM_ATTEMPTS, score_attempts
from src.solver.base import Solver
from src.solver.registry import SOLVERS, get_solver

RUNS_ROOT = REPO_ROOT / "runs"


@dataclass
class TaskResult:
    task_id: str
    num_test_outputs: int
    num_correct: int
    score: float  # mean over this task's test outputs
    per_test: list[int]  # 0/1 per test input, in order


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Torch/numpy not imported here on purpose — keep the harness import-light.
    # Solvers that use them should seed inside their own setup.


def _run_task(solver: Solver, task: Task) -> TaskResult:
    predictions = solver.predict(task)
    if len(predictions) != len(task.test):
        raise ValueError(
            f"solver {solver.name!r} returned {len(predictions)} prediction sets "
            f"for task {task.task_id!r}, expected {len(task.test)}"
        )
    per_test: list[int] = []
    for attempts, test_pair in zip(predictions, task.test):
        if len(attempts) != NUM_ATTEMPTS:
            raise ValueError(
                f"solver {solver.name!r} returned {len(attempts)} attempts for a "
                f"test input on task {task.task_id!r}, expected {NUM_ATTEMPTS}"
            )
        if test_pair.output is None:
            raise ValueError(
                f"task {task.task_id!r} has no ground truth; cannot score locally"
            )
        per_test.append(score_attempts(tuple(attempts), test_pair.output))
    return TaskResult(
        task_id=task.task_id,
        num_test_outputs=len(per_test),
        num_correct=sum(per_test),
        score=sum(per_test) / len(per_test) if per_test else 0.0,
        per_test=per_test,
    )


def run(split: str, solver_name: str, limit: int | None, seed: int, out_dir: Path) -> dict:
    _seed_everything(seed)
    solver = get_solver(solver_name)

    started = time.monotonic()
    task_results: list[TaskResult] = []
    for task in iter_split(split, limit=limit):
        task_results.append(_run_task(solver, task))
    elapsed_s = time.monotonic() - started

    total_outputs = sum(r.num_test_outputs for r in task_results)
    total_correct = sum(r.num_correct for r in task_results)
    overall_score = total_correct / total_outputs if total_outputs else 0.0

    summary = {
        "run_id": out_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "solver": solver_name,
        "limit": limit,
        "seed": seed,
        "num_tasks": len(task_results),
        "num_test_outputs": total_outputs,
        "num_correct": total_correct,
        "score": overall_score,
        "elapsed_s": round(elapsed_s, 3),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_task.json", "w") as f:
        json.dump([asdict(r) for r in task_results], f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {"split": split, "solver": solver_name, "limit": limit, "seed": seed},
            f,
            indent=2,
        )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARC-AGI-2 local eval harness")
    p.add_argument("--split", default="public_eval", choices=["public_train", "public_eval"])
    p.add_argument("--solver", default="identity", choices=sorted(SOLVERS))
    p.add_argument("--limit", type=int, default=None, help="run at most N tasks (for smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory; default runs/<UTC-timestamp>-<solver>",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out = RUNS_ROOT / f"{stamp}-{args.solver}"

    summary = run(
        split=args.split,
        solver_name=args.solver,
        limit=args.limit,
        seed=args.seed,
        out_dir=args.out,
    )
    print(
        f"[{summary['run_id']}] split={summary['split']} solver={summary['solver']} "
        f"tasks={summary['num_tasks']} outputs={summary['num_test_outputs']} "
        f"score={summary['score']:.4f} "
        f"({summary['num_correct']}/{summary['num_test_outputs']}) "
        f"elapsed={summary['elapsed_s']}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
