"""Score a Kaggle-format submission.json against a local split.

This is the bridge between "we have predictions from somewhere (Kaggle GPU
notebook, a teammate's run)" and "our local harness independently scored
them with the official 2-attempt metric." It's load-bearing for G1: the
Kaggle notebook prints pass@2 from TRM's evaluator, this prints pass@2 from
our metric, and the two must agree.

    python -m src.solver.from_submission \\
        --submission runs/trm_repro_kaggle/submission.json \\
        --split public_eval

Prints overall score and per-test-output count. Optionally writes a
runs/<id>/ summary so it shows up alongside other harness runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.eval.data import Grid, REPO_ROOT, iter_split
from src.eval.metric import NUM_ATTEMPTS, score_attempts


def _grid(raw) -> Grid:
    return tuple(tuple(int(v) for v in row) for row in raw)


def score_submission(submission_path: Path, split: str) -> dict:
    submission = json.loads(submission_path.read_text())

    total_outputs = 0
    total_correct = 0
    missing_tasks: list[str] = []
    mismatched_tasks: list[str] = []
    per_task: list[dict] = []

    for task in iter_split(split):
        if task.task_id not in submission:
            missing_tasks.append(task.task_id)
            # Missing predictions = automatic 0 for every test output in the task.
            total_outputs += len(task.test)
            per_task.append({
                "task_id": task.task_id,
                "num_test_outputs": len(task.test),
                "num_correct": 0,
                "missing": True,
            })
            continue

        attempts_per_test = submission[task.task_id]
        if len(attempts_per_test) != len(task.test):
            mismatched_tasks.append(
                f"{task.task_id}: submission has {len(attempts_per_test)} test "
                f"slots, dataset has {len(task.test)}"
            )

        per_test: list[int] = []
        for i, test_pair in enumerate(task.test):
            if i >= len(attempts_per_test):
                per_test.append(0)
                continue
            entry = attempts_per_test[i]
            attempts = (_grid(entry["attempt_1"]), _grid(entry["attempt_2"]))
            assert test_pair.output is not None
            per_test.append(score_attempts(attempts, test_pair.output))

        per_task.append({
            "task_id": task.task_id,
            "num_test_outputs": len(per_test),
            "num_correct": sum(per_test),
            "per_test": per_test,
        })
        total_outputs += len(per_test)
        total_correct += sum(per_test)

    return {
        "submission": str(submission_path),
        "split": split,
        "num_tasks": len(per_task),
        "num_test_outputs": total_outputs,
        "num_correct": total_correct,
        "score": total_correct / total_outputs if total_outputs else 0.0,
        "missing_tasks": missing_tasks,
        "mismatched_tasks": mismatched_tasks,
        "per_task": per_task,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score a Kaggle submission.json locally")
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--split", default="public_eval", choices=["public_train", "public_eval"])
    p.add_argument(
        "--write-run",
        action="store_true",
        help="also write runs/<utc-ts>-from_submission/{summary,per_task}.json",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    assert NUM_ATTEMPTS == 2, "metric must enforce exactly 2 attempts"

    result = score_submission(args.submission, args.split)

    if args.write_run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = REPO_ROOT / "runs" / f"{stamp}-from_submission"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps({
            "run_id": out.name,
            "split": result["split"],
            "solver": "from_submission",
            "submission": result["submission"],
            "num_tasks": result["num_tasks"],
            "num_test_outputs": result["num_test_outputs"],
            "num_correct": result["num_correct"],
            "score": result["score"],
        }, indent=2))
        (out / "per_task.json").write_text(json.dumps(result["per_task"], indent=2))
        print(f"wrote {out}")

    print(
        f"score={result['score']:.4f} ({result['num_correct']}/{result['num_test_outputs']}) "
        f"tasks={result['num_tasks']} split={result['split']}"
    )
    if result["missing_tasks"]:
        print(f"[warn] {len(result['missing_tasks'])} tasks missing from submission "
              f"(first 5: {result['missing_tasks'][:5]})")
    if result["mismatched_tasks"]:
        print(f"[warn] {len(result['mismatched_tasks'])} tasks with test-count mismatch:")
        for m in result["mismatched_tasks"][:5]:
            print(f"  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
