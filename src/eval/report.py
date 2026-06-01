"""Per-task pass/fail + simple failure clustering for a completed run.

    python -m src.eval.report --run runs/<id>

Clusters failures by coarse, model-agnostic features (output grid area, color
count) so we can eyeball "where does the solver lose ground?" before doing
deeper error analysis. This is intentionally crude — the goal is to surface
the next thing to look at, not to be the analysis itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from src.eval.data import REPO_ROOT, load_task

RUNS_ROOT = REPO_ROOT / "runs"


def _bucket_area(area: int) -> str:
    # ARC grids are bounded at 30x30 = 900 cells.
    if area <= 25:
        return "<=5x5"
    if area <= 100:
        return "<=10x10"
    if area <= 400:
        return "<=20x20"
    return ">20x20"


def report(run_dir: Path) -> None:
    summary = json.loads((run_dir / "summary.json").read_text())
    per_task = json.loads((run_dir / "per_task.json").read_text())

    print(f"Run: {summary['run_id']}")
    print(
        f"  split={summary['split']} solver={summary['solver']} "
        f"score={summary['score']:.4f} "
        f"({summary['num_correct']}/{summary['num_test_outputs']}) "
        f"tasks={summary['num_tasks']}"
    )

    fail_ids = [r["task_id"] for r in per_task if r["num_correct"] < r["num_test_outputs"]]
    pass_ids = [r["task_id"] for r in per_task if r["num_correct"] == r["num_test_outputs"]]
    print(f"  fully solved: {len(pass_ids)}    failed (>=1 wrong test): {len(fail_ids)}")

    split = summary["split"]
    by_area: dict[str, list[int]] = defaultdict(list)  # bucket -> list of 0/1 per test output
    by_colors: dict[int, list[int]] = defaultdict(list)

    for r in per_task:
        task = load_task(_task_path(split, r["task_id"]))
        for score, test_pair in zip(r["per_test"], task.test):
            truth = test_pair.output
            assert truth is not None, "report only handles scored splits"
            h, w = len(truth), len(truth[0]) if truth else 0
            by_area[_bucket_area(h * w)].append(score)
            colors = {v for row in truth for v in row}
            by_colors[len(colors)].append(score)

    print("\n  failures by output area:")
    for bucket in ("<=5x5", "<=10x10", "<=20x20", ">20x20"):
        ss = by_area.get(bucket, [])
        if not ss:
            continue
        acc = sum(ss) / len(ss)
        print(f"    {bucket:>10s}  n={len(ss):4d}  acc={acc:.3f}  fails={len(ss) - sum(ss)}")

    print("\n  failures by output color count:")
    for k in sorted(by_colors):
        ss = by_colors[k]
        acc = sum(ss) / len(ss)
        print(f"    {k:>2d} colors  n={len(ss):4d}  acc={acc:.3f}  fails={len(ss) - sum(ss)}")


def _task_path(split: str, task_id: str) -> Path:
    # Resolve via iter_split's SPLIT_DIRS to stay consistent with the loader.
    from src.eval.data import SPLIT_DIRS

    return SPLIT_DIRS[split] / f"{task_id}.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARC-AGI-2 eval report")
    p.add_argument("--run", type=Path, required=True, help="path to runs/<id> directory")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = args.run if args.run.is_absolute() else (REPO_ROOT / args.run)
    if not run_dir.is_dir():
        print(f"run directory not found: {run_dir}", file=sys.stderr)
        return 2
    report(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
