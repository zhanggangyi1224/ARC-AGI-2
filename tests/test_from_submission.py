"""Pin the from_submission scorer.

This is the bridge between TRM's evaluator (Kaggle-side) and our metric
(local). If TRM reports pass@2 = X and we report Y on the same submission
file, X == Y is the G1 acceptance criterion. These tests pin what "our Y"
means concretely.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.data import iter_split, SPLIT_DIRS
from src.solver.from_submission import score_submission


# --- helpers ----------------------------------------------------------------


def _grid_to_list(g) -> list[list[int]]:
    return [list(r) for r in g]


def _synthetic_submission(tasks, attempt_1: str, attempt_2: str) -> dict:
    """Build a Kaggle-format submission picking known sources for each attempt.

    attempt_1 / attempt_2 ∈ {"truth", "input", "wrong"}.
    """
    sub: dict = {}
    for t in tasks:
        rows = []
        for p in t.test:
            assert p.output is not None
            wrong = [[(v + 1) % 10 for v in row] for row in p.output]
            picks = {"truth": p.output, "input": p.input, "wrong": tuple(map(tuple, wrong))}
            rows.append({
                "attempt_1": _grid_to_list(picks[attempt_1]),
                "attempt_2": _grid_to_list(picks[attempt_2]),
            })
        sub[t.task_id] = rows
    return sub


@pytest.fixture(scope="module")
def sample_tasks():
    if not SPLIT_DIRS["public_eval"].is_dir():
        pytest.skip("public_eval data not fetched")
    return list(iter_split("public_eval", limit=10))


# --- tests ------------------------------------------------------------------


def test_perfect_attempt_1_scores_one(tmp_path: Path, sample_tasks) -> None:
    sub = _synthetic_submission(sample_tasks, attempt_1="truth", attempt_2="wrong")
    p = tmp_path / "sub.json"
    p.write_text(json.dumps(sub))
    r = score_submission(p, "public_eval")

    n_outputs_in_sub = sum(len(t.test) for t in sample_tasks)
    assert r["num_correct"] == n_outputs_in_sub, "every attempt_1 matched truth"
    assert r["score"] == pytest.approx(n_outputs_in_sub / r["num_test_outputs"])
    # 10 of 120 tasks scored; remainder counted as missing → 0.
    assert len(r["missing_tasks"]) == r["num_tasks"] - len(sample_tasks)


def test_perfect_attempt_2_also_scores_one(tmp_path: Path, sample_tasks) -> None:
    sub = _synthetic_submission(sample_tasks, attempt_1="wrong", attempt_2="truth")
    p = tmp_path / "sub.json"
    p.write_text(json.dumps(sub))
    r = score_submission(p, "public_eval")
    n_outputs_in_sub = sum(len(t.test) for t in sample_tasks)
    assert r["num_correct"] == n_outputs_in_sub, "every attempt_2 matched truth"


def test_both_wrong_scores_zero(tmp_path: Path, sample_tasks) -> None:
    sub = _synthetic_submission(sample_tasks, attempt_1="wrong", attempt_2="input")
    p = tmp_path / "sub.json"
    p.write_text(json.dumps(sub))
    r = score_submission(p, "public_eval")
    # All sample tasks contribute 0; missing tasks also 0. Total must be 0.
    assert r["num_correct"] == 0
    assert r["score"] == 0.0


def test_missing_tasks_surface_as_warning(tmp_path: Path, sample_tasks) -> None:
    sub = _synthetic_submission(sample_tasks[:3], attempt_1="truth", attempt_2="truth")
    p = tmp_path / "sub.json"
    p.write_text(json.dumps(sub))
    r = score_submission(p, "public_eval")
    assert len(r["missing_tasks"]) == 120 - 3
    # Missing tasks are explicitly listed (not silently dropped).
    assert all(isinstance(tid, str) for tid in r["missing_tasks"])


def test_test_input_count_mismatch_is_flagged(tmp_path: Path, sample_tasks) -> None:
    # Find a multi-test task and submit only one attempt slot for it.
    multi = next(t for t in sample_tasks if len(t.test) > 1)
    sub = _synthetic_submission(sample_tasks, attempt_1="truth", attempt_2="truth")
    # Drop one slot to force a mismatch
    sub[multi.task_id] = sub[multi.task_id][:1]
    p = tmp_path / "sub.json"
    p.write_text(json.dumps(sub))
    r = score_submission(p, "public_eval")
    assert any(multi.task_id in msg for msg in r["mismatched_tasks"]), \
        f"expected {multi.task_id} in mismatched_tasks, got {r['mismatched_tasks']}"


def test_denominator_is_total_test_outputs_not_tasks(tmp_path: Path, sample_tasks) -> None:
    sub = _synthetic_submission(sample_tasks, attempt_1="truth", attempt_2="wrong")
    p = tmp_path / "sub.json"
    p.write_text(json.dumps(sub))
    r = score_submission(p, "public_eval")
    # The official metric averages over outputs, not over tasks. Pin that.
    assert r["num_test_outputs"] == 167, "public_eval has 167 test outputs"
    assert r["num_tasks"] == 120
