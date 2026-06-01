"""Pin the data-loader contract.

These tests assume `data/arc-agi-2/{training,evaluation}/` is present
(fetched via README / scripts/fetch_data.sh). They are NOT pure unit tests —
they verify our reading of the public dataset, which is itself part of the
contract. If these break, either the data layout changed or our reader did.
"""

from __future__ import annotations

import pytest

from src.eval.data import Grid, Pair, Task, iter_split, load_task, SPLIT_DIRS


# Fixed expectations for the public ARC-AGI-2 dataset as of 2026-06.
EXPECTED_TRAIN_TASKS = 1000
EXPECTED_EVAL_TASKS = 120
EXPECTED_EVAL_TEST_OUTPUTS = 167


@pytest.fixture(scope="module")
def eval_tasks() -> list[Task]:
    if not SPLIT_DIRS["public_eval"].is_dir():
        pytest.skip("public_eval data not fetched; see scripts/fetch_data.sh")
    return list(iter_split("public_eval"))


def test_public_eval_task_count(eval_tasks: list[Task]) -> None:
    assert len(eval_tasks) == EXPECTED_EVAL_TASKS


def test_public_eval_total_test_outputs(eval_tasks: list[Task]) -> None:
    # Per-output denominator: scoring is over test outputs, not tasks.
    total = sum(len(t.test) for t in eval_tasks)
    assert total == EXPECTED_EVAL_TEST_OUTPUTS


def test_grids_are_hashable(eval_tasks: list[Task]) -> None:
    # The metric and the from_submission scorer rely on Grid == Grid working
    # for tuple-of-tuple-of-int. If anyone ever changes Grid to lists, this
    # silently breaks scoring (lists hash differently and == still works, but
    # downstream content-hash voting and set membership don't).
    task = eval_tasks[0]
    {task.train[0].input, task.train[0].output}  # set construction requires hashable


def test_train_pairs_have_outputs(eval_tasks: list[Task]) -> None:
    # Demo pairs must always have outputs (that's what makes them demos).
    for task in eval_tasks[:5]:
        for pair in task.train:
            assert pair.output is not None, f"task {task.task_id} train pair missing output"


def test_test_pairs_have_outputs_on_public_eval(eval_tasks: list[Task]) -> None:
    # The public eval ships with truth so we can score locally. The Kaggle
    # hidden test set is a different file we don't ship.
    for task in eval_tasks[:5]:
        for pair in task.test:
            assert pair.output is not None, (
                f"public_eval task {task.task_id} unexpectedly missing test output"
            )


def test_grid_values_in_arc_palette(eval_tasks: list[Task]) -> None:
    # ARC grids use 0..9 only.
    for task in eval_tasks[:3]:
        for pair in task.train + task.test:
            for grid in (pair.input, pair.output):
                if grid is None:
                    continue
                for row in grid:
                    for v in row:
                        assert 0 <= v <= 9, f"out-of-palette value {v} in {task.task_id}"


def test_iter_split_is_deterministic(eval_tasks: list[Task]) -> None:
    again = list(iter_split("public_eval"))
    assert [t.task_id for t in again] == [t.task_id for t in eval_tasks]


def test_iter_split_limit_applies() -> None:
    if not SPLIT_DIRS["public_eval"].is_dir():
        pytest.skip("public_eval data not fetched")
    limited = list(iter_split("public_eval", limit=7))
    assert len(limited) == 7


def test_iter_split_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="unknown split"):
        list(iter_split("not_a_real_split"))


def test_load_task_matches_iter_split(eval_tasks: list[Task]) -> None:
    first = eval_tasks[0]
    reloaded = load_task(SPLIT_DIRS["public_eval"] / f"{first.task_id}.json")
    assert reloaded == first


def test_some_tasks_have_multiple_test_inputs(eval_tasks: list[Task]) -> None:
    # 167 outputs across 120 tasks means at least 47 tasks have >1 test input.
    multi = [t for t in eval_tasks if len(t.test) > 1]
    assert len(multi) >= 40, f"expected >=40 multi-test tasks, got {len(multi)}"
