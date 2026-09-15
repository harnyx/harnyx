from uuid import UUID

import pytest

from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer
from harnyx_validator.application.miner_task_time_limit import assigned_miner_task_execution_time_limit_seconds


def _task(*, task_id: int, fast: bool) -> MinerTask:
    return MinerTask(
        task_id=UUID(int=task_id),
        query=Query(text="time-limit policy", fast=fast),
        reference_answer=ReferenceAnswer(text="reference"),
    )


@pytest.mark.parametrize(
    ("batch_id", "task_id", "expected_limit_seconds"),
    [
        (1, 1, 210.0),
        (1, 2, 180.0),
        (1, 3, 150.0),
    ],
)
def test_fast_limit_mapping_has_stable_versioned_golden_vectors(
    batch_id: int,
    task_id: int,
    expected_limit_seconds: float,
) -> None:
    assert (
        assigned_miner_task_execution_time_limit_seconds(
            batch_id=UUID(int=batch_id),
            task=_task(task_id=task_id, fast=True),
        )
        == expected_limit_seconds
    )


def test_fast_limit_is_stable_for_the_same_batch_task_occurrence() -> None:
    batch_id = UUID(int=7)
    task = _task(task_id=3, fast=True)

    first = assigned_miner_task_execution_time_limit_seconds(batch_id=batch_id, task=task)
    second = assigned_miner_task_execution_time_limit_seconds(batch_id=batch_id, task=task)

    assert first == second == 180.0


def test_reused_fast_task_can_select_another_limit_in_a_new_batch() -> None:
    task = _task(task_id=2, fast=True)

    first_batch_limit = assigned_miner_task_execution_time_limit_seconds(batch_id=UUID(int=1), task=task)
    second_batch_limit = assigned_miner_task_execution_time_limit_seconds(batch_id=UUID(int=2), task=task)

    assert first_batch_limit == 180.0
    assert second_batch_limit == 150.0


@pytest.mark.parametrize(
    ("batch_id", "task_id", "expected_limit_seconds"),
    [
        (1, 6, 180.0),
        (1, 4, 210.0),
        (1, 1, 240.0),
    ],
)
def test_ordinary_limit_mapping_has_stable_versioned_golden_vectors(
    batch_id: int,
    task_id: int,
    expected_limit_seconds: float,
) -> None:
    assert (
        assigned_miner_task_execution_time_limit_seconds(
            batch_id=UUID(int=batch_id),
            task=_task(task_id=task_id, fast=False),
        )
        == expected_limit_seconds
    )


def test_ordinary_limit_is_stable_for_the_same_batch_task_occurrence() -> None:
    batch_id = UUID(int=7)
    task = _task(task_id=3, fast=False)

    first = assigned_miner_task_execution_time_limit_seconds(batch_id=batch_id, task=task)
    second = assigned_miner_task_execution_time_limit_seconds(batch_id=batch_id, task=task)

    assert first == second == 180.0


def test_reused_ordinary_task_can_select_another_limit_in_a_new_batch() -> None:
    task = _task(task_id=1, fast=False)

    first_batch_limit = assigned_miner_task_execution_time_limit_seconds(batch_id=UUID(int=1), task=task)
    second_batch_limit = assigned_miner_task_execution_time_limit_seconds(batch_id=UUID(int=2), task=task)

    assert first_batch_limit == 240.0
    assert second_batch_limit == 210.0
