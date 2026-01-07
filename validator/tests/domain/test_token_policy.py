from __future__ import annotations

import pytest

from caster_commons.errors import ConcurrencyLimitError
from caster_commons.tools.token_semaphore import TokenSemaphore


def test_token_semaphore_allows_within_limit() -> None:
    semaphore = TokenSemaphore(max_parallel_calls=2)

    semaphore.acquire("token")
    semaphore.acquire("token")

    assert semaphore.in_flight("token") == 2

    semaphore.release("token")
    semaphore.release("token")

    assert semaphore.in_flight("token") == 0


def test_token_semaphore_blocks_excess_parallelism() -> None:
    semaphore = TokenSemaphore(max_parallel_calls=1)
    semaphore.acquire("token")

    with pytest.raises(ConcurrencyLimitError):
        semaphore.acquire("token")

    semaphore.release("token")


def test_token_semaphore_release_without_acquire_errors() -> None:
    semaphore = TokenSemaphore(max_parallel_calls=1)

    with pytest.raises(RuntimeError):
        semaphore.release("token")
