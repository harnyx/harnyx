from __future__ import annotations

import asyncio

import pytest

from harnyx_commons.errors import ConcurrencyLimitError
from harnyx_commons.tools.token_semaphore import TokenSemaphore


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


@pytest.mark.anyio("asyncio")
async def test_token_semaphore_async_waits_for_released_permit() -> None:
    semaphore = TokenSemaphore(max_parallel_calls=1)
    acquired = asyncio.Event()

    async def wait_for_permit() -> None:
        await semaphore.acquire_async("token")
        acquired.set()

    semaphore.acquire("token")
    waiter = asyncio.create_task(wait_for_permit())
    await asyncio.sleep(0.05)
    assert not acquired.is_set()

    semaphore.release("token")
    await asyncio.wait_for(acquired.wait(), timeout=1.0)
    semaphore.release("token")
    await waiter


def test_token_semaphore_release_without_acquire_errors() -> None:
    semaphore = TokenSemaphore(max_parallel_calls=1)

    with pytest.raises(RuntimeError):
        semaphore.release("token")
