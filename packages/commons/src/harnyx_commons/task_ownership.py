from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextlib import suppress
from typing import Any, TypeVar

_T = TypeVar("_T")


async def wait_for_owned_task(task: asyncio.Task[_T]) -> asyncio.CancelledError | None:
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                break
            cancellation = exc
        except BaseException:
            # The owner reads the settled task result after this wait.
            return cancellation
    return cancellation


async def await_owned_task(task: asyncio.Task[_T]) -> _T:
    cancellation = await wait_for_owned_task(task)
    if cancellation is not None:
        with suppress(BaseException):
            task.result()
        raise cancellation
    return task.result()


async def run_owned_task_until_deadline(
    work: Coroutine[Any, Any, None],
    *,
    delay_seconds: float,
    on_deadline: Callable[[], Awaitable[None]],
) -> None:
    """Own one work task and its deadline callback until both have settled."""
    if delay_seconds <= 0:
        work.close()
        raise ValueError("delay_seconds must be positive")

    work_task = asyncio.create_task(work)

    async def run_deadline() -> None:
        await _wait_for_delay(delay_seconds)
        await on_deadline()

    deadline_task = asyncio.create_task(run_deadline())
    tasks = (work_task, deadline_task)
    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        await _cancel_and_join_owned_tasks(tasks)
        raise

    if work_task in done:
        deadline_task.cancel()
    else:
        work_task.cancel()
    await _join_owned_tasks(tasks)


async def _wait_for_delay(delay_seconds: float) -> None:
    """Wait on a loop timer without inheriting worker-specific sleep patches."""
    loop = asyncio.get_running_loop()
    elapsed = asyncio.Event()
    handle = loop.call_later(delay_seconds, elapsed.set)
    try:
        await elapsed.wait()
    finally:
        handle.cancel()


async def _cancel_and_join_owned_tasks(tasks: Sequence[asyncio.Task[None]]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    for task in tasks:
        await wait_for_owned_task(task)
    for task in tasks:
        with suppress(BaseException):
            task.result()


async def _join_owned_tasks(tasks: Sequence[asyncio.Task[None]]) -> None:
    cancellation: asyncio.CancelledError | None = None
    for task in tasks:
        task_cancellation = await wait_for_owned_task(task)
        cancellation = task_cancellation or cancellation
    failures: list[BaseException] = []
    for task in tasks:
        try:
            task.result()
        except asyncio.CancelledError:
            continue
        except BaseException as exc:
            failures.append(exc)
    if cancellation is not None:
        raise cancellation
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("owned work and deadline tasks failed", failures)


__all__ = [
    "await_owned_task",
    "run_owned_task_until_deadline",
    "wait_for_owned_task",
]
