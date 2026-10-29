from __future__ import annotations

import asyncio
import socket
from typing import Any, cast

import pytest

from harnyx_validator.infrastructure.http import local_tool_host

pytestmark = pytest.mark.anyio("asyncio")


class _BlockingServer:
    created: list[_BlockingServer] = []

    def __init__(self, config: object) -> None:
        self.config = config
        self.started = False
        self.should_exit = False
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.sockets: list[socket.socket] = []
        self.__class__.created.append(self)

    async def serve(self, sockets: list[socket.socket] | None = None) -> None:
        assert sockets is not None
        self.sockets = sockets
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


async def test_start_local_tool_host_cleans_up_when_cancelled_during_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _BlockingServer.created.clear()
    monkeypatch.setattr(local_tool_host, "Server", _BlockingServer)

    start_task = asyncio.create_task(
        local_tool_host.start_local_tool_host(
            tool_executor=cast(Any, object()),
            tool_concurrency_limiter=cast(Any, object()),
        )
    )

    while not _BlockingServer.created:
        await asyncio.sleep(0)
    server = _BlockingServer.created[0]
    await asyncio.wait_for(server.entered.wait(), timeout=1.0)

    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start_task

    assert server.should_exit is True
    await asyncio.wait_for(server.cancelled.wait(), timeout=1.0)
    assert all(listener.fileno() == -1 for listener in server.sockets)


async def test_concurrent_hosts_hold_distinct_ports_until_startup_and_release_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _BlockingServer.created.clear()
    monkeypatch.setattr(local_tool_host, "Server", _BlockingServer)
    tasks = [asyncio.create_task(local_tool_host.start_local_tool_host(
        tool_executor=cast(Any, object()), tool_concurrency_limiter=cast(Any, object()),
        host="127.0.0.1")) for _ in range(8)]
    while len(_BlockingServer.created) < 8:
        await asyncio.sleep(0)
    await asyncio.gather(*(server.entered.wait() for server in _BlockingServer.created))
    listeners = [server.sockets[0] for server in _BlockingServer.created]
    assert len({listener.getsockname() for listener in listeners}) == 8
    for listener in listeners:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as contender:
            with pytest.raises(OSError):
                contender.bind(listener.getsockname())
    for task in tasks:
        task.cancel()
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in outcomes)
    assert all(listener.fileno() == -1 for listener in listeners)
