"""Synthetic concurrent miner workload, with no paid or external tool calls."""

import asyncio
import os

from harnyx_miner_sdk.decorators import entrypoint

loads = 1


@entrypoint("isolated")
async def shared(request: dict[str, object]) -> dict[str, int]:
    if request.get("block"):
        while True:
            pass
    await asyncio.sleep(float(request.get("delay", 0.03)))
    return {"pid": os.getpid(), "loads": loads}
