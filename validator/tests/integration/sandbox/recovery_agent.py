"""Disposable workload that exposes when a recovery test's query has started."""

import asyncio

from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response


@entrypoint("query")
async def query(request: Query) -> Response:
    if request.text == "unavailable":
        print("RECOVERY_QUERY_STARTED", flush=True)
        await asyncio.sleep(60)
    return Response(text="recovered")
