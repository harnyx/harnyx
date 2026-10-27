"""Disposable workload that exposes when a recovery test's query has started."""

import asyncio

from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response


@entrypoint("query")
async def query(request: Query) -> Response:
    if request.text == "bad_error":

        class BadMeta(type):
            @property
            def __name__(cls):
                raise RuntimeError("unsafe metaclass name")

        class BadError(Exception, metaclass=BadMeta):
            def __str__(self):
                raise ValueError("formatter failed")

        class BadName(str):
            def __getitem__(self, key):
                raise RuntimeError("unsafe name formatting")

        vars(type)["__name__"].__set__(BadError, BadName("BadError"))
        raise BadError()
    if request.text in ("nested_exit", "detached_exit", "callback_exit"):

        def fatal():
            raise SystemExit(23)

        async def child():
            fatal()

        if request.text == "nested_exit":
            await asyncio.gather(child())
        elif request.text == "detached_exit":
            asyncio.create_task(child())
            await asyncio.sleep(1)
        else:
            asyncio.get_running_loop().call_soon(fatal)
            await asyncio.sleep(1)
    if request.text == "unavailable":
        print("RECOVERY_QUERY_STARTED", flush=True)
        await asyncio.sleep(60)
    return Response(text="recovered")
