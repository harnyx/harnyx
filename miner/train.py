from __future__ import annotations

from pathlib import Path

from harnyx_miner_sdk.context import ContextSnapshot
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response


@entrypoint("query")
async def query(query: Query, _context: ContextSnapshot) -> Response:
    return Response(text=f"default miner agent: {query.text}")


if __name__ == "__main__":
    from prepare import run_experiment

    raise SystemExit(run_experiment(Path(__file__)))
