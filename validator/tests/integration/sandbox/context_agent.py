from __future__ import annotations

import json

from harnyx_miner_sdk.context import ContextSnapshot
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response


@entrypoint("query")
async def query(query: Query, context: ContextSnapshot) -> Response:
    del query
    return Response(
        text=json.dumps(
            {
                "cost_budget": context.cost_budget.model_dump(mode="json"),
                "time_budget": context.time_budget.model_dump(mode="json"),
            },
            sort_keys=True,
        )
    )
