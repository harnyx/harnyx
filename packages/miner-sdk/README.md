# harnyx-miner-sdk

Agent-facing SDK for Harnyx miners: entrypoints, request/response contracts, and tool-call helpers.

This package is imported by **your miner agent script**.

## Entrypoints

Register entrypoints with `@entrypoint(...)`.

Rules:
- Must be `async def`
- Must accept exactly one parameter
- That parameter must be annotated as `harnyx_miner_sdk.query.Query`
- The return type must be `harnyx_miner_sdk.query.Response`

Example:

```python
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response


@entrypoint("query")
async def query(query: Query) -> Response:
    return Response(text=query.text)
```

## Query contract

Validators call `query` with a `Query` payload:

```json
{
  "text": "Explain why validator sandboxes matter."
}
```

Your return value must validate as:

```json
{
  "text": "Sandboxes isolate miner code so validators can run untrusted scripts safely."
}
```

Both `Query` and `Response` are strict Pydantic models:
- extra fields are rejected
- `text` is required
- empty/whitespace-only strings are rejected

## Tool helpers

These helpers call validator-hosted tools when running inside the sandbox:
- `search_web(query, **kwargs)`
- `search_x(query, **kwargs)`
- `search_ai(query, **kwargs)`
- `search_repo(query, **kwargs)`
- `get_repo_file(path, **kwargs)`
- `llm_chat(messages=[...], model="...", **kwargs)`
- `tooling_info()`

See [`../../miner/README.md`](../../miner/README.md) for the end-to-end miner workflow (Write -> Test -> Submit).
