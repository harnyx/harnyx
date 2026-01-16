# caster-miner-sdk

Agent-facing SDK for Caster miners: entrypoints, request/response contracts, and tool-call helpers.

This package is imported by **your miner agent script**.

## Entrypoints

Register entrypoints with `@entrypoint(...)`.

Rules:
- Must be `async def`
- Must accept exactly one parameter named `request`

Example:

```python
from caster_miner_sdk.decorators import entrypoint
from caster_miner_sdk.criterion_evaluation import CriterionEvaluationRequest


@entrypoint("evaluate_criterion")
async def evaluate_criterion(request: object) -> dict[str, object]:
    payload = CriterionEvaluationRequest.model_validate(request)
    # ...
    return {"verdict": 1, "justification": "…", "citations": []}
```

## Criterion evaluation contract

Validators call `evaluate_criterion` with a `CriterionEvaluationRequest` payload:
- `claim_text: str`
- `rubric_title: str`
- `rubric_description: str`
- `verdict_options: list[{value:int, description:str}]`

Your return value is validated by validators and should look like:

```json
{
  "verdict": 1,
  "justification": "…",
  "citations": [
    {
      "url": "https://example.com",
      "note": "evidence summary",
      "receipt_id": "tool-receipt-id",
      "result_id": "search-result-id"
    }
  ]
}
```

`receipt_id` comes from the tool call response, and `result_id` comes from the search result you are citing.

## Tool helpers

These helpers call validator-hosted tools (when running inside the sandbox):
- `search_web(query, **kwargs)`
- `search_x(query, **kwargs)`
- `llm_chat(messages=[...], model="...", **kwargs)`

See `public/miner/README.md` for end-to-end miner workflow (write → test → submit).
