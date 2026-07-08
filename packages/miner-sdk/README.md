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

or, when your answer needs receipt-backed support:

```json
{
  "text": "Sandboxes isolate miner code so validators can run untrusted scripts safely.",
  "citations": [
    {"receipt_id": "receipt-123", "result_id": "result-abc"}
  ]
}
```

Both `Query` and `Response` are strict Pydantic models:
- extra fields are rejected
- `text` is required
- empty/whitespace-only strings are rejected
- `citations` is optional
- `text` may contain at most 80,000 characters
- `citations`, when present, may contain at most 200 receipt refs
- each citation must include `receipt_id` and `result_id`
- citation refs may also include `slices=[CitationSlice(start=..., end=...)]`; refs without slices use the entire referenced result text
- citations may materialize at most 400 evidence segments and 120,000 source-text characters per answer

For practical scoring, treat `citations` as required for answers that make non-obvious factual claims or depend on search/tool evidence. A response without citations only makes sense when the answer is obvious enough that no external support is reasonably needed. Facts presented without citations can be dismissed by the judge when they are load-bearing to the answer.

When citations are present, validators hydrate them into shared citations shaped like
`{url, title?, note?}` before scoring. Hydrated citation notes are materialized by the validator from the referenced tool result's `note` text. A ref without slices materializes the full result note. A ref with slices materializes only those offsets. Miner-authored citation text is not accepted as evidence.

## Receipts and citations

Hosted tool calls return two layers of identifiers:

- `receipt_id`: the tool call itself
- `result_id`: a specific referenceable result from that tool call

Your `Response.citations` must point at the exact result(s) that support your answer:

```python
from harnyx_miner_sdk.api import search_web
from harnyx_miner_sdk.query import CitationRef, Query, Response


async def query(query: Query) -> Response:
    search = await search_web(query.text, provider="parallel", num=5)
    top_result = search.results[0]
    return Response(
        text="...",
        citations=[
            CitationRef(
                receipt_id=search.receipt_id,
                result_id=top_result.result_id,
            )
        ],
    )
```

How to extract them:

- call a hosted tool such as `search_web(...)`
- read the tool-call envelope `search.receipt_id`
- choose the specific supporting result from `search.results`
- read that result's `result_id`
- return `CitationRef(receipt_id=..., result_id=...)` for whole-result evidence
- return `CitationRef(receipt_id=..., result_id=..., slices=[CitationSlice(start=0, end=180)])` when a narrower excerpt is enough

Targeted slice example:

```python
from harnyx_miner_sdk.query import CitationRef, CitationSlice

CitationRef(
    receipt_id=search.receipt_id,
    result_id=top_result.result_id,
    slices=[CitationSlice(start=0, end=180)],
)
```

The relevant SDK fields are:

```python
search.receipt_id
search.results[i].result_id
search.results[i].url
search.results[i].title
search.results[i].note
```

Use the citation only when that result actually supports a material claim in your response. Prefer results whose `note` text already contains the factoid or excerpt your answer depends on. Whole-result citations are valid; targeted slices are useful when a large result contains both relevant and irrelevant text. Irrelevant citations do not help, and citation spam makes the response worse.

## Tool helpers

These helpers call validator-hosted tools when running inside the sandbox:
- `search_web(query, provider="parallel" | "desearch", timeout=..., **kwargs)`
- `search_ai(query, provider="parallel" | "desearch", timeout=..., **kwargs)`
- `fetch_page(url, provider="parallel" | "desearch", timeout=...)`
- `llm_chat(provider="chutes" | "openrouter" | "ai_gateway", messages=[...], model="<provider-specific model id>", timeout=..., temperature=0.0, thinking={"enabled": True}, provider_extra=...)`
- `embed_text(texts, input_type="query" | "document", provider="chutes" | "openrouter", model="<provider-specific embedding model id>", instruction=..., dimensions=..., timeout=...)`
- `tooling_info(timeout=...)`
- `test_tool(message, timeout=...)`

Every hosted tool helper accepts an optional positive finite `timeout` in seconds. For provider-backed tools other than `llm_chat`, the tool host bounds the complete provider-backed invocation, including host-owned retries/backoff, and raises a tool invocation error if the deadline expires. `llm_chat` makes one provider attempt per SDK call; retry loops belong in miner script code when desired. `tooling_info` and `test_tool` accept the same parameter for interface consistency, but they complete locally and do not perform provider deadline enforcement.

`llm_chat` model ids are provider-specific. Use `tooling_info().response["allowed_llm_provider_models"][provider]` as the runtime source of truth and pass the selected provider's model id exactly.

`embed_text` model ids are provider-specific too. Use `tooling_info().response["allowed_embedding_provider_models"][provider]` as the runtime source of truth. The current miner-facing embedding model ids are `Qwen/Qwen3-Embedding-8B-TEE` on `chutes` and `qwen/qwen3-embedding-8b` on `openrouter`, with pricing exposed under `tooling_info().response["pricing"]["embed_text"]["provider_models"]`.

Use `input_type="query"` for query or instruction-style embeddings and `input_type="document"` for document embeddings. Query embeddings use Qwen's retrieval instruction by default and accept an optional `instruction` override. Document embeddings are sent as raw text and reject `instruction`.

```python
from harnyx_miner_sdk.api import embed_text

query_embedding = await embed_text(
    query.text,
    provider="openrouter",
    model="qwen/qwen3-embedding-8b",
    input_type="query",
)
vector = query_embedding.response.data[0].embedding

document_embeddings = await embed_text(
    ["First passage text.", "Second passage text."],
    provider="openrouter",
    model="qwen/qwen3-embedding-8b",
    input_type="document",
)
```

Embedding outputs are ordinary tool responses for miner code. They are not citation sources, so they do not replace `search_web`, `search_ai`, or `fetch_page` evidence when an answer needs citations.

`provider_extra` is strict and selected by `provider`. Use it only for selected-provider-specific request additions that are not already common `llm_chat` parameters. OpenRouter supports provider selection:

```python
await llm_chat(
    provider="openrouter",
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Reply with only ok."}],
    provider_extra={"provider": {"only": ["cerebras"]}},
)
```

OpenRouter also accepts an optional `provider.allow_fallbacks` boolean. Omit it to use OpenRouter's default fallback behavior; set it only when your miner needs to explicitly choose whether OpenRouter may fall back to another hosted provider after the selected provider fails. You can pass it with `provider.only`, or by itself as `provider_extra={"provider": {"allow_fallbacks": False}}`.

AI Gateway accepts Vercel's top-level `provider` shorthand or the `providerOptions.gateway` form. Use these for request-level upstream provider selection:

```python
await llm_chat(
    provider="ai_gateway",
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Reply with only ok."}],
    provider_extra={"provider": {"only": ["cerebras"]}},
)

await llm_chat(
    provider="ai_gateway",
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Reply with only ok."}],
    provider_extra={"providerOptions": {"gateway": {"only": ["cerebras"]}}},
)
```

Do not pass `provider_extra={"provider": "cerebras"}`. The SDK/runtime rejects the raw string form.

AI Gateway model ids currently allowed by the tool contract are `zai/glm-5.2-fast`, `openai/gpt-oss-20b`, `zai/glm-4.7`, `google/gemma-4-31b-it`, `openai/gpt-oss-120b`, `alibaba/qwen3.7-plus`, `minimax/minimax-m2.7`, and `zai/glm-4.7-flash`. Use `tooling_info().response["pricing"]["llm_chat"]["provider_models"]["ai_gateway"]` for representative static rates; actual AI Gateway returned cost wins when present.

Do not put common behavior in `provider_extra`. For example, reasoning controls belong in `thinking` even when a provider's raw API spells them differently. Chutes raw reasoning options are handled by `thinking`, not `provider_extra`. Other OpenRouter provider-preference fields such as `order`, `require_parameters`, `ignore`, `quantizations`, `sort`, and `max_price` are not supported here.

`llm_chat` accepts a typed `thinking` option:

| Provider | Model | `enabled=True` / `enabled=False` | `effort` | `budget` |
|----------|-------|----------------------------------|----------|----------|
| `openrouter` | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` | Supported via OpenRouter `reasoning.enabled` / `reasoning.effort="none"` | Supported via OpenRouter `reasoning.effort` | Supported via OpenRouter `reasoning.max_tokens` |
| `openrouter` | `deepseek/deepseek-v3.2`, `z-ai/glm-5`, `qwen/qwen3.6-27b`, `google/gemma-4-31b-it` | Supported via OpenRouter `reasoning.enabled` / `reasoning.effort="none"` | Supported via OpenRouter `reasoning.effort` | Supported via OpenRouter `reasoning.max_tokens` |
| `ai_gateway` | All allowed AI Gateway models | Supported via AI Gateway `reasoning.enabled` / `reasoning.effort="none"` | Supported via AI Gateway `reasoning.effort` | Supported via AI Gateway `reasoning.max_tokens` |
| `chutes` | `deepseek-ai/DeepSeek-V3.2-TEE` | Supported via `chat_template_kwargs.thinking` | Unsupported for Chutes; not serialized | Unsupported for Chutes; not serialized |
| `chutes` | `zai-org/GLM-5-TEE` | Supported via `chat_template_kwargs.enable_thinking` | Unsupported for Chutes; not serialized | Unsupported for Chutes; not serialized |
| `chutes` | `Qwen/Qwen3.6-27B-TEE`, `google/gemma-4-31B-turbo-TEE` | Supported via `chat_template_kwargs.enable_thinking` | Unsupported for Chutes; not serialized | Unsupported for Chutes; not serialized |

```python
await llm_chat(
    provider="chutes",
    model="deepseek-ai/DeepSeek-V3.2-TEE",
    messages=[{"role": "user", "content": "Solve 17 * 23."}],
    temperature=0.0,
    thinking={"enabled": True},
)

await llm_chat(
    provider="chutes",
    model="zai-org/GLM-5-TEE",
    messages=[{"role": "user", "content": "Reply with only ok."}],
    temperature=0.0,
    thinking={"enabled": False},
)

await llm_chat(
    provider="openrouter",
    model="deepseek/deepseek-v3.2",
    messages=[{"role": "user", "content": "Reply with only ok."}],
    temperature=0.0,
    thinking={"effort": "low"},
)
```

Omit `thinking` to use provider defaults. `effort` accepts `"low"`, `"medium"`, or `"high"` and `budget` must be a positive integer. OpenRouter-selected and AI Gateway-selected models honor those fields through provider reasoning controls. Do not send `effort` and `budget` together; that is a validation error. Provider support is best effort, so unsupported level/budget hints are not serialized into raw provider-body fields.

See [`../../miner/README.md`](../../miner/README.md) for the end-to-end miner workflow (Write -> Test -> Submit).
