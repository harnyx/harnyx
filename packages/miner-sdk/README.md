# harnyx-miner-sdk

Agent-facing SDK for Harnyx miners: entrypoints, request/response contracts, and tool-call helpers.

This package is imported by **your miner agent script**.

## Generated Python execution

SDK `0.1.9` can execute generated Python inside the miner sandbox:

```python
from harnyx_miner_sdk.safe_exec import safe_exec

average = safe_exec(
    """
import statistics
result = statistics.mean(values)
""",
    {"values": [2, 4, 6]},
)
```

Miner scripts must use this exact import and call the protected binding directly.
Aliasing, rebinding, shadowing, or carrying `safe_exec` as a value is rejected by
the Platform upload policy. Direct calls named `eval`, `exec`, or `compile` are
also rejected.

`safe_exec(code, variables=None)` accepts multi-statement Python and an exact
built-in `dict` of JSON-compatible values. The code runs with normal Python
builtins and imports in a fresh namespace inside the existing miner sandbox.
Supplied variables are copied into that namespace, code must assign `result`,
and the returned result must be JSON-compatible. Inputs and results are detached
by JSON serialization. The namespace is not a process-isolation boundary:
generated code can explicitly inspect other same-process state, including caller
frames. The existing miner sandbox is the security boundary. Generated-code
exceptions propagate normally; sandbox resource and network limits remain the
responsibility of the sandbox and execution lifecycle.

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
  "text": "Sandboxes isolate miner code so validators can run untrusted scripts safely [[1]].",
  "citations": [
    {"receipt_id": "receipt-123", "result_id": "result-abc"}
  ]
}
```

For structured output, pass a self-contained JSON Schema Draft 2020-12 object in
`output_schema`:

```json
{
  "text": "Summarize the result by region.",
  "output_schema": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
      "region": {"type": "string"},
      "finding": {"type": "string"}
    },
    "required": ["region", "finding"],
    "additionalProperties": false
  }
}
```

Return the JSON value directly in `output`, not as encoded JSON inside `text`:

```json
{
  "output": {"region": "North", "finding": "Demand increased."},
  "citations": [
    {"receipt_id": "receipt-123", "result_id": "result-abc"}
  ]
}
```

`output_schema` is an entrypoint contract. It is not an `llm_chat` request
option. A missing or `null` schema requires `Response.text` for a plain-text
answer; a present schema, including `{}`, requires `Response.output`. Return
exactly one of `text` or `output`, never both. Top-level `null` does not count as an answer,
although nested nulls are valid when the schema permits them. The schema applies
only to `output`; citation refs remain Harnyx-owned response siblings and are
hydrated after output validation.

Schemas must use Draft 2020-12 when `$schema` is declared, must resolve entirely
within the submitted schema, and cannot use external `$ref` or `$dynamicRef`
targets. The compact JSON encoding of the schema and of the returned structured
value is limited to 80,000 characters each. Invalid structured output is
rejected and never falls back to `Response.text`.

Both `Query` and `Response` are strict Pydantic models:
- extra fields are rejected
- `Query.text` is required and empty/whitespace-only strings are rejected
- `Response` requires exactly one non-null answer field for the query mode
- `citations` is optional
- plain-text response `text` may contain at most 80,000 characters
- structured schemas and outputs may contain at most 80,000 compact JSON characters each
- `citations`, when present, may contain at most 200 receipt refs
- each citation must include `receipt_id` and `result_id`
- citation refs may also include `slices=[CitationSlice(start=..., end=...)]`; refs without slices use the entire referenced result text
- citations may materialize at most 400 evidence segments and 120,000 source-text characters per answer

For practical scoring, treat `citations` as required for answers that make non-obvious factual claims or depend on search/tool evidence. A response without citations only makes sense when the answer is obvious enough that no external support is reasonably needed. Facts presented without citations can be dismissed by the judge when they are load-bearing to the answer.

When citations are present, validators hydrate them into shared citations shaped like
`{url, title?, note?}` before scoring. Hydrated citation notes are materialized by the validator from the referenced tool result's `note` text. A ref without slices materializes the full result note. A ref with slices materializes only those offsets. Miner-authored citation text is not accepted as evidence.

For prose answers, `[[n]]` is an exact one-based pointer to `Response.citations[n - 1]`; ordinary `[n]` text is not a citation. Unless the query explicitly rejects citations, give every material researched claim a valid pointer to the evidence that supports it. Submission order is authoritative: duplicate refs occupy separate positions, and validators never deduplicate, renumber, or remap them. Miners submit only non-null `CitationRef` items. If a submitted position cannot be resolved or hydrated, the public hydrated response contains `null` at that position, that position provides no factual support, and later pointers do not move. A missing, unresolved, out-of-range, irrelevant, or mismatched pointer is a judge-visible quality defect, not an invalid response or automatic loss; malformed citation payloads and the documented hard limits still invalidate the response.

Structured output does not require inline pointers by default. Add them only when the query or a prose-capable field description explicitly requires citations, and only in that prose-capable field. Do not add citation syntax to atomic integer, number, boolean, enum, identifier, date-token, or similar fields.

When the query does not request a conflicting form, prefer a clear, self-contained, reader-facing synthesis and use Markdown only when it reduces reader effort. An explicit requested form such as XML or a terse answer overrides that default. Correctness, requested coverage, instruction following, evidence support, and calibrated uncertainty take priority over presentation.

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
        text=f"The researched result is {top_result.title} [[1]].",
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
- `search_web(query, provider="parallel" | "desearch" | "firecrawl" | "exa" | "tavily", timeout=..., provider_extra=..., **kwargs)`
- `fetch_page(url, provider="parallel" | "desearch" | "firecrawl" | "exa" | "tavily", timeout=..., provider_extra=...)`

`provider_extra` is strictly validated for the selected provider and operation. It exposes retrieval and extraction controls only; provider answers, deep research, autonomous reasoning, and generated-output controls are rejected.

| Provider | `search_web.provider_extra` | `fetch_page.provider_extra` |
|---|---|---|
| `desearch` | `start` | `format`, `js`, `wait` |
| `parallel` | `mode` (`turbo`, `basic`, or `advanced`), `max_chars_total`, `source_policy`, `fetch_policy`, `excerpt_settings`, `location` | `objective`, `max_chars_total`, `fetch_policy`, `excerpt_settings`, `full_content` |
| `firecrawl` | `categories`, domain filters, `tbs`, `location`, `country`, invalid-URL and privacy controls | `formats` (`markdown` and/or `rawHtml`), main-content/tag/cache/wait/mobile/PDF/location/image/ad/proxy/cache-retention controls |
| `exa` | `type` (`auto`, `instant`, or `fast`), category, domain/date/location/moderation filters | `text` (`true` or text options), `max_age_hours`, `livecrawl_timeout` |
| `tavily` | `search_depth` (`basic`, `fast`, `advanced`, or `ultra-fast`), chunks, topic/time/date/domain/country/exact/safe controls | `query`, `chunks_per_source`, `extract_depth`, `format` |

Common `search_queries`, `num`, and `timeout` remain top-level fields and are rejected if duplicated in `provider_extra`. Unknown fields and documented incompatible combinations fail before the tool proxy is called.

Firecrawl `fetch_page` uses Firecrawl's provider-native plural `formats` list. The default is `["markdown"]`. When multiple formats are requested, one result is returned for each format in the same order:

```python
page = await fetch_page(
    "https://example.com",
    provider="firecrawl",
    provider_extra={"formats": ["markdown", "rawHtml"]},
)
markdown, raw_html = page.response.data
```

Use `provider_extra={"formats": ["rawHtml"]}` when only raw HTML is needed. If Firecrawl omits or returns blank content for any requested format, the fetch fails instead of substituting another representation.

- `llm_chat(provider="chutes" | "openrouter" | "ai_gateway", messages=[...], model="<provider-specific model id>", timeout=..., temperature=0.0, thinking={"enabled": True}, provider_extra=...)`
- `embed_text(texts, input_type="query" | "document", provider="chutes" | "openrouter", model="<provider-specific embedding model id>", instruction=..., dimensions=..., provider_extra=..., timeout=...)`
- `tooling_info(timeout=...)`
- `test_tool(message, timeout=...)`

Every hosted tool helper accepts an optional positive finite `timeout` in seconds. For provider-backed tools other than `llm_chat`, the tool host bounds the complete provider-backed invocation, including host-owned retries/backoff, and raises a tool invocation error if the deadline expires. `llm_chat` makes one provider attempt per SDK call; retry loops belong in miner script code when desired. `tooling_info` and `test_tool` accept the same parameter for interface consistency, but they complete locally and do not perform provider deadline enforcement.

Firecrawl is an ordinary-web provider: it supports `search_web` and `fetch_page`.

`llm_chat` model ids are provider-specific. Use `tooling_info().response["allowed_llm_provider_models"][provider]` as the runtime source of truth and pass the selected provider's model id exactly.

### Function tool calls

Pass function definitions through `tools`. Functions may include `description`, recursive JSON Schema `parameters`, and `strict`. Use `tool_choice="none"`, `"auto"`, `"required"`, or name one declared function. Set `parallel_tool_calls` when you need to control whether the model may request more than one function in a turn.

The model returns flattened tool calls with `id`, `type`, `name`, and JSON-string `arguments`. Run those functions in your miner, then make a second `llm_chat` call with the assistant message followed by one linked tool-result message per call. `to_input_message()` preserves the assistant's tool calls and any ordered `reasoning_details` needed for replay.

```python
import json

from harnyx_miner_sdk.api import llm_chat

question = {"role": "user", "content": "What is the weather in Paris?"}
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Return the current weather for one city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

first = await llm_chat(
    provider="openrouter",
    model="openai/gpt-oss-20b",
    messages=[question],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
    parallel_tool_calls=False,
)
assistant = first.llm.choices[0].message
assert assistant.tool_calls is not None
call = assistant.tool_calls[0]
arguments = json.loads(call.arguments)
tool_output = {"city": arguments["city"], "temperature_c": 19}

final = await llm_chat(
    provider="openrouter",
    model="openai/gpt-oss-20b",
    messages=[
        question,
        assistant.to_input_message(),
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(tool_output),
        },
    ],
    tools=tools,
    tool_choice="auto",
    parallel_tool_calls=False,
)
answer = final.llm.raw_text
```

Tool-result messages must immediately resolve every call in the preceding assistant message, once each, before another user or assistant message. Parallel results may be supplied in any order. Provider and model support for the forwarded controls can vary; an upstream rejection is returned as a tool failure.

The canonical miner request field is `max_output_tokens`. The SDK accepts `max_tokens` as a compatibility alias and normalizes it to `max_output_tokens`; setting both is rejected. The former miner-facing `include` and `response_format` fields are rejected. Internal validator structured-output postprocessing is separate and is not exposed through `llm_chat`.

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

Embedding outputs are ordinary tool responses for miner code. They are not citation sources, so they do not replace `search_web` or `fetch_page` evidence when an answer needs citations.

`provider_extra` is strict and selected by `provider`. Use it only for selected-provider-specific request additions that are not already common tool parameters. OpenRouter supports provider selection for both `llm_chat` and `embed_text`:

```python
await llm_chat(
    provider="openrouter",
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Reply with only ok."}],
    provider_extra={"provider": {"only": ["cerebras"]}},
)

await embed_text(
    "What is Harnyx?",
    provider="openrouter",
    model="qwen/qwen3-embedding-8b",
    input_type="query",
    provider_extra={"provider": {"only": ["nebius"]}},
)
```

OpenRouter also accepts an optional `provider.allow_fallbacks` boolean. Omit it to use OpenRouter's default fallback behavior; set it only when your miner needs to explicitly choose whether OpenRouter may fall back to another hosted provider after the selected provider fails. You can pass it with `provider.only`, or by itself as `provider_extra={"provider": {"allow_fallbacks": False}}`.

AI Gateway uses `providerOptions.gateway` for request-level upstream provider selection:

```python
await llm_chat(
    provider="ai_gateway",
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Reply with only ok."}],
    provider_extra={"providerOptions": {"gateway": {"only": ["cerebras"]}}},
)
```

The SDK still accepts the legacy top-level `provider.only` input and normalizes it to `providerOptions.gateway.only` before invoking AI Gateway. New miner code should use the canonical `providerOptions.gateway` form shown above.

Do not pass `provider_extra={"provider": "cerebras"}`. The SDK/runtime rejects the raw string form.

**Gemma 4 reasoning through AI Gateway pinned to Cerebras requires an explicit typed effort.** For example, use `thinking={"enabled": True, "effort": "medium"}` with `google/gemma-4-31b-it`; `low` and `high` are also supported. The AI Gateway adapter translates that common control to Cerebras's provider-specific `reasoningEffort`, and the response exposes `reasoning`, `reasoning_details`, and positive `reasoning_tokens`. Keep reasoning controls in `thinking`; do not pass `providerOptions.cerebras` through `provider_extra`.

AI Gateway model ids currently allowed by the tool contract are `thinkingmachines/inkling`, `zai/glm-5.2-fast`, `openai/gpt-oss-20b`, `zai/glm-4.7`, `google/gemma-4-31b-it`, `openai/gpt-oss-120b`, `minimax/minimax-m2.7`, `zai/glm-4.7-flash`, `deepseek/deepseek-v4-flash`, `deepseek/deepseek-v4-flash-0731`, `deepseek/deepseek-v4-pro`, `meta/muse-glimmer-30b`, and `alibaba/qwen3.8-27b`. Use `tooling_info().response["pricing"]["llm_chat"]["provider_models"]["ai_gateway"]` for representative static rates; actual AI Gateway returned cost wins when present.

Do not put common behavior in `provider_extra`. For example, reasoning controls belong in `thinking` even when a provider's raw API spells them differently. Chutes raw reasoning options are handled by `thinking`, not `provider_extra`. Other OpenRouter provider-preference fields such as `order`, `require_parameters`, `ignore`, `quantizations`, `sort`, and `max_price` are not supported here.

`llm_chat` accepts a typed `thinking` option:

| Provider | Model | `enabled=True` / `enabled=False` | `effort` | `budget` |
|----------|-------|----------------------------------|----------|----------|
| `openrouter` | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` | Supported via OpenRouter `reasoning.enabled` / `reasoning.effort="none"` | Supported via OpenRouter `reasoning.effort` | Supported via OpenRouter `reasoning.max_tokens` |
| `openrouter` | `deepseek/deepseek-v3.2`, `z-ai/glm-5`, `qwen/qwen3.6-27b`, `qwen/qwen3.8-27b`, `google/gemma-4-31b-it` | Supported via OpenRouter `reasoning.enabled` / `reasoning.effort="none"` | Supported via OpenRouter `reasoning.effort` | Supported via OpenRouter `reasoning.max_tokens` |
| `openrouter` | `deepseek/deepseek-v4-flash`, `deepseek/deepseek-v4-flash-0731`, `deepseek/deepseek-v4-pro`, `z-ai/glm-5.2`, `thinkingmachines/inkling`, `qwen/qwen3.5-397b-a17b`, `meta/muse-glimmer-30b` | Supported via OpenRouter `reasoning.enabled` / `reasoning.effort="none"` | Supported via OpenRouter `reasoning.effort` | Supported via OpenRouter `reasoning.max_tokens` |
| `ai_gateway` | Allowed AI Gateway models except `google/gemma-4-31b-it` pinned to Cerebras | Supported via AI Gateway `reasoning.enabled` / `reasoning.effort="none"` | Supported via AI Gateway `reasoning.effort` | Supported via AI Gateway `reasoning.max_tokens` |
| `ai_gateway` | `google/gemma-4-31b-it` pinned to Cerebras | Enable by supplying an explicit `effort`; disabling uses Gemma's disabled provider default | Supported via Cerebras `reasoningEffort` | Unsupported for this route; not serialized into a Cerebras provider option |
| `chutes` | `deepseek-ai/DeepSeek-V3.2-TEE` | Supported via `chat_template_kwargs.thinking` | Unsupported for Chutes; not serialized | Unsupported for Chutes; not serialized |
| `chutes` | `Qwen/Qwen3.6-27B-TEE`, `Qwen/Qwen3.8-27B-TEE`, `google/gemma-4-31B-turbo-TEE` | Supported via `chat_template_kwargs.enable_thinking` | Unsupported for Chutes; not serialized | Unsupported for Chutes; not serialized |
| `chutes` | `moonshotai/Kimi-K2.6-TEE`, `zai-org/GLM-5.2-TEE`, `Qwen/Qwen3.5-397B-A17B-TEE` | No verified Chutes toggle; typed hints are not serialized and provider defaults apply | Unsupported for Chutes; not serialized | Unsupported for Chutes; not serialized |

```python
await llm_chat(
    provider="chutes",
    model="deepseek-ai/DeepSeek-V3.2-TEE",
    messages=[{"role": "user", "content": "Solve 17 * 23."}],
    temperature=0.0,
    thinking={"enabled": True},
)

await llm_chat(
    provider="openrouter",
    model="deepseek/deepseek-v3.2",
    messages=[{"role": "user", "content": "Reply with only ok."}],
    temperature=0.0,
    thinking={"effort": "low"},
)
```

Omit `thinking` to use provider defaults. `effort` accepts `"low"`, `"medium"`, or `"high"` and `budget` must be a positive integer. OpenRouter-selected and AI Gateway-selected models honor those fields through provider reasoning controls. Gemma 4 pinned to Cerebras requires an explicit `effort` and does not support `budget`. Do not send `effort` and `budget` together; that is a validation error. Provider support is best effort, so unsupported level/budget hints are not serialized into raw provider-body fields.

See [`../../miner/README.md`](../../miner/README.md) for the end-to-end miner workflow (Write -> Test -> Submit).
