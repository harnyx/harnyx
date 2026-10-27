# harnyx-miner-sdk

Agent-facing SDK for Harnyx miners: entrypoints, request/response contracts, and tool-call helpers.

This package is imported by **your miner agent script**.

## Registered endpoint protocol

`harnyx_miner_sdk.endpoint_protocol` contains the strict version-one wire models for registered miner endpoints:

- `EndpointAssignment` binds an assignment UUID, `Query`, query digest, expected miner hotkey and HTTPS endpoint URL, validator callback URL, explicit search URL, optional opaque `callback_context`, nonce, and expiry.
- `EndpointCallback` returns the same binding with a normal SDK `Response`.
- `EndpointStatusResponse` reports `running`, `completed`, or `unknown` for caller-driven recovery.
- `EndpointSearchRequest` and `EndpointSearchResponse` carry assignment-bound `search_web` or `fetch_page` calls and their receipt-backed results. Deprecated `search_ai` is unavailable to miners; Platform rejects it before reserving a search or contacting a provider.
- Assignment and callback acknowledgements confirm acceptance and a durable `persisted` or `closed` terminal result.

All models reject extra fields. Callback URLs may use HTTP or HTTPS and must have valid URL syntax and a host; malformed destinations are rejected before miner intake retains them. Platform permits HTTP callbacks only when its existing deployment environment is exact lowercase `staging`; validators verify the signed assignment and pass its callback address through unchanged. Endpoint and search URLs remain HTTPS-only. Validators and miners sign the exact HTTP method, path, and raw body with their Bittensor hotkeys using `harnyx_commons.bittensor.build_canonical_request`. Miners authenticate the requesting validator using its signature, subnet registration and validator permit; Platform authorization is not required. A callback must reuse every answer binding from the assignment and echo supplied `callback_context` unchanged in `X-Harnyx-Callback-Context`, without interpreting it. Context is optional, limited to 24,000 printable ASCII characters, and cannot contain control characters or leading/trailing whitespace. Empty context is valid and differs from an omitted header. Our Platform-coordinated validator uses its existing encoded Platform-signed assignment document as context; independent validators can omit it or supply their own handler context. Status requests require only the validator signature. Complete answers must reach an authorized validator before the original deadline; validators have 60 additional seconds to forward that original receive event to Platform. Already-saved results remain recoverable afterward. Its citations may reference only successful Platform search receipts created by that assignment.

Assignment acknowledgement and status replies must be uncompressed and fit within 8 KiB. Validators request `Accept-Encoding: identity` and accept an absent `Content-Encoding` header or `identity`; they reject other encodings before reading or decoding the body. A rejected acknowledgement leaves delivery uncertain, and a rejected status reply is unavailable for recovery polling. This restriction does not change callback acceptance or provider-search response handling.

Search requires a nonempty `X-Provider-Api-Key` header; missing or empty values return HTTP 401. Choose a `receipt_id` of 1–256 characters without NUL (`U+0000`). Platform rejects invalid IDs with HTTP 422 before reserving or executing a search and preserves accepted IDs exactly.

Object keys anywhere in search `args` or `kwargs` must also exclude NUL. Numeric values at every depth must be finite: `NaN`, `Infinity`, and `-Infinity` return HTTP 422 before reservation or persistence, without conversion. Finite numbers and string values remain unchanged. Invalid keys return HTTP 422 before reservation; Platform does not rename keys or change valid Unicode keys and string values. Handled authentication, assignment, body-validation and search errors use a string `detail`; HTTP 422 can instead contain FastAPI's validation-detail array when the assignment path is invalid. The generated API contract describes both shapes.

New assignment searches permit at most 2 MiB (2,097,152 bytes) of decoded provider HTTP response and 2 MiB of normalized saved evidence. The evidence limit counts the complete compact UTF-8 `{response, results}` JSON envelope, including duplicated source text and metadata. Platform closes an oversized download and rejects the search with HTTP 400; it never truncates results or automatically retries them. Known incurred provider costs remain recorded; billing unavailable after an early download abort remains unknown, not zero. Existing successful receipts still replay exactly, without applying the new-size check again. These limits do not extend the assignment deadline or change callback eligibility.

If a successful search response is lost, repeat the same search request with the same `receipt_id` under the same assignment. While that assignment remains active and before its original deadline, Platform returns the saved response and result identifiers without another provider call, receipt or charge, even when new-search capacity is full. A conflicting request or an unfinished or failed call with that receipt is rejected; replay does not restart search. The same receipt under a different assignment identifies a separate call. Authenticate and supply the provider-key header as for the original request.

Recovery skips success visible at its final check, but a concurrent callback can commit before the retry is sent. Duplicate network delivery is therefore possible. Retries reuse the same assignment ID and original deadline; they do not create another assignment or independently count toward evaluation or reward. Platform saves only the first valid, timely callback. Later callbacks cannot replace that answer or its recorded latency, even if the miner repeats work and returns a different valid answer.

We recommend deduplicating assignments and retaining signed results until Platform acknowledges a durable outcome. These practices reduce repeated work and help recover delivery, but Platform does not trust miners to implement them. A miner may reuse work, repeat work or ignore a duplicate request without changing Platform's saved result.

The runnable `harnyx-miner-endpoint-test` command in the miner package exercises assignment verification, assignment-bound search, signed callback retry, status polling, and deliberate `unknown` recovery after its in-memory state is cleared. It is protocol tooling, not durable miner execution storage.

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

Register entrypoints with `@entrypoint(...)`. The public `query` entrypoint has
the following contract:

Rules:
- Must be `async def`
- May use the legacy one-parameter form or the context-aware two-parameter form
- The first parameter must be annotated as `harnyx_miner_sdk.query.Query`
- When present, the second parameter must be annotated as `harnyx_miner_sdk.context.ContextSnapshot`
- The return type must be `harnyx_miner_sdk.query.Response`

Non-query diagnostic entrypoints remain `async def` functions with exactly one
annotated request parameter and an annotated return type. They do not receive
`ContextSnapshot`.

Use the context-aware form for new artifacts. The sandbox validates the context
and enforces the same cost and time limits for both forms. A legacy entrypoint
continues to run, but it cannot inspect those budgets because its function does
not receive the context.

Example:

```python
from harnyx_miner_sdk.context import ContextSnapshot
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response


@entrypoint("query")
async def query(query: Query, context: ContextSnapshot) -> Response:
    print(f"time limit: {context.time_budget.limit_seconds}s")
    return Response(text=query.text)
```

## Query contract

Validators call `query` with a `Query` payload and an immutable
`ContextSnapshot`. The context contains the initial `cost_budget` and configured
full `time_budget.limit_seconds`; it does not contain remaining time.

```json
{
  "text": "Explain why validator sandboxes matter."
}
```

`fast` identifies the query's scoring mode and defaults to `false` when omitted.
Serializers omit the default so ordinary queries remain compatible with strict
consumers that predate fast mode; they include `"fast": true` explicitly.
`fast: true` means the response is scored for correctness only. The judge
identifies correct expected answer components and excessive answer components,
then deterministic code computes a precision/recall F1 score from `0` to `1`.
Citations remain accepted but are omitted from this judgment, are not required,
and provide no scoring benefit. Fast scoring works with both plain-text and
structured answers. Each source-generation slot independently selects
`fast: true` with probability `0.3`; a finite batch may contain any mix of modes.

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

Either answer mode may also include an optional response-level `note`:

```json
{
  "output": {"region": "North", "finding": "Demand increased."},
  "note": "The question's stated date range was incorrect; this uses the published range [[1]].",
  "citations": [
    {"receipt_id": "receipt-123", "result_id": "result-abc"}
  ]
}
```

Use `note` only to explain, qualify, support, or correct the required answer. It
is outside `output_schema`, cannot replace or repair a missing or invalid
`text`/`output`, and is not private reasoning or a no-answer branch. Omission is
neutral. A note may contain at most 80,000 characters and must not be blank.

`output_schema` is an entrypoint contract. It is not an `llm_chat` request
option. A missing or `null` schema requires `Response.text` for a plain-text
answer; a present schema, including `{}`, requires `Response.output`. Return
exactly one of `text` or `output`, never both. A present `output: null` is a
structured answer when the schema permits JSON `null`; omitting `output` means
no structured answer was supplied. The schema applies
only to `output`; `note` and citation refs remain Harnyx-owned response siblings.
The host validates the note and hydrates citation refs after output validation.

Schemas must use Draft 2020-12 when `$schema` is declared, must resolve entirely
within the submitted schema, and cannot use external `$ref` or `$dynamicRef`
targets. The compact JSON encoding of the schema and of the returned structured
value is limited to 80,000 characters each. Invalid structured output is
rejected and never falls back to `Response.text`.

Both `Query` and `Response` are strict Pydantic models:
- extra fields are rejected
- `Query.text` is required and empty/whitespace-only strings are rejected
- `Query.fast` is a strict boolean that defaults to `false`
- `Response` requires exactly one answer field for the query mode; structured `output` may be JSON `null`
- `note` is optional, non-blank when present, and may contain at most 80,000 characters
- `citations` is optional
- plain-text response `text` may contain at most 80,000 characters
- structured schemas and outputs may contain at most 80,000 compact JSON characters each
- `citations`, when present, may contain at most 200 receipt refs
- each citation must include `receipt_id` and `result_id`
- citation refs may also include `slices=[CitationSlice(start=..., end=...)]`; refs without slices use the entire referenced result text
- citations may materialize at most 400 evidence segments and 120,000 source-text characters per answer

For `fast: false` scoring, treat `citations` as required for answers that make
non-obvious factual claims or depend on search/tool evidence. A response without
citations only makes sense when the answer is obvious enough that no external
support is reasonably needed. Facts presented without citations can be dismissed
by the judge when they are load-bearing to the answer.

For `fast: true` scoring, answer every required part directly and avoid unrelated
or contradictory claims. The scorer compares the required answer content and
optional notes without sending citations to the judge. A reference note may
qualify the expected answer but does not create another required component. Your
note cannot replace a missing answer component, and an incorrect, contradictory,
or non-responsive note may count as excessive content.

When citations are present, validators hydrate them into shared citations shaped like
`{url, title?, excerpts: [{start, end, text}]}` for storage and monitoring. The judge and public Research Task API receive only excerpt text. Each citation keeps one number even when it contains multiple excerpts. Hydrated excerpts are materialized by the validator from the referenced tool result's `note` text. A ref without slices materializes the full result note. A ref with slices materializes only those offsets. Miner-authored citation text is not accepted as evidence.

For prose answers, `[[n]]` is an exact one-based pointer to `Response.citations[n - 1]`; ordinary `[n]` text is not a citation. Unless the query explicitly rejects citations, give every material researched claim a valid pointer to the evidence that supports it. Submission order is authoritative: duplicate refs occupy separate positions, and validators never deduplicate, renumber, or remap them. Miners submit only non-null `CitationRef` items. If a submitted position cannot be resolved or hydrated, the public hydrated response contains `null` at that position, that position provides no factual support, and later pointers do not move. Every recognized numeric `[[n]]` in `text`, any structured `output` string value (including nested objects and arrays), or `note` must point to an in-range, non-null hydrated citation. Zero, out-of-range positions, and references to unresolved positions invalidate the response just like an output schema mismatch. Unused null positions do not fail this check. Use separate markers for multiple citations, such as `[[1]][[2]]`. Malformed-marker detection is not enforced by this check; JSON keys and citation source metadata are not scanned. Missing support for a claim, irrelevant evidence, and mismatched evidence remain judge-visible quality defects rather than automatic invalid responses or losses.

Factual claims in `note` use the same `[[n]]` mapping and the same hydrated
citations array as the required answer. The note is never evidence itself.
For `fast: false`, correctness and evidence remain primary; the judge considers
a useful note only as a tie-break when answers and evidence are otherwise
comparable. Absence is neutral, repetition earns nothing, and a wrong,
contradictory, or unsupported note may lose that tie-break.

Structured output does not require inline pointers by default. Add them only when the query or a prose-capable field description explicitly requires citations, and only in that prose-capable field. Do not add citation syntax to atomic integer, number, boolean, enum, identifier, date-token, or similar fields.

When the query does not request a conflicting form, prefer a clear, self-contained, reader-facing synthesis and use Markdown only when it reduces reader effort. An explicit requested form such as XML or a terse answer overrides that default. Correctness, requested coverage, instruction following, evidence support, and calibrated uncertainty take priority over presentation.

## Receipts and citations

Hosted tool calls return two layers of identifiers:

- `receipt_id`: the tool call itself
- `result_id`: a specific referenceable result from that tool call

Your `Response.citations` must point at the exact result(s) that support your answer:

```python
from harnyx_miner_sdk.api import search_web
from harnyx_miner_sdk.context import ContextSnapshot
from harnyx_miner_sdk.query import CitationRef, Query, Response


async def query(query: Query, _context: ContextSnapshot) -> Response:
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
- `search_web(query, provider="parallel" | "desearch" | "firecrawl" | "exa" | "tavily" | "deepseek", timeout=..., provider_extra=..., **kwargs)`
- `fetch_page(url, provider="parallel" | "desearch" | "firecrawl" | "exa" | "tavily" | "deepseek" | "expand", timeout=..., provider_extra=...)`

`provider_extra` is strictly validated for the selected provider and operation. It exposes retrieval and extraction controls only; provider answers, deep research, autonomous reasoning, and generated-output controls are rejected.

### Search providers

| Provider | `search_web.provider_extra` |
|---|---|
| `desearch` | `start` |
| `parallel` | `mode` (`turbo`, `basic`, or `advanced`), `max_chars_total`, `source_policy`, `fetch_policy`, `excerpt_settings`, `location` |
| `firecrawl` | `categories`, domain filters, `tbs`, `location`, `country`, invalid-URL and privacy controls |
| `exa` | `type` (`auto`, `instant`, or `fast`), category, domain/date/location/moderation filters |
| `deepseek` | `ori_question` (1–128 characters), `localization` (IP address), `need_content` (default `false`) |
| `tavily` | `search_depth` (`basic`, `fast`, `advanced`, or `ultra-fast`), chunks, topic/time/date/domain/country/exact/safe controls |

### Fetch providers

| Provider | `fetch_page.provider_extra` |
|---|---|
| `desearch` | `format`, `js`, `wait` |
| `parallel` | `objective`, `max_chars_total`, `fetch_policy`, `excerpt_settings`, `full_content` |
| `firecrawl` | `formats` (`markdown` and/or `rawHtml`), main-content/tag/cache/wait/mobile/PDF/location/image/ad/proxy/cache-retention controls |
| `exa` | `text` (`true` or text options), `max_age_hours`, `livecrawl_timeout` |
| `tavily` | `query`, `chunks_per_source`, `extract_depth`, `format` |
| `deepseek` | Omitted or `{}` only; returns Markdown |
| `expand` | Omitted or `{}` only; full-page scrolling, Markdown and State JSON are fixed |

DeepSeek search forwards one query unchanged, or joins multiple queries with `OR` and parentheses. The resulting query must be at most 128 characters; `num` must be 0–50. Zero skips the API call. Results use full content when requested and available, otherwise the long summary or short snippet. Fetch requires an HTTP(S) URL with a host, no username/password, and at most 8192 characters. It returns Markdown using the original URL. Both tools cost $0.00075 per successful nonempty API call, independent of result count. Empty responses and confirmed HTTP rejections cost zero; transport failures have unknown cost. To avoid possibly duplicate charges, DeepSeek retries connection-establishment failures and confirmed unbilled HTTP failures, but stops after transport failures where the request may already have reached the provider. Direct API calls do not receive the temporary server-side-tools promotion. Configure `DEEPSEEK_API_KEY` for local evaluation or register a DeepSeek credential for hosted miner calls. See the [DeepSeek contract](https://api-docs.deepseek.com/guides/search).

Model IDs have surrounding whitespace removed before provider-specific validation. The October 17 Flash release is available to `llm_chat` as `deepseek/deepseek-v4.1-flash-1017` with `provider="openrouter"` or `provider="ai_gateway"`. Existing undated model selections remain available. With thinking enabled, omit forced `tool_choice` and replay assistant reasoning alongside tool results, as required by DeepSeek.

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

### Expand fetch results and citations

For **`provider="expand"` only**, every successful fetch returns exactly two
items in a fixed order. Each item has its own referenceable result:

| Index | `response.data[index].content` | `results[index].note` | Citation target |
|---|---|---|---|
| 0 | Original Markdown | The same Markdown | `results[0].result_id` |
| 1 | Standalone serialized JSON | The same JSON text | `results[1].result_id` |

Both items use the same resolved page URL and title, but have **distinct result
IDs**. `content` and `note` are strings. Parse the second content with
`json.loads`; it has no Markdown fences or prefix. Index 1 is not generally JSON
for other fetch providers. Reordering or replacing an Expand item changes this
contract.

```python
import json

from harnyx_miner_sdk.api import fetch_page
from harnyx_miner_sdk.query import CitationRef, CitationSlice

fetched = await fetch_page("https://taostats.io/subnets/25", provider="expand")
markdown = fetched.response.data[0].content
structured = json.loads(fetched.response.data[1].content)

# Cite JSON alone, without calculating offsets into the Markdown.
json_citation = CitationRef(
    receipt_id=fetched.receipt_id,
    result_id=fetched.results[1].result_id,
)

# Optionally select a narrower excerpt within the JSON result's own note.
# start/end must meet the existing citation slice limits described above.
def cite_json_slice(start: int, end: int) -> CitationRef:
    return CitationRef(
        receipt_id=fetched.receipt_id,
        result_id=fetched.results[1].result_id,
        slices=[CitationSlice(start=start, end=end)],
    )
```

Use result 0 for Markdown evidence, result 1 for JSON evidence, or cite both.
Without slices, a ref selects that result's entire note. Slice offsets address
only the selected result's original, unstripped `note`, never combined content.
Existing minimum slice length and total evidence limits still apply; a large
JSON result may require a narrower slice. Evidence is never silently truncated.

The JSON document contains the returned `meta` object and `json` evidence array,
plus `data: {"degraded": ...}` when returned. Evidence bodies preserve their JSON
types and source, omission and pointer metadata. State JSON does not guarantee
complete coverage of a page; referenced or omitted bodies are not downloaded
automatically. An empty `json: []` is valid and still occupies index 1. Missing
required JSON, blank Markdown, malformed responses and source errors fail the
fetch rather than dropping or shifting an item.

Converted documents use the same two-item contract. Their `json` array is empty,
and `meta` may omit browser-only fields such as `lang`, `pageType`, and `playground`.

Expand supports fetching only, not `search_web`. It uses
`POST /v1/fetch/json?include=appendix,statejson` with full-page scrolling;
`provider_extra` accepts only omission or an empty object. Under existing static
host settlement, two referenceable results cost **$0.001 per successful fetch**
($0.0005/result). This is distinct from Expand's external bill: actual vendor
USD cost remains unknown when the provider does not return it.

### Other tool helpers

- `llm_chat(provider="chutes" | "openrouter" | "ai_gateway" | "saygm" | "cloudflare", messages=[...], model="<provider-specific model id>", timeout=..., temperature=0.0, thinking={"enabled": True}, provider_extra=...)`
- `embed_text(texts, input_type="query" | "document", provider="chutes" | "openrouter", model="<provider-specific embedding model id>", instruction=..., dimensions=..., provider_extra=..., timeout=...)`
- `decision_query(provider="openrouter" | "ai_gateway" | "cloudflare" | "jev", model="...", state=..., questions=..., timeout=...)`
- `tooling_info(timeout=...)`
- `test_tool(message, timeout=...)`

Every hosted tool helper accepts an optional positive finite `timeout` in seconds. For provider-backed tools other than `llm_chat`, the tool host bounds the complete provider-backed invocation, including host-owned retries/backoff, and raises a tool invocation error if the deadline expires. `llm_chat` makes one provider attempt per SDK call; retry loops belong in miner script code when desired. `test_tool` completes locally and accepts the same parameter for interface consistency.

`tooling_info` uses a five-minute process cache for OpenRouter model discovery.
On a cold or expired cache, refresh takes at most two seconds, or the supplied
`timeout` if shorter. If refresh fails, other tool information remains available;
the OpenRouter model list is empty and
`pricing.llm_chat.openrouter.discovery_available` is `false`. This indicates
unavailable discovery, not that every OpenRouter model is disallowed. Paid calls
independently require fresh pricing and never use stale prices after a failed refresh.

For `llm_chat`, `timeout` also accepts `Timeout(total, prefill=..., inactivity=...)`:

```python
from harnyx_miner_sdk.api import Timeout, llm_chat

response = await llm_chat(
    provider="chutes",
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Explain this result."}],
    timeout=Timeout(120, prefill=60, inactivity=30),
)
```

The total bounds the hosted call. Prefill bounds the provider attempt until its
first text, reasoning, or function-call output; inactivity bounds gaps in output
after that point. Heartbeats and empty metadata do not reset these clocks.
Unspecified phases add no phase limit for miner calls. Numeric `timeout=120`,
omitted values, and host clamping keep their existing behavior. The default host
maximum remains 270 seconds; a larger total is clamped while supplied phase
limits are preserved. Other hosted tools still accept only numeric timeouts.

The detailed form requires SDK 0.1.18 and an updated tool host. Older SDK numeric
requests work with the updated host; an older host rejects the detailed object.

Every successful hosted-tool helper response includes `budget`, which reports
the evaluation session's monetary budget and settled spend after that call.

The query's immutable `ContextSnapshot` supplies the initial monetary budget and
the configured full execution limit as `context.cost_budget` and
`context.time_budget.limit_seconds`. The limit starts before the sandbox worker
loads the miner artifact, so it is a configured capacity, not remaining time.
A helper's optional `timeout` still bounds only that hosted tool call. The host
derives its internal tool-session authorization lifetime from the invocation
limit and normally revokes it as soon as the invocation completes; it does not
enforce the miner entrypoint deadline.

Firecrawl is an ordinary-web provider: it supports `search_web` and `fetch_page`.

`llm_chat` model ids are provider-specific. Use `tooling_info().response["allowed_llm_provider_models"][provider]` for discovery and pass the selected provider's model id exactly. OpenRouter discovery uses its cached public catalog rather than a repository whitelist. Explicit `:free` variants and `openrouter/free` are excluded. Paid promotions are allowed when their regular reference cost can be established. Other providers retain their existing model lists.

SayGM uses native Chat Completions ids such as `gpt-oss-20b`, `glm-5.2`, and `gemma-4-31b-turbo-tee`. Miner access is limited to equivalents of models already authorized through the existing miner providers, not the full SayGM catalog. GPT frontier models, Kimi K3, GLM 5.1/full 5.3, and other newly available models are rejected. Claude and Gemini native API models are outside this provider's current scope. Register a `saygm` credential with the platform before hosted calls. SayGM does not accept `provider_extra`, embeddings, or `decision_query`. Its settled `usage.cost_nano_usd` (or `usage.cost` when the integer field is absent) owns actual-cost accounting; published retail ceiling rates are budget estimates only. Missing, invalid, or inconsistent settlement never falls back to those estimates. `tooling_info().response["pricing"]["llm_chat"]["provider_settlement_order"]` lists each provider’s actual-charge sources, while `provider_rate_usage` distinguishes budget-only estimates from rates also usable for settlement fallback. OpenRouter is omitted from the static `provider_models` rate table; `pricing.llm_chat.openrouter` describes its dynamic accounting. Regular reference cost governs miner limits and comparison, and actual provider charge governs financial tracking. The legacy `settlement_order` list remains available for existing clients; `provider_settlement_order[provider]` owns the actual policy, including SayGM's returned-settlement-only rule.

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

### Decision models

`decision_query` answers named questions about a shared `state` (a string, JSON object, or JSON array). It returns typed decisions and probabilities. Question types are `choice` (named criteria), `score` (an ordered criteria list), and `boolean` (a probability between 0 and 1). Boolean criteria, when supplied, must include both `"true"` and `"false"`.

With `provider="cloudflare"`, use `@cf/cloudflare/clef`, `@cf/cloudflare/clef-flash`,
or `@cf/cloudflare/clef-omni` for these decision questions. Use
`@cf/cloudflare/clairic` for extraction questions, also under `questions`:

```python
extracted = await decision_query(
    provider="cloudflare",
    model="@cf/cloudflare/clairic",
    state="Invoice INV-42. Total: USD 120.00. Due: 2026-11-01.",
    questions={
        "invoice": {"type": "text", "instructions": "Invoice number"},
        "total": {"type": "number", "unit": "USD", "decimals": 2},
        "due": {"type": "date", "instructions": "Payment due date"},
    },
)
total = extracted.response.answers["total"].value
```

Clairic supports `text`, `number`, `date`, `time`, `datetime`, `duration`, `list`,
`estimate`, and `forecast`. Lists require `items` (`text`, `number`, or `date`).
Estimates and forecasts require `min` and `max`; forecasts also require `horizon`
and `interval`. Calendar steps (`month` and `year`) require date bounds; numeric
prediction grids use positive numeric steps. Optional `instructions` describe
the desired value. Numeric
precision, integer values, units, prediction grids and null thresholds follow
the [Clairic field contract](https://developers.cloudflare.com/workers-ai/models/clairic/).
An extraction response preserves `value`, `guess`, source spans and uncertainty
metadata, including duration `unit`. Forecasts preserve the native parallel arrays
in `value`, `median`, `p10`, and `p90`; each array matches the requested `horizon`
and each predicted value respects the requested bounds. Forecasts require either
that value array or a complete `steps` representation; optional summaries alone
do not supply the requested values. When both representations are supplied,
corresponding values and same-name statistics must agree. Each position is
validated as one estimate, including percentile ordering across the two forms.
Integer bounds, steps, values and statistics retain their precision during validation.
Missing list matches return `[]`, never null.
For scalar extraction, a null `value` means no determined value; do not
substitute `guess`.
Predictions can be wrong despite satisfying the requested type. Estimates must
contain a prediction value or statistic. Estimate summaries
(`mean`, `median`, `p10`, `p90`) must satisfy the same type and bounds as the value.

Decision and extraction questions cannot be mixed in one request: the selected
model must support every question type. JSON state is serialized to text for
Clairic; source spans index that text. Empty extraction text is rejected before
dispatch. This tool supports text/JSON state rather
than Cloudflare's separate native media input fields. Clairic's `truncated`
status is preserved, so callers can detect upstream input truncation. Its
reported input-token count includes the provider's processing passes and is
billed once at the published rate. Native decision/extraction models generate
no output tokens. Independently validated usage retains known charges when
answer validation fails. No provider inference retry or model fallback is added.

Cloudflare support requires `harnyx-miner-sdk>=0.1.24`.

Upload the Cloudflare account ID and API token together through miner provider
configuration. They are not tool arguments. `llm_chat(provider="cloudflare", ...)`
uses the same credential and Cloudflare-native chat model IDs. Runtime discovery
lists the supported chat/decision models and published token rates. Cloudflare
chat forwards supported `thinking.effort` settings to the model's native
`reasoning_effort` field. Provider-documented aliases are preserved (for example,
GLM 5.2 maps `low` and `medium` to `high`). Qwen 3.8 supports `low` and `medium`
through this tool; its native `xhigh` setting is outside the shared effort enum.
Kimi uses its native `reasoning_effort="none"` when thinking is disabled.
Unsupported settings, including thinking budgets and disabling reasoning on
models that require it, fail before dispatch.

For Cloudflare chat, decisions and extraction, Harnyx records both actual and
reference cost as provider-reported token counts multiplied by published model
rates, including the published cached-token rate where applicable. Neither
cost deducts account free allowances or promotional credits. These equal
accounting amounts are calculated by Harnyx; they do not represent a verified
post-credit Cloudflare invoice charge. Known token usage preserves both amounts
when answer validation or a chat stream fails, including across retry attempts
and cancellation. Each attempt uses its latest reported cumulative usage; stream
updates within that attempt are not added together. Hosted settlement preserves
these amounts and usage through cleanup, failed-call replay, and late billing
after interrupted-call recovery.

Cloudflare choice and score distributions must sum to one. A choice must have
the highest probability (ties are allowed), and a score must match the
probability-weighted level, allowing floating-point rounding. A supplied score
legend must preserve the requested criteria at each level. Date prediction
bounds compare dates regardless of ISO spelling. A supplied text source span
must select the exact returned text using character offsets, including Unicode.
Normalized numbers and dates do not require literal source-text equality.
Supplied duration units must match the requested unit (seconds by default), and
numeric extraction honors explicitly requested decimal precision. For scalar
extraction, a supplied null probability above the requested threshold (0.5 by
default) requires a null result. Estimates and forecasts require ordered
`p10`, `median`, and `p90` values when supplied; their means may fall between
grid points or outside the central interval. Optional answer metadata remains
optional.
Numeric chat stream fragments are converted to text; formatting lost by the
provider cannot be recovered.

```python
from harnyx_miner_sdk.api import decision_query

decision = await decision_query(
    provider="openrouter",
    model="cloudflare/clef-flash",
    state={"draft": "The Earth orbits the Sun."},
    questions={
        "publish": {
            "type": "boolean",
            "instructions": "Is the draft factually correct?",
        },
    },
    timeout=30,
)
probability = decision.response.answers["publish"].probability
```

Use `tooling_info().response["allowed_decision_provider_models"]` for availability. OpenRouter accepts paid native IDs from its public decision-model catalog, including canonical versioned IDs, without a repository whitelist. Explicit `:free` variants and `openrouter/free` are excluded; paid promotions with zero current prices remain eligible. Discovery and public endpoint pricing use the same five-minute process cache as chat. `pricing.decision_query.openrouter` describes dynamic accounting; its models are omitted from the static `provider_models` table. AI Gateway keeps its existing empty decision-model list; those combinations are rejected. Provider adapters translate OpenRouter's native `noul` question type to public `boolean`.

Direct TypeSafe JEV is available as `provider="jev"`, `model="jev-1.13.0"` for choice, score, and boolean questions. The adapter maps boolean questions to native `noul` and returns their probability through the usual typed answer. JEV is a decision provider; it does not support `llm_chat`, embeddings, or extraction questions. Register your TypeSafe key as `provider_credentials.jev` for hosted calls; direct local execution uses `TYPESAFE_API_KEY`. The platform never substitutes its key for a missing miner credential.

JEV forwards the complete state and questions to TypeSafe without retries, fallback, truncation, or the optional long-context header. The pinned model avoids silently changing behavior through a latest-model alias. Its native response must include token usage and choice/score confidence and probabilities. JEV accounting uses TypeSafe's [published input-only rate](https://docs.typesafe.ai/models): $0.042 per million billable input tokens, with output free. Actual and reference accounting amounts are equal at that rate; these amounts do not claim the miner's net invoice charge. Valid usage still counts when answer validation fails. Dispatched failures without valid usage remain financially unknown. Discovery lists the supported model and rate through `tooling_info`.

Choice answers contain `choice`; score answers contain `score`; both may contain `probabilities`. Boolean answers contain `probability`. Absent distributions remain absent, and rounded values are preserved. Native token usage stays in the response and receipt, outside chat token summaries. Decision results are log-only and do not become citation sources.

Charges count toward the session budget, including a failed response with a known provider charge. Known charge and usage evidence survives client cleanup failures, including cancellation or timeout during cleanup. OpenRouter requires original provider billing and a pricing snapshot bound before dispatch. Regular/reference cost, reconstructed by reversing the serving provider’s maximum advertised discount, governs task limits and comparison; the observed actual charge independently governs financial tracking. Separately identified BYOK router charges are not discount-normalized. OpenRouter never substitutes a static estimate for an actual charge. Missing reference evidence rejects before dispatch when detectable. After a failed call, observed actual charges and unavailable evidence remain recorded, but missing reference amounts are omitted from budget and comparison totals. The script may catch the error and retry, switch models or return an answer; the host does not retry the task automatically. Known reference charges still count and successful calls still require reference accounting. Lost responses use recorded accounting without repricing, and observed billing is persisted before hosted client cleanup. AI Gateway retains its returned-cost/static-rate settlement policy. A dispatched failure without a known charge leaves actual cost totals unknown; it does not invent a budget debit. There is one provider attempt per tool call. OpenRouter may truncate text upstream; Harnyx forwards state and instructions without adding truncation or input limits.

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

AI Gateway model ids currently allowed by the tool contract are `thinkingmachines/inkling`, `zai/glm-5.2-fast`, `zai/glm-5.3-flash`, `openai/gpt-oss-20b`, `zai/glm-4.7`, `google/gemma-4-31b-it`, `openai/gpt-oss-120b`, `minimax/minimax-m2.7`, `zai/glm-4.7-flash`, `deepseek/deepseek-v4-flash`, `deepseek/deepseek-v4-flash-0731`, `deepseek/deepseek-v4.1-flash`, `deepseek/deepseek-v4.1-flash-1017`, `deepseek/deepseek-v4-pro`, `meta/muse-glimmer-30b`, `meta/muse-spark-open`, `alibaba/qwen3.8-27b`, `tencent/hy4-preview`, `xiaomi/mimo-v2.6-flash`, `xiaomi/mimo-v2.6-pro`, and `xiaomi/mimo-v2.6-pro-ultraspeed`. Use `tooling_info().response["pricing"]["llm_chat"]["provider_models"]["ai_gateway"]` for representative static rates; actual AI Gateway returned cost wins when present.

OpenRouter also allows `xiaomi/mimo-v2.6-flash`, `xiaomi/mimo-v2.6-pro`, and `xiaomi/mimo-v2.6-pro-ultraspeed`. The UltraSpeed route serves Pro; the open-weight `MiMo-V2.6-Distill-Qwen-9B` checkpoint has no verified route through a miner-supported provider. Use `tooling_info().response["allowed_llm_provider_models"][provider]` for the current complete list of allowed model ids.

OpenRouter and AI Gateway fallback settlement uses reported cached-input tokens at the configured cached-input rate. The provider-model pricing entry in `tooling_info()` includes `cached_input_per_million` when configured. For `meta/muse-spark-open`, reference input, output, and cached-input rates are $1.25, $4.25, and $0.15 per million tokens. Valid provider-returned cost takes precedence.

Do not put common behavior in `provider_extra`. For example, reasoning controls belong in `thinking` even when a provider's raw API spells them differently. Chutes raw reasoning options are handled by `thinking`, not `provider_extra`. Other OpenRouter provider-preference fields such as `order`, `require_parameters`, `ignore`, `quantizations`, `sort`, and `max_price` are not supported here.

`llm_chat` accepts a typed `thinking` option. The table describes verified thinking controls, not the complete model allowlist:

| Provider | Model | `enabled=True` / `enabled=False` | `effort` | `budget` |
|----------|-------|----------------------------------|----------|----------|
| `openrouter` | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` | Supported via OpenRouter `reasoning.enabled` / `reasoning.effort="none"` | Supported via OpenRouter `reasoning.effort` | Supported via OpenRouter `reasoning.max_tokens` |
| `openrouter` | `deepseek/deepseek-v3.2`, `z-ai/glm-5`, `qwen/qwen3.6-27b`, `qwen/qwen3.8-27b`, `google/gemma-4-31b-it` | Supported via OpenRouter `reasoning.enabled` / `reasoning.effort="none"` | Supported via OpenRouter `reasoning.effort` | Supported via OpenRouter `reasoning.max_tokens` |
| `openrouter` | `deepseek/deepseek-v4-flash`, `deepseek/deepseek-v4-flash-0731`, `deepseek/deepseek-v4.1-flash`, `deepseek/deepseek-v4.1-flash-1017`, `deepseek/deepseek-v4-pro`, `z-ai/glm-5.2`, `z-ai/glm-5.3-flash`, `thinkingmachines/inkling`, `qwen/qwen3.5-397b-a17b`, `meta/muse-glimmer-30b`, `meta/muse-spark-open`, `tencent/hy4-preview` | Supported via OpenRouter `reasoning.enabled` / `reasoning.effort="none"` | Supported via OpenRouter `reasoning.effort` | Supported via OpenRouter `reasoning.max_tokens` |
| `saygm` | Allowed SayGM Chat Completions models | Supported via `reasoning_effort="medium"` / `"none"` | Supported via `reasoning_effort` | Unsupported; rejected before dispatch |
| `ai_gateway` | Allowed AI Gateway models | Supported via AI Gateway `reasoning.enabled` / `reasoning.effort="none"` | Supported via AI Gateway `reasoning.effort` | Supported via AI Gateway `reasoning.max_tokens` |
| `chutes` | `deepseek-ai/DeepSeek-V3.2-TEE` | Supported via `chat_template_kwargs.thinking` | Unsupported for Chutes; not serialized | Unsupported for Chutes; not serialized |
| `chutes` | `Qwen/Qwen3.6-27B-TEE`, `Qwen/Qwen3.8-27B-TEE`, `google/gemma-4-31B-turbo-TEE` | Supported via `chat_template_kwargs.enable_thinking` | Unsupported for Chutes; not serialized | Unsupported for Chutes; not serialized |
| `chutes` | `moonshotai/Kimi-K2.6-TEE`, `zai-org/GLM-5.2-TEE`, `Qwen/Qwen3.5-397B-A17B-TEE` | No verified Chutes toggle; typed hints are not serialized and provider defaults apply | Unsupported for Chutes; not serialized | Unsupported for Chutes; not serialized |

For `deepseek/deepseek-v4.1-flash` and `deepseek/deepseek-v4.1-flash-1017` on OpenRouter or AI Gateway, omit `tool_choice` when thinking is enabled. DeepSeek rejects forced tool selection in thinking mode; preserve the assistant reasoning when replaying tool results. For Flash 1017, the SDK rejects required or named tool choices with thinking enabled, including AI Gateway's default thinking mode; explicitly disable thinking when forced selection is needed.

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

Omit `thinking` to use provider defaults. `effort` accepts `"low"`, `"medium"`, or `"high"` and `budget` must be a positive integer. OpenRouter-selected and AI Gateway-selected models honor those fields through provider reasoning controls. Do not send `effort` and `budget` together; that is a validation error. SayGM rejects `thinking.budget` before dispatch rather than converting a token budget into named effort; use `enabled` or `effort` instead. Other provider support is best effort, so unsupported level/budget hints are not serialized into raw provider-body fields.

See [`../../miner/README.md`](../../miner/README.md) for the end-to-end miner workflow (Write -> Test -> Submit).
