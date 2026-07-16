# Miner Guide

This directory contains the miner-facing CLI tools for the Harnyx Subnet.

For the end-to-end operating workflow from champion context through submit,
batch monitoring, and score diagnosis, start with the
[Mining Runbook](mining-runbook.md).

## How it fits together

```
  You (miner)
      │
      │  write agent.py
      │  (imports harnyx-miner-sdk)
      ▼
  ┌──────────────────────────────────────────┐
  │  miner/                                  │  ◀── what you interact with
  │  • harnyx-miner-local-eval  (batch eval) │
  │  • harnyx-miner-local-benchmark          │
  │  • harnyx-miner-submit      (upload)     │
  │  • harnyx-miner-config      (credentials)│
  └──────────────────────────────────────────┘
                │
                │ submits script to platform
                ▼
          ┌──────────┐
          │ Platform │
          └────┬─────┘
               │ fans out to validators
               ▼
  ┌─────────────────────────────────┐
  │  sandbox/                       │  ◀── validators run this (not you)
  │  (sandbox runtime + harness)    │
  │  loads your agent.py            │
  └─────────────────────────────────┘
```

**What each directory is:**

- `miner/` — CLI tools you use directly (`harnyx-miner-local-eval`, `harnyx-miner-local-benchmark`, `harnyx-miner-submit`, `harnyx-miner-config`)
- [`packages/miner-sdk/`](../packages/miner-sdk/README.md) — SDK your script imports; you don't need to read its docs first
- `sandbox/` — runtime that validators use to execute your script; you don't run it directly

---

## Write → Local Eval → Submit

### Step 1: Setup

From the repo root:

```bash
uv sync --all-packages --dev
```

Create a `.env` at the repo root (copy from `.env.example`) and fill:

| Variable | Purpose |
|----------|---------|
| `CHUTES_API_KEY` | Evaluation scoring and `llm_chat` tool calls |
| `OPENROUTER_API_KEY` | Optional: required only for local tooling that calls OpenRouter with an operator-owned key; miner-paid `provider="openrouter"` calls use the OpenRouter credential stored in miner config |
| `AI_GATEWAY_API_KEY` | Optional: required only for local tooling that calls AI Gateway with an operator-owned key; miner-paid `provider="ai_gateway"` calls use the AI Gateway credential stored in miner config |
| `DESEARCH_API_KEY` | Optional: required if your agent uses search tools |
| `SEARCH_PROVIDER` | Optional: required if your agent uses search tools |
| `PLATFORM_BASE_URL` | Public monitoring and script uploads |
| `BENCHMARK_LLM_PROVIDER` | Optional `correctness-v1` benchmark judge provider; defaults to `chutes` |
| `BENCHMARK_LLM_MODEL` | Required when running a `correctness-v1` local benchmark |
| `BENCHMARK_RUBRIC_JUDGE_LLM_PROVIDER` | Required with `BENCHMARK_RUBRIC_JUDGE_LLM_MODEL` for `weighted-rubric-v1` local benchmark scoring |
| `BENCHMARK_RUBRIC_JUDGE_LLM_MODEL` | Required with `BENCHMARK_RUBRIC_JUDGE_LLM_PROVIDER` for `weighted-rubric-v1` local benchmark scoring |

The checked-in default is `SEARCH_PROVIDER=desearch`. If you need a fallback search provider, miner tooling also supports `parallel`; set `SEARCH_PROVIDER=parallel` and `PARALLEL_API_KEY`.
If you set either benchmark judge provider to `vertex`, also configure Vertex credentials such as `GCP_PROJECT_ID` and `GCP_LOCATION`. For DRACO with Gemini 3.1 Pro Preview, use `BENCHMARK_RUBRIC_JUDGE_LLM_PROVIDER=vertex`, `BENCHMARK_RUBRIC_JUDGE_LLM_MODEL=gemini-3.1-pro-preview`, and `GCP_LOCATION=global`. `BENCHMARK_LLM_*` settings do not enable `weighted-rubric-v1` by fallback; rubric scoring uses only the dedicated `BENCHMARK_RUBRIC_JUDGE_LLM_*` settings.

#### Provider credentials on the platform

Use `harnyx-miner-config` to manage provider API keys stored by the platform for your signing hotkey:

```bash
harnyx-miner-config --wallet-name <wallet> --hotkey-name <hotkey> --get
harnyx-miner-config --wallet-name <wallet> --hotkey-name <hotkey> --provider chutes --api-key <provider-api-key>
harnyx-miner-config --wallet-name <wallet> --hotkey-name <hotkey> --delete-provider chutes
```

Supported providers are `chutes`, `openrouter`, `ai_gateway`, `desearch`, and `parallel`.
Reads return only whether each provider credential exists and timestamps; raw API keys are never returned.
Active miner-task batch execution uses these stored credentials through platform tool proxy execution. Validators receive only short-lived platform-tool-proxy tokens for one batch artifact/task/validator attempt. Retry attempts receive fresh validator sessions and fresh tokens, while the platform still enforces each artifact snapshot's configured `task_retry_count`. Raw provider API keys stay inside the platform boundary.
When your artifact becomes the active champion and receives champion emission, the platform also uses those stored provider API keys to run benchmark suites for that champion artifact.

---

### Step 2: Write your agent

You submit **one UTF-8 Python source file** (<= 1,000,000 bytes / 1 MB). Validators will:

1. Stage it as `agent.py`
2. Load it via `runpy.run_path`
3. Call your `query` entrypoint with a strict `Query` JSON payload

If `./agent.py` does not exist yet, start with a minimal stub:

```python
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response


@entrypoint("query")
async def query(query: Query) -> Response:
    return Response(text=query.text)
```

Your script must define this entrypoint:

```python
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response

@entrypoint("query")
async def query(query: Query) -> Response:
    # ... call tools (search_web, llm_chat)
    return Response(text="...")
```

The `query` entrypoint must stay `async def`, accept exactly one parameter annotated as `Query`, and return `Response`. The parameter name itself does not matter.

The entrypoint also supports caller-selected structured output. When
`query.output_schema` is absent or `None`, return the existing text response.
When it is present, return the JSON value directly in `Response.output`:

```python
from harnyx_miner_sdk.query import CitationRef


@entrypoint("query")
async def query(query: Query) -> Response:
    refs = [CitationRef(receipt_id="receipt-123", result_id="result-abc")]
    if query.output_schema is None:
        return Response(text="legacy answer", citations=refs)

    return Response(
        output={"summary": "structured answer"},
        citations=refs,
    )
```

Return exactly one of `text` or `output`. Top-level `None` is treated as a
missing answer; nested nulls remain ordinary JSON values when the caller's
schema allows them. Citation refs remain response-level siblings rather than
fields inside the caller's schema. Invalid structured output is rejected rather
than converted to text. See the
[miner SDK query contract](../packages/miner-sdk/README.md#query-contract) for
the Draft 2020-12 restrictions and 80,000-character compact JSON limits.

`Response.citations` is optional at the schema level, but for miner quality it should be treated as required whenever your answer makes non-obvious factual claims or depends on tool/search evidence. Answers without citations only make sense when the answer is obvious enough that no external support is reasonably needed. Facts presented without citations can be dismissed by the judge when they are material to the response. When present, `Response.citations` is capped at 200 refs; if you return more than 200, the response is invalid. `Response.text` is capped at 80,000 characters.

When citations are present, validators hydrate them into shared citations shaped like
`{url, title?, note?}` before scoring and monitoring. Hydrated citation notes are materialized by the validator from the referenced result's `note` text. A ref without slices materializes the full result note. A ref with `slices` materializes only those offsets. Across an answer, validators materialize at most 400 evidence segments and 120,000 source-text characters.

When your answer depends on a tool result that should be carried forward into scoring or monitoring, return receipt refs rather than freeform URLs:

```python
from harnyx_miner_sdk.query import CitationRef, Query, Response


@entrypoint("query")
async def query(query: Query) -> Response:
    return Response(
        text="...",
        citations=[CitationRef(receipt_id="receipt-123", result_id="result-abc")],
    )
```

Keep citations targeted at the claims your argument materially depends on. They are not a citation-count contest, and answers for obvious questions can still omit them.

#### How to extract `receipt_id` and `result_id`

Hosted tools return a tool-call envelope plus referenceable results. You need both pieces:

- `receipt_id`: the tool call
- `result_id`: the specific result that supports the claim

Example with `search_web`:

```python
from harnyx_miner_sdk.api import search_web
from harnyx_miner_sdk.query import CitationRef, CitationSlice, Query, Response


@entrypoint("query")
async def query(query: Query) -> Response:
    search = await search_web(query.text, provider="parallel", num=5)
    result = search.results[0]
    return Response(
        text=f"{result.title}: {result.note}",
        citations=[
            CitationRef(
                receipt_id=search.receipt_id,
                result_id=result.result_id,
            )
        ],
    )
```

The fields to read are:

```python
search.receipt_id
search.results[i].result_id
search.results[i].url
search.results[i].title
search.results[i].note
```

Workflow:

1. Call a hosted tool.
2. Pick the result that actually supports the claim you are making.
3. Use the tool call's `receipt_id`.
4. Use that supporting result's `result_id`.
5. Return only the targeted supporting refs in `Response.citations`, keeping the list at 200 or fewer.

Do not cite every tool result you saw. Cite only the specific results that carry the load-bearing facts in your answer. Prefer cited results whose `note` text already contains the factoid or excerpt your answer depends on. Use `CitationRef(receipt_id=..., result_id=...)` when the whole result is relevant. Use `CitationRef(receipt_id=..., result_id=..., slices=[CitationSlice(start=..., end=...)])` when only a narrower excerpt should be carried into scoring. Irrelevant citations do not help, and citation spam makes the response worse.

For large result notes, use `CitationSlice` offsets into the unstripped `note` text:

```python
CitationRef(
    receipt_id=search.receipt_id,
    result_id=result.result_id,
    slices=[CitationSlice(start=0, end=180)],
)
```

#### Tools and budgeting

Miner evaluations run under a per-session budget, and that budget **may vary between evaluations** — don’t assume a fixed value.

Tool calls return a budget snapshot:
- `session_budget_usd`
- `session_hard_limit_usd`
- `session_used_budget_usd`
- `session_remaining_budget_usd`

`session_budget_usd` is the communicated budget for the evaluation. `session_hard_limit_usd` is the actual enforcement ceiling for the session. `session_remaining_budget_usd` is clamped at `0` once usage exceeds the communicated budget, even if the hard limit is still higher.

For miner-task batch evaluation, the run is strict: if execution hits the hard limit, validators record the run as `session_budget_exhausted` and stop before scoring/finalization. Return a best-effort `Response` before that point if you can.

Tool calls are also concurrency-limited per evaluation session. For one session/token, validators allow up to 20 in-flight tool calls total across the registered miner tools. The cap is shared across the whole miner script session; it is not split by tool type, provider, or `llm_chat` model. If your agent starts another call after the shared cap is full, that extra call waits for a free slot instead of failing immediately.

Treat that limit as a runtime constraint, not a free queue. Waiting calls still consume wall-clock time, and they can still fail later if the session budget is exhausted or the upstream tool call fails.

You can call `tooling_info` (free) to fetch pricing metadata for available tools and provider/model pairs:

```python
from harnyx_miner_sdk.api import tooling_info

info = await tooling_info()
budget = info.budget
provider_models = info.response["allowed_llm_provider_models"]
embedding_provider_models = info.response["allowed_embedding_provider_models"]
pricing = info.response["pricing"]
```

Treat `allowed_llm_provider_models[provider]` as the runtime source of truth for `llm_chat` model ids instead of hardcoding a fixed list in your miner. Model ids are provider-specific: pass the id exactly as listed for the selected provider.

Treat `allowed_embedding_provider_models[provider]` the same way for `embed_text`. The current miner-facing embedding models are provider-specific Qwen3 Embedding 8B ids, and their pricing is listed under `pricing["embed_text"]["provider_models"]`.

Current allowed `llm_chat` provider/model ids in this repo:

| Provider | Model ids |
|----------|-----------|
| `chutes` | `deepseek-ai/DeepSeek-V3.2-TEE`, `zai-org/GLM-5-TEE`, `Qwen/Qwen3.6-27B-TEE`, `google/gemma-4-31B-turbo-TEE` |
| `openrouter` | `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `deepseek/deepseek-v3.2`, `z-ai/glm-5`, `qwen/qwen3.6-27b`, `google/gemma-4-31b-it` |
| `ai_gateway` | `thinkingmachines/inkling`, `zai/glm-5.2-fast`, `openai/gpt-oss-20b`, `zai/glm-4.7`, `google/gemma-4-31b-it`, `openai/gpt-oss-120b`, `minimax/minimax-m2.7`, `zai/glm-4.7-flash` |

`tooling_info().response["pricing"]["llm_chat"]["provider_models"]` exposes representative static rates for each provider/model pair. For OpenRouter and AI Gateway, those are reference prices for budgeting and fallback settlement; actual provider-returned cost wins when the provider returns one.

### Run a function tool loop

Define functions in `tools`, let the model request them, run them in your miner, then replay the assistant message and linked results. Keep the real provider call ID. `assistant.to_input_message()` also preserves any ordered `reasoning_details` returned by the provider.

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

`tool_choice` accepts `"none"`, `"auto"`, `"required"`, or one declared function. Function definitions can include `description`, recursive JSON Schema `parameters`, and `strict`. If the model requests parallel calls, return one contiguous tool-result message per call before adding another user or assistant message; the results may be in any order.

These controls pass through to the selected provider and model. Provider support can vary, and upstream rejections remain visible as tool failures. Use `max_output_tokens` in new code. The SDK still accepts `max_tokens` as a compatibility alias and normalizes it to `max_output_tokens`; setting both is rejected. Miner requests with `include` or `response_format` are rejected. Internal evaluation postprocessing is not part of the miner `llm_chat` API.

Current allowed `embed_text` provider/model ids in this repo:

| Provider | Model ids |
|----------|-----------|
| `chutes` | `Qwen/Qwen3-Embedding-8B-TEE` |
| `openrouter` | `qwen/qwen3-embedding-8b` |

Use `input_type="query"` for query or instruction-style embeddings and `input_type="document"` for document embeddings:

```python
from harnyx_miner_sdk.api import embed_text

query_embedding = await embed_text(
    query.text,
    provider="openrouter",
    model="qwen/qwen3-embedding-8b",
    input_type="query",
)
document_embeddings = await embed_text(
    ["First passage text.", "Second passage text."],
    provider="openrouter",
    model="qwen/qwen3-embedding-8b",
    input_type="document",
)
```

Query embeddings use Qwen's retrieval instruction by default and accept an optional `instruction` override. Document embeddings are sent as raw text and reject `instruction`. Embedding outputs are not citation sources; keep using `search_web`, `search_ai`, or `fetch_page` for cited evidence.

Use `provider_extra` only for selected-provider-specific request additions that do not already have common tool parameters. The schema is selected by the sibling `provider` value and is strict. OpenRouter supports provider selection for both `llm_chat` and `embed_text`:

```python
response = await llm_chat(
    provider="openrouter",
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Reply with only ok."}],
    provider_extra={"provider": {"only": ["cerebras"]}},
)

query_embedding = await embed_text(
    "What is Harnyx?",
    provider="openrouter",
    model="qwen/qwen3-embedding-8b",
    input_type="query",
    provider_extra={"provider": {"only": ["nebius"]}},
)
```

OpenRouter also accepts an optional `provider.allow_fallbacks` boolean. Omit it to use OpenRouter's default fallback behavior; set it only when your miner needs to explicitly choose whether OpenRouter may fall back to another hosted provider after the selected provider fails. You can pass it with `provider.only`, or by itself as `provider_extra={"provider": {"allow_fallbacks": False}}`.

AI Gateway uses `providerOptions.gateway` for request-level upstream provider selection, for example Cerebras through AI Gateway:

```python
await llm_chat(
    provider="ai_gateway",
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Reply with only ok."}],
    provider_extra={"providerOptions": {"gateway": {"only": ["cerebras"]}}},
)
```

The SDK still accepts the legacy top-level `provider.only` input and normalizes it to `providerOptions.gateway.only` before invoking AI Gateway. New miner code should use the canonical `providerOptions.gateway` form shown above.

Do not pass `provider_extra={"provider": "cerebras"}`. Vercel expects provider routing options as objects, and the SDK/runtime rejects the raw string form.

**Gemma 4 reasoning through AI Gateway pinned to Cerebras requires an explicit typed effort.** For example, use `thinking={"enabled": True, "effort": "medium"}` with `google/gemma-4-31b-it`; `low` and `high` are also supported. The AI Gateway adapter translates that common control to Cerebras's provider-specific `reasoningEffort`, and the response exposes `reasoning`, `reasoning_details`, and positive `reasoning_tokens`. Keep reasoning controls in `thinking`; do not pass `providerOptions.cerebras` through `provider_extra`.

Do not put common behavior in `provider_extra`. For example, reasoning controls belong in `thinking` even when a provider's raw API spells them differently. Chutes raw reasoning options are handled by `thinking`, not `provider_extra`. Other OpenRouter provider-preference fields such as `order`, `require_parameters`, `ignore`, `quantizations`, `sort`, and `max_price` are not supported here.

You can request model thinking/reasoning through the typed `thinking` option on `llm_chat`.
Omit it when you want the validator/provider default behavior.

Thinking controls are provider/model specific:

| Provider | Model | `enabled=True` / `enabled=False` | `effort` | `budget` |
|----------|-------|----------------------------------|----------|----------|
| `openrouter` | `openai/gpt-oss-20b` | Supported via OpenRouter `reasoning.enabled` / `reasoning.effort="none"` | Supported via OpenRouter `reasoning.effort` | Supported via OpenRouter `reasoning.max_tokens` |
| `openrouter` | `openai/gpt-oss-120b` | Supported via OpenRouter `reasoning.enabled` / `reasoning.effort="none"` | Supported via OpenRouter `reasoning.effort` | Supported via OpenRouter `reasoning.max_tokens` |
| `openrouter` | `deepseek/deepseek-v3.2`, `z-ai/glm-5`, `qwen/qwen3.6-27b`, `google/gemma-4-31b-it` | Supported via OpenRouter `reasoning.enabled` / `reasoning.effort="none"` | Supported via OpenRouter `reasoning.effort` | Supported via OpenRouter `reasoning.max_tokens` |
| `ai_gateway` | Allowed AI Gateway models except `google/gemma-4-31b-it` pinned to Cerebras | Supported via AI Gateway `reasoning.enabled` / `reasoning.effort="none"` | Supported via AI Gateway `reasoning.effort` | Supported via AI Gateway `reasoning.max_tokens` |
| `ai_gateway` | `google/gemma-4-31b-it` pinned to Cerebras | Enable by supplying an explicit `effort`; disabling uses Gemma's disabled provider default | Supported via Cerebras `reasoningEffort` | Unsupported for this route; not serialized into a Cerebras provider option |
| `chutes` | `deepseek-ai/DeepSeek-V3.2-TEE` | Supported via `chat_template_kwargs.thinking` | Unsupported for Chutes; not serialized | Unsupported for Chutes; not serialized |
| `chutes` | `zai-org/GLM-5-TEE` | Supported via `chat_template_kwargs.enable_thinking` | Unsupported for Chutes; not serialized | Unsupported for Chutes; not serialized |
| `chutes` | `Qwen/Qwen3.6-27B-TEE`, `google/gemma-4-31B-turbo-TEE` | Supported via `chat_template_kwargs.enable_thinking` | Unsupported for Chutes; not serialized | Unsupported for Chutes; not serialized |

```python
from harnyx_miner_sdk.api import llm_chat

response = await llm_chat(
    provider="chutes",
    model="deepseek-ai/DeepSeek-V3.2-TEE",
    messages=[{"role": "user", "content": "Solve 17 * 23. Return only the answer."}],
    temperature=0.0,
    thinking={"enabled": True},
)
```

To explicitly disable thinking where the provider/model supports a disable control:

```python
response = await llm_chat(
    provider="chutes",
    model="zai-org/GLM-5-TEE",
    messages=[{"role": "user", "content": "Reply with only ok."}],
    temperature=0.0,
    thinking={"enabled": False},
)
```

`effort` (`"low"`, `"medium"`, `"high"`) and `budget` are supported when the selected provider/model uses OpenRouter or AI Gateway reasoning controls. Gemma 4 pinned to Cerebras requires an explicit `effort` and does not support `budget`. The two fields cannot be sent together, and invalid scalar values are rejected; for example, `"false"` is not accepted as a boolean. Thinking controls are best effort across providers: if the selected model/provider has no verified control, the request still runs and unsupported hints are not serialized into guessed provider fields.

Core subnet-facing tools today:
- `search_web`: web search results; pass `timeout=<seconds>` to bound the full search call
- `search_ai`: AI search results; pass `timeout=<seconds>` to bound the full AI search call
- `fetch_page`: fetched page content; pass `timeout=<seconds>` to bound slow page fetches
- `llm_chat`: hosted LLM chat; pass `timeout=<seconds>` to bound the full hosted chat call
- `embed_text`: hosted text embeddings; pass `provider`, `model`, `input_type="query"` or `"document"`, optional query-only `instruction`, `dimensions`, and `timeout=<seconds>`
- `tooling_info`: available tool names/models/pricing metadata; accepts `timeout` for call-surface consistency
- `test_tool`: invocation sanity check; accepts `timeout` for call-surface consistency and is not used in subnet evaluation

`llm_chat` does not automatically retry provider attempts. If a miner wants retry behavior, the script should catch the tool failure and call `llm_chat` again explicitly; each call has its own receipt, cost, and budget accounting.

Pricing for all tools is described by `tooling_info.response["pricing"]`, including the settlement order used when provider-returned cost is unavailable.

Repository-grounding tools exist elsewhere in the monorepo for content-review flows, but they are not part of the subnet-facing miner workflow.

**Reference implementation:** [`tests/docker_sandbox_entrypoint.py`](tests/docker_sandbox_entrypoint.py)

---

### Step 3: Run local batch evaluation

Use `harnyx-miner-local-eval` to benchmark your local artifact against a completed public miner-task batch before you submit.

Run it:

```bash
uv run --package harnyx-miner harnyx-miner-local-eval --agent-path ./agent.py
```

By default it selects the latest completed public batch and runs `vs-champion`. It also supports `target-only`, specific `--batch-id` selection, and writes JSON + Markdown reports you can use for your improvement loop.

See [`local-eval.md`](local-eval.md) for prerequisites, modes, reports, and the full local-eval workflow. If you are using a code agent, the public step-based skills in [`skills/README.md`](skills/README.md) can help structure that loop.

To run an open benchmark suite against a pinned source batch, list the available
suites and then choose one explicitly:

```bash
uv run --package harnyx-miner harnyx-miner-local-benchmark --list-suites
```

```bash
uv run --package harnyx-miner harnyx-miner-local-benchmark \
  --suite webwalkerqa \
  --agent-path ./agent.py \
  --source-batch-id <completed-batch-id>
```

This writes a structured JSON report with the open benchmark question, reference answer or rubric payload, generated answer, score, optional `score_detail`, cost, runtime, and errors for each item.
The local benchmark uses the public `harnyx_commons.miner_task_benchmark` boundary for packaged benchmark data, deterministic benchmark IDs, scoring, sampling, metric aggregation, and scoring-version checks. Miners must pass `--suite` so the report names the exact benchmark suite and version under test; there is no default suite. Current-suite listing shows active public benchmark suites, including DRACO. DRACO local runs require dedicated `BENCHMARK_RUBRIC_JUDGE_LLM_*` settings and use the current weighted-rubric snapshot when dataset/scoring flags are omitted. The command does not depend on private platform internals.

For upstream-style autonomous experimentation, see [`AUTO-RESEARCH.md`](AUTO-RESEARCH.md). That guide gives the operator prompt, environment checklist, setup commands, and safety boundaries. The agent-facing research policy lives in [`program.md`](program.md).

That flow keeps the surface intentionally small:

- `prepare.py` pins the local-eval batch plus the explicitly selected benchmark snapshot and owns fixed support.
- `train.py` is the only file the agent edits and the command run for each experiment.
- `results.tsv` records Score A, Score B, cost, and status for each experiment and stays untracked.

---

### Step 4: Submit to the platform

Set the platform base URL:

```bash
export PLATFORM_BASE_URL="https://api.harnyx.ai"
```

Upload your agent with your registered hotkey:

```bash
uv run --package harnyx-miner harnyx-miner-submit \
  --agent-path ./agent.py \
  --wallet-name <wallet-name> \
  --hotkey-name <hotkey-name>
```

**What happens:**

- Calls `POST ${PLATFORM_BASE_URL}/v1/miners/scripts`
- Payload: `{ "script_b64": "...", "sha256": "..." }`
- Success response includes `content_hash`
- Signed with: `Authorization: Bittensor ss58="…",sig="…"`
- Signature is over: `METHOD + "\n" + PATH_QS + "\n" + sha256(body_bytes)`

Verify the returned hash against your local file:

```bash
uv run --package harnyx-miner harnyx-miner-hash --agent-path ./agent.py
```

That command computes the same SHA-256 the platform validates for upload, so it should
match the response field `content_hash`.

After upload, use the [Mining Runbook](mining-runbook.md) to verify platform
acceptance, check batch eligibility, inspect completed results, and decide the
next improvement step.

## Miner Script Python Subset

Server upload validates miner scripts with an AST policy before duplicate
fingerprinting. Normal SDK imports, common stdlib imports, async helpers,
loops, lambdas, comprehensions, f-strings, dataclass-style classes,
`json.dumps(...)`, `re.compile(...)`, and ordinary bound method calls are
supported.

Accepted scripts are also hashed structurally to catch exact copies, formatting
or comment changes, common helper/local/import alias renames, and unused inert
top-level helper padding. This is duplicate-submission hardening, not a proof
that two arbitrary Python programs are semantically equivalent.

The duplicate hash uses an allow-listed normalization policy:

| Transform class | Decision-hash behavior |
|-----------------|------------------------|
| Comments, whitespace, parser-erased literal spelling, and source locations | canonicalized |
| Common lexical renames, import alias order, inert helper padding/order, and stable local keyword names | canonicalized when the transform's mechanical assumptions are true |
| Keyword argument order, arbitrary statement/import reordering, docstring removal, dead-code removal, constant folding, and semantic equivalence | not normalized in the duplicate-rejection hash |

`canonical_ast_hash_v1` is not semantic equivalence. It intentionally ignores
some syntactic and identifier-level differences for duplicate rejection. Scripts
that depend on reflection over helper order, local names, or parameter names are
not guaranteed to receive distinct hashes.

Review comments on the duplicate hash should distinguish unreliable binding
assumptions from semantic-equivalence requests. Binding ambiguity is guarded;
semantic equivalence is out of scope.

Use literal optional-field access:

```python
title = getattr(result, "title", None)
```

Direct builtin `eval(...)` calls are allowed when they improve answer accuracy.
The submitted `eval(...)` call is structurally hashed, but code generated at
runtime is not inspected by duplicate detection.

Do not alias or pass around `eval`, `getattr`, or `hasattr`, and do not use other
dynamic reflection or execution APIs such as `exec`, `compile`, `globals`,
`locals`, `vars`, `dir`, `type`, `help`, `__import__`, `importlib`, `inspect`,
`setattr`, `delattr`, or dunder attributes such as `__dict__`, `__code__`,
`__globals__`, `__mro__`, and `__subclasses__`. Reflection helpers such as
`typing.get_type_hints`, `dataclasses.fields`, and `dataclasses.asdict` are not
part of the accepted upload subset.

Pattern matching, `del`, `global`, and `nonlocal` are not part of the upload
subset. Keep class bodies to docstrings, pass statements, field assignments or
annotations, and method definitions.

---

## Common errors

### Authentication (401)

| Error | Cause |
|-------|-------|
| `missing_authorization` | No `Authorization` header |
| `invalid_signature` | Signature does not verify |
| `invalid_signature_hex` | Signature is not valid hex |
| `invalid_signature_length` | Signature has wrong length |

### Authorization (403)

| Error | Cause |
|-------|-------|
| `unknown_hotkey` | Hotkey is not registered on the subnet metagraph |

### Validation (4xx)

| Error | Cause |
|-------|-------|
| `sha_mismatch` (422) | Your `sha256` does not match the decoded `script_b64` |
| `invalid_script_payload` (422) | The script is not valid UTF-8/Python or uses unsupported dynamic/reflection syntax |
| `duplicate_script` (409) | The same script already exists globally |

### Runtime (during evaluation)

If your script fails during preload because of your own code, or violates the `query` contract, monitoring usually shows `script_validation_failed`. If `query` starts and your code crashes, it usually shows `miner_unhandled_exception`. `sandbox_invocation_failed` is reserved for validator-owned sandbox startup/preload faults.

Tool calls can fail transiently (timeouts / upstream errors). Treat them like external APIs: catch tool errors and still return a valid `Response` so you don’t crash the whole evaluation run.

For slow tools, pass a positive finite timeout such as `await search_web(query.text, provider="parallel", num=5, timeout=10.0)`, `await fetch_page(url, provider="parallel", timeout=10.0)`, `await llm_chat(provider="chutes", ..., timeout=20.0)`, or `await embed_text(texts, provider="openrouter", model="qwen/qwen3-embedding-8b", input_type="document", timeout=10.0)`, and catch the tool error so a single slow call does not consume the whole evaluation run.

During batch evaluation, failed attempts can retry only when your stored miner config has remaining `task_retry_count`. Each retry is a fresh validator session and fresh platform-tool-proxy token; a terminated attempt is kept as internal audit evidence, while public batch results still show one final task result per validator/artifact/task pair.

Validator-side provider evidence is aggregate and batch-scoped for monitoring and
debugging. Because miners choose their providers, provider errors do not by
themselves fail validator delivery or source batch health. Historical
`provider_batch_failure` rows may still appear for old batches, but active
validator runtime no longer emits that code.

```python
try:
    search = await search_web(query.text, provider="parallel", num=5)
except Exception:
    search = None

summary = "no search evidence"
if search and search.results:
    summary = search.results[0].title or search.results[0].url or "search evidence"

return Response(text=summary)
```
