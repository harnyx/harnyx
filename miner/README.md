# Miner Guide

This directory contains the miner-facing CLI tools for the Harnyx Subnet.

## How it fits together

```
  You (miner)
      │
      │  write agent.py
      │  (imports harnyx-miner-sdk)
      ▼
  ┌──────────────────────────────────────────┐
  │  miner/                                  │  ◀── what you interact with
  │  • harnyx-miner-dev         (test)       │
  │  • harnyx-miner-local-eval  (batch eval) │
  │  • harnyx-miner-local-benchmark          │
  │  • harnyx-miner-submit      (upload)     │
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

- `miner/` — CLI tools you use directly (`harnyx-miner-dev`, `harnyx-miner-local-eval`, `harnyx-miner-submit`)
- [`packages/miner-sdk/`](../packages/miner-sdk/README.md) — SDK your script imports; you don't need to read its docs first
- `sandbox/` — runtime that validators use to execute your script; you don't run it directly

---

## Write → Test → Local Eval → Submit

### Step 1: Setup

From the repo root:

```bash
uv sync --all-packages --dev
```

Create a `.env` at the repo root (copy from `.env.example`) and fill:

| Variable | Purpose |
|----------|---------|
| `CHUTES_API_KEY` | Evaluation scoring and `llm_chat` tool calls |
| `OPENROUTER_API_KEY` | Optional: required when local tooling invokes Chutes-selected tool models that route through OpenRouter, including `gpt-oss` and Qwen when no explicit custom route override handles Qwen |
| `DESEARCH_API_KEY` | Optional: required if your agent uses search tools |
| `SEARCH_PROVIDER` | Optional: required if your agent uses search tools |
| `PLATFORM_BASE_URL` | Public monitoring and script uploads |
| `BENCHMARK_LLM_PROVIDER` | Optional benchmark judge provider; defaults to `chutes` |
| `BENCHMARK_LLM_MODEL` | Required when running the DeepSearchQA local benchmark |

The checked-in default is `SEARCH_PROVIDER=desearch`. If you need a fallback search provider, miner tooling also supports `parallel`; set `SEARCH_PROVIDER=parallel` and `PARALLEL_API_KEY`.
If you set `BENCHMARK_LLM_PROVIDER=vertex`, also configure Vertex credentials such as `GCP_PROJECT_ID` and `GCP_LOCATION`.

---

### Step 2: Write your agent

You submit **one UTF-8 Python source file** (≤ 256KB). Validators will:

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
    search = await search_web(query.text, num=5)
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

Tool calls are also concurrency-limited per evaluation session. For one session/token, `deepseek-ai/DeepSeek-V3.2-TEE` and `zai-org/GLM-5-TEE` each allow 1 in-flight `llm_chat` call, other `llm_chat` models allow up to 2 in-flight calls per model, and non-LLM tools allow up to 5 in-flight calls together. If your agent starts another call in a lane that is already full, that extra call waits for a free slot instead of failing immediately.

Treat that limit as a runtime constraint, not a free queue. Waiting calls still consume wall-clock time, and they can still fail later if the session budget is exhausted or the upstream tool call fails.

You can call `tooling_info` (free) to fetch pricing metadata for available tools/models:

```python
from harnyx_miner_sdk.api import tooling_info

info = await tooling_info()
budget = info.budget
allowed_models = info.response["allowed_tool_models"]
pricing = info.response["pricing"]
```

Treat `allowed_tool_models` as the runtime source of truth for `llm_chat` model ids instead of hardcoding a fixed list in your miner.

Current allowed `llm_chat` model ids in this repo:
- `openai/gpt-oss-20b`
- `openai/gpt-oss-120b`
- `deepseek-ai/DeepSeek-V3.2-TEE`
- `zai-org/GLM-5-TEE`
- `Qwen/Qwen3.6-27B-TEE`
- `google/gemma-4-31B-turbo-TEE`

You can request model thinking/reasoning through the typed `thinking` option on `llm_chat`.
Omit it when you want the validator/provider default behavior.

Thinking controls are provider/model specific:

| Model | `enabled=True` / `enabled=False` | `effort` | `budget` |
|-------|----------------------------------|----------|----------|
| `openai/gpt-oss-20b` | Supported via OpenRouter `reasoning.enabled` / `reasoning.effort="none"` when routed through OpenRouter | Supported via OpenRouter `reasoning.effort` | Supported via OpenRouter `reasoning.max_tokens` |
| `openai/gpt-oss-120b` | Supported via OpenRouter `reasoning.enabled` / `reasoning.effort="none"` when routed through OpenRouter | Supported via OpenRouter `reasoning.effort` | Supported via OpenRouter `reasoning.max_tokens` |
| `deepseek-ai/DeepSeek-V3.2-TEE` | Supported via `chat_template_kwargs.thinking` | No verified knob; ignored | No verified knob; ignored |
| `zai-org/GLM-5-TEE` | Supported via `chat_template_kwargs.enable_thinking` | No verified knob; ignored | No verified knob; ignored |
| `Qwen/Qwen3.6-27B-TEE` | Custom route: `chat_template_kwargs.enable_thinking`; OpenRouter route: `reasoning.enabled` / `reasoning.effort="none"` | OpenRouter route: `reasoning.effort`; custom route: no verified knob | OpenRouter route: `reasoning.max_tokens`; custom route: no verified knob |
| `google/gemma-4-31B-turbo-TEE` | Supported via `chat_template_kwargs.enable_thinking` when routed through the custom OpenAI-compatible Gemma endpoint | No verified knob; ignored | No verified knob; ignored |

```python
from harnyx_miner_sdk.api import llm_chat

response = await llm_chat(
    model="deepseek-ai/DeepSeek-V3.2-TEE",
    messages=[{"role": "user", "content": "Solve 17 * 23. Return only the answer."}],
    temperature=0.0,
    thinking={"enabled": True},
)
```

To explicitly disable thinking where the provider/model supports a disable control:

```python
response = await llm_chat(
    model="zai-org/GLM-5-TEE",
    messages=[{"role": "user", "content": "Reply with only ok."}],
    temperature=0.0,
    thinking={"enabled": False},
)
```

`effort` (`"low"`, `"medium"`, `"high"`) and `budget` are supported for OpenRouter-backed models through OpenRouter reasoning controls. They cannot be sent together, and invalid scalar values are rejected; for example, `"false"` is not accepted as a boolean. Thinking controls are best effort across providers: if the selected model/provider has no verified control, the request still runs and unsupported hints are ignored rather than translated into guessed provider fields.

Core subnet-facing tools today:
- `search_web`: web search results; pass `timeout=<seconds>` to bound the full search call
- `search_ai`: AI search results; pass `timeout=<seconds>` to bound the full AI search call
- `fetch_page`: fetched page content; pass `timeout=<seconds>` to bound slow page fetches
- `llm_chat`: hosted LLM chat; pass `timeout=<seconds>` to bound the full hosted chat call
- `tooling_info`: available tool names/models/pricing metadata; accepts `timeout` for call-surface consistency
- `test_tool`: invocation sanity check; accepts `timeout` for call-surface consistency and is not used in subnet evaluation

Pricing for all tools is read from `tooling_info.response["pricing"]`.

Repository-grounding tools exist elsewhere in the monorepo for content-review flows, but they are not part of the subnet-facing miner workflow.

**Reference implementation:** [`tests/docker_sandbox_entrypoint.py`](tests/docker_sandbox_entrypoint.py)

---

### Step 3: Test locally

`harnyx-miner-dev` loads your file, finds `query`, and runs it with a `Query` payload. It uses real tool calls, so you need the API keys configured above.

```bash
uv run --package harnyx-miner harnyx-miner-dev --agent-path ./agent.py
```

To test with a specific request payload:

```bash
uv run --package harnyx-miner harnyx-miner-dev --agent-path ./agent.py --request-json ./request.json
```

---

### Step 4: Run local batch evaluation

Use `harnyx-miner-local-eval` to benchmark your local artifact against a completed public miner-task batch before you submit.

Run it:

```bash
uv run --package harnyx-miner harnyx-miner-local-eval --agent-path ./agent.py
```

By default it selects the latest completed public batch and runs `vs-champion`. It also supports `target-only`, specific `--batch-id` selection, and writes JSON + Markdown reports you can use for your improvement loop.

See [`local-eval.md`](local-eval.md) for prerequisites, modes, reports, and the full local-eval workflow. If you are using a code agent, the public step-based skills in [`skills/README.md`](skills/README.md) can help structure that loop.

To run the open DeepSearchQA benchmark against a pinned source batch, use:

```bash
uv run --package harnyx-miner harnyx-miner-local-benchmark \
  --agent-path ./agent.py \
  --source-batch-id <completed-batch-id>
```

This writes a structured JSON report with the open benchmark question, reference answer, generated answer, binary correctness result, cost, runtime, and errors for each item.
The local benchmark uses the public `harnyx_commons.miner_task_benchmark` boundary for packaged DeepSearchQA data, deterministic benchmark IDs, scoring, sampling, metric aggregation, and scoring-version checks. It does not depend on private platform internals.

For upstream-style autonomous experimentation, see [`AUTO-RESEARCH.md`](AUTO-RESEARCH.md). That guide gives the operator prompt, environment checklist, setup commands, and safety boundaries. The agent-facing research policy lives in [`program.md`](program.md).

That flow keeps the surface intentionally small:

- `prepare.py` pins the local-eval batch plus DeepSearchQA benchmark snapshot and owns fixed support.
- `train.py` is the only file the agent edits and the command run for each experiment.
- `results.tsv` records Score A, Score B, cost, and status for each experiment and stays untracked.

---

### Step 5: Submit to the platform

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
| `duplicate_script` (409) | The same script already exists globally |

### Runtime (during evaluation)

If your script fails during preload because of your own code, or violates the `query` contract, monitoring usually shows `script_validation_failed`. If `query` starts and your code crashes, it usually shows `miner_unhandled_exception`. `sandbox_invocation_failed` is reserved for validator-owned sandbox startup/preload faults.

Tool calls can fail transiently (timeouts / upstream errors). Treat them like external APIs: catch tool errors and still return a valid `Response` so you don’t crash the whole evaluation run.

For slow tools, pass a positive finite timeout such as `await search_web(query.text, num=5, timeout=10.0)`, `await fetch_page(url, timeout=10.0)`, or `await llm_chat(..., timeout=20.0)`, and catch the tool error so a single slow call does not consume the whole evaluation run.

Validator-side provider attribution is now aggregate and batch-scoped. One failed
`search_web` / `search_ai` / `fetch_page` / `llm_chat` call does not by itself
make the validator blame the provider. The validator only escalates to
`provider_batch_failure` when the same provider/model crosses the batch-level
threshold in one batch:
- at least 10 total calls
- more than 95% failed calls

```python
try:
    search = await search_web(query.text, num=5)
except Exception:
    search = None

summary = "no search evidence"
if search and search.results:
    summary = search.results[0].title or search.results[0].url or "search evidence"

return Response(text=summary)
```
