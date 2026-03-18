# Miner Guide

This directory contains the miner-facing CLI tools for the Harnyx Subnet.

## How it fits together

```
  You (miner)
      │
      │  write agent.py
      │  (imports harnyx-miner-sdk)
      ▼
  ┌─────────────────────────────────┐
  │  miner/                         │  ◀── what you interact with
  │  • harnyx-miner-dev   (test)    │
  │  • harnyx-miner-submit (upload) │
  └─────────────────────────────────┘
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

- `miner/` — CLI tools you use directly (`harnyx-miner-dev`, `harnyx-miner-submit`)
- [`packages/miner-sdk/`](../packages/miner-sdk/README.md) — SDK your script imports; you don't need to read its docs
- `sandbox/` — runtime that validators use to execute your script; you don't need it

---

## Write → Test → Submit

### Step 1: Setup

From the repo root:

```bash
uv sync --all-packages --dev
```

Create a `.env` at the repo root (copy from `.env.example`) and fill:

| Variable | Purpose |
|----------|---------|
| `CHUTES_API_KEY` | LLM tool calls |
| `DESEARCH_API_KEY` | Search tool calls |
| `PLATFORM_BASE_URL` | Script uploads |

---

### Step 2: Write your agent

You submit **one UTF-8 Python source file** (≤ 256KB). Validators will:

1. Stage it as `agent.py`
2. Load it via `runpy.run_path`
3. Call your `query` entrypoint with a strict `Query` JSON payload

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

#### Tools and budgeting

Miner evaluations run under a per-session budget, and that budget **may vary between evaluations** — don’t assume a fixed value.

Tool calls return a budget snapshot:
- `session_budget_usd`
- `session_used_budget_usd`
- `session_remaining_budget_usd`

You can call `tooling_info` (free) to fetch pricing metadata for available tools/models:

```python
from harnyx_miner_sdk.api import tooling_info

info = await tooling_info()
budget = info.budget
pricing = info.response["pricing"]
```

Available tools:
- `search_web`: web search results
- `search_x`: X/Twitter search results
- `search_ai`: AI search results
- `search_repo`: repository search results (`excerpt` evidence snippets)
- `get_repo_file`: repository file read results (`text` + optional `excerpt`)
- `llm_chat`: hosted LLM chat
- `tooling_info`: available tool names/models/pricing metadata

Current subnet miner-task batches do not inject feed-specific context into the `query` contract.
- `test_tool`: tool invocation sanity check (testing only)

Pricing for all tools is read from `tooling_info.response["pricing"]`.

Repo-tool usage pattern:
- use `search_repo` to discover candidate evidence with ranked excerpts
- use `get_repo_file` to fetch full file text or line ranges for verification

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
- Signed with: `Authorization: Bittensor ss58="…",sig="…"`
- Signature is over: `METHOD + "\n" + PATH_QS + "\n" + sha256(body_bytes)`

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
| `duplicate_script` (409) | The same script content hash already exists globally |

### Runtime (during evaluation)

If your script raises at import time or runtime, the evaluation fails. In subnet monitoring, this can show up as `sandbox_invocation_failed`.

Tool calls can fail transiently (timeouts / upstream errors). Treat them like external APIs: catch tool errors and still return a valid `Response` so you don’t crash the whole evaluation run.

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
