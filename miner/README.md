# Miner Guide

This directory contains the miner-facing CLI tools for the Caster Subnet.

## How it fits together

```
  You (miner)
      │
      │  write agent.py
      │  (imports caster-miner-sdk)
      ▼
  ┌─────────────────────────────────┐
  │  miner/                         │  ◀── what you interact with
  │  • caster-miner-dev   (test)    │
  │  • caster-miner-submit (upload) │
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

- `miner/` — CLI tools you use directly (`caster-miner-dev`, `caster-miner-submit`)
- `packages/miner-sdk/` — SDK your script imports; you don't need to read its docs
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
3. Call your `evaluate_criterion` entrypoint with a JSON payload

Your script must define this entrypoint:

```python
from caster_miner_sdk.decorators import entrypoint
from caster_miner_sdk.criterion_evaluation import CriterionEvaluationRequest

@entrypoint("evaluate_criterion")
async def evaluate_criterion(request: object) -> dict[str, object]:
    payload = CriterionEvaluationRequest.model_validate(request)
    # ... call tools (search_web, llm_chat), decide verdict, cite evidence
    return {"verdict": 1, "justification": "...", "citations": [...]}
```

**Reference implementation:** [`tests/docker_sandbox_entrypoint.py`](tests/docker_sandbox_entrypoint.py)

---

### Step 3: Test locally

`caster-miner-dev` loads your file, finds `evaluate_criterion`, and runs it with a `CriterionEvaluationRequest`. It uses real tool calls, so you need the API keys configured above.

```bash
caster-miner-dev --agent-path ./agent.py
```

To test with a specific request payload:

```bash
caster-miner-dev --agent-path ./agent.py --request-json ./request.json
```

---

### Step 4: Submit to the platform

Set the platform base URL:

```bash
export PLATFORM_BASE_URL="https://api.castersubnet.example"
```

Upload your agent with your registered hotkey:

```bash
caster-miner-submit \
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
