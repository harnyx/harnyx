# caster-miner

Miner-facing utilities for the Caster Subnet.

This package is for **miner script authors**:
- Write a single Python “agent” script that implements `evaluate_criterion`.
- Test it locally the same way validators will run it.
- Submit the script to the platform API (signed by your subnet-registered hotkey).

## What miners submit

Miners submit **one UTF-8 Python source file** (“agent script”). Validators will:
- stage it as `agent.py`
- load it via `CASTER_AGENT_PATH` (using `runpy.run_path`)
- call the `evaluate_criterion` entrypoint with a JSON payload

Practical constraints:
- Your entrypoint must be `async def` and accept exactly one parameter named `request`.
- Keep scripts small; validators enforce a hard size limit (`<= 256KB`).

## Setup

From the repo root:

```bash
uv sync --all-packages --dev
```

Create a `.env` at the repo root (copy from `.env.example`) and fill:
- `CHUTES_API_KEY`
- `DESEARCH_API_KEY`
- `PLATFORM_BASE_URL` (for uploads)

## Write an agent (entrypoint contract)

Entrypoints are registered with `caster-miner-sdk`:

```python
from caster_miner_sdk.decorators import entrypoint
from caster_miner_sdk.criterion_evaluation import CriterionEvaluationRequest


@entrypoint("evaluate_criterion")
async def evaluate_criterion(request: object) -> dict[str, object]:
    payload = CriterionEvaluationRequest.model_validate(request)

    # Your logic here: call tools (search_web, llm_chat), decide verdict, cite evidence.
    return {
        "verdict": 1,
        "justification": "…",
        "citations": [
            {
                "url": "https://example.com",
                "note": "evidence summary",
                "receipt_id": "tool-receipt-id",
                "result_id": "search-result-id",
            }
        ],
    }
```

Full working reference agent:
- `public/miner/tests/docker_sandbox_entrypoint.py`

## Local testing (recommended)

### 1) Run the agent locally (with real tool calls)

`caster-miner-dev` loads your file, finds `evaluate_criterion`, and runs it with a `CriterionEvaluationRequest`.

It uses the in-process tool host, so you need tool credentials:
- `DESEARCH_API_KEY`
- `CHUTES_API_KEY`

```bash
caster-miner-dev --agent-path ./agent.py
```

You can also pass a full request payload:

```bash
caster-miner-dev --agent-path ./agent.py --request-json ./request.json
```

## Submit to the platform (signed by your hotkey)

Set the platform base URL (no hardcoded default; env or `.env`):

```bash
export PLATFORM_BASE_URL="http://localhost:8200"
```

Upload your agent with your registered hotkey:

```bash
caster-miner-submit \
  --agent-path ./agent.py \
  --wallet-name <wallet-name> \
  --hotkey-name <hotkey-name>
```

This calls `POST ${PLATFORM_BASE_URL}/v1/miners/scripts` with:
- JSON payload `{ "script_b64": "...", "sha256": "..." }`
- `Authorization: Bittensor ss58="…",sig="…"` where the signature is over the canonical request:
  `METHOD + "\n" + PATH_QS + "\n" + sha256(body_bytes)`

### Common errors

- `missing_authorization` (401): no `Authorization` header.
- `unknown_hotkey` (403): hotkey is not registered on the subnet metagraph.
- `invalid_signature` / `invalid_signature_hex` / `invalid_signature_length` (401): signature does not verify.
- `sha_mismatch` (422): your `sha256` does not match the decoded `script_b64`.
- `duplicate_script` (409): the same script content hash already exists globally.
