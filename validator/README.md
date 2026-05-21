# Validator Operator Runbook

This directory contains the validator runtime package plus an operator-ready Docker Compose stack (`validator` + `watchtower`).

## Prerequisites

Before starting, ensure you have:

1. **Docker** installed and running (Docker Compose v2+)
2. **A Bittensor wallet/hotkey** registered on the subnet metagraph
3. **A public endpoint** reachable by the platform (for registration + evaluation callbacks)
4. **API keys** for LLM and search tools (see env vars below)

### Hardware + networking (quick sizing)

- vCPU: 2 (4 recommended)
- RAM: 32 GiB
- Disk: 20 GB
- Network: platform must reach your validator on TCP 8100; set `VALIDATOR_PUBLIC_BASE_URL` accordingly
- Third-party APIs: Chutes (`CHUTES_API_KEY`) + DeSearch (`DESEARCH_API_KEY`), plus OpenRouter (`OPENROUTER_API_KEY`) if your validator serves Chutes-selected tool models that route through OpenRouter.

## Step 1: Create your env file

```bash
cp .env.example .env
```

## Step 2: Configure environment variables

Edit `.env` and set at least:

| Variable | Description |
|----------|-------------|
| `PLATFORM_BASE_URL` | Platform API endpoint (finney/mainnet: `https://api.harnyx.ai`, testnet: `https://api.staging.harnyx.ai`) |
| `VALIDATOR_PUBLIC_BASE_URL` | How the platform can reach your validator |
| `CHUTES_API_KEY` | API key for LLM calls |
| `OPENROUTER_API_KEY` | Optional API key for Chutes-selected OpenRouter-routed `llm_chat` models such as `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, and `Qwen/Qwen3.6-27B-TEE` when no explicit custom route override handles that model |
| `DESEARCH_API_KEY` | API key for search tools |

The defaults in `.env.example` already target mainnet (`finney`) and netuid `67`. The checked-in default is `SEARCH_PROVIDER=desearch`. If you need a fallback search provider, the validator also supports `parallel`; set `SEARCH_PROVIDER=parallel` and `PARALLEL_API_KEY`. Validator sandbox execution defaults to `harnyx/harnyx-subnet-sandbox:finney`; set `SANDBOX_IMAGE=harnyx/harnyx-subnet-sandbox:testnet` for staging/testnet, or use another explicit value only when you intentionally want to test or pin a different sandbox image. `VALIDATOR_ARTIFACT_PARALLELISM` controls concurrent artifact sandboxes and defaults to `4`; `VALIDATOR_TASK_PARALLELISM` controls batch-wide task sessions across those artifacts and defaults to `20`.

Completed-run execution logs are stored under the validator state volume while the platform drains `/runs` pages. `VALIDATOR_RUN_PROGRESS_RETENTION_SECONDS` controls how long terminal batch blobs are retained before cleanup; it defaults to 24 hours. `VALIDATOR_RUN_PROGRESS_CLEANUP_INTERVAL_SECONDS` controls the idle cleanup cadence and defaults to 10 minutes. Operators with intentionally delayed platform sync can increase the retention window.

Validator scoring keeps `SCORING_LLM_PROVIDER` configurable, but the scoring model contract is fixed in code to `moonshotai/Kimi-K2.5-TEE` with `reasoning_effort="high"`. The pairwise scoring prompt, request shape, and score mapping live in `public/packages/commons/src/harnyx_commons/miner_task_scoring.py`; validator runtime code only wires providers, sandbox execution, and submission flow.

When `TOOL_LLM_PROVIDER=chutes`, validator tool routing internally sends configured OpenRouter-routed models to OpenRouter. `OPENROUTER_API_KEY` is checked only when one of those models is invoked; validators that do not serve them can leave the key blank. Explicit `LLM_MODEL_PROVIDER_OVERRIDES_JSON` entries still win, so internal deployments can route Qwen to `custom-openai-compatible:qwen36-cloud-run` without invoking OpenRouter for Qwen. Do not set `TOOL_LLM_PROVIDER=openrouter`; `openrouter` is an internal route target, not an operator-selectable provider.

### Optional Sentry

- The checked-in `.env.example` defaults `SENTRY_DSN` to the shared Harnyx validator Sentry project so we can monitor operator issues centrally.
- Recommended production defaults are `SENTRY_ENVIRONMENT=prod` and `SENTRY_TRACES_SAMPLE_RATE=0.05`.
- `SENTRY_RELEASE` is optional.
- Clear `SENTRY_DSN` only if you intentionally want to opt out of Harnyx-managed monitoring.
- Validator follows the same Sentry model as platform: framework request-path and fatal top-level crash failures can be auto-captured, while swallowed background-worker failures are captured explicitly.
- Expected translated request/tool 4xx paths stay low-noise and should not create Sentry events during normal control flow.

### Wallet configuration

Choose one of these options:

- **Existing hotkey file**: If you have a hotkey in `~/.bittensor/wallets`, set:
  - `SUBTENSOR_WALLET_NAME`
  - `SUBTENSOR_HOTKEY_NAME`

- **Generate from mnemonic**: If you don't have a hotkey file yet, set:
  - `SUBTENSOR_HOTKEY_MNEMONIC` — the hotkey will be generated on first start

## Step 3: Start the validator

```bash
bash scripts/operator_up.sh
```

## Step 4: Verify it's working

Check logs for successful startup:

```bash
bash scripts/operator_logs.sh
```

Look for:
- Successful connection to the platform
- Validator endpoint registration confirmation
- Evaluation batches being received and processed

## Operations

### Start / Stop / Logs

| Action | Command |
|--------|---------|
| Start or update | `bash scripts/operator_up.sh` |
| View logs | `bash scripts/operator_logs.sh` |
| Stop | `bash scripts/operator_down.sh` |

### Auto-updates (Watchtower)

Watchtower polls Docker Hub every 5 minutes and will pull/restart the validator when `VALIDATOR_IMAGE` changes.

The stack now uses a normal short shutdown budget (`60s`), not a long correctness-critical drain window.

Miner-task restart safety no longer depends on the validator staying alive until every in-flight batch fully finishes. Platform persists completed validator submissions while a batch is still active, and if the validator later restarts and reports `unknown` for that batch, platform can redispatch the same batch back to that same validator with restore data so the validator resumes only the remaining work.

## Troubleshooting

### Weight response semantics

`GET /v1/weights` returns `champion_uid` and `weights`.

When latest champion weights exist, total miner weight is capped by the platform's configured maximum miner emission fraction times the latest champion batch score. Owner `uid=0` receives the remainder.

If there is no champion selection, miner emission is burned for that round: `champion_uid=null`, `weights={0: 1.0}`.

Use the live benchmark page to inspect benchmark history and run detail: [`dashboard.harnyx.ai/benchmark`](https://dashboard.harnyx.ai/benchmark).

### HTTP 403 when querying weights

The platform will deny weight queries (`GET /v1/weights`) unless the validator:

1. Is a metagraph validator and signs the request
2. Has a registered validator endpoint (`POST /v1/validators/register`)

If weight queries still return `403`, verify that the validator has registered its public base URL with platform under the same hotkey used to sign the request.

**Note:** Registration is still required; an unregistered hotkey remains blocked.
