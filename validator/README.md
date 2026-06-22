# Validator Operator Runbook

This directory contains the validator runtime package plus an operator-ready Docker Compose stack (`validator` + `watchtower`).

## Prerequisites

Before starting, ensure you have:

1. **Docker** installed and running (Docker Compose v2+)
2. **A Bittensor wallet/hotkey** registered on the subnet metagraph
3. **A public endpoint** reachable by the platform (for registration + evaluation callbacks)
4. **API key** for validator scoring/similarity LLM calls (see env vars below)

### Hardware + networking (quick sizing)

- vCPU: 2 (4 recommended)
- RAM: 32 GiB
- Disk: 20 GB
- Network: platform must reach your validator on TCP 8100; set `VALIDATOR_PUBLIC_BASE_URL` accordingly
- Third-party APIs: Chutes (`CHUTES_API_KEY`) for validator scoring/similarity by default; provider-backed miner script tools use miner-stored credentials through platform tool proxy.

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
| `CHUTES_API_KEY` | API key for the default validator scoring/similarity provider |
| `SCORING_LLM_PROVIDER` | Optional scoring/similarity provider selector; defaults to `chutes` |
| `SCORING_LLM_RETRY_ATTEMPTS` | Optional pairwise scoring LLM request retry attempts; defaults to `6` |
| `SCORING_LLM_RETRY_INITIAL_MS` | Optional pairwise scoring LLM request retry initial backoff; defaults to `30000` |
| `SCORING_LLM_RETRY_MAX_MS` | Optional pairwise scoring LLM request retry maximum backoff; defaults to `300000` |
| `SCORING_LLM_RETRY_JITTER` | Optional pairwise scoring LLM request retry jitter ratio; defaults to `0.2` |

The defaults in `.env.example` already target mainnet (`finney`) and netuid `67`. Validator sandbox execution defaults to `harnyx/harnyx-subnet-sandbox:finney`; set `SANDBOX_IMAGE=harnyx/harnyx-subnet-sandbox:testnet` for staging/testnet, or use another explicit value only when you intentionally want to test or pin a different sandbox image. Validator batch execution uses fixed runtime concurrency: 4 concurrent artifact sandboxes and 20 batch-wide task sessions across those artifacts.

Provider-backed miner script tools execute through platform tool proxy with miner-stored credentials. Validators do not need `SEARCH_PROVIDER`, `DESEARCH_API_KEY`, `PARALLEL_API_KEY`, or `TOOL_LLM_PROVIDER` for normal miner task provider-backed tool execution.

Completed-run execution logs are stored under the validator state volume while the platform drains `/runs` pages. `VALIDATOR_RUN_PROGRESS_RETENTION_SECONDS` controls how long terminal batch blobs are retained before cleanup; it defaults to 24 hours. `VALIDATOR_RUN_PROGRESS_CLEANUP_INTERVAL_SECONDS` controls the idle cleanup cadence and defaults to 10 minutes. Operators with intentionally delayed platform sync can increase the retention window.

Validator scoring keeps `SCORING_LLM_PROVIDER` configurable, but the primary scoring model contract is fixed in code to `moonshotai/Kimi-K2.5-TEE` with `reasoning_effort="high"`. Pairwise scoring uses `SCORING_LLM_RETRY_*` for the scoring LLM request retry loop, including retryable provider errors, response verification failures, and structured-output repair retries; exhausting that policy records `scoring_llm_retry_exhausted` and does not schedule a new miner task attempt. If the primary candidate exhausts provider retries, scoring tries `zai-org/GLM-5-TEE` and then `google/gemma-4-31B-turbo-TEE`; non-retryable provider failures fail fast. The pairwise scoring prompt, request shape, fallback loop, and score mapping live in `public/packages/commons/src/harnyx_commons/miner_task_scoring.py`; validator runtime code wires providers, sandbox execution, and submission flow.

Post-dethrone similarity judging uses the same candidate order as scoring and is routed through the `duplication_detection` surface in `LLM_MODEL_PROVIDER_OVERRIDES_JSON`. The platform computes the dethrone order and sends the original incumbent script plus candidate diff to validators; validators own the similarity prompt internally and return a duplicate/not-duplicate verdict with provider reasoning metadata.

Shared route override variables may still appear in internal/local tooling deployments, including miner local eval, but they are not validator server miner-task provider credential requirements.

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

Champion emission remains active. Participant emission uses the latest terminal source batch with artifacts. Failed terminal batches add the base participant-emission level for each distinct participant miner hotkey. Successful terminal batches tier participant emission by normalized artifact score: the top 10% of all eligible participants gets `2x`, participants through the top 50% get `1x`, and lower-ranked participants get `0`. Score `0` still gets `0`. Completed terminal batches without successful scoring evidence burn participant emission instead of using historical monitoring fallback rows. Registered participant hotkeys receive weight at their current metagraph UID, in addition to any champion emission for the same UID. Owner `uid=0` receives the final remainder, including shares for participant hotkeys that are not currently registered. `champion_uid` reports the active champion emission component's current UID when champion emission is active.

Failed terminal batches with artifacts count for participant emission. Initializing/running batches and terminal batches without artifacts do not update the emitted participant source. If no terminal source batch with artifacts exists, only the champion component remains active; if neither champion nor participant components project to registered miners, miner emission is burned for that round: `champion_uid=null`, `weights={0: 1.0}`.

Use the live benchmark page to inspect benchmark history and run detail: [`dashboard.harnyx.ai/benchmark`](https://dashboard.harnyx.ai/benchmark).

### HTTP 403 when querying weights

The platform will deny weight queries (`GET /v1/weights`) unless the validator:

1. Is a metagraph validator and signs the request
2. Has a registered validator endpoint (`POST /v1/validators/register`)

The platform's validator allowlist only controls miner-task batch delivery. It does not control validator registration or `GET /v1/weights`.

If weight queries still return `403`, verify that the validator has registered its public base URL with platform under the same hotkey used to sign the request.

**Note:** Registration is still required; an unregistered hotkey remains blocked.
