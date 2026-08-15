# Validator Operator Runbook

This directory contains the validator runtime package plus an operator-ready Docker Compose stack (`validator` + `watchtower`).

## Prerequisites

Before starting, ensure you have:

1. **Docker** installed and running (Docker Compose v2+)
2. **A Bittensor wallet/hotkey** registered on the subnet metagraph
3. **A public endpoint** reachable by the platform (for registration + evaluation callbacks)
4. **API key** for validator pairwise scoring and similarity LLM calls (see env vars below)

### Hardware + networking (quick sizing)

- vCPU: 2 (4 recommended)
- RAM: 32 GiB
- Disk: 20 GB
- Network: platform must reach your validator on TCP 8100; set `VALIDATOR_PUBLIC_BASE_URL` accordingly
- Third-party APIs: Chutes (`CHUTES_API_KEY`) for validator pairwise scoring and similarity by default. Brightmount-managed routes also use OpenRouter (`OPENROUTER_API_KEY`) for the DeepSeek similarity fallback; provider-backed miner script tools use miner-stored credentials through platform tool proxy.

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
| `CHUTES_API_KEY` | API key for the default validator pairwise scoring and similarity provider |
| `OPENROUTER_API_KEY` | API key required only when an explicit validator route uses OpenRouter, including the Brightmount-managed DeepSeek similarity route |
| `SCORING_LLM_PROVIDER` | Optional pairwise scoring provider selector; defaults to `chutes` |
| `SCORING_LLM_TEMPERATURE` | Optional pairwise scoring request temperature; defaults to provider default |
| `SCORING_LLM_TIMEOUT_SECONDS` | Optional pairwise scoring LLM request timeout; defaults to `300` |
| `SCORING_LLM_MAX_OUTPUT_TOKENS` | Optional pairwise scoring max output tokens; defaults to `20480` |
| `SCORING_LLM_RETRY_ATTEMPTS` | Optional pairwise scoring LLM request retry attempts; defaults to `6` |
| `SCORING_LLM_RETRY_INITIAL_MS` | Optional pairwise scoring LLM request retry initial backoff; defaults to `30000` |
| `SCORING_LLM_RETRY_MAX_MS` | Optional pairwise scoring LLM request retry maximum backoff; defaults to `300000` |
| `SCORING_LLM_RETRY_JITTER` | Optional pairwise scoring LLM request retry jitter ratio; defaults to `0.2` |
| `SIMILARITY_LLM_PROVIDER` | Optional duplicate-preflight similarity provider selector; defaults to `chutes` |
| `SIMILARITY_LLM_TEMPERATURE` | Optional duplicate-preflight similarity request temperature; defaults to provider default |
| `SIMILARITY_LLM_TIMEOUT_SECONDS` | Optional duplicate-preflight similarity LLM request timeout; defaults to `300` |
| `SIMILARITY_LLM_MAX_OUTPUT_TOKENS` | Optional duplicate-preflight similarity max output tokens; defaults to `20480` |
| `SIMILARITY_LLM_RETRY_ATTEMPTS` | Optional duplicate-preflight similarity LLM request retry attempts; defaults to `1` |
| `SIMILARITY_LLM_RETRY_INITIAL_MS` | Optional duplicate-preflight similarity LLM request retry initial backoff; defaults to `0` |
| `SIMILARITY_LLM_RETRY_MAX_MS` | Optional duplicate-preflight similarity LLM request retry maximum backoff; defaults to `0` |
| `SIMILARITY_LLM_RETRY_JITTER` | Optional duplicate-preflight similarity LLM request retry jitter ratio; defaults to `0.0` |

The defaults in `.env.example` already target mainnet (`finney`) and netuid `67`. Validator sandbox execution defaults to `harnyx/harnyx-subnet-sandbox:finney`; set `SANDBOX_IMAGE=harnyx/harnyx-subnet-sandbox:testnet` for staging/testnet, or use another explicit value only when you intentionally want to test or pin a different sandbox image. Validator miner-task execution uses fixed runtime concurrency: 4 concurrent artifact sandboxes and 20 task-attempt slots across active artifacts. Each sandbox retains a one-CPU limit. All validator-owned sandbox processes share at most 4 allowed logical CPUs; validators exposing 4 or fewer CPU IDs use all of them.

Provider-backed miner script tools execute through platform tool proxy with miner-stored credentials. Validators do not need `SEARCH_PROVIDER`, `DESEARCH_API_KEY`, `PARALLEL_API_KEY`, or `TOOL_LLM_PROVIDER` for normal miner task provider-backed tool execution.

Completed-run execution logs are stored under the validator state volume for local audit and troubleshooting. Platform owns assignment, result acceptance, delivery projection, and finalization. `VALIDATOR_RUN_PROGRESS_RETENTION_SECONDS` controls how long terminal local progress blobs are retained before cleanup; it defaults to 24 hours. `VALIDATOR_RUN_PROGRESS_CLEANUP_INTERVAL_SECONDS` controls the idle cleanup cadence and defaults to 10 minutes.

Validator pairwise scoring keeps `SCORING_LLM_PROVIDER` configurable for the default scoring provider and embeddings, but scoreable-execution judge distribution is fixed in code with `ScoringSlotConfig(entries=...)`. The current entries are Gemma 4 and Qwen3.6, each with 10 scoreable-execution slots. The validator fetches one model-unaware scoreable-execution batch from Platform using total free slots, then assigns each execution to one entry by deterministic rotation over entries with free slots. Each assigned entry uses `SCORING_LLM_RETRY_*` for every model in its request retry loop. If the assigned primary exhausts scoring LLM retries, the entry tries GLM-5.2, GLM-5.1, and Kimi K3 in that order before recording `scoring_llm_retry_exhausted` or the existing scoring failure detail for that execution. With the default six attempts per model, a fully exhausted chain can make 24 LLM requests in each judge direction. Because pairwise scoring runs miner-first and reference-first directions concurrently, one fully exhausted scoreable execution can make up to 48 requests. This fallback stays inside the assigned entry and does not requeue work to another configured primary entry. Validators use `LLM_MODEL_PROVIDER_OVERRIDES_JSON.scoring` to route configured entry and fallback models to provider targets. Brightmount-managed route files send the three current fallbacks to Chutes, keep Gemma 4 and Qwen3.6 on their private routes, and retain explicit legacy routes for GLM-5 and Kimi K2.5. External validators use Chutes by default or need their own provider access and route overrides for another backend. The pairwise scoring prompt, request shape, retry loop, and score mapping live in `public/packages/commons/src/harnyx_commons/miner_task_scoring.py`; validator runtime code wires providers, slot assignment, sandbox execution, and submission flow.

Similarity judging uses `SIMILARITY_LLM_*` for provider, temperature, timeout, output tokens, and retry policy. The default model chain is `google/gemma-4-31B-turbo-TEE`, then `deepseek-ai/DeepSeek-V4-Flash-0731-TEE`, then `zai-org/GLM-5.2-TEE`, then `moonshotai/Kimi-K3-TEE`. Default external-validator routes use Chutes unless the operator supplies compatible overrides. Brightmount-managed validators route Gemma 4 through the private Cloud Run endpoint, the exact DeepSeek candidate through OpenRouter as `deepseek/deepseek-v4-flash-0731`, and GLM 5.2 plus Kimi K3 through Chutes. Validator bootstrap rejects OpenRouter similarity overrides for every other model. The judge advances after any provider/model attempt failure, including a non-retryable provider rejection, exhausted retries, an invalid structured response, or an incomplete response. Each activation is logged; unexpected implementation errors and cancellation are not treated as model fallback. Platform selects the structurally closest eligible artifact from the established corpus of up to 10 champion artifacts, every member of the two preceding successful batches, and earlier current-batch challengers. Duplicate artifacts are retained at their champion-corpus position, and artifacts owned by the evaluated hotkey are excluded. Validators return `duplicate`, `near_duplicate`, `notable_change` or `novel` with structured reasoning. `Near_duplicate` means localized changes inside substantially the same pipeline. `Notable_change` means substantial reorganization or extension while at least one part of the same architectural root remains. Relative to the selected reference, `novel` requires affirmative evidence that all three reachable ordinary-case architectural dimensions are replaced: the primary controller, evidence state and flow, and answer-production path. Preserving, wrapping, or extending any one of those dimensions limits the result to `notable_change` or lower. If structural selection returns `no_eligible_reference`, Platform assigns `novel` automatically without requesting a validator judgment.

If the validator cannot produce a similarity verdict after receiving a valid request because judge execution or the upstream judge provider failed, it returns a structured `502` JSON body with `error_code`, `retryable`, `detail`, and nullable `judge_usage`. Request validation failures remain `422`. Service-availability failures such as auth unavailable or missing similarity judge configuration remain `503` and use normal status-based Platform retry behavior unless a structured failure body says otherwise. Platform reads the explicit `retryable` field from valid structured similarity judge failure bodies when deciding whether to retry a failed similarity judge request.

Internal route controls may still appear in internal/local tooling deployments, including miner local eval, but they are not validator server miner-task provider credential requirements.

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
- Miner-task assignments being polled, executed, and submitted

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

Miner-task restart safety no longer depends on the validator staying alive until every in-flight batch fully finishes. Platform owns expected work, attempt numbering, accepted results, delivery projection, and finalization. After restart, the validator starts with empty active-task memory and polls Platform for new assignments; unfinished work is resolved by Platform deadline and retry handling.

## Troubleshooting

### Weight response semantics

`GET /v1/weights` returns `champion_uid` and `weights`.

Champion emission remains active and additive. Participant emission uses the latest terminal source batch with finalized tasks and artifacts. A failed terminal batch divides the entire post-champion remainder equally among distinct participant hotkeys. A successful data-version-7-or-later batch divides that remainder per reward-eligible artifact using `participation stage × novelty`: top 50% `1`, top 10% `2`, main `5`; near duplicate `1`, notable change `3`, novel `5`. The highest distribution weight is therefore `25`. Shares are aggregated to hotkeys after artifact allocation. Champion emission is never novelty-multiplied. Data versions 1 through 6 retain their historical reward calculation. Registered participant hotkeys receive weight at their current metagraph UID; unregistered shares burn through owner `uid=0`.

Monitoring reports these allocations as hotkey totals with expandable artifact rows containing nullable stage and novelty multipliers, nullable distribution weight, and participant reward fraction. Pre-change and failed source batches return null multipliers with their preserved share.

Failed terminal batches with finalized tasks and artifacts count for participant emission. Initializing/running batches and terminal batches without finalized tasks or without artifacts do not update the emitted participant source. If no terminal source batch with finalized tasks and artifacts exists, only the champion component remains active; if neither champion nor participant components project to registered miners, miner emission is burned for that round: `champion_uid=null`, `weights={0: 1.0}`.

Use the live benchmark page to inspect benchmark history and run detail: [`dashboard.harnyx.ai/benchmark`](https://dashboard.harnyx.ai/benchmark).

### HTTP 403 when querying weights

The platform will deny weight queries (`GET /v1/weights`) unless the validator:

1. Is a metagraph validator and signs the request
2. Has a registered validator endpoint (`POST /v1/validators/register`)

The platform's validator allowlist only controls miner-task batch delivery. It does not control validator registration or `GET /v1/weights`.

If weight queries still return `403`, verify that the validator has registered its public base URL with platform under the same hotkey used to sign the request.

**Note:** Registration is still required; an unregistered hotkey remains blocked.
