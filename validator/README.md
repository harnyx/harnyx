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
- Disk: space for Docker images, logs, normal validator operation and a 128 MiB storage compatibility probe
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
| `CHUTES_API_KEY` | API key for the default validator miner-task scoring and similarity provider |
| `OPENROUTER_API_KEY` | API key required only when an explicit validator route uses OpenRouter, including the Brightmount-managed DeepSeek similarity route |
| `SCORING_LLM_PROVIDER` | Optional miner-task scoring provider selector; defaults to `chutes` |
| `SCORING_LLM_TEMPERATURE` | Optional miner-task scoring request temperature; defaults to provider default |
| `SCORING_LLM_MAX_OUTPUT_TOKENS` | Optional miner-task scoring max output tokens; defaults to `32768` |
| `SCORING_LLM_RETRY_ATTEMPTS` | Optional miner-task scoring LLM request retry attempts; defaults to `6` |
| `SCORING_LLM_RETRY_INITIAL_MS` | Optional miner-task scoring LLM request retry initial backoff; defaults to `30000` |
| `SCORING_LLM_RETRY_MAX_MS` | Optional miner-task scoring LLM request retry maximum backoff; defaults to `300000` |
| `SCORING_LLM_RETRY_JITTER` | Optional miner-task scoring LLM request retry jitter ratio; defaults to `0.2` |
| `SIMILARITY_LLM_PROVIDER` | Optional duplicate-preflight similarity provider selector; defaults to `chutes` |
| `SIMILARITY_LLM_TEMPERATURE` | Optional duplicate-preflight similarity request temperature; defaults to provider default |
| `SIMILARITY_LLM_TIMEOUT_SECONDS` | Total duration per similarity model attempt; defaults to `900` seconds |
| `SIMILARITY_LLM_PREFILL_TIMEOUT_SECONDS` | Time until the first actual text or reasoning output; defaults to `300` seconds |
| `SIMILARITY_LLM_INACTIVITY_TIMEOUT_SECONDS` | Maximum gap between outputs after the first output; defaults to `60` seconds |
| `SIMILARITY_LLM_MAX_OUTPUT_TOKENS` | Optional duplicate-preflight similarity max output tokens; defaults to `20480` |
| `SIMILARITY_LLM_RETRY_ATTEMPTS` | Optional duplicate-preflight similarity LLM request retry attempts; defaults to `1` |
| `SIMILARITY_LLM_RETRY_INITIAL_MS` | Optional duplicate-preflight similarity LLM request retry initial backoff; defaults to `0` |
| `SIMILARITY_LLM_RETRY_MAX_MS` | Optional duplicate-preflight similarity LLM request retry maximum backoff; defaults to `0` |
| `SIMILARITY_LLM_RETRY_JITTER` | Optional duplicate-preflight similarity LLM request retry jitter ratio; defaults to `0.0` |

Scoring attempts use fixed limits of 300 seconds to first text or reasoning output and 900 seconds total. These limits have no environment settings.

Similarity limits apply separately to each provider attempt. Headers,
heartbeats, and empty metadata do not count as output. The total does not restart
when output begins. Existing explicit environment values override these defaults.
Platform allows 80 minutes for the complete similarity request, including all
provider retries and model fallbacks. Each model attempt retains its own limits;
operator overrides can still exceed the Platform budget.

The defaults in `.env.example` already target mainnet (`finney`) and netuid `67`. Validator sandbox execution defaults to `harnyx/harnyx-subnet-sandbox:finney`; set `SANDBOX_IMAGE=harnyx/harnyx-subnet-sandbox:testnet` for staging/testnet, or use another explicit value only when you intentionally want to test or pin a different sandbox image. Validator miner-task execution uses fixed runtime concurrency: 15 concurrent artifact sandboxes and 45 task-attempt slots across active artifacts. Platform assigns three-task chunks, prioritizing the same tasks across artifacts. Each sandbox finishes its chunk before taking a later one; other sandboxes advance independently. Scoring remains separate at 10 Gemma and 10 Qwen response jobs and waits for every required execution and retry of the task to close. Each sandbox retains a one-CPU limit. All validator-owned sandbox processes share at most 4 allowed logical CPUs; validators exposing 4 or fewer CPU IDs use all of them.

Provider-backed miner script tools execute through platform tool proxy with miner-stored credentials. Validators do not need `SEARCH_PROVIDER`, `DESEARCH_API_KEY`, or `PARALLEL_API_KEY` for normal miner task provider-backed tool execution. Each `llm_chat` request selects its provider; the former `TOOL_LLM_PROVIDER` setting has been removed.

Completed-run execution logs are stored under the validator state volume for local audit and troubleshooting. Platform owns assignment, result acceptance, delivery projection, and finalization. `VALIDATOR_RUN_PROGRESS_RETENTION_SECONDS` controls how long terminal local progress blobs are retained before cleanup; it defaults to 24 hours. `VALIDATOR_RUN_PROGRESS_CLEANUP_INTERVAL_SECONDS` controls the idle cleanup cadence and defaults to 10 minutes.

Validator pairwise scoring keeps `SCORING_LLM_PROVIDER` configurable for the default scoring provider and embeddings, but scoreable-execution judge distribution is fixed in code with `ScoringSlotConfig(entries=...)`. The current entries are Gemma 4 and Qwen3.6, each with 10 scoreable-execution slots. The validator fetches one model-unaware scoreable-execution batch from Platform using total free slots, then assigns each execution to one entry by deterministic rotation over entries with free slots. Each assigned entry uses `SCORING_LLM_RETRY_*` for every model in its request retry loop. If the assigned primary exhausts scoring LLM retries, the entry tries GLM-5.2, GLM-5.1, and Kimi K3 in that order before recording `scoring_llm_retry_exhausted` or the existing scoring failure detail for that execution. With the default six attempts per model, a fully exhausted chain can make 24 LLM requests in each judge direction. Because pairwise scoring runs miner-first and reference-first directions concurrently, one fully exhausted scoreable execution can make up to 48 requests. This fallback stays inside the assigned entry and does not requeue work to another configured primary entry. Validators use `LLM_MODEL_PROVIDER_OVERRIDES_JSON.scoring` to route configured entry and fallback models to provider targets. Brightmount-managed route files send the three current fallbacks to Chutes, keep Gemma 4 and Qwen3.6 on their private routes, and retain explicit legacy routes for GLM-5 and Kimi K2.5. External validators use Chutes by default or need their own provider access and route overrides for another backend. The pairwise scoring prompt, request shape, retry loop, and score mapping live in `public/packages/commons/src/harnyx_commons/miner_task_scoring.py`; validator runtime code wires providers, slot assignment, sandbox execution, and submission flow.

Override fallback selection with `SCORING_LLM_FALLBACK_MODELS_JSON` and `SIMILARITY_LLM_FALLBACK_MODELS_JSON`, each an ordered JSON array of model IDs. An unset variable or JSON `null` preserves the built-in chain; `[]` disables fallbacks. A supplied list replaces the fallback list, preserving its order and skipping the active primary if included. Blank, non-string, or repeated IDs fail settings validation. The scoring list applies to both scoring slots and direct evaluation; primary models, slot counts, retry rules, and failure reporting stay unchanged. Provider selection remains in `LLM_MODEL_PROVIDER_OVERRIDES_JSON.scoring` or `.duplication_detection`: use a provider-native model ID as the key and, for example, `openrouter` as its provider target. Both judges support those OpenRouter routes. The shared provider registry registers explicitly configured fallback IDs routed to OpenRouter with the real provider; models routed elsewhere and unconfigured IDs retain the existing restrictions. Configuring a model does not verify provider availability or credentials.

Similarity judging uses `SIMILARITY_LLM_*` for provider, temperature, timeout, output tokens, and retry policy. The default model chain is `google/gemma-4-31B-turbo-TEE`, then `deepseek-ai/DeepSeek-V4-Flash-0731-TEE`, then `zai-org/GLM-5.2-TEE`, then `moonshotai/Kimi-K3-TEE`. Default external-validator routes use Chutes unless the operator supplies compatible overrides. Brightmount-managed validators route Gemma 4 through OpenRouter and explicitly override the fallback list with DeepSeek V4.1 Flash 1017, DeepSeek V4 Flash 0731, MiniMax M3.1 Flash `:free`, and GLM 5.3 Flash, in that order through OpenRouter. Flash 0731 retains its internal alias and provider exclusions; the other three use their OpenRouter-native IDs. The built-in chain retains its provider compatibility checks; additional provider-native model IDs may route to OpenRouter. The judge advances after any provider/model attempt failure, including a non-retryable provider rejection, exhausted retries, an invalid structured response, or an incomplete response. Each activation is logged; unexpected implementation errors and cancellation are not treated as model fallback. Platform selects two independently ranked, different references from one structural evidence pass over the established corpus: the unchanged symmetric-similarity primary and a candidate-coverage challenge. Validators always decide the primary round first. Platform opens the separate challenge round only if the primary aggregate is `novel`, then uses the lower classification. The judge evaluates the union of every reachable non-error successful branch, so query hashes, schema hashes, IDs, timestamps, randomness, or branch frequency cannot hide preserved reference behavior. `Novel` requires affirmative evidence that the primary controller, evidence state and flow, and answer-production path are all replaced on every completed comparison. If primary selection returns `no_eligible_reference`, Platform assigns `novel` without a validator judgment; if only the challenge lacks an alternative, the primary result is final.

Platform requests live status on the existing similarity POST with `Accept: text/event-stream`.
The response reports model-attempt progress, sends a heartbeat every 30 seconds,
and ends with the existing result or error plus attempt summaries. Heartbeats
indicate a live connection, not model output, and do not extend the 80-minute
budget. Callers without this Accept header receive the existing JSON response;
Platform also accepts JSON from older validators without sending another request.
A disconnected caller or a response write blocked for 60 seconds cancels local
judging. Neither path produces a verdict. Diagnostic retention and Platform log
fields are documented in [observability](../../.docs/ops/observability.md#similarity-response-status).

If the validator cannot produce a similarity verdict after receiving a valid request because judge execution or the upstream judge provider failed, it returns a structured `502` JSON body with `error_code`, `retryable`, `detail`, and nullable `judge_usage`. Timeouts, exhausted model fallbacks, and invalid judge output report `retryable=true`. Known provider configuration failures and upstream authentication/endpoint rejections (`401`, `403`, `404`) report `retryable=false`, including when wrapped in a provider exception. A missing similarity judge returns a structured `503` with `retryable=false`. Request validation failures remain `422`; authentication rejections remain `401`/`403`. Temporary service unavailability remains retryable. Platform reads the explicit `retryable` field and permits at most one additional request, which runs the model chain again. Invalid completed success responses also permit that retry. Incomplete streams and transport failures retain the ownership-deadline wait before recovery, preventing overlapping requests. External validators must upgrade to emit the corrected retry classification; Platform continues to honor older validators' explicit `retryable=false` responses.

Internal route controls may still appear in internal/local tooling deployments, including miner local eval, but they are not validator server miner-task provider credential requirements.

### Miner communication

Platform assignments contain queries and execution settings without reference answers. An artifact can begin execution once its duplicate decision is final while other artifacts are still being checked. Platform waits for the complete participant/work set and all executions of a task across validators before generating that task's black-box reference. Once its reference resolves, validators can score that task's saved responses while other tasks continue executing. Qualifying and main retain separate complete-work boundaries. Execution defaults to 24 hours and evaluation keeps the approximate six-hour round window with reference-processing allowance. The existing query-only execution and scoreable-execution payloads are unchanged.

Platform can assign registered-miner execution through `POST /validator/endpoint-assignments`. Your public HTTPS base must also accept miner callbacks at `/validator/endpoint-assignments/{assignment_id}/callback`. Preserve the signed raw path through proxies. The runtime verifies Platform delegation and saves start timing on Platform before sending. It places the existing encoded signed document in optional `callback_context`; miners authenticate the validator directly and return that string unchanged in `X-Harnyx-Callback-Context`. Our callback route requires and verifies this context, then verifies the original miner-signed answer and forwards it to the configured Platform for durable storage. An empty in-memory registry after restart does not require an assignment lookup to reconstruct context. Existing Platform-signer trust initialization from chain must still succeed. During a Platform reporting outage, repeated callbacks retain the original receipt timestamp and receive no success acknowledgement until storage is confirmed. Pending reports are not persisted across another validator restart.

Healthy progress checks retain the current validator. After unavailability, another validator can continue the same assignment; either may deliver the first valid answer. A signed acceptance acknowledgement or known miner status confirms admission. Failed status checks trigger an identical assignment retry only while admission remains unconfirmed; an explicit `unknown` status permits retry even after earlier confirmation. Immediately before resending, the validator checks that no answer is pending forwarding or already saved and that the original deadline remains open. Neither retries nor missing ACKs extend the original deadline. Complete answers received on time may be forwarded for 60 more seconds. Validator disappearance alone carries no miner penalty; an authenticated actual-attempt failure report does. Saved outcomes remain recoverable afterward. Execution readiness is checked on the actual execute route, independently of rating judging.

### Optional black-box rating judging

The signed `/validator/status` response reports `rating_worker_ready` from the running rating worker. Platform requires a true report before assigning new comparisons; disabled or stopped workers report false. Previously assigned results remain acceptable. Pausing the Platform scheduler also pauses making newly received answers available for judging; pending orchestration resumes when the scheduler restarts.

The rating worker starts with the normal Platform client and scoring-provider configuration. Platform assigns evaluation work only in its staging environment; other environments return an authenticated empty work page. Idle polling does not call the paid judge. Registration, health, signed worker readiness and metagraph permission still govern new assignments. Source revision and version metadata remain available for monitoring but do not gate work. This lane does not change white-box work or activate black-box emission.

The worker polls comparisons assigned to this validator, verifies each miner's signed answer and original callback path, hydrates receipt-backed citations, and judges answer quality in both orders. Platform alone applies the response-time tie-break. Each worker retains at most 10 comparisons, counting active judging, judging retry waits and completed results awaiting delivery together. It requests only available capacity and leaves other comparisons queued on Platform. Paging skips already-held and permanently rejected IDs so later work can fill free slots. Successful delivery or permanent rejection frees its slot independently; a slow comparison does not form a group barrier. Temporary delivery failures retain their slot for retry. A whole failed judging attempt waits 30 seconds before retrying, doubling after each failure up to five minutes and resetting after success. This preserves provider-level retries and does not penalize either miner. Retry waits keep their slots while other judging and deliveries continue independently; ten waiting comparisons pause new fetching. Backoff state is in memory and restarts with the process.

Completed results are retried after temporary delivery failures without rejudging while the process remains alive. Platform limits each judgment body to 1 MiB and rejects oversized evidence with HTTP 413 without truncation. HTTP 413 or HTTP 409 with `rating_judgment_already_accepted` stops further automatic delivery and judging of that comparison in the worker and reports the delivery failure; neither miner is penalized for it. Other 409 responses remain retryable. These suppression records, like completed results, are in memory. A daily rating boundary does not expire a pending result. Platform saves the first valid accepted judgment and acknowledges identical retries. A validator restart before Platform accepts the result can require rejudging; already accepted evidence is not replaced. Assignment ownership remains fixed across release-metadata changes. Brief Platform/validator deployment mismatches can cause temporary failures; the existing error reporting and retries remain in effect without a separate release pin or coordinated drain.

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

Miner-task restart safety no longer depends on the validator staying alive until every in-flight batch fully finishes. Platform owns expected work, attempt numbering, accepted results, delivery projection, and finalization. After restart, the validator replays saved, unacknowledged execution/results from `VALIDATOR_STATE_DIR` before polling new assignments. Platform resumes scoring from accepted execution evidence. Only unrecoverable unfinished work may execute again within existing limits; exhausted validator interruption fails that delivery. Keep the state volume across restarts. The sandbox and validator must both support the private completion-timestamp contract; before rolling back to code without durable pending replay, let pending submissions finish and preserve the state directory.

## Troubleshooting

### Weight response semantics

`GET /v1/weights` returns `champion_uid` and `weights`.

The endpoint serves the allocation saved for the latest terminal source batch with finalized tasks and artifacts. It maps saved hotkeys to current metagraph UIDs; unregistered shares burn through owner `uid=0`. Deploying a policy change or changing the champion ceiling does not recalculate a saved allocation. The new policy applies when a later terminal batch records its allocation.

From source data version 12, successful allocations combine independent Overall and F1 champion rewards with participant shares. The response's `champion_uid` identifies the Overall champion; `weights` includes both champion rewards and participation, added by current UID. One artifact may hold both titles.

The [public emission rules](../README.md#how-participant-miner-emission-works) explain the champion budget and split, independent retention decay, participation and novelty multipliers, qualifying score used for percentile cutoffs, and historical policies.

Failed batches divide the entire pool equally among distinct participant hotkeys, including participating champions, without separate champion allocations; `champion_uid` is null. They do not advance or break either title's retention sequence or change payouts already received.

Legacy allocations saved as combined hotkey totals remain unchanged, including allocations for already-failed batches. Monitoring cannot recover their champion/participant split and reports it as unavailable. For categorized allocations, monitoring displays the saved component and hotkey totals. Expandable artifact details explain those amounts when the supporting cache is available; missing detail does not remove the saved totals. Failed batches and versions 1 through 6 have null stage and novelty multipliers.

Initializing/running batches and terminal batches without finalized tasks or artifacts do not replace the recorded source. If no eligible terminal source exists, the response burns the full allocation: `champion_uid=null`, `weights={0: 1.0}`. Shares also burn when their saved hotkeys cannot be mapped to current registered miners.

Use the live benchmark page to inspect benchmark history and run detail: [`dashboard.harnyx.ai/benchmark`](https://dashboard.harnyx.ai/benchmark).

### HTTP 403 when querying weights

The platform will deny weight queries (`GET /v1/weights`) unless the validator:

1. Is a metagraph validator and signs the request
2. Has a registered validator endpoint (`POST /v1/validators/register`)

The platform's validator allowlist only controls miner-task batch delivery. It does not control validator registration or `GET /v1/weights`.

If weight queries still return `403`, verify that the validator has registered its public base URL with platform under the same hotkey used to sign the request.

**Note:** Registration is still required; an unregistered hotkey remains blocked.

## Miner temporary storage

This release runs **only a background compatibility probe**. Existing miner execution keeps its read-only filesystem; no new workspace volume is attached, even after a successful probe. File I/O activation remains a separate release decision.

Once per validator process, an isolated `harnyx-storage-probe` pool tests a 128 MiB image through allocation, mount, kernel I/O limits, read/write and cleanup. Failures do not block startup or existing work. Success and failure reports include the stage, bounded sanitized exception traces (including helper errors), cleanup outcome, validator hotkey and version, and Docker daemon Engine/API versions when available. Docker version lookup has a ten-second timeout; lookup failure does not skip the trial. The validator sends them through a separate signed registration request; normal registration never includes diagnostics. Delivery has a ten-second request timeout and at most three attempts, spaced thirty seconds apart. Delivery failure or rejection by an older platform does not change readiness. Look for `validator storage probe` in platform logs; use `probe_id` to recognize repeated deliveries. Reports live in logs, not a database or monitoring endpoint; missing reports are not proof of success. No environment dump, credentials or frame-local values are collected.

The probe uses fixed names and ownership-fenced recovery. It does not rerun allocation on delivery retries; unresolved cleanup blocks the next probe before another allocation. Normal cleanup releases the loop/image and removes the helper and filesystem volume; the small named backing volume and bookkeeping remain. Probe resources are separate from miner containers. Inspect the validator log and `docker logs harnyx-storage-probe-helper` if that helper remains. Do not prune or detach unrelated resources. Existing stale miner-container cleanup still gates execution independently of this probe. Shutdown signals the probe without waiting for its thread, so slow Docker diagnostics cannot consume the execution workers’ shutdown grace. In-flight probe cleanup may finish while the service stops; interrupted resources are recovered on the next startup.

The storage implementation retained for later activation gives each artifact a shared 3 GiB ext4 disk image for its query workspaces. Filesystem metadata reduces usable file capacity. One query can consume all remaining space for that artifact; other artifacts have separate images. Fifteen fixed slots bound total workspace image capacity to 45 GiB, including allocations retained after cleanup failures. This does not limit Docker image storage, logs or other host consumers.

Each artifact's loop device is limited to **32 IOPS and 4 MiB/s for reads, and independently the same limits for writes**, shared by its concurrent queries. Override these with `VALIDATOR_ARTIFACT_IOPS` and `VALIDATOR_ARTIFACT_IO_MIB_PER_SECOND` in the validator environment, then recreate the validator container. Both accept positive integers; unset or empty values use the defaults. Zero does not disable throttling. At fifteen active slots, the configured sums are 480 IOPS and 60 MiB/s in each direction. These are loop-device limits, not a physical host-wide budget: cached reads, helper image preparation, and unrelated disk activity are outside them. Low IOPS especially slows random access and frequent fsync workloads.

The validator starts a trusted storage helper from its own immutable Docker image. Required utilities are included in that image; operators do not install tools inside running containers. The helper receives UID 0, `MKNOD`, `/dev/loop-control:rw` and device rule `b 7:* rwm`, with all other capabilities dropped, no network and a read-only root. It does not receive the Docker socket, credentials or validator state. The Docker daemon mounts the prepared filesystem. The miner receives no additional capabilities or devices, and keeps its read-only root and existing memory/process restrictions.

This requires a rootful Linux Docker daemon with loop and ext4 support and permission to grant the helper those device accesses. A normal host-Docker validator needs no additional capabilities on its own container; Docker must be allowed to start the helper. Docker-in-Docker requires its outer daemon deployment to permit these operations (the existing privileged staging daemon does). Rootless Docker and restrictive device/security policies are not assumed supported. The daemon must support block I/O throttling and be accessible through a local Unix socket. The validator creates each consumer without starting it, configures all four limits through the Engine API using the same owned device node as the filesystem mount, then starts it. This does not require a daemon-side `/dev/loopN` pathname. Existing CLI security and launch options are preserved. The background probe verifies all four kernel cgroup limits and performs an allocation, mount, read/write and cleanup check. Failure is reported without enabling storage or stopping existing execution. The eventual artifact rollout will require space for 45 GiB plus normal operating headroom; that capacity is not allocated by this probe.

The helper owns the labeled `harnyx-artifact-storage-backing` volume. It is never mounted into miners. Fixed slot records account for incomplete operations across restarts. Only confirmed container removal, filesystem-volume removal and loop detachment permit a slot to be freed. Cleanup failure preserves completed query results, logs the consumer identity and retains the allocation. Unknown ownership blocks recovery; operators must not manually delete backing images or volumes while they might still be mounted or attached.

After Docker mounts an artifact workspace, the helper verifies and enables kernel loop `autoclear`, then unlinks the owned image file. The mounted loop keeps the image data alive: existing I/O continues even if the helper dies. When the last user releases the loop, the kernel detaches it and releases the image storage. Preparation retains a named image with autoclear disabled so the device survives until Docker mounts it. A crash before activation finishes can therefore leave an image for normal cleanup or startup recovery. Slot records remain until ownership-fenced cleanup confirms consumer, filesystem-volume and loop removal; uncertain mappings are never detached using a missing image as proof of ownership. Live consumers still hold storage, and Docker does not automatically remove the named backing volume or its small bookkeeping files.

For later artifact-storage activation diagnosis, inspect `docker logs harnyx-artifact-storage-helper`, the validator error, and `docker volume ls --filter label=ai.harnyx.artifact-storage`. Preserve exact resource identities. Restarting the validator after its previous incarnation has stopped triggers owned-resource recovery; a second live validator on the same daemon is rejected. If recovery cannot prove ownership or detachment, stop new execution and investigate those exact resources. Never use global volume prune, global loop detach or forced unmount as recovery.

Real Docker/loop storage acceptance tests are manual-only; CI retains unit tests and ordinary sandbox tests without loop devices. On a suitable development host, build both images with `task build:sandbox` and `task build:validator`, then run:

```bash
task test:integration -- --skip-setup --deterministic-defaults --artifact-storage public/validator/tests/integration/test_artifact_storage.py
```

Without `--artifact-storage`, these tests are deselected before provisioning resources. They use 128 MiB images, not a full-capacity 45 GiB trial.

Deploy the matching sandbox and validator images together. Before rollback, drain execution and reclaim its storage using the new manager; if reclamation cannot be confirmed, leave execution stopped and retain the diagnostic identities. Reverting to unlimited workspaces is not an automatic fallback.
