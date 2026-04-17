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
- RAM: 4 GiB (8 GiB recommended)
- Disk: 20 GB
- Network: platform must reach your validator on TCP 8100; set `VALIDATOR_PUBLIC_BASE_URL` accordingly
- Third-party APIs: Chutes (`CHUTES_API_KEY`) + DeSearch (`DESEARCH_API_KEY`)

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
| `DESEARCH_API_KEY` | API key for search tools |

The defaults in `.env.example` already target mainnet (`finney`) and netuid `67`. The checked-in default is `SEARCH_PROVIDER=desearch`. If you need a fallback search provider, the validator also supports `parallel`; set `SEARCH_PROVIDER=parallel` and `PARALLEL_API_KEY`. Validator sandbox execution defaults to `harnyx/harnyx-subnet-sandbox:finney`; set `SANDBOX_IMAGE=harnyx/harnyx-subnet-sandbox:testnet` for staging/testnet, or use another explicit value only when you intentionally want to test or pin a different sandbox image.

Validator scoring keeps `SCORING_LLM_PROVIDER` configurable, but the scoring model contract is fixed in code to `moonshotai/Kimi-K2.5-TEE` with `reasoning_effort="high"`.

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

### HTTP 403 when querying weights

The platform will deny weight queries (`GET /v1/weights`) unless the validator:

1. Is a metagraph validator and signs the request
2. Has a registered validator endpoint (`POST /v1/validators/register`)

If weight queries still return `403`, verify that the validator has registered its public base URL with platform under the same hotkey used to sign the request.

**Note:** Registration is still required; an unregistered hotkey remains blocked.
