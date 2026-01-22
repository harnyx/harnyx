# Validator Operator Runbook

This directory contains the validator runtime package plus an operator-ready Docker Compose stack (`validator` + `watchtower`).

## Prerequisites

Before starting, ensure you have:

1. **Docker** installed and running (Docker Compose v2+)
2. **A Bittensor wallet/hotkey** registered on the subnet metagraph
3. **A public endpoint** reachable by the platform (for registration + evaluation callbacks)
4. **API keys** for LLM and search tools (see env vars below)

## Step 1: Create your env file

```bash
cp .env.example .env
```

## Step 2: Configure environment variables

Edit `.env` and set at least:

| Variable | Description |
|----------|-------------|
| `PLATFORM_BASE_URL` | Platform API endpoint |
| `VALIDATOR_PUBLIC_BASE_URL` | How the platform can reach your validator |
| `CHUTES_API_KEY` | API key for LLM calls |
| `DESEARCH_API_KEY` | API key for search tools |

The defaults in `.env.example` already target mainnet (`finney`) and netuid `67`, and use `castersubnet/caster-subnet-sandbox:latest` for sandbox execution.

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

Watchtower polls Docker Hub every 5 minutes and will pull/restart the validator when `CASTER_VALIDATOR_IMAGE` changes.

The stack is configured for a graceful restart window (default `30m`) so in-progress evaluations can finish before the container is restarted.

## Troubleshooting

### HTTP 403 when querying weights

The platform will deny weight queries (`GET /v1/weights`) unless the validator:

1. Is a metagraph validator and signs the request
2. Has a registered validator endpoint (`POST /v1/validators/register`)
3. Is considered "functioning" — has successfully completed an evaluation batch within the last 120 hours

If your validator has never completed an evaluation batch (or hasn't completed one in 120+ hours), weight queries are blocked until it completes a later batch.

**Note:** Changing or re-registering your endpoint does not clear this block — it is tracked by hotkey.
