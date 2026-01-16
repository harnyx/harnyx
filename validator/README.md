# Validator operator quickstart

This directory contains the validator runtime package plus an operator-ready Docker Compose stack (`validator` + `watchtower`).

## Configure

1) Create your env file:
```bash
cp .env.example .env
```

2) Edit `.env` and set at least:
- `PLATFORM_BASE_URL` (platform API endpoint)
- `VALIDATOR_PUBLIC_BASE_URL` (how the platform can reach your validator)
- `CHUTES_API_KEY` (LLM calls)
- `DESEARCH_API_KEY` (search tools)
- `SCORING_LLM_MODEL` (scoring model; default `openai/gpt-oss-20b`)

The defaults in `.env.example` already target mainnet (`finney`) and netuid `67`, and use `castersubnet/caster-subnet-sandbox:latest` for sandbox execution.

Wallet notes:
- If you already have a hotkey file in `~/.bittensor/wallets`, set `SUBTENSOR_WALLET_NAME` / `SUBTENSOR_HOTKEY_NAME` to match.
- If you *don’t* have a hotkey file yet, set `SUBTENSOR_HOTKEY_MNEMONIC` to generate it on first start.

## Run

- Start/update: `bash scripts/operator_up.sh`
- Logs: `bash scripts/operator_logs.sh`
- Stop: `bash scripts/operator_down.sh`

## Auto-updates (Watchtower)

Watchtower polls Docker Hub every 5 minutes and will pull/restart the validator when `CASTER_VALIDATOR_IMAGE` changes.

The stack is configured for a graceful restart window (default `30m`) so in-progress evaluations can finish before the container is restarted.
