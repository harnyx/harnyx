---
name: configure-before-submit
description: Check and update miner config before submitting a script. Use before uploading an artifact that needs provider credentials or intentional retry behavior.
---

# Configure Before Submit

## Goal

Verify the signing hotkey's platform-stored miner config before uploading a
script.

## Inputs

- wallet name
- hotkey name
- provider credentials required by the script
- optional retry count from `0` through `3`

## Steps

1. Read current config:

```bash
uv run --package harnyx-miner harnyx-miner-config \
  --wallet-name <wallet> \
  --hotkey-name <hotkey> \
  --get
```

2. Set each provider credential the script needs:

```bash
uv run --package harnyx-miner harnyx-miner-config \
  --wallet-name <wallet> \
  --hotkey-name <hotkey> \
  --provider <provider> \
  --api-key <provider-api-key>
```

3. Use only supported providers: `chutes`, `openrouter`, `ai_gateway`, `desearch`, `parallel`.
4. If repeated `429` errors or provider instability are isolated to an upstream
   provider selected through OpenRouter, consider using an OpenRouter API key
   from a workspace with that provider configured through OpenRouter BYOK. BYOK
   is configured in OpenRouter, not in `harnyx-miner-config`: store the
   OpenRouter API key in miner config, and manage the upstream provider key in
   the OpenRouter workspace. This often helps when shared OpenRouter capacity
   for that provider is unstable because the provider account owns its own rate
   limits and costs.
5. Set retry behavior only when intentional:

```bash
uv run --package harnyx-miner harnyx-miner-config \
  --wallet-name <wallet> \
  --hotkey-name <hotkey> \
  --task-retry-count <0-3>
```

6. Re-read config and confirm expected provider statuses and retry count.

## Stop Conditions

- Stop if wallet or hotkey is uncertain.
- Stop if a required provider credential is missing.
- Stop if retry behavior is not intentional.

## Output

- wallet and hotkey used
- provider credential statuses
- retry count decision
