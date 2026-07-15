---
name: connect-platform-mcp
description: Connect to the public platform monitoring MCP and verify workflow tools are available. Use before champion lookup, submission verification, batch monitoring, or score diagnosis through MCP.
---

# Connect Platform MCP

## Goal

Configure an MCP client for the public platform monitoring endpoint and confirm
the workflow tools are available.

## Inputs

- MCP client or harness support for Streamable HTTP MCP
- endpoint: `https://api.harnyx.ai/mcp`

## Steps

1. Configure the MCP client with `https://api.harnyx.ai/mcp`.
2. Use the client's normal session initialization.
3. List available tools.
4. Confirm these workflow tools are present:
   - `get_champion`
   - `get_validators`
   - `get_benchmark`
   - `get_latest_submissions`
   - `get_miner_script`
   - `list_miner_task_batches`
   - `get_miner_task_batch`
   - `get_miner_task_batch_comparison`
   - `get_miner_task_batch_results`
   - `get_task_results`

## Stop Conditions

- Stop if the MCP client cannot initialize.
- Stop if the needed tool is not listed.
- Do not implement raw MCP transport inside the miner workflow when a normal MCP
  client is available.

## Output

- endpoint used
- available workflow tools
- missing tools, if any
