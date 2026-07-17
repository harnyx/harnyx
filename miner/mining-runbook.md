# Mining Runbook

Use this runbook when you are operating a miner workflow from a local candidate
script through platform submission, batch monitoring, completed-result analysis,
and the next improvement decision.

The runbook is written for both humans and code agents. Follow the sections in
order when starting a new candidate. Jump to the diagnosis section when a score,
batch state, or submitted artifact looks wrong.

Code agents can also load the step-level workflow skills in
[`skills/`](skills/). Harnesses that support skill discovery can read
[`skills/manifest.json`](skills/manifest.json), then load the matching
`SKILL.md` file for the current workflow step.

## Connect To Platform Monitoring MCP

Configure an MCP client with this Streamable HTTP endpoint:

```text
https://api.harnyx.ai/mcp
```

Use your MCP client's normal session initialization and tool listing. Raw MCP
transport implementation is outside this miner workflow runbook; prefer a normal
MCP client.

Workflow tools used below:

- `get_champion`
- `get_validators`
- `get_benchmark`
- `get_latest_submissions`
- `get_miner_script`
- `list_miner_task_batches`
- `get_miner_task_batch`
- `get_miner_task_batch_comparison`
- `list_miner_task_batch_artifact_comparisons`
- `get_miner_task_batch_artifact_comparison`
- `list_miner_task_batch_challenger_steps`
- `get_miner_task_batch_challenger_step`
- `get_miner_task_batch_similarity_round`
- `get_miner_task_batch_results`
- `get_task_results`

For `get_miner_task_batch_results` and `get_task_results`, read the returned
rows from `results[]`.

Call `get_validators` and read `runtime.next_scheduled_batch_at` when you need
the next configured batch time. The value is UTC on the wire; `null` means
automatic miner-task batch scheduling is disabled.

## Start From Current Champion Context

First, get the current public champion context:

1. Call `get_champion`.
2. If `champion.script_id` is present, call `get_miner_script` with:
   - `artifact_id=<champion.script_id>`
   - `include_content=true`
3. If you need source-batch context instead, call `get_benchmark` and inspect:
   - `current_champion.script_id`
   - `latest_source_batch.champion_artifact_id`
4. If script content is revealed, decode `content_b64` and use it as a baseline
   to inspect, compare, or seed `agent.py`.
5. If script content is not available, start from local eval against the latest
   completed batch instead.

Champion code is a baseline, not finished work. The useful question is what it
does well, where it fails, and which focused change you can test next.

## Improve Locally Before Submit

Run local eval before uploading unless you are only testing platform plumbing:

```bash
uv run --package harnyx-miner harnyx-miner-local-eval --agent-path ./agent.py
```

Use a specific completed batch when investigating a known comparison:

```bash
uv run --package harnyx-miner harnyx-miner-local-eval \
  --agent-path ./agent.py \
  --batch-id <completed-batch-id>
```

Read the JSON report first. Classify losses by repeated cause:

- missing evidence
- weak synthesis
- unsupported claims
- slow answers
- over-spending
- retry or tool handling failures
- script exceptions

Make one focused change, re-run local eval, and compare reports before deciding
to submit.

## Configure Before Submit

Use the same wallet and hotkey that will sign the script upload.

Read current platform-stored miner config:

```bash
uv run --package harnyx-miner harnyx-miner-config \
  --wallet-name <wallet> \
  --hotkey-name <hotkey> \
  --get
```

Set provider credentials that your script needs during validator execution:

```bash
uv run --package harnyx-miner harnyx-miner-config \
  --wallet-name <wallet> \
  --hotkey-name <hotkey> \
  --provider chutes \
  --api-key <provider-api-key>
```

Supported providers are `chutes`, `openrouter`, `ai_gateway`, `desearch`, and `parallel`.

If repeated `429` errors or provider instability are isolated to an upstream
provider selected through OpenRouter, consider using an OpenRouter API key from
a workspace with that provider configured through OpenRouter BYOK. BYOK is
configured in OpenRouter, not in `harnyx-miner-config`: store the OpenRouter
API key in miner config, and manage the upstream provider key in the OpenRouter
workspace. This often helps when shared OpenRouter capacity for that provider is
unstable because the provider account owns its own rate limits and costs.

Set retry behavior only when it is intentional:

```bash
uv run --package harnyx-miner harnyx-miner-config \
  --wallet-name <wallet> \
  --hotkey-name <hotkey> \
  --task-retry-count <0-3>
```

## Submit And Confirm Acceptance

Set the production platform base URL:

```bash
export PLATFORM_BASE_URL="https://api.harnyx.ai"
```

Hash the local script:

```bash
uv run --package harnyx-miner harnyx-miner-hash --agent-path ./agent.py
```

Upload the script:

```bash
uv run --package harnyx-miner harnyx-miner-submit \
  --agent-path ./agent.py \
  --wallet-name <wallet> \
  --hotkey-name <hotkey>
```

After submit:

1. Compare the returned `content_hash` with the local hash.
2. Call MCP `get_latest_submissions`; if the artifact has already moved, inspect finalized `initializing` or `running` batch detail.
3. Find the returned `artifact_id`, miner hotkey, and `content_hash` in one of those two disjoint views.
4. Record `artifact_id`, `content_hash`, the signing hotkey, and submit time.

Current candidates or finalized non-terminal batch membership confirm accepted
identity metadata. Neither view exposes script, task, or result content.

## Monitor Waiting, Running, And Completed Batches

Use `list_miner_task_batches` to find recent source batches, then
`get_miner_task_batch(batch_id)` for details.

### Waiting For The Next Batch

Check:

- `get_latest_submissions` contains your `artifact_id`, miner hotkey, and `content_hash`, or a finalized non-terminal batch already contains the same identity metadata
- your `submitted_at`
- the candidate batch `cutoff_at`

If `submitted_at` is after `cutoff_at`, expect to wait for a later batch.

### While A Batch Is Running

Use `get_miner_task_batch(batch_id)` for:

- batch state
- finalized artifact membership as UID, hotkey, and script hash
- delivery state
- delivery progress
- validator progress and last error fields when present

Every challenger in finalized membership was considered by duplicate preflight,
but some artifacts receive no scoring tasks. Do not expect task rows, task result
rows, miner responses, reference answers, or script content before completion.

### After A Batch Completes

Use completed-batch tools by purpose:

- `get_miner_task_batch_comparison(batch_id)` for the batch outcome and total
  costs without downloading artifact, challenger-step, or duplicate-vote collections.
- `list_miner_task_batch_artifact_comparisons(batch_id)` for the compact artifact
  index, then `get_miner_task_batch_artifact_comparison(batch_id, artifact_id)`
  for one artifact's scores, costs, observed work, `error_counts`, and per-model
  LLM usage.
- `list_miner_task_batch_challenger_steps(batch_id)` for the compact champion
  sequence, then `get_miner_task_batch_challenger_step(batch_id, step_number)`
  for one step's rule evaluations.
- `get_miner_task_batch_similarity_round(batch_id, candidate_artifact_id)` only
  when full validator votes, reasoning, errors, and judge usage are needed for
  one candidate.
- `get_miner_task_batch_results(batch_id, artifact_id, ...)` for
  artifact-scoped result rows in `results[]`. Add `task_id`, `validator_hotkey`, or
  `miner_uid` only when narrowing the query.
- `get_task_results(batch_id, artifact_id, task_id)` for full result detail,
  attempts, and `execution_log` evidence for one task in `results[]`.

## Diagnose Bad Or Weird Scores

Work from the first failed condition that applies.

| Symptom | Check | Tool or Source |
|---------|-------|----------------|
| Upload not accepted | Submit response, local hash, signing wallet/hotkey, current candidate or finalized non-terminal membership metadata | `harnyx-miner-submit`, `harnyx-miner-hash`, `get_latest_submissions`, `get_miner_task_batch` |
| Accepted but not in completed batch | `submitted_at` versus `cutoff_at`, current candidate or finalized non-terminal membership, completed batch artifact/result visibility | `get_latest_submissions`, `get_miner_task_batch` |
| Batch still running | Delivery state and progress only; do not look for public result rows yet | `get_miner_task_batch` |
| Execution did not happen | Delivery state/progress, then the selected artifact's `error_counts` after completion | `get_miner_task_batch`, `get_miner_task_batch_artifact_comparison` |
| Timeout, crash, or budget issue | Attempts, `elapsed_ms`, `execution_log`, `specifics.error`, cost totals | `get_miner_task_batch_results`, then `get_task_results` for the chosen `task_id` |
| Weak answer | Reference answer from batch task metadata; miner response/citations/score details from artifact result rows joined by `task_id` | `get_miner_task_batch`, `get_miner_task_batch_results`, then `get_task_results` when full task detail is needed |
| Judge rationale is surprising | `specifics.score_breakdown.reasoning.text` when present | `get_miner_task_batch_results` or `get_task_results` |
| Need direct target-vs-champion answer comparison | Re-run local eval in `vs-champion` mode and compare the report's `target` and `opponent` fields | `harnyx-miner-local-eval` |

For weak answers, join the data like this:

1. Use `get_miner_task_batch(batch_id).batch.tasks[]` for `task_id` and
   `reference_answer`.
2. Use artifact-scoped result rows for `response`, `citations`, and
   `specifics.score_breakdown`.
3. Join task metadata and result rows by `task_id`.
4. When you need full attempts or execution logs, call
   `get_task_results(batch_id, artifact_id, task_id)` for that one task.

Then choose the next action:

- config fix
- script robustness fix
- retrieval fix
- synthesis fix
- retry policy change
- wait for the next eligible batch

## Preserve Debug Evidence

Before changing direction, record:

- `artifact_id`
- `content_hash`
- `batch_id`
- `task_id` examples
- signing hotkey
- observed errors
- the diagnosis already attempted

Keep this evidence with your local eval reports so the next iteration can start
from the actual failure instead of from memory.
