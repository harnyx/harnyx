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
- `get_benchmark`
- `get_latest_submissions`
- `get_miner_script`
- `list_miner_task_batches`
- `get_miner_task_batch`
- `get_miner_task_batch_comparison`
- `get_miner_task_batch_results`
- `get_task_results`

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

Supported providers are `chutes`, `openrouter`, `desearch`, and `parallel`.

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
2. Call MCP `get_latest_submissions`.
3. Find the returned `artifact_id` and `content_hash`.
4. Record `artifact_id`, `content_hash`, the signing hotkey, and submit time.

`get_latest_submissions` confirms platform acceptance. It does not prove that a
running batch has already exposed artifact or result rows.

## Monitor Waiting, Running, And Completed Batches

Use `list_miner_task_batches` to find recent source batches, then
`get_miner_task_batch(batch_id)` for details.

### Waiting For The Next Batch

Check:

- `get_latest_submissions` contains your `artifact_id` and `content_hash`
- your `submitted_at`
- the candidate batch `cutoff_at`

If `submitted_at` is after `cutoff_at`, expect to wait for a later batch.

### While A Batch Is Running

Use `get_miner_task_batch(batch_id)` for:

- batch state
- delivery state
- delivery progress
- validator progress and last error fields when present

Do not expect public artifact rows, task result rows, miner responses, reference
answers, or script content before the source batch is completed.

### After A Batch Completes

Use completed-batch tools by purpose:

- `get_miner_task_batch_comparison(batch_id)` for aggregate comparison,
  artifact totals, scores, cost totals, and `error_counts`.
- `get_miner_task_batch_results(batch_id, artifact_id, ...)` for
  artifact-scoped result rows. Add `task_id`, `validator_hotkey`, or
  `miner_uid` only when narrowing the query.
- `get_task_results(batch_id, artifact_id, task_id)` for full result detail,
  attempts, and `execution_log` evidence for one task.

## Diagnose Bad Or Weird Scores

Work from the first failed condition that applies.

| Symptom | Check | Tool or Source |
|---------|-------|----------------|
| Upload not accepted | Submit response, local hash, signing wallet/hotkey, latest submission metadata | `harnyx-miner-submit`, `harnyx-miner-hash`, `get_latest_submissions` |
| Accepted but not in completed batch | `submitted_at` versus `cutoff_at`, completed batch artifact/result visibility | `get_latest_submissions`, `get_miner_task_batch` |
| Batch still running | Delivery state and progress only; do not look for public result rows yet | `get_miner_task_batch` |
| Execution did not happen | Delivery state/progress, then aggregate `error_counts` after completion | `get_miner_task_batch`, `get_miner_task_batch_comparison` |
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
