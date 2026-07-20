# Local Eval Workflow

`harnyx-miner-local-eval` lets you evaluate a local artifact such as `./agent.py` against a completed public miner-task batch from your own machine.

This is the detailed local-eval guide linked from [`README.md`](README.md).

## Prerequisites

- Docker must be installed and available on your machine.
- The sandbox image configured by `SANDBOX_IMAGE` must be pullable or already present locally.
- `PLATFORM_BASE_URL` must be configured so the CLI can resolve public batches and fetch recorded artifact context.
- `CHUTES_API_KEY` must be configured for evaluation scoring and for agents that call `llm_chat`.
- Search-tool configuration is only required if your agent uses search tools:
  - `SEARCH_PROVIDER`
  - `DESEARCH_API_KEY`

Tool-free agents can create the local-eval runtime without search configuration.

The checked-in default is `SEARCH_PROVIDER=desearch`. If you need a fallback search provider, miner tooling also supports `parallel`; set `SEARCH_PROVIDER=parallel` and `PARALLEL_API_KEY`.

## Quick Start

Latest completed public batch, default mode:

```bash
uv run --package harnyx-miner harnyx-miner-local-eval --agent-path ./agent.py
```

Specific batch:

```bash
uv run --package harnyx-miner harnyx-miner-local-eval \
  --agent-path ./agent.py \
  --batch-id <batch-id>
```

Target only:

```bash
uv run --package harnyx-miner harnyx-miner-local-eval \
  --agent-path ./agent.py \
  --mode target-only
```

## Modes

### `vs-champion`

- default mode
- runs your local artifact
- runs the recorded champion artifact for the selected batch
- writes raw local head-to-head totals
- writes a local simulated champion-selection result using the platform ranking cascade over the local cohort

### `target-only`

- runs only your local artifact
- still includes bounded recorded platform batch context in the report when a champion artifact is available
- useful when you want a quick iteration loop or when the batch has no champion artifact

## Batch Source

The command can:

- discover the latest completed public batch
- fetch a specific public batch by id
- fetch the recorded batch detail, artifact metadata, and champion-artifact recorded result rows needed for comparison
- continue with degraded recorded-platform context when batch detail succeeds but the public artifact-results endpoint is temporarily unavailable

## Execution Boundary

- local eval now stages both your target artifact and the fetched champion artifact into short-lived Docker sandboxes
- the CLI starts a short-lived local HTTP tool host so sandboxed runs can call the normal tool contract
- fetched champion code is not executed in the host Python process during `vs-champion`
- task execution within an artifact now uses the same validator-style sandbox worker parallelism as validator runtime
- in `vs-champion`, target and champion evaluations can run concurrently in separate sandboxes
- scoring, retries, budgeting, and report generation still reuse the shared evaluation path

## Output

By default, the command writes both reports to the current working directory:

- `local-eval-report-<batch-id>-<mode>.json`
- `local-eval-report-<batch-id>-<mode>.md`

During the run, the CLI prints human progress logs to `stderr` so you can see batch selection, runtime startup, and task completion progress. The final report-path summary remains machine-readable JSON on `stdout`.

If sandbox startup or the evaluated agent fails, the CLI writes a failure bundle under `/tmp/harnyx-local-eval-failures/<run-id>/...` before cleanup and prints the failure category plus bundle path to `stderr`. The bundle includes the evaluated `agent.py`, local-eval context, redacted sandbox options, redacted Docker run arguments, and Docker inspect/log output when Docker created a container.

## Use DEBUG Logs To Inspect Tool Calls

The default `LOG_LEVEL` is `WARNING`. Set `LOG_LEVEL=DEBUG` when you need the
detailed tool-call execution log for a diagnosis. DEBUG events include the tool
name, request, response, usage, cost, budget, elapsed time, and error details,
with identifiers such as `task_id`, `attempt`, and `call_id` for correlation.

Pin the batch while comparing changes. Use `target-only` when you only need to
inspect your artifact and want to avoid interleaved champion logs:

```bash
LOG_LEVEL=DEBUG uv run --package harnyx-miner harnyx-miner-local-eval \
  --agent-path ./agent.py \
  --batch-id <completed-batch-id> \
  --mode target-only \
  > local-eval-paths.json \
  2> local-eval-debug.log
```

`local-eval-paths.json` contains the machine-readable paths to the generated
reports. `local-eval-debug.log` contains DEBUG events such as
`miner_tool_call.started`, `miner_tool_call.completed`, and
`miner_tool_call.failed`.

Use the DEBUG log together with the JSON report:

1. Start with a weak or failed task in the report.
2. Filter the DEBUG log by its `task_id` and `attempt`.
3. Compare the tool request, returned evidence, cost, and error details with the
   final answer and score details.
4. Change one behavior in `agent.py`, then rerun the same pinned batch.

DEBUG logging changes log verbosity only; it does not change evaluation or add
the tool-call execution log to the generated reports. Logs can contain task
content, retrieved pages, and model responses, so review them before sharing.

## What The Reports Contain

Both reports include:

- batch metadata
- selected mode
- target, champion, and batch identifiers
- evaluation config snapshot
- scoring-config context
- local leaderboard
- local simulated champion-selection summary
- raw head-to-head comparison in `vs-champion`
- bounded recorded platform context for comparison
- explicit recorded-results availability metadata when champion-artifact recorded rows are unavailable
- per-task details

The evaluation config snapshot now also records the sandbox execution boundary, sandbox image, local tool-host mode, and the task-level / artifact-level parallelism used for the run.

The JSON report is the machine-readable source of truth for automated analysis. The Markdown report is the human-readable summary.

If batch detail resolves but champion-artifact recorded monitoring rows cannot be fetched, local eval still completes the local run and writes a degraded report:

- `recorded_platform_context.results` is `null`
- `recorded_platform_context.results_status` explains the outage
- `recorded_platform_context.results_scope` is `null`
- per-task `recorded_platform_rows` are marked unavailable instead of pretending zero rows were fetched

## Local Selection Semantics

The report contains two different local comparison views:

- raw head-to-head totals, wins, losses, and ties for quick analysis
- a local simulated champion-selection result that applies the same ranking cascade the platform uses for champion dethroning

That simulated champion result is still **local**, not the platform's official winner election. The platform uses a successful-validator cohort. Local eval uses the cohort available on your machine for that run.

## Per-Task Detail

Each task record includes enough detail to drive your own analysis loop:

- question
- reference answer / reference context when available
- target answer
- opponent answer in `vs-champion`
- score and score details
- cost and token usage
- provider/model usage
- elapsed time
- retries / attempt count
- errors when present

## How To Use The Reports

Recommended loop:

1. Find the tasks where your artifact lost to the champion or scored poorly.
2. When the report is not enough to explain the result, rerun the same batch
   with [DEBUG tool-call logs](#use-debug-logs-to-inspect-tool-calls).
3. Look for repeated patterns: missing evidence, weak synthesis, over-spending, brittle tool handling, slow answers.
4. Change `./agent.py` to test one or two specific hypotheses.
5. Re-run local eval.
6. Compare the new JSON/Markdown report and DEBUG evidence with the previous run.

After local eval shows a candidate is worth submitting, or after it exposes a
failure pattern you need to diagnose against platform state, use the
[Mining Runbook](mining-runbook.md). It covers config checks, submit
verification, batch eligibility, completed-result inspection, and score
diagnosis.

## Fresh Start

If `./agent.py` does not exist yet, create a minimal working stub first and then start the loop:

```python
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response


@entrypoint("query")
async def query(query: Query) -> Response:
    return Response(text=query.text)
```

## Optional Public Workflow Skills

If your code agent supports this repository's public step-based skills, they can
help with the local improvement loop. They do not replace the
[Mining Runbook](mining-runbook.md) for submit, monitoring, or post-batch
diagnosis.

- `prepare-benchmark-context`
- `improve-artifact`
- `run-local-eval`
