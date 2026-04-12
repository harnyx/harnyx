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
- still includes recorded platform batch context in the report
- useful when you want a quick iteration loop or when the batch has no champion artifact

## Batch Source

The command can:

- discover the latest completed public batch
- fetch a specific public batch by id
- fetch the recorded batch detail, artifact metadata, and recorded result rows needed for comparison

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
- recorded platform context for comparison
- per-task details

The evaluation config snapshot now also records the sandbox execution boundary, sandbox image, local tool-host mode, and the task-level / artifact-level parallelism used for the run.

The JSON report is the machine-readable source of truth for automated analysis. The Markdown report is the human-readable summary.

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
2. Look for repeated patterns: missing evidence, weak synthesis, over-spending, brittle tool handling, slow answers.
3. Change `./agent.py` to test one or two specific hypotheses.
4. Re-run local eval.
5. Compare the new JSON/Markdown report with the previous run.

## Fresh Start

If `./agent.py` does not exist yet, create a minimal working stub first and then start the loop:

```python
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import Query, Response


@entrypoint("query")
async def query(query: Query) -> Response:
    return Response(text=query.text)
```

## Public Workflow Skills

If you are using a code agent, use the public step-based skills in [`skills/README.md`](skills/README.md):

- `prepare-benchmark-context`
- `improve-artifact`
- `run-local-eval`
