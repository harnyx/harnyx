---
name: run-local-eval
description: Run miner local batch evaluation and extract the next iteration signal.
---

# Run Local Eval

## Goal

Run `harnyx-miner-local-eval`, collect the reports, and decide the next move.

## Inputs

- target artifact path, usually `./agent.py`
- mode: `vs-champion` or `target-only`
- optional completed `batch_id`
- log detail: default logs or `LOG_LEVEL=DEBUG` for tool-call diagnosis

## Steps

1. Choose the mode:
   - `vs-champion` for the default comparison loop
   - `target-only` for a quicker isolated check
2. Run:

```bash
uv run --package harnyx-miner harnyx-miner-local-eval --agent-path ./agent.py
```

   If a report does not explain a weak or failed task, pin its batch and capture
   DEBUG logs. Prefer `target-only` when diagnosing only the target artifact:

```bash
LOG_LEVEL=DEBUG uv run --package harnyx-miner harnyx-miner-local-eval \
  --agent-path ./agent.py \
  --batch-id <completed-batch-id> \
  --mode target-only \
  > local-eval-paths.json \
  2> local-eval-debug.log
```

3. Read:
   - `local-eval-report-<batch-id>-<mode>.json`
   - `local-eval-report-<batch-id>-<mode>.md`
   - `local-eval-debug.log` when DEBUG logging was enabled
4. Extract:
   - score deltas
   - task-level wins / losses
   - cost regressions
   - retries / failures
   - tool requests, responses, and errors for the selected `task_id` and `attempt`
5. Decide whether to:
   - keep iterating on the artifact
   - switch to `target-only`
   - submit the current artifact

## Stop Conditions

- Stop if Docker or required evaluation environment variables are missing.
- Stop if the artifact cannot load.
- Stop if the report was not written; inspect the failure bundle before editing.

## Output

- command used
- report paths
- DEBUG log path when captured
- key findings from the run
- next recommended action
