---
name: diagnose-score
description: Diagnose weak, missing, failed, or surprising miner scores from public MCP evidence. Use when a submitted artifact has a confusing score, missing result, timeout, crash, or weak answer.
---

# Diagnose Score

## Goal

Classify why an artifact scored poorly, failed, or looks missing, then choose
the next workflow action.

## Inputs

- `artifact_id`
- `content_hash`
- `batch_id`
- task examples when available
- completed-batch MCP evidence

## Steps

1. If upload acceptance is uncertain, check submit response, local hash, signing wallet/hotkey, `get_latest_submissions`, and finalized non-terminal batch membership.
2. If the artifact is absent from a completed batch, compare `submitted_at` with
   `cutoff_at`.
3. If the batch is still running, confirm UID/hotkey/hash membership and inspect delivery state/progress. Membership proves duplicate-preflight consideration, not scoring-task emission; wait for completion before reading result rows.
4. If execution did not happen, inspect:
   - delivery state/progress from `get_miner_task_batch(batch_id)`
   - selected-artifact `error_counts` from
     `get_miner_task_batch_artifact_comparison(batch_id, artifact_id)`
5. If timeout, crash, or budget is suspected, inspect attempts, `elapsed_ms`,
   `execution_log`, `specifics.error`, and cost totals:
   - use `get_miner_task_batch_results(batch_id, artifact_id, ...)` to find the
     affected `task_id`
   - call `get_task_results(batch_id, artifact_id, task_id)` for full task
     result detail
6. If the answer is weak:
   - read `reference_answer` from `get_miner_task_batch(batch_id).batch.tasks[]`
   - read `response`, `citations`, and `specifics.score_breakdown` from
     artifact-scoped result rows
   - join by `task_id`
7. If judge rationale is surprising, read
   `specifics.score_breakdown.reasoning.text` when present.
8. If direct target-vs-champion answer comparison is needed, run local eval in
   `vs-champion` mode and compare the report's `target` and `opponent` fields.

## Stop Conditions

- Stop if the batch is not completed and the question requires result rows.
- Stop if required identifiers are missing; preserve evidence first.

## Output

- diagnosis category
- evidence used
- next action: config fix, script robustness fix, retrieval fix, synthesis fix,
  retry policy change, or wait for the next eligible batch
