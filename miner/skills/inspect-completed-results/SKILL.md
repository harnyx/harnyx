---
name: inspect-completed-results
description: Inspect completed batch comparison and artifact-scoped result rows. Use after a source batch is completed and artifact/result rows are public.
---

# Inspect Completed Results

## Goal

Collect completed-batch evidence for one submitted artifact.

## Inputs

- `batch_id`
- `artifact_id`
- MCP tools:
  - `get_miner_task_batch`
  - `get_miner_task_batch_comparison`
  - `get_miner_task_batch_results`
  - `get_task_results`

## Steps

1. Call `get_miner_task_batch(batch_id)` and confirm the batch is completed.
2. Read `batch.tasks[]` for `task_id`, query, and `reference_answer`.
3. Call `get_miner_task_batch_comparison(batch_id)` for aggregate score,
   cost totals, and `error_counts`.
4. Call `get_miner_task_batch_results(batch_id, artifact_id, ...)` for
   artifact-scoped result rows.
5. Call `get_task_results(batch_id, artifact_id, task_id)` when attempts or
   `execution_log` detail are needed for one task.
6. Join task metadata and result rows by `task_id`.

## Stop Conditions

- Stop if the batch is not completed.
- Stop if `artifact_id` is unknown; find it from submit evidence or completed
  batch artifact visibility first.

## Output

- batch task metadata
- aggregate comparison
- artifact-scoped result rows
- attempt and execution-log evidence when needed
