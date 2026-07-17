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
  - `list_miner_task_batch_artifact_comparisons`
  - `get_miner_task_batch_artifact_comparison`
  - `get_miner_task_batch_challenger_step`
  - `get_miner_task_batch_similarity_round`
  - `get_miner_task_batch_results`
  - `get_task_results`

## Steps

1. Call `get_miner_task_batch(batch_id)` and confirm the batch is completed.
2. Read `batch.tasks[]` for `task_id`, query, and `reference_answer`.
3. Call `get_miner_task_batch_comparison(batch_id)` for batch outcome and total
   costs, without downloading all artifact or vote evidence.
4. Call `get_miner_task_batch_artifact_comparison(batch_id, artifact_id)` for
   the selected artifact's scores, costs, observed work, and `error_counts`.
5. Call `get_miner_task_batch_challenger_step(batch_id, step_number)` or
   `get_miner_task_batch_similarity_round(batch_id, artifact_id)` only when the
   selected artifact needs rule-evaluation or duplicate-vote evidence.
6. Call `get_miner_task_batch_results(batch_id, artifact_id, ...)` for
   artifact-scoped result rows, then read those rows from `results[]`.
7. Call `get_task_results(batch_id, artifact_id, task_id)` when attempts or
   `execution_log` detail are needed for one task, then read those rows from
   `results[]`.
8. Join task metadata and result rows by `task_id`.

## Stop Conditions

- Stop if the batch is not completed.
- Stop if `artifact_id` is unknown; find it from submit evidence or completed
  batch artifact visibility first.

## Output

- batch task metadata
- compact batch comparison and selected artifact comparison
- selected challenger-step or similarity-round evidence when needed
- artifact-scoped result rows
- attempt and execution-log evidence when needed
