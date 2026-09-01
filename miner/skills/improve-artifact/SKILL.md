---
name: improve-artifact
description: Change a miner artifact using hypotheses from local-eval reports.
---

# Improve Artifact

## Goal

Make a focused change to `./agent.py` based on evidence from a local-eval
report.

## Inputs

- target artifact path, usually `./agent.py`
- task-level problems
- improvement hypotheses

## Steps

1. Start from one or two specific hypotheses, not a full rewrite.
2. Keep the public contract stable:
   - one Python file
   - `@entrypoint("query")`
   - `async def query(query: Query, context: ContextSnapshot) -> Response`
3. Prefer changes that can be tested against the failing or weak tasks you already identified.
4. Preserve graceful failure behavior for tools and budgets.
5. Record what changed and what you expect to improve in the next report.

## Stop Conditions

- Stop if there is no specific hypothesis.
- Stop if the change would rewrite the artifact without task-level evidence.
- Stop if the change breaks the one-file
  `query(Query, ContextSnapshot) -> Response` contract.

## Output

- updated artifact
- expected task-level impact
- risks to watch in the next local-eval run
