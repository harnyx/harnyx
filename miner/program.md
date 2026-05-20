# autoresearch

This is an experiment to have an LLM improve the Harnyx miner agent by running fixed local evaluation and keeping only changes that improve the scores.

The format is intentionally the upstream-style autoresearch format: this `program.md` file is the research policy, `prepare.py` is fixed support, `train.py` is the only editable candidate, `run.log` captures evaluator output, and `results.tsv` records kept, discarded, and crashed experiments.

The agent must behave like a rigorous research engineer, not like a shallow benchmark hill-climber.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date. The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: create `autoresearch/<tag>` from the current public miner branch.
3. **Work from this miner directory**. Run the commands below from the directory that contains `program.md`, `prepare.py`, and `train.py`.
4. **Read the in-scope files**. The autoresearch surface is intentionally small:
   - `README.md` - miner context.
   - `prepare.py` - fixed batch pinning, local-eval invocation, and report parsing. Do not modify.
   - `train.py` - the only file you modify. This is the miner agent and experiment command.
5. **Pin the evaluators**: run `uv run prepare.py --benchmark-suite <suite-slug>`. If the user wants a specific completed batch, run `uv run prepare.py --benchmark-suite <suite-slug> --batch-id <batch-id>`.
6. **Initialize results.tsv** with just the header row:

```
commit	score_a	score_b	cost_usd	status	description
```

7. **Initialize the rich research ledger** at `.autoresearch/experiment-ledger.md`. This file is untracked and complements `results.tsv`; it does not replace it.
8. **Confirm and go**: confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Candidate constraints

The final kept harness must respect these constraints:

- Modify only `train.py`.
- Keep `train.py` as a self-contained single Python miner agent with `@entrypoint("query")`.
- Use only the provided `llm_chat` and `search_web` tools in the final candidate.
- Do not install packages or add dependencies.
- Do not modify `prepare.py`, local-eval implementation, benchmark implementation, or submission/upload implementation.
- Do not modify files other than `train.py` during experiments, except untracked generated artifacts such as `run.log`, `results.tsv`, and `.autoresearch/experiment-ledger.md`.
- Do not hardcode benchmark item IDs, exact benchmark questions, exact diagnostic questions, sources, queries, or answer lookup tables in `train.py`.
- Do not upload automatically.

Debug or oracle experiments are allowed during research when they help isolate a mechanism, but they must not be kept in the final candidate. Remove any debug-only code, oracle answer access, or diagnostic-only shortcut before a keep decision.

## Fixed evaluators

Each committed full experiment evaluates `train.py` against the pinned local-eval batch and the pinned benchmark snapshot selected by `prepare.py`. Launch it simply as:

```
uv run train.py
```

Score A is the primary local batch-eval score against the current champion. Score B is the selected benchmark score against open canonical answers. Neither score is a perfect generalization test, but the pair is the fixed metric for this run.

Cost and complexity are soft constraints. A cost increase is acceptable for a meaningful score gain, but do not make the agent expensive or fragile for tiny gains. All else equal, simpler is better. A small improvement from deleting code is valuable. A tiny score gain from a large brittle rewrite is usually not worth keeping.

The first run should establish the baseline by running `train.py` as is.

## Output format

Once the script finishes, it prints a summary like this:

```
---
score_a:              0.670000
score_b:              0.550000
champion_score_a:     0.600000
delta_vs_champion_a:  0.070000
total_seconds:        410.2
cost_usd:             0.035000
local_eval_cost_usd:  0.020000
benchmark_cost_usd:   0.015000
error_count:          0
batch_id:             00000000-0000-0000-0000-000000000001
benchmark:            deepsearchqa dataset_version=... scoring_version=...
local_eval_json_report:  .autoresearch/reports/20260430-120000/local-eval/local-eval-report-...
benchmark_json_report:   .autoresearch/reports/20260430-120000/benchmark/local-benchmark-report-...
```

Extract the key metrics from the log file:

```
grep "^score_a:\|^score_b:\|^champion_score_a:\|^delta_vs_champion_a:\|^cost_usd:\|^error_count:" run.log
```

## Research policy

Do not randomly tweak the harness, run the full evaluation after every tiny change, abandon an idea after one failed score, overfit one diagnostic case with wording, summarize external research without turning it into an experiment, or move to a new idea before understanding the current failure.

### Start from failures

Before changing code, inspect concrete weak or failed cases. Use the latest `run.log`, structured local-eval report, structured benchmark report, and any diagnostic output you created.

Look at:

- question
- current answer
- search queries
- search results
- selected sources
- extracted evidence, if present
- missing facts
- final answer
- score or reference comparison, if available

Do not rely only on aggregate `score_a` or `score_b`.

### Build a failure taxonomy

Classify observed failures into concrete categories and attach examples whenever possible:

- question interpretation failure
- missing required facts
- failure to detect recency/currentness requirement
- poor search planning
- weak query generation
- insufficient search depth
- poor source selection
- failure to extract key claims
- weak synthesis
- unsupported final claims
- generic answer
- stale evidence
- formatting mismatch
- scoring mismatch
- tool budget waste
- implementation bug

The taxonomy is working memory. Update it in `.autoresearch/experiment-ledger.md` as new evidence arrives.

### Pick one bottleneck per cycle

Each research cycle must target exactly one mechanism-level bottleneck.

Good bottlenecks:

- The harness answers before checking whether required facts are missing.
- The harness retrieves sources but fails to extract decisive claims.
- The harness does not diversify queries for multi-hop questions.
- The harness uses stale sources for current-information questions.
- The harness has no validation or retry loop for missing evidence.

Bad bottlenecks:

- Improve answer quality.
- Make the prompt better.
- Try a new strategy.
- Increase score.

### Write the hypothesis before editing

Before editing `train.py`, write this block in `.autoresearch/experiment-ledger.md`:

```
## Experiment <id>

- Bottleneck:
- Hypothesis:
- Expected observable behavior:
- Minimal change:
- Diagnostic cases:
- Intermediate artifacts to inspect:
- Success criteria:
- What would falsify this hypothesis:
```

Only then edit `train.py`. Do not make unstructured edits.

### Use focused diagnostic cases before full evaluation

For the chosen bottleneck, select 1-3 focused diagnostic cases that expose it. For each diagnostic case, record:

- why this case exposes the bottleneck
- current baseline behavior
- what should change
- which intermediate artifact should prove the fix worked

During active development, run focused diagnostics first. Do not run the full evaluation after every small change.

Focused diagnostics may use existing tools and generated artifacts:

- inspect prior local-eval and benchmark JSON reports
- create temporary request JSON files under `.autoresearch/`
- run `uv run harnyx-miner-dev --agent-path ./train.py --request-json <request.json>` for a small case
- add temporary debug logging or oracle checks in `train.py` only while diagnosing, then remove them before any keep decision
- compare baseline and modified outputs line by line

Full evaluation with `uv run train.py > run.log 2>&1` is allowed only when:

- focused diagnostic cases show clear improvement
- a focused hypothesis cycle is complete
- the change is global and cannot be tested locally
- regression checking is needed after a promising local improvement
- the user explicitly asks for full evaluation

### Inspect intermediate artifacts, not just score

After a change, compare before and after:

- research plan
- required facts
- generated search queries
- retrieved sources
- extracted evidence
- missing facts
- final answer
- score, if available

A score change without a mechanism explanation is not enough. Passing a diagnostic case is not enough unless it passed for the right reason.

Verify:

- Did the harness identify the right missing facts?
- Did it generate better searches?
- Did it retrieve better sources?
- Did it extract the needed evidence?
- Did the final answer use that evidence?
- Would the same mechanism apply to nearby questions?

### Do not abandon a hypothesis after one failed attempt

A failed diagnostic run or score drop is only an observation. Before abandoning a hypothesis, do at least two of:

- verify the new code path actually executed
- inspect logs to see whether intended behavior changed
- simplify the change and test a narrower variant
- create an oracle or debug variant
- test on an even smaller case
- compare baseline vs modified outputs line by line
- determine whether the failure came from implementation, hypothesis, test case, scoring noise, or another component

Before moving on, write a short failure report:

- What did we expect to change?
- Did it change?
- If not, why not?
- If yes, why did quality or score not improve?
- Was the hypothesis wrong, implementation wrong, test wrong, or metric misleading?
- What evidence supports this conclusion?

### Use the intervention ladder

Prompt tuning is the lowest-level intervention, not the default solution. Prompt changes are allowed, but they are not the main solution.

Use the lowest level that reliably fixes the mechanism:

1. **Prompt clarification**: use only when the failure is caused by ambiguous instructions or superficial formatting.
2. **Structured output contract**: use when the model fails to reliably preserve required intermediate information.
3. **Validator plus retry**: use when the model can sometimes produce correct behavior but not consistently.
4. **Control-flow change**: use when the pipeline order is wrong, such as answering before evidence is complete.
5. **Algorithmic strategy change**: use when the whole research strategy is wrong.

If a failure persists after one or two prompt-only attempts, stop prompt-tweaking and analyze the mechanism-level failure.

When prompt-only fixes are brittle, consider:

- explicit research plan contract
- required facts list
- source acceptance criteria
- evidence extraction table
- missing-fact retry loop
- answer-from-evidence constraint
- separate critic or revision pass
- deterministic validation of intermediate outputs
- fallback search strategy
- query diversification strategy
- per-question-type control flow

The goal is not to merely tell the LLM to behave better. The goal is to make the desired behavior structurally more likely or explicitly validated.

### Convert external research into experiments

If you look up papers, blog posts, or agent techniques, do not merely summarize them. Convert each idea into an implementation hypothesis:

- What mechanism is relevant?
- Which observed bottleneck does it address?
- What minimal `train.py` change would test it?
- Which diagnostic case would reveal whether it works?
- What result would convince us to keep or discard it?

## Research ledger

Maintain `.autoresearch/experiment-ledger.md` throughout the run. Each entry should include:

- experiment id
- target bottleneck
- hypothesis
- code change
- diagnostic cases
- observed behavior
- intermediate artifact comparison
- score or result, if available
- interpretation
- decision: keep / refine / revert / escalate / abandon
- next action

The ledger should prevent repeating the same failed ideas without learning.

## Logging results

When a full experiment is done, log it to `results.tsv`. It is tab-separated, not comma-separated.

The TSV has a header row and 6 columns:

```
commit	score_a	score_b	cost_usd	status	description
```

1. git commit hash, short 7 chars
2. Score A achieved, such as `0.670000`; use `0.000000` for crashes
3. Score B achieved, such as `0.550000`; use `0.000000` for crashes
4. combined target total cost in USD, rounded to 6 decimals; use `0.000000` for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	score_a	score_b	cost_usd	status	description
a1b2c3d	0.600000	0.450000	0.020000	keep	baseline
b2c3d4e	0.670000	0.550000	0.021500	keep	add citation-backed synthesis pass
c3d4e5f	0.590000	0.520000	0.025000	discard	over-aggressive query expansion
d4e5f6g	0.000000	0.000000	0.000000	crash	invalid response schema
```

Do not commit `results.tsv`; leave it untracked.

## Keep / discard criteria

Keep a commit when Score A improves meaningfully and Score B does not reveal a serious regression. A Score B improvement can justify keeping a small Score A tie, especially when the change is simple and reduces errors.

Discard a commit when Score A regresses, when Score B collapses, when either evaluator reports errors, or when the change looks like overfitting to benchmark examples instead of improving the general agent.

If a score regresses after diagnostics looked better, do not immediately jump to a new idea. First inspect whether the mechanism improved, whether the change hurt another bottleneck, whether the diagnostic case was too narrow, or whether the score movement looks like evaluator noise.

## The experiment loop

The experiment runs on a dedicated branch such as `autoresearch/apr30` or `autoresearch/apr30-gpu0`.

LOOP FOREVER:

1. Look at the git state: the current branch and commit.
2. Inspect concrete failures from the latest reports, `run.log`, diagnostics, and `results.tsv`.
3. Update the failure taxonomy in `.autoresearch/experiment-ledger.md`.
4. Pick exactly one mechanism-level bottleneck.
5. Write the hypothesis block before editing.
6. Select 1-3 focused diagnostic cases.
7. Edit `train.py` with the minimal change needed to test the hypothesis.
8. Run focused diagnostics and inspect intermediate artifacts.
9. If the intended behavior did not execute or did not change, refine the same hypothesis before full evaluation.
10. If diagnostics show the mechanism improved for the right reason, git commit the `train.py` change.
11. Run the full experiment: `uv run train.py > run.log 2>&1`. Redirect everything; do not let output flood your context.
12. Read out the results with `grep "^score_a:\|^score_b:\|^champion_score_a:\|^delta_vs_champion_a:\|^cost_usd:\|^error_count:" run.log`.
13. If the grep output is empty, the run crashed. Run `tail -n 80 run.log` to read the error and decide whether to fix the implementation bug or abandon the idea.
14. Inspect the structured report paths printed in `run.log`; compare intermediate artifacts and diagnostic behavior against the hypothesis.
15. Record the result in `results.tsv`. Do not commit `results.tsv`; leave it untracked.
16. Update `.autoresearch/experiment-ledger.md` with interpretation, decision, and next action.
17. If the score improved for a plausible mechanism-level reason, advance the branch by keeping the git commit.
18. If the score is equal or worse, write the failure report, then either refine the same hypothesis or reset back to where you started before that experiment.

If a run exceeds the user's expected local-eval time budget, kill it and treat it as a failure.

If a run crashes because of a small bug such as a typo, fix it and rerun. If the idea itself is broken, log `crash`, reset, and move on only after writing the failure report.

## Cycle report format

At the end of each cycle, report:

- Targeted bottleneck
- Diagnostic cases used
- Hypothesis
- Change made
- What changed in behavior
- Evidence from intermediate artifacts
- Score or result, if available
- Keep / refine / revert decision
- Next experiment

Be concrete. Do not report only "score improved" or "score decreased." Explain the mechanism.

**NEVER STOP**: Once the experiment loop has begun, do not pause to ask the human whether to continue. The human expects the loop to continue until manually interrupted. If you run out of ideas, think harder: reread `README.md`, `prepare.py`, `train.py`, the latest `run.log`, structured reports, diagnostics, `results.tsv`, and `.autoresearch/experiment-ledger.md`; combine previous near-misses; try focused changes; try simplifications.
