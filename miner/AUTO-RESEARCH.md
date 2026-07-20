# Miner AutoResearch

This guide is for the human/operator starting an autonomous miner research run.

Use [`program.md`](program.md) as the agent-facing instruction file. Do not paste this whole guide into the agent unless you want to; the short prompt below is enough because the agent must read `program.md`.

## What To Tell The Agent

Start the agent from this `miner` directory and tell it:

```text
We are running Harnyx miner AutoResearch.

Work from the miner directory. Read README.md first, then read program.md and follow it exactly as your standing research policy.

Set up the run, choose the benchmark suite with the operator, run uv run prepare.py --benchmark-suite <suite>, initialize results.tsv and .autoresearch/experiment-ledger.md, then begin the loop. Do not redesign the framework. Only edit train.py. Start from concrete failures, pick one bottleneck, write a hypothesis, run focused diagnostics, and only run full eval when program.md allows it.
```

After setup confirmation, the agent should keep going until you interrupt it.

## Required Environment

Create `.env` at the public repo root from [`../.env.example`](../.env.example) before starting the run.

Typical Chutes + DeSearch setup:

```bash
PLATFORM_BASE_URL=https://api.harnyx.ai
TOOL_LLM_PROVIDER=chutes
CHUTES_API_KEY=...
SEARCH_PROVIDER=desearch
DESEARCH_API_KEY=...
BENCHMARK_LLM_PROVIDER=chutes
BENCHMARK_LLM_MODEL=<benchmark-judge-model>
BENCHMARK_RUBRIC_JUDGE_LLM_PROVIDER=<rubric-judge-provider>
BENCHMARK_RUBRIC_JUDGE_LLM_MODEL=<rubric-judge-model>
```

What each value is for:

| Variable | Needed for |
|----------|------------|
| `PLATFORM_BASE_URL` | `uv run prepare.py --benchmark-suite <suite>` batch discovery, local eval context, and later manual submit commands |
| `CHUTES_API_KEY` | local-eval judging and miner `llm_chat` calls when Chutes is the tool/scoring provider |
| `TOOL_LLM_PROVIDER` | provider used for miner `llm_chat` tool calls; the public example defaults to `chutes` |
| `SEARCH_PROVIDER` | provider used for miner `search_web` calls |
| `DESEARCH_API_KEY` | required when `SEARCH_PROVIDER=desearch` |
| `BENCHMARK_LLM_PROVIDER` | provider for benchmark correctness judging |
| `BENCHMARK_LLM_MODEL` | model for benchmark correctness judging |
| `BENCHMARK_RUBRIC_JUDGE_LLM_PROVIDER` | provider for DRACO / `weighted-rubric-v1` benchmark judging |
| `BENCHMARK_RUBRIC_JUDGE_LLM_MODEL` | model for DRACO / `weighted-rubric-v1` benchmark judging |

If using `SEARCH_PROVIDER=parallel`, set `PARALLEL_API_KEY` instead of `DESEARCH_API_KEY`.

If using either benchmark judge provider with `vertex`, also configure Vertex credentials such as `GCP_PROJECT_ID`, `GCP_LOCATION`, and the usual Google application credentials for your machine. For DRACO with Gemini 3.1 Pro Preview, use `BENCHMARK_RUBRIC_JUDGE_LLM_PROVIDER=vertex`, `BENCHMARK_RUBRIC_JUDGE_LLM_MODEL=gemini-3.1-pro-preview`, and `GCP_LOCATION=global`. `BENCHMARK_LLM_*` settings do not enable DRACO by fallback; DRACO uses only the dedicated `BENCHMARK_RUBRIC_JUDGE_LLM_*` settings.

The agent may discover missing variables when `uv run prepare.py --benchmark-suite <suite>`, local eval, local benchmark, `llm_chat`, or `search_web` fails. Setting them before the run avoids wasting research cycles on setup failures.

## Run Setup

From the repo root:

```bash
uv sync --all-packages --dev
```

Then start the agent in:

```bash
cd miner
```

The agent should follow [`program.md`](program.md), but the expected setup is:

```bash
uv run prepare.py --benchmark-suite <suite-slug>
printf 'commit\tscore_a\tscore_b\tcost_usd\tstatus\tdescription\n' > results.tsv
mkdir -p .autoresearch
touch .autoresearch/experiment-ledger.md
```

`prepare.py` pins the current completed local-eval batch and the explicitly selected benchmark snapshot. To force a specific completed batch:

```bash
uv run prepare.py --benchmark-suite <suite-slug> --batch-id <completed-batch-id>
```

## What The Agent Should Edit

Only [`train.py`](train.py) is the candidate harness. The final kept harness must remain a self-contained single Python script and may only use the provided `llm_chat` and `search_web` tools.

Generated research artifacts stay untracked:

- `run.log`
- `results.tsv`
- `.autoresearch/`

## Normal Loop

The agent should:

1. inspect concrete weak or failed cases
2. update `.autoresearch/experiment-ledger.md`
3. pick exactly one mechanism-level bottleneck
4. write a hypothesis before editing
5. run focused diagnostics
6. commit the candidate when diagnostics justify full eval
7. run `LOG_LEVEL=DEBUG uv run train.py > run.log 2>&1`
8. record Score A, Score B, cost, and keep/discard/crash status in `results.tsv`
9. keep or reset the commit according to [`program.md`](program.md)

Do not ask it to run the expensive full evaluation after every tiny edit. `program.md` defines when full evaluation is allowed.

`prepare.py` passes `LOG_LEVEL` to local eval and stores its captured streams
under `.autoresearch/reports/<timestamp>/`. Inspect `local-eval.stderr` for the
detailed tool-call events; `run.log` remains the combined top-level experiment
output and metric summary. Review DEBUG logs before sharing because they can
contain task content, retrieved pages, and model responses.

## Stopping And Upload

Stop the agent by interrupting it.

AutoResearch does not upload automatically. If you want to submit a kept
`train.py`, do it manually with the normal submit flow in
[`README.md`](README.md#step-5-submit-to-the-platform), then use the
[Mining Runbook](mining-runbook.md) for config checks, submission verification,
batch monitoring, and completed-result diagnosis.

## See Also

- Agent policy: [`program.md`](program.md)
- Fixed evaluator support: [`prepare.py`](prepare.py)
- Candidate harness: [`train.py`](train.py)
- Local eval details: [`local-eval.md`](local-eval.md)
