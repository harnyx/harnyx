# Harnyx Subnet

**A Deep Research harness under continuous competitive pressure — always adapting, never static.**

Harnyx (SN 67) is a Bittensor subnet for deep research. It turns research execution into a competitive harness: miners compete on better workflows, validators enforce the runtime, and the network returns intelligence with provenance.

The core thesis is simple: better models matter, but better harnesses compound faster. Deep research is not one reasoning step. It is decomposition, retrieval, ranking, cross-checking, and synthesis under real constraints. Harnyx makes that harness an open competitive system instead of a closed product team.

## Start here

- **Validator operators**: see [`validator/README.md`](validator/README.md)
- **Miner developers**: see [`miner/README.md`](miner/README.md)
- **Miner AutoResearch**: see [`miner/AUTO-RESEARCH.md`](miner/AUTO-RESEARCH.md)
- **Miner SDK reference**: see [`packages/miner-sdk/README.md`](packages/miner-sdk/README.md)
- **Live benchmark**: see [`dashboard.harnyx.ai/benchmark`](https://dashboard.harnyx.ai/benchmark)

## Install dependencies (local dev)

```bash
uv sync --all-packages --dev
```

## How the subnet works today

Today, miners submit Python agents. Validators run those agents in sandboxes against subnet tasks, score the results, and submit weights on-chain. The runtime contract centers on miner-task batches, validator scoring, and public monitoring.

A **task** is one research-style query plus one stronger reference answer.

<details>
<summary><strong>Exact task contract (JSON)</strong></summary>

Miners implement the `query` entrypoint. Validators call it with this payload:

```json
{
  "text": "Harnyx Subnet validators manage sandboxed miners."
}
```

Your script must return:

```json
{
  "text": "Validators execute miner scripts inside sandboxed environments."
}
```

Notes:
- Requests and responses are plain text wrapped in typed objects so the contract can expand later without breaking the entrypoint shape.

**Dig deeper**
- [Miner entrypoint contract (SDK)](packages/miner-sdk/README.md#query-contract)
- [Flow: miner-task batch](docs/api/flows.md#miner-task-batch)
- [Flow: tool execution](docs/api/flows.md#tool-execution)
- [API auth conventions + index](docs/api/README.md)
</details>

**How the task set is built**
- The platform generates batches of research-style standalone queries.
- For each query, the platform generates a stronger **reference answer** using a more expensive model than the typical miner budget allows.
- Tasks are intentionally mixed across factual recall, explanation, comparison, and synthesis so miners need real search/reasoning behavior rather than memorized outputs.

**How miners are evaluated**
- Miners submit scripts that answer the query under a tight tool budget.
- Validators score each response against the reference answer with:
  - `comparison_score`: pairwise judge vs reference answer, run twice with swapped order
  - `total_score = comparison_score`
- Candidate totals are aggregated across validators, and ties prefer lower total tool cost.

**Validator flow + gating**
- The platform owns the miner-task work ledger; validators poll for assigned task attempts, run script x task combinations, and submit task results.
- One successful validator delivery is enough to satisfy the validator quorum. Failures from other validators do not, by themselves, prevent the batch from completing.
- Registered validators can query the latest weights for on-chain emission submission.
- Miner emission keeps champion emission active and adds participant emission from the latest terminal source batch with finalized tasks and artifacts. Successful batches use score tiers plus the artifact's novelty classification. Failed batches divide the entire post-champion remainder equally among distinct participant hotkeys. The exact allocations are described below. The final owner `uid=0` remainder, including unregistered participant shares, burns miner emission and is not paid to the owner.
- The [live benchmark page](https://dashboard.harnyx.ai/benchmark) shows benchmark history and run detail for inspecting champion quality.

**Roles**
- **Miners** submit Python agent scripts that answer queries
- **Validators** execute miner scripts in sandboxed containers and score results
- **Platform** coordinates runs, aggregates scores, and computes weights
- **Bittensor** records weights on-chain for emission distribution

```mermaid
sequenceDiagram
    participant Platform
    participant Validator
    participant Sandbox
    participant Bittensor

    Validator->>Platform: 1) Poll assigned task attempts
    Platform-->>Validator: 2) Task, script, and attempt metadata
    Validator->>Sandbox: 3) Execute script x task
    Sandbox-->>Validator: 4) Miner response
    Validator-->>Platform: 5) Submit task results
    Validator->>Bittensor: 6) submit_weights
```

### How champion selection works

Champion selection is not the same as "highest score in the batch wins."

The platform starts from the incumbent champion and compares challengers in batch order. For current data-version-8 batches, a challenger only replaces the incumbent when it clears one of these hard-coded dethroning rules:

- its score is strictly higher and at least `min(perfect score, incumbent score + 10 percentage points)`, regardless of cost or runtime
- its score and runtime do not regress and median cost falls by at least 10%, or
- its score and cost do not regress and median runtime falls by at least 10% and at least 1,000 ms

Data versions 1 through 7 retain their historical 20% relative score, cost, and runtime thresholds when replayed. The 1,000 ms runtime floor also applies historically.

Because of that:

- the champion is not always the highest score in the batch
- score-margin improvements can replace the champion regardless of cost or runtime
- challenger order matters
- a newer eligible submission from the current champion hotkey gets the first challenger position
- small score differences inside the tolerance band do not automatically replace the incumbent

### How participant miner emission works

`GET /v1/weights` uses latest champion weights for champion emission and the latest terminal source batch with finalized tasks and artifacts for participant emission. For a failed terminal batch, champion emission is reserved first and the entire remainder is divided equally among distinct participant hotkeys. For a successful data-version-7-or-later batch, the remainder is divided per reward-eligible artifact. The participation-stage multiplier is `1` for top 50%, `2` for top 10%, or `5` for main; the novelty multiplier is `1` for `near_duplicate`, `3` for `notable_change`, or `5` for `novel`. An artifact's share is proportional to the product of those multipliers, up to `25`. Shares are aggregated to hotkeys afterward. A challenger that becomes champion receives champion emission plus its participant share; champion emission itself is never multiplied. The unchanged entering incumbent, duplicates, zero-response artifacts and artifacts outside the reward tiers receive no participant share. Data versions 1 through 6 retain their historical calculation. Registered hotkeys project to current metagraph UIDs; unregistered shares burn through owner `uid=0`.

Public emission monitoring groups totals by participant hotkey and nests one entry per source-batch artifact. Artifact entries expose nullable stage and novelty multipliers, nullable distribution weight, and the participant reward fraction. Pre-change and failed-batch rows use null multipliers while preserving their correct share.

Failed terminal batches with finalized tasks and artifacts count for participant emission. Initializing/running batches and terminal batches without finalized tasks or without artifacts do not update the emitted participant source. If no terminal source batch with finalized tasks and artifacts exists, only the champion component remains active.



## Repo layout

```text
miner/                # miner-facing CLI tooling (local-eval, local-benchmark, submit)
validator/            # validator runtime + operator docs
sandbox/              # sandbox runtime (run by validators, not miners)
packages/
  miner-sdk/          # SDK imported by miner scripts
  commons/            # shared utilities plus public miner incentive logic
```
