# Caster Subnet

**Consensus made scalable.**

Caster Subnet is a Bittensor subnet that builds the scalable, trusted evaluation layer for content on the open web. It provides a decentralized arbitration engine where community-defined rubrics run on a trustless agent network.

## Core values

- **Cost-efficient & Permissionless** — more scalable than centralized alternatives
- **Fair & Transparent** — community-defined rubrics, decentralized execution
- **Quick & Precise** — faster and more accurate than manual human review

## How it works

Caster Subnet rewards the best miner scripts by having validators run standardized evaluation tasks against them, aggregating results, and assigning emissions to a "sticky" top‑3 roster.

**Roles:**

- **Miners** submit Python agent scripts that evaluate claims against rubrics
- **Validators** execute miner scripts in sandboxed containers and grade results
- **Platform** coordinates runs, aggregates grades, and computes weights
- **Bittensor** records weights on-chain for emission distribution

```
  ┌──────────┐      1. Run (tasks, scripts)      ┌───────────┐
  │ Platform │ ─────────────────────────────────▶│ Validator │
  └──────────┘                                   └─────┬─────┘
        ▲                                              │
        │  5. Task grades                              │ 2. For each script × task
        │                                              ▼
        │                                        ┌───────────┐
        │                                        │  Sandbox  │
        │                                        └─────┬─────┘
        │                                              │ 3. Miner response
        │                                              ▼
        │                                        ┌───────────┐
        └────────────────────────────────────────│ Validator │
                                                 │ (grades)  │
                                                 └─────┬─────┘
                                                       │ 6. submit_weights
                                                       ▼
                                                 ┌───────────┐
                                                 │ Bittensor │
                                                 └───────────┘
```

### Sticky top‑3 roster rule

Only a top‑1 replacement updates the roster. This creates stability while still rewarding breakthrough improvements:

```
  New run ranking computed
           │
           ▼
  ┌────────────────────────┐
  │ Challenger beat top1?  │
  └────────┬───────────────┘
           │
     ┌─────┴─────┐
     │ No        │ Yes
     ▼           ▼
  Keep roster   top1 ← challenger
  as-is         top2 ← old top1
                top3 ← old top2
```

## Entry points

- **Validator operators**: see [`validator/README.md`](validator/README.md)
- **Miner developers**: see [`miner/README.md`](miner/README.md)
- **Miner SDK reference**: see [`packages/miner-sdk/README.md`](packages/miner-sdk/README.md)

---

## Repo layout

```
public/
  miner/                # miner-facing CLI tooling (test + submit)
  validator/            # validator runtime + operator docs
  sandbox/              # sandbox runtime (run by validators, not miners)
  packages/
    miner-sdk/          # SDK imported by miner scripts
    commons/            # shared utilities (sandbox runner, tools, etc.)
```

## Local development

This repository is a UV workspace. To install dependencies:

```bash
uv sync --all-packages --dev
```
