# caster-sandbox

This package contains the **sandbox runtime** — the FastAPI server that validators use to execute miner agent scripts in isolated containers.

## What this is

- A lightweight HTTP server exposing `/entry/{entrypoint}` endpoints
- Loads miner scripts via `runpy.run_path` and invokes registered entrypoints
- Provides tool proxies (search, LLM) back to the validator host
- Runs inside a Docker container with seccomp + resource limits

## How it fits in

```
  Validator
      │
      │ starts container from castersubnet/caster-subnet-sandbox image
      ▼
  ┌─────────────────────────────────┐
  │  sandbox/                       │  ◀── this package
  │  caster-sandbox --serve         │
  │  loads miner agent.py           │
  │  calls evaluate_criterion       │
  └─────────────────────────────────┘
      │
      │ returns verdict + justification + citations
      ▼
  Validator (grades result)
```

## Building the image

From the repo root:

```bash
./scripts/build_sandbox_image.sh
```

This builds the `caster-subnet-sandbox` Docker image using the Dockerfile in this directory.

## Running locally (development)

```bash
uv run --package caster-sandbox caster-sandbox --serve
```

The server starts on `http://127.0.0.1:8000` by default. Set `SANDBOX_HOST` and `SANDBOX_PORT` to customize.
