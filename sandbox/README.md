# harnyx-sandbox

This package contains the **sandbox runtime** — the FastAPI server that validators use to execute miner agent scripts in isolated containers.

## What this is

- A lightweight HTTP server exposing `/entry/{entrypoint}` endpoints
- Loads miner scripts via `runpy.run_path` and invokes registered entrypoints
- Provides tool proxies (search, LLM) back to the validator host
- In the subnet miner-task path, validators typically call `/entry/query` with `{ "text": "..." }`
- Runs inside a Docker container with seccomp + resource limits

## How it fits in

```
  Validator
      │
      │ starts container from harnyx/harnyx-subnet-sandbox image
      ▼
  ┌─────────────────────────────────┐
  │  sandbox/                       │  ◀── this package
  │  harnyx-sandbox --serve         │
  │  loads miner agent.py           │
  │  calls query                    │
  └─────────────────────────────────┘
      │
      │ returns structured response text
      ▼
  Validator (grades result)
```

## Building the image

From the repo root:

```bash
docker build -f sandbox/Dockerfile -t harnyx/harnyx-subnet-sandbox:local .
```

This builds the `harnyx/harnyx-subnet-sandbox:local` Docker image using `sandbox/Dockerfile`.

## Running locally (development)

```bash
uv run --package harnyx-sandbox harnyx-sandbox --serve
```

The server starts on `http://127.0.0.1:8000` by default. Set `SANDBOX_HOST` and `SANDBOX_PORT` to customize.
