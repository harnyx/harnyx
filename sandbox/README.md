# harnyx-sandbox

This package contains the **sandbox runtime** — the FastAPI server that validators use to execute miner agent scripts in isolated containers.

## What this is

- A lightweight HTTP server exposing `/entry/{entrypoint}` endpoints
- Compiles miner source once and invokes registered entrypoints in isolated query workers
- Provides tool proxies (search, LLM) back to the validator host
- In the subnet miner-task path, validators call `/entry/query` with a sandbox envelope whose `context` contains the initial `cost_budget` and `time_budget.limit_seconds` alongside the query `payload`.
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
      │ returns plain text or direct structured output
      ▼
  Validator (grades result)
```

The sandbox validates the miner entrypoint query and response envelope through
the miner SDK. A query without `output_schema` requires `Response.text` for a
plain-text answer; a query with a schema requires direct `Response.output`. The
host that submitted the query validates that output against the originating schema before it
hydrates response-level citation refs. This keeps citations outside the
caller-owned output schema and preserves invalid-response classification.

## Execution lifetime

Each artifact container has a trusted compiler process and a separate worker for
each query. The compiler starts in a fresh interpreter, compiles the source once,
and never executes miner code or receives query credentials. Query workers inherit
the compiled code through copy-on-write memory. Each worker executes the module
and initializes its own imports and globals; mutable state is not shared.

The HTTP supervisor holds tool credentials. Each worker sends bounded JSON tool
requests through its own connection, which the supervisor binds to that query.
Worker-supplied IDs cannot select another session or result destination. Process
memory and descriptor protections prevent workers from reading the supervisor or
siblings. Miner code runs only after the supervisor owns its process handle and a default-deny syscall policy is installed. The policy allows required runtime operations and self-directed resource management, while denying peer control and new processes/threads.

The validator reserves admission through `/admission` before starting a query
session, then passes the reservation to `/entry/query`. Unstarted assignments keep
their original dispatch lease. Direct client invocations count admission waiting
against their existing time budget. Compilation and readiness remain bounded;
health checks remain responsive.

At a query deadline, the supervisor signals the worker directly through its process handle: TERM, up to one second of grace, then KILL if needed. Normal reaping remains compiler-owned. An unresponsive compiler triggers bounded generation retirement; unconfirmed cleanup is reported as HTTP 503 with code `SandboxCleanupUnconfirmed`, requiring host retirement. Successful and failed queries also
release their worker and its background tasks. Other queries continue. Compiler
failure retires its workers before replacement; started queries are never replayed.
Closing a worker does not guarantee cancellation of remote provider requests;
the trusted host retains tool settlement and accounting ownership.

Raw query stdout/stderr is attributed using its parent-owned session binding.
Compiler/runtime output remains artifact-scoped. The existing 1 GiB container
limit and concurrency settings are unchanged. Compilation is reused, but imports,
module initialization, and query allocations still consume memory per worker;
arbitrary artifacts can still exhaust the container.

## Building the image

From the repo root:

```bash
docker build -f sandbox/Dockerfile -t harnyx/harnyx-subnet-sandbox:local .
```

This builds the `harnyx/harnyx-subnet-sandbox:local` Docker image using `sandbox/Dockerfile`.

## Running locally (development)

```bash
export SANDBOX_CONTROL_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run --package harnyx-sandbox harnyx-sandbox --serve
```

Send the same per-run value as `x-sandbox-control-token` on admission, release and entry requests. Existing tool-session headers are still required separately. The standard Docker manager provisions and redacts this credential automatically; standalone clients must supply it explicitly. A missing startup credential fails closed. Do not put it in miner code or commit it.

The server starts on `http://127.0.0.1:8000` by default. Set `SANDBOX_HOST` and `SANDBOX_PORT` to customize.

## Tool completion and worker authority

Accepted tool requests stay owned by the invocation. Before returning a worker result (including a reported miner error), the supervisor waits for those requests to complete within the query deadline. On deadline, cancellation, or broken worker transport, it cancels and joins local tool tasks and closes their proxy before returning. This does not guarantee remote provider cancellation or receipt completion after an HTTP request is cancelled; the transport cannot acknowledge that stronger contract.

Worker signal syscalls are restricted to the worker's own PID/thread. Process-group signals and pidfd-based signals are denied, so a miner cannot stop the compiler, supervisor, or sibling worker. Supervisor-to-child termination remains available.
