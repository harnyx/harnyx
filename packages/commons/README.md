# harnyx-commons

Shared utilities for the Harnyx Subnet workspace.

This package provides:

- **Sandbox runner** — Docker container lifecycle management for validators (start/stop sandbox containers, HTTP client for calling sandbox entrypoints)
- **Tool infrastructure** — LLM and search tool adapters, cost tracking, budget validation
- **Observability** — logging and tracing utilities
- **Domain types** — shared data structures (queries, responses, sessions, scores, etc.)

This is an internal package used by `harnyx-validator`.
