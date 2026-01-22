# caster-commons

Shared utilities for the Caster Subnet workspace.

This package provides:

- **Sandbox runner** — Docker container lifecycle management for validators (start/stop sandbox containers, HTTP client for calling sandbox entrypoints)
- **Tool infrastructure** — LLM and search tool adapters, cost tracking, budget validation
- **Observability** — logging and tracing utilities
- **Domain types** — shared data structures (claims, sessions, verdicts, etc.)

This is an internal package used by `caster-validator`.
