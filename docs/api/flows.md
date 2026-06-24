# Harnyx API flows (sequence diagrams)

All Mermaid **sequence diagrams** live here (one document). For request/response shapes, use the generated endpoint references:
- Platform: [generated/platform.md](generated/platform.md)
- Validator: [generated/validator.md](generated/validator.md)
- Sandbox: [generated/sandbox.md](generated/sandbox.md)

## Diagram style

These diagrams are intentionally **linear** (no `alt` / `par` / `loop`) to keep them easy to read. Any optional or repeated behavior is described in short notes next to the diagram.

## Quick index

- Subnet runtime (Platform ↔ Validator ↔ Miner)
  - [Miner config](#miner-config)
  - [Miner script upload](#miner-script-upload)
  - [Miner-task batch](#miner-task-batch)
  - [Tool execution](#tool-execution)
- Subnet ops (Platform ↔ Validator)
  - [Validator registration and weights](#validator-registration-and-weights)

## Flow catalog (fast scan)

| Domain | Flow | Goal | Actors | Auth / Context |
|--------|------|------|--------|------|
| Subnet runtime | Miner config | configure retry count; upload, read, or delete redacted provider credential status | Miner ↔ Platform | `Authorization: Bittensor ...` |
| Subnet runtime | Miner script upload | upload script artifact | Miner ↔ Platform | `Authorization: Bittensor ...` |
| Subnet runtime | Miner-task batch | forward batch to allowlisted validators + run sandbox + poll status and drain run pages | Platform ↔ Validator ↔ Sandbox | `Authorization: Bittensor ...` + `x-platform-token` + `x-session-id` + `x-host-container-url` |
| Subnet runtime | Tool execution | agent invokes host tools | Sandbox agent ↔ Tool host | `x-platform-token` + `x-session-id` |
| Subnet ops | Validator registration and weights | register API base URL; read weights | Validator ↔ Platform | `Authorization: Bittensor ...` |

---

## Subnet runtime (Platform ↔ Validator ↔ Miner)

These flows are the subnet’s core evaluation path.

### Miner config

| Overview | |
|---|---|
| **What’s happening** | Miner manages platform-stored config for a signing hotkey. |
| **Execution status** | Stored provider credentials are used by active miner-task batch execution through scoped platform tool proxy calls. Raw provider API keys are never returned to validators or sandboxes. |
| **Credential cleanup** | On successful metagraph refresh, platform prunes provider credentials for miner hotkeys absent from the refreshed metagraph. Empty registered-hotkey snapshots and suspiciously broad cleanup candidates are skipped. |
| **Actors** | Miner ↔ Platform |
| **Auth** | `Authorization: Bittensor ss58="...",sig="..."` |
| **Happy path** | `GET`, `PUT`, or `DELETE /v1/miner-config` returns retry count and redacted provider status. |

```mermaid
sequenceDiagram
  participant M as Miner
  participant P as Platform API

  Note over M,P: Authorization: Bittensor ss58="...",sig="..."

  M->>P: GET /v1/miner-config
  P-->>M: 200 { task_retry_count, provider_credentials:{...exists/timestamps...} }

  M->>P: PUT /v1/miner-config<br/>{ key:"task_retry_count", value:"3" }
  P-->>M: 200 { task_retry_count:3, provider_credentials:{...} }

  M->>P: PUT /v1/miner-config<br/>{ key:"provider_credentials.chutes", value:"..." }
  P-->>M: 200 { task_retry_count, provider_credentials:{ chutes:{ exists:true } } }

  M->>P: DELETE /v1/miner-config<br/>{ key:"provider_credentials.chutes" }
  P-->>M: 200 { task_retry_count, provider_credentials:{ chutes:{ exists:false } } }
```

**Endpoints involved**
- Platform:
  - [GET /v1/miner-config](generated/platform.md#endpoint-get-v1-miner-config)
  - [PUT /v1/miner-config](generated/platform.md#endpoint-put-v1-miner-config)
  - [DELETE /v1/miner-config](generated/platform.md#endpoint-delete-v1-miner-config)

---

### Miner script upload

| Overview | |
|---|---|
| **What’s happening** | Miner uploads a script artifact that later becomes a batch candidate. |
| **Actors** | Miner ↔ Platform |
| **Auth** | `Authorization: Bittensor ss58="...",sig="..."` |
| **Happy path** | `POST /v1/miners/scripts` returns `{ artifact_id, ... }` |

```mermaid
sequenceDiagram
  participant M as Miner
  participant P as Platform API

  Note over M,P: Authorization: Bittensor ss58="...",sig="..."

  M->>P: POST /v1/miners/scripts<br/>{ script_b64, sha256 }
  P-->>M: 200 { uid, artifact_id, content_hash, size_bytes }
```

**Endpoints involved**
- Platform (miner): [POST /v1/miners/scripts](generated/platform.md#endpoint-post-v1-miners-scripts)

---

### Miner-task batch

| Overview | |
|---|---|
| **What’s happening** | Platform owns the batch work ledger; validator polls for assigned task attempts, fetches artifacts, runs `query`, and submits results back to platform. |
| **Actors** | Platform ↔ Validator worker ↔ Sandbox |
| **Auth** | Validator↔Platform is Bittensor-signed; Validator↔Sandbox uses `x-platform-token` + `x-session-id` + `x-host-container-url`. |
| **Happy path** | materialize expected work → poll task assignment → fetch artifact → run `query` → submit result |
| **Assignment gate** | Platform assigns work only to registered, healthy, metagraph-authorized validators that have a `validator_allowlist_entry` row for `miner_task_batch_delivery`. |

#### 1) Platform creates a batch and materializes expected work

```mermaid
sequenceDiagram
  participant P as Platform
  participant DB as Platform DB

  Note over P: Manual or scheduled creation
  P->>P: POST /v1/miner-task-batches/batch<br/>{ tasks, artifacts, cutoff_at }
  P->>DB: Store batch, artifact refs, and expected validator task rows
  P->>DB: Initialize delivery projection from platform-owned rows

  Note over P,DB: No batch payload is pushed to validators.
```

#### 2) Validator polls platform for work

```mermaid
sequenceDiagram
  participant V as Validator
  participant P as Platform

  Note over V,P: Authorization: Bittensor ss58="...",sig="..."
  V->>P: POST /v2/miner-task-work/tasks<br/>{ target_concurrency, max_active_artifacts, active_attempts }
  P-->>V: 200 { tasks:[{ batch_id, artifact_id, task_id, attempt_number, ... }] }

  Note over V,P: Validator polls again whenever local execution slots are free.
```

#### 3) Validator fetches artifacts and invokes sandbox query

```mermaid
sequenceDiagram
  participant V as Validator
  participant P as Platform
  participant S as Sandbox
  participant VA as Validator API (tools)

  Note over V,P: Authorization: Bittensor ss58="...",sig="..."
  V->>P: GET /v1/miner-task-batches/{batch_id}/artifacts/{artifact_id}
  P-->>V: 200 <application/octet-stream>

  Note over V,S: Headers: x-session-id + x-platform-token + x-host-container-url
  V->>S: POST /entry/query<br/>{ text: "..." }

  Note over S,VA: Headers: x-session-id + x-platform-token
  S->>VA: POST /v1/tools/execute<br/>ToolExecuteRequestDTO (0+ times)
  VA-->>S: 200 ToolExecuteResponseDTO

  S-->>V: 200 { ok:true, result:{ text: "..." } }
```

#### 4) Validator submits task results to platform

```mermaid
sequenceDiagram
  participant V as Validator
  participant P as Platform
  participant DB as Platform DB

  Note over P,V: Authorization: Bittensor ss58="...",sig="..."
  V->>P: POST /v2/miner-task-work/results<br/>{ results:[{ batch_id, artifact_id, task_id, attempt_number, ... }] }
  P->>DB: Accept result idempotently and update run/delivery state
  P-->>V: 200 { results:[{ outcome:"accepted"|"retry_later"|"rejected", canonical:true|false, reason_code:null|"already_accepted"|... }] }
  Note over V: Act only on outcome; reason_code is diagnostic.

  Note over V,P: Validator retries only pending result submissions whose outcome is unknown or transient.
```

**Endpoints involved**
- Platform:
  - [POST /v1/miner-task-batches/batch](generated/platform.md#endpoint-post-v1-miner-task-batches-batch)
  - [GET /v1/miner-task-batches/{batch_id}/artifacts/{artifact_id}](generated/platform.md#endpoint-get-v1-miner-task-batches-batch_id-artifacts-artifact_id)
  - `POST /v2/miner-task-work/tasks`
  - `POST /v2/miner-task-work/results`
- Validator:
  - [GET /validator/status](generated/validator.md#endpoint-get-validator-status)
  - [POST /validator/miner-task-batches/{batch_id}/similarity](generated/validator.md#endpoint-post-validator-miner-task-batches-batch_id-similarity)
  - [POST /v1/tools/execute](generated/validator.md#endpoint-post-v1-tools-execute)
- Sandbox:
  - [POST /entry/{entrypoint_name}](generated/sandbox.md#endpoint-post-entry-entrypoint_name)

---

### Tool execution

| Overview | |
|---|---|
| **What’s happening** | Sandboxed agent code invokes host-managed tools (search/LLM/etc.) over HTTP. |
| **Actors** | Agent (in sandbox) ↔ Tool host (validator) |
| **Auth** | `x-platform-token` + `x-session-id` |
| **Happy path** | `POST /v1/tools/execute` returns `ToolExecuteResponseDTO` |

```mermaid
sequenceDiagram
  participant A as Agent code (in sandbox)
  participant H as Tool host (validator)

  Note over A,H: ToolProxy sends x-platform-token + x-session-id headers
  A->>H: POST /v1/tools/execute<br/>ToolExecuteRequestDTO
  H-->>A: 200 ToolExecuteResponseDTO
```

**Endpoints involved**
- Validator: [POST /v1/tools/execute](generated/validator.md#endpoint-post-v1-tools-execute)

---

## Subnet ops (Platform ↔ Validator)

These flows are about validator lifecycle and platform coordination.

### Validator registration and weights

| Overview | |
|---|---|
| **What’s happening** | Validator registers its public API base URL, then reads the current weights. |
| **Actors** | Validator ↔ Platform |
| **Auth** | `Authorization: Bittensor ss58="...",sig="..."` |
| **Happy path** | `POST /v1/validators/register` → `GET /v1/weights` |
| **Allowlist** | The batch-delivery allowlist does not gate registration or `GET /v1/weights`. |

```mermaid
sequenceDiagram
  participant V as Validator
  participant P as Platform API

  Note over V,P: Authorization: Bittensor ss58="...",sig="..."

  V->>P: POST /v1/validators/register<br/>{ base_url }
  P-->>V: 200 { status:"ok" }

  V->>P: GET /v1/weights
  P-->>V: 200 { weights, champion_uid }

  Note over V,P: On error, platform returns 4xx { error_code, message }.
```

**Endpoints involved**
- Platform:
  - [POST /v1/validators/register](generated/platform.md#endpoint-post-v1-validators-register)
  - [GET /v1/weights](generated/platform.md#endpoint-get-v1-weights)
