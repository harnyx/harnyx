# Validator API reference (generated)

Generated from FastAPI OpenAPI.

## Domains
- [miner-task-batches](#miner-task-batches)
  - [POST /validator/miner-task-batches/batch](#endpoint-post-validator-miner-task-batches-batch)
  - [GET /validator/miner-task-batches/{batch_id}/progress](#endpoint-get-validator-miner-task-batches-batch_id-progress)
- [status](#status)
  - [GET /validator/status](#endpoint-get-validator-status)
- [tools](#tools)
  - [POST /v1/tools/execute](#endpoint-post-v1-tools-execute)

## miner-task-batches

### batch

<a id="endpoint-post-validator-miner-task-batches-batch"></a>
#### POST /validator/miner-task-batches/batch

Accept a miner task batch and start processing it.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Request**
Content-Type: `application/json`
Body: [MinerTaskBatchRequestModel](#model-minertaskbatchrequestmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifacts` |  |  | req | array[[ScriptArtifactRequestModel](#model-scriptartifactrequestmodel)] |
|  | `artifact_id` |  | req | `string` |
|  | `content_hash` |  | req | `string` |
|  | `size_bytes` |  | req | `integer` |
|  | `uid` |  | req | `integer` |
| `batch_id` |  |  | req | `string` |
| `created_at` |  |  | req | `string` |
| `cutoff_at` |  |  | req | `string` |
| `tasks` |  |  | req | array[[MinerTaskRequestModel](#model-minertaskrequestmodel)] |
|  | `budget_usd` |  | opt | `number` (default: 0.5) |
|  | `query` |  | req | [Query](#model-query) |
|  |  | `text` | req | `string` |
|  | `reference_answer` |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `text` | req | `string` |
|  | `task_id` |  | req | `string` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [BatchAcceptResponse](#model-batchacceptresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batch_id` |  |  | req | `string` |
| `caller` |  |  | req | `string` |
| `status` |  |  | req | `string` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


### {batch_id}

#### progress

<a id="endpoint-get-validator-miner-task-batches-batch_id-progress"></a>
##### GET /validator/miner-task-batches/{batch_id}/progress

Return progress and results for a miner task batch.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `batch_id` | path | req | `string` (format: uuid) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [ProgressResponse](#model-progressresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batch_id` |  |  | req | `string` |
| `completed` |  |  | req | `integer` |
| `miner_task_runs` |  |  | req | array[[MinerTaskRunSubmissionModel](#model-minertaskrunsubmissionmodel)] |
|  | `batch_id` |  | req | `string` |
|  | `run` |  | req | [MinerTaskRunModel](#model-minertaskrunmodel) |
|  |  | `artifact_id` | req | `string` |
|  |  | `query` | req | [Query](#model-query) |
|  |  | `reference_answer` | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `response` | opt | [Response](#model-response) (nullable) |
|  |  | `task_id` | req | `string` |
|  |  | `uid` | req | `integer` |
|  | `score` |  | req | `number` |
|  | `session` |  | req | [SessionModel](#model-sessionmodel) |
|  |  | `expires_at` | req | `string` |
|  |  | `issued_at` | req | `string` |
|  |  | `session_id` | req | `string` |
|  |  | `status` | req | `string` |
|  |  | `uid` | req | `integer` |
|  | `specifics` |  | req | [EvaluationDetails](#model-evaluationdetails) |
|  |  | `elapsed_ms` | opt | `number` (nullable) |
|  |  | `error` | opt | [EvaluationError](#model-evaluationerror) (nullable) |
|  |  | `score_breakdown` | opt | [ScoreBreakdown](#model-scorebreakdown) (nullable) |
|  |  | `total_tool_usage` | opt | [ToolUsageSummary](#model-toolusagesummary) |
|  | `usage` |  | req | [UsageModel](#model-usagemodel) |
|  |  | `by_provider` | opt | `object` |
|  |  | `call_count` | req | `integer` |
|  |  | `total_completion_tokens` | req | `integer` |
|  |  | `total_prompt_tokens` | req | `integer` |
|  |  | `total_tokens` | req | `integer` |
|  | `validator` |  | req | [ValidatorModel](#model-validatormodel) |
|  |  | `uid` | req | `integer` |
| `remaining` |  |  | req | `integer` |
| `total` |  |  | req | `integer` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |



## status

<a id="endpoint-get-validator-status"></a>
### GET /validator/status

Return a validator status snapshot for platform health checks.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [ValidatorStatusResponse](#model-validatorstatusresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `last_batch_id` |  |  | opt | `string` (nullable) |
| `last_completed_at` |  |  | opt | `string` (nullable) |
| `last_error` |  |  | opt | `string` (nullable) |
| `last_started_at` |  |  | opt | `string` (nullable) |
| `last_weight_error` |  |  | opt | `string` (nullable) |
| `last_weight_submission_at` |  |  | opt | `string` (nullable) |
| `queued_batches` |  |  | opt | `integer` (default: 0) |
| `running` |  |  | opt | `boolean` (default: False) |
| `status` |  |  | req | `string` |



## tools

### execute

<a id="endpoint-post-v1-tools-execute"></a>
#### POST /v1/tools/execute

Execute a tool invocation and return the tool result and usage.

**Auth**: Tool token (`x-platform-token` header)

**Headers**
| Header | Req | Notes |
| --- | --- | --- |
| `x-session-id` | req | `string` (format: uuid) |

**Request**
Content-Type: `application/json`
Body: [ToolExecuteRequestDTO](#model-toolexecuterequestdto)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `args` |  |  | opt | array[[JsonValue](#model-jsonvalue)] (default: []) |
| `kwargs` |  |  | opt | `object` (default: {}) |
| `tool` |  |  | req | `string` (enum: [search_web, search_x, search_ai, llm_chat, search_items, test_tool, tooling_info]) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [ToolExecuteResponseDTO](#model-toolexecuteresponsedto)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `budget` |  |  | req | [ToolBudgetDTO](#model-toolbudgetdto) |
|  | `session_budget_usd` |  | req | `number` |
|  | `session_hard_limit_usd` |  | req | `number` |
|  | `session_remaining_budget_usd` |  | req | `number` |
|  | `session_used_budget_usd` |  | req | `number` |
| `cost_usd` |  |  | opt | `number` (nullable) |
| `receipt_id` |  |  | req | `string` |
| `response` |  |  | req | [JsonValue](#model-jsonvalue) |
| `result_policy` |  |  | req | `string` |
| `results` |  |  | req | array[[ToolResultDTO](#model-toolresultdto)] |
|  | `index` |  | req | `integer` |
|  | `note` |  | opt | `string` (nullable) |
|  | `raw` |  | opt | [JsonValue](#model-jsonvalue) (nullable) |
|  | `result_id` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (nullable) |
| `usage` |  |  | opt | [ToolUsageDTO](#model-toolusagedto) (nullable) |
|  | `completion_tokens` |  | opt | `integer` (nullable) |
|  | `prompt_tokens` |  | opt | `integer` (nullable) |
|  | `total_tokens` |  | opt | `integer` (nullable) |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |



## Models

<a id="model-batchacceptresponse"></a>
### Model: BatchAcceptResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batch_id` |  |  | req | `string` |
| `caller` |  |  | req | `string` |
| `status` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "batch_id": {
      "minLength": 1,
      "title": "Batch Id",
      "type": "string"
    },
    "caller": {
      "minLength": 1,
      "title": "Caller",
      "type": "string"
    },
    "status": {
      "minLength": 1,
      "title": "Status",
      "type": "string"
    }
  },
  "required": [
    "status",
    "batch_id",
    "caller"
  ],
  "title": "BatchAcceptResponse",
  "type": "object"
}
```

</details>

<a id="model-evaluationdetails"></a>
### Model: EvaluationDetails

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `elapsed_ms` |  |  | opt | `number` (nullable) |
| `error` |  |  | opt | [EvaluationError](#model-evaluationerror) (nullable) |
|  | `code` |  | req | `string` |
|  | `message` |  | req | `string` |
| `score_breakdown` |  |  | opt | [ScoreBreakdown](#model-scorebreakdown) (nullable) |
|  | `comparison_score` |  | req | `number` |
|  | `scoring_version` |  | req | `string` |
|  | `similarity_score` |  | req | `number` |
|  | `total_score` |  | req | `number` |
| `total_tool_usage` |  |  | opt | [ToolUsageSummary](#model-toolusagesummary) |
|  | `llm` |  | opt | [LlmUsageSummary](#model-llmusagesummary) |
|  |  | `call_count` | opt | `integer` (default: 0) |
|  |  | `completion_tokens` | opt | `integer` (default: 0) |
|  |  | `cost` | opt | `number` (default: 0.0) |
|  |  | `prompt_tokens` | opt | `integer` (default: 0) |
|  |  | `providers` | opt | `object` |
|  |  | `total_tokens` | opt | `integer` (default: 0) |
|  | `llm_cost` |  | opt | `number` (default: 0.0) |
|  | `search_tool` |  | opt | [SearchToolUsageSummary](#model-searchtoolusagesummary) |
|  |  | `call_count` | opt | `integer` (default: 0) |
|  |  | `cost` | opt | `number` (default: 0.0) |
|  | `search_tool_cost` |  | opt | `number` (default: 0.0) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "elapsed_ms": {
      "anyOf": [
        {
          "minimum": 0.0,
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Elapsed Ms"
    },
    "error": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/EvaluationError"
        },
        {
          "type": "null"
        }
      ]
    },
    "score_breakdown": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/ScoreBreakdown"
        },
        {
          "type": "null"
        }
      ]
    },
    "total_tool_usage": {
      "$ref": "#/components/schemas/ToolUsageSummary"
    }
  },
  "title": "EvaluationDetails",
  "type": "object"
}
```

</details>

<a id="model-evaluationerror"></a>
### Model: EvaluationError

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `code` |  |  | req | `string` |
| `message` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "code": {
      "minLength": 1,
      "title": "Code",
      "type": "string"
    },
    "message": {
      "minLength": 1,
      "title": "Message",
      "type": "string"
    }
  },
  "required": [
    "code",
    "message"
  ],
  "title": "EvaluationError",
  "type": "object"
}
```

</details>

<a id="model-httpvalidationerror"></a>
### Model: HTTPValidationError

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "detail": {
      "items": {
        "$ref": "#/components/schemas/ValidationError"
      },
      "title": "Detail",
      "type": "array"
    }
  },
  "title": "HTTPValidationError",
  "type": "object"
}
```

</details>

<a id="model-jsonvalue"></a>
### Model: JsonValue

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{}
```

</details>

<a id="model-llmmodelusagecost"></a>
### Model: LlmModelUsageCost

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `cost` |  |  | opt | `number` (default: 0.0) |
| `usage` |  |  | opt | [LlmUsageTotals](#model-llmusagetotals) |
|  | `call_count` |  | opt | `integer` (default: 0) |
|  | `completion_tokens` |  | opt | `integer` (default: 0) |
|  | `prompt_tokens` |  | opt | `integer` (default: 0) |
|  | `total_tokens` |  | opt | `integer` (default: 0) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "cost": {
      "default": 0.0,
      "title": "Cost",
      "type": "number"
    },
    "usage": {
      "$ref": "#/components/schemas/LlmUsageTotals"
    }
  },
  "title": "LlmModelUsageCost",
  "type": "object"
}
```

</details>

<a id="model-llmusagesummary"></a>
### Model: LlmUsageSummary

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `call_count` |  |  | opt | `integer` (default: 0) |
| `completion_tokens` |  |  | opt | `integer` (default: 0) |
| `cost` |  |  | opt | `number` (default: 0.0) |
| `prompt_tokens` |  |  | opt | `integer` (default: 0) |
| `providers` |  |  | opt | `object` |
| `total_tokens` |  |  | opt | `integer` (default: 0) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "call_count": {
      "default": 0,
      "title": "Call Count",
      "type": "integer"
    },
    "completion_tokens": {
      "default": 0,
      "title": "Completion Tokens",
      "type": "integer"
    },
    "cost": {
      "default": 0.0,
      "title": "Cost",
      "type": "number"
    },
    "prompt_tokens": {
      "default": 0,
      "title": "Prompt Tokens",
      "type": "integer"
    },
    "providers": {
      "additionalProperties": {
        "additionalProperties": {
          "$ref": "#/components/schemas/LlmModelUsageCost"
        },
        "type": "object"
      },
      "title": "Providers",
      "type": "object"
    },
    "total_tokens": {
      "default": 0,
      "title": "Total Tokens",
      "type": "integer"
    }
  },
  "title": "LlmUsageSummary",
  "type": "object"
}
```

</details>

<a id="model-llmusagetotals"></a>
### Model: LlmUsageTotals

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `call_count` |  |  | opt | `integer` (default: 0) |
| `completion_tokens` |  |  | opt | `integer` (default: 0) |
| `prompt_tokens` |  |  | opt | `integer` (default: 0) |
| `total_tokens` |  |  | opt | `integer` (default: 0) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "call_count": {
      "default": 0,
      "title": "Call Count",
      "type": "integer"
    },
    "completion_tokens": {
      "default": 0,
      "title": "Completion Tokens",
      "type": "integer"
    },
    "prompt_tokens": {
      "default": 0,
      "title": "Prompt Tokens",
      "type": "integer"
    },
    "total_tokens": {
      "default": 0,
      "title": "Total Tokens",
      "type": "integer"
    }
  },
  "title": "LlmUsageTotals",
  "type": "object"
}
```

</details>

<a id="model-minertaskbatchrequestmodel"></a>
### Model: MinerTaskBatchRequestModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifacts` |  |  | req | array[[ScriptArtifactRequestModel](#model-scriptartifactrequestmodel)] |
|  | `artifact_id` |  | req | `string` |
|  | `content_hash` |  | req | `string` |
|  | `size_bytes` |  | req | `integer` |
|  | `uid` |  | req | `integer` |
| `batch_id` |  |  | req | `string` |
| `created_at` |  |  | req | `string` |
| `cutoff_at` |  |  | req | `string` |
| `tasks` |  |  | req | array[[MinerTaskRequestModel](#model-minertaskrequestmodel)] |
|  | `budget_usd` |  | opt | `number` (default: 0.5) |
|  | `query` |  | req | [Query](#model-query) |
|  |  | `text` | req | `string` |
|  | `reference_answer` |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `text` | req | `string` |
|  | `task_id` |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "artifacts": {
      "items": {
        "$ref": "#/components/schemas/ScriptArtifactRequestModel"
      },
      "minItems": 1,
      "title": "Artifacts",
      "type": "array"
    },
    "batch_id": {
      "minLength": 1,
      "title": "Batch Id",
      "type": "string"
    },
    "created_at": {
      "minLength": 1,
      "title": "Created At",
      "type": "string"
    },
    "cutoff_at": {
      "minLength": 1,
      "title": "Cutoff At",
      "type": "string"
    },
    "tasks": {
      "items": {
        "$ref": "#/components/schemas/MinerTaskRequestModel"
      },
      "minItems": 1,
      "title": "Tasks",
      "type": "array"
    }
  },
  "required": [
    "batch_id",
    "cutoff_at",
    "created_at",
    "tasks",
    "artifacts"
  ],
  "title": "MinerTaskBatchRequestModel",
  "type": "object"
}
```

</details>

<a id="model-minertaskrequestmodel"></a>
### Model: MinerTaskRequestModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `budget_usd` |  |  | opt | `number` (default: 0.5) |
| `query` |  |  | req | [Query](#model-query) |
|  | `text` |  | req | `string` |
| `reference_answer` |  |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  | `text` |  | req | `string` |
| `task_id` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "budget_usd": {
      "default": 0.5,
      "minimum": 0.0,
      "title": "Budget Usd",
      "type": "number"
    },
    "query": {
      "$ref": "#/components/schemas/Query"
    },
    "reference_answer": {
      "$ref": "#/components/schemas/ReferenceAnswer"
    },
    "task_id": {
      "minLength": 1,
      "title": "Task Id",
      "type": "string"
    }
  },
  "required": [
    "task_id",
    "query",
    "reference_answer"
  ],
  "title": "MinerTaskRequestModel",
  "type": "object"
}
```

</details>

<a id="model-minertaskrunmodel"></a>
### Model: MinerTaskRunModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` |
| `query` |  |  | req | [Query](#model-query) |
|  | `text` |  | req | `string` |
| `reference_answer` |  |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  | `text` |  | req | `string` |
| `response` |  |  | opt | [Response](#model-response) (nullable) |
|  | `text` |  | req | `string` |
| `task_id` |  |  | req | `string` |
| `uid` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "artifact_id": {
      "minLength": 1,
      "title": "Artifact Id",
      "type": "string"
    },
    "query": {
      "$ref": "#/components/schemas/Query"
    },
    "reference_answer": {
      "$ref": "#/components/schemas/ReferenceAnswer"
    },
    "response": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/Response"
        },
        {
          "type": "null"
        }
      ]
    },
    "task_id": {
      "minLength": 1,
      "title": "Task Id",
      "type": "string"
    },
    "uid": {
      "minimum": 0.0,
      "title": "Uid",
      "type": "integer"
    }
  },
  "required": [
    "uid",
    "artifact_id",
    "task_id",
    "query",
    "reference_answer"
  ],
  "title": "MinerTaskRunModel",
  "type": "object"
}
```

</details>

<a id="model-minertaskrunsubmissionmodel"></a>
### Model: MinerTaskRunSubmissionModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batch_id` |  |  | req | `string` |
| `run` |  |  | req | [MinerTaskRunModel](#model-minertaskrunmodel) |
|  | `artifact_id` |  | req | `string` |
|  | `query` |  | req | [Query](#model-query) |
|  |  | `text` | req | `string` |
|  | `reference_answer` |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `text` | req | `string` |
|  | `response` |  | opt | [Response](#model-response) (nullable) |
|  |  | `text` | req | `string` |
|  | `task_id` |  | req | `string` |
|  | `uid` |  | req | `integer` |
| `score` |  |  | req | `number` |
| `session` |  |  | req | [SessionModel](#model-sessionmodel) |
|  | `expires_at` |  | req | `string` |
|  | `issued_at` |  | req | `string` |
|  | `session_id` |  | req | `string` |
|  | `status` |  | req | `string` |
|  | `uid` |  | req | `integer` |
| `specifics` |  |  | req | [EvaluationDetails](#model-evaluationdetails) |
|  | `elapsed_ms` |  | opt | `number` (nullable) |
|  | `error` |  | opt | [EvaluationError](#model-evaluationerror) (nullable) |
|  |  | `code` | req | `string` |
|  |  | `message` | req | `string` |
|  | `score_breakdown` |  | opt | [ScoreBreakdown](#model-scorebreakdown) (nullable) |
|  |  | `comparison_score` | req | `number` |
|  |  | `scoring_version` | req | `string` |
|  |  | `similarity_score` | req | `number` |
|  |  | `total_score` | req | `number` |
|  | `total_tool_usage` |  | opt | [ToolUsageSummary](#model-toolusagesummary) |
|  |  | `llm` | opt | [LlmUsageSummary](#model-llmusagesummary) |
|  |  | `llm_cost` | opt | `number` (default: 0.0) |
|  |  | `search_tool` | opt | [SearchToolUsageSummary](#model-searchtoolusagesummary) |
|  |  | `search_tool_cost` | opt | `number` (default: 0.0) |
| `usage` |  |  | req | [UsageModel](#model-usagemodel) |
|  | `by_provider` |  | opt | `object` |
|  | `call_count` |  | req | `integer` |
|  | `total_completion_tokens` |  | req | `integer` |
|  | `total_prompt_tokens` |  | req | `integer` |
|  | `total_tokens` |  | req | `integer` |
| `validator` |  |  | req | [ValidatorModel](#model-validatormodel) |
|  | `uid` |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "batch_id": {
      "minLength": 1,
      "title": "Batch Id",
      "type": "string"
    },
    "run": {
      "$ref": "#/components/schemas/MinerTaskRunModel"
    },
    "score": {
      "maximum": 1.0,
      "minimum": 0.0,
      "title": "Score",
      "type": "number"
    },
    "session": {
      "$ref": "#/components/schemas/SessionModel"
    },
    "specifics": {
      "$ref": "#/components/schemas/EvaluationDetails"
    },
    "usage": {
      "$ref": "#/components/schemas/UsageModel"
    },
    "validator": {
      "$ref": "#/components/schemas/ValidatorModel"
    }
  },
  "required": [
    "batch_id",
    "validator",
    "run",
    "score",
    "usage",
    "session",
    "specifics"
  ],
  "title": "MinerTaskRunSubmissionModel",
  "type": "object"
}
```

</details>

<a id="model-progressresponse"></a>
### Model: ProgressResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batch_id` |  |  | req | `string` |
| `completed` |  |  | req | `integer` |
| `miner_task_runs` |  |  | req | array[[MinerTaskRunSubmissionModel](#model-minertaskrunsubmissionmodel)] |
|  | `batch_id` |  | req | `string` |
|  | `run` |  | req | [MinerTaskRunModel](#model-minertaskrunmodel) |
|  |  | `artifact_id` | req | `string` |
|  |  | `query` | req | [Query](#model-query) |
|  |  | `reference_answer` | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `response` | opt | [Response](#model-response) (nullable) |
|  |  | `task_id` | req | `string` |
|  |  | `uid` | req | `integer` |
|  | `score` |  | req | `number` |
|  | `session` |  | req | [SessionModel](#model-sessionmodel) |
|  |  | `expires_at` | req | `string` |
|  |  | `issued_at` | req | `string` |
|  |  | `session_id` | req | `string` |
|  |  | `status` | req | `string` |
|  |  | `uid` | req | `integer` |
|  | `specifics` |  | req | [EvaluationDetails](#model-evaluationdetails) |
|  |  | `elapsed_ms` | opt | `number` (nullable) |
|  |  | `error` | opt | [EvaluationError](#model-evaluationerror) (nullable) |
|  |  | `score_breakdown` | opt | [ScoreBreakdown](#model-scorebreakdown) (nullable) |
|  |  | `total_tool_usage` | opt | [ToolUsageSummary](#model-toolusagesummary) |
|  | `usage` |  | req | [UsageModel](#model-usagemodel) |
|  |  | `by_provider` | opt | `object` |
|  |  | `call_count` | req | `integer` |
|  |  | `total_completion_tokens` | req | `integer` |
|  |  | `total_prompt_tokens` | req | `integer` |
|  |  | `total_tokens` | req | `integer` |
|  | `validator` |  | req | [ValidatorModel](#model-validatormodel) |
|  |  | `uid` | req | `integer` |
| `remaining` |  |  | req | `integer` |
| `total` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "batch_id": {
      "minLength": 1,
      "title": "Batch Id",
      "type": "string"
    },
    "completed": {
      "minimum": 0.0,
      "title": "Completed",
      "type": "integer"
    },
    "miner_task_runs": {
      "items": {
        "$ref": "#/components/schemas/MinerTaskRunSubmissionModel"
      },
      "title": "Miner Task Runs",
      "type": "array"
    },
    "remaining": {
      "minimum": 0.0,
      "title": "Remaining",
      "type": "integer"
    },
    "total": {
      "minimum": 0.0,
      "title": "Total",
      "type": "integer"
    }
  },
  "required": [
    "batch_id",
    "total",
    "completed",
    "remaining",
    "miner_task_runs"
  ],
  "title": "ProgressResponse",
  "type": "object"
}
```

</details>

<a id="model-query"></a>
### Model: Query

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `text` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "text": {
      "minLength": 1,
      "title": "Text",
      "type": "string"
    }
  },
  "required": [
    "text"
  ],
  "title": "Query",
  "type": "object"
}
```

</details>

<a id="model-referenceanswer"></a>
### Model: ReferenceAnswer

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `text` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "text": {
      "minLength": 1,
      "title": "Text",
      "type": "string"
    }
  },
  "required": [
    "text"
  ],
  "title": "ReferenceAnswer",
  "type": "object"
}
```

</details>

<a id="model-response"></a>
### Model: Response

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `text` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "text": {
      "minLength": 1,
      "title": "Text",
      "type": "string"
    }
  },
  "required": [
    "text"
  ],
  "title": "Response",
  "type": "object"
}
```

</details>

<a id="model-scorebreakdown"></a>
### Model: ScoreBreakdown

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `comparison_score` |  |  | req | `number` |
| `scoring_version` |  |  | req | `string` |
| `similarity_score` |  |  | req | `number` |
| `total_score` |  |  | req | `number` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "comparison_score": {
      "maximum": 1.0,
      "minimum": 0.0,
      "title": "Comparison Score",
      "type": "number"
    },
    "scoring_version": {
      "minLength": 1,
      "title": "Scoring Version",
      "type": "string"
    },
    "similarity_score": {
      "maximum": 1.0,
      "minimum": 0.0,
      "title": "Similarity Score",
      "type": "number"
    },
    "total_score": {
      "maximum": 1.0,
      "minimum": 0.0,
      "title": "Total Score",
      "type": "number"
    }
  },
  "required": [
    "comparison_score",
    "similarity_score",
    "total_score",
    "scoring_version"
  ],
  "title": "ScoreBreakdown",
  "type": "object"
}
```

</details>

<a id="model-scriptartifactrequestmodel"></a>
### Model: ScriptArtifactRequestModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` |
| `content_hash` |  |  | req | `string` |
| `size_bytes` |  |  | req | `integer` |
| `uid` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "artifact_id": {
      "minLength": 1,
      "title": "Artifact Id",
      "type": "string"
    },
    "content_hash": {
      "minLength": 1,
      "title": "Content Hash",
      "type": "string"
    },
    "size_bytes": {
      "minimum": 0.0,
      "title": "Size Bytes",
      "type": "integer"
    },
    "uid": {
      "minimum": 0.0,
      "title": "Uid",
      "type": "integer"
    }
  },
  "required": [
    "uid",
    "artifact_id",
    "content_hash",
    "size_bytes"
  ],
  "title": "ScriptArtifactRequestModel",
  "type": "object"
}
```

</details>

<a id="model-searchtoolusagesummary"></a>
### Model: SearchToolUsageSummary

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `call_count` |  |  | opt | `integer` (default: 0) |
| `cost` |  |  | opt | `number` (default: 0.0) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "call_count": {
      "default": 0,
      "title": "Call Count",
      "type": "integer"
    },
    "cost": {
      "default": 0.0,
      "title": "Cost",
      "type": "number"
    }
  },
  "title": "SearchToolUsageSummary",
  "type": "object"
}
```

</details>

<a id="model-sessionmodel"></a>
### Model: SessionModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `expires_at` |  |  | req | `string` |
| `issued_at` |  |  | req | `string` |
| `session_id` |  |  | req | `string` |
| `status` |  |  | req | `string` |
| `uid` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "expires_at": {
      "minLength": 1,
      "title": "Expires At",
      "type": "string"
    },
    "issued_at": {
      "minLength": 1,
      "title": "Issued At",
      "type": "string"
    },
    "session_id": {
      "minLength": 1,
      "title": "Session Id",
      "type": "string"
    },
    "status": {
      "minLength": 1,
      "title": "Status",
      "type": "string"
    },
    "uid": {
      "minimum": 0.0,
      "title": "Uid",
      "type": "integer"
    }
  },
  "required": [
    "session_id",
    "uid",
    "status",
    "issued_at",
    "expires_at"
  ],
  "title": "SessionModel",
  "type": "object"
}
```

</details>

<a id="model-toolbudgetdto"></a>
### Model: ToolBudgetDTO

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `session_budget_usd` |  |  | req | `number` |
| `session_hard_limit_usd` |  |  | req | `number` |
| `session_remaining_budget_usd` |  |  | req | `number` |
| `session_used_budget_usd` |  |  | req | `number` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "session_budget_usd": {
      "minimum": 0.0,
      "title": "Session Budget Usd",
      "type": "number"
    },
    "session_hard_limit_usd": {
      "minimum": 0.0,
      "title": "Session Hard Limit Usd",
      "type": "number"
    },
    "session_remaining_budget_usd": {
      "minimum": 0.0,
      "title": "Session Remaining Budget Usd",
      "type": "number"
    },
    "session_used_budget_usd": {
      "minimum": 0.0,
      "title": "Session Used Budget Usd",
      "type": "number"
    }
  },
  "required": [
    "session_budget_usd",
    "session_hard_limit_usd",
    "session_used_budget_usd",
    "session_remaining_budget_usd"
  ],
  "title": "ToolBudgetDTO",
  "type": "object"
}
```

</details>

<a id="model-toolexecuterequestdto"></a>
### Model: ToolExecuteRequestDTO

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `args` |  |  | opt | array[[JsonValue](#model-jsonvalue)] (default: []) |
| `kwargs` |  |  | opt | `object` (default: {}) |
| `tool` |  |  | req | `string` (enum: [search_web, search_x, search_ai, llm_chat, search_items, test_tool, tooling_info]) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "args": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/JsonValue"
      },
      "title": "Args",
      "type": "array"
    },
    "kwargs": {
      "additionalProperties": {
        "$ref": "#/components/schemas/JsonValue"
      },
      "default": {},
      "title": "Kwargs",
      "type": "object"
    },
    "tool": {
      "enum": [
        "search_web",
        "search_x",
        "search_ai",
        "llm_chat",
        "search_items",
        "test_tool",
        "tooling_info"
      ],
      "title": "Tool",
      "type": "string"
    }
  },
  "required": [
    "tool"
  ],
  "title": "ToolExecuteRequestDTO",
  "type": "object"
}
```

</details>

<a id="model-toolexecuteresponsedto"></a>
### Model: ToolExecuteResponseDTO

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `budget` |  |  | req | [ToolBudgetDTO](#model-toolbudgetdto) |
|  | `session_budget_usd` |  | req | `number` |
|  | `session_hard_limit_usd` |  | req | `number` |
|  | `session_remaining_budget_usd` |  | req | `number` |
|  | `session_used_budget_usd` |  | req | `number` |
| `cost_usd` |  |  | opt | `number` (nullable) |
| `receipt_id` |  |  | req | `string` |
| `response` |  |  | req | [JsonValue](#model-jsonvalue) |
| `result_policy` |  |  | req | `string` |
| `results` |  |  | req | array[[ToolResultDTO](#model-toolresultdto)] |
|  | `index` |  | req | `integer` |
|  | `note` |  | opt | `string` (nullable) |
|  | `raw` |  | opt | [JsonValue](#model-jsonvalue) (nullable) |
|  | `result_id` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (nullable) |
| `usage` |  |  | opt | [ToolUsageDTO](#model-toolusagedto) (nullable) |
|  | `completion_tokens` |  | opt | `integer` (nullable) |
|  | `prompt_tokens` |  | opt | `integer` (nullable) |
|  | `total_tokens` |  | opt | `integer` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "budget": {
      "$ref": "#/components/schemas/ToolBudgetDTO"
    },
    "cost_usd": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Cost Usd"
    },
    "receipt_id": {
      "title": "Receipt Id",
      "type": "string"
    },
    "response": {
      "$ref": "#/components/schemas/JsonValue"
    },
    "result_policy": {
      "title": "Result Policy",
      "type": "string"
    },
    "results": {
      "items": {
        "$ref": "#/components/schemas/ToolResultDTO"
      },
      "title": "Results",
      "type": "array"
    },
    "usage": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/ToolUsageDTO"
        },
        {
          "type": "null"
        }
      ]
    }
  },
  "required": [
    "receipt_id",
    "response",
    "results",
    "result_policy",
    "budget"
  ],
  "title": "ToolExecuteResponseDTO",
  "type": "object"
}
```

</details>

<a id="model-toolresultdto"></a>
### Model: ToolResultDTO

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `index` |  |  | req | `integer` |
| `note` |  |  | opt | `string` (nullable) |
| `raw` |  |  | opt | [JsonValue](#model-jsonvalue) (nullable) |
| `result_id` |  |  | req | `string` |
| `title` |  |  | opt | `string` (nullable) |
| `url` |  |  | opt | `string` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "index": {
      "title": "Index",
      "type": "integer"
    },
    "note": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Note"
    },
    "raw": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/JsonValue"
        },
        {
          "type": "null"
        }
      ]
    },
    "result_id": {
      "title": "Result Id",
      "type": "string"
    },
    "title": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Title"
    },
    "url": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Url"
    }
  },
  "required": [
    "index",
    "result_id"
  ],
  "title": "ToolResultDTO",
  "type": "object"
}
```

</details>

<a id="model-toolusagedto"></a>
### Model: ToolUsageDTO

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `completion_tokens` |  |  | opt | `integer` (nullable) |
| `prompt_tokens` |  |  | opt | `integer` (nullable) |
| `total_tokens` |  |  | opt | `integer` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "completion_tokens": {
      "anyOf": [
        {
          "minimum": 0.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Completion Tokens"
    },
    "prompt_tokens": {
      "anyOf": [
        {
          "minimum": 0.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Prompt Tokens"
    },
    "total_tokens": {
      "anyOf": [
        {
          "minimum": 0.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Total Tokens"
    }
  },
  "title": "ToolUsageDTO",
  "type": "object"
}
```

</details>

<a id="model-toolusagesummary"></a>
### Model: ToolUsageSummary

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `llm` |  |  | opt | [LlmUsageSummary](#model-llmusagesummary) |
|  | `call_count` |  | opt | `integer` (default: 0) |
|  | `completion_tokens` |  | opt | `integer` (default: 0) |
|  | `cost` |  | opt | `number` (default: 0.0) |
|  | `prompt_tokens` |  | opt | `integer` (default: 0) |
|  | `providers` |  | opt | `object` |
|  | `total_tokens` |  | opt | `integer` (default: 0) |
| `llm_cost` |  |  | opt | `number` (default: 0.0) |
| `search_tool` |  |  | opt | [SearchToolUsageSummary](#model-searchtoolusagesummary) |
|  | `call_count` |  | opt | `integer` (default: 0) |
|  | `cost` |  | opt | `number` (default: 0.0) |
| `search_tool_cost` |  |  | opt | `number` (default: 0.0) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "llm": {
      "$ref": "#/components/schemas/LlmUsageSummary"
    },
    "llm_cost": {
      "default": 0.0,
      "title": "Llm Cost",
      "type": "number"
    },
    "search_tool": {
      "$ref": "#/components/schemas/SearchToolUsageSummary"
    },
    "search_tool_cost": {
      "default": 0.0,
      "title": "Search Tool Cost",
      "type": "number"
    }
  },
  "title": "ToolUsageSummary",
  "type": "object"
}
```

</details>

<a id="model-usagemodel"></a>
### Model: UsageModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `by_provider` |  |  | opt | `object` |
| `call_count` |  |  | req | `integer` |
| `total_completion_tokens` |  |  | req | `integer` |
| `total_prompt_tokens` |  |  | req | `integer` |
| `total_tokens` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "by_provider": {
      "additionalProperties": {
        "additionalProperties": {
          "$ref": "#/components/schemas/UsageModelEntry"
        },
        "type": "object"
      },
      "title": "By Provider",
      "type": "object"
    },
    "call_count": {
      "minimum": 0.0,
      "title": "Call Count",
      "type": "integer"
    },
    "total_completion_tokens": {
      "minimum": 0.0,
      "title": "Total Completion Tokens",
      "type": "integer"
    },
    "total_prompt_tokens": {
      "minimum": 0.0,
      "title": "Total Prompt Tokens",
      "type": "integer"
    },
    "total_tokens": {
      "minimum": 0.0,
      "title": "Total Tokens",
      "type": "integer"
    }
  },
  "required": [
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_tokens",
    "call_count"
  ],
  "title": "UsageModel",
  "type": "object"
}
```

</details>

<a id="model-usagemodelentry"></a>
### Model: UsageModelEntry

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `call_count` |  |  | req | `integer` |
| `completion_tokens` |  |  | req | `integer` |
| `prompt_tokens` |  |  | req | `integer` |
| `total_tokens` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "call_count": {
      "minimum": 0.0,
      "title": "Call Count",
      "type": "integer"
    },
    "completion_tokens": {
      "minimum": 0.0,
      "title": "Completion Tokens",
      "type": "integer"
    },
    "prompt_tokens": {
      "minimum": 0.0,
      "title": "Prompt Tokens",
      "type": "integer"
    },
    "total_tokens": {
      "minimum": 0.0,
      "title": "Total Tokens",
      "type": "integer"
    }
  },
  "required": [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "call_count"
  ],
  "title": "UsageModelEntry",
  "type": "object"
}
```

</details>

<a id="model-validationerror"></a>
### Model: ValidationError

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `loc` |  |  | req | array[anyOf: `string` OR `integer`] |
| `msg` |  |  | req | `string` |
| `type` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "loc": {
      "items": {
        "anyOf": [
          {
            "type": "string"
          },
          {
            "type": "integer"
          }
        ]
      },
      "title": "Location",
      "type": "array"
    },
    "msg": {
      "title": "Message",
      "type": "string"
    },
    "type": {
      "title": "Error Type",
      "type": "string"
    }
  },
  "required": [
    "loc",
    "msg",
    "type"
  ],
  "title": "ValidationError",
  "type": "object"
}
```

</details>

<a id="model-validatormodel"></a>
### Model: ValidatorModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `uid` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "uid": {
      "minimum": 0.0,
      "title": "Uid",
      "type": "integer"
    }
  },
  "required": [
    "uid"
  ],
  "title": "ValidatorModel",
  "type": "object"
}
```

</details>

<a id="model-validatorstatusresponse"></a>
### Model: ValidatorStatusResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `last_batch_id` |  |  | opt | `string` (nullable) |
| `last_completed_at` |  |  | opt | `string` (nullable) |
| `last_error` |  |  | opt | `string` (nullable) |
| `last_started_at` |  |  | opt | `string` (nullable) |
| `last_weight_error` |  |  | opt | `string` (nullable) |
| `last_weight_submission_at` |  |  | opt | `string` (nullable) |
| `queued_batches` |  |  | opt | `integer` (default: 0) |
| `running` |  |  | opt | `boolean` (default: False) |
| `status` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "last_batch_id": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Last Batch Id"
    },
    "last_completed_at": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Last Completed At"
    },
    "last_error": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Last Error"
    },
    "last_started_at": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Last Started At"
    },
    "last_weight_error": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Last Weight Error"
    },
    "last_weight_submission_at": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Last Weight Submission At"
    },
    "queued_batches": {
      "default": 0,
      "minimum": 0.0,
      "title": "Queued Batches",
      "type": "integer"
    },
    "running": {
      "default": false,
      "title": "Running",
      "type": "boolean"
    },
    "status": {
      "minLength": 1,
      "title": "Status",
      "type": "string"
    }
  },
  "required": [
    "status"
  ],
  "title": "ValidatorStatusResponse",
  "type": "object"
}
```

</details>
