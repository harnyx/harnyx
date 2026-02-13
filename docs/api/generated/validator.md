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
Body: [MinerTaskBatchSpec](#model-minertaskbatchspec)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `candidates` |  |  | req | array[[ScriptArtifactSpec](#model-scriptartifactspec)] |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `content_hash` |  | req | `string` |
|  | `size_bytes` |  | req | `integer` |
|  | `uid` |  | req | `integer` |
| `claims` |  |  | req | array[[MinerTaskClaim](#model-minertaskclaim)] |
|  | `budget_usd` |  | opt | `number` (default: 0.05) |
|  | `claim_id` |  | req | `string` (format: uuid) |
|  | `context` |  | opt | [FeedSearchContext](#model-feedsearchcontext) (nullable) |
|  |  | `enqueue_seq` | req | `integer` |
|  |  | `feed_id` | req | `string` (format: uuid) |
|  | `reference_answer` |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `citations` | opt | array[[Citation](#model-citation)] (default: []) |
|  |  | `justification` | req | `string` |
|  |  | `spans` | opt | array[[Span](#model-span)] (default: []) |
|  |  | `verdict` | req | `integer` |
|  | `rubric` |  | req | [Rubric](#model-rubric) |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | [VerdictOptions](#model-verdictoptions) |
|  | `text` |  | req | `string` |
| `created_at_iso` |  |  | req | `string` |
| `cutoff_at_iso` |  |  | req | `string` |
| `entrypoint` |  |  | req | `string` |

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
| `miner_task_results` |  |  | req | array[[MinerTaskResultModel](#model-minertaskresultmodel)] |
|  | `batch_id` |  | req | `string` |
|  | `criterion_evaluation` |  | req | [MinerTaskResultCriterionEvaluationModel](#model-minertaskresultcriterionevaluationmodel) |
|  |  | `artifact_id` | req | `string` |
|  |  | `citations` | req | array[[MinerTaskResultCitationModel](#model-minertaskresultcitationmodel)] |
|  |  | `claim_id` | req | `string` |
|  |  | `criterion_evaluation_id` | req | `string` |
|  |  | `justification` | req | `string` |
|  |  | `uid` | req | `integer` |
|  |  | `verdict` | req | `integer` |
|  | `score` |  | req | [MinerTaskResultScoreModel](#model-minertaskresultscoremodel) |
|  |  | `error_code` | opt | `string` (nullable) |
|  |  | `error_message` | opt | `string` (nullable) |
|  |  | `failed_citation_ids` | req | array[`string`] |
|  |  | `grader_rationale` | opt | `string` (nullable) |
|  |  | `justification_pass` | req | `boolean` |
|  |  | `support_score` | req | `number` |
|  |  | `verdict_score` | req | `number` |
|  | `session` |  | req | [SessionModel](#model-sessionmodel) |
|  |  | `expires_at` | req | `string` |
|  |  | `issued_at` | req | `string` |
|  |  | `session_id` | req | `string` |
|  |  | `status` | req | `string` |
|  |  | `uid` | req | `integer` |
|  | `total_tool_usage` |  | req | [ToolUsageSummary](#model-toolusagesummary) |
|  |  | `llm` | opt | [LlmUsageSummary](#model-llmusagesummary) |
|  |  | `llm_cost` | opt | `number` (default: 0.0) |
|  |  | `search_tool` | opt | [SearchToolUsageSummary](#model-searchtoolusagesummary) |
|  |  | `search_tool_cost` | opt | `number` (default: 0.0) |
|  | `usage` |  | req | [UsageModel](#model-usagemodel) |
|  |  | `by_provider` | req | `object` |
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

**Auth**: Token + session in body (ToolExecuteRequestDTO).

**Request**
Content-Type: `application/json`
Body: [ToolExecuteRequestDTO](#model-toolexecuterequestdto)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `args` |  |  | opt | array[[JsonValue](#model-jsonvalue)] (default: []) |
| `kwargs` |  |  | opt | `object` (default: {}) |
| `session_id` |  |  | req | `string` (format: uuid) |
| `token` |  |  | req | `string` |
| `tool` |  |  | req | `string` (enum: [search_web, search_x, search_ai, llm_chat, search_items, test_tool, tooling_info]) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [ToolExecuteResponseDTO](#model-toolexecuteresponsedto)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `budget` |  |  | req | [ToolBudgetDTO](#model-toolbudgetdto) |
|  | `session_budget_usd` |  | req | `number` |
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
  "properties": {
    "batch_id": {
      "title": "Batch Id",
      "type": "string"
    },
    "caller": {
      "title": "Caller",
      "type": "string"
    },
    "status": {
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

<a id="model-citation"></a>
### Model: Citation

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `note` |  |  | req | `string` |
| `url` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "note": {
      "title": "Note",
      "type": "string"
    },
    "url": {
      "title": "Url",
      "type": "string"
    }
  },
  "required": [
    "url",
    "note"
  ],
  "title": "Citation",
  "type": "object"
}
```

</details>

<a id="model-feedsearchcontext"></a>
### Model: FeedSearchContext

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `enqueue_seq` |  |  | req | `integer` |
| `feed_id` |  |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "enqueue_seq": {
      "minimum": 0.0,
      "title": "Enqueue Seq",
      "type": "integer"
    },
    "feed_id": {
      "format": "uuid",
      "title": "Feed Id",
      "type": "string"
    }
  },
  "required": [
    "feed_id",
    "enqueue_seq"
  ],
  "title": "FeedSearchContext",
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

<a id="model-minertaskbatchspec"></a>
### Model: MinerTaskBatchSpec

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `candidates` |  |  | req | array[[ScriptArtifactSpec](#model-scriptartifactspec)] |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `content_hash` |  | req | `string` |
|  | `size_bytes` |  | req | `integer` |
|  | `uid` |  | req | `integer` |
| `claims` |  |  | req | array[[MinerTaskClaim](#model-minertaskclaim)] |
|  | `budget_usd` |  | opt | `number` (default: 0.05) |
|  | `claim_id` |  | req | `string` (format: uuid) |
|  | `context` |  | opt | [FeedSearchContext](#model-feedsearchcontext) (nullable) |
|  |  | `enqueue_seq` | req | `integer` |
|  |  | `feed_id` | req | `string` (format: uuid) |
|  | `reference_answer` |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `citations` | opt | array[[Citation](#model-citation)] (default: []) |
|  |  | `justification` | req | `string` |
|  |  | `spans` | opt | array[[Span](#model-span)] (default: []) |
|  |  | `verdict` | req | `integer` |
|  | `rubric` |  | req | [Rubric](#model-rubric) |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | [VerdictOptions](#model-verdictoptions) |
|  | `text` |  | req | `string` |
| `created_at_iso` |  |  | req | `string` |
| `cutoff_at_iso` |  |  | req | `string` |
| `entrypoint` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "batch_id": {
      "format": "uuid",
      "title": "Batch Id",
      "type": "string"
    },
    "candidates": {
      "items": {
        "$ref": "#/components/schemas/ScriptArtifactSpec"
      },
      "title": "Candidates",
      "type": "array"
    },
    "claims": {
      "items": {
        "$ref": "#/components/schemas/MinerTaskClaim"
      },
      "title": "Claims",
      "type": "array"
    },
    "created_at_iso": {
      "title": "Created At Iso",
      "type": "string"
    },
    "cutoff_at_iso": {
      "title": "Cutoff At Iso",
      "type": "string"
    },
    "entrypoint": {
      "title": "Entrypoint",
      "type": "string"
    }
  },
  "required": [
    "batch_id",
    "entrypoint",
    "cutoff_at_iso",
    "created_at_iso",
    "claims",
    "candidates"
  ],
  "title": "MinerTaskBatchSpec",
  "type": "object"
}
```

</details>

<a id="model-minertaskclaim"></a>
### Model: MinerTaskClaim

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `budget_usd` |  |  | opt | `number` (default: 0.05) |
| `claim_id` |  |  | req | `string` (format: uuid) |
| `context` |  |  | opt | [FeedSearchContext](#model-feedsearchcontext) (nullable) |
|  | `enqueue_seq` |  | req | `integer` |
|  | `feed_id` |  | req | `string` (format: uuid) |
| `reference_answer` |  |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  | `citations` |  | opt | array[[Citation](#model-citation)] (default: []) |
|  |  | `note` | req | `string` |
|  |  | `url` | req | `string` |
|  | `justification` |  | req | `string` |
|  | `spans` |  | opt | array[[Span](#model-span)] (default: []) |
|  |  | `end` | req | `integer` |
|  |  | `excerpt` | req | `string` |
|  |  | `start` | req | `integer` |
|  | `verdict` |  | req | `integer` |
| `rubric` |  |  | req | [Rubric](#model-rubric) |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
|  | `verdict_options` |  | req | [VerdictOptions](#model-verdictoptions) |
|  |  | `options` | req | array[[VerdictOption](#model-verdictoption)] |
| `text` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "budget_usd": {
      "default": 0.05,
      "title": "Budget Usd",
      "type": "number"
    },
    "claim_id": {
      "format": "uuid",
      "title": "Claim Id",
      "type": "string"
    },
    "context": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/FeedSearchContext"
        },
        {
          "type": "null"
        }
      ]
    },
    "reference_answer": {
      "$ref": "#/components/schemas/ReferenceAnswer"
    },
    "rubric": {
      "$ref": "#/components/schemas/Rubric"
    },
    "text": {
      "title": "Text",
      "type": "string"
    }
  },
  "required": [
    "claim_id",
    "text",
    "rubric",
    "reference_answer"
  ],
  "title": "MinerTaskClaim",
  "type": "object"
}
```

</details>

<a id="model-minertaskresultcitationmodel"></a>
### Model: MinerTaskResultCitationModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `note` |  |  | opt | `string` (nullable) |
| `receipt_id` |  |  | opt | `string` (nullable) |
| `result_id` |  |  | opt | `string` (nullable) |
| `url` |  |  | opt | `string` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
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
    "receipt_id": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Receipt Id"
    },
    "result_id": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Result Id"
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
  "title": "MinerTaskResultCitationModel",
  "type": "object"
}
```

</details>

<a id="model-minertaskresultcriterionevaluationmodel"></a>
### Model: MinerTaskResultCriterionEvaluationModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` |
| `citations` |  |  | req | array[[MinerTaskResultCitationModel](#model-minertaskresultcitationmodel)] |
|  | `note` |  | opt | `string` (nullable) |
|  | `receipt_id` |  | opt | `string` (nullable) |
|  | `result_id` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (nullable) |
| `claim_id` |  |  | req | `string` |
| `criterion_evaluation_id` |  |  | req | `string` |
| `justification` |  |  | req | `string` |
| `uid` |  |  | req | `integer` |
| `verdict` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "artifact_id": {
      "title": "Artifact Id",
      "type": "string"
    },
    "citations": {
      "items": {
        "$ref": "#/components/schemas/MinerTaskResultCitationModel"
      },
      "title": "Citations",
      "type": "array"
    },
    "claim_id": {
      "title": "Claim Id",
      "type": "string"
    },
    "criterion_evaluation_id": {
      "title": "Criterion Evaluation Id",
      "type": "string"
    },
    "justification": {
      "title": "Justification",
      "type": "string"
    },
    "uid": {
      "title": "Uid",
      "type": "integer"
    },
    "verdict": {
      "title": "Verdict",
      "type": "integer"
    }
  },
  "required": [
    "criterion_evaluation_id",
    "uid",
    "artifact_id",
    "claim_id",
    "verdict",
    "justification",
    "citations"
  ],
  "title": "MinerTaskResultCriterionEvaluationModel",
  "type": "object"
}
```

</details>

<a id="model-minertaskresultmodel"></a>
### Model: MinerTaskResultModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batch_id` |  |  | req | `string` |
| `criterion_evaluation` |  |  | req | [MinerTaskResultCriterionEvaluationModel](#model-minertaskresultcriterionevaluationmodel) |
|  | `artifact_id` |  | req | `string` |
|  | `citations` |  | req | array[[MinerTaskResultCitationModel](#model-minertaskresultcitationmodel)] |
|  |  | `note` | opt | `string` (nullable) |
|  |  | `receipt_id` | opt | `string` (nullable) |
|  |  | `result_id` | opt | `string` (nullable) |
|  |  | `url` | opt | `string` (nullable) |
|  | `claim_id` |  | req | `string` |
|  | `criterion_evaluation_id` |  | req | `string` |
|  | `justification` |  | req | `string` |
|  | `uid` |  | req | `integer` |
|  | `verdict` |  | req | `integer` |
| `score` |  |  | req | [MinerTaskResultScoreModel](#model-minertaskresultscoremodel) |
|  | `error_code` |  | opt | `string` (nullable) |
|  | `error_message` |  | opt | `string` (nullable) |
|  | `failed_citation_ids` |  | req | array[`string`] |
|  | `grader_rationale` |  | opt | `string` (nullable) |
|  | `justification_pass` |  | req | `boolean` |
|  | `support_score` |  | req | `number` |
|  | `verdict_score` |  | req | `number` |
| `session` |  |  | req | [SessionModel](#model-sessionmodel) |
|  | `expires_at` |  | req | `string` |
|  | `issued_at` |  | req | `string` |
|  | `session_id` |  | req | `string` |
|  | `status` |  | req | `string` |
|  | `uid` |  | req | `integer` |
| `total_tool_usage` |  |  | req | [ToolUsageSummary](#model-toolusagesummary) |
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
| `usage` |  |  | req | [UsageModel](#model-usagemodel) |
|  | `by_provider` |  | req | `object` |
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
  "properties": {
    "batch_id": {
      "title": "Batch Id",
      "type": "string"
    },
    "criterion_evaluation": {
      "$ref": "#/components/schemas/MinerTaskResultCriterionEvaluationModel"
    },
    "score": {
      "$ref": "#/components/schemas/MinerTaskResultScoreModel"
    },
    "session": {
      "$ref": "#/components/schemas/SessionModel"
    },
    "total_tool_usage": {
      "$ref": "#/components/schemas/ToolUsageSummary"
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
    "criterion_evaluation",
    "score",
    "usage",
    "session",
    "total_tool_usage"
  ],
  "title": "MinerTaskResultModel",
  "type": "object"
}
```

</details>

<a id="model-minertaskresultscoremodel"></a>
### Model: MinerTaskResultScoreModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `error_code` |  |  | opt | `string` (nullable) |
| `error_message` |  |  | opt | `string` (nullable) |
| `failed_citation_ids` |  |  | req | array[`string`] |
| `grader_rationale` |  |  | opt | `string` (nullable) |
| `justification_pass` |  |  | req | `boolean` |
| `support_score` |  |  | req | `number` |
| `verdict_score` |  |  | req | `number` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "error_code": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Error Code"
    },
    "error_message": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Error Message"
    },
    "failed_citation_ids": {
      "items": {
        "type": "string"
      },
      "title": "Failed Citation Ids",
      "type": "array"
    },
    "grader_rationale": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Grader Rationale"
    },
    "justification_pass": {
      "title": "Justification Pass",
      "type": "boolean"
    },
    "support_score": {
      "title": "Support Score",
      "type": "number"
    },
    "verdict_score": {
      "title": "Verdict Score",
      "type": "number"
    }
  },
  "required": [
    "verdict_score",
    "support_score",
    "justification_pass",
    "failed_citation_ids"
  ],
  "title": "MinerTaskResultScoreModel",
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
| `miner_task_results` |  |  | req | array[[MinerTaskResultModel](#model-minertaskresultmodel)] |
|  | `batch_id` |  | req | `string` |
|  | `criterion_evaluation` |  | req | [MinerTaskResultCriterionEvaluationModel](#model-minertaskresultcriterionevaluationmodel) |
|  |  | `artifact_id` | req | `string` |
|  |  | `citations` | req | array[[MinerTaskResultCitationModel](#model-minertaskresultcitationmodel)] |
|  |  | `claim_id` | req | `string` |
|  |  | `criterion_evaluation_id` | req | `string` |
|  |  | `justification` | req | `string` |
|  |  | `uid` | req | `integer` |
|  |  | `verdict` | req | `integer` |
|  | `score` |  | req | [MinerTaskResultScoreModel](#model-minertaskresultscoremodel) |
|  |  | `error_code` | opt | `string` (nullable) |
|  |  | `error_message` | opt | `string` (nullable) |
|  |  | `failed_citation_ids` | req | array[`string`] |
|  |  | `grader_rationale` | opt | `string` (nullable) |
|  |  | `justification_pass` | req | `boolean` |
|  |  | `support_score` | req | `number` |
|  |  | `verdict_score` | req | `number` |
|  | `session` |  | req | [SessionModel](#model-sessionmodel) |
|  |  | `expires_at` | req | `string` |
|  |  | `issued_at` | req | `string` |
|  |  | `session_id` | req | `string` |
|  |  | `status` | req | `string` |
|  |  | `uid` | req | `integer` |
|  | `total_tool_usage` |  | req | [ToolUsageSummary](#model-toolusagesummary) |
|  |  | `llm` | opt | [LlmUsageSummary](#model-llmusagesummary) |
|  |  | `llm_cost` | opt | `number` (default: 0.0) |
|  |  | `search_tool` | opt | [SearchToolUsageSummary](#model-searchtoolusagesummary) |
|  |  | `search_tool_cost` | opt | `number` (default: 0.0) |
|  | `usage` |  | req | [UsageModel](#model-usagemodel) |
|  |  | `by_provider` | req | `object` |
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
  "properties": {
    "batch_id": {
      "title": "Batch Id",
      "type": "string"
    },
    "completed": {
      "title": "Completed",
      "type": "integer"
    },
    "miner_task_results": {
      "items": {
        "$ref": "#/components/schemas/MinerTaskResultModel"
      },
      "title": "Miner Task Results",
      "type": "array"
    },
    "remaining": {
      "title": "Remaining",
      "type": "integer"
    },
    "total": {
      "title": "Total",
      "type": "integer"
    }
  },
  "required": [
    "batch_id",
    "total",
    "completed",
    "remaining",
    "miner_task_results"
  ],
  "title": "ProgressResponse",
  "type": "object"
}
```

</details>

<a id="model-referenceanswer"></a>
### Model: ReferenceAnswer

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `citations` |  |  | opt | array[[Citation](#model-citation)] (default: []) |
|  | `note` |  | req | `string` |
|  | `url` |  | req | `string` |
| `justification` |  |  | req | `string` |
| `spans` |  |  | opt | array[[Span](#model-span)] (default: []) |
|  | `end` |  | req | `integer` |
|  | `excerpt` |  | req | `string` |
|  | `start` |  | req | `integer` |
| `verdict` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "citations": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/Citation"
      },
      "title": "Citations",
      "type": "array"
    },
    "justification": {
      "title": "Justification",
      "type": "string"
    },
    "spans": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/Span"
      },
      "title": "Spans",
      "type": "array"
    },
    "verdict": {
      "title": "Verdict",
      "type": "integer"
    }
  },
  "required": [
    "verdict",
    "justification"
  ],
  "title": "ReferenceAnswer",
  "type": "object"
}
```

</details>

<a id="model-rubric"></a>
### Model: Rubric

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `description` |  |  | req | `string` |
| `title` |  |  | req | `string` |
| `verdict_options` |  |  | req | [VerdictOptions](#model-verdictoptions) |
|  | `options` |  | req | array[[VerdictOption](#model-verdictoption)] |
|  |  | `description` | req | `string` |
|  |  | `value` | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "description": {
      "title": "Description",
      "type": "string"
    },
    "title": {
      "title": "Title",
      "type": "string"
    },
    "verdict_options": {
      "$ref": "#/components/schemas/VerdictOptions"
    }
  },
  "required": [
    "title",
    "description",
    "verdict_options"
  ],
  "title": "Rubric",
  "type": "object"
}
```

</details>

<a id="model-scriptartifactspec"></a>
### Model: ScriptArtifactSpec

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `content_hash` |  |  | req | `string` |
| `size_bytes` |  |  | req | `integer` |
| `uid` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "artifact_id": {
      "format": "uuid",
      "title": "Artifact Id",
      "type": "string"
    },
    "content_hash": {
      "title": "Content Hash",
      "type": "string"
    },
    "size_bytes": {
      "title": "Size Bytes",
      "type": "integer"
    },
    "uid": {
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
  "title": "ScriptArtifactSpec",
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
  "properties": {
    "expires_at": {
      "title": "Expires At",
      "type": "string"
    },
    "issued_at": {
      "title": "Issued At",
      "type": "string"
    },
    "session_id": {
      "title": "Session Id",
      "type": "string"
    },
    "status": {
      "title": "Status",
      "type": "string"
    },
    "uid": {
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

<a id="model-span"></a>
### Model: Span

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `end` |  |  | req | `integer` |
| `excerpt` |  |  | req | `string` |
| `start` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "end": {
      "title": "End",
      "type": "integer"
    },
    "excerpt": {
      "title": "Excerpt",
      "type": "string"
    },
    "start": {
      "title": "Start",
      "type": "integer"
    }
  },
  "required": [
    "excerpt",
    "start",
    "end"
  ],
  "title": "Span",
  "type": "object"
}
```

</details>

<a id="model-toolbudgetdto"></a>
### Model: ToolBudgetDTO

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `session_budget_usd` |  |  | req | `number` |
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
| `session_id` |  |  | req | `string` (format: uuid) |
| `token` |  |  | req | `string` |
| `tool` |  |  | req | `string` (enum: [search_web, search_x, search_ai, llm_chat, search_items, test_tool, tooling_info]) |

<details>
<summary>JSON schema</summary>

```json
{
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
    "session_id": {
      "format": "uuid",
      "title": "Session Id",
      "type": "string"
    },
    "token": {
      "title": "Token",
      "type": "string"
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
    "session_id",
    "token",
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
| `by_provider` |  |  | req | `object` |
| `call_count` |  |  | req | `integer` |
| `total_completion_tokens` |  |  | req | `integer` |
| `total_prompt_tokens` |  |  | req | `integer` |
| `total_tokens` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
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
      "title": "Call Count",
      "type": "integer"
    },
    "total_completion_tokens": {
      "title": "Total Completion Tokens",
      "type": "integer"
    },
    "total_prompt_tokens": {
      "title": "Total Prompt Tokens",
      "type": "integer"
    },
    "total_tokens": {
      "title": "Total Tokens",
      "type": "integer"
    }
  },
  "required": [
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_tokens",
    "call_count",
    "by_provider"
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
  "properties": {
    "call_count": {
      "title": "Call Count",
      "type": "integer"
    },
    "completion_tokens": {
      "title": "Completion Tokens",
      "type": "integer"
    },
    "prompt_tokens": {
      "title": "Prompt Tokens",
      "type": "integer"
    },
    "total_tokens": {
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
  "properties": {
    "uid": {
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
      "title": "Queued Batches",
      "type": "integer"
    },
    "running": {
      "default": false,
      "title": "Running",
      "type": "boolean"
    },
    "status": {
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

<a id="model-verdictoption"></a>
### Model: VerdictOption

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `description` |  |  | req | `string` |
| `value` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "description": {
      "title": "Description",
      "type": "string"
    },
    "value": {
      "title": "Value",
      "type": "integer"
    }
  },
  "required": [
    "value",
    "description"
  ],
  "title": "VerdictOption",
  "type": "object"
}
```

</details>

<a id="model-verdictoptions"></a>
### Model: VerdictOptions

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `options` |  |  | req | array[[VerdictOption](#model-verdictoption)] |
|  | `description` |  | req | `string` |
|  | `value` |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "options": {
      "items": {
        "$ref": "#/components/schemas/VerdictOption"
      },
      "title": "Options",
      "type": "array"
    }
  },
  "required": [
    "options"
  ],
  "title": "VerdictOptions",
  "type": "object"
}
```

</details>
