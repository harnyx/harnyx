# Validator API reference (generated)

Generated from FastAPI OpenAPI.

## Domains
- [miner-task-batches](#miner-task-batches)
  - [POST /validator/miner-task-batches/{batch_id}/similarity](#endpoint-post-validator-miner-task-batches-batch_id-similarity)
- [status](#status)
  - [GET /validator/status](#endpoint-get-validator-status)
- [tools](#tools)
  - [POST /v1/tools/execute](#endpoint-post-v1-tools-execute)

## miner-task-batches

### {batch_id}

#### similarity

<a id="endpoint-post-validator-miner-task-batches-batch_id-similarity"></a>
##### POST /validator/miner-task-batches/{batch_id}/similarity

Run a validator-owned similarity judge for a dethroning miner script candidate.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `batch_id` | path | req | `string` (format: uuid) |

**Request**
Content-Type: `application/json`
Body: [SimilarityJudgeRequestModel](#model-similarityjudgerequestmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `candidate_artifact_id` |  |  | req | `string` |
| `candidate_diff` |  |  | req | `string` |
| `candidate_miner_uid` |  |  | req | `integer` |
| `incumbent_artifact_id` |  |  | req | `string` |
| `incumbent_miner_uid` |  |  | req | `integer` |
| `incumbent_script` |  |  | req | `string` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [SimilarityJudgeResponseModel](#model-similarityjudgeresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `model` |  |  | req | `string` |
| `provider` |  |  | req | `string` |
| `reasoning` |  |  | opt | `string` (nullable) |
| `reasoning_tokens` |  |  | opt | `integer` (nullable) |
| `verdict` |  |  | req | `string` (enum: [not_duplicate, duplicate]) |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `ctx` |  | opt | `object` |
|  | `input` |  | opt | `object` |
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
| `hotkey` |  |  | req | `string` |
| `is_chutes_configured` |  |  | opt | `boolean` (default: False) |
| `is_openrouter_configured` |  |  | opt | `boolean` (default: False) |
| `last_batch_id` |  |  | opt | `string` (nullable) |
| `last_completed_at` |  |  | opt | `string` (nullable) |
| `last_error` |  |  | opt | `string` (nullable) |
| `last_started_at` |  |  | opt | `string` (nullable) |
| `last_weight_error` |  |  | opt | `string` (nullable) |
| `last_weight_submission_at` |  |  | opt | `string` (nullable) |
| `queued_batches` |  |  | opt | `integer` (default: 0) |
| `resource_usage` |  |  | opt | [ValidatorResourceUsageResponse](#model-validatorresourceusageresponse) (nullable) |
|  | `captured_at` |  | req | `string` |
|  | `cpu_capacity_cores` |  | req | `number` |
|  | `cpu_percent` |  | req | `number` |
|  | `disk_percent` |  | req | `number` |
|  | `disk_total_bytes` |  | req | `integer` |
|  | `disk_used_bytes` |  | req | `integer` |
|  | `memory_percent` |  | req | `number` |
|  | `memory_total_bytes` |  | req | `integer` |
|  | `memory_used_bytes` |  | req | `integer` |
| `running` |  |  | opt | `boolean` (default: False) |
| `signature_hex` |  |  | opt | `string` (nullable) |
| `status` |  |  | req | `string` |

`500` Internal Server Error
Content-Type: `application/json`
Body: [ValidatorInternalErrorResponse](#model-validatorinternalerrorresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `error_code` |  |  | req | `string` |
| `error_message` |  |  | req | `string` |
| `exception_type` |  |  | req | `string` |
| `request_id` |  |  | req | `string` |
| `traceback` |  |  | req | `string` |



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
| `tool` |  |  | req | `string` (enum: [search_web, search_ai, fetch_page, llm_chat, test_tool, tooling_info]) |

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
|  | `ctx` |  | opt | `object` |
|  | `input` |  | opt | `object` |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |



## Models

<a id="model-httpvalidationerror"></a>
### Model: HTTPValidationError

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `ctx` |  | opt | `object` |
|  | `input` |  | opt | `object` |
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

<a id="model-similarityjudgerequestmodel"></a>
### Model: SimilarityJudgeRequestModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `candidate_artifact_id` |  |  | req | `string` |
| `candidate_diff` |  |  | req | `string` |
| `candidate_miner_uid` |  |  | req | `integer` |
| `incumbent_artifact_id` |  |  | req | `string` |
| `incumbent_miner_uid` |  |  | req | `integer` |
| `incumbent_script` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "candidate_artifact_id": {
      "minLength": 1,
      "title": "Candidate Artifact Id",
      "type": "string"
    },
    "candidate_diff": {
      "minLength": 1,
      "title": "Candidate Diff",
      "type": "string"
    },
    "candidate_miner_uid": {
      "minimum": 0.0,
      "title": "Candidate Miner Uid",
      "type": "integer"
    },
    "incumbent_artifact_id": {
      "minLength": 1,
      "title": "Incumbent Artifact Id",
      "type": "string"
    },
    "incumbent_miner_uid": {
      "minimum": 0.0,
      "title": "Incumbent Miner Uid",
      "type": "integer"
    },
    "incumbent_script": {
      "minLength": 1,
      "title": "Incumbent Script",
      "type": "string"
    }
  },
  "required": [
    "candidate_artifact_id",
    "incumbent_artifact_id",
    "candidate_miner_uid",
    "incumbent_miner_uid",
    "incumbent_script",
    "candidate_diff"
  ],
  "title": "SimilarityJudgeRequestModel",
  "type": "object"
}
```

</details>

<a id="model-similarityjudgeresponsemodel"></a>
### Model: SimilarityJudgeResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `model` |  |  | req | `string` |
| `provider` |  |  | req | `string` |
| `reasoning` |  |  | opt | `string` (nullable) |
| `reasoning_tokens` |  |  | opt | `integer` (nullable) |
| `verdict` |  |  | req | `string` (enum: [not_duplicate, duplicate]) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "model": {
      "minLength": 1,
      "title": "Model",
      "type": "string"
    },
    "provider": {
      "minLength": 1,
      "title": "Provider",
      "type": "string"
    },
    "reasoning": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Reasoning"
    },
    "reasoning_tokens": {
      "anyOf": [
        {
          "minimum": 0.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Reasoning Tokens"
    },
    "verdict": {
      "enum": [
        "not_duplicate",
        "duplicate"
      ],
      "title": "Verdict",
      "type": "string"
    }
  },
  "required": [
    "verdict",
    "model",
    "provider"
  ],
  "title": "SimilarityJudgeResponseModel",
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
| `tool` |  |  | req | `string` (enum: [search_web, search_ai, fetch_page, llm_chat, test_tool, tooling_info]) |

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
        "search_ai",
        "fetch_page",
        "llm_chat",
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

<a id="model-validationerror"></a>
### Model: ValidationError

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `ctx` |  |  | opt | `object` |
| `input` |  |  | opt | `object` |
| `loc` |  |  | req | array[anyOf: `string` OR `integer`] |
| `msg` |  |  | req | `string` |
| `type` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "ctx": {
      "title": "Context",
      "type": "object"
    },
    "input": {
      "title": "Input"
    },
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

<a id="model-validatorinternalerrorresponse"></a>
### Model: ValidatorInternalErrorResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `error_code` |  |  | req | `string` |
| `error_message` |  |  | req | `string` |
| `exception_type` |  |  | req | `string` |
| `request_id` |  |  | req | `string` |
| `traceback` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "error_code": {
      "minLength": 1,
      "title": "Error Code",
      "type": "string"
    },
    "error_message": {
      "minLength": 1,
      "title": "Error Message",
      "type": "string"
    },
    "exception_type": {
      "minLength": 1,
      "title": "Exception Type",
      "type": "string"
    },
    "request_id": {
      "minLength": 1,
      "title": "Request Id",
      "type": "string"
    },
    "traceback": {
      "minLength": 1,
      "title": "Traceback",
      "type": "string"
    }
  },
  "required": [
    "error_code",
    "error_message",
    "exception_type",
    "request_id",
    "traceback"
  ],
  "title": "ValidatorInternalErrorResponse",
  "type": "object"
}
```

</details>

<a id="model-validatorresourceusageresponse"></a>
### Model: ValidatorResourceUsageResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `captured_at` |  |  | req | `string` |
| `cpu_capacity_cores` |  |  | req | `number` |
| `cpu_percent` |  |  | req | `number` |
| `disk_percent` |  |  | req | `number` |
| `disk_total_bytes` |  |  | req | `integer` |
| `disk_used_bytes` |  |  | req | `integer` |
| `memory_percent` |  |  | req | `number` |
| `memory_total_bytes` |  |  | req | `integer` |
| `memory_used_bytes` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "captured_at": {
      "minLength": 1,
      "title": "Captured At",
      "type": "string"
    },
    "cpu_capacity_cores": {
      "minimum": 0.0,
      "title": "Cpu Capacity Cores",
      "type": "number"
    },
    "cpu_percent": {
      "minimum": 0.0,
      "title": "Cpu Percent",
      "type": "number"
    },
    "disk_percent": {
      "minimum": 0.0,
      "title": "Disk Percent",
      "type": "number"
    },
    "disk_total_bytes": {
      "minimum": 0.0,
      "title": "Disk Total Bytes",
      "type": "integer"
    },
    "disk_used_bytes": {
      "minimum": 0.0,
      "title": "Disk Used Bytes",
      "type": "integer"
    },
    "memory_percent": {
      "minimum": 0.0,
      "title": "Memory Percent",
      "type": "number"
    },
    "memory_total_bytes": {
      "minimum": 0.0,
      "title": "Memory Total Bytes",
      "type": "integer"
    },
    "memory_used_bytes": {
      "minimum": 0.0,
      "title": "Memory Used Bytes",
      "type": "integer"
    }
  },
  "required": [
    "captured_at",
    "cpu_percent",
    "cpu_capacity_cores",
    "memory_used_bytes",
    "memory_total_bytes",
    "memory_percent",
    "disk_used_bytes",
    "disk_total_bytes",
    "disk_percent"
  ],
  "title": "ValidatorResourceUsageResponse",
  "type": "object"
}
```

</details>

<a id="model-validatorstatusresponse"></a>
### Model: ValidatorStatusResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `hotkey` |  |  | req | `string` |
| `is_chutes_configured` |  |  | opt | `boolean` (default: False) |
| `is_openrouter_configured` |  |  | opt | `boolean` (default: False) |
| `last_batch_id` |  |  | opt | `string` (nullable) |
| `last_completed_at` |  |  | opt | `string` (nullable) |
| `last_error` |  |  | opt | `string` (nullable) |
| `last_started_at` |  |  | opt | `string` (nullable) |
| `last_weight_error` |  |  | opt | `string` (nullable) |
| `last_weight_submission_at` |  |  | opt | `string` (nullable) |
| `queued_batches` |  |  | opt | `integer` (default: 0) |
| `resource_usage` |  |  | opt | [ValidatorResourceUsageResponse](#model-validatorresourceusageresponse) (nullable) |
|  | `captured_at` |  | req | `string` |
|  | `cpu_capacity_cores` |  | req | `number` |
|  | `cpu_percent` |  | req | `number` |
|  | `disk_percent` |  | req | `number` |
|  | `disk_total_bytes` |  | req | `integer` |
|  | `disk_used_bytes` |  | req | `integer` |
|  | `memory_percent` |  | req | `number` |
|  | `memory_total_bytes` |  | req | `integer` |
|  | `memory_used_bytes` |  | req | `integer` |
| `running` |  |  | opt | `boolean` (default: False) |
| `signature_hex` |  |  | opt | `string` (nullable) |
| `status` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": true,
  "properties": {
    "hotkey": {
      "minLength": 1,
      "title": "Hotkey",
      "type": "string"
    },
    "is_chutes_configured": {
      "default": false,
      "title": "Is Chutes Configured",
      "type": "boolean"
    },
    "is_openrouter_configured": {
      "default": false,
      "title": "Is Openrouter Configured",
      "type": "boolean"
    },
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
    "resource_usage": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/ValidatorResourceUsageResponse"
        },
        {
          "type": "null"
        }
      ]
    },
    "running": {
      "default": false,
      "title": "Running",
      "type": "boolean"
    },
    "signature_hex": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Signature Hex"
    },
    "status": {
      "minLength": 1,
      "title": "Status",
      "type": "string"
    }
  },
  "required": [
    "status",
    "hotkey"
  ],
  "title": "ValidatorStatusResponse",
  "type": "object"
}
```

</details>
