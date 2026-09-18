# Validator API reference (generated)

Generated from FastAPI OpenAPI.

## Domains
- [endpoint-assignments](#endpoint-assignments)
  - [POST /validator/endpoint-assignments](#endpoint-post-validator-endpoint-assignments)
  - [POST /validator/endpoint-assignments/{assignment_id}/callback](#endpoint-post-validator-endpoint-assignments-assignment_id-callback)
- [miner-task-batches](#miner-task-batches)
  - [POST /validator/miner-task-batches/{batch_id}/reference-selection](#endpoint-post-validator-miner-task-batches-batch_id-reference-selection)
  - [POST /validator/miner-task-batches/{batch_id}/similarity](#endpoint-post-validator-miner-task-batches-batch_id-similarity)
- [status](#status)
  - [GET /validator/status](#endpoint-get-validator-status)
- [tools](#tools)
  - [POST /v1/tools/execute](#endpoint-post-v1-tools-execute)
- [Misc](#misc)
  - [GET /healthz](#endpoint-get-healthz)
  - [GET /readyz](#endpoint-get-readyz)

## endpoint-assignments

<a id="endpoint-post-validator-endpoint-assignments"></a>
### POST /validator/endpoint-assignments

Execute Endpoint

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Request**
Content-Type: `application/json`
Body: [EndpointExecutionWork](#model-endpointexecutionwork)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `delegation` |  |  | req | [EndpointExecutionWorkEndpointDelegation](#model-endpointexecutionworkendpointdelegation) |
|  | `body_utf8` |  | req | `string` |
|  | `platform_hotkey` |  | req | `string` |
|  | `signature_hex` |  | req | `string` |
| `query` |  |  | req | [EndpointExecutionWorkQuery](#model-endpointexecutionworkquery) |
|  | `fast` |  | opt | `boolean` (default: False) |
|  | `output_schema` |  | opt | [EndpointExecutionWorkJsonObject](#model-endpointexecutionworkjsonobject) (nullable) |
|  | `text` |  | req | `string` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [EndpointProgress](#model-endpointprogress)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `assignment_id` |  |  | req | `string` (format: uuid) |
| `state` |  |  | req | `string` (enum: [queued, active, reporting, saved]) |


### {assignment_id}

#### callback

<a id="endpoint-post-validator-endpoint-assignments-assignment_id-callback"></a>
##### POST /validator/endpoint-assignments/{assignment_id}/callback

Endpoint Callback

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Headers**
| Header | Req | Notes |
| --- | --- | --- |
| `X-Harnyx-Callback-Context` | req | `string` |

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `assignment_id` | path | req | `string` (format: uuid) |

**Request**
Content-Type: `application/json`
Body: [EndpointCallback](#model-endpointcallback)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `assignment_id` |  |  | req | `string` (format: uuid) |
| `expires_at` |  |  | req | `string` (format: date-time) |
| `nonce` |  |  | req | `string` |
| `query_digest` |  |  | req | `string` |
| `response` |  |  | req | [EndpointCallbackResponse](#model-endpointcallbackresponse) |
|  | `citations` |  | opt | array[[EndpointCallbackCitationRef](#model-endpointcallbackcitationref)] (nullable) |
|  |  | `receipt_id` | req | `string` |
|  |  | `result_id` | req | `string` |
|  |  | `slices` | opt | array[[EndpointCallbackCitationSlice](#model-endpointcallbackcitationslice)] |
|  | `note` |  | opt | `string` (nullable) |
|  | `output` |  | opt | [EndpointCallbackJsonValue](#model-endpointcallbackjsonvalue) (nullable) |
|  | `text` |  | opt | `string` (nullable) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [EndpointCallbackAcknowledgement](#model-endpointcallbackacknowledgement)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `durable_terminal_result` |  |  | req | [EndpointDurableTerminalResult](#model-endpointdurableterminalresult) |

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



## miner-task-batches

### {batch_id}

#### reference-selection

<a id="endpoint-post-validator-miner-task-batches-batch_id-reference-selection"></a>
##### POST /validator/miner-task-batches/{batch_id}/reference-selection

Compare a signed endpoint answer against the dataset reference in both orders.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `batch_id` | path | req | `string` (format: uuid) |

**Request**
Content-Type: `application/json`
Body: [ReferenceSelectionRequest](#model-referenceselectionrequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `candidate` |  |  | req | [ReferenceSelectionRequestEndpointAnswer](#model-referenceselectionrequestendpointanswer) |
|  | `assignment_id` |  | req | `string` (format: uuid) |
|  | `callback_body_utf8` |  | req | `string` |
|  | `expected_hotkey` |  | req | `string` |
|  | `receipt_logs` |  | req | array[[ReferenceSelectionRequestEndpointReceipt](#model-referenceselectionrequestendpointreceipt)] |
|  |  | `assignment_id` | req | `string` (format: uuid) |
|  |  | `issued_at` | req | `string` (format: date-time) |
|  |  | `receipt_id` | req | `string` |
|  |  | `results` | req | array[[ReferenceSelectionRequestSearchToolResult](#model-referenceselectionrequestsearchtoolresult)] |
|  |  | `tool` | req | `string` (enum: [search_web, search_ai, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |
|  | `signature_hex` |  | req | `string` |
|  | `signed_callback_path` |  | req | `string` |
| `task` |  |  | req | [ReferenceSelectionRequestMinerTask](#model-referenceselectionrequestminertask) |
|  | `budget_usd` |  | opt | `number` (default: 0.5) |
|  | `query` |  | req | [ReferenceSelectionRequestQuery](#model-referenceselectionrequestquery) |
|  |  | `fast` | opt | `boolean` (default: False) |
|  |  | `output_schema` | opt | [ReferenceSelectionRequestJsonObject](#model-referenceselectionrequestjsonobject) (nullable) |
|  |  | `text` | req | `string` |
|  | `reference_answer` |  | req | [ReferenceSelectionRequestReferenceAnswer](#model-referenceselectionrequestreferenceanswer) |
|  |  | `citations` | opt | array[[ReferenceSelectionRequestAnswerCitation](#model-referenceselectionrequestanswercitation) (nullable)] (nullable) |
|  |  | `note` | opt | `string` (nullable) |
|  |  | `text` | req | `string` |
|  | `task_id` |  | req | `string` (format: uuid) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [ScoreBreakdown](#model-scorebreakdown)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `comparison_score` |  |  | req | `number` |
| `fast_score_evidence` |  |  | opt | [FastScoreEvidence](#model-fastscoreevidence) (nullable) |
|  | `excessive_components` |  | req | array[[FastScoreExcessiveComponent](#model-fastscoreexcessivecomponent)] |
|  |  | `component_id` | req | `string` |
|  | `expected_components` |  | req | array[[FastScoreExpectedComponent](#model-fastscoreexpectedcomponent)] |
|  |  | `component_id` | req | `string` |
|  |  | `is_correct` | req | `boolean` |
|  | `precision` |  | req | `number` |
|  | `recall` |  | req | `number` |
| `reasoning` |  |  | opt | [ScorerReasoning](#model-scorerreasoning) (nullable) |
|  | `reasoning_tokens` |  | opt | `integer` (nullable) |
|  | `text` |  | opt | `string` (nullable) |
| `scoring_version` |  |  | req | `string` |
| `total_score` |  |  | req | `number` |

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
| `classification` |  |  | req | `string` (enum: [duplicate, near_duplicate, notable_change, novel]) |
| `judge_usage` |  |  | opt | [JudgeUsageSummary](#model-judgeusagesummary) (nullable) |
|  | `actual_cost_usd` |  | req | `number` (nullable) |
|  | `call_count` |  | req | `integer` |
|  | `completion_tokens` |  | req | `integer` |
|  | `models` |  | req | array[[JudgeModelUsage](#model-judgemodelusage)] |
|  |  | `actual_cost_evidence` | opt | `string` (nullable) |
|  |  | `actual_cost_provider` | opt | `string` (nullable) |
|  |  | `actual_cost_source` | req | `string` (enum: [provider_actual, unavailable]) |
|  |  | `actual_cost_usd` | req | `number` (nullable) |
|  |  | `call_count` | req | `integer` |
|  |  | `completion_tokens` | req | `integer` |
|  |  | `model` | req | `string` |
|  |  | `prompt_tokens` | req | `integer` |
|  |  | `provider` | req | `string` |
|  |  | `reasoning_tokens` | req | `integer` (nullable) |
|  |  | `total_tokens` | req | `integer` |
|  | `prompt_tokens` |  | req | `integer` |
|  | `reasoning_tokens` |  | req | `integer` |
|  | `total_tokens` |  | req | `integer` |
| `model` |  |  | req | `string` |
| `provider` |  |  | req | `string` |
| `reasoning` |  |  | opt | `string` (nullable) |
| `reasoning_tokens` |  |  | opt | `integer` (nullable) |

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

`502` Bad Gateway
Content-Type: `application/json`
Body: [SimilarityJudgeFailureResponseModel](#model-similarityjudgefailureresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | req | `string` |
| `error_code` |  |  | req | `string` |
| `judge_usage` |  |  | opt | [JudgeUsageSummary](#model-judgeusagesummary) (nullable) |
|  | `actual_cost_usd` |  | req | `number` (nullable) |
|  | `call_count` |  | req | `integer` |
|  | `completion_tokens` |  | req | `integer` |
|  | `models` |  | req | array[[JudgeModelUsage](#model-judgemodelusage)] |
|  |  | `actual_cost_evidence` | opt | `string` (nullable) |
|  |  | `actual_cost_provider` | opt | `string` (nullable) |
|  |  | `actual_cost_source` | req | `string` (enum: [provider_actual, unavailable]) |
|  |  | `actual_cost_usd` | req | `number` (nullable) |
|  |  | `call_count` | req | `integer` |
|  |  | `completion_tokens` | req | `integer` |
|  |  | `model` | req | `string` |
|  |  | `prompt_tokens` | req | `integer` |
|  |  | `provider` | req | `string` |
|  |  | `reasoning_tokens` | req | `integer` (nullable) |
|  |  | `total_tokens` | req | `integer` |
|  | `prompt_tokens` |  | req | `integer` |
|  | `reasoning_tokens` |  | req | `integer` |
|  | `total_tokens` |  | req | `integer` |
| `retryable` |  |  | req | `boolean` |



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
| `rating_worker_ready` |  |  | opt | `boolean` (default: False) |
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
| `tool` |  |  | req | `string` (enum: [search_web, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |

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



## Misc

### healthz

<a id="endpoint-get-healthz"></a>
#### GET /healthz

Validator health check.

**Auth**: None.

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [ValidatorHealthResponse](#model-validatorhealthresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `status` |  |  | req | `string` |


### readyz

<a id="endpoint-get-readyz"></a>
#### GET /readyz

Validator readiness check.

**Auth**: None.

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [ValidatorReadinessSuccessResponse](#model-validatorreadinesssuccessresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `status` |  |  | req | `string` |

`503` Validator is not ready.
Content-Type: `application/json`
Body: [ValidatorReadinessFailureResponse](#model-validatorreadinessfailureresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | `string` (nullable) |
| `status` |  |  | req | `string` (enum: [waiting_for_platform_registration, waiting_for_auth_warmup, registration_failed, auth_unavailable]) |



## Models

<a id="model-endpointcallback"></a>
### Model: EndpointCallback

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `assignment_id` |  |  | req | `string` (format: uuid) |
| `expires_at` |  |  | req | `string` (format: date-time) |
| `nonce` |  |  | req | `string` |
| `query_digest` |  |  | req | `string` |
| `response` |  |  | req | [EndpointCallbackResponse](#model-endpointcallbackresponse) |
|  | `citations` |  | opt | array[[EndpointCallbackCitationRef](#model-endpointcallbackcitationref)] (nullable) |
|  |  | `receipt_id` | req | `string` |
|  |  | `result_id` | req | `string` |
|  |  | `slices` | opt | array[[EndpointCallbackCitationSlice](#model-endpointcallbackcitationslice)] |
|  | `note` |  | opt | `string` (nullable) |
|  | `output` |  | opt | [EndpointCallbackJsonValue](#model-endpointcallbackjsonvalue) (nullable) |
|  | `text` |  | opt | `string` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "assignment_id": {
      "format": "uuid",
      "title": "Assignment Id",
      "type": "string"
    },
    "expires_at": {
      "format": "date-time",
      "title": "Expires At",
      "type": "string"
    },
    "nonce": {
      "maxLength": 128,
      "minLength": 32,
      "title": "Nonce",
      "type": "string"
    },
    "query_digest": {
      "pattern": "^[0-9a-f]{64}$",
      "title": "Query Digest",
      "type": "string"
    },
    "response": {
      "$ref": "#/components/schemas/EndpointCallbackResponse"
    }
  },
  "required": [
    "assignment_id",
    "query_digest",
    "nonce",
    "expires_at",
    "response"
  ],
  "title": "EndpointCallback",
  "type": "object"
}
```

</details>

<a id="model-endpointcallbackacknowledgement"></a>
### Model: EndpointCallbackAcknowledgement

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `durable_terminal_result` |  |  | req | [EndpointDurableTerminalResult](#model-endpointdurableterminalresult) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "durable_terminal_result": {
      "$ref": "#/components/schemas/EndpointDurableTerminalResult"
    }
  },
  "required": [
    "durable_terminal_result"
  ],
  "title": "EndpointCallbackAcknowledgement",
  "type": "object"
}
```

</details>

<a id="model-endpointcallbackcitationref"></a>
### Model: EndpointCallbackCitationRef

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `receipt_id` |  |  | req | `string` |
| `result_id` |  |  | req | `string` |
| `slices` |  |  | opt | array[[EndpointCallbackCitationSlice](#model-endpointcallbackcitationslice)] |
|  | `end` |  | req | `integer` |
|  | `start` |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "receipt_id": {
      "minLength": 1,
      "title": "Receipt Id",
      "type": "string"
    },
    "result_id": {
      "minLength": 1,
      "title": "Result Id",
      "type": "string"
    },
    "slices": {
      "items": {
        "$ref": "#/components/schemas/EndpointCallbackCitationSlice"
      },
      "title": "Slices",
      "type": "array"
    }
  },
  "required": [
    "receipt_id",
    "result_id"
  ],
  "title": "CitationRef",
  "type": "object"
}
```

</details>

<a id="model-endpointcallbackcitationslice"></a>
### Model: EndpointCallbackCitationSlice

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `end` |  |  | req | `integer` |
| `start` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "end": {
      "exclusiveMinimum": 0,
      "title": "End",
      "type": "integer"
    },
    "start": {
      "minimum": 0,
      "title": "Start",
      "type": "integer"
    }
  },
  "required": [
    "start",
    "end"
  ],
  "title": "CitationSlice",
  "type": "object"
}
```

</details>

<a id="model-endpointcallbackjsonvalue"></a>
### Model: EndpointCallbackJsonValue

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "integer"
    },
    {
      "type": "number"
    },
    {
      "type": "boolean"
    },
    {
      "items": {
        "$ref": "#/components/schemas/EndpointCallbackJsonValue"
      },
      "type": "array"
    },
    {
      "additionalProperties": {
        "$ref": "#/components/schemas/EndpointCallbackJsonValue"
      },
      "type": "object"
    },
    {
      "type": "null"
    }
  ]
}
```

</details>

<a id="model-endpointcallbackresponse"></a>
### Model: EndpointCallbackResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `citations` |  |  | opt | array[[EndpointCallbackCitationRef](#model-endpointcallbackcitationref)] (nullable) |
|  | `receipt_id` |  | req | `string` |
|  | `result_id` |  | req | `string` |
|  | `slices` |  | opt | array[[EndpointCallbackCitationSlice](#model-endpointcallbackcitationslice)] |
|  |  | `end` | req | `integer` |
|  |  | `start` | req | `integer` |
| `note` |  |  | opt | `string` (nullable) |
| `output` |  |  | opt | [EndpointCallbackJsonValue](#model-endpointcallbackjsonvalue) (nullable) |
| `text` |  |  | opt | `string` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "oneOf": [
    {
      "properties": {
        "output": {
          "type": "null"
        },
        "text": {
          "type": "string"
        }
      },
      "required": [
        "text"
      ]
    },
    {
      "properties": {
        "output": {},
        "text": {
          "allOf": [
            {
              "type": "string"
            },
            {
              "type": "number"
            }
          ]
        }
      },
      "required": [
        "output"
      ]
    }
  ],
  "properties": {
    "citations": {
      "anyOf": [
        {
          "items": {
            "$ref": "#/components/schemas/EndpointCallbackCitationRef"
          },
          "maxItems": 200,
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Citations"
    },
    "note": {
      "anyOf": [
        {
          "maxLength": 80000,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Optional public supplementary content that may explain, qualify, support, or correct the required answer. It cannot replace or repair a missing or invalid answer. Factual claims use the same citations array.",
      "title": "Note"
    },
    "output": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/EndpointCallbackJsonValue"
        },
        {
          "type": "null"
        }
      ],
      "default": null
    },
    "text": {
      "anyOf": [
        {
          "maxLength": 80000,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Text"
    }
  },
  "title": "Response",
  "type": "object"
}
```

</details>

<a id="model-endpointdurableterminalresult"></a>
### Model: EndpointDurableTerminalResult

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "enum": [
    "persisted",
    "closed"
  ],
  "title": "EndpointDurableTerminalResult",
  "type": "string"
}
```

</details>

<a id="model-endpointexecutionwork"></a>
### Model: EndpointExecutionWork

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `delegation` |  |  | req | [EndpointExecutionWorkEndpointDelegation](#model-endpointexecutionworkendpointdelegation) |
|  | `body_utf8` |  | req | `string` |
|  | `platform_hotkey` |  | req | `string` |
|  | `signature_hex` |  | req | `string` |
| `query` |  |  | req | [EndpointExecutionWorkQuery](#model-endpointexecutionworkquery) |
|  | `fast` |  | opt | `boolean` (default: False) |
|  | `output_schema` |  | opt | [EndpointExecutionWorkJsonObject](#model-endpointexecutionworkjsonobject) (nullable) |
|  | `text` |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "delegation": {
      "$ref": "#/components/schemas/EndpointExecutionWorkEndpointDelegation"
    },
    "query": {
      "$ref": "#/components/schemas/EndpointExecutionWorkQuery"
    }
  },
  "required": [
    "query",
    "delegation"
  ],
  "title": "EndpointExecutionWork",
  "type": "object"
}
```

</details>

<a id="model-endpointexecutionworkendpointdelegation"></a>
### Model: EndpointExecutionWorkEndpointDelegation

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `body_utf8` |  |  | req | `string` |
| `platform_hotkey` |  |  | req | `string` |
| `signature_hex` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "description": "Original Platform-signed assignment authority, retained across validator retries.",
  "properties": {
    "body_utf8": {
      "maxLength": 12000,
      "minLength": 1,
      "title": "Body Utf8",
      "type": "string"
    },
    "platform_hotkey": {
      "minLength": 1,
      "title": "Platform Hotkey",
      "type": "string"
    },
    "signature_hex": {
      "maxLength": 128,
      "minLength": 128,
      "pattern": "^[0-9a-f]+$",
      "title": "Signature Hex",
      "type": "string"
    }
  },
  "required": [
    "platform_hotkey",
    "body_utf8",
    "signature_hex"
  ],
  "title": "EndpointDelegation",
  "type": "object"
}
```

</details>

<a id="model-endpointexecutionworkjsonobject"></a>
### Model: EndpointExecutionWorkJsonObject

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": {
    "$ref": "#/components/schemas/EndpointExecutionWorkJsonValue"
  },
  "type": "object"
}
```

</details>

<a id="model-endpointexecutionworkjsonvalue"></a>
### Model: EndpointExecutionWorkJsonValue

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "integer"
    },
    {
      "type": "number"
    },
    {
      "type": "boolean"
    },
    {
      "items": {
        "$ref": "#/components/schemas/EndpointExecutionWorkJsonValue"
      },
      "type": "array"
    },
    {
      "additionalProperties": {
        "$ref": "#/components/schemas/EndpointExecutionWorkJsonValue"
      },
      "type": "object"
    },
    {
      "type": "null"
    }
  ]
}
```

</details>

<a id="model-endpointexecutionworkquery"></a>
### Model: EndpointExecutionWorkQuery

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `fast` |  |  | opt | `boolean` (default: False) |
| `output_schema` |  |  | opt | [EndpointExecutionWorkJsonObject](#model-endpointexecutionworkjsonobject) (nullable) |
| `text` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "fast": {
      "default": false,
      "description": "Whether the query uses correctness-only fast-mode scoring.",
      "title": "Fast",
      "type": "boolean"
    },
    "output_schema": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/EndpointExecutionWorkJsonObject"
        },
        {
          "type": "null"
        }
      ],
      "default": null
    },
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

<a id="model-endpointprogress"></a>
### Model: EndpointProgress

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `assignment_id` |  |  | req | `string` (format: uuid) |
| `state` |  |  | req | `string` (enum: [queued, active, reporting, saved]) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "assignment_id": {
      "format": "uuid",
      "title": "Assignment Id",
      "type": "string"
    },
    "state": {
      "enum": [
        "queued",
        "active",
        "reporting",
        "saved"
      ],
      "title": "State",
      "type": "string"
    }
  },
  "required": [
    "assignment_id",
    "state"
  ],
  "title": "EndpointProgress",
  "type": "object"
}
```

</details>

<a id="model-fastscoreevidence"></a>
### Model: FastScoreEvidence

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `excessive_components` |  |  | req | array[[FastScoreExcessiveComponent](#model-fastscoreexcessivecomponent)] |
|  | `component_id` |  | req | `string` |
| `expected_components` |  |  | req | array[[FastScoreExpectedComponent](#model-fastscoreexpectedcomponent)] |
|  | `component_id` |  | req | `string` |
|  | `is_correct` |  | req | `boolean` |
| `precision` |  |  | req | `number` |
| `recall` |  |  | req | `number` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "description": "Persisted component judgment and deterministic metrics for one fast score.",
  "properties": {
    "excessive_components": {
      "items": {
        "$ref": "#/components/schemas/FastScoreExcessiveComponent"
      },
      "title": "Excessive Components",
      "type": "array"
    },
    "expected_components": {
      "items": {
        "$ref": "#/components/schemas/FastScoreExpectedComponent"
      },
      "minItems": 1,
      "title": "Expected Components",
      "type": "array"
    },
    "precision": {
      "maximum": 1.0,
      "minimum": 0.0,
      "title": "Precision",
      "type": "number"
    },
    "recall": {
      "maximum": 1.0,
      "minimum": 0.0,
      "title": "Recall",
      "type": "number"
    }
  },
  "required": [
    "expected_components",
    "excessive_components",
    "precision",
    "recall"
  ],
  "title": "FastScoreEvidence",
  "type": "object"
}
```

</details>

<a id="model-fastscoreexcessivecomponent"></a>
### Model: FastScoreExcessiveComponent

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `component_id` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "description": "One excessive answer component retained as fast-score evidence.",
  "properties": {
    "component_id": {
      "minLength": 1,
      "title": "Component Id",
      "type": "string"
    }
  },
  "required": [
    "component_id"
  ],
  "title": "FastScoreExcessiveComponent",
  "type": "object"
}
```

</details>

<a id="model-fastscoreexpectedcomponent"></a>
### Model: FastScoreExpectedComponent

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `component_id` |  |  | req | `string` |
| `is_correct` |  |  | req | `boolean` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "description": "One required answer component retained as fast-score evidence.",
  "properties": {
    "component_id": {
      "minLength": 1,
      "title": "Component Id",
      "type": "string"
    },
    "is_correct": {
      "title": "Is Correct",
      "type": "boolean"
    }
  },
  "required": [
    "component_id",
    "is_correct"
  ],
  "title": "FastScoreExpectedComponent",
  "type": "object"
}
```

</details>

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

<a id="model-judgemodelusage"></a>
### Model: JudgeModelUsage

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `actual_cost_evidence` |  |  | opt | `string` (nullable) |
| `actual_cost_provider` |  |  | opt | `string` (nullable) |
| `actual_cost_source` |  |  | req | `string` (enum: [provider_actual, unavailable]) |
| `actual_cost_usd` |  |  | req | `number` (nullable) |
| `call_count` |  |  | req | `integer` |
| `completion_tokens` |  |  | req | `integer` |
| `model` |  |  | req | `string` |
| `prompt_tokens` |  |  | req | `integer` |
| `provider` |  |  | req | `string` |
| `reasoning_tokens` |  |  | req | `integer` (nullable) |
| `total_tokens` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "actual_cost_evidence": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Actual Cost Evidence"
    },
    "actual_cost_provider": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Actual Cost Provider"
    },
    "actual_cost_source": {
      "enum": [
        "provider_actual",
        "unavailable"
      ],
      "title": "Actual Cost Source",
      "type": "string"
    },
    "actual_cost_usd": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Actual Cost Usd"
    },
    "call_count": {
      "title": "Call Count",
      "type": "integer"
    },
    "completion_tokens": {
      "title": "Completion Tokens",
      "type": "integer"
    },
    "model": {
      "title": "Model",
      "type": "string"
    },
    "prompt_tokens": {
      "title": "Prompt Tokens",
      "type": "integer"
    },
    "provider": {
      "title": "Provider",
      "type": "string"
    },
    "reasoning_tokens": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Reasoning Tokens"
    },
    "total_tokens": {
      "title": "Total Tokens",
      "type": "integer"
    }
  },
  "required": [
    "provider",
    "model",
    "call_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "actual_cost_usd",
    "actual_cost_source"
  ],
  "title": "JudgeModelUsage",
  "type": "object"
}
```

</details>

<a id="model-judgeusagesummary"></a>
### Model: JudgeUsageSummary

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `actual_cost_usd` |  |  | req | `number` (nullable) |
| `call_count` |  |  | req | `integer` |
| `completion_tokens` |  |  | req | `integer` |
| `models` |  |  | req | array[[JudgeModelUsage](#model-judgemodelusage)] |
|  | `actual_cost_evidence` |  | opt | `string` (nullable) |
|  | `actual_cost_provider` |  | opt | `string` (nullable) |
|  | `actual_cost_source` |  | req | `string` (enum: [provider_actual, unavailable]) |
|  | `actual_cost_usd` |  | req | `number` (nullable) |
|  | `call_count` |  | req | `integer` |
|  | `completion_tokens` |  | req | `integer` |
|  | `model` |  | req | `string` |
|  | `prompt_tokens` |  | req | `integer` |
|  | `provider` |  | req | `string` |
|  | `reasoning_tokens` |  | req | `integer` (nullable) |
|  | `total_tokens` |  | req | `integer` |
| `prompt_tokens` |  |  | req | `integer` |
| `reasoning_tokens` |  |  | req | `integer` |
| `total_tokens` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "actual_cost_usd": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Actual Cost Usd"
    },
    "call_count": {
      "title": "Call Count",
      "type": "integer"
    },
    "completion_tokens": {
      "title": "Completion Tokens",
      "type": "integer"
    },
    "models": {
      "items": {
        "$ref": "#/components/schemas/JudgeModelUsage"
      },
      "title": "Models",
      "type": "array"
    },
    "prompt_tokens": {
      "title": "Prompt Tokens",
      "type": "integer"
    },
    "reasoning_tokens": {
      "title": "Reasoning Tokens",
      "type": "integer"
    },
    "total_tokens": {
      "title": "Total Tokens",
      "type": "integer"
    }
  },
  "required": [
    "call_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "actual_cost_usd",
    "models"
  ],
  "title": "JudgeUsageSummary",
  "type": "object"
}
```

</details>

<a id="model-referenceselectionrequest"></a>
### Model: ReferenceSelectionRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `candidate` |  |  | req | [ReferenceSelectionRequestEndpointAnswer](#model-referenceselectionrequestendpointanswer) |
|  | `assignment_id` |  | req | `string` (format: uuid) |
|  | `callback_body_utf8` |  | req | `string` |
|  | `expected_hotkey` |  | req | `string` |
|  | `receipt_logs` |  | req | array[[ReferenceSelectionRequestEndpointReceipt](#model-referenceselectionrequestendpointreceipt)] |
|  |  | `assignment_id` | req | `string` (format: uuid) |
|  |  | `issued_at` | req | `string` (format: date-time) |
|  |  | `receipt_id` | req | `string` |
|  |  | `results` | req | array[[ReferenceSelectionRequestSearchToolResult](#model-referenceselectionrequestsearchtoolresult)] |
|  |  | `tool` | req | `string` (enum: [search_web, search_ai, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |
|  | `signature_hex` |  | req | `string` |
|  | `signed_callback_path` |  | req | `string` |
| `task` |  |  | req | [ReferenceSelectionRequestMinerTask](#model-referenceselectionrequestminertask) |
|  | `budget_usd` |  | opt | `number` (default: 0.5) |
|  | `query` |  | req | [ReferenceSelectionRequestQuery](#model-referenceselectionrequestquery) |
|  |  | `fast` | opt | `boolean` (default: False) |
|  |  | `output_schema` | opt | [ReferenceSelectionRequestJsonObject](#model-referenceselectionrequestjsonobject) (nullable) |
|  |  | `text` | req | `string` |
|  | `reference_answer` |  | req | [ReferenceSelectionRequestReferenceAnswer](#model-referenceselectionrequestreferenceanswer) |
|  |  | `citations` | opt | array[[ReferenceSelectionRequestAnswerCitation](#model-referenceselectionrequestanswercitation) (nullable)] (nullable) |
|  |  | `note` | opt | `string` (nullable) |
|  |  | `text` | req | `string` |
|  | `task_id` |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "batch_id": {
      "format": "uuid",
      "title": "Batch Id",
      "type": "string"
    },
    "candidate": {
      "$ref": "#/components/schemas/ReferenceSelectionRequestEndpointAnswer"
    },
    "task": {
      "$ref": "#/components/schemas/ReferenceSelectionRequestMinerTask"
    }
  },
  "required": [
    "batch_id",
    "task",
    "candidate"
  ],
  "title": "ReferenceSelectionRequest",
  "type": "object"
}
```

</details>

<a id="model-referenceselectionrequestanswercitation"></a>
### Model: ReferenceSelectionRequestAnswerCitation

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `excerpts` |  |  | opt | array[[ReferenceSelectionRequestCitationExcerpt](#model-referenceselectionrequestcitationexcerpt)] (default: []) |
|  | `end` |  | opt | `integer` (nullable) |
|  | `start` |  | opt | `integer` (nullable) |
|  | `text` |  | req | `string` |
| `title` |  |  | opt | `string` (nullable) |
| `url` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "excerpts": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/ReferenceSelectionRequestCitationExcerpt"
      },
      "title": "Excerpts",
      "type": "array"
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
      "default": null,
      "title": "Title"
    },
    "url": {
      "minLength": 1,
      "title": "Url",
      "type": "string"
    }
  },
  "required": [
    "url"
  ],
  "title": "AnswerCitation",
  "type": "object"
}
```

</details>

<a id="model-referenceselectionrequestcitationexcerpt"></a>
### Model: ReferenceSelectionRequestCitationExcerpt

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `end` |  |  | opt | `integer` (nullable) |
| `start` |  |  | opt | `integer` (nullable) |
| `text` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "description": "Exact source passage; unknown positions are retained only for legacy evidence.",
  "properties": {
    "end": {
      "anyOf": [
        {
          "exclusiveMinimum": 0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "End"
    },
    "start": {
      "anyOf": [
        {
          "minimum": 0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Start"
    },
    "text": {
      "title": "Text",
      "type": "string"
    }
  },
  "required": [
    "text"
  ],
  "title": "CitationExcerpt",
  "type": "object"
}
```

</details>

<a id="model-referenceselectionrequestendpointanswer"></a>
### Model: ReferenceSelectionRequestEndpointAnswer

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `assignment_id` |  |  | req | `string` (format: uuid) |
| `callback_body_utf8` |  |  | req | `string` |
| `expected_hotkey` |  |  | req | `string` |
| `receipt_logs` |  |  | req | array[[ReferenceSelectionRequestEndpointReceipt](#model-referenceselectionrequestendpointreceipt)] |
|  | `assignment_id` |  | req | `string` (format: uuid) |
|  | `issued_at` |  | req | `string` (format: date-time) |
|  | `receipt_id` |  | req | `string` |
|  | `results` |  | req | array[[ReferenceSelectionRequestSearchToolResult](#model-referenceselectionrequestsearchtoolresult)] |
|  |  | `index` | req | `integer` |
|  |  | `note` | opt | `string` (nullable) |
|  |  | `raw` | opt | [ReferenceSelectionRequestJsonValue](#model-referenceselectionrequestjsonvalue) (nullable) |
|  |  | `result_id` | req | `string` |
|  |  | `title` | opt | `string` (nullable) |
|  |  | `url` | opt | `string` (default: ) |
|  | `tool` |  | req | `string` (enum: [search_web, search_ai, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |
| `signature_hex` |  |  | req | `string` |
| `signed_callback_path` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "assignment_id": {
      "format": "uuid",
      "title": "Assignment Id",
      "type": "string"
    },
    "callback_body_utf8": {
      "title": "Callback Body Utf8",
      "type": "string"
    },
    "expected_hotkey": {
      "minLength": 1,
      "title": "Expected Hotkey",
      "type": "string"
    },
    "receipt_logs": {
      "items": {
        "$ref": "#/components/schemas/ReferenceSelectionRequestEndpointReceipt"
      },
      "title": "Receipt Logs",
      "type": "array"
    },
    "signature_hex": {
      "minLength": 1,
      "title": "Signature Hex",
      "type": "string"
    },
    "signed_callback_path": {
      "pattern": "^/",
      "title": "Signed Callback Path",
      "type": "string"
    }
  },
  "required": [
    "assignment_id",
    "expected_hotkey",
    "callback_body_utf8",
    "signature_hex",
    "signed_callback_path",
    "receipt_logs"
  ],
  "title": "EndpointAnswer",
  "type": "object"
}
```

</details>

<a id="model-referenceselectionrequestendpointreceipt"></a>
### Model: ReferenceSelectionRequestEndpointReceipt

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `assignment_id` |  |  | req | `string` (format: uuid) |
| `issued_at` |  |  | req | `string` (format: date-time) |
| `receipt_id` |  |  | req | `string` |
| `results` |  |  | req | array[[ReferenceSelectionRequestSearchToolResult](#model-referenceselectionrequestsearchtoolresult)] |
|  | `index` |  | req | `integer` |
|  | `note` |  | opt | `string` (nullable) |
|  | `raw` |  | opt | [ReferenceSelectionRequestJsonValue](#model-referenceselectionrequestjsonvalue) (nullable) |
|  | `result_id` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (default: ) |
| `tool` |  |  | req | `string` (enum: [search_web, search_ai, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "assignment_id": {
      "format": "uuid",
      "title": "Assignment Id",
      "type": "string"
    },
    "issued_at": {
      "format": "date-time",
      "title": "Issued At",
      "type": "string"
    },
    "receipt_id": {
      "title": "Receipt Id",
      "type": "string"
    },
    "results": {
      "items": {
        "$ref": "#/components/schemas/ReferenceSelectionRequestSearchToolResult"
      },
      "title": "Results",
      "type": "array"
    },
    "tool": {
      "enum": [
        "search_web",
        "search_ai",
        "fetch_page",
        "embed_text",
        "llm_chat",
        "test_tool",
        "tooling_info"
      ],
      "title": "Tool",
      "type": "string"
    }
  },
  "required": [
    "receipt_id",
    "assignment_id",
    "tool",
    "issued_at",
    "results"
  ],
  "title": "EndpointReceipt",
  "type": "object"
}
```

</details>

<a id="model-referenceselectionrequestjsonobject"></a>
### Model: ReferenceSelectionRequestJsonObject

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": {
    "$ref": "#/components/schemas/ReferenceSelectionRequestJsonValue"
  },
  "type": "object"
}
```

</details>

<a id="model-referenceselectionrequestjsonvalue"></a>
### Model: ReferenceSelectionRequestJsonValue

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "anyOf": [
    {
      "type": "string"
    },
    {
      "type": "integer"
    },
    {
      "type": "number"
    },
    {
      "type": "boolean"
    },
    {
      "items": {
        "$ref": "#/components/schemas/ReferenceSelectionRequestJsonValue"
      },
      "type": "array"
    },
    {
      "additionalProperties": {
        "$ref": "#/components/schemas/ReferenceSelectionRequestJsonValue"
      },
      "type": "object"
    },
    {
      "type": "null"
    }
  ]
}
```

</details>

<a id="model-referenceselectionrequestminertask"></a>
### Model: ReferenceSelectionRequestMinerTask

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `budget_usd` |  |  | opt | `number` (default: 0.5) |
| `query` |  |  | req | [ReferenceSelectionRequestQuery](#model-referenceselectionrequestquery) |
|  | `fast` |  | opt | `boolean` (default: False) |
|  | `output_schema` |  | opt | [ReferenceSelectionRequestJsonObject](#model-referenceselectionrequestjsonobject) (nullable) |
|  | `text` |  | req | `string` |
| `reference_answer` |  |  | req | [ReferenceSelectionRequestReferenceAnswer](#model-referenceselectionrequestreferenceanswer) |
|  | `citations` |  | opt | array[[ReferenceSelectionRequestAnswerCitation](#model-referenceselectionrequestanswercitation) (nullable)] (nullable) |
|  |  | `excerpts` | opt | array[[ReferenceSelectionRequestCitationExcerpt](#model-referenceselectionrequestcitationexcerpt)] (default: []) |
|  |  | `title` | opt | `string` (nullable) |
|  |  | `url` | req | `string` |
|  | `note` |  | opt | `string` (nullable) |
|  | `text` |  | req | `string` |
| `task_id` |  |  | req | `string` (format: uuid) |

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
      "$ref": "#/components/schemas/ReferenceSelectionRequestQuery"
    },
    "reference_answer": {
      "$ref": "#/components/schemas/ReferenceSelectionRequestReferenceAnswer"
    },
    "task_id": {
      "format": "uuid",
      "title": "Task Id",
      "type": "string"
    }
  },
  "required": [
    "task_id",
    "query",
    "reference_answer"
  ],
  "title": "MinerTask",
  "type": "object"
}
```

</details>

<a id="model-referenceselectionrequestquery"></a>
### Model: ReferenceSelectionRequestQuery

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `fast` |  |  | opt | `boolean` (default: False) |
| `output_schema` |  |  | opt | [ReferenceSelectionRequestJsonObject](#model-referenceselectionrequestjsonobject) (nullable) |
| `text` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "fast": {
      "default": false,
      "description": "Whether the query uses correctness-only fast-mode scoring.",
      "title": "Fast",
      "type": "boolean"
    },
    "output_schema": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/ReferenceSelectionRequestJsonObject"
        },
        {
          "type": "null"
        }
      ],
      "default": null
    },
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

<a id="model-referenceselectionrequestreferenceanswer"></a>
### Model: ReferenceSelectionRequestReferenceAnswer

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `citations` |  |  | opt | array[[ReferenceSelectionRequestAnswerCitation](#model-referenceselectionrequestanswercitation) (nullable)] (nullable) |
|  | `excerpts` |  | opt | array[[ReferenceSelectionRequestCitationExcerpt](#model-referenceselectionrequestcitationexcerpt)] (default: []) |
|  |  | `end` | opt | `integer` (nullable) |
|  |  | `start` | opt | `integer` (nullable) |
|  |  | `text` | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | req | `string` |
| `note` |  |  | opt | `string` (nullable) |
| `text` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "citations": {
      "anyOf": [
        {
          "items": {
            "anyOf": [
              {
                "$ref": "#/components/schemas/ReferenceSelectionRequestAnswerCitation"
              },
              {
                "type": "null"
              }
            ]
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Hydrated submitted citation positions in order. Miners submit only non-null CitationRef entries. An AnswerCitation means that the submitted position resolved to authoritative public evidence; null means that the submitted position could not be resolved or hydrated. A null provides no factual support, and submitted positions are never deleted, renumbered, or remapped.",
      "title": "Citations"
    },
    "note": {
      "anyOf": [
        {
          "maxLength": 80000,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Optional public supplementary content that may explain, qualify, support, or correct the required answer. It cannot replace or repair a missing or invalid answer. Factual claims use the same citations array.",
      "title": "Note"
    },
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

<a id="model-referenceselectionrequestsearchtoolresult"></a>
### Model: ReferenceSelectionRequestSearchToolResult

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `index` |  |  | req | `integer` |
| `note` |  |  | opt | `string` (nullable) |
| `raw` |  |  | opt | [ReferenceSelectionRequestJsonValue](#model-referenceselectionrequestjsonvalue) (nullable) |
| `result_id` |  |  | req | `string` |
| `title` |  |  | opt | `string` (nullable) |
| `url` |  |  | opt | `string` (default: ) |

<details>
<summary>JSON schema</summary>

```json
{
  "description": "Normalized search result that miners may cite.",
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
      "default": null,
      "title": "Note"
    },
    "raw": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/ReferenceSelectionRequestJsonValue"
        },
        {
          "type": "null"
        }
      ],
      "default": null
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
      "default": null,
      "title": "Title"
    },
    "url": {
      "default": "",
      "title": "Url",
      "type": "string"
    }
  },
  "required": [
    "index",
    "result_id"
  ],
  "title": "SearchToolResult",
  "type": "object"
}
```

</details>

<a id="model-scorebreakdown"></a>
### Model: ScoreBreakdown

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `comparison_score` |  |  | req | `number` |
| `fast_score_evidence` |  |  | opt | [FastScoreEvidence](#model-fastscoreevidence) (nullable) |
|  | `excessive_components` |  | req | array[[FastScoreExcessiveComponent](#model-fastscoreexcessivecomponent)] |
|  |  | `component_id` | req | `string` |
|  | `expected_components` |  | req | array[[FastScoreExpectedComponent](#model-fastscoreexpectedcomponent)] |
|  |  | `component_id` | req | `string` |
|  |  | `is_correct` | req | `boolean` |
|  | `precision` |  | req | `number` |
|  | `recall` |  | req | `number` |
| `reasoning` |  |  | opt | [ScorerReasoning](#model-scorerreasoning) (nullable) |
|  | `reasoning_tokens` |  | opt | `integer` (nullable) |
|  | `text` |  | opt | `string` (nullable) |
| `scoring_version` |  |  | req | `string` |
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
    "fast_score_evidence": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/FastScoreEvidence"
        },
        {
          "type": "null"
        }
      ]
    },
    "reasoning": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/ScorerReasoning"
        },
        {
          "type": "null"
        }
      ]
    },
    "scoring_version": {
      "minLength": 1,
      "title": "Scoring Version",
      "type": "string"
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
    "total_score",
    "scoring_version"
  ],
  "title": "ScoreBreakdown",
  "type": "object"
}
```

</details>

<a id="model-scorerreasoning"></a>
### Model: ScorerReasoning

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `reasoning_tokens` |  |  | opt | `integer` (nullable) |
| `text` |  |  | opt | `string` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
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
    "text": {
      "anyOf": [
        {
          "minLength": 1,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Text"
    }
  },
  "title": "ScorerReasoning",
  "type": "object"
}
```

</details>

<a id="model-similarityjudgefailureresponsemodel"></a>
### Model: SimilarityJudgeFailureResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | req | `string` |
| `error_code` |  |  | req | `string` |
| `judge_usage` |  |  | opt | [JudgeUsageSummary](#model-judgeusagesummary) (nullable) |
|  | `actual_cost_usd` |  | req | `number` (nullable) |
|  | `call_count` |  | req | `integer` |
|  | `completion_tokens` |  | req | `integer` |
|  | `models` |  | req | array[[JudgeModelUsage](#model-judgemodelusage)] |
|  |  | `actual_cost_evidence` | opt | `string` (nullable) |
|  |  | `actual_cost_provider` | opt | `string` (nullable) |
|  |  | `actual_cost_source` | req | `string` (enum: [provider_actual, unavailable]) |
|  |  | `actual_cost_usd` | req | `number` (nullable) |
|  |  | `call_count` | req | `integer` |
|  |  | `completion_tokens` | req | `integer` |
|  |  | `model` | req | `string` |
|  |  | `prompt_tokens` | req | `integer` |
|  |  | `provider` | req | `string` |
|  |  | `reasoning_tokens` | req | `integer` (nullable) |
|  |  | `total_tokens` | req | `integer` |
|  | `prompt_tokens` |  | req | `integer` |
|  | `reasoning_tokens` |  | req | `integer` |
|  | `total_tokens` |  | req | `integer` |
| `retryable` |  |  | req | `boolean` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "detail": {
      "minLength": 1,
      "title": "Detail",
      "type": "string"
    },
    "error_code": {
      "const": "similarity_judge_failed",
      "title": "Error Code",
      "type": "string"
    },
    "judge_usage": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/JudgeUsageSummary"
        },
        {
          "type": "null"
        }
      ]
    },
    "retryable": {
      "title": "Retryable",
      "type": "boolean"
    }
  },
  "required": [
    "error_code",
    "retryable",
    "detail"
  ],
  "title": "SimilarityJudgeFailureResponseModel",
  "type": "object"
}
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
| `classification` |  |  | req | `string` (enum: [duplicate, near_duplicate, notable_change, novel]) |
| `judge_usage` |  |  | opt | [JudgeUsageSummary](#model-judgeusagesummary) (nullable) |
|  | `actual_cost_usd` |  | req | `number` (nullable) |
|  | `call_count` |  | req | `integer` |
|  | `completion_tokens` |  | req | `integer` |
|  | `models` |  | req | array[[JudgeModelUsage](#model-judgemodelusage)] |
|  |  | `actual_cost_evidence` | opt | `string` (nullable) |
|  |  | `actual_cost_provider` | opt | `string` (nullable) |
|  |  | `actual_cost_source` | req | `string` (enum: [provider_actual, unavailable]) |
|  |  | `actual_cost_usd` | req | `number` (nullable) |
|  |  | `call_count` | req | `integer` |
|  |  | `completion_tokens` | req | `integer` |
|  |  | `model` | req | `string` |
|  |  | `prompt_tokens` | req | `integer` |
|  |  | `provider` | req | `string` |
|  |  | `reasoning_tokens` | req | `integer` (nullable) |
|  |  | `total_tokens` | req | `integer` |
|  | `prompt_tokens` |  | req | `integer` |
|  | `reasoning_tokens` |  | req | `integer` |
|  | `total_tokens` |  | req | `integer` |
| `model` |  |  | req | `string` |
| `provider` |  |  | req | `string` |
| `reasoning` |  |  | opt | `string` (nullable) |
| `reasoning_tokens` |  |  | opt | `integer` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "classification": {
      "enum": [
        "duplicate",
        "near_duplicate",
        "notable_change",
        "novel"
      ],
      "title": "Classification",
      "type": "string"
    },
    "judge_usage": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/JudgeUsageSummary"
        },
        {
          "type": "null"
        }
      ]
    },
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
    }
  },
  "required": [
    "classification",
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
| `tool` |  |  | req | `string` (enum: [search_web, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |

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
        "fetch_page",
        "embed_text",
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

<a id="model-validatorhealthresponse"></a>
### Model: ValidatorHealthResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `status` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "status": {
      "const": "ok",
      "title": "Status",
      "type": "string"
    }
  },
  "required": [
    "status"
  ],
  "title": "ValidatorHealthResponse",
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

<a id="model-validatorreadinessfailureresponse"></a>
### Model: ValidatorReadinessFailureResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | `string` (nullable) |
| `status` |  |  | req | `string` (enum: [waiting_for_platform_registration, waiting_for_auth_warmup, registration_failed, auth_unavailable]) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "detail": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Detail"
    },
    "status": {
      "enum": [
        "waiting_for_platform_registration",
        "waiting_for_auth_warmup",
        "registration_failed",
        "auth_unavailable"
      ],
      "title": "Status",
      "type": "string"
    }
  },
  "required": [
    "status"
  ],
  "title": "ValidatorReadinessFailureResponse",
  "type": "object"
}
```

</details>

<a id="model-validatorreadinesssuccessresponse"></a>
### Model: ValidatorReadinessSuccessResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `status` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "status": {
      "const": "ok",
      "title": "Status",
      "type": "string"
    }
  },
  "required": [
    "status"
  ],
  "title": "ValidatorReadinessSuccessResponse",
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
| `rating_worker_ready` |  |  | opt | `boolean` (default: False) |
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
    "rating_worker_ready": {
      "default": false,
      "title": "Rating Worker Ready",
      "type": "boolean"
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
