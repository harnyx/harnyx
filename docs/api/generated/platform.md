# Platform API reference (generated)

Generated from FastAPI OpenAPI.

## Domains
- [feeds](#feeds)
  - [POST /v1/feeds/search](#endpoint-post-v1-feeds-search)
  - [POST /v1/feeds/{feed_id}/tool/search](#endpoint-post-v1-feeds-feed_id-tool-search)
- [miner-config](#miner-config)
  - [GET /v1/miner-config](#endpoint-get-v1-miner-config)
  - [PUT /v1/miner-config](#endpoint-put-v1-miner-config)
  - [DELETE /v1/miner-config](#endpoint-delete-v1-miner-config)
- [miner-task-batches](#miner-task-batches)
  - [POST /v1/miner-task-batches/batch](#endpoint-post-v1-miner-task-batches-batch)
  - [GET /v1/miner-task-batches/batch/{batch_id}](#endpoint-get-v1-miner-task-batches-batch-batch_id)
  - [GET /v1/miner-task-batches/{batch_id}/artifacts/{artifact_id}](#endpoint-get-v1-miner-task-batches-batch_id-artifacts-artifact_id)
- [miner-task-work](#miner-task-work)
  - [POST /v2/miner-task-work/executions](#endpoint-post-v2-miner-task-work-executions)
  - [POST /v2/miner-task-work/results](#endpoint-post-v2-miner-task-work-results)
  - [POST /v2/miner-task-work/scoreable-executions](#endpoint-post-v2-miner-task-work-scoreable-executions)
  - [POST /v2/miner-task-work/tasks](#endpoint-post-v2-miner-task-work-tasks)
- [miners](#miners)
  - [POST /v1/miners/scripts](#endpoint-post-v1-miners-scripts)
- [platform-tool-proxy](#platform-tool-proxy)
  - [POST /v1/platform-tool-proxy/grants](#endpoint-post-v1-platform-tool-proxy-grants)
- [repo-search](#repo-search)
  - [POST /v1/repo-search/ensure-index](#endpoint-post-v1-repo-search-ensure-index)
  - [POST /v1/repo-search/get-file](#endpoint-post-v1-repo-search-get-file)
  - [POST /v1/repo-search/search](#endpoint-post-v1-repo-search-search)
  - [POST /v1/repo-search/tool/get-file](#endpoint-post-v1-repo-search-tool-get-file)
  - [POST /v1/repo-search/tool/search](#endpoint-post-v1-repo-search-tool-search)
- [validators](#validators)
  - [POST /v1/validators/register](#endpoint-post-v1-validators-register)
- [weights](#weights)
  - [GET /v1/weights](#endpoint-get-v1-weights)

## feeds

### search

<a id="endpoint-post-v1-feeds-search"></a>
#### POST /v1/feeds/search

Search for similar indexed feed items, optionally scoped to strict prior items.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ApiKey OR Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Request**
Content-Type: `application/json`
Body: [FeedSearchRequestModel](#model-feedsearchrequestmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `enqueue_seq` |  |  | opt | `integer` (nullable) |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `num_hit` |  |  | opt | `integer` (default: 20) |
| `search_queries` |  |  | req | array[`string`] |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [FeedSearchResponseModel](#model-feedsearchresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `hits` |  |  | req | array[[FeedSearchHitModel](#model-feedsearchhitmodel)] |
|  | `content_id` |  | req | `string` (format: uuid) |
|  | `content_review_rubric_result` |  | opt | [ExternalEvalResultModel](#model-externalevalresultmodel) (nullable) |
|  |  | `criteria` | req | array[[CriterionAssessmentModel](#model-criterionassessmentmodel)] |
|  |  | `overall_rationale` | opt | `string` (nullable) |
|  |  | `rubric_id` | req | `string` |
|  |  | `rubric_score` | req | `number` |
|  | `content_review_topic_gate` |  | opt | [TopicGateModel](#model-topicgatemodel) (nullable) |
|  |  | `criteria` | opt | array[[CriterionAssessmentModel](#model-criterionassessmentmodel)] (default: []) |
|  |  | `score` | opt | `number` (nullable) |
|  | `decision` |  | opt | `string` (nullable) |
|  | `enqueue_seq` |  | req | `integer` |
|  | `external_id` |  | req | `string` |
|  | `is_excluded` |  | opt | `boolean` (nullable) |
|  | `job_error_code` |  | opt | `string` (nullable) |
|  | `job_error_message` |  | opt | `string` (nullable) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `job_status` |  | opt | `string` (nullable) |
|  | `provider` |  | req | `string` |
|  | `requested_at_epoch_ms` |  | req | `integer` |
|  | `score` |  | opt | `number` (nullable) |
|  | `text` |  | req | `string` |
|  | `url` |  | opt | `string` (nullable) |

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


### {feed_id}

#### tool

##### search

<a id="endpoint-post-v1-feeds-feed_id-tool-search"></a>
###### POST /v1/feeds/{feed_id}/tool/search

Provider-native simple search endpoint for feed-item grounding with optional enqueue boundary.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`) OR ApiKey

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |
| `enqueue_seq` | query | opt | `integer` (nullable) |

**Request**
Content-Type: `application/json`
Body: [_RepoSimpleSearchRequest](#model-_reposimplesearchrequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `query` |  |  | req | `string` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: array[[_RepoSimpleSearchHit](#model-_reposimplesearchhit)]

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `snippet` |  |  | req | `string` |
| `uri` |  |  | req | `string` |

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



## miner-config

<a id="endpoint-get-v1-miner-config"></a>
### GET /v1/miner-config

Read redacted miner provider credential status.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [MinerConfigResponse](#model-minerconfigresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `miner_hotkey_ss58` |  |  | req | `string` |
| `provider_credentials` |  |  | req | `object` |
| `task_retry_count` |  |  | req | `integer` |
| `uid` |  |  | req | `integer` (nullable) |


<a id="endpoint-put-v1-miner-config"></a>
### PUT /v1/miner-config

Create or update one miner provider credential.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Request**
Content-Type: `application/json`
Body: [MinerConfigPutRequest](#model-minerconfigputrequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `key` |  |  | req | `string` |
| `value` |  |  | req | `string` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [MinerConfigResponse](#model-minerconfigresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `miner_hotkey_ss58` |  |  | req | `string` |
| `provider_credentials` |  |  | req | `object` |
| `task_retry_count` |  |  | req | `integer` |
| `uid` |  |  | req | `integer` (nullable) |

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


<a id="endpoint-delete-v1-miner-config"></a>
### DELETE /v1/miner-config

Delete one miner provider credential.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Request**
Content-Type: `application/json`
Body: [MinerConfigDeleteRequest](#model-minerconfigdeleterequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `key` |  |  | req | `string` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [MinerConfigResponse](#model-minerconfigresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `miner_hotkey_ss58` |  |  | req | `string` |
| `provider_credentials` |  |  | req | `object` |
| `task_retry_count` |  |  | req | `integer` |
| `uid` |  |  | req | `integer` (nullable) |

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

### batch

<a id="endpoint-post-v1-miner-task-batches-batch"></a>
#### POST /v1/miner-task-batches/batch

Subnet owner or platform admin API-key emergency recovery route. Each request force-creates a fresh batch, is not replay-safe, and fails fast with 409 while batch creation is already in progress or another batch is running. Returns once the worker has persisted the build claim, started the background continuation, and can identify the accepted batch.

**Auth**: ConfiguredApiKey OR Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Request**
Content-Type: `application/json`
Body: [CreateBatchRequest](#model-createbatchrequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `champion_artifact_id` |  |  | opt | `string` (format: uuid; nullable) |
| `override_task_dataset` |  |  | opt | [OverrideMinerTaskDatasetModel](#model-overrideminertaskdatasetmodel) (nullable) |
|  | `tasks` |  | req | array[[MinerTaskInputModel](#model-minertaskinputmodel)] |
|  |  | `budget_usd` | opt | `number` (default: 0.5) |
|  |  | `query` | req | [Query](#model-query) |
|  |  | `reference_answer` | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `task_id` | req | `string` (format: uuid) |
| `use_previous_task_dataset` |  |  | opt | `boolean` (default: False) |

**Responses**
`202` Successful Response
Content-Type: `application/json`
Body: [CreateBatchAcceptedResponse](#model-createbatchacceptedresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batch_id` |  |  | req | `string` (format: uuid) |

`409` Batch creation already in progress or another batch is running.
Content-Type: `application/json`
Body: [ErrorResponse](#model-errorresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `error_code` |  |  | req | `string` |
| `message` |  |  | req | `string` |

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

`502` Platform API could not validate the worker acknowledgement for the async start.
Content-Type: `application/json`
Body: [ErrorResponse](#model-errorresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `error_code` |  |  | req | `string` |
| `message` |  |  | req | `string` |

`503` Platform API could not reach the worker for batch creation delegation.
Content-Type: `application/json`
Body: [ErrorResponse](#model-errorresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `error_code` |  |  | req | `string` |
| `message` |  |  | req | `string` |


#### {batch_id}

<a id="endpoint-get-v1-miner-task-batches-batch-batch_id"></a>
##### GET /v1/miner-task-batches/batch/{batch_id}

Fetch a previously created miner-task batch.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `batch_id` | path | req | `string` (format: uuid) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [MinerTaskBatchModel](#model-minertaskbatchmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifacts` |  |  | req | array[[MinerTaskBatchArtifactModel](#model-minertaskbatchartifactmodel)] |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `content_hash` |  | req | `string` |
|  | `miner_hotkey_ss58` |  | req | `string` |
|  | `size_bytes` |  | req | `integer` |
|  | `submitted_at` |  | req | `string` (format: date-time) |
|  | `task_retry_count` |  | req | `integer` |
|  | `uid` |  | req | `integer` |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `champion_artifact_id` |  |  | req | `string` (format: uuid; nullable) |
| `completed_at` |  |  | opt | `string` (format: date-time; nullable) |
| `created_at` |  |  | req | `string` (format: date-time) |
| `cutoff_at` |  |  | req | `string` (format: date-time) |
| `failed_at` |  |  | opt | `string` (format: date-time; nullable) |
| `tasks` |  |  | req | array[[MinerTask](#model-minertask)] |
|  | `budget_usd` |  | opt | `number` (default: 0.5) |
|  | `query` |  | req | [Query](#model-query) |
|  |  | `output_schema` | opt | [JsonObject-Input](#model-jsonobject-input) (nullable) |
|  |  | `text` | req | `string` |
|  | `reference_answer` |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `citations` | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  |  | `text` | req | `string` |
|  | `task_id` |  | req | `string` (format: uuid) |

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


### {batch_id}

#### artifacts

##### {artifact_id}

<a id="endpoint-get-v1-miner-task-batches-batch_id-artifacts-artifact_id"></a>
###### GET /v1/miner-task-batches/{batch_id}/artifacts/{artifact_id}

Download a stored script artifact for a batch.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `batch_id` | path | req | `string` (format: uuid) |
| `artifact_id` | path | req | `string` (format: uuid) |

**Responses**
`200` Successful Response

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



## miner-task-work

### executions

<a id="endpoint-post-v2-miner-task-work-executions"></a>
#### POST /v2/miner-task-work/executions

Submit completed platform-owned miner-task execution evidence for the caller validator.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Request**
Content-Type: `application/json`
Body: [MinerTaskWorkExecutionsRequest](#model-minertaskworkexecutionsrequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `executions` |  |  | opt | array[[MinerTaskWorkExecutionEnvelope](#model-minertaskworkexecutionenvelope)] (default: []) |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `execution_completed_at` |  | req | `string` (format: date-time) |
|  | `execution_log` |  | opt | array[[ToolCall-Input](#model-toolcall-input)] (default: []) |
|  |  | `details` | req | [ToolCallDetails-Input](#model-toolcalldetails-input) |
|  |  | `issued_at` | req | `string` (format: date-time) |
|  |  | `outcome` | req | [ToolCallOutcome](#model-toolcalloutcome) |
|  |  | `receipt_id` | req | `string` |
|  |  | `session_id` | req | `string` (format: uuid) |
|  |  | `tool` | req | `string` (enum: [search_web, search_ai, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |
|  |  | `uid` | req | `integer` |
|  | `miner_hotkey_ss58` |  | req | `string` |
|  | `response` |  | req | [Response](#model-response) |
|  |  | `citations` | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  |  | `output` | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  |  | `text` | opt | `string` (nullable) |
|  | `session` |  | req | [SessionModel](#model-sessionmodel) |
|  |  | `expires_at` | req | `string` |
|  |  | `issued_at` | req | `string` |
|  |  | `session_id` | req | `string` (format: uuid) |
|  |  | `status` | req | `string` |
|  |  | `uid` | req | `integer` |
|  | `started_at` |  | req | `string` (format: date-time) |
|  | `task_id` |  | req | `string` (format: uuid) |
|  | `total_tool_usage` |  | req | [ToolUsageSummary](#model-toolusagesummary) |
|  |  | `actual_cost_by_provider` | opt | `object` |
|  |  | `actual_total_cost_usd` | opt | `number` (nullable) |
|  |  | `embedding` | opt | [EmbeddingToolUsageSummary](#model-embeddingtoolusagesummary) |
|  |  | `embedding_cost` | opt | `number` (default: 0.0) |
|  |  | `llm` | opt | [LlmUsageSummary](#model-llmusagesummary) |
|  |  | `llm_cost` | opt | `number` (default: 0.0) |
|  |  | `reference_cost_by_provider` | opt | `object` |
|  |  | `reference_total_cost_usd` | opt | `number` (default: 0.0) |
|  |  | `search_tool` | opt | [SearchToolUsageSummary](#model-searchtoolusagesummary) |
|  |  | `search_tool_cost` | opt | `number` (default: 0.0) |
|  | `trace` |  | opt | [EvaluationTrace](#model-evaluationtrace) (nullable) |
|  |  | `entrypoint_invocation_ms` | opt | `number` (nullable) |
|  |  | `orchestration_ms` | opt | `number` (nullable) |
|  |  | `scoring_judge_attempt_count` | opt | `integer` (nullable) |
|  |  | `scoring_judge_duration_ms` | opt | `number` (nullable) |
|  |  | `scoring_judge_retry_count` | opt | `integer` (nullable) |
|  |  | `scoring_judge_retry_reasons` | opt | array[`string`] (default: []) |
|  |  | `scoring_judge_selected_routes` | opt | array[`string`] (default: []) |
|  |  | `scoring_judge_status` | opt | `string` (enum: [ok, exhausted, failed]; nullable) |
|  |  | `scoring_ms` | opt | `number` (nullable) |
|  | `uid` |  | req | `integer` |
|  | `usage` |  | req | [UsageModel](#model-usagemodel) |
|  |  | `by_provider` | opt | `object` |
|  |  | `call_count` | req | `integer` |
|  |  | `total_completion_tokens` | req | `integer` |
|  |  | `total_prompt_tokens` | req | `integer` |
|  |  | `total_tokens` | req | `integer` |
|  | `validator_session_id` |  | req | `string` (format: uuid) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [MinerTaskWorkExecutionsResponse](#model-minertaskworkexecutionsresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `executions` |  |  | opt | array[[MinerTaskWorkExecutionItemResponse](#model-minertaskworkexecutionitemresponse)] (default: []) |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `canonical` |  | req | `boolean` |
|  | `outcome` |  | req | [MinerTaskResultOutcome](#model-minertaskresultoutcome) |
|  | `reason` |  | opt | `string` (nullable) |
|  | `reason_code` |  | opt | [MinerTaskResultReasonCode](#model-minertaskresultreasoncode) (nullable) |
|  | `task_id` |  | req | `string` (format: uuid) |
| `server_time` |  |  | req | `string` (format: date-time) |

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


### results

<a id="endpoint-post-v2-miner-task-work-results"></a>
#### POST /v2/miner-task-work/results

Submit final scored platform-owned miner-task results for the caller validator.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Request**
Content-Type: `application/json`
Body: [MinerTaskWorkResultsRequest](#model-minertaskworkresultsrequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `results` |  |  | opt | array[[MinerTaskWorkResultEnvelope](#model-minertaskworkresultenvelope)] (default: []) |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `result` |  | opt | [MinerTaskRunRequest](#model-minertaskrunrequest) (nullable) |
|  |  | `batch_id` | req | `string` (format: uuid) |
|  |  | `execution_log` | opt | array[[ToolCall-Input](#model-toolcall-input)] (default: []) |
|  |  | `run` | req | [MinerTaskRunSection](#model-minertaskrunsection) |
|  |  | `score` | opt | `number` (nullable) |
|  |  | `session` | req | [SessionModel](#model-sessionmodel) |
|  |  | `specifics` | req | [EvaluationDetails](#model-evaluationdetails) |
|  |  | `usage` | req | [UsageModel](#model-usagemodel) |
|  | `task_id` |  | req | `string` (format: uuid) |
|  | `terminal_attempt` |  | req | [MinerTaskAttemptAuditPayload](#model-minertaskattemptauditpayload) |
|  |  | `artifact_id` | req | `string` (format: uuid) |
|  |  | `attempt_number` | req | `integer` |
|  |  | `batch_id` | req | `string` (format: uuid) |
|  |  | `delivery_failure_detail` | opt | [FailureDetailPayload](#model-failuredetailpayload) (nullable) |
|  |  | `diagnostics` | opt | [MinerTaskAttemptDiagnosticsPayload](#model-minertaskattemptdiagnosticspayload) (nullable) |
|  |  | `error_code` | opt | `string` (nullable) |
|  |  | `error_summary_code` | opt | `string` (nullable) |
|  |  | `execution_log` | opt | array[[ToolCall-Input](#model-toolcall-input)] (default: []) |
|  |  | `finished_at` | req | `string` (format: date-time) |
|  |  | `max_attempts` | req | `integer` |
|  |  | `miner_hotkey_ss58` | req | `string` |
|  |  | `retry_decision` | req | [MinerTaskAttemptRetryDecision](#model-minertaskattemptretrydecision) |
|  |  | `started_at` | req | `string` (format: date-time) |
|  |  | `status` | req | [MinerTaskAttemptStatus](#model-minertaskattemptstatus) |
|  |  | `task_id` | req | `string` (format: uuid) |
|  |  | `terminal_effect` | req | [MinerTaskAttemptTerminalEffect](#model-minertaskattemptterminaleffect) (nullable) |
|  |  | `uid` | req | `integer` |
|  |  | `validator_session_id` | req | `string` (format: uuid) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [MinerTaskWorkResultsResponse](#model-minertaskworkresultsresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `results` |  |  | opt | array[[MinerTaskWorkResultItemResponse](#model-minertaskworkresultitemresponse)] (default: []) |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `canonical` |  | req | `boolean` |
|  | `outcome` |  | req | [MinerTaskResultOutcome](#model-minertaskresultoutcome) |
|  | `reason` |  | opt | `string` (nullable) |
|  | `reason_code` |  | opt | [MinerTaskResultReasonCode](#model-minertaskresultreasoncode) (nullable) |
|  | `task_id` |  | req | `string` (format: uuid) |
| `server_time` |  |  | req | `string` (format: date-time) |

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


### scoreable-executions

<a id="endpoint-post-v2-miner-task-work-scoreable-executions"></a>
#### POST /v2/miner-task-work/scoreable-executions

Return accepted miner-task executions that still need final scoring.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Request**
Content-Type: `application/json`
Body: [MinerTaskWorkScoreableExecutionsRequest](#model-minertaskworkscoreableexecutionsrequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `active_scoring` |  |  | opt | array[[MinerTaskAttemptIdentityPayload](#model-minertaskattemptidentitypayload)] (default: []) |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `task_id` |  | req | `string` (format: uuid) |
|  | `validator_session_id` |  | opt | `string` (format: uuid; nullable) |
| `limit` |  |  | req | `integer` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [MinerTaskWorkScoreableExecutionsResponse](#model-minertaskworkscoreableexecutionsresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `executions` |  |  | opt | array[[MinerTaskWorkScoreableExecutionPayload](#model-minertaskworkscoreableexecutionpayload)] (default: []) |
|  | `artifact` |  | req | [MinerTaskWorkArtifactPayload](#model-minertaskworkartifactpayload) |
|  |  | `artifact_id` | req | `string` (format: uuid) |
|  |  | `content_hash` | req | `string` |
|  |  | `miner_hotkey_ss58` | opt | `string` (nullable) |
|  |  | `size_bytes` | req | `integer` |
|  |  | `uid` | req | `integer` |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `execution_completed_at` |  | req | `string` (format: date-time) |
|  | `execution_log` |  | opt | array[[ToolCall-Output](#model-toolcall-output)] (default: []) |
|  |  | `details` | req | [ToolCallDetails-Output](#model-toolcalldetails-output) |
|  |  | `issued_at` | req | `string` (format: date-time) |
|  |  | `outcome` | req | [ToolCallOutcome](#model-toolcalloutcome) |
|  |  | `receipt_id` | req | `string` |
|  |  | `session_id` | req | `string` (format: uuid) |
|  |  | `tool` | req | `string` (enum: [search_web, search_ai, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |
|  |  | `uid` | req | `integer` |
|  | `max_attempts` |  | req | `integer` |
|  | `miner_hotkey_ss58` |  | req | `string` |
|  | `response` |  | req | [Response](#model-response) (nullable) |
|  |  | `citations` | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  |  | `output` | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  |  | `text` | opt | `string` (nullable) |
|  | `session` |  | req | [SessionModel](#model-sessionmodel) |
|  |  | `expires_at` | req | `string` |
|  |  | `issued_at` | req | `string` |
|  |  | `session_id` | req | `string` (format: uuid) |
|  |  | `status` | req | `string` |
|  |  | `uid` | req | `integer` |
|  | `started_at` |  | req | `string` (format: date-time) |
|  | `task` |  | req | [MinerTask](#model-minertask) |
|  |  | `budget_usd` | opt | `number` (default: 0.5) |
|  |  | `query` | req | [Query](#model-query) |
|  |  | `reference_answer` | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `task_id` | req | `string` (format: uuid) |
|  | `total_tool_usage` |  | req | [ToolUsageSummary](#model-toolusagesummary) |
|  |  | `actual_cost_by_provider` | opt | `object` |
|  |  | `actual_total_cost_usd` | opt | `number` (nullable) |
|  |  | `embedding` | opt | [EmbeddingToolUsageSummary](#model-embeddingtoolusagesummary) |
|  |  | `embedding_cost` | opt | `number` (default: 0.0) |
|  |  | `llm` | opt | [LlmUsageSummary](#model-llmusagesummary) |
|  |  | `llm_cost` | opt | `number` (default: 0.0) |
|  |  | `reference_cost_by_provider` | opt | `object` |
|  |  | `reference_total_cost_usd` | opt | `number` (default: 0.0) |
|  |  | `search_tool` | opt | [SearchToolUsageSummary](#model-searchtoolusagesummary) |
|  |  | `search_tool_cost` | opt | `number` (default: 0.0) |
|  | `trace` |  | opt | [EvaluationTrace](#model-evaluationtrace) (nullable) |
|  |  | `entrypoint_invocation_ms` | opt | `number` (nullable) |
|  |  | `orchestration_ms` | opt | `number` (nullable) |
|  |  | `scoring_judge_attempt_count` | opt | `integer` (nullable) |
|  |  | `scoring_judge_duration_ms` | opt | `number` (nullable) |
|  |  | `scoring_judge_retry_count` | opt | `integer` (nullable) |
|  |  | `scoring_judge_retry_reasons` | opt | array[`string`] (default: []) |
|  |  | `scoring_judge_selected_routes` | opt | array[`string`] (default: []) |
|  |  | `scoring_judge_status` | opt | `string` (enum: [ok, exhausted, failed]; nullable) |
|  |  | `scoring_ms` | opt | `number` (nullable) |
|  | `uid` |  | req | `integer` |
|  | `usage` |  | req | [UsageModel](#model-usagemodel) |
|  |  | `by_provider` | opt | `object` |
|  |  | `call_count` | req | `integer` |
|  |  | `total_completion_tokens` | req | `integer` |
|  |  | `total_prompt_tokens` | req | `integer` |
|  |  | `total_tokens` | req | `integer` |
|  | `validator_session_id` |  | req | `string` (format: uuid) |
| `server_time` |  |  | req | `string` (format: date-time) |

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


### tasks

<a id="endpoint-post-v2-miner-task-work-tasks"></a>
#### POST /v2/miner-task-work/tasks

Return the next platform-owned miner-task attempts for the caller validator.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Request**
Content-Type: `application/json`
Body: [MinerTaskWorkTasksRequest](#model-minertaskworktasksrequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `active_attempts` |  |  | opt | array[[MinerTaskAttemptIdentityPayload](#model-minertaskattemptidentitypayload)] (default: []) |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `task_id` |  | req | `string` (format: uuid) |
|  | `validator_session_id` |  | opt | `string` (format: uuid; nullable) |
| `max_active_artifacts` |  |  | req | `integer` |
| `target_concurrency` |  |  | req | `integer` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [MinerTaskWorkTasksResponse](#model-minertaskworktasksresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `server_time` |  |  | req | `string` (format: date-time) |
| `tasks` |  |  | opt | array[[MinerTaskWorkAssignmentPayload](#model-minertaskworkassignmentpayload)] (default: []) |
|  | `artifact` |  | req | [MinerTaskWorkArtifactPayload](#model-minertaskworkartifactpayload) |
|  |  | `artifact_id` | req | `string` (format: uuid) |
|  |  | `content_hash` | req | `string` |
|  |  | `miner_hotkey_ss58` | opt | `string` (nullable) |
|  |  | `size_bytes` | req | `integer` |
|  |  | `uid` | req | `integer` |
|  | `assignment_token` |  | req | `string` |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `max_attempts` |  | req | `integer` |
|  | `task` |  | req | [MinerTask](#model-minertask) |
|  |  | `budget_usd` | opt | `number` (default: 0.5) |
|  |  | `query` | req | [Query](#model-query) |
|  |  | `reference_answer` | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `task_id` | req | `string` (format: uuid) |

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



## miners

### scripts

<a id="endpoint-post-v1-miners-scripts"></a>
#### POST /v1/miners/scripts

Upload a miner script artifact (base64 + sha256) for later evaluation.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Request**
Content-Type: `application/json`
Body: [UploadScriptRequest](#model-uploadscriptrequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `script_b64` |  |  | req | `string` |
| `sha256` |  |  | req | `string` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [ScriptArtifactModel](#model-scriptartifactmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `content_hash` |  |  | req | `string` |
| `size_bytes` |  |  | req | `integer` |
| `submitted_at` |  |  | req | `string` (format: date-time) |
| `uid` |  |  | req | `integer` |

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



## platform-tool-proxy

### grants

<a id="endpoint-post-v1-platform-tool-proxy-grants"></a>
#### POST /v1/platform-tool-proxy/grants

Create a short-lived platform-tool-proxy token for a validator-owned batch delivery.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Request**
Content-Type: `application/json`
Body: [PlatformToolProxyGrantRequest](#model-platformtoolproxygrantrequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `assignment_token` |  |  | req | `string` |
| `attempt_number` |  |  | req | `integer` |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `task_id` |  |  | req | `string` (format: uuid) |
| `validator_session_id` |  |  | req | `string` (format: uuid) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [PlatformToolProxyGrantResponse](#model-platformtoolproxygrantresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `expires_at` |  |  | req | `string` (format: date-time) |
| `token` |  |  | req | `string` |

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



## repo-search

### ensure-index

<a id="endpoint-post-v1-repo-search-ensure-index"></a>
#### POST /v1/repo-search/ensure-index

Ensure a repository index is available for repo tools.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`) OR ApiKey

**Request**
Content-Type: `application/json`
Body: [RepoSearchEnsureIndexRequestModel](#model-reposearchensureindexrequestmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `commit_sha` |  |  | req | `string` |
| `repo_url` |  |  | req | `string` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [StatusResponse](#model-statusresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `status` |  |  | req | `string` |

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


### get-file

<a id="endpoint-post-v1-repo-search-get-file"></a>
#### POST /v1/repo-search/get-file

Fetch a markdown file from a repository snapshot.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`) OR ApiKey

**Request**
Content-Type: `application/json`
Body: [GetRepoFileRequest](#model-getrepofilerequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `commit_sha` |  |  | req | `string` |
| `end_line` |  |  | opt | `integer` (nullable) |
| `path` |  |  | req | `string` |
| `repo_url` |  |  | req | `string` |
| `start_line` |  |  | opt | `integer` (nullable) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [GetRepoFileResponse](#model-getrepofileresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `data` |  |  | opt | array[[GetRepoFileResult](#model-getrepofileresult)] |
|  | `excerpt` |  | opt | `string` (nullable) |
|  | `path` |  | req | `string` |
|  | `text` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | req | `string` |

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


### search

<a id="endpoint-post-v1-repo-search-search"></a>
#### POST /v1/repo-search/search

Search markdown files in a repository snapshot.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`) OR ApiKey

**Request**
Content-Type: `application/json`
Body: [SearchRepoSearchRequest](#model-searchreposearchrequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `commit_sha` |  |  | req | `string` |
| `limit` |  |  | opt | `integer` (default: 10) |
| `path_glob` |  |  | opt | `string` (nullable) |
| `query` |  |  | req | `string` |
| `repo_url` |  |  | req | `string` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [SearchRepoSearchResponse](#model-searchreposearchresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `data` |  |  | opt | array[[SearchRepoResult](#model-searchreporesult)] |
|  | `bm25` |  | opt | `number` (nullable) |
|  | `excerpt` |  | opt | `string` (nullable) |
|  | `path` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | req | `string` |

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


### tool

#### get-file

<a id="endpoint-post-v1-repo-search-tool-get-file"></a>
##### POST /v1/repo-search/tool/get-file

Provider-native simple search endpoint for full-file repo grounding.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`) OR ApiKey

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `repo_url` | query | req | `string` |
| `commit_sha` | query | req | `string` |

**Request**
Content-Type: `application/json`
Body: [_RepoSimpleSearchRequest](#model-_reposimplesearchrequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `query` |  |  | req | `string` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: array[[_RepoSimpleSearchHit](#model-_reposimplesearchhit)]

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `snippet` |  |  | req | `string` |
| `uri` |  |  | req | `string` |

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


#### search

<a id="endpoint-post-v1-repo-search-tool-search"></a>
##### POST /v1/repo-search/tool/search

Provider-native simple search endpoint for repo-diff grounding.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`) OR ApiKey

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `repo_url` | query | req | `string` |
| `commit_sha` | query | req | `string` |

**Request**
Content-Type: `application/json`
Body: [_RepoSimpleSearchRequest](#model-_reposimplesearchrequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `query` |  |  | req | `string` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: array[[_RepoSimpleSearchHit](#model-_reposimplesearchhit)]

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `snippet` |  |  | req | `string` |
| `uri` |  |  | req | `string` |

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



## validators

### register

<a id="endpoint-post-v1-validators-register"></a>
#### POST /v1/validators/register

Register (or refresh) the caller validator's base URL.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Request**
Content-Type: `application/json`
Body: [RegisterValidatorRequest](#model-registervalidatorrequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `base_url` |  |  | req | `string` |
| `local_image_id` |  |  | opt | `string` (nullable) |
| `registry_digest` |  |  | opt | `string` (nullable) |
| `source_revision` |  |  | opt | `string` (nullable) |
| `validator_version` |  |  | opt | `string` (nullable) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [StatusResponse](#model-statusresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `status` |  |  | req | `string` |

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



## weights

<a id="endpoint-get-v1-weights"></a>
### GET /v1/weights

Fetch the latest weights for the caller validator.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [WeightsResponse](#model-weightsresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `champion_uid` |  |  | opt | `integer` (nullable) |
| `weights` |  |  | req | `object` |



## Models

<a id="model-_reposimplesearchhit"></a>
### Model: _RepoSimpleSearchHit

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `snippet` |  |  | req | `string` |
| `uri` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "snippet": {
      "title": "Snippet",
      "type": "string"
    },
    "uri": {
      "title": "Uri",
      "type": "string"
    }
  },
  "required": [
    "snippet",
    "uri"
  ],
  "title": "_RepoSimpleSearchHit",
  "type": "object"
}
```

</details>

<a id="model-_reposimplesearchrequest"></a>
### Model: _RepoSimpleSearchRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `query` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "query": {
      "minLength": 1,
      "title": "Query",
      "type": "string"
    }
  },
  "required": [
    "query"
  ],
  "title": "_RepoSimpleSearchRequest",
  "type": "object"
}
```

</details>

<a id="model-answercitation"></a>
### Model: AnswerCitation

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `note` |  |  | opt | `string` (nullable) |
| `title` |  |  | opt | `string` (nullable) |
| `url` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
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

<a id="model-citationmodel"></a>
### Model: CitationModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `note` |  |  | opt | `string` (nullable) |
| `title` |  |  | opt | `string` (nullable) |
| `url` |  |  | req | `string` |

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
      "title": "Url",
      "type": "string"
    }
  },
  "required": [
    "url"
  ],
  "title": "CitationModel",
  "type": "object"
}
```

</details>

<a id="model-createbatchacceptedresponse"></a>
### Model: CreateBatchAcceptedResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batch_id` |  |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "batch_id": {
      "format": "uuid",
      "title": "Batch Id",
      "type": "string"
    }
  },
  "required": [
    "batch_id"
  ],
  "title": "CreateBatchAcceptedResponse",
  "type": "object"
}
```

</details>

<a id="model-createbatchrequest"></a>
### Model: CreateBatchRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `champion_artifact_id` |  |  | opt | `string` (format: uuid; nullable) |
| `override_task_dataset` |  |  | opt | [OverrideMinerTaskDatasetModel](#model-overrideminertaskdatasetmodel) (nullable) |
|  | `tasks` |  | req | array[[MinerTaskInputModel](#model-minertaskinputmodel)] |
|  |  | `budget_usd` | opt | `number` (default: 0.5) |
|  |  | `query` | req | [Query](#model-query) |
|  |  | `reference_answer` | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `task_id` | req | `string` (format: uuid) |
| `use_previous_task_dataset` |  |  | opt | `boolean` (default: False) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "champion_artifact_id": {
      "anyOf": [
        {
          "format": "uuid",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Champion Artifact Id"
    },
    "override_task_dataset": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/OverrideMinerTaskDatasetModel"
        },
        {
          "type": "null"
        }
      ]
    },
    "use_previous_task_dataset": {
      "default": false,
      "title": "Use Previous Task Dataset",
      "type": "boolean"
    }
  },
  "title": "CreateBatchRequest",
  "type": "object"
}
```

</details>

<a id="model-criterionassessmentmodel"></a>
### Model: CriterionAssessmentModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `aggregate_score` |  |  | req | `number` |
| `criterion_evaluations` |  |  | req | array[[CriterionEvaluationModel](#model-criterionevaluationmodel)] |
|  | `citations` |  | opt | array[[CitationModel](#model-citationmodel)] (default: []) |
|  |  | `note` | opt | `string` (nullable) |
|  |  | `title` | opt | `string` (nullable) |
|  |  | `url` | req | `string` |
|  | `internal_metadata` |  | opt | `object` (nullable) |
|  | `justification` |  | req | `string` |
|  | `spans` |  | opt | array[[SpanModel](#model-spanmodel)] (default: []) |
|  |  | `end` | req | `integer` |
|  |  | `excerpt` | req | `string` |
|  |  | `start` | req | `integer` |
|  | `verdict` |  | req | `integer` |
| `criterion_id` |  |  | req | `string` |
| `verdict_options` |  |  | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  | `description` |  | req | `string` |
|  | `value` |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "aggregate_score": {
      "title": "Aggregate Score",
      "type": "number"
    },
    "criterion_evaluations": {
      "items": {
        "anyOf": [
          {
            "$ref": "#/components/schemas/CriterionEvaluationModel"
          },
          {
            "type": "null"
          }
        ]
      },
      "title": "Criterion Evaluations",
      "type": "array"
    },
    "criterion_id": {
      "title": "Criterion Id",
      "type": "string"
    },
    "verdict_options": {
      "items": {
        "$ref": "#/components/schemas/VerdictOptionModel"
      },
      "title": "Verdict Options",
      "type": "array"
    }
  },
  "required": [
    "criterion_id",
    "verdict_options",
    "aggregate_score",
    "criterion_evaluations"
  ],
  "title": "CriterionAssessmentModel",
  "type": "object"
}
```

</details>

<a id="model-criterionevaluationmodel"></a>
### Model: CriterionEvaluationModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `citations` |  |  | opt | array[[CitationModel](#model-citationmodel)] (default: []) |
|  | `note` |  | opt | `string` (nullable) |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | req | `string` |
| `internal_metadata` |  |  | opt | `object` (nullable) |
| `justification` |  |  | req | `string` |
| `spans` |  |  | opt | array[[SpanModel](#model-spanmodel)] (default: []) |
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
        "$ref": "#/components/schemas/CitationModel"
      },
      "title": "Citations",
      "type": "array"
    },
    "internal_metadata": {
      "anyOf": [
        {
          "additionalProperties": true,
          "type": "object"
        },
        {
          "type": "null"
        }
      ],
      "title": "Internal Metadata"
    },
    "justification": {
      "title": "Justification",
      "type": "string"
    },
    "spans": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/SpanModel"
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
  "title": "CriterionEvaluationModel",
  "type": "object"
}
```

</details>

<a id="model-embeddingtoolusagesummary"></a>
### Model: EmbeddingToolUsageSummary

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `actual_cost` |  |  | opt | `number` (nullable) |
| `call_count` |  |  | opt | `integer` (default: 0) |
| `cost` |  |  | opt | `number` (default: 0.0) |
| `reference_cost` |  |  | opt | `number` (default: 0.0) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "actual_cost": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Actual Cost"
    },
    "call_count": {
      "default": 0,
      "title": "Call Count",
      "type": "integer"
    },
    "cost": {
      "default": 0.0,
      "title": "Cost",
      "type": "number"
    },
    "reference_cost": {
      "default": 0.0,
      "title": "Reference Cost",
      "type": "number"
    }
  },
  "title": "EmbeddingToolUsageSummary",
  "type": "object"
}
```

</details>

<a id="model-errorresponse"></a>
### Model: ErrorResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `error_code` |  |  | req | `string` |
| `message` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "error_code": {
      "title": "Error Code",
      "type": "string"
    },
    "message": {
      "title": "Message",
      "type": "string"
    }
  },
  "required": [
    "error_code",
    "message"
  ],
  "title": "ErrorResponse",
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
|  | `code` |  | req | [MinerTaskErrorCode](#model-minertaskerrorcode) |
|  | `message` |  | req | `string` |
| `score_breakdown` |  |  | opt | [ScoreBreakdown](#model-scorebreakdown) (nullable) |
|  | `comparison_score` |  | req | `number` |
|  | `reasoning` |  | opt | [ScorerReasoning](#model-scorerreasoning) (nullable) |
|  |  | `reasoning_tokens` | opt | `integer` (nullable) |
|  |  | `text` | opt | `string` (nullable) |
|  | `scoring_version` |  | req | `string` |
|  | `total_score` |  | req | `number` |
| `scoring_judge_usage` |  |  | opt | [JudgeUsageSummary](#model-judgeusagesummary) (nullable) |
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
| `total_tool_usage` |  |  | opt | [ToolUsageSummary](#model-toolusagesummary) |
|  | `actual_cost_by_provider` |  | opt | `object` |
|  | `actual_total_cost_usd` |  | opt | `number` (nullable) |
|  | `embedding` |  | opt | [EmbeddingToolUsageSummary](#model-embeddingtoolusagesummary) |
|  |  | `actual_cost` | opt | `number` (nullable) |
|  |  | `call_count` | opt | `integer` (default: 0) |
|  |  | `cost` | opt | `number` (default: 0.0) |
|  |  | `reference_cost` | opt | `number` (default: 0.0) |
|  | `embedding_cost` |  | opt | `number` (default: 0.0) |
|  | `llm` |  | opt | [LlmUsageSummary](#model-llmusagesummary) |
|  |  | `actual_cost` | opt | `number` (nullable) |
|  |  | `call_count` | opt | `integer` (default: 0) |
|  |  | `completion_tokens` | opt | `integer` (default: 0) |
|  |  | `cost` | opt | `number` (default: 0.0) |
|  |  | `prompt_tokens` | opt | `integer` (default: 0) |
|  |  | `providers` | opt | `object` |
|  |  | `reasoning_tokens` | opt | `integer` (default: 0) |
|  |  | `reference_cost` | opt | `number` (default: 0.0) |
|  |  | `total_tokens` | opt | `integer` (default: 0) |
|  | `llm_cost` |  | opt | `number` (default: 0.0) |
|  | `reference_cost_by_provider` |  | opt | `object` |
|  | `reference_total_cost_usd` |  | opt | `number` (default: 0.0) |
|  | `search_tool` |  | opt | [SearchToolUsageSummary](#model-searchtoolusagesummary) |
|  |  | `actual_cost` | opt | `number` (nullable) |
|  |  | `call_count` | opt | `integer` (default: 0) |
|  |  | `cost` | opt | `number` (default: 0.0) |
|  |  | `reference_cost` | opt | `number` (default: 0.0) |
|  | `search_tool_cost` |  | opt | `number` (default: 0.0) |
| `trace` |  |  | opt | [EvaluationTrace](#model-evaluationtrace) (nullable) |
|  | `entrypoint_invocation_ms` |  | opt | `number` (nullable) |
|  | `orchestration_ms` |  | opt | `number` (nullable) |
|  | `scoring_judge_attempt_count` |  | opt | `integer` (nullable) |
|  | `scoring_judge_duration_ms` |  | opt | `number` (nullable) |
|  | `scoring_judge_retry_count` |  | opt | `integer` (nullable) |
|  | `scoring_judge_retry_reasons` |  | opt | array[`string`] (default: []) |
|  | `scoring_judge_selected_routes` |  | opt | array[`string`] (default: []) |
|  | `scoring_judge_status` |  | opt | `string` (enum: [ok, exhausted, failed]; nullable) |
|  | `scoring_ms` |  | opt | `number` (nullable) |

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
    "scoring_judge_usage": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/JudgeUsageSummary"
        },
        {
          "type": "null"
        }
      ]
    },
    "total_tool_usage": {
      "$ref": "#/components/schemas/ToolUsageSummary"
    },
    "trace": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/EvaluationTrace"
        },
        {
          "type": "null"
        }
      ]
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
| `code` |  |  | req | [MinerTaskErrorCode](#model-minertaskerrorcode) |
| `message` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "code": {
      "$ref": "#/components/schemas/MinerTaskErrorCode"
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

<a id="model-evaluationtrace"></a>
### Model: EvaluationTrace

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `entrypoint_invocation_ms` |  |  | opt | `number` (nullable) |
| `orchestration_ms` |  |  | opt | `number` (nullable) |
| `scoring_judge_attempt_count` |  |  | opt | `integer` (nullable) |
| `scoring_judge_duration_ms` |  |  | opt | `number` (nullable) |
| `scoring_judge_retry_count` |  |  | opt | `integer` (nullable) |
| `scoring_judge_retry_reasons` |  |  | opt | array[`string`] (default: []) |
| `scoring_judge_selected_routes` |  |  | opt | array[`string`] (default: []) |
| `scoring_judge_status` |  |  | opt | `string` (enum: [ok, exhausted, failed]; nullable) |
| `scoring_ms` |  |  | opt | `number` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "entrypoint_invocation_ms": {
      "anyOf": [
        {
          "minimum": 0.0,
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Entrypoint Invocation Ms"
    },
    "orchestration_ms": {
      "anyOf": [
        {
          "minimum": 0.0,
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Orchestration Ms"
    },
    "scoring_judge_attempt_count": {
      "anyOf": [
        {
          "minimum": 0.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Scoring Judge Attempt Count"
    },
    "scoring_judge_duration_ms": {
      "anyOf": [
        {
          "minimum": 0.0,
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Scoring Judge Duration Ms"
    },
    "scoring_judge_retry_count": {
      "anyOf": [
        {
          "minimum": 0.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Scoring Judge Retry Count"
    },
    "scoring_judge_retry_reasons": {
      "default": [],
      "items": {
        "type": "string"
      },
      "title": "Scoring Judge Retry Reasons",
      "type": "array"
    },
    "scoring_judge_selected_routes": {
      "default": [],
      "items": {
        "type": "string"
      },
      "title": "Scoring Judge Selected Routes",
      "type": "array"
    },
    "scoring_judge_status": {
      "anyOf": [
        {
          "enum": [
            "ok",
            "exhausted",
            "failed"
          ],
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Scoring Judge Status"
    },
    "scoring_ms": {
      "anyOf": [
        {
          "minimum": 0.0,
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Scoring Ms"
    }
  },
  "title": "EvaluationTrace",
  "type": "object"
}
```

</details>

<a id="model-externalevalresultmodel"></a>
### Model: ExternalEvalResultModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `criteria` |  |  | req | array[[CriterionAssessmentModel](#model-criterionassessmentmodel)] |
|  | `aggregate_score` |  | req | `number` |
|  | `criterion_evaluations` |  | req | array[[CriterionEvaluationModel](#model-criterionevaluationmodel)] |
|  |  | `citations` | opt | array[[CitationModel](#model-citationmodel)] (default: []) |
|  |  | `internal_metadata` | opt | `object` (nullable) |
|  |  | `justification` | req | `string` |
|  |  | `spans` | opt | array[[SpanModel](#model-spanmodel)] (default: []) |
|  |  | `verdict` | req | `integer` |
|  | `criterion_id` |  | req | `string` |
|  | `verdict_options` |  | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `description` | req | `string` |
|  |  | `value` | req | `integer` |
| `overall_rationale` |  |  | opt | `string` (nullable) |
| `rubric_id` |  |  | req | `string` |
| `rubric_score` |  |  | req | `number` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "criteria": {
      "items": {
        "$ref": "#/components/schemas/CriterionAssessmentModel"
      },
      "title": "Criteria",
      "type": "array"
    },
    "overall_rationale": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Overall Rationale"
    },
    "rubric_id": {
      "title": "Rubric Id",
      "type": "string"
    },
    "rubric_score": {
      "title": "Rubric Score",
      "type": "number"
    }
  },
  "required": [
    "rubric_id",
    "criteria",
    "rubric_score"
  ],
  "title": "ExternalEvalResultModel",
  "type": "object"
}
```

</details>

<a id="model-failuredetailpayload"></a>
### Model: FailureDetailPayload

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | opt | `string` (format: uuid; nullable) |
| `error_code` |  |  | req | `string` |
| `error_message` |  |  | req | `string` |
| `exception_type` |  |  | opt | `string` (nullable) |
| `occurred_at` |  |  | req | `string` (format: date-time) |
| `sandbox_diagnostics` |  |  | opt | [SandboxFailureDiagnostics](#model-sandboxfailurediagnostics) (nullable) |
|  | `container_id` |  | opt | `string` (nullable) |
|  | `container_name` |  | opt | `string` (nullable) |
|  | `docker_inspect_error_tail` |  | opt | `string` (nullable) |
|  | `docker_logs_error_tail` |  | opt | `string` (nullable) |
|  | `docker_logs_tail` |  | opt | `string` (nullable) |
|  | `error_text` |  | opt | `string` (nullable) |
|  | `exit_code` |  | opt | `integer` (nullable) |
|  | `image` |  | opt | `string` (nullable) |
|  | `oom_killed` |  | opt | `boolean` (nullable) |
|  | `pull_policy` |  | opt | `string` (nullable) |
|  | `pull_returncode` |  | opt | `integer` (nullable) |
|  | `pull_stderr_tail` |  | opt | `string` (nullable) |
|  | `pull_stdout_tail` |  | opt | `string` (nullable) |
|  | `run_returncode` |  | opt | `integer` (nullable) |
|  | `run_stderr_tail` |  | opt | `string` (nullable) |
|  | `run_stdout_tail` |  | opt | `string` (nullable) |
|  | `state_error` |  | opt | `string` (nullable) |
|  | `status` |  | opt | `string` (nullable) |
| `task_id` |  |  | opt | `string` (format: uuid; nullable) |
| `traceback` |  |  | opt | `string` (nullable) |
| `uid` |  |  | opt | `integer` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "artifact_id": {
      "anyOf": [
        {
          "format": "uuid",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Artifact Id"
    },
    "error_code": {
      "title": "Error Code",
      "type": "string"
    },
    "error_message": {
      "title": "Error Message",
      "type": "string"
    },
    "exception_type": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Exception Type"
    },
    "occurred_at": {
      "format": "date-time",
      "title": "Occurred At",
      "type": "string"
    },
    "sandbox_diagnostics": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/SandboxFailureDiagnostics"
        },
        {
          "type": "null"
        }
      ]
    },
    "task_id": {
      "anyOf": [
        {
          "format": "uuid",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Task Id"
    },
    "traceback": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Traceback"
    },
    "uid": {
      "anyOf": [
        {
          "minimum": 0.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Uid"
    }
  },
  "required": [
    "error_code",
    "error_message",
    "occurred_at"
  ],
  "title": "FailureDetailPayload",
  "type": "object"
}
```

</details>

<a id="model-feedsearchhitmodel"></a>
### Model: FeedSearchHitModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `content_id` |  |  | req | `string` (format: uuid) |
| `content_review_rubric_result` |  |  | opt | [ExternalEvalResultModel](#model-externalevalresultmodel) (nullable) |
|  | `criteria` |  | req | array[[CriterionAssessmentModel](#model-criterionassessmentmodel)] |
|  |  | `aggregate_score` | req | `number` |
|  |  | `criterion_evaluations` | req | array[[CriterionEvaluationModel](#model-criterionevaluationmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  | `overall_rationale` |  | opt | `string` (nullable) |
|  | `rubric_id` |  | req | `string` |
|  | `rubric_score` |  | req | `number` |
| `content_review_topic_gate` |  |  | opt | [TopicGateModel](#model-topicgatemodel) (nullable) |
|  | `criteria` |  | opt | array[[CriterionAssessmentModel](#model-criterionassessmentmodel)] (default: []) |
|  |  | `aggregate_score` | req | `number` |
|  |  | `criterion_evaluations` | req | array[[CriterionEvaluationModel](#model-criterionevaluationmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  | `score` |  | opt | `number` (nullable) |
| `decision` |  |  | opt | `string` (nullable) |
| `enqueue_seq` |  |  | req | `integer` |
| `external_id` |  |  | req | `string` |
| `is_excluded` |  |  | opt | `boolean` (nullable) |
| `job_error_code` |  |  | opt | `string` (nullable) |
| `job_error_message` |  |  | opt | `string` (nullable) |
| `job_id` |  |  | req | `string` (format: uuid) |
| `job_status` |  |  | opt | `string` (nullable) |
| `provider` |  |  | req | `string` |
| `requested_at_epoch_ms` |  |  | req | `integer` |
| `score` |  |  | opt | `number` (nullable) |
| `text` |  |  | req | `string` |
| `url` |  |  | opt | `string` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "content_id": {
      "format": "uuid",
      "title": "Content Id",
      "type": "string"
    },
    "content_review_rubric_result": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/ExternalEvalResultModel"
        },
        {
          "type": "null"
        }
      ]
    },
    "content_review_topic_gate": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/TopicGateModel"
        },
        {
          "type": "null"
        }
      ]
    },
    "decision": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Decision"
    },
    "enqueue_seq": {
      "title": "Enqueue Seq",
      "type": "integer"
    },
    "external_id": {
      "title": "External Id",
      "type": "string"
    },
    "is_excluded": {
      "anyOf": [
        {
          "type": "boolean"
        },
        {
          "type": "null"
        }
      ],
      "title": "Is Excluded"
    },
    "job_error_code": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Job Error Code"
    },
    "job_error_message": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Job Error Message"
    },
    "job_id": {
      "format": "uuid",
      "title": "Job Id",
      "type": "string"
    },
    "job_status": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Job Status"
    },
    "provider": {
      "title": "Provider",
      "type": "string"
    },
    "requested_at_epoch_ms": {
      "title": "Requested At Epoch Ms",
      "type": "integer"
    },
    "score": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Score"
    },
    "text": {
      "title": "Text",
      "type": "string"
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
    "job_id",
    "content_id",
    "provider",
    "external_id",
    "text",
    "requested_at_epoch_ms",
    "enqueue_seq"
  ],
  "title": "FeedSearchHitModel",
  "type": "object"
}
```

</details>

<a id="model-feedsearchrequestmodel"></a>
### Model: FeedSearchRequestModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `enqueue_seq` |  |  | opt | `integer` (nullable) |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `num_hit` |  |  | opt | `integer` (default: 20) |
| `search_queries` |  |  | req | array[`string`] |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "enqueue_seq": {
      "anyOf": [
        {
          "minimum": 0.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Enqueue Seq"
    },
    "feed_id": {
      "format": "uuid",
      "title": "Feed Id",
      "type": "string"
    },
    "num_hit": {
      "default": 20,
      "maximum": 200.0,
      "minimum": 1.0,
      "title": "Num Hit",
      "type": "integer"
    },
    "search_queries": {
      "items": {
        "type": "string"
      },
      "minItems": 1,
      "title": "Search Queries",
      "type": "array"
    }
  },
  "required": [
    "feed_id",
    "search_queries"
  ],
  "title": "FeedSearchRequestModel",
  "type": "object"
}
```

</details>

<a id="model-feedsearchresponsemodel"></a>
### Model: FeedSearchResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `hits` |  |  | req | array[[FeedSearchHitModel](#model-feedsearchhitmodel)] |
|  | `content_id` |  | req | `string` (format: uuid) |
|  | `content_review_rubric_result` |  | opt | [ExternalEvalResultModel](#model-externalevalresultmodel) (nullable) |
|  |  | `criteria` | req | array[[CriterionAssessmentModel](#model-criterionassessmentmodel)] |
|  |  | `overall_rationale` | opt | `string` (nullable) |
|  |  | `rubric_id` | req | `string` |
|  |  | `rubric_score` | req | `number` |
|  | `content_review_topic_gate` |  | opt | [TopicGateModel](#model-topicgatemodel) (nullable) |
|  |  | `criteria` | opt | array[[CriterionAssessmentModel](#model-criterionassessmentmodel)] (default: []) |
|  |  | `score` | opt | `number` (nullable) |
|  | `decision` |  | opt | `string` (nullable) |
|  | `enqueue_seq` |  | req | `integer` |
|  | `external_id` |  | req | `string` |
|  | `is_excluded` |  | opt | `boolean` (nullable) |
|  | `job_error_code` |  | opt | `string` (nullable) |
|  | `job_error_message` |  | opt | `string` (nullable) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `job_status` |  | opt | `string` (nullable) |
|  | `provider` |  | req | `string` |
|  | `requested_at_epoch_ms` |  | req | `integer` |
|  | `score` |  | opt | `number` (nullable) |
|  | `text` |  | req | `string` |
|  | `url` |  | opt | `string` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "hits": {
      "items": {
        "$ref": "#/components/schemas/FeedSearchHitModel"
      },
      "title": "Hits",
      "type": "array"
    }
  },
  "required": [
    "hits"
  ],
  "title": "FeedSearchResponseModel",
  "type": "object"
}
```

</details>

<a id="model-getrepofilerequest"></a>
### Model: GetRepoFileRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `commit_sha` |  |  | req | `string` |
| `end_line` |  |  | opt | `integer` (nullable) |
| `path` |  |  | req | `string` |
| `repo_url` |  |  | req | `string` |
| `start_line` |  |  | opt | `integer` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "description": "Query parameters for the platform repo file callback.",
  "properties": {
    "commit_sha": {
      "pattern": "^[0-9a-f]{40}$",
      "title": "Commit Sha",
      "type": "string"
    },
    "end_line": {
      "anyOf": [
        {
          "minimum": 1.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "End Line"
    },
    "path": {
      "title": "Path",
      "type": "string"
    },
    "repo_url": {
      "title": "Repo Url",
      "type": "string"
    },
    "start_line": {
      "anyOf": [
        {
          "minimum": 1.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Start Line"
    }
  },
  "required": [
    "repo_url",
    "commit_sha",
    "path"
  ],
  "title": "GetRepoFileRequest",
  "type": "object"
}
```

</details>

<a id="model-getrepofileresponse"></a>
### Model: GetRepoFileResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `data` |  |  | opt | array[[GetRepoFileResult](#model-getrepofileresult)] |
|  | `excerpt` |  | opt | `string` (nullable) |
|  | `path` |  | req | `string` |
|  | `text` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "description": "Response payload for the platform repo file callback.",
  "properties": {
    "data": {
      "items": {
        "$ref": "#/components/schemas/GetRepoFileResult"
      },
      "title": "Data",
      "type": "array"
    }
  },
  "title": "GetRepoFileResponse",
  "type": "object"
}
```

</details>

<a id="model-getrepofileresult"></a>
### Model: GetRepoFileResult

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `excerpt` |  |  | opt | `string` (nullable) |
| `path` |  |  | req | `string` |
| `text` |  |  | req | `string` |
| `title` |  |  | opt | `string` (nullable) |
| `url` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "description": "Single repository file response item.",
  "properties": {
    "excerpt": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Excerpt"
    },
    "path": {
      "title": "Path",
      "type": "string"
    },
    "text": {
      "title": "Text",
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
      "title": "Url",
      "type": "string"
    }
  },
  "required": [
    "path",
    "url",
    "text"
  ],
  "title": "GetRepoFileResult",
  "type": "object"
}
```

</details>

<a id="model-harnyx_miner_sdk__json_types__jsonvalue-input"></a>
### Model: harnyx_miner_sdk__json_types__JsonValue-Input

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
        "$ref": "#/components/schemas/harnyx_miner_sdk__json_types__JsonValue-Input"
      },
      "type": "array"
    },
    {
      "additionalProperties": {
        "$ref": "#/components/schemas/harnyx_miner_sdk__json_types__JsonValue-Input"
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

<a id="model-harnyx_miner_sdk__json_types__jsonvalue-output"></a>
### Model: harnyx_miner_sdk__json_types__JsonValue-Output

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
        "$ref": "#/components/schemas/harnyx_miner_sdk__json_types__JsonValue-Output"
      },
      "type": "array"
    },
    {
      "additionalProperties": {
        "$ref": "#/components/schemas/harnyx_miner_sdk__json_types__JsonValue-Output"
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

<a id="model-jsonobject-input"></a>
### Model: JsonObject-Input

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": {
    "$ref": "#/components/schemas/harnyx_miner_sdk__json_types__JsonValue-Input"
  },
  "type": "object"
}
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

<a id="model-llmmodelusagecost"></a>
### Model: LlmModelUsageCost

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `actual_cost` |  |  | opt | `number` (nullable) |
| `cost` |  |  | opt | `number` (default: 0.0) |
| `reference_cost` |  |  | opt | `number` (default: 0.0) |
| `usage` |  |  | opt | [LlmUsageTotals](#model-llmusagetotals) |
|  | `call_count` |  | opt | `integer` (default: 0) |
|  | `completion_tokens` |  | opt | `integer` (default: 0) |
|  | `prompt_tokens` |  | opt | `integer` (default: 0) |
|  | `reasoning_tokens` |  | opt | `integer` (default: 0) |
|  | `total_tokens` |  | opt | `integer` (default: 0) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "actual_cost": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Actual Cost"
    },
    "cost": {
      "default": 0.0,
      "title": "Cost",
      "type": "number"
    },
    "reference_cost": {
      "default": 0.0,
      "title": "Reference Cost",
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
| `actual_cost` |  |  | opt | `number` (nullable) |
| `call_count` |  |  | opt | `integer` (default: 0) |
| `completion_tokens` |  |  | opt | `integer` (default: 0) |
| `cost` |  |  | opt | `number` (default: 0.0) |
| `prompt_tokens` |  |  | opt | `integer` (default: 0) |
| `providers` |  |  | opt | `object` |
| `reasoning_tokens` |  |  | opt | `integer` (default: 0) |
| `reference_cost` |  |  | opt | `number` (default: 0.0) |
| `total_tokens` |  |  | opt | `integer` (default: 0) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "actual_cost": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Actual Cost"
    },
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
    "reasoning_tokens": {
      "default": 0,
      "title": "Reasoning Tokens",
      "type": "integer"
    },
    "reference_cost": {
      "default": 0.0,
      "title": "Reference Cost",
      "type": "number"
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
| `reasoning_tokens` |  |  | opt | `integer` (default: 0) |
| `total_tokens` |  |  | opt | `integer` (default: 0) |

<details>
<summary>JSON schema</summary>

```json
{
  "description": "Accumulated token usage for a single provider/model pair.",
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
    "reasoning_tokens": {
      "default": 0,
      "title": "Reasoning Tokens",
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

<a id="model-minerconfigdeleterequest"></a>
### Model: MinerConfigDeleteRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `key` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "key": {
      "title": "Key",
      "type": "string"
    }
  },
  "required": [
    "key"
  ],
  "title": "MinerConfigDeleteRequest",
  "type": "object"
}
```

</details>

<a id="model-minerconfigputrequest"></a>
### Model: MinerConfigPutRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `key` |  |  | req | `string` |
| `value` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "key": {
      "title": "Key",
      "type": "string"
    },
    "value": {
      "title": "Value",
      "type": "string"
    }
  },
  "required": [
    "key",
    "value"
  ],
  "title": "MinerConfigPutRequest",
  "type": "object"
}
```

</details>

<a id="model-minerconfigresponse"></a>
### Model: MinerConfigResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `miner_hotkey_ss58` |  |  | req | `string` |
| `provider_credentials` |  |  | req | `object` |
| `task_retry_count` |  |  | req | `integer` |
| `uid` |  |  | req | `integer` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "miner_hotkey_ss58": {
      "title": "Miner Hotkey Ss58",
      "type": "string"
    },
    "provider_credentials": {
      "additionalProperties": {
        "$ref": "#/components/schemas/MinerProviderCredentialStatusModel"
      },
      "title": "Provider Credentials",
      "type": "object"
    },
    "task_retry_count": {
      "title": "Task Retry Count",
      "type": "integer"
    },
    "uid": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Uid"
    }
  },
  "required": [
    "miner_hotkey_ss58",
    "uid",
    "task_retry_count",
    "provider_credentials"
  ],
  "title": "MinerConfigResponse",
  "type": "object"
}
```

</details>

<a id="model-minerprovidercredentialstatusmodel"></a>
### Model: MinerProviderCredentialStatusModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `created_at` |  |  | req | `string` (format: date-time; nullable) |
| `exists` |  |  | req | `boolean` |
| `provider` |  |  | req | `string` |
| `updated_at` |  |  | req | `string` (format: date-time; nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "created_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Created At"
    },
    "exists": {
      "title": "Exists",
      "type": "boolean"
    },
    "provider": {
      "title": "Provider",
      "type": "string"
    },
    "updated_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Updated At"
    }
  },
  "required": [
    "provider",
    "exists",
    "created_at",
    "updated_at"
  ],
  "title": "MinerProviderCredentialStatusModel",
  "type": "object"
}
```

</details>

<a id="model-minertask"></a>
### Model: MinerTask

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `budget_usd` |  |  | opt | `number` (default: 0.5) |
| `query` |  |  | req | [Query](#model-query) |
|  | `output_schema` |  | opt | [JsonObject-Input](#model-jsonobject-input) (nullable) |
|  | `text` |  | req | `string` |
| `reference_answer` |  |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  | `citations` |  | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  |  | `note` | opt | `string` (nullable) |
|  |  | `title` | opt | `string` (nullable) |
|  |  | `url` | req | `string` |
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
      "$ref": "#/components/schemas/Query"
    },
    "reference_answer": {
      "$ref": "#/components/schemas/ReferenceAnswer"
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

<a id="model-minertaskattemptauditpayload"></a>
### Model: MinerTaskAttemptAuditPayload

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `attempt_number` |  |  | req | `integer` |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `delivery_failure_detail` |  |  | opt | [FailureDetailPayload](#model-failuredetailpayload) (nullable) |
|  | `artifact_id` |  | opt | `string` (format: uuid; nullable) |
|  | `error_code` |  | req | `string` |
|  | `error_message` |  | req | `string` |
|  | `exception_type` |  | opt | `string` (nullable) |
|  | `occurred_at` |  | req | `string` (format: date-time) |
|  | `sandbox_diagnostics` |  | opt | [SandboxFailureDiagnostics](#model-sandboxfailurediagnostics) (nullable) |
|  |  | `container_id` | opt | `string` (nullable) |
|  |  | `container_name` | opt | `string` (nullable) |
|  |  | `docker_inspect_error_tail` | opt | `string` (nullable) |
|  |  | `docker_logs_error_tail` | opt | `string` (nullable) |
|  |  | `docker_logs_tail` | opt | `string` (nullable) |
|  |  | `error_text` | opt | `string` (nullable) |
|  |  | `exit_code` | opt | `integer` (nullable) |
|  |  | `image` | opt | `string` (nullable) |
|  |  | `oom_killed` | opt | `boolean` (nullable) |
|  |  | `pull_policy` | opt | `string` (nullable) |
|  |  | `pull_returncode` | opt | `integer` (nullable) |
|  |  | `pull_stderr_tail` | opt | `string` (nullable) |
|  |  | `pull_stdout_tail` | opt | `string` (nullable) |
|  |  | `run_returncode` | opt | `integer` (nullable) |
|  |  | `run_stderr_tail` | opt | `string` (nullable) |
|  |  | `run_stdout_tail` | opt | `string` (nullable) |
|  |  | `state_error` | opt | `string` (nullable) |
|  |  | `status` | opt | `string` (nullable) |
|  | `task_id` |  | opt | `string` (format: uuid; nullable) |
|  | `traceback` |  | opt | `string` (nullable) |
|  | `uid` |  | opt | `integer` (nullable) |
| `diagnostics` |  |  | opt | [MinerTaskAttemptDiagnosticsPayload](#model-minertaskattemptdiagnosticspayload) (nullable) |
|  | `elapsed_ms` |  | opt | `number` (nullable) |
|  | `failure_owner` |  | opt | `string` (nullable) |
|  | `phase` |  | req | `string` |
|  | `platform_tool_activity_observed` |  | opt | `boolean` (default: False) |
|  | `timeout_owner` |  | opt | `string` (nullable) |
| `error_code` |  |  | opt | `string` (nullable) |
| `error_summary_code` |  |  | opt | `string` (nullable) |
| `execution_log` |  |  | opt | array[[ToolCall-Input](#model-toolcall-input)] (default: []) |
|  | `details` |  | req | [ToolCallDetails-Input](#model-toolcalldetails-input) |
|  |  | `actual_cost_provider` | opt | `string` (nullable) |
|  |  | `actual_cost_usd` | opt | `number` (nullable) |
|  |  | `cost_usd` | opt | `number` (nullable) |
|  |  | `execution` | opt | [ToolExecutionFacts](#model-toolexecutionfacts) (nullable) |
|  |  | `extra` | opt | `object` (nullable) |
|  |  | `reference_cost_usd` | opt | `number` (nullable) |
|  |  | `request_hash` | req | `string` |
|  |  | `request_payload` | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  |  | `response_hash` | opt | `string` (nullable) |
|  |  | `response_payload` | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  |  | `result_policy` | opt | [ToolResultPolicy](#model-toolresultpolicy) (default: log_only) |
|  |  | `results` | opt | array[[ToolResult-Input](#model-toolresult-input)] (default: []) |
|  | `issued_at` |  | req | `string` (format: date-time) |
|  | `outcome` |  | req | [ToolCallOutcome](#model-toolcalloutcome) |
|  | `receipt_id` |  | req | `string` |
|  | `session_id` |  | req | `string` (format: uuid) |
|  | `tool` |  | req | `string` (enum: [search_web, search_ai, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |
|  | `uid` |  | req | `integer` |
| `finished_at` |  |  | req | `string` (format: date-time) |
| `max_attempts` |  |  | req | `integer` |
| `miner_hotkey_ss58` |  |  | req | `string` |
| `retry_decision` |  |  | req | [MinerTaskAttemptRetryDecision](#model-minertaskattemptretrydecision) |
| `started_at` |  |  | req | `string` (format: date-time) |
| `status` |  |  | req | [MinerTaskAttemptStatus](#model-minertaskattemptstatus) |
| `task_id` |  |  | req | `string` (format: uuid) |
| `terminal_effect` |  |  | req | [MinerTaskAttemptTerminalEffect](#model-minertaskattemptterminaleffect) (nullable) |
| `uid` |  |  | req | `integer` |
| `validator_session_id` |  |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "artifact_id": {
      "format": "uuid",
      "title": "Artifact Id",
      "type": "string"
    },
    "attempt_number": {
      "minimum": 1.0,
      "title": "Attempt Number",
      "type": "integer"
    },
    "batch_id": {
      "format": "uuid",
      "title": "Batch Id",
      "type": "string"
    },
    "delivery_failure_detail": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/FailureDetailPayload"
        },
        {
          "type": "null"
        }
      ]
    },
    "diagnostics": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/MinerTaskAttemptDiagnosticsPayload"
        },
        {
          "type": "null"
        }
      ]
    },
    "error_code": {
      "anyOf": [
        {
          "maxLength": 128,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Error Code"
    },
    "error_summary_code": {
      "anyOf": [
        {
          "maxLength": 128,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Error Summary Code"
    },
    "execution_log": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/ToolCall-Input"
      },
      "title": "Execution Log",
      "type": "array"
    },
    "finished_at": {
      "format": "date-time",
      "title": "Finished At",
      "type": "string"
    },
    "max_attempts": {
      "minimum": 1.0,
      "title": "Max Attempts",
      "type": "integer"
    },
    "miner_hotkey_ss58": {
      "minLength": 1,
      "title": "Miner Hotkey Ss58",
      "type": "string"
    },
    "retry_decision": {
      "$ref": "#/components/schemas/MinerTaskAttemptRetryDecision"
    },
    "started_at": {
      "format": "date-time",
      "title": "Started At",
      "type": "string"
    },
    "status": {
      "$ref": "#/components/schemas/MinerTaskAttemptStatus"
    },
    "task_id": {
      "format": "uuid",
      "title": "Task Id",
      "type": "string"
    },
    "terminal_effect": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/MinerTaskAttemptTerminalEffect"
        },
        {
          "type": "null"
        }
      ]
    },
    "uid": {
      "minimum": 0.0,
      "title": "Uid",
      "type": "integer"
    },
    "validator_session_id": {
      "format": "uuid",
      "title": "Validator Session Id",
      "type": "string"
    }
  },
  "required": [
    "validator_session_id",
    "batch_id",
    "artifact_id",
    "task_id",
    "attempt_number",
    "uid",
    "miner_hotkey_ss58",
    "started_at",
    "finished_at",
    "status",
    "retry_decision",
    "terminal_effect",
    "max_attempts"
  ],
  "title": "MinerTaskAttemptAuditPayload",
  "type": "object"
}
```

</details>

<a id="model-minertaskattemptdiagnosticspayload"></a>
### Model: MinerTaskAttemptDiagnosticsPayload

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `elapsed_ms` |  |  | opt | `number` (nullable) |
| `failure_owner` |  |  | opt | `string` (nullable) |
| `phase` |  |  | req | `string` |
| `platform_tool_activity_observed` |  |  | opt | `boolean` (default: False) |
| `timeout_owner` |  |  | opt | `string` (nullable) |

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
    "failure_owner": {
      "anyOf": [
        {
          "maxLength": 128,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Failure Owner"
    },
    "phase": {
      "maxLength": 128,
      "minLength": 1,
      "title": "Phase",
      "type": "string"
    },
    "platform_tool_activity_observed": {
      "default": false,
      "title": "Platform Tool Activity Observed",
      "type": "boolean"
    },
    "timeout_owner": {
      "anyOf": [
        {
          "maxLength": 128,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Timeout Owner"
    }
  },
  "required": [
    "phase"
  ],
  "title": "MinerTaskAttemptDiagnosticsPayload",
  "type": "object"
}
```

</details>

<a id="model-minertaskattemptidentitypayload"></a>
### Model: MinerTaskAttemptIdentityPayload

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `attempt_number` |  |  | req | `integer` |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `task_id` |  |  | req | `string` (format: uuid) |
| `validator_session_id` |  |  | opt | `string` (format: uuid; nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "artifact_id": {
      "format": "uuid",
      "title": "Artifact Id",
      "type": "string"
    },
    "attempt_number": {
      "minimum": 1.0,
      "title": "Attempt Number",
      "type": "integer"
    },
    "batch_id": {
      "format": "uuid",
      "title": "Batch Id",
      "type": "string"
    },
    "task_id": {
      "format": "uuid",
      "title": "Task Id",
      "type": "string"
    },
    "validator_session_id": {
      "anyOf": [
        {
          "format": "uuid",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Validator Session Id"
    }
  },
  "required": [
    "batch_id",
    "artifact_id",
    "task_id",
    "attempt_number"
  ],
  "title": "MinerTaskAttemptIdentityPayload",
  "type": "object"
}
```

</details>

<a id="model-minertaskattemptretrydecision"></a>
### Model: MinerTaskAttemptRetryDecision

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "enum": [
    "will_retry",
    "will_not_retry"
  ],
  "title": "MinerTaskAttemptRetryDecision",
  "type": "string"
}
```

</details>

<a id="model-minertaskattemptstatus"></a>
### Model: MinerTaskAttemptStatus

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "enum": [
    "succeeded",
    "failed"
  ],
  "title": "MinerTaskAttemptStatus",
  "type": "string"
}
```

</details>

<a id="model-minertaskattemptterminaleffect"></a>
### Model: MinerTaskAttemptTerminalEffect

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "enum": [
    "task_result",
    "delivery_failure",
    "attempt_failure"
  ],
  "title": "MinerTaskAttemptTerminalEffect",
  "type": "string"
}
```

</details>

<a id="model-minertaskbatchartifactmodel"></a>
### Model: MinerTaskBatchArtifactModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `content_hash` |  |  | req | `string` |
| `miner_hotkey_ss58` |  |  | req | `string` |
| `size_bytes` |  |  | req | `integer` |
| `submitted_at` |  |  | req | `string` (format: date-time) |
| `task_retry_count` |  |  | req | `integer` |
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
    "miner_hotkey_ss58": {
      "title": "Miner Hotkey Ss58",
      "type": "string"
    },
    "size_bytes": {
      "title": "Size Bytes",
      "type": "integer"
    },
    "submitted_at": {
      "format": "date-time",
      "title": "Submitted At",
      "type": "string"
    },
    "task_retry_count": {
      "title": "Task Retry Count",
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
    "size_bytes",
    "submitted_at",
    "miner_hotkey_ss58",
    "task_retry_count"
  ],
  "title": "MinerTaskBatchArtifactModel",
  "type": "object"
}
```

</details>

<a id="model-minertaskbatchmodel"></a>
### Model: MinerTaskBatchModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifacts` |  |  | req | array[[MinerTaskBatchArtifactModel](#model-minertaskbatchartifactmodel)] |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `content_hash` |  | req | `string` |
|  | `miner_hotkey_ss58` |  | req | `string` |
|  | `size_bytes` |  | req | `integer` |
|  | `submitted_at` |  | req | `string` (format: date-time) |
|  | `task_retry_count` |  | req | `integer` |
|  | `uid` |  | req | `integer` |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `champion_artifact_id` |  |  | req | `string` (format: uuid; nullable) |
| `completed_at` |  |  | opt | `string` (format: date-time; nullable) |
| `created_at` |  |  | req | `string` (format: date-time) |
| `cutoff_at` |  |  | req | `string` (format: date-time) |
| `failed_at` |  |  | opt | `string` (format: date-time; nullable) |
| `tasks` |  |  | req | array[[MinerTask](#model-minertask)] |
|  | `budget_usd` |  | opt | `number` (default: 0.5) |
|  | `query` |  | req | [Query](#model-query) |
|  |  | `output_schema` | opt | [JsonObject-Input](#model-jsonobject-input) (nullable) |
|  |  | `text` | req | `string` |
|  | `reference_answer` |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `citations` | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  |  | `text` | req | `string` |
|  | `task_id` |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "artifacts": {
      "items": {
        "$ref": "#/components/schemas/MinerTaskBatchArtifactModel"
      },
      "title": "Artifacts",
      "type": "array"
    },
    "batch_id": {
      "format": "uuid",
      "title": "Batch Id",
      "type": "string"
    },
    "champion_artifact_id": {
      "anyOf": [
        {
          "format": "uuid",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Champion Artifact Id"
    },
    "completed_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Completed At"
    },
    "created_at": {
      "format": "date-time",
      "title": "Created At",
      "type": "string"
    },
    "cutoff_at": {
      "format": "date-time",
      "title": "Cutoff At",
      "type": "string"
    },
    "failed_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Failed At"
    },
    "tasks": {
      "items": {
        "$ref": "#/components/schemas/MinerTask"
      },
      "title": "Tasks",
      "type": "array"
    }
  },
  "required": [
    "batch_id",
    "cutoff_at",
    "created_at",
    "tasks",
    "artifacts",
    "champion_artifact_id"
  ],
  "title": "MinerTaskBatchModel",
  "type": "object"
}
```

</details>

<a id="model-minertaskerrorcode"></a>
### Model: MinerTaskErrorCode

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "enum": [
    "artifact_breaker_tripped",
    "artifact_fetch_failed",
    "artifact_hash_mismatch",
    "artifact_setup_failed",
    "artifact_size_invalid",
    "artifact_staging_failed",
    "batch_execution_failed",
    "miner_response_invalid",
    "miner_unhandled_exception",
    "never_ran",
    "progress_snapshot_failed",
    "provider_batch_failure",
    "sandbox_failed",
    "sandbox_invocation_failed",
    "sandbox_start_failed",
    "scoring_llm_retry_exhausted",
    "script_validation_failed",
    "session_budget_exhausted",
    "timeout_inconclusive",
    "timeout_miner_owned",
    "tool_provider_failed",
    "unexpected_validator_failure",
    "validator_failed",
    "validator_internal_timeout",
    "validator_timeout"
  ],
  "title": "MinerTaskErrorCode",
  "type": "string"
}
```

</details>

<a id="model-minertaskinputmodel"></a>
### Model: MinerTaskInputModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `budget_usd` |  |  | opt | `number` (default: 0.5) |
| `query` |  |  | req | [Query](#model-query) |
|  | `output_schema` |  | opt | [JsonObject-Input](#model-jsonobject-input) (nullable) |
|  | `text` |  | req | `string` |
| `reference_answer` |  |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  | `citations` |  | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  |  | `note` | opt | `string` (nullable) |
|  |  | `title` | opt | `string` (nullable) |
|  |  | `url` | req | `string` |
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
      "$ref": "#/components/schemas/Query"
    },
    "reference_answer": {
      "$ref": "#/components/schemas/ReferenceAnswer"
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
  "title": "MinerTaskInputModel",
  "type": "object"
}
```

</details>

<a id="model-minertaskresultoutcome"></a>
### Model: MinerTaskResultOutcome

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "enum": [
    "accepted",
    "rejected"
  ],
  "title": "MinerTaskResultOutcome",
  "type": "string"
}
```

</details>

<a id="model-minertaskresultreasoncode"></a>
### Model: MinerTaskResultReasonCode

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "enum": [
    "already_accepted",
    "stale_attempt",
    "conflicting_replay",
    "invalid_attempt",
    "invalid_state"
  ],
  "title": "MinerTaskResultReasonCode",
  "type": "string"
}
```

</details>

<a id="model-minertaskrunrequest"></a>
### Model: MinerTaskRunRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `execution_log` |  |  | opt | array[[ToolCall-Input](#model-toolcall-input)] (default: []) |
|  | `details` |  | req | [ToolCallDetails-Input](#model-toolcalldetails-input) |
|  |  | `actual_cost_provider` | opt | `string` (nullable) |
|  |  | `actual_cost_usd` | opt | `number` (nullable) |
|  |  | `cost_usd` | opt | `number` (nullable) |
|  |  | `execution` | opt | [ToolExecutionFacts](#model-toolexecutionfacts) (nullable) |
|  |  | `extra` | opt | `object` (nullable) |
|  |  | `reference_cost_usd` | opt | `number` (nullable) |
|  |  | `request_hash` | req | `string` |
|  |  | `request_payload` | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  |  | `response_hash` | opt | `string` (nullable) |
|  |  | `response_payload` | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  |  | `result_policy` | opt | [ToolResultPolicy](#model-toolresultpolicy) (default: log_only) |
|  |  | `results` | opt | array[[ToolResult-Input](#model-toolresult-input)] (default: []) |
|  | `issued_at` |  | req | `string` (format: date-time) |
|  | `outcome` |  | req | [ToolCallOutcome](#model-toolcalloutcome) |
|  | `receipt_id` |  | req | `string` |
|  | `session_id` |  | req | `string` (format: uuid) |
|  | `tool` |  | req | `string` (enum: [search_web, search_ai, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |
|  | `uid` |  | req | `integer` |
| `run` |  |  | req | [MinerTaskRunSection](#model-minertaskrunsection) |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `completed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `response` |  | opt | [Response](#model-response) (nullable) |
|  |  | `citations` | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  |  | `output` | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  |  | `text` | opt | `string` (nullable) |
|  | `task_id` |  | req | `string` (format: uuid) |
| `score` |  |  | opt | `number` (nullable) |
| `session` |  |  | req | [SessionModel](#model-sessionmodel) |
|  | `expires_at` |  | req | `string` |
|  | `issued_at` |  | req | `string` |
|  | `session_id` |  | req | `string` (format: uuid) |
|  | `status` |  | req | `string` |
|  | `uid` |  | req | `integer` |
| `specifics` |  |  | req | [EvaluationDetails](#model-evaluationdetails) |
|  | `elapsed_ms` |  | opt | `number` (nullable) |
|  | `error` |  | opt | [EvaluationError](#model-evaluationerror) (nullable) |
|  |  | `code` | req | [MinerTaskErrorCode](#model-minertaskerrorcode) |
|  |  | `message` | req | `string` |
|  | `score_breakdown` |  | opt | [ScoreBreakdown](#model-scorebreakdown) (nullable) |
|  |  | `comparison_score` | req | `number` |
|  |  | `reasoning` | opt | [ScorerReasoning](#model-scorerreasoning) (nullable) |
|  |  | `scoring_version` | req | `string` |
|  |  | `total_score` | req | `number` |
|  | `scoring_judge_usage` |  | opt | [JudgeUsageSummary](#model-judgeusagesummary) (nullable) |
|  |  | `actual_cost_usd` | req | `number` (nullable) |
|  |  | `call_count` | req | `integer` |
|  |  | `completion_tokens` | req | `integer` |
|  |  | `models` | req | array[[JudgeModelUsage](#model-judgemodelusage)] |
|  |  | `prompt_tokens` | req | `integer` |
|  |  | `reasoning_tokens` | req | `integer` |
|  |  | `total_tokens` | req | `integer` |
|  | `total_tool_usage` |  | opt | [ToolUsageSummary](#model-toolusagesummary) |
|  |  | `actual_cost_by_provider` | opt | `object` |
|  |  | `actual_total_cost_usd` | opt | `number` (nullable) |
|  |  | `embedding` | opt | [EmbeddingToolUsageSummary](#model-embeddingtoolusagesummary) |
|  |  | `embedding_cost` | opt | `number` (default: 0.0) |
|  |  | `llm` | opt | [LlmUsageSummary](#model-llmusagesummary) |
|  |  | `llm_cost` | opt | `number` (default: 0.0) |
|  |  | `reference_cost_by_provider` | opt | `object` |
|  |  | `reference_total_cost_usd` | opt | `number` (default: 0.0) |
|  |  | `search_tool` | opt | [SearchToolUsageSummary](#model-searchtoolusagesummary) |
|  |  | `search_tool_cost` | opt | `number` (default: 0.0) |
|  | `trace` |  | opt | [EvaluationTrace](#model-evaluationtrace) (nullable) |
|  |  | `entrypoint_invocation_ms` | opt | `number` (nullable) |
|  |  | `orchestration_ms` | opt | `number` (nullable) |
|  |  | `scoring_judge_attempt_count` | opt | `integer` (nullable) |
|  |  | `scoring_judge_duration_ms` | opt | `number` (nullable) |
|  |  | `scoring_judge_retry_count` | opt | `integer` (nullable) |
|  |  | `scoring_judge_retry_reasons` | opt | array[`string`] (default: []) |
|  |  | `scoring_judge_selected_routes` | opt | array[`string`] (default: []) |
|  |  | `scoring_judge_status` | opt | `string` (enum: [ok, exhausted, failed]; nullable) |
|  |  | `scoring_ms` | opt | `number` (nullable) |
| `usage` |  |  | req | [UsageModel](#model-usagemodel) |
|  | `by_provider` |  | opt | `object` |
|  | `call_count` |  | req | `integer` |
|  | `total_completion_tokens` |  | req | `integer` |
|  | `total_prompt_tokens` |  | req | `integer` |
|  | `total_tokens` |  | req | `integer` |

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
    "execution_log": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/ToolCall-Input"
      },
      "title": "Execution Log",
      "type": "array"
    },
    "run": {
      "$ref": "#/components/schemas/MinerTaskRunSection"
    },
    "score": {
      "anyOf": [
        {
          "maximum": 1.0,
          "minimum": 0.0,
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Score"
    },
    "session": {
      "$ref": "#/components/schemas/SessionModel"
    },
    "specifics": {
      "$ref": "#/components/schemas/EvaluationDetails"
    },
    "usage": {
      "$ref": "#/components/schemas/UsageModel"
    }
  },
  "required": [
    "batch_id",
    "run",
    "usage",
    "session",
    "specifics"
  ],
  "title": "MinerTaskRunRequest",
  "type": "object"
}
```

</details>

<a id="model-minertaskrunsection"></a>
### Model: MinerTaskRunSection

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `completed_at` |  |  | opt | `string` (format: date-time; nullable) |
| `response` |  |  | opt | [Response](#model-response) (nullable) |
|  | `citations` |  | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  |  | `note` | opt | `string` (nullable) |
|  |  | `title` | opt | `string` (nullable) |
|  |  | `url` | req | `string` |
|  | `output` |  | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  | `text` |  | opt | `string` (nullable) |
| `task_id` |  |  | req | `string` (format: uuid) |

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
    "completed_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Completed At"
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
      "format": "uuid",
      "title": "Task Id",
      "type": "string"
    }
  },
  "required": [
    "artifact_id",
    "task_id"
  ],
  "title": "MinerTaskRunSection",
  "type": "object"
}
```

</details>

<a id="model-minertaskworkartifactpayload"></a>
### Model: MinerTaskWorkArtifactPayload

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `content_hash` |  |  | req | `string` |
| `miner_hotkey_ss58` |  |  | opt | `string` (nullable) |
| `size_bytes` |  |  | req | `integer` |
| `uid` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "artifact_id": {
      "format": "uuid",
      "title": "Artifact Id",
      "type": "string"
    },
    "content_hash": {
      "minLength": 1,
      "title": "Content Hash",
      "type": "string"
    },
    "miner_hotkey_ss58": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Miner Hotkey Ss58"
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
    "artifact_id",
    "uid",
    "content_hash",
    "size_bytes"
  ],
  "title": "MinerTaskWorkArtifactPayload",
  "type": "object"
}
```

</details>

<a id="model-minertaskworkassignmentpayload"></a>
### Model: MinerTaskWorkAssignmentPayload

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact` |  |  | req | [MinerTaskWorkArtifactPayload](#model-minertaskworkartifactpayload) |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `content_hash` |  | req | `string` |
|  | `miner_hotkey_ss58` |  | opt | `string` (nullable) |
|  | `size_bytes` |  | req | `integer` |
|  | `uid` |  | req | `integer` |
| `assignment_token` |  |  | req | `string` |
| `attempt_number` |  |  | req | `integer` |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `max_attempts` |  |  | req | `integer` |
| `task` |  |  | req | [MinerTask](#model-minertask) |
|  | `budget_usd` |  | opt | `number` (default: 0.5) |
|  | `query` |  | req | [Query](#model-query) |
|  |  | `output_schema` | opt | [JsonObject-Input](#model-jsonobject-input) (nullable) |
|  |  | `text` | req | `string` |
|  | `reference_answer` |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `citations` | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  |  | `text` | req | `string` |
|  | `task_id` |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "artifact": {
      "$ref": "#/components/schemas/MinerTaskWorkArtifactPayload"
    },
    "assignment_token": {
      "minLength": 1,
      "title": "Assignment Token",
      "type": "string"
    },
    "attempt_number": {
      "minimum": 1.0,
      "title": "Attempt Number",
      "type": "integer"
    },
    "batch_id": {
      "format": "uuid",
      "title": "Batch Id",
      "type": "string"
    },
    "max_attempts": {
      "minimum": 1.0,
      "title": "Max Attempts",
      "type": "integer"
    },
    "task": {
      "$ref": "#/components/schemas/MinerTask"
    }
  },
  "required": [
    "batch_id",
    "artifact",
    "task",
    "attempt_number",
    "max_attempts",
    "assignment_token"
  ],
  "title": "MinerTaskWorkAssignmentPayload",
  "type": "object"
}
```

</details>

<a id="model-minertaskworkexecutionenvelope"></a>
### Model: MinerTaskWorkExecutionEnvelope

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `attempt_number` |  |  | req | `integer` |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `execution_completed_at` |  |  | req | `string` (format: date-time) |
| `execution_log` |  |  | opt | array[[ToolCall-Input](#model-toolcall-input)] (default: []) |
|  | `details` |  | req | [ToolCallDetails-Input](#model-toolcalldetails-input) |
|  |  | `actual_cost_provider` | opt | `string` (nullable) |
|  |  | `actual_cost_usd` | opt | `number` (nullable) |
|  |  | `cost_usd` | opt | `number` (nullable) |
|  |  | `execution` | opt | [ToolExecutionFacts](#model-toolexecutionfacts) (nullable) |
|  |  | `extra` | opt | `object` (nullable) |
|  |  | `reference_cost_usd` | opt | `number` (nullable) |
|  |  | `request_hash` | req | `string` |
|  |  | `request_payload` | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  |  | `response_hash` | opt | `string` (nullable) |
|  |  | `response_payload` | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  |  | `result_policy` | opt | [ToolResultPolicy](#model-toolresultpolicy) (default: log_only) |
|  |  | `results` | opt | array[[ToolResult-Input](#model-toolresult-input)] (default: []) |
|  | `issued_at` |  | req | `string` (format: date-time) |
|  | `outcome` |  | req | [ToolCallOutcome](#model-toolcalloutcome) |
|  | `receipt_id` |  | req | `string` |
|  | `session_id` |  | req | `string` (format: uuid) |
|  | `tool` |  | req | `string` (enum: [search_web, search_ai, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |
|  | `uid` |  | req | `integer` |
| `miner_hotkey_ss58` |  |  | req | `string` |
| `response` |  |  | req | [Response](#model-response) |
|  | `citations` |  | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  |  | `note` | opt | `string` (nullable) |
|  |  | `title` | opt | `string` (nullable) |
|  |  | `url` | req | `string` |
|  | `output` |  | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  | `text` |  | opt | `string` (nullable) |
| `session` |  |  | req | [SessionModel](#model-sessionmodel) |
|  | `expires_at` |  | req | `string` |
|  | `issued_at` |  | req | `string` |
|  | `session_id` |  | req | `string` (format: uuid) |
|  | `status` |  | req | `string` |
|  | `uid` |  | req | `integer` |
| `started_at` |  |  | req | `string` (format: date-time) |
| `task_id` |  |  | req | `string` (format: uuid) |
| `total_tool_usage` |  |  | req | [ToolUsageSummary](#model-toolusagesummary) |
|  | `actual_cost_by_provider` |  | opt | `object` |
|  | `actual_total_cost_usd` |  | opt | `number` (nullable) |
|  | `embedding` |  | opt | [EmbeddingToolUsageSummary](#model-embeddingtoolusagesummary) |
|  |  | `actual_cost` | opt | `number` (nullable) |
|  |  | `call_count` | opt | `integer` (default: 0) |
|  |  | `cost` | opt | `number` (default: 0.0) |
|  |  | `reference_cost` | opt | `number` (default: 0.0) |
|  | `embedding_cost` |  | opt | `number` (default: 0.0) |
|  | `llm` |  | opt | [LlmUsageSummary](#model-llmusagesummary) |
|  |  | `actual_cost` | opt | `number` (nullable) |
|  |  | `call_count` | opt | `integer` (default: 0) |
|  |  | `completion_tokens` | opt | `integer` (default: 0) |
|  |  | `cost` | opt | `number` (default: 0.0) |
|  |  | `prompt_tokens` | opt | `integer` (default: 0) |
|  |  | `providers` | opt | `object` |
|  |  | `reasoning_tokens` | opt | `integer` (default: 0) |
|  |  | `reference_cost` | opt | `number` (default: 0.0) |
|  |  | `total_tokens` | opt | `integer` (default: 0) |
|  | `llm_cost` |  | opt | `number` (default: 0.0) |
|  | `reference_cost_by_provider` |  | opt | `object` |
|  | `reference_total_cost_usd` |  | opt | `number` (default: 0.0) |
|  | `search_tool` |  | opt | [SearchToolUsageSummary](#model-searchtoolusagesummary) |
|  |  | `actual_cost` | opt | `number` (nullable) |
|  |  | `call_count` | opt | `integer` (default: 0) |
|  |  | `cost` | opt | `number` (default: 0.0) |
|  |  | `reference_cost` | opt | `number` (default: 0.0) |
|  | `search_tool_cost` |  | opt | `number` (default: 0.0) |
| `trace` |  |  | opt | [EvaluationTrace](#model-evaluationtrace) (nullable) |
|  | `entrypoint_invocation_ms` |  | opt | `number` (nullable) |
|  | `orchestration_ms` |  | opt | `number` (nullable) |
|  | `scoring_judge_attempt_count` |  | opt | `integer` (nullable) |
|  | `scoring_judge_duration_ms` |  | opt | `number` (nullable) |
|  | `scoring_judge_retry_count` |  | opt | `integer` (nullable) |
|  | `scoring_judge_retry_reasons` |  | opt | array[`string`] (default: []) |
|  | `scoring_judge_selected_routes` |  | opt | array[`string`] (default: []) |
|  | `scoring_judge_status` |  | opt | `string` (enum: [ok, exhausted, failed]; nullable) |
|  | `scoring_ms` |  | opt | `number` (nullable) |
| `uid` |  |  | req | `integer` |
| `usage` |  |  | req | [UsageModel](#model-usagemodel) |
|  | `by_provider` |  | opt | `object` |
|  | `call_count` |  | req | `integer` |
|  | `total_completion_tokens` |  | req | `integer` |
|  | `total_prompt_tokens` |  | req | `integer` |
|  | `total_tokens` |  | req | `integer` |
| `validator_session_id` |  |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "artifact_id": {
      "format": "uuid",
      "title": "Artifact Id",
      "type": "string"
    },
    "attempt_number": {
      "minimum": 1.0,
      "title": "Attempt Number",
      "type": "integer"
    },
    "batch_id": {
      "format": "uuid",
      "title": "Batch Id",
      "type": "string"
    },
    "execution_completed_at": {
      "format": "date-time",
      "title": "Execution Completed At",
      "type": "string"
    },
    "execution_log": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/ToolCall-Input"
      },
      "title": "Execution Log",
      "type": "array"
    },
    "miner_hotkey_ss58": {
      "minLength": 1,
      "title": "Miner Hotkey Ss58",
      "type": "string"
    },
    "response": {
      "$ref": "#/components/schemas/Response"
    },
    "session": {
      "$ref": "#/components/schemas/SessionModel"
    },
    "started_at": {
      "format": "date-time",
      "title": "Started At",
      "type": "string"
    },
    "task_id": {
      "format": "uuid",
      "title": "Task Id",
      "type": "string"
    },
    "total_tool_usage": {
      "$ref": "#/components/schemas/ToolUsageSummary"
    },
    "trace": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/EvaluationTrace"
        },
        {
          "type": "null"
        }
      ]
    },
    "uid": {
      "minimum": 0.0,
      "title": "Uid",
      "type": "integer"
    },
    "usage": {
      "$ref": "#/components/schemas/UsageModel"
    },
    "validator_session_id": {
      "format": "uuid",
      "title": "Validator Session Id",
      "type": "string"
    }
  },
  "required": [
    "batch_id",
    "artifact_id",
    "task_id",
    "attempt_number",
    "validator_session_id",
    "uid",
    "miner_hotkey_ss58",
    "started_at",
    "execution_completed_at",
    "response",
    "session",
    "usage",
    "total_tool_usage"
  ],
  "title": "MinerTaskWorkExecutionEnvelope",
  "type": "object"
}
```

</details>

<a id="model-minertaskworkexecutionitemresponse"></a>
### Model: MinerTaskWorkExecutionItemResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `attempt_number` |  |  | req | `integer` |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `canonical` |  |  | req | `boolean` |
| `outcome` |  |  | req | [MinerTaskResultOutcome](#model-minertaskresultoutcome) |
| `reason` |  |  | opt | `string` (nullable) |
| `reason_code` |  |  | opt | [MinerTaskResultReasonCode](#model-minertaskresultreasoncode) (nullable) |
| `task_id` |  |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "artifact_id": {
      "format": "uuid",
      "title": "Artifact Id",
      "type": "string"
    },
    "attempt_number": {
      "minimum": 1.0,
      "title": "Attempt Number",
      "type": "integer"
    },
    "batch_id": {
      "format": "uuid",
      "title": "Batch Id",
      "type": "string"
    },
    "canonical": {
      "title": "Canonical",
      "type": "boolean"
    },
    "outcome": {
      "$ref": "#/components/schemas/MinerTaskResultOutcome"
    },
    "reason": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Reason"
    },
    "reason_code": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/MinerTaskResultReasonCode"
        },
        {
          "type": "null"
        }
      ]
    },
    "task_id": {
      "format": "uuid",
      "title": "Task Id",
      "type": "string"
    }
  },
  "required": [
    "batch_id",
    "artifact_id",
    "task_id",
    "attempt_number",
    "outcome",
    "canonical"
  ],
  "title": "MinerTaskWorkExecutionItemResponse",
  "type": "object"
}
```

</details>

<a id="model-minertaskworkexecutionsrequest"></a>
### Model: MinerTaskWorkExecutionsRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `executions` |  |  | opt | array[[MinerTaskWorkExecutionEnvelope](#model-minertaskworkexecutionenvelope)] (default: []) |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `execution_completed_at` |  | req | `string` (format: date-time) |
|  | `execution_log` |  | opt | array[[ToolCall-Input](#model-toolcall-input)] (default: []) |
|  |  | `details` | req | [ToolCallDetails-Input](#model-toolcalldetails-input) |
|  |  | `issued_at` | req | `string` (format: date-time) |
|  |  | `outcome` | req | [ToolCallOutcome](#model-toolcalloutcome) |
|  |  | `receipt_id` | req | `string` |
|  |  | `session_id` | req | `string` (format: uuid) |
|  |  | `tool` | req | `string` (enum: [search_web, search_ai, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |
|  |  | `uid` | req | `integer` |
|  | `miner_hotkey_ss58` |  | req | `string` |
|  | `response` |  | req | [Response](#model-response) |
|  |  | `citations` | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  |  | `output` | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  |  | `text` | opt | `string` (nullable) |
|  | `session` |  | req | [SessionModel](#model-sessionmodel) |
|  |  | `expires_at` | req | `string` |
|  |  | `issued_at` | req | `string` |
|  |  | `session_id` | req | `string` (format: uuid) |
|  |  | `status` | req | `string` |
|  |  | `uid` | req | `integer` |
|  | `started_at` |  | req | `string` (format: date-time) |
|  | `task_id` |  | req | `string` (format: uuid) |
|  | `total_tool_usage` |  | req | [ToolUsageSummary](#model-toolusagesummary) |
|  |  | `actual_cost_by_provider` | opt | `object` |
|  |  | `actual_total_cost_usd` | opt | `number` (nullable) |
|  |  | `embedding` | opt | [EmbeddingToolUsageSummary](#model-embeddingtoolusagesummary) |
|  |  | `embedding_cost` | opt | `number` (default: 0.0) |
|  |  | `llm` | opt | [LlmUsageSummary](#model-llmusagesummary) |
|  |  | `llm_cost` | opt | `number` (default: 0.0) |
|  |  | `reference_cost_by_provider` | opt | `object` |
|  |  | `reference_total_cost_usd` | opt | `number` (default: 0.0) |
|  |  | `search_tool` | opt | [SearchToolUsageSummary](#model-searchtoolusagesummary) |
|  |  | `search_tool_cost` | opt | `number` (default: 0.0) |
|  | `trace` |  | opt | [EvaluationTrace](#model-evaluationtrace) (nullable) |
|  |  | `entrypoint_invocation_ms` | opt | `number` (nullable) |
|  |  | `orchestration_ms` | opt | `number` (nullable) |
|  |  | `scoring_judge_attempt_count` | opt | `integer` (nullable) |
|  |  | `scoring_judge_duration_ms` | opt | `number` (nullable) |
|  |  | `scoring_judge_retry_count` | opt | `integer` (nullable) |
|  |  | `scoring_judge_retry_reasons` | opt | array[`string`] (default: []) |
|  |  | `scoring_judge_selected_routes` | opt | array[`string`] (default: []) |
|  |  | `scoring_judge_status` | opt | `string` (enum: [ok, exhausted, failed]; nullable) |
|  |  | `scoring_ms` | opt | `number` (nullable) |
|  | `uid` |  | req | `integer` |
|  | `usage` |  | req | [UsageModel](#model-usagemodel) |
|  |  | `by_provider` | opt | `object` |
|  |  | `call_count` | req | `integer` |
|  |  | `total_completion_tokens` | req | `integer` |
|  |  | `total_prompt_tokens` | req | `integer` |
|  |  | `total_tokens` | req | `integer` |
|  | `validator_session_id` |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "executions": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/MinerTaskWorkExecutionEnvelope"
      },
      "title": "Executions",
      "type": "array"
    }
  },
  "title": "MinerTaskWorkExecutionsRequest",
  "type": "object"
}
```

</details>

<a id="model-minertaskworkexecutionsresponse"></a>
### Model: MinerTaskWorkExecutionsResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `executions` |  |  | opt | array[[MinerTaskWorkExecutionItemResponse](#model-minertaskworkexecutionitemresponse)] (default: []) |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `canonical` |  | req | `boolean` |
|  | `outcome` |  | req | [MinerTaskResultOutcome](#model-minertaskresultoutcome) |
|  | `reason` |  | opt | `string` (nullable) |
|  | `reason_code` |  | opt | [MinerTaskResultReasonCode](#model-minertaskresultreasoncode) (nullable) |
|  | `task_id` |  | req | `string` (format: uuid) |
| `server_time` |  |  | req | `string` (format: date-time) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "executions": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/MinerTaskWorkExecutionItemResponse"
      },
      "title": "Executions",
      "type": "array"
    },
    "server_time": {
      "format": "date-time",
      "title": "Server Time",
      "type": "string"
    }
  },
  "required": [
    "server_time"
  ],
  "title": "MinerTaskWorkExecutionsResponse",
  "type": "object"
}
```

</details>

<a id="model-minertaskworkresultenvelope"></a>
### Model: MinerTaskWorkResultEnvelope

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `attempt_number` |  |  | req | `integer` |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `result` |  |  | opt | [MinerTaskRunRequest](#model-minertaskrunrequest) (nullable) |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `execution_log` |  | opt | array[[ToolCall-Input](#model-toolcall-input)] (default: []) |
|  |  | `details` | req | [ToolCallDetails-Input](#model-toolcalldetails-input) |
|  |  | `issued_at` | req | `string` (format: date-time) |
|  |  | `outcome` | req | [ToolCallOutcome](#model-toolcalloutcome) |
|  |  | `receipt_id` | req | `string` |
|  |  | `session_id` | req | `string` (format: uuid) |
|  |  | `tool` | req | `string` (enum: [search_web, search_ai, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |
|  |  | `uid` | req | `integer` |
|  | `run` |  | req | [MinerTaskRunSection](#model-minertaskrunsection) |
|  |  | `artifact_id` | req | `string` (format: uuid) |
|  |  | `completed_at` | opt | `string` (format: date-time; nullable) |
|  |  | `response` | opt | [Response](#model-response) (nullable) |
|  |  | `task_id` | req | `string` (format: uuid) |
|  | `score` |  | opt | `number` (nullable) |
|  | `session` |  | req | [SessionModel](#model-sessionmodel) |
|  |  | `expires_at` | req | `string` |
|  |  | `issued_at` | req | `string` |
|  |  | `session_id` | req | `string` (format: uuid) |
|  |  | `status` | req | `string` |
|  |  | `uid` | req | `integer` |
|  | `specifics` |  | req | [EvaluationDetails](#model-evaluationdetails) |
|  |  | `elapsed_ms` | opt | `number` (nullable) |
|  |  | `error` | opt | [EvaluationError](#model-evaluationerror) (nullable) |
|  |  | `score_breakdown` | opt | [ScoreBreakdown](#model-scorebreakdown) (nullable) |
|  |  | `scoring_judge_usage` | opt | [JudgeUsageSummary](#model-judgeusagesummary) (nullable) |
|  |  | `total_tool_usage` | opt | [ToolUsageSummary](#model-toolusagesummary) |
|  |  | `trace` | opt | [EvaluationTrace](#model-evaluationtrace) (nullable) |
|  | `usage` |  | req | [UsageModel](#model-usagemodel) |
|  |  | `by_provider` | opt | `object` |
|  |  | `call_count` | req | `integer` |
|  |  | `total_completion_tokens` | req | `integer` |
|  |  | `total_prompt_tokens` | req | `integer` |
|  |  | `total_tokens` | req | `integer` |
| `task_id` |  |  | req | `string` (format: uuid) |
| `terminal_attempt` |  |  | req | [MinerTaskAttemptAuditPayload](#model-minertaskattemptauditpayload) |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `delivery_failure_detail` |  | opt | [FailureDetailPayload](#model-failuredetailpayload) (nullable) |
|  |  | `artifact_id` | opt | `string` (format: uuid; nullable) |
|  |  | `error_code` | req | `string` |
|  |  | `error_message` | req | `string` |
|  |  | `exception_type` | opt | `string` (nullable) |
|  |  | `occurred_at` | req | `string` (format: date-time) |
|  |  | `sandbox_diagnostics` | opt | [SandboxFailureDiagnostics](#model-sandboxfailurediagnostics) (nullable) |
|  |  | `task_id` | opt | `string` (format: uuid; nullable) |
|  |  | `traceback` | opt | `string` (nullable) |
|  |  | `uid` | opt | `integer` (nullable) |
|  | `diagnostics` |  | opt | [MinerTaskAttemptDiagnosticsPayload](#model-minertaskattemptdiagnosticspayload) (nullable) |
|  |  | `elapsed_ms` | opt | `number` (nullable) |
|  |  | `failure_owner` | opt | `string` (nullable) |
|  |  | `phase` | req | `string` |
|  |  | `platform_tool_activity_observed` | opt | `boolean` (default: False) |
|  |  | `timeout_owner` | opt | `string` (nullable) |
|  | `error_code` |  | opt | `string` (nullable) |
|  | `error_summary_code` |  | opt | `string` (nullable) |
|  | `execution_log` |  | opt | array[[ToolCall-Input](#model-toolcall-input)] (default: []) |
|  |  | `details` | req | [ToolCallDetails-Input](#model-toolcalldetails-input) |
|  |  | `issued_at` | req | `string` (format: date-time) |
|  |  | `outcome` | req | [ToolCallOutcome](#model-toolcalloutcome) |
|  |  | `receipt_id` | req | `string` |
|  |  | `session_id` | req | `string` (format: uuid) |
|  |  | `tool` | req | `string` (enum: [search_web, search_ai, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |
|  |  | `uid` | req | `integer` |
|  | `finished_at` |  | req | `string` (format: date-time) |
|  | `max_attempts` |  | req | `integer` |
|  | `miner_hotkey_ss58` |  | req | `string` |
|  | `retry_decision` |  | req | [MinerTaskAttemptRetryDecision](#model-minertaskattemptretrydecision) |
|  | `started_at` |  | req | `string` (format: date-time) |
|  | `status` |  | req | [MinerTaskAttemptStatus](#model-minertaskattemptstatus) |
|  | `task_id` |  | req | `string` (format: uuid) |
|  | `terminal_effect` |  | req | [MinerTaskAttemptTerminalEffect](#model-minertaskattemptterminaleffect) (nullable) |
|  | `uid` |  | req | `integer` |
|  | `validator_session_id` |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "artifact_id": {
      "format": "uuid",
      "title": "Artifact Id",
      "type": "string"
    },
    "attempt_number": {
      "minimum": 1.0,
      "title": "Attempt Number",
      "type": "integer"
    },
    "batch_id": {
      "format": "uuid",
      "title": "Batch Id",
      "type": "string"
    },
    "result": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/MinerTaskRunRequest"
        },
        {
          "type": "null"
        }
      ]
    },
    "task_id": {
      "format": "uuid",
      "title": "Task Id",
      "type": "string"
    },
    "terminal_attempt": {
      "$ref": "#/components/schemas/MinerTaskAttemptAuditPayload"
    }
  },
  "required": [
    "batch_id",
    "artifact_id",
    "task_id",
    "attempt_number",
    "terminal_attempt"
  ],
  "title": "MinerTaskWorkResultEnvelope",
  "type": "object"
}
```

</details>

<a id="model-minertaskworkresultitemresponse"></a>
### Model: MinerTaskWorkResultItemResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `attempt_number` |  |  | req | `integer` |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `canonical` |  |  | req | `boolean` |
| `outcome` |  |  | req | [MinerTaskResultOutcome](#model-minertaskresultoutcome) |
| `reason` |  |  | opt | `string` (nullable) |
| `reason_code` |  |  | opt | [MinerTaskResultReasonCode](#model-minertaskresultreasoncode) (nullable) |
| `task_id` |  |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "artifact_id": {
      "format": "uuid",
      "title": "Artifact Id",
      "type": "string"
    },
    "attempt_number": {
      "minimum": 1.0,
      "title": "Attempt Number",
      "type": "integer"
    },
    "batch_id": {
      "format": "uuid",
      "title": "Batch Id",
      "type": "string"
    },
    "canonical": {
      "title": "Canonical",
      "type": "boolean"
    },
    "outcome": {
      "$ref": "#/components/schemas/MinerTaskResultOutcome"
    },
    "reason": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Reason"
    },
    "reason_code": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/MinerTaskResultReasonCode"
        },
        {
          "type": "null"
        }
      ]
    },
    "task_id": {
      "format": "uuid",
      "title": "Task Id",
      "type": "string"
    }
  },
  "required": [
    "batch_id",
    "artifact_id",
    "task_id",
    "attempt_number",
    "outcome",
    "canonical"
  ],
  "title": "MinerTaskWorkResultItemResponse",
  "type": "object"
}
```

</details>

<a id="model-minertaskworkresultsrequest"></a>
### Model: MinerTaskWorkResultsRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `results` |  |  | opt | array[[MinerTaskWorkResultEnvelope](#model-minertaskworkresultenvelope)] (default: []) |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `result` |  | opt | [MinerTaskRunRequest](#model-minertaskrunrequest) (nullable) |
|  |  | `batch_id` | req | `string` (format: uuid) |
|  |  | `execution_log` | opt | array[[ToolCall-Input](#model-toolcall-input)] (default: []) |
|  |  | `run` | req | [MinerTaskRunSection](#model-minertaskrunsection) |
|  |  | `score` | opt | `number` (nullable) |
|  |  | `session` | req | [SessionModel](#model-sessionmodel) |
|  |  | `specifics` | req | [EvaluationDetails](#model-evaluationdetails) |
|  |  | `usage` | req | [UsageModel](#model-usagemodel) |
|  | `task_id` |  | req | `string` (format: uuid) |
|  | `terminal_attempt` |  | req | [MinerTaskAttemptAuditPayload](#model-minertaskattemptauditpayload) |
|  |  | `artifact_id` | req | `string` (format: uuid) |
|  |  | `attempt_number` | req | `integer` |
|  |  | `batch_id` | req | `string` (format: uuid) |
|  |  | `delivery_failure_detail` | opt | [FailureDetailPayload](#model-failuredetailpayload) (nullable) |
|  |  | `diagnostics` | opt | [MinerTaskAttemptDiagnosticsPayload](#model-minertaskattemptdiagnosticspayload) (nullable) |
|  |  | `error_code` | opt | `string` (nullable) |
|  |  | `error_summary_code` | opt | `string` (nullable) |
|  |  | `execution_log` | opt | array[[ToolCall-Input](#model-toolcall-input)] (default: []) |
|  |  | `finished_at` | req | `string` (format: date-time) |
|  |  | `max_attempts` | req | `integer` |
|  |  | `miner_hotkey_ss58` | req | `string` |
|  |  | `retry_decision` | req | [MinerTaskAttemptRetryDecision](#model-minertaskattemptretrydecision) |
|  |  | `started_at` | req | `string` (format: date-time) |
|  |  | `status` | req | [MinerTaskAttemptStatus](#model-minertaskattemptstatus) |
|  |  | `task_id` | req | `string` (format: uuid) |
|  |  | `terminal_effect` | req | [MinerTaskAttemptTerminalEffect](#model-minertaskattemptterminaleffect) (nullable) |
|  |  | `uid` | req | `integer` |
|  |  | `validator_session_id` | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "results": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/MinerTaskWorkResultEnvelope"
      },
      "title": "Results",
      "type": "array"
    }
  },
  "title": "MinerTaskWorkResultsRequest",
  "type": "object"
}
```

</details>

<a id="model-minertaskworkresultsresponse"></a>
### Model: MinerTaskWorkResultsResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `results` |  |  | opt | array[[MinerTaskWorkResultItemResponse](#model-minertaskworkresultitemresponse)] (default: []) |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `canonical` |  | req | `boolean` |
|  | `outcome` |  | req | [MinerTaskResultOutcome](#model-minertaskresultoutcome) |
|  | `reason` |  | opt | `string` (nullable) |
|  | `reason_code` |  | opt | [MinerTaskResultReasonCode](#model-minertaskresultreasoncode) (nullable) |
|  | `task_id` |  | req | `string` (format: uuid) |
| `server_time` |  |  | req | `string` (format: date-time) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "results": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/MinerTaskWorkResultItemResponse"
      },
      "title": "Results",
      "type": "array"
    },
    "server_time": {
      "format": "date-time",
      "title": "Server Time",
      "type": "string"
    }
  },
  "required": [
    "server_time"
  ],
  "title": "MinerTaskWorkResultsResponse",
  "type": "object"
}
```

</details>

<a id="model-minertaskworkscoreableexecutionpayload"></a>
### Model: MinerTaskWorkScoreableExecutionPayload

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact` |  |  | req | [MinerTaskWorkArtifactPayload](#model-minertaskworkartifactpayload) |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `content_hash` |  | req | `string` |
|  | `miner_hotkey_ss58` |  | opt | `string` (nullable) |
|  | `size_bytes` |  | req | `integer` |
|  | `uid` |  | req | `integer` |
| `attempt_number` |  |  | req | `integer` |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `execution_completed_at` |  |  | req | `string` (format: date-time) |
| `execution_log` |  |  | opt | array[[ToolCall-Output](#model-toolcall-output)] (default: []) |
|  | `details` |  | req | [ToolCallDetails-Output](#model-toolcalldetails-output) |
|  |  | `actual_cost_provider` | opt | `string` (nullable) |
|  |  | `actual_cost_usd` | opt | `number` (nullable) |
|  |  | `cost_usd` | opt | `number` (nullable) |
|  |  | `execution` | opt | [ToolExecutionFacts](#model-toolexecutionfacts) (nullable) |
|  |  | `extra` | opt | `object` (nullable) |
|  |  | `reference_cost_usd` | opt | `number` (nullable) |
|  |  | `request_hash` | req | `string` |
|  |  | `request_payload` | opt | [harnyx_miner_sdk__json_types__JsonValue-Output](#model-harnyx_miner_sdk__json_types__jsonvalue-output) (nullable) |
|  |  | `response_hash` | opt | `string` (nullable) |
|  |  | `response_payload` | opt | [harnyx_miner_sdk__json_types__JsonValue-Output](#model-harnyx_miner_sdk__json_types__jsonvalue-output) (nullable) |
|  |  | `result_policy` | opt | [ToolResultPolicy](#model-toolresultpolicy) (default: log_only) |
|  |  | `results` | opt | array[[ToolResult-Output](#model-toolresult-output)] (default: []) |
|  | `issued_at` |  | req | `string` (format: date-time) |
|  | `outcome` |  | req | [ToolCallOutcome](#model-toolcalloutcome) |
|  | `receipt_id` |  | req | `string` |
|  | `session_id` |  | req | `string` (format: uuid) |
|  | `tool` |  | req | `string` (enum: [search_web, search_ai, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |
|  | `uid` |  | req | `integer` |
| `max_attempts` |  |  | req | `integer` |
| `miner_hotkey_ss58` |  |  | req | `string` |
| `response` |  |  | req | [Response](#model-response) (nullable) |
|  | `citations` |  | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  |  | `note` | opt | `string` (nullable) |
|  |  | `title` | opt | `string` (nullable) |
|  |  | `url` | req | `string` |
|  | `output` |  | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  | `text` |  | opt | `string` (nullable) |
| `session` |  |  | req | [SessionModel](#model-sessionmodel) |
|  | `expires_at` |  | req | `string` |
|  | `issued_at` |  | req | `string` |
|  | `session_id` |  | req | `string` (format: uuid) |
|  | `status` |  | req | `string` |
|  | `uid` |  | req | `integer` |
| `started_at` |  |  | req | `string` (format: date-time) |
| `task` |  |  | req | [MinerTask](#model-minertask) |
|  | `budget_usd` |  | opt | `number` (default: 0.5) |
|  | `query` |  | req | [Query](#model-query) |
|  |  | `output_schema` | opt | [JsonObject-Input](#model-jsonobject-input) (nullable) |
|  |  | `text` | req | `string` |
|  | `reference_answer` |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `citations` | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  |  | `text` | req | `string` |
|  | `task_id` |  | req | `string` (format: uuid) |
| `total_tool_usage` |  |  | req | [ToolUsageSummary](#model-toolusagesummary) |
|  | `actual_cost_by_provider` |  | opt | `object` |
|  | `actual_total_cost_usd` |  | opt | `number` (nullable) |
|  | `embedding` |  | opt | [EmbeddingToolUsageSummary](#model-embeddingtoolusagesummary) |
|  |  | `actual_cost` | opt | `number` (nullable) |
|  |  | `call_count` | opt | `integer` (default: 0) |
|  |  | `cost` | opt | `number` (default: 0.0) |
|  |  | `reference_cost` | opt | `number` (default: 0.0) |
|  | `embedding_cost` |  | opt | `number` (default: 0.0) |
|  | `llm` |  | opt | [LlmUsageSummary](#model-llmusagesummary) |
|  |  | `actual_cost` | opt | `number` (nullable) |
|  |  | `call_count` | opt | `integer` (default: 0) |
|  |  | `completion_tokens` | opt | `integer` (default: 0) |
|  |  | `cost` | opt | `number` (default: 0.0) |
|  |  | `prompt_tokens` | opt | `integer` (default: 0) |
|  |  | `providers` | opt | `object` |
|  |  | `reasoning_tokens` | opt | `integer` (default: 0) |
|  |  | `reference_cost` | opt | `number` (default: 0.0) |
|  |  | `total_tokens` | opt | `integer` (default: 0) |
|  | `llm_cost` |  | opt | `number` (default: 0.0) |
|  | `reference_cost_by_provider` |  | opt | `object` |
|  | `reference_total_cost_usd` |  | opt | `number` (default: 0.0) |
|  | `search_tool` |  | opt | [SearchToolUsageSummary](#model-searchtoolusagesummary) |
|  |  | `actual_cost` | opt | `number` (nullable) |
|  |  | `call_count` | opt | `integer` (default: 0) |
|  |  | `cost` | opt | `number` (default: 0.0) |
|  |  | `reference_cost` | opt | `number` (default: 0.0) |
|  | `search_tool_cost` |  | opt | `number` (default: 0.0) |
| `trace` |  |  | opt | [EvaluationTrace](#model-evaluationtrace) (nullable) |
|  | `entrypoint_invocation_ms` |  | opt | `number` (nullable) |
|  | `orchestration_ms` |  | opt | `number` (nullable) |
|  | `scoring_judge_attempt_count` |  | opt | `integer` (nullable) |
|  | `scoring_judge_duration_ms` |  | opt | `number` (nullable) |
|  | `scoring_judge_retry_count` |  | opt | `integer` (nullable) |
|  | `scoring_judge_retry_reasons` |  | opt | array[`string`] (default: []) |
|  | `scoring_judge_selected_routes` |  | opt | array[`string`] (default: []) |
|  | `scoring_judge_status` |  | opt | `string` (enum: [ok, exhausted, failed]; nullable) |
|  | `scoring_ms` |  | opt | `number` (nullable) |
| `uid` |  |  | req | `integer` |
| `usage` |  |  | req | [UsageModel](#model-usagemodel) |
|  | `by_provider` |  | opt | `object` |
|  | `call_count` |  | req | `integer` |
|  | `total_completion_tokens` |  | req | `integer` |
|  | `total_prompt_tokens` |  | req | `integer` |
|  | `total_tokens` |  | req | `integer` |
| `validator_session_id` |  |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "artifact": {
      "$ref": "#/components/schemas/MinerTaskWorkArtifactPayload"
    },
    "attempt_number": {
      "minimum": 1.0,
      "title": "Attempt Number",
      "type": "integer"
    },
    "batch_id": {
      "format": "uuid",
      "title": "Batch Id",
      "type": "string"
    },
    "execution_completed_at": {
      "format": "date-time",
      "title": "Execution Completed At",
      "type": "string"
    },
    "execution_log": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/ToolCall-Output"
      },
      "title": "Execution Log",
      "type": "array"
    },
    "max_attempts": {
      "minimum": 1.0,
      "title": "Max Attempts",
      "type": "integer"
    },
    "miner_hotkey_ss58": {
      "minLength": 1,
      "title": "Miner Hotkey Ss58",
      "type": "string"
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
    "session": {
      "$ref": "#/components/schemas/SessionModel"
    },
    "started_at": {
      "format": "date-time",
      "title": "Started At",
      "type": "string"
    },
    "task": {
      "$ref": "#/components/schemas/MinerTask"
    },
    "total_tool_usage": {
      "$ref": "#/components/schemas/ToolUsageSummary"
    },
    "trace": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/EvaluationTrace"
        },
        {
          "type": "null"
        }
      ]
    },
    "uid": {
      "minimum": 0.0,
      "title": "Uid",
      "type": "integer"
    },
    "usage": {
      "$ref": "#/components/schemas/UsageModel"
    },
    "validator_session_id": {
      "format": "uuid",
      "title": "Validator Session Id",
      "type": "string"
    }
  },
  "required": [
    "batch_id",
    "artifact",
    "task",
    "attempt_number",
    "max_attempts",
    "validator_session_id",
    "uid",
    "miner_hotkey_ss58",
    "started_at",
    "execution_completed_at",
    "response",
    "session",
    "usage",
    "total_tool_usage"
  ],
  "title": "MinerTaskWorkScoreableExecutionPayload",
  "type": "object"
}
```

</details>

<a id="model-minertaskworkscoreableexecutionsrequest"></a>
### Model: MinerTaskWorkScoreableExecutionsRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `active_scoring` |  |  | opt | array[[MinerTaskAttemptIdentityPayload](#model-minertaskattemptidentitypayload)] (default: []) |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `task_id` |  | req | `string` (format: uuid) |
|  | `validator_session_id` |  | opt | `string` (format: uuid; nullable) |
| `limit` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "active_scoring": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/MinerTaskAttemptIdentityPayload"
      },
      "title": "Active Scoring",
      "type": "array"
    },
    "limit": {
      "maximum": 20.0,
      "minimum": 1.0,
      "title": "Limit",
      "type": "integer"
    }
  },
  "required": [
    "limit"
  ],
  "title": "MinerTaskWorkScoreableExecutionsRequest",
  "type": "object"
}
```

</details>

<a id="model-minertaskworkscoreableexecutionsresponse"></a>
### Model: MinerTaskWorkScoreableExecutionsResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `executions` |  |  | opt | array[[MinerTaskWorkScoreableExecutionPayload](#model-minertaskworkscoreableexecutionpayload)] (default: []) |
|  | `artifact` |  | req | [MinerTaskWorkArtifactPayload](#model-minertaskworkartifactpayload) |
|  |  | `artifact_id` | req | `string` (format: uuid) |
|  |  | `content_hash` | req | `string` |
|  |  | `miner_hotkey_ss58` | opt | `string` (nullable) |
|  |  | `size_bytes` | req | `integer` |
|  |  | `uid` | req | `integer` |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `execution_completed_at` |  | req | `string` (format: date-time) |
|  | `execution_log` |  | opt | array[[ToolCall-Output](#model-toolcall-output)] (default: []) |
|  |  | `details` | req | [ToolCallDetails-Output](#model-toolcalldetails-output) |
|  |  | `issued_at` | req | `string` (format: date-time) |
|  |  | `outcome` | req | [ToolCallOutcome](#model-toolcalloutcome) |
|  |  | `receipt_id` | req | `string` |
|  |  | `session_id` | req | `string` (format: uuid) |
|  |  | `tool` | req | `string` (enum: [search_web, search_ai, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |
|  |  | `uid` | req | `integer` |
|  | `max_attempts` |  | req | `integer` |
|  | `miner_hotkey_ss58` |  | req | `string` |
|  | `response` |  | req | [Response](#model-response) (nullable) |
|  |  | `citations` | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  |  | `output` | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  |  | `text` | opt | `string` (nullable) |
|  | `session` |  | req | [SessionModel](#model-sessionmodel) |
|  |  | `expires_at` | req | `string` |
|  |  | `issued_at` | req | `string` |
|  |  | `session_id` | req | `string` (format: uuid) |
|  |  | `status` | req | `string` |
|  |  | `uid` | req | `integer` |
|  | `started_at` |  | req | `string` (format: date-time) |
|  | `task` |  | req | [MinerTask](#model-minertask) |
|  |  | `budget_usd` | opt | `number` (default: 0.5) |
|  |  | `query` | req | [Query](#model-query) |
|  |  | `reference_answer` | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `task_id` | req | `string` (format: uuid) |
|  | `total_tool_usage` |  | req | [ToolUsageSummary](#model-toolusagesummary) |
|  |  | `actual_cost_by_provider` | opt | `object` |
|  |  | `actual_total_cost_usd` | opt | `number` (nullable) |
|  |  | `embedding` | opt | [EmbeddingToolUsageSummary](#model-embeddingtoolusagesummary) |
|  |  | `embedding_cost` | opt | `number` (default: 0.0) |
|  |  | `llm` | opt | [LlmUsageSummary](#model-llmusagesummary) |
|  |  | `llm_cost` | opt | `number` (default: 0.0) |
|  |  | `reference_cost_by_provider` | opt | `object` |
|  |  | `reference_total_cost_usd` | opt | `number` (default: 0.0) |
|  |  | `search_tool` | opt | [SearchToolUsageSummary](#model-searchtoolusagesummary) |
|  |  | `search_tool_cost` | opt | `number` (default: 0.0) |
|  | `trace` |  | opt | [EvaluationTrace](#model-evaluationtrace) (nullable) |
|  |  | `entrypoint_invocation_ms` | opt | `number` (nullable) |
|  |  | `orchestration_ms` | opt | `number` (nullable) |
|  |  | `scoring_judge_attempt_count` | opt | `integer` (nullable) |
|  |  | `scoring_judge_duration_ms` | opt | `number` (nullable) |
|  |  | `scoring_judge_retry_count` | opt | `integer` (nullable) |
|  |  | `scoring_judge_retry_reasons` | opt | array[`string`] (default: []) |
|  |  | `scoring_judge_selected_routes` | opt | array[`string`] (default: []) |
|  |  | `scoring_judge_status` | opt | `string` (enum: [ok, exhausted, failed]; nullable) |
|  |  | `scoring_ms` | opt | `number` (nullable) |
|  | `uid` |  | req | `integer` |
|  | `usage` |  | req | [UsageModel](#model-usagemodel) |
|  |  | `by_provider` | opt | `object` |
|  |  | `call_count` | req | `integer` |
|  |  | `total_completion_tokens` | req | `integer` |
|  |  | `total_prompt_tokens` | req | `integer` |
|  |  | `total_tokens` | req | `integer` |
|  | `validator_session_id` |  | req | `string` (format: uuid) |
| `server_time` |  |  | req | `string` (format: date-time) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "executions": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/MinerTaskWorkScoreableExecutionPayload"
      },
      "title": "Executions",
      "type": "array"
    },
    "server_time": {
      "format": "date-time",
      "title": "Server Time",
      "type": "string"
    }
  },
  "required": [
    "server_time"
  ],
  "title": "MinerTaskWorkScoreableExecutionsResponse",
  "type": "object"
}
```

</details>

<a id="model-minertaskworktasksrequest"></a>
### Model: MinerTaskWorkTasksRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `active_attempts` |  |  | opt | array[[MinerTaskAttemptIdentityPayload](#model-minertaskattemptidentitypayload)] (default: []) |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `task_id` |  | req | `string` (format: uuid) |
|  | `validator_session_id` |  | opt | `string` (format: uuid; nullable) |
| `max_active_artifacts` |  |  | req | `integer` |
| `target_concurrency` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "active_attempts": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/MinerTaskAttemptIdentityPayload"
      },
      "title": "Active Attempts",
      "type": "array"
    },
    "max_active_artifacts": {
      "maximum": 256.0,
      "minimum": 1.0,
      "title": "Max Active Artifacts",
      "type": "integer"
    },
    "target_concurrency": {
      "maximum": 256.0,
      "minimum": 1.0,
      "title": "Target Concurrency",
      "type": "integer"
    }
  },
  "required": [
    "target_concurrency",
    "max_active_artifacts"
  ],
  "title": "MinerTaskWorkTasksRequest",
  "type": "object"
}
```

</details>

<a id="model-minertaskworktasksresponse"></a>
### Model: MinerTaskWorkTasksResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `server_time` |  |  | req | `string` (format: date-time) |
| `tasks` |  |  | opt | array[[MinerTaskWorkAssignmentPayload](#model-minertaskworkassignmentpayload)] (default: []) |
|  | `artifact` |  | req | [MinerTaskWorkArtifactPayload](#model-minertaskworkartifactpayload) |
|  |  | `artifact_id` | req | `string` (format: uuid) |
|  |  | `content_hash` | req | `string` |
|  |  | `miner_hotkey_ss58` | opt | `string` (nullable) |
|  |  | `size_bytes` | req | `integer` |
|  |  | `uid` | req | `integer` |
|  | `assignment_token` |  | req | `string` |
|  | `attempt_number` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `max_attempts` |  | req | `integer` |
|  | `task` |  | req | [MinerTask](#model-minertask) |
|  |  | `budget_usd` | opt | `number` (default: 0.5) |
|  |  | `query` | req | [Query](#model-query) |
|  |  | `reference_answer` | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `task_id` | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "server_time": {
      "format": "date-time",
      "title": "Server Time",
      "type": "string"
    },
    "tasks": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/MinerTaskWorkAssignmentPayload"
      },
      "title": "Tasks",
      "type": "array"
    }
  },
  "required": [
    "server_time"
  ],
  "title": "MinerTaskWorkTasksResponse",
  "type": "object"
}
```

</details>

<a id="model-overrideminertaskdatasetmodel"></a>
### Model: OverrideMinerTaskDatasetModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `tasks` |  |  | req | array[[MinerTaskInputModel](#model-minertaskinputmodel)] |
|  | `budget_usd` |  | opt | `number` (default: 0.5) |
|  | `query` |  | req | [Query](#model-query) |
|  |  | `output_schema` | opt | [JsonObject-Input](#model-jsonobject-input) (nullable) |
|  |  | `text` | req | `string` |
|  | `reference_answer` |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `citations` | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  |  | `text` | req | `string` |
|  | `task_id` |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "tasks": {
      "items": {
        "$ref": "#/components/schemas/MinerTaskInputModel"
      },
      "minItems": 1,
      "title": "Tasks",
      "type": "array"
    }
  },
  "required": [
    "tasks"
  ],
  "title": "OverrideMinerTaskDatasetModel",
  "type": "object"
}
```

</details>

<a id="model-platformtoolproxygrantrequest"></a>
### Model: PlatformToolProxyGrantRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `assignment_token` |  |  | req | `string` |
| `attempt_number` |  |  | req | `integer` |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `task_id` |  |  | req | `string` (format: uuid) |
| `validator_session_id` |  |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "artifact_id": {
      "format": "uuid",
      "title": "Artifact Id",
      "type": "string"
    },
    "assignment_token": {
      "minLength": 1,
      "title": "Assignment Token",
      "type": "string"
    },
    "attempt_number": {
      "minimum": 1.0,
      "title": "Attempt Number",
      "type": "integer"
    },
    "batch_id": {
      "format": "uuid",
      "title": "Batch Id",
      "type": "string"
    },
    "task_id": {
      "format": "uuid",
      "title": "Task Id",
      "type": "string"
    },
    "validator_session_id": {
      "format": "uuid",
      "title": "Validator Session Id",
      "type": "string"
    }
  },
  "required": [
    "batch_id",
    "artifact_id",
    "task_id",
    "validator_session_id",
    "attempt_number",
    "assignment_token"
  ],
  "title": "PlatformToolProxyGrantRequest",
  "type": "object"
}
```

</details>

<a id="model-platformtoolproxygrantresponse"></a>
### Model: PlatformToolProxyGrantResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `expires_at` |  |  | req | `string` (format: date-time) |
| `token` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "expires_at": {
      "format": "date-time",
      "title": "Expires At",
      "type": "string"
    },
    "token": {
      "title": "Token",
      "type": "string"
    }
  },
  "required": [
    "token",
    "expires_at"
  ],
  "title": "PlatformToolProxyGrantResponse",
  "type": "object"
}
```

</details>

<a id="model-query"></a>
### Model: Query

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `output_schema` |  |  | opt | [JsonObject-Input](#model-jsonobject-input) (nullable) |
| `text` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "output_schema": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/JsonObject-Input"
        },
        {
          "type": "null"
        }
      ]
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

<a id="model-referenceanswer"></a>
### Model: ReferenceAnswer

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `citations` |  |  | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  | `note` |  | opt | `string` (nullable) |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | req | `string` |
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
            "$ref": "#/components/schemas/AnswerCitation"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "title": "Citations"
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

<a id="model-registervalidatorrequest"></a>
### Model: RegisterValidatorRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `base_url` |  |  | req | `string` |
| `local_image_id` |  |  | opt | `string` (nullable) |
| `registry_digest` |  |  | opt | `string` (nullable) |
| `source_revision` |  |  | opt | `string` (nullable) |
| `validator_version` |  |  | opt | `string` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "base_url": {
      "minLength": 1,
      "title": "Base Url",
      "type": "string"
    },
    "local_image_id": {
      "anyOf": [
        {
          "minLength": 1,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Local Image Id"
    },
    "registry_digest": {
      "anyOf": [
        {
          "minLength": 1,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Registry Digest"
    },
    "source_revision": {
      "anyOf": [
        {
          "minLength": 1,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Source Revision"
    },
    "validator_version": {
      "anyOf": [
        {
          "minLength": 1,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Validator Version"
    }
  },
  "required": [
    "base_url"
  ],
  "title": "RegisterValidatorRequest",
  "type": "object"
}
```

</details>

<a id="model-reposearchensureindexrequestmodel"></a>
### Model: RepoSearchEnsureIndexRequestModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `commit_sha` |  |  | req | `string` |
| `repo_url` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "commit_sha": {
      "pattern": "^[0-9a-f]{40}$",
      "title": "Commit Sha",
      "type": "string"
    },
    "repo_url": {
      "title": "Repo Url",
      "type": "string"
    }
  },
  "required": [
    "repo_url",
    "commit_sha"
  ],
  "title": "RepoSearchEnsureIndexRequestModel",
  "type": "object"
}
```

</details>

<a id="model-response"></a>
### Model: Response

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `citations` |  |  | opt | array[[AnswerCitation](#model-answercitation)] (nullable) |
|  | `note` |  | opt | `string` (nullable) |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | req | `string` |
| `output` |  |  | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
| `text` |  |  | opt | `string` (nullable) |

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
            "$ref": "#/components/schemas/AnswerCitation"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "title": "Citations"
    },
    "output": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/harnyx_miner_sdk__json_types__JsonValue-Input"
        },
        {
          "type": "null"
        }
      ]
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
      "title": "Text"
    }
  },
  "title": "Response",
  "type": "object"
}
```

</details>

<a id="model-sandboxfailurediagnostics"></a>
### Model: SandboxFailureDiagnostics

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `container_id` |  |  | opt | `string` (nullable) |
| `container_name` |  |  | opt | `string` (nullable) |
| `docker_inspect_error_tail` |  |  | opt | `string` (nullable) |
| `docker_logs_error_tail` |  |  | opt | `string` (nullable) |
| `docker_logs_tail` |  |  | opt | `string` (nullable) |
| `error_text` |  |  | opt | `string` (nullable) |
| `exit_code` |  |  | opt | `integer` (nullable) |
| `image` |  |  | opt | `string` (nullable) |
| `oom_killed` |  |  | opt | `boolean` (nullable) |
| `pull_policy` |  |  | opt | `string` (nullable) |
| `pull_returncode` |  |  | opt | `integer` (nullable) |
| `pull_stderr_tail` |  |  | opt | `string` (nullable) |
| `pull_stdout_tail` |  |  | opt | `string` (nullable) |
| `run_returncode` |  |  | opt | `integer` (nullable) |
| `run_stderr_tail` |  |  | opt | `string` (nullable) |
| `run_stdout_tail` |  |  | opt | `string` (nullable) |
| `state_error` |  |  | opt | `string` (nullable) |
| `status` |  |  | opt | `string` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "container_id": {
      "anyOf": [
        {
          "maxLength": 512,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Container Id"
    },
    "container_name": {
      "anyOf": [
        {
          "maxLength": 512,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Container Name"
    },
    "docker_inspect_error_tail": {
      "anyOf": [
        {
          "maxLength": 4096,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Docker Inspect Error Tail"
    },
    "docker_logs_error_tail": {
      "anyOf": [
        {
          "maxLength": 4096,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Docker Logs Error Tail"
    },
    "docker_logs_tail": {
      "anyOf": [
        {
          "maxLength": 8192,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Docker Logs Tail"
    },
    "error_text": {
      "anyOf": [
        {
          "maxLength": 4096,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Error Text"
    },
    "exit_code": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Exit Code"
    },
    "image": {
      "anyOf": [
        {
          "maxLength": 512,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Image"
    },
    "oom_killed": {
      "anyOf": [
        {
          "type": "boolean"
        },
        {
          "type": "null"
        }
      ],
      "title": "Oom Killed"
    },
    "pull_policy": {
      "anyOf": [
        {
          "maxLength": 512,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Pull Policy"
    },
    "pull_returncode": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Pull Returncode"
    },
    "pull_stderr_tail": {
      "anyOf": [
        {
          "maxLength": 4096,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Pull Stderr Tail"
    },
    "pull_stdout_tail": {
      "anyOf": [
        {
          "maxLength": 4096,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Pull Stdout Tail"
    },
    "run_returncode": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Run Returncode"
    },
    "run_stderr_tail": {
      "anyOf": [
        {
          "maxLength": 4096,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Run Stderr Tail"
    },
    "run_stdout_tail": {
      "anyOf": [
        {
          "maxLength": 4096,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Run Stdout Tail"
    },
    "state_error": {
      "anyOf": [
        {
          "maxLength": 2048,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "State Error"
    },
    "status": {
      "anyOf": [
        {
          "maxLength": 512,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Status"
    }
  },
  "title": "SandboxFailureDiagnostics",
  "type": "object"
}
```

</details>

<a id="model-scorebreakdown"></a>
### Model: ScoreBreakdown

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `comparison_score` |  |  | req | `number` |
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

<a id="model-scriptartifactmodel"></a>
### Model: ScriptArtifactModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `content_hash` |  |  | req | `string` |
| `size_bytes` |  |  | req | `integer` |
| `submitted_at` |  |  | req | `string` (format: date-time) |
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
    "submitted_at": {
      "format": "date-time",
      "title": "Submitted At",
      "type": "string"
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
    "size_bytes",
    "submitted_at"
  ],
  "title": "ScriptArtifactModel",
  "type": "object"
}
```

</details>

<a id="model-searchreporesult"></a>
### Model: SearchRepoResult

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `bm25` |  |  | opt | `number` (nullable) |
| `excerpt` |  |  | opt | `string` (nullable) |
| `path` |  |  | req | `string` |
| `title` |  |  | opt | `string` (nullable) |
| `url` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "description": "Single repository search result item.",
  "properties": {
    "bm25": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Bm25"
    },
    "excerpt": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Excerpt"
    },
    "path": {
      "title": "Path",
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
      "title": "Url",
      "type": "string"
    }
  },
  "required": [
    "path",
    "url"
  ],
  "title": "SearchRepoResult",
  "type": "object"
}
```

</details>

<a id="model-searchreposearchrequest"></a>
### Model: SearchRepoSearchRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `commit_sha` |  |  | req | `string` |
| `limit` |  |  | opt | `integer` (default: 10) |
| `path_glob` |  |  | opt | `string` (nullable) |
| `query` |  |  | req | `string` |
| `repo_url` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "description": "Query parameters for the platform repo-search callback.",
  "properties": {
    "commit_sha": {
      "pattern": "^[0-9a-f]{40}$",
      "title": "Commit Sha",
      "type": "string"
    },
    "limit": {
      "default": 10,
      "maximum": 50.0,
      "minimum": 1.0,
      "title": "Limit",
      "type": "integer"
    },
    "path_glob": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Path Glob"
    },
    "query": {
      "title": "Query",
      "type": "string"
    },
    "repo_url": {
      "title": "Repo Url",
      "type": "string"
    }
  },
  "required": [
    "repo_url",
    "commit_sha",
    "query"
  ],
  "title": "SearchRepoSearchRequest",
  "type": "object"
}
```

</details>

<a id="model-searchreposearchresponse"></a>
### Model: SearchRepoSearchResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `data` |  |  | opt | array[[SearchRepoResult](#model-searchreporesult)] |
|  | `bm25` |  | opt | `number` (nullable) |
|  | `excerpt` |  | opt | `string` (nullable) |
|  | `path` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "description": "Response payload for the platform repo-search callback.",
  "properties": {
    "data": {
      "items": {
        "$ref": "#/components/schemas/SearchRepoResult"
      },
      "title": "Data",
      "type": "array"
    }
  },
  "title": "SearchRepoSearchResponse",
  "type": "object"
}
```

</details>

<a id="model-searchtoolusagesummary"></a>
### Model: SearchToolUsageSummary

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `actual_cost` |  |  | opt | `number` (nullable) |
| `call_count` |  |  | opt | `integer` (default: 0) |
| `cost` |  |  | opt | `number` (default: 0.0) |
| `reference_cost` |  |  | opt | `number` (default: 0.0) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "actual_cost": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Actual Cost"
    },
    "call_count": {
      "default": 0,
      "title": "Call Count",
      "type": "integer"
    },
    "cost": {
      "default": 0.0,
      "title": "Cost",
      "type": "number"
    },
    "reference_cost": {
      "default": 0.0,
      "title": "Reference Cost",
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
| `session_id` |  |  | req | `string` (format: uuid) |
| `status` |  |  | req | `string` |
| `uid` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
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
      "format": "uuid",
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

<a id="model-spanmodel"></a>
### Model: SpanModel

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
  "title": "SpanModel",
  "type": "object"
}
```

</details>

<a id="model-statusresponse"></a>
### Model: StatusResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `status` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "status": {
      "title": "Status",
      "type": "string"
    }
  },
  "required": [
    "status"
  ],
  "title": "StatusResponse",
  "type": "object"
}
```

</details>

<a id="model-toolcall-input"></a>
### Model: ToolCall-Input

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `details` |  |  | req | [ToolCallDetails-Input](#model-toolcalldetails-input) |
|  | `actual_cost_provider` |  | opt | `string` (nullable) |
|  | `actual_cost_usd` |  | opt | `number` (nullable) |
|  | `cost_usd` |  | opt | `number` (nullable) |
|  | `execution` |  | opt | [ToolExecutionFacts](#model-toolexecutionfacts) (nullable) |
|  |  | `elapsed_ms` | opt | `number` (nullable) |
|  |  | `finished_at` | opt | `string` (format: date-time; nullable) |
|  |  | `started_at` | opt | `string` (format: date-time; nullable) |
|  |  | `ttft_ms` | opt | `number` (nullable) |
|  | `extra` |  | opt | `object` (nullable) |
|  | `reference_cost_usd` |  | opt | `number` (nullable) |
|  | `request_hash` |  | req | `string` |
|  | `request_payload` |  | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  | `response_hash` |  | opt | `string` (nullable) |
|  | `response_payload` |  | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  | `result_policy` |  | opt | [ToolResultPolicy](#model-toolresultpolicy) (default: log_only) |
|  | `results` |  | opt | array[[ToolResult-Input](#model-toolresult-input)] (default: []) |
|  |  | `index` | req | `integer` |
|  |  | `raw` | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  |  | `result_id` | req | `string` |
| `issued_at` |  |  | req | `string` (format: date-time) |
| `outcome` |  |  | req | [ToolCallOutcome](#model-toolcalloutcome) |
| `receipt_id` |  |  | req | `string` |
| `session_id` |  |  | req | `string` (format: uuid) |
| `tool` |  |  | req | `string` (enum: [search_web, search_ai, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |
| `uid` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "description": "Immutable audit trail for a tool invocation.",
  "properties": {
    "details": {
      "$ref": "#/components/schemas/ToolCallDetails-Input"
    },
    "issued_at": {
      "format": "date-time",
      "title": "Issued At",
      "type": "string"
    },
    "outcome": {
      "$ref": "#/components/schemas/ToolCallOutcome"
    },
    "receipt_id": {
      "title": "Receipt Id",
      "type": "string"
    },
    "session_id": {
      "format": "uuid",
      "title": "Session Id",
      "type": "string"
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
    },
    "uid": {
      "title": "Uid",
      "type": "integer"
    }
  },
  "required": [
    "receipt_id",
    "session_id",
    "uid",
    "tool",
    "issued_at",
    "outcome",
    "details"
  ],
  "title": "ToolCall",
  "type": "object"
}
```

</details>

<a id="model-toolcall-output"></a>
### Model: ToolCall-Output

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `details` |  |  | req | [ToolCallDetails-Output](#model-toolcalldetails-output) |
|  | `actual_cost_provider` |  | opt | `string` (nullable) |
|  | `actual_cost_usd` |  | opt | `number` (nullable) |
|  | `cost_usd` |  | opt | `number` (nullable) |
|  | `execution` |  | opt | [ToolExecutionFacts](#model-toolexecutionfacts) (nullable) |
|  |  | `elapsed_ms` | opt | `number` (nullable) |
|  |  | `finished_at` | opt | `string` (format: date-time; nullable) |
|  |  | `started_at` | opt | `string` (format: date-time; nullable) |
|  |  | `ttft_ms` | opt | `number` (nullable) |
|  | `extra` |  | opt | `object` (nullable) |
|  | `reference_cost_usd` |  | opt | `number` (nullable) |
|  | `request_hash` |  | req | `string` |
|  | `request_payload` |  | opt | [harnyx_miner_sdk__json_types__JsonValue-Output](#model-harnyx_miner_sdk__json_types__jsonvalue-output) (nullable) |
|  | `response_hash` |  | opt | `string` (nullable) |
|  | `response_payload` |  | opt | [harnyx_miner_sdk__json_types__JsonValue-Output](#model-harnyx_miner_sdk__json_types__jsonvalue-output) (nullable) |
|  | `result_policy` |  | opt | [ToolResultPolicy](#model-toolresultpolicy) (default: log_only) |
|  | `results` |  | opt | array[[ToolResult-Output](#model-toolresult-output)] (default: []) |
|  |  | `index` | req | `integer` |
|  |  | `raw` | opt | [harnyx_miner_sdk__json_types__JsonValue-Output](#model-harnyx_miner_sdk__json_types__jsonvalue-output) (nullable) |
|  |  | `result_id` | req | `string` |
| `issued_at` |  |  | req | `string` (format: date-time) |
| `outcome` |  |  | req | [ToolCallOutcome](#model-toolcalloutcome) |
| `receipt_id` |  |  | req | `string` |
| `session_id` |  |  | req | `string` (format: uuid) |
| `tool` |  |  | req | `string` (enum: [search_web, search_ai, fetch_page, embed_text, llm_chat, test_tool, tooling_info]) |
| `uid` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "description": "Immutable audit trail for a tool invocation.",
  "properties": {
    "details": {
      "$ref": "#/components/schemas/ToolCallDetails-Output"
    },
    "issued_at": {
      "format": "date-time",
      "title": "Issued At",
      "type": "string"
    },
    "outcome": {
      "$ref": "#/components/schemas/ToolCallOutcome"
    },
    "receipt_id": {
      "title": "Receipt Id",
      "type": "string"
    },
    "session_id": {
      "format": "uuid",
      "title": "Session Id",
      "type": "string"
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
    },
    "uid": {
      "title": "Uid",
      "type": "integer"
    }
  },
  "required": [
    "receipt_id",
    "session_id",
    "uid",
    "tool",
    "issued_at",
    "outcome",
    "details"
  ],
  "title": "ToolCall",
  "type": "object"
}
```

</details>

<a id="model-toolcalldetails-input"></a>
### Model: ToolCallDetails-Input

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `actual_cost_provider` |  |  | opt | `string` (nullable) |
| `actual_cost_usd` |  |  | opt | `number` (nullable) |
| `cost_usd` |  |  | opt | `number` (nullable) |
| `execution` |  |  | opt | [ToolExecutionFacts](#model-toolexecutionfacts) (nullable) |
|  | `elapsed_ms` |  | opt | `number` (nullable) |
|  | `finished_at` |  | opt | `string` (format: date-time; nullable) |
|  | `started_at` |  | opt | `string` (format: date-time; nullable) |
|  | `ttft_ms` |  | opt | `number` (nullable) |
| `extra` |  |  | opt | `object` (nullable) |
| `reference_cost_usd` |  |  | opt | `number` (nullable) |
| `request_hash` |  |  | req | `string` |
| `request_payload` |  |  | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
| `response_hash` |  |  | opt | `string` (nullable) |
| `response_payload` |  |  | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
| `result_policy` |  |  | opt | [ToolResultPolicy](#model-toolresultpolicy) (default: log_only) |
| `results` |  |  | opt | array[[ToolResult-Input](#model-toolresult-input)] (default: []) |
|  | `index` |  | req | `integer` |
|  | `raw` |  | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
|  | `result_id` |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "description": "Supplemental details stored alongside a tool call receipt.",
  "properties": {
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
    "execution": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/ToolExecutionFacts"
        },
        {
          "type": "null"
        }
      ]
    },
    "extra": {
      "anyOf": [
        {
          "additionalProperties": {
            "$ref": "#/components/schemas/harnyx_miner_sdk__json_types__JsonValue-Input"
          },
          "type": "object"
        },
        {
          "type": "null"
        }
      ],
      "title": "Extra"
    },
    "reference_cost_usd": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Reference Cost Usd"
    },
    "request_hash": {
      "title": "Request Hash",
      "type": "string"
    },
    "request_payload": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/harnyx_miner_sdk__json_types__JsonValue-Input"
        },
        {
          "type": "null"
        }
      ]
    },
    "response_hash": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Response Hash"
    },
    "response_payload": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/harnyx_miner_sdk__json_types__JsonValue-Input"
        },
        {
          "type": "null"
        }
      ]
    },
    "result_policy": {
      "$ref": "#/components/schemas/ToolResultPolicy",
      "default": "log_only"
    },
    "results": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/ToolResult-Input"
      },
      "title": "Results",
      "type": "array"
    }
  },
  "required": [
    "request_hash"
  ],
  "title": "ToolCallDetails",
  "type": "object"
}
```

</details>

<a id="model-toolcalldetails-output"></a>
### Model: ToolCallDetails-Output

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `actual_cost_provider` |  |  | opt | `string` (nullable) |
| `actual_cost_usd` |  |  | opt | `number` (nullable) |
| `cost_usd` |  |  | opt | `number` (nullable) |
| `execution` |  |  | opt | [ToolExecutionFacts](#model-toolexecutionfacts) (nullable) |
|  | `elapsed_ms` |  | opt | `number` (nullable) |
|  | `finished_at` |  | opt | `string` (format: date-time; nullable) |
|  | `started_at` |  | opt | `string` (format: date-time; nullable) |
|  | `ttft_ms` |  | opt | `number` (nullable) |
| `extra` |  |  | opt | `object` (nullable) |
| `reference_cost_usd` |  |  | opt | `number` (nullable) |
| `request_hash` |  |  | req | `string` |
| `request_payload` |  |  | opt | [harnyx_miner_sdk__json_types__JsonValue-Output](#model-harnyx_miner_sdk__json_types__jsonvalue-output) (nullable) |
| `response_hash` |  |  | opt | `string` (nullable) |
| `response_payload` |  |  | opt | [harnyx_miner_sdk__json_types__JsonValue-Output](#model-harnyx_miner_sdk__json_types__jsonvalue-output) (nullable) |
| `result_policy` |  |  | opt | [ToolResultPolicy](#model-toolresultpolicy) (default: log_only) |
| `results` |  |  | opt | array[[ToolResult-Output](#model-toolresult-output)] (default: []) |
|  | `index` |  | req | `integer` |
|  | `raw` |  | opt | [harnyx_miner_sdk__json_types__JsonValue-Output](#model-harnyx_miner_sdk__json_types__jsonvalue-output) (nullable) |
|  | `result_id` |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "description": "Supplemental details stored alongside a tool call receipt.",
  "properties": {
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
    "execution": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/ToolExecutionFacts"
        },
        {
          "type": "null"
        }
      ]
    },
    "extra": {
      "anyOf": [
        {
          "additionalProperties": {
            "$ref": "#/components/schemas/harnyx_miner_sdk__json_types__JsonValue-Output"
          },
          "type": "object"
        },
        {
          "type": "null"
        }
      ],
      "title": "Extra"
    },
    "reference_cost_usd": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Reference Cost Usd"
    },
    "request_hash": {
      "title": "Request Hash",
      "type": "string"
    },
    "request_payload": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/harnyx_miner_sdk__json_types__JsonValue-Output"
        },
        {
          "type": "null"
        }
      ]
    },
    "response_hash": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Response Hash"
    },
    "response_payload": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/harnyx_miner_sdk__json_types__JsonValue-Output"
        },
        {
          "type": "null"
        }
      ]
    },
    "result_policy": {
      "$ref": "#/components/schemas/ToolResultPolicy",
      "default": "log_only"
    },
    "results": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/ToolResult-Output"
      },
      "title": "Results",
      "type": "array"
    }
  },
  "required": [
    "request_hash"
  ],
  "title": "ToolCallDetails",
  "type": "object"
}
```

</details>

<a id="model-toolcalloutcome"></a>
### Model: ToolCallOutcome

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "description": "High-level outcome for a tool invocation.",
  "enum": [
    "ok",
    "provider_error",
    "budget_exceeded",
    "internal_error",
    "timeout"
  ],
  "title": "ToolCallOutcome",
  "type": "string"
}
```

</details>

<a id="model-toolexecutionfacts"></a>
### Model: ToolExecutionFacts

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `elapsed_ms` |  |  | opt | `number` (nullable) |
| `finished_at` |  |  | opt | `string` (format: date-time; nullable) |
| `started_at` |  |  | opt | `string` (format: date-time; nullable) |
| `ttft_ms` |  |  | opt | `number` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "description": "Execution facts captured at the private tool runtime boundary.",
  "properties": {
    "elapsed_ms": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Elapsed Ms"
    },
    "finished_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Finished At"
    },
    "started_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Started At"
    },
    "ttft_ms": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Ttft Ms"
    }
  },
  "title": "ToolExecutionFacts",
  "type": "object"
}
```

</details>

<a id="model-toolresult-input"></a>
### Model: ToolResult-Input

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `index` |  |  | req | `integer` |
| `raw` |  |  | opt | [harnyx_miner_sdk__json_types__JsonValue-Input](#model-harnyx_miner_sdk__json_types__jsonvalue-input) (nullable) |
| `result_id` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "description": "Structured representation of a tool result for auditing.",
  "properties": {
    "index": {
      "title": "Index",
      "type": "integer"
    },
    "raw": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/harnyx_miner_sdk__json_types__JsonValue-Input"
        },
        {
          "type": "null"
        }
      ]
    },
    "result_id": {
      "title": "Result Id",
      "type": "string"
    }
  },
  "required": [
    "index",
    "result_id"
  ],
  "title": "ToolResult",
  "type": "object"
}
```

</details>

<a id="model-toolresult-output"></a>
### Model: ToolResult-Output

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `index` |  |  | req | `integer` |
| `raw` |  |  | opt | [harnyx_miner_sdk__json_types__JsonValue-Output](#model-harnyx_miner_sdk__json_types__jsonvalue-output) (nullable) |
| `result_id` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "description": "Structured representation of a tool result for auditing.",
  "properties": {
    "index": {
      "title": "Index",
      "type": "integer"
    },
    "raw": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/harnyx_miner_sdk__json_types__JsonValue-Output"
        },
        {
          "type": "null"
        }
      ]
    },
    "result_id": {
      "title": "Result Id",
      "type": "string"
    }
  },
  "required": [
    "index",
    "result_id"
  ],
  "title": "ToolResult",
  "type": "object"
}
```

</details>

<a id="model-toolresultpolicy"></a>
### Model: ToolResultPolicy

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "description": "Indicates whether tool results can be cited.",
  "enum": [
    "referenceable",
    "log_only"
  ],
  "title": "ToolResultPolicy",
  "type": "string"
}
```

</details>

<a id="model-toolusagesummary"></a>
### Model: ToolUsageSummary

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `actual_cost_by_provider` |  |  | opt | `object` |
| `actual_total_cost_usd` |  |  | opt | `number` (nullable) |
| `embedding` |  |  | opt | [EmbeddingToolUsageSummary](#model-embeddingtoolusagesummary) |
|  | `actual_cost` |  | opt | `number` (nullable) |
|  | `call_count` |  | opt | `integer` (default: 0) |
|  | `cost` |  | opt | `number` (default: 0.0) |
|  | `reference_cost` |  | opt | `number` (default: 0.0) |
| `embedding_cost` |  |  | opt | `number` (default: 0.0) |
| `llm` |  |  | opt | [LlmUsageSummary](#model-llmusagesummary) |
|  | `actual_cost` |  | opt | `number` (nullable) |
|  | `call_count` |  | opt | `integer` (default: 0) |
|  | `completion_tokens` |  | opt | `integer` (default: 0) |
|  | `cost` |  | opt | `number` (default: 0.0) |
|  | `prompt_tokens` |  | opt | `integer` (default: 0) |
|  | `providers` |  | opt | `object` |
|  | `reasoning_tokens` |  | opt | `integer` (default: 0) |
|  | `reference_cost` |  | opt | `number` (default: 0.0) |
|  | `total_tokens` |  | opt | `integer` (default: 0) |
| `llm_cost` |  |  | opt | `number` (default: 0.0) |
| `reference_cost_by_provider` |  |  | opt | `object` |
| `reference_total_cost_usd` |  |  | opt | `number` (default: 0.0) |
| `search_tool` |  |  | opt | [SearchToolUsageSummary](#model-searchtoolusagesummary) |
|  | `actual_cost` |  | opt | `number` (nullable) |
|  | `call_count` |  | opt | `integer` (default: 0) |
|  | `cost` |  | opt | `number` (default: 0.0) |
|  | `reference_cost` |  | opt | `number` (default: 0.0) |
| `search_tool_cost` |  |  | opt | `number` (default: 0.0) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "actual_cost_by_provider": {
      "additionalProperties": {
        "type": "number"
      },
      "title": "Actual Cost By Provider",
      "type": "object"
    },
    "actual_total_cost_usd": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Actual Total Cost Usd"
    },
    "embedding": {
      "$ref": "#/components/schemas/EmbeddingToolUsageSummary"
    },
    "embedding_cost": {
      "default": 0.0,
      "title": "Embedding Cost",
      "type": "number"
    },
    "llm": {
      "$ref": "#/components/schemas/LlmUsageSummary"
    },
    "llm_cost": {
      "default": 0.0,
      "title": "Llm Cost",
      "type": "number"
    },
    "reference_cost_by_provider": {
      "additionalProperties": {
        "type": "number"
      },
      "title": "Reference Cost By Provider",
      "type": "object"
    },
    "reference_total_cost_usd": {
      "default": 0.0,
      "title": "Reference Total Cost Usd",
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

<a id="model-topicgatemodel"></a>
### Model: TopicGateModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `criteria` |  |  | opt | array[[CriterionAssessmentModel](#model-criterionassessmentmodel)] (default: []) |
|  | `aggregate_score` |  | req | `number` |
|  | `criterion_evaluations` |  | req | array[[CriterionEvaluationModel](#model-criterionevaluationmodel)] |
|  |  | `citations` | opt | array[[CitationModel](#model-citationmodel)] (default: []) |
|  |  | `internal_metadata` | opt | `object` (nullable) |
|  |  | `justification` | req | `string` |
|  |  | `spans` | opt | array[[SpanModel](#model-spanmodel)] (default: []) |
|  |  | `verdict` | req | `integer` |
|  | `criterion_id` |  | req | `string` |
|  | `verdict_options` |  | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `description` | req | `string` |
|  |  | `value` | req | `integer` |
| `score` |  |  | opt | `number` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "criteria": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/CriterionAssessmentModel"
      },
      "title": "Criteria",
      "type": "array"
    },
    "score": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "title": "Score"
    }
  },
  "title": "TopicGateModel",
  "type": "object"
}
```

</details>

<a id="model-uploadscriptrequest"></a>
### Model: UploadScriptRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `script_b64` |  |  | req | `string` |
| `sha256` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "script_b64": {
      "title": "Script B64",
      "type": "string"
    },
    "sha256": {
      "title": "Sha256",
      "type": "string"
    }
  },
  "required": [
    "script_b64",
    "sha256"
  ],
  "title": "UploadScriptRequest",
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

<a id="model-verdictoptionmodel"></a>
### Model: VerdictOptionModel

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
  "title": "VerdictOptionModel",
  "type": "object"
}
```

</details>

<a id="model-weightsresponse"></a>
### Model: WeightsResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `champion_uid` |  |  | opt | `integer` (nullable) |
| `weights` |  |  | req | `object` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "champion_uid": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Champion Uid"
    },
    "weights": {
      "additionalProperties": {
        "type": "number"
      },
      "title": "Weights",
      "type": "object"
    }
  },
  "required": [
    "weights"
  ],
  "title": "WeightsResponse",
  "type": "object"
}
```

</details>
