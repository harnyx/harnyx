# Platform API reference (generated)

Generated from FastAPI OpenAPI.

## Domains
- [admin](#admin)
  - [POST /v1/admin/users](#endpoint-post-v1-admin-users)
- [console](#console)
  - [GET /v1/console/session](#endpoint-get-v1-console-session)
  - [POST /v1/console/session](#endpoint-post-v1-console-session)
  - [DELETE /v1/console/session](#endpoint-delete-v1-console-session)
- [feeds](#feeds)
  - [GET /v1/feeds](#endpoint-get-v1-feeds)
  - [POST /v1/feeds](#endpoint-post-v1-feeds)
  - [GET /v1/feeds/runs/in-progress](#endpoint-get-v1-feeds-runs-in-progress)
  - [POST /v1/feeds/search](#endpoint-post-v1-feeds-search)
  - [GET /v1/feeds/{feed_id}](#endpoint-get-v1-feeds-feed_id)
  - [PUT /v1/feeds/{feed_id}](#endpoint-put-v1-feeds-feed_id)
  - [POST /v1/feeds/{feed_id}/force-run](#endpoint-post-v1-feeds-feed_id-force-run)
  - [PUT /v1/feeds/{feed_id}/pause](#endpoint-put-v1-feeds-feed_id-pause)
  - [GET /v1/feeds/{feed_id}/runs](#endpoint-get-v1-feeds-feed_id-runs)
  - [PUT /v1/feeds/{feed_id}/runs/{run_date}/override](#endpoint-put-v1-feeds-feed_id-runs-run_date-override)
  - [DELETE /v1/feeds/{feed_id}/runs/{run_date}/override](#endpoint-delete-v1-feeds-feed_id-runs-run_date-override)
  - [POST /v1/feeds/{feed_id}/runs/{run_id}/cancel](#endpoint-post-v1-feeds-feed_id-runs-run_id-cancel)
  - [POST /v1/feeds/{feed_id}/runs/{run_id}/digest/regenerate](#endpoint-post-v1-feeds-feed_id-runs-run_id-digest-regenerate)
  - [GET /v1/feeds/{feed_id}/runs/{run_id}/items](#endpoint-get-v1-feeds-feed_id-runs-run_id-items)
  - [POST /v1/feeds/{feed_id}/runs/{run_id}/items/{job_id}/exclude](#endpoint-post-v1-feeds-feed_id-runs-run_id-items-job_id-exclude)
  - [POST /v1/feeds/{feed_id}/submissions](#endpoint-post-v1-feeds-feed_id-submissions)
  - [GET /v1/feeds/{feed_id}/submissions/{job_id}](#endpoint-get-v1-feeds-feed_id-submissions-job_id)
  - [POST /v1/feeds/{feed_id}/tool/search](#endpoint-post-v1-feeds-feed_id-tool-search)
- [internal](#internal)
  - [POST /v1/internal/query/execute](#endpoint-post-v1-internal-query-execute)
- [manual-evals](#manual-evals)
  - [GET /v1/manual-evals](#endpoint-get-v1-manual-evals)
  - [POST /v1/manual-evals](#endpoint-post-v1-manual-evals)
  - [GET /v1/manual-evals/{job_id}](#endpoint-get-v1-manual-evals-job_id)
- [miner-task-batches](#miner-task-batches)
  - [POST /v1/miner-task-batches/batch](#endpoint-post-v1-miner-task-batches-batch)
  - [GET /v1/miner-task-batches/batch/{batch_id}](#endpoint-get-v1-miner-task-batches-batch-batch_id)
  - [GET /v1/miner-task-batches/progress/{batch_id}](#endpoint-get-v1-miner-task-batches-progress-batch_id)
  - [GET /v1/miner-task-batches/{batch_id}/artifacts/{artifact_id}](#endpoint-get-v1-miner-task-batches-batch_id-artifacts-artifact_id)
- [miners](#miners)
  - [POST /v1/miners/scripts](#endpoint-post-v1-miners-scripts)
- [monitoring](#monitoring)
  - [GET /v1/monitoring/miner-scripts/{artifact_id}](#endpoint-get-v1-monitoring-miner-scripts-artifact_id)
  - [GET /v1/monitoring/miner-task-batches](#endpoint-get-v1-monitoring-miner-task-batches)
  - [GET /v1/monitoring/miner-task-batches/{batch_id}](#endpoint-get-v1-monitoring-miner-task-batches-batch_id)
  - [GET /v1/monitoring/miner-task-batches/{batch_id}/results](#endpoint-get-v1-monitoring-miner-task-batches-batch_id-results)
  - [GET /v1/monitoring/overview](#endpoint-get-v1-monitoring-overview)
- [public](#public)
  - [GET /v1/public/feeds/top-scoring-posts](#endpoint-get-v1-public-feeds-top-scoring-posts)
  - [GET /v1/public/feeds/{feed_id}/leaderboard/today](#endpoint-get-v1-public-feeds-feed_id-leaderboard-today)
  - [GET /v1/public/feeds/{feed_id}/run-dates/{run_date}/digest](#endpoint-get-v1-public-feeds-feed_id-run-dates-run_date-digest)
  - [GET /v1/public/feeds/{feed_id}/runs](#endpoint-get-v1-public-feeds-feed_id-runs)
  - [GET /v1/public/feeds/{feed_id}/runs/{run_date}](#endpoint-get-v1-public-feeds-feed_id-runs-run_date)
  - [GET /v1/public/feeds/{feed_id}/runs/{run_id}/digest](#endpoint-get-v1-public-feeds-feed_id-runs-run_id-digest)
  - [GET /v1/public/feeds/{feed_id}/submissions/{job_id}](#endpoint-get-v1-public-feeds-feed_id-submissions-job_id)
- [query](#query)
  - [POST /v1/query/execute](#endpoint-post-v1-query-execute)
- [repo-search](#repo-search)
  - [POST /v1/repo-search/ensure-index](#endpoint-post-v1-repo-search-ensure-index)
  - [POST /v1/repo-search/get-file](#endpoint-post-v1-repo-search-get-file)
  - [POST /v1/repo-search/search](#endpoint-post-v1-repo-search-search)
  - [POST /v1/repo-search/tool/get-file](#endpoint-post-v1-repo-search-tool-get-file)
  - [POST /v1/repo-search/tool/search](#endpoint-post-v1-repo-search-tool-search)
- [tools](#tools)
  - [POST /v1/tools/execute](#endpoint-post-v1-tools-execute)
- [validators](#validators)
  - [POST /v1/validators/register](#endpoint-post-v1-validators-register)
- [weights](#weights)
  - [GET /v1/weights](#endpoint-get-v1-weights)
- [Misc](#misc)
  - [GET /healthz](#endpoint-get-healthz)
  - [GET /metrics](#endpoint-get-metrics)

## admin

### users

<a id="endpoint-post-v1-admin-users"></a>
#### POST /v1/admin/users

Create an identity-less user and issue a one-time API key (admin only).

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ApiKey OR ConsoleSessionCookie

**Request**
Content-Type: `application/json`
Body: [AdminCreateUserRequestModel](#model-admincreateuserrequestmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `role` |  |  | req | `string` (enum: [member, admin]) |
| `username` |  |  | req | `string` |

**Responses**
`201` Successful Response
Content-Type: `application/json`
Body: [AdminCreateUserResponseModel](#model-admincreateuserresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `api_key` |  |  | req | `string` |
| `api_key_id` |  |  | req | `string` (format: uuid) |
| `status` |  |  | req | `string` |
| `user_id` |  |  | req | `string` (format: uuid) |
| `username` |  |  | req | `string` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |



## console

### session

<a id="endpoint-get-v1-console-session"></a>
#### GET /v1/console/session

Return current console session status from bearer auth or session cookie.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ConsoleSessionCookie

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [ConsoleSessionResponseModel](#model-consolesessionresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `is_admin` |  |  | req | `boolean` |
| `label` |  |  | req | `string` |
| `status` |  |  | req | `string` |


<a id="endpoint-post-v1-console-session"></a>
#### POST /v1/console/session

Exchange Google bearer auth for a persisted console session cookie.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`)

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [ConsoleSessionResponseModel](#model-consolesessionresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `is_admin` |  |  | req | `boolean` |
| `label` |  |  | req | `string` |
| `status` |  |  | req | `string` |


<a id="endpoint-delete-v1-console-session"></a>
#### DELETE /v1/console/session

Clear persisted console session cookie.

**Auth**: None.

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [StatusResponse](#model-statusresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `status` |  |  | req | `string` |



## feeds

<a id="endpoint-get-v1-feeds"></a>
### GET /v1/feeds

List feeds with their current state and latest run snapshot.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ConsoleSessionCookie

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [FeedsListResponseModel](#model-feedslistresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `feeds` |  |  | req | array[[FeedReadModel](#model-feedreadmodel)] |
|  | `created_at` |  | req | `string` (format: date-time) |
|  | `feed_type` |  | req | `string` |
|  | `id` |  | req | `string` (format: uuid) |
|  | `interval_hours` |  | opt | `integer` (nullable) |
|  | `is_paused` |  | req | `boolean` |
|  | `last_error` |  | opt | `string` (nullable) |
|  | `latest_run` |  | opt | [FeedRunReadModel](#model-feedrunreadmodel) (nullable) |
|  |  | `attempt_number` | req | `integer` |
|  |  | `cancel_requested_at` | opt | `string` (format: date-time; nullable) |
|  |  | `completed_at` | opt | `string` (format: date-time; nullable) |
|  |  | `error` | opt | `string` (nullable) |
|  |  | `id` | req | `string` (format: uuid) |
|  |  | `slot_at` | req | `string` (format: date-time) |
|  |  | `started_at` | req | `string` (format: date-time) |
|  |  | `status` | req | `string` |
|  |  | `trigger_kind` | req | `string` |
|  | `next_run_at` |  | opt | `string` (format: date-time; nullable) |
|  | `progress` |  | opt | [FeedProgressModel](#model-feedprogressmodel) |
|  |  | `accepted` | opt | `integer` (default: 0) |
|  |  | `content_review_complete` | opt | `boolean` (default: False) |
|  |  | `failed` | opt | `integer` (default: 0) |
|  |  | `queued` | opt | `integer` (default: 0) |
|  |  | `rejected` | opt | `integer` (default: 0) |
|  |  | `running` | opt | `integer` (default: 0) |
|  |  | `succeeded` | opt | `integer` (default: 0) |
|  | `rubric` |  | req | [FeedRubricReadModel](#model-feedrubricreadmodel) |
|  |  | `description` | req | `string` |
|  |  | `id` | req | `string` (format: uuid) |
|  |  | `slug` | req | `string` |
|  |  | `title` | req | `string` |
|  | `schedule_first_run_at` |  | opt | `string` (format: date-time; nullable) |
|  | `state` |  | req | `string` |
|  | `submission_access_policy` |  | opt | `string` (enum: [owner_editor_only, authenticated_any, paid_any, authenticated_or_paid]; default: owner_editor_only) |
|  | `topic` |  | req | [FeedTopicReadModel](#model-feedtopicreadmodel) |
|  |  | `criterion_id` | req | `string` |
|  |  | `description` | req | `string` |
|  |  | `id` | req | `string` (format: uuid) |
|  |  | `keyword` | req | `string` |
|  |  | `title` | req | `string` |
|  | `trigger_kind` |  | req | `string` |
|  | `updated_at` |  | req | `string` (format: date-time) |


<a id="endpoint-post-v1-feeds"></a>
### POST /v1/feeds

Create a new feed and return its id.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ApiKey OR ConsoleSessionCookie

**Request**
Content-Type: `application/json`
Body: [FeedCreateRequestModel-Input](#model-feedcreaterequestmodel-input)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `digest_threshold_pct` |  |  | opt | `integer` (default: 70) |
| `feed_type` |  |  | opt | `string` (enum: [batch, continuous]; default: batch) |
| `item_limit` |  |  | opt | `integer` (nullable) |
| `lookback_minutes` |  |  | opt | `integer` (nullable) |
| `providers` |  |  | req | array[`string`] |
| `rubric` |  |  | req | [InlineRubricRequestModel-Input](#model-inlinerubricrequestmodel-input) |
|  | `criteria` |  | req | array[[InlineRubricCriterionModel](#model-inlinerubriccriterionmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `weight_pct` | req | `integer` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
| `schedule` |  |  | req | oneOf: [FeedManualSchedule](#model-feedmanualschedule) OR [FeedScheduledSchedule](#model-feedscheduledschedule) |
| `submission_access_policy` |  |  | opt | `string` (enum: [owner_editor_only, authenticated_any, paid_any, authenticated_or_paid]; default: owner_editor_only) |
| `topic_description` |  |  | opt | `string` (nullable) |
| `topic_keyword` |  |  | req | `string` |
| `topic_title` |  |  | req | `string` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [FeedCreateResponseModel](#model-feedcreateresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `feed_id` |  |  | req | `string` (format: uuid) |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


### runs

#### in-progress

<a id="endpoint-get-v1-feeds-runs-in-progress"></a>
##### GET /v1/feeds/runs/in-progress

List all feed runs currently in progress (admin only).

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ConsoleSessionCookie

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `limit` | query | opt | `integer` (default: 200) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [InProgressFeedRunsResponseModel](#model-inprogressfeedrunsresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `limit` |  |  | req | `integer` |
| `runs` |  |  | req | array[[FeedRunInProgressModel](#model-feedruninprogressmodel)] |
|  | `attempt_number` |  | req | `integer` |
|  | `cancel_requested_at` |  | opt | `string` (format: date-time; nullable) |
|  | `feed_id` |  | req | `string` (format: uuid) |
|  | `run_id` |  | req | `string` (format: uuid) |
|  | `slot_at` |  | req | `string` (format: date-time) |
|  | `started_at` |  | req | `string` (format: date-time) |
|  | `status` |  | req | `string` |
|  | `trigger_kind` |  | req | `string` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


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
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


### {feed_id}

<a id="endpoint-get-v1-feeds-feed_id"></a>
#### GET /v1/feeds/{feed_id}

Fetch the full configuration for a single feed.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ApiKey OR ConsoleSessionCookie

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [FeedCreateRequestModel-Output](#model-feedcreaterequestmodel-output)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `digest_threshold_pct` |  |  | opt | `integer` (default: 70) |
| `feed_type` |  |  | opt | `string` (enum: [batch, continuous]; default: batch) |
| `item_limit` |  |  | opt | `integer` (nullable) |
| `lookback_minutes` |  |  | opt | `integer` (nullable) |
| `providers` |  |  | req | array[`string`] |
| `rubric` |  |  | req | [InlineRubricRequestModel-Output](#model-inlinerubricrequestmodel-output) |
|  | `criteria` |  | req | array[[InlineRubricCriterionModel](#model-inlinerubriccriterionmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `weight_pct` | req | `integer` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
| `schedule` |  |  | req | oneOf: [FeedManualSchedule](#model-feedmanualschedule) OR [FeedScheduledSchedule](#model-feedscheduledschedule) |
| `submission_access_policy` |  |  | opt | `string` (enum: [owner_editor_only, authenticated_any, paid_any, authenticated_or_paid]; default: owner_editor_only) |
| `topic_description` |  |  | opt | `string` (nullable) |
| `topic_keyword` |  |  | req | `string` |
| `topic_title` |  |  | req | `string` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


<a id="endpoint-put-v1-feeds-feed_id"></a>
#### PUT /v1/feeds/{feed_id}

Update an existing feed definition.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ApiKey OR ConsoleSessionCookie

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |

**Request**
Content-Type: `application/json`
Body: [FeedUpdateRequestModel](#model-feedupdaterequestmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `digest_threshold_pct` |  |  | opt | `integer` (nullable) |
| `feed_type` |  |  | opt | `string` (enum: [batch, continuous]; nullable) |
| `item_limit` |  |  | opt | `integer` (nullable) |
| `lookback_minutes` |  |  | opt | `integer` (nullable) |
| `providers` |  |  | opt | array[`string`] (nullable) |
| `rubric` |  |  | opt | [InlineRubricRequestModel-Input](#model-inlinerubricrequestmodel-input) (nullable) |
|  | `criteria` |  | req | array[[InlineRubricCriterionModel](#model-inlinerubriccriterionmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `weight_pct` | req | `integer` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
| `schedule` |  |  | opt | oneOf: [FeedManualSchedule](#model-feedmanualschedule) OR [FeedScheduledSchedule](#model-feedscheduledschedule) (nullable) |
| `submission_access_policy` |  |  | opt | `string` (enum: [owner_editor_only, authenticated_any, paid_any, authenticated_or_paid]; nullable) |
| `topic_description` |  |  | opt | `string` (nullable) |
| `topic_keyword` |  |  | opt | `string` (nullable) |
| `topic_title` |  |  | opt | `string` (nullable) |

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
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


#### force-run

<a id="endpoint-post-v1-feeds-feed_id-force-run"></a>
##### POST /v1/feeds/{feed_id}/force-run

Trigger a feed run immediately (outside its schedule).

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ConsoleSessionCookie

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |

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
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


#### pause

<a id="endpoint-put-v1-feeds-feed_id-pause"></a>
##### PUT /v1/feeds/{feed_id}/pause

Pause or unpause a feed (stops new runs; continuous feeds also reject new submissions).

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ApiKey OR ConsoleSessionCookie

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |

**Request**
Content-Type: `application/json`
Body: [FeedPauseRequestModel](#model-feedpauserequestmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `is_paused` |  |  | req | `boolean` |

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
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


#### runs

<a id="endpoint-get-v1-feeds-feed_id-runs"></a>
##### GET /v1/feeds/{feed_id}/runs

List runs for a feed.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ConsoleSessionCookie

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |
| `limit` | query | opt | `integer` (default: 20) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [FeedRunsResponseModel](#model-feedrunsresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `runs` |  |  | req | array[[FeedRunReadModel](#model-feedrunreadmodel)] |
|  | `attempt_number` |  | req | `integer` |
|  | `cancel_requested_at` |  | opt | `string` (format: date-time; nullable) |
|  | `completed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `error` |  | opt | `string` (nullable) |
|  | `id` |  | req | `string` (format: uuid) |
|  | `slot_at` |  | req | `string` (format: date-time) |
|  | `started_at` |  | req | `string` (format: date-time) |
|  | `status` |  | req | `string` |
|  | `trigger_kind` |  | req | `string` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


##### {run_date}

###### override

<a id="endpoint-put-v1-feeds-feed_id-runs-run_date-override"></a>
###### PUT /v1/feeds/{feed_id}/runs/{run_date}/override

Override a feed's run for a given date.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ConsoleSessionCookie

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |
| `run_date` | path | req | `string` (format: date) |

**Request**
Content-Type: `application/json`
Body: [FeedRunDateOverrideRequestModel](#model-feedrundateoverriderequestmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `run_id` |  |  | req | `string` (format: uuid) |

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
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


<a id="endpoint-delete-v1-feeds-feed_id-runs-run_date-override"></a>
###### DELETE /v1/feeds/{feed_id}/runs/{run_date}/override

Clear a feed run-date override.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ConsoleSessionCookie

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |
| `run_date` | path | req | `string` (format: date) |

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
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


##### {run_id}

###### cancel

<a id="endpoint-post-v1-feeds-feed_id-runs-run_id-cancel"></a>
###### POST /v1/feeds/{feed_id}/runs/{run_id}/cancel

Request cancellation of a running feed run.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ConsoleSessionCookie

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |
| `run_id` | path | req | `string` (format: uuid) |

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
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


###### digest

###### regenerate

<a id="endpoint-post-v1-feeds-feed_id-runs-run_id-digest-regenerate"></a>
###### POST /v1/feeds/{feed_id}/runs/{run_id}/digest/regenerate

Queue digest re-generation for a succeeded run.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ConsoleSessionCookie

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |
| `run_id` | path | req | `string` (format: uuid) |

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
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


###### items

<a id="endpoint-get-v1-feeds-feed_id-runs-run_id-items"></a>
###### GET /v1/feeds/{feed_id}/runs/{run_id}/items

List items for a run, including evaluation results.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ConsoleSessionCookie

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |
| `run_id` | path | req | `string` (format: uuid) |
| `limit` | query | opt | `integer` (default: 50) |
| `sort` | query | opt | [FeedRunItemSort](#model-feedrunitemsort) (default: score) |
| `status` | query | opt | [FeedRunItemBucket](#model-feedrunitembucket) (default: all) |
| `q` | query | opt | `string` (nullable) |
| `cursor` | query | opt | `string` (nullable) |
| `exclude` | query | opt | `boolean` (default: False) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [FeedRunItemsResponseModel](#model-feedrunitemsresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `cursor` |  |  | opt | `string` (nullable) |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `items` |  |  | req | array[[FeedRunItemReadModel](#model-feedrunitemreadmodel)] |
|  | `author` |  | opt | `string` (nullable) |
|  | `content_id` |  | req | `string` (format: uuid) |
|  | `content_review_rubric_result` |  | opt | [ExternalEvalResultModel](#model-externalevalresultmodel) (nullable) |
|  |  | `criteria` | req | array[[CriterionAssessmentModel](#model-criterionassessmentmodel)] |
|  |  | `overall_rationale` | opt | `string` (nullable) |
|  |  | `rubric_id` | req | `string` |
|  |  | `rubric_score` | req | `number` |
|  | `content_review_topic_gate` |  | req | [TopicGateModel](#model-topicgatemodel) |
|  |  | `criteria` | opt | array[[CriterionAssessmentModel](#model-criterionassessmentmodel)] (default: []) |
|  |  | `score` | opt | `number` (nullable) |
|  | `decision` |  | opt | `string` (nullable) |
|  | `evaluated_at` |  | opt | `string` (format: date-time; nullable) |
|  | `external_id` |  | req | `string` |
|  | `is_excluded` |  | opt | `boolean` (default: False) |
|  | `job_error_code` |  | opt | `string` (nullable) |
|  | `job_error_message` |  | opt | `string` (nullable) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `job_status` |  | req | `string` |
|  | `provider` |  | req | `string` |
|  | `provider_context` |  | opt | `object` (nullable) |
|  | `requested_at` |  | req | `string` (format: date-time) |
|  | `source_created_at` |  | opt | `string` (format: date-time; nullable) |
|  | `text` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (nullable) |
| `limit` |  |  | req | `integer` |
| `next_cursor` |  |  | opt | `string` (nullable) |
| `q` |  |  | opt | `string` (nullable) |
| `run_id` |  |  | req | `string` (format: uuid) |
| `sort` |  |  | req | `string` |
| `status` |  |  | req | `string` |
| `total_count` |  |  | req | `integer` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


###### {job_id}

###### exclude

<a id="endpoint-post-v1-feeds-feed_id-runs-run_id-items-job_id-exclude"></a>
###### POST /v1/feeds/{feed_id}/runs/{run_id}/items/{job_id}/exclude

Set or toggle exclusion for a feed run item.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ConsoleSessionCookie

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |
| `run_id` | path | req | `string` (format: uuid) |
| `job_id` | path | req | `string` (format: uuid) |

**Request**
Content-Type: `application/json`
Body: [FeedRunItemExcludeRequestModel](#model-feedrunitemexcluderequestmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `is_excluded` |  |  | opt | `boolean` (nullable) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [FeedRunItemExcludeResponseModel](#model-feedrunitemexcluderesponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `is_excluded` |  |  | req | `boolean` |
| `job_id` |  |  | req | `string` (format: uuid) |
| `run_id` |  |  | req | `string` (format: uuid) |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


#### submissions

<a id="endpoint-post-v1-feeds-feed_id-submissions"></a>
##### POST /v1/feeds/{feed_id}/submissions

Submit content for realtime evaluation (API key, Google, or x402 fallback).

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ApiKey OR ConsoleSessionCookie

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |

**Request**
Content-Type: `application/json`
Body: oneOf: [FeedSubmissionUrlModel](#model-feedsubmissionurlmodel) OR [FeedSubmissionOriginalModel](#model-feedsubmissionoriginalmodel) OR [FeedSubmissionRepoDiffModel](#model-feedsubmissionrepodiffmodel)

(no documented fields)

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [FeedSubmissionCreateResponseModel](#model-feedsubmissioncreateresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `job_id` |  |  | req | `string` (format: uuid) |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


##### {job_id}

<a id="endpoint-get-v1-feeds-feed_id-submissions-job_id"></a>
###### GET /v1/feeds/{feed_id}/submissions/{job_id}

Fetch submission status/results by job id (console or API key).

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ApiKey OR ConsoleSessionCookie

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |
| `job_id` | path | req | `string` (format: uuid) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [FeedRunItemReadModel](#model-feedrunitemreadmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `author` |  |  | opt | `string` (nullable) |
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
| `content_review_topic_gate` |  |  | req | [TopicGateModel](#model-topicgatemodel) |
|  | `criteria` |  | opt | array[[CriterionAssessmentModel](#model-criterionassessmentmodel)] (default: []) |
|  |  | `aggregate_score` | req | `number` |
|  |  | `criterion_evaluations` | req | array[[CriterionEvaluationModel](#model-criterionevaluationmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  | `score` |  | opt | `number` (nullable) |
| `decision` |  |  | opt | `string` (nullable) |
| `evaluated_at` |  |  | opt | `string` (format: date-time; nullable) |
| `external_id` |  |  | req | `string` |
| `is_excluded` |  |  | opt | `boolean` (default: False) |
| `job_error_code` |  |  | opt | `string` (nullable) |
| `job_error_message` |  |  | opt | `string` (nullable) |
| `job_id` |  |  | req | `string` (format: uuid) |
| `job_status` |  |  | req | `string` |
| `provider` |  |  | req | `string` |
| `provider_context` |  |  | opt | `object` (nullable) |
| `requested_at` |  |  | req | `string` (format: date-time) |
| `source_created_at` |  |  | opt | `string` (format: date-time; nullable) |
| `text` |  |  | req | `string` |
| `title` |  |  | opt | `string` (nullable) |
| `url` |  |  | opt | `string` (nullable) |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


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
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |



## internal

### query

#### execute

<a id="endpoint-post-v1-internal-query-execute"></a>
##### POST /v1/internal/query/execute

Execute an internal query on the worker and return the selected artifact result.

**Auth**: Internal secret (`x-internal-secret` header)

**Request**
Content-Type: `application/json`
Body: [Query](#model-query)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `text` |  |  | req | `string` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [QueryExecutionResultDTO](#model-queryexecutionresultdto)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `response` |  |  | req | [Response](#model-response) |
|  | `text` |  | req | `string` |
| `uid` |  |  | req | `integer` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |



## manual-evals

<a id="endpoint-get-v1-manual-evals"></a>
### GET /v1/manual-evals

List recent manual evaluation jobs.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ConsoleSessionCookie

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `limit` | query | opt | `integer` (default: 20) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [ManualEvalJobsResponseModel](#model-manualevaljobsresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `jobs` |  |  | req | array[[ManualEvalJobSummaryModel](#model-manualevaljobsummarymodel)] |
|  | `author` |  | opt | `string` (nullable) |
|  | `evaluated_at` |  | opt | `string` (format: date-time; nullable) |
|  | `external_id` |  | req | `string` |
|  | `job_error_code` |  | opt | `string` (nullable) |
|  | `job_error_message` |  | opt | `string` (nullable) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `job_status` |  | req | `string` |
|  | `requested_at` |  | req | `string` (format: date-time) |
|  | `rubric_slug` |  | req | `string` |
|  | `rubric_title` |  | req | `string` |
|  | `source_created_at` |  | opt | `string` (format: date-time; nullable) |
|  | `text` |  | req | `string` |
|  | `url` |  | opt | `string` (nullable) |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


<a id="endpoint-post-v1-manual-evals"></a>
### POST /v1/manual-evals

Queue an on-demand evaluation for a single URL.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ConsoleSessionCookie

**Request**
Content-Type: `application/json`
Body: [ManualEvalCreateRequestModel](#model-manualevalcreaterequestmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `rubric` |  |  | req | [InlineRubricRequestModel-Input](#model-inlinerubricrequestmodel-input) |
|  | `criteria` |  | req | array[[InlineRubricCriterionModel](#model-inlinerubriccriterionmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `weight_pct` | req | `integer` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
| `url` |  |  | req | `string` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [ManualEvalCreateResponseModel](#model-manualevalcreateresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `job_id` |  |  | req | `string` (format: uuid) |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


### {job_id}

<a id="endpoint-get-v1-manual-evals-job_id"></a>
#### GET /v1/manual-evals/{job_id}

Fetch the evaluated item (and rubric results) for a manual evaluation job.

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ConsoleSessionCookie

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `job_id` | path | req | `string` (format: uuid) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [FeedRunItemReadModel](#model-feedrunitemreadmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `author` |  |  | opt | `string` (nullable) |
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
| `content_review_topic_gate` |  |  | req | [TopicGateModel](#model-topicgatemodel) |
|  | `criteria` |  | opt | array[[CriterionAssessmentModel](#model-criterionassessmentmodel)] (default: []) |
|  |  | `aggregate_score` | req | `number` |
|  |  | `criterion_evaluations` | req | array[[CriterionEvaluationModel](#model-criterionevaluationmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  | `score` |  | opt | `number` (nullable) |
| `decision` |  |  | opt | `string` (nullable) |
| `evaluated_at` |  |  | opt | `string` (format: date-time; nullable) |
| `external_id` |  |  | req | `string` |
| `is_excluded` |  |  | opt | `boolean` (default: False) |
| `job_error_code` |  |  | opt | `string` (nullable) |
| `job_error_message` |  |  | opt | `string` (nullable) |
| `job_id` |  |  | req | `string` (format: uuid) |
| `job_status` |  |  | req | `string` |
| `provider` |  |  | req | `string` |
| `provider_context` |  |  | opt | `object` (nullable) |
| `requested_at` |  |  | req | `string` (format: date-time) |
| `source_created_at` |  |  | opt | `string` (format: date-time; nullable) |
| `text` |  |  | req | `string` |
| `title` |  |  | opt | `string` (nullable) |
| `url` |  |  | opt | `string` (nullable) |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |



## miner-task-batches

### batch

<a id="endpoint-post-v1-miner-task-batches-batch"></a>
#### POST /v1/miner-task-batches/batch

Owner emergency recovery route. Each request force-creates a fresh batch, is not replay-safe, and fails fast with 409 while batch creation is already in progress or another batch is running. Returns an explicit terminal delivery failure when every observed validator definitively rejects or misses dispatch.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

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

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [MinerTaskBatchModel](#model-minertaskbatchmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifacts` |  |  | req | array[[ScriptArtifactModel](#model-scriptartifactmodel)] |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `content_hash` |  | req | `string` |
|  | `size_bytes` |  | req | `integer` |
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
|  |  | `text` | req | `string` |
|  | `reference_answer` |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `text` | req | `string` |
|  | `task_id` |  | req | `string` (format: uuid) |

`409` Batch creation already in progress or another batch is running.
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
| `artifacts` |  |  | req | array[[ScriptArtifactModel](#model-scriptartifactmodel)] |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `content_hash` |  | req | `string` |
|  | `size_bytes` |  | req | `integer` |
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
|  |  | `text` | req | `string` |
|  | `reference_answer` |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `text` | req | `string` |
|  | `task_id` |  | req | `string` (format: uuid) |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


### progress

#### {batch_id}

<a id="endpoint-get-v1-miner-task-batches-progress-batch_id"></a>
##### GET /v1/miner-task-batches/progress/{batch_id}

Return a lightweight progress snapshot for a batch.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `batch_id` | path | req | `string` (format: uuid) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [ProgressSnapshotResponse](#model-progresssnapshotresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_count` |  |  | req | `integer` |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `created_at` |  |  | req | `string` (format: date-time) |
| `cutoff_at` |  |  | req | `string` (format: date-time) |
| `status` |  |  | req | `string` |
| `task_count` |  |  | req | `integer` |

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
| `uid` |  |  | req | `integer` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |



## monitoring

### miner-scripts

#### {artifact_id}

<a id="endpoint-get-v1-monitoring-miner-scripts-artifact_id"></a>
##### GET /v1/monitoring/miner-scripts/{artifact_id}

Fetch miner script metadata and content.

**Auth**: None.

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `artifact_id` | path | req | `string` (format: uuid) |
| `include_content` | query | opt | `boolean` (default: False) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [MonitoringScriptModel](#model-monitoringscriptmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `content_b64` |  |  | opt | `string` (nullable) |
| `content_hash` |  |  | req | `string` |
| `revealed_at` |  |  | opt | `string` (format: date-time; nullable) |
| `size_bytes` |  |  | req | `integer` |
| `submitted_at` |  |  | req | `string` (format: date-time) |
| `uid` |  |  | req | `integer` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


### miner-task-batches

<a id="endpoint-get-v1-monitoring-miner-task-batches"></a>
#### GET /v1/monitoring/miner-task-batches

List recent miner-task batches.

**Auth**: None.

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `limit` | query | opt | `integer` (default: 20) |
| `before` | query | opt | `string` (format: date-time; nullable) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [MonitoringBatchListModel](#model-monitoringbatchlistmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batches` |  |  | req | array[[MonitoringBatchSummaryModel](#model-monitoringbatchsummarymodel)] |
|  | `artifact_count` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `champion_artifact_id` |  | opt | `string` (format: uuid; nullable) |
|  | `completed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `created_at` |  | req | `string` (format: date-time) |
|  | `cutoff_at` |  | req | `string` (format: date-time) |
|  | `failed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `status` |  | req | `string` |
|  | `task_count` |  | req | `integer` |
| `before` |  |  | opt | `string` (format: date-time; nullable) |
| `limit` |  |  | req | `integer` |
| `next_before` |  |  | opt | `string` (format: date-time; nullable) |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


#### {batch_id}

<a id="endpoint-get-v1-monitoring-miner-task-batches-batch_id"></a>
##### GET /v1/monitoring/miner-task-batches/{batch_id}

Fetch a batch (tasks + artifacts) plus aggregates.

**Auth**: None.

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `batch_id` | path | req | `string` (format: uuid) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [MonitoringBatchDetailModel](#model-monitoringbatchdetailmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_aggregates` |  |  | req | array[[MonitoringArtifactAggregateModel](#model-monitoringartifactaggregatemodel)] |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `avg_score` |  | req | `number` |
|  | `completed_run_count` |  | req | `integer` |
|  | `cost_totals` |  | req | [MonitoringCostTotalsModel](#model-monitoringcosttotalsmodel) |
|  |  | `llm_call_count` | req | `integer` |
|  |  | `llm_cost_usd` | req | `number` |
|  |  | `llm_total_tokens` | req | `integer` |
|  |  | `search_tool_call_count` | req | `integer` |
|  |  | `search_tool_cost_usd` | req | `number` |
|  |  | `total_cost_usd` | req | `number` |
|  | `error_counts` |  | opt | array[[MonitoringErrorCountModel](#model-monitoringerrorcountmodel)] (default: []) |
|  |  | `count` | req | `integer` |
|  |  | `error_code` | req | `string` |
|  | `expected_run_count` |  | req | `integer` |
|  | `llm_models` |  | opt | array[[MonitoringLlmModelUsageModel](#model-monitoringllmmodelusagemodel)] (default: []) |
|  |  | `call_count` | req | `integer` |
|  |  | `completion_tokens` | req | `integer` |
|  |  | `cost_usd` | req | `number` |
|  |  | `model` | req | `string` |
|  |  | `prompt_tokens` | req | `integer` |
|  |  | `total_tokens` | req | `integer` |
|  | `miner_uid` |  | req | `integer` |
|  | `total_score` |  | req | `number` |
| `batch` |  |  | req | [MinerTaskBatchModel](#model-minertaskbatchmodel) |
|  | `artifacts` |  | req | array[[ScriptArtifactModel](#model-scriptartifactmodel)] |
|  |  | `artifact_id` | req | `string` (format: uuid) |
|  |  | `content_hash` | req | `string` |
|  |  | `size_bytes` | req | `integer` |
|  |  | `uid` | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `champion_artifact_id` |  | req | `string` (format: uuid; nullable) |
|  | `completed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `created_at` |  | req | `string` (format: date-time) |
|  | `cutoff_at` |  | req | `string` (format: date-time) |
|  | `failed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `tasks` |  | req | array[[MinerTask](#model-minertask)] |
|  |  | `budget_usd` | opt | `number` (default: 0.05) |
|  |  | `query` | req | [Query](#model-query) |
|  |  | `reference_answer` | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `task_id` | req | `string` (format: uuid) |
| `completed_validator_hotkeys` |  |  | req | array[`string`] |
| `cost_totals` |  |  | req | [MonitoringCostTotalsModel](#model-monitoringcosttotalsmodel) |
|  | `llm_call_count` |  | req | `integer` |
|  | `llm_cost_usd` |  | req | `number` |
|  | `llm_total_tokens` |  | req | `integer` |
|  | `search_tool_call_count` |  | req | `integer` |
|  | `search_tool_cost_usd` |  | req | `number` |
|  | `total_cost_usd` |  | req | `number` |
| `dispatch_failed_validator_hotkeys` |  |  | req | array[`string`] |
| `pending_validator_hotkeys` |  |  | req | array[`string`] |
| `summary` |  |  | req | [MonitoringBatchSummaryModel](#model-monitoringbatchsummarymodel) |
|  | `artifact_count` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `champion_artifact_id` |  | opt | `string` (format: uuid; nullable) |
|  | `completed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `created_at` |  | req | `string` (format: date-time) |
|  | `cutoff_at` |  | req | `string` (format: date-time) |
|  | `failed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `status` |  | req | `string` |
|  | `task_count` |  | req | `integer` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


##### results

<a id="endpoint-get-v1-monitoring-miner-task-batches-batch_id-results"></a>
###### GET /v1/monitoring/miner-task-batches/{batch_id}/results

List completed per-task run rows for a batch.

**Auth**: None.

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `batch_id` | path | req | `string` (format: uuid) |
| `include_payload` | query | opt | `boolean` (default: False) |
| `miner_uid` | query | opt | `integer` (nullable) |
| `artifact_id` | query | opt | `string` (format: uuid; nullable) |
| `task_id` | query | opt | `string` (format: uuid; nullable) |
| `validator_hotkey` | query | opt | `string` (nullable) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: array[[MonitoringResultRowModel](#model-monitoringresultrowmodel)]

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `cost_totals` |  |  | req | [MonitoringCostTotalsModel](#model-monitoringcosttotalsmodel) |
|  | `llm_call_count` |  | req | `integer` |
|  | `llm_cost_usd` |  | req | `number` |
|  | `llm_total_tokens` |  | req | `integer` |
|  | `search_tool_call_count` |  | req | `integer` |
|  | `search_tool_cost_usd` |  | req | `number` |
|  | `total_cost_usd` |  | req | `number` |
| `llm_models` |  |  | opt | array[[MonitoringLlmModelUsageModel](#model-monitoringllmmodelusagemodel)] (default: []) |
|  | `call_count` |  | req | `integer` |
|  | `completion_tokens` |  | req | `integer` |
|  | `cost_usd` |  | req | `number` |
|  | `model` |  | req | `string` |
|  | `prompt_tokens` |  | req | `integer` |
|  | `total_tokens` |  | req | `integer` |
| `miner_uid` |  |  | req | `integer` |
| `payload_json` |  |  | opt | `object` (nullable) |
| `received_at` |  |  | req | `string` (format: date-time) |
| `response` |  |  | opt | [Response](#model-response) (nullable) |
|  | `text` |  | req | `string` |
| `score` |  |  | req | `number` |
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
| `task_id` |  |  | req | `string` (format: uuid) |
| `validator_hotkey` |  |  | req | `string` |
| `validator_uid` |  |  | req | `integer` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


### overview

<a id="endpoint-get-v1-monitoring-overview"></a>
#### GET /v1/monitoring/overview

Monitoring overview.

**Auth**: None.

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [MonitoringOverviewModel](#model-monitoringoverviewmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `champion` |  |  | req | [MonitoringChampionModel](#model-monitoringchampionmodel) |
|  | `miner_uid` |  | opt | `integer` (nullable) |
|  | `script_id` |  | opt | `string` (nullable) |
| `latest_batch` |  |  | opt | [MonitoringBatchSummaryModel](#model-monitoringbatchsummarymodel) (nullable) |
|  | `artifact_count` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `champion_artifact_id` |  | opt | `string` (format: uuid; nullable) |
|  | `completed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `created_at` |  | req | `string` (format: date-time) |
|  | `cutoff_at` |  | req | `string` (format: date-time) |
|  | `failed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `status` |  | req | `string` |
|  | `task_count` |  | req | `integer` |
| `latest_batch_cost_totals` |  |  | opt | [MonitoringCostTotalsModel](#model-monitoringcosttotalsmodel) (nullable) |
|  | `llm_call_count` |  | req | `integer` |
|  | `llm_cost_usd` |  | req | `number` |
|  | `llm_total_tokens` |  | req | `integer` |
|  | `search_tool_call_count` |  | req | `integer` |
|  | `search_tool_cost_usd` |  | req | `number` |
|  | `total_cost_usd` |  | req | `number` |
| `runtime` |  |  | req | [MonitoringRuntimeConfigModel](#model-monitoringruntimeconfigmodel) |
|  | `miner_evaluation_timeout_seconds` |  | req | `number` |
|  | `miner_task_interval_minutes` |  | req | `integer` |
|  | `sandbox_image` |  | req | `string` |
| `validator_endpoints` |  |  | req | array[[MonitoringValidatorEndpointModel](#model-monitoringvalidatorendpointmodel)] |
|  | `base_url` |  | req | `string` |
|  | `first_registered_at` |  | req | `string` (format: date-time) |
|  | `health_checked_at` |  | opt | `string` (format: date-time; nullable) |
|  | `health_error` |  | opt | `string` (nullable) |
|  | `health_status` |  | req | `string` |
|  | `hotkey` |  | req | `string` |
|  | `last_eval_completed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `last_registered_at` |  | req | `string` (format: date-time) |
| `validator_health` |  |  | req | [MonitoringValidatorHealthCountsModel](#model-monitoringvalidatorhealthcountsmodel) |
|  | `healthy` |  | req | `integer` |
|  | `unhealthy` |  | req | `integer` |
|  | `unknown` |  | req | `integer` |
| `weights` |  |  | opt | [MonitoringWeightsModel](#model-monitoringweightsmodel) (nullable) |
|  | `champion_uid` |  | opt | `integer` (nullable) |
|  | `weights` |  | req | `object` |



## public

### feeds

#### top-scoring-posts

<a id="endpoint-get-v1-public-feeds-top-scoring-posts"></a>
##### GET /v1/public/feeds/top-scoring-posts

Fetch a merged list of accepted posts across multiple feeds (public view).

**Auth**: None.

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | query | req | array[`string`] |
| `lookback_hours` | query | opt | `integer` (default: 24) |
| `limit` | query | opt | `integer` (default: 200) |
| `sort` | query | opt | [FeedRunItemSort](#model-feedrunitemsort) (default: score) |
| `q` | query | opt | `string` (nullable) |
| `cursor` | query | opt | `string` (nullable) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [PublicTopScoringPostsResponseModel](#model-publictopscoringpostsresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `cursor` |  |  | opt | `string` (nullable) |
| `feed_ids` |  |  | req | array[`string`] |
| `items` |  |  | req | array[[PublicTopScoringPostModel](#model-publictopscoringpostmodel)] |
|  | `author` |  | opt | `string` (nullable) |
|  | `content_review_rubric_result` |  | opt | [PublicExternalEvalResultModel](#model-publicexternalevalresultmodel) (nullable) |
|  |  | `criteria` | req | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] |
|  |  | `overall_rationale` | opt | `string` (nullable) |
|  |  | `rubric_score` | req | `number` |
|  | `content_review_topic_gate` |  | req | [PublicTopicGateModel](#model-publictopicgatemodel) |
|  |  | `criteria` | opt | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] (default: []) |
|  |  | `score` | opt | `number` (nullable) |
|  | `external_id` |  | req | `string` |
|  | `feed_id` |  | req | `string` (format: uuid) |
|  | `is_excluded` |  | opt | `boolean` (default: False) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `provider_context` |  | opt | `object` (nullable) |
|  | `source_created_at` |  | opt | `string` (format: date-time; nullable) |
|  | `text` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (nullable) |
| `limit` |  |  | req | `integer` |
| `lookback_hours` |  |  | req | `integer` |
| `next_cursor` |  |  | opt | `string` (nullable) |
| `q` |  |  | opt | `string` (nullable) |
| `total_count` |  |  | req | `integer` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


#### {feed_id}

##### leaderboard

###### today

<a id="endpoint-get-v1-public-feeds-feed_id-leaderboard-today"></a>
###### GET /v1/public/feeds/{feed_id}/leaderboard/today

Fetch the current leaderboard: active run if any, else today's succeeded run, else empty.

**Auth**: None.

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |
| `limit` | query | opt | `integer` (default: 200) |
| `sort` | query | opt | [FeedRunItemSort](#model-feedrunitemsort) (default: score) |
| `q` | query | opt | `string` (nullable) |
| `cursor` | query | opt | `string` (nullable) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [PublicFeedLeaderboardResponseModel](#model-publicfeedleaderboardresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `cursor` |  |  | opt | `string` (nullable) |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `items` |  |  | req | array[[PublicFeedRunItemModel](#model-publicfeedrunitemmodel)] |
|  | `author` |  | opt | `string` (nullable) |
|  | `content_review_rubric_result` |  | opt | [PublicExternalEvalResultModel](#model-publicexternalevalresultmodel) (nullable) |
|  |  | `criteria` | req | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] |
|  |  | `overall_rationale` | opt | `string` (nullable) |
|  |  | `rubric_score` | req | `number` |
|  | `content_review_topic_gate` |  | req | [PublicTopicGateModel](#model-publictopicgatemodel) |
|  |  | `criteria` | opt | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] (default: []) |
|  |  | `score` | opt | `number` (nullable) |
|  | `external_id` |  | req | `string` |
|  | `is_excluded` |  | opt | `boolean` (default: False) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `provider_context` |  | opt | `object` (nullable) |
|  | `source_created_at` |  | opt | `string` (format: date-time; nullable) |
|  | `text` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (nullable) |
| `limit` |  |  | req | `integer` |
| `next_cursor` |  |  | opt | `string` (nullable) |
| `q` |  |  | opt | `string` (nullable) |
| `rubric` |  |  | req | [PublicRubricModel](#model-publicrubricmodel) |
|  | `criteria` |  | req | array[[PublicRubricCriterionModel](#model-publicrubriccriterionmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `weight_pct` | req | `integer` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
| `run_at` |  |  | opt | `string` (format: date-time; nullable) |
| `run_id` |  |  | opt | `string` (format: uuid; nullable) |
| `sort` |  |  | req | `string` |
| `status` |  |  | opt | `string` (nullable) |
| `topic` |  |  | req | [PublicFeedTopicModel](#model-publicfeedtopicmodel) |
|  | `description` |  | opt | `string` (nullable) |
|  | `keyword` |  | req | `string` |
|  | `title` |  | req | `string` |
| `total_count` |  |  | req | `integer` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


##### run-dates

###### {run_date}

###### digest

<a id="endpoint-get-v1-public-feeds-feed_id-run-dates-run_date-digest"></a>
###### GET /v1/public/feeds/{feed_id}/run-dates/{run_date}/digest

Fetch a curated digest for a succeeded run date (public view).

**Auth**: None.

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |
| `run_date` | path | req | `string` (format: date) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [PublicFeedRunDigestResponseModel](#model-publicfeedrundigestresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `curated_count` |  |  | req | `integer` |
| `digest_status` |  |  | req | `string` |
| `evaluated_count` |  |  | req | `integer` |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `items` |  |  | req | array[[PublicFeedRunItemModel](#model-publicfeedrunitemmodel)] |
|  | `author` |  | opt | `string` (nullable) |
|  | `content_review_rubric_result` |  | opt | [PublicExternalEvalResultModel](#model-publicexternalevalresultmodel) (nullable) |
|  |  | `criteria` | req | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] |
|  |  | `overall_rationale` | opt | `string` (nullable) |
|  |  | `rubric_score` | req | `number` |
|  | `content_review_topic_gate` |  | req | [PublicTopicGateModel](#model-publictopicgatemodel) |
|  |  | `criteria` | opt | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] (default: []) |
|  |  | `score` | opt | `number` (nullable) |
|  | `external_id` |  | req | `string` |
|  | `is_excluded` |  | opt | `boolean` (default: False) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `provider_context` |  | opt | `object` (nullable) |
|  | `source_created_at` |  | opt | `string` (format: date-time; nullable) |
|  | `text` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (nullable) |
| `run_at` |  |  | req | `string` (format: date-time) |
| `run_id` |  |  | req | `string` (format: uuid) |
| `settings` |  |  | req | [PublicFeedRunSettingsModel](#model-publicfeedrunsettingsmodel) |
|  | `digest_threshold_pct` |  | req | `integer` |
|  | `interval_hours` |  | opt | `integer` (nullable) |
|  | `item_limit` |  | req | `integer` |
|  | `lookback_minutes` |  | req | `integer` |
|  | `min_likes` |  | req | `integer` |
|  | `providers` |  | req | array[`string`] |
| `summary_markdown` |  |  | opt | `string` (nullable) |
| `threshold_pct` |  |  | req | `integer` |
| `topic` |  |  | req | [PublicFeedTopicModel](#model-publicfeedtopicmodel) |
|  | `description` |  | opt | `string` (nullable) |
|  | `keyword` |  | req | `string` |
|  | `title` |  | req | `string` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


##### runs

<a id="endpoint-get-v1-public-feeds-feed_id-runs"></a>
###### GET /v1/public/feeds/{feed_id}/runs

List run dates for a feed: succeeded runs plus the current active run when that date has no succeeded run (public view).

**Auth**: None.

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |
| `limit` | query | opt | `integer` (default: 50) |
| `cursor` | query | opt | `string` (format: date; nullable) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [PublicFeedRunsResponseModel](#model-publicfeedrunsresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `cursor` |  |  | opt | `string` (nullable) |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `limit` |  |  | req | `integer` |
| `next_cursor` |  |  | opt | `string` (nullable) |
| `runs` |  |  | req | array[[PublicFeedRunReadModel](#model-publicfeedrunreadmodel)] |
|  | `date` |  | req | `string` (format: date) |
|  | `is_overridden` |  | opt | `boolean` (default: False) |
|  | `run_at` |  | req | `string` (format: date-time) |
|  | `run_id` |  | req | `string` (format: uuid) |
|  | `status` |  | req | `string` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


###### {run_date}

<a id="endpoint-get-v1-public-feeds-feed_id-runs-run_date"></a>
###### GET /v1/public/feeds/{feed_id}/runs/{run_date}

Fetch the succeeded run for a date (items + rubric), or the active run when that date has no succeeded run (public view).

**Auth**: None.

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |
| `run_date` | path | req | `string` (format: date) |
| `limit` | query | opt | `integer` (default: 200) |
| `sort` | query | opt | [FeedRunItemSort](#model-feedrunitemsort) (default: score) |
| `q` | query | opt | `string` (nullable) |
| `cursor` | query | opt | `string` (nullable) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [PublicFeedRunResponseModel](#model-publicfeedrunresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `cursor` |  |  | opt | `string` (nullable) |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `items` |  |  | req | array[[PublicFeedRunItemModel](#model-publicfeedrunitemmodel)] |
|  | `author` |  | opt | `string` (nullable) |
|  | `content_review_rubric_result` |  | opt | [PublicExternalEvalResultModel](#model-publicexternalevalresultmodel) (nullable) |
|  |  | `criteria` | req | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] |
|  |  | `overall_rationale` | opt | `string` (nullable) |
|  |  | `rubric_score` | req | `number` |
|  | `content_review_topic_gate` |  | req | [PublicTopicGateModel](#model-publictopicgatemodel) |
|  |  | `criteria` | opt | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] (default: []) |
|  |  | `score` | opt | `number` (nullable) |
|  | `external_id` |  | req | `string` |
|  | `is_excluded` |  | opt | `boolean` (default: False) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `provider_context` |  | opt | `object` (nullable) |
|  | `source_created_at` |  | opt | `string` (format: date-time; nullable) |
|  | `text` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (nullable) |
| `limit` |  |  | req | `integer` |
| `next_cursor` |  |  | opt | `string` (nullable) |
| `q` |  |  | opt | `string` (nullable) |
| `rubric` |  |  | req | [PublicRubricModel](#model-publicrubricmodel) |
|  | `criteria` |  | req | array[[PublicRubricCriterionModel](#model-publicrubriccriterionmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `weight_pct` | req | `integer` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
| `run_at` |  |  | req | `string` (format: date-time) |
| `run_id` |  |  | req | `string` (format: uuid) |
| `settings` |  |  | req | [PublicFeedRunSettingsModel](#model-publicfeedrunsettingsmodel) |
|  | `digest_threshold_pct` |  | req | `integer` |
|  | `interval_hours` |  | opt | `integer` (nullable) |
|  | `item_limit` |  | req | `integer` |
|  | `lookback_minutes` |  | req | `integer` |
|  | `min_likes` |  | req | `integer` |
|  | `providers` |  | req | array[`string`] |
| `sort` |  |  | req | `string` |
| `status` |  |  | req | `string` |
| `topic` |  |  | req | [PublicFeedTopicModel](#model-publicfeedtopicmodel) |
|  | `description` |  | opt | `string` (nullable) |
|  | `keyword` |  | req | `string` |
|  | `title` |  | req | `string` |
| `total_count` |  |  | req | `integer` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


###### {run_id}

###### digest

<a id="endpoint-get-v1-public-feeds-feed_id-runs-run_id-digest"></a>
###### GET /v1/public/feeds/{feed_id}/runs/{run_id}/digest

Fetch a curated digest for a run (public view).

**Auth**: None.

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |
| `run_id` | path | req | `string` (format: uuid) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [PublicFeedRunDigestResponseModel](#model-publicfeedrundigestresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `curated_count` |  |  | req | `integer` |
| `digest_status` |  |  | req | `string` |
| `evaluated_count` |  |  | req | `integer` |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `items` |  |  | req | array[[PublicFeedRunItemModel](#model-publicfeedrunitemmodel)] |
|  | `author` |  | opt | `string` (nullable) |
|  | `content_review_rubric_result` |  | opt | [PublicExternalEvalResultModel](#model-publicexternalevalresultmodel) (nullable) |
|  |  | `criteria` | req | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] |
|  |  | `overall_rationale` | opt | `string` (nullable) |
|  |  | `rubric_score` | req | `number` |
|  | `content_review_topic_gate` |  | req | [PublicTopicGateModel](#model-publictopicgatemodel) |
|  |  | `criteria` | opt | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] (default: []) |
|  |  | `score` | opt | `number` (nullable) |
|  | `external_id` |  | req | `string` |
|  | `is_excluded` |  | opt | `boolean` (default: False) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `provider_context` |  | opt | `object` (nullable) |
|  | `source_created_at` |  | opt | `string` (format: date-time; nullable) |
|  | `text` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (nullable) |
| `run_at` |  |  | req | `string` (format: date-time) |
| `run_id` |  |  | req | `string` (format: uuid) |
| `settings` |  |  | req | [PublicFeedRunSettingsModel](#model-publicfeedrunsettingsmodel) |
|  | `digest_threshold_pct` |  | req | `integer` |
|  | `interval_hours` |  | opt | `integer` (nullable) |
|  | `item_limit` |  | req | `integer` |
|  | `lookback_minutes` |  | req | `integer` |
|  | `min_likes` |  | req | `integer` |
|  | `providers` |  | req | array[`string`] |
| `summary_markdown` |  |  | opt | `string` (nullable) |
| `threshold_pct` |  |  | req | `integer` |
| `topic` |  |  | req | [PublicFeedTopicModel](#model-publicfeedtopicmodel) |
|  | `description` |  | opt | `string` (nullable) |
|  | `keyword` |  | req | `string` |
|  | `title` |  | req | `string` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |


##### submissions

###### {job_id}

<a id="endpoint-get-v1-public-feeds-feed_id-submissions-job_id"></a>
###### GET /v1/public/feeds/{feed_id}/submissions/{job_id}

Fetch submission status/results by job id (public view).

**Auth**: None.

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `feed_id` | path | req | `string` (format: uuid) |
| `job_id` | path | req | `string` (format: uuid) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [PublicFeedSubmissionResponseModel](#model-publicfeedsubmissionresponsemodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `decision` |  |  | opt | `string` (nullable) |
| `evaluated_at` |  |  | opt | `string` (format: date-time; nullable) |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `item` |  |  | req | [PublicFeedRunItemModel](#model-publicfeedrunitemmodel) |
|  | `author` |  | opt | `string` (nullable) |
|  | `content_review_rubric_result` |  | opt | [PublicExternalEvalResultModel](#model-publicexternalevalresultmodel) (nullable) |
|  |  | `criteria` | req | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] |
|  |  | `overall_rationale` | opt | `string` (nullable) |
|  |  | `rubric_score` | req | `number` |
|  | `content_review_topic_gate` |  | req | [PublicTopicGateModel](#model-publictopicgatemodel) |
|  |  | `criteria` | opt | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] (default: []) |
|  |  | `score` | opt | `number` (nullable) |
|  | `external_id` |  | req | `string` |
|  | `is_excluded` |  | opt | `boolean` (default: False) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `provider_context` |  | opt | `object` (nullable) |
|  | `source_created_at` |  | opt | `string` (format: date-time; nullable) |
|  | `text` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (nullable) |
| `job_error_code` |  |  | opt | `string` (nullable) |
| `job_error_message` |  |  | opt | `string` (nullable) |
| `job_id` |  |  | req | `string` (format: uuid) |
| `job_status` |  |  | req | `string` |
| `requested_at` |  |  | req | `string` (format: date-time) |
| `rubric` |  |  | req | [PublicRubricModel](#model-publicrubricmodel) |
|  | `criteria` |  | req | array[[PublicRubricCriterionModel](#model-publicrubriccriterionmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `weight_pct` | req | `integer` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
| `topic` |  |  | req | [PublicFeedTopicModel](#model-publicfeedtopicmodel) |
|  | `description` |  | opt | `string` (nullable) |
|  | `keyword` |  | req | `string` |
|  | `title` |  | req | `string` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |



## query

### execute

<a id="endpoint-post-v1-query-execute"></a>
#### POST /v1/query/execute

Execute a query against the current platform-selected artifact (admin only).

**Auth**: Google Bearer (`Authorization: Bearer <google_id_token>`) OR ApiKey OR ConsoleSessionCookie

**Request**
Content-Type: `application/json`
Body: [Query](#model-query)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `text` |  |  | req | `string` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [Response](#model-response)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `text` |  |  | req | `string` |

`422` Validation Error
Content-Type: `application/json`
Body: [HTTPValidationError](#model-httpvalidationerror)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `detail` |  |  | opt | array[[ValidationError](#model-validationerror)] |
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
|  | `loc` |  | req | array[anyOf: `string` OR `integer`] |
|  | `msg` |  | req | `string` |
|  | `type` |  | req | `string` |



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



## Misc

### healthz

<a id="endpoint-get-healthz"></a>
#### GET /healthz

Healthz

**Auth**: None.

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [StatusResponse](#model-statusresponse)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `status` |  |  | req | `string` |


### metrics

<a id="endpoint-get-metrics"></a>
#### GET /metrics

Prometheus metrics endpoint.

**Auth**: None.

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: `unknown`

(no documented fields)



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

<a id="model-admincreateuserrequestmodel"></a>
### Model: AdminCreateUserRequestModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `role` |  |  | req | `string` (enum: [member, admin]) |
| `username` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "role": {
      "enum": [
        "member",
        "admin"
      ],
      "title": "Role",
      "type": "string"
    },
    "username": {
      "maxLength": 256,
      "minLength": 1,
      "title": "Username",
      "type": "string"
    }
  },
  "required": [
    "username",
    "role"
  ],
  "title": "AdminCreateUserRequestModel",
  "type": "object"
}
```

</details>

<a id="model-admincreateuserresponsemodel"></a>
### Model: AdminCreateUserResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `api_key` |  |  | req | `string` |
| `api_key_id` |  |  | req | `string` (format: uuid) |
| `status` |  |  | req | `string` |
| `user_id` |  |  | req | `string` (format: uuid) |
| `username` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "api_key": {
      "title": "Api Key",
      "type": "string"
    },
    "api_key_id": {
      "format": "uuid",
      "title": "Api Key Id",
      "type": "string"
    },
    "status": {
      "const": "ok",
      "title": "Status",
      "type": "string"
    },
    "user_id": {
      "format": "uuid",
      "title": "User Id",
      "type": "string"
    },
    "username": {
      "title": "Username",
      "type": "string"
    }
  },
  "required": [
    "status",
    "username",
    "user_id",
    "api_key_id",
    "api_key"
  ],
  "title": "AdminCreateUserResponseModel",
  "type": "object"
}
```

</details>

<a id="model-citationmodel"></a>
### Model: CitationModel

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
  "title": "CitationModel",
  "type": "object"
}
```

</details>

<a id="model-consolesessionresponsemodel"></a>
### Model: ConsoleSessionResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `is_admin` |  |  | req | `boolean` |
| `label` |  |  | req | `string` |
| `status` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "is_admin": {
      "title": "Is Admin",
      "type": "boolean"
    },
    "label": {
      "title": "Label",
      "type": "string"
    },
    "status": {
      "const": "ok",
      "title": "Status",
      "type": "string"
    }
  },
  "required": [
    "status",
    "label",
    "is_admin"
  ],
  "title": "ConsoleSessionResponseModel",
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
|  |  | `note` | req | `string` |
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
|  | `note` |  | req | `string` |
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

<a id="model-feedcreaterequestmodel-input"></a>
### Model: FeedCreateRequestModel-Input

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `digest_threshold_pct` |  |  | opt | `integer` (default: 70) |
| `feed_type` |  |  | opt | `string` (enum: [batch, continuous]; default: batch) |
| `item_limit` |  |  | opt | `integer` (nullable) |
| `lookback_minutes` |  |  | opt | `integer` (nullable) |
| `providers` |  |  | req | array[`string`] |
| `rubric` |  |  | req | [InlineRubricRequestModel-Input](#model-inlinerubricrequestmodel-input) |
|  | `criteria` |  | req | array[[InlineRubricCriterionModel](#model-inlinerubriccriterionmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `weight_pct` | req | `integer` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
| `schedule` |  |  | req | oneOf: [FeedManualSchedule](#model-feedmanualschedule) OR [FeedScheduledSchedule](#model-feedscheduledschedule) |
| `submission_access_policy` |  |  | opt | `string` (enum: [owner_editor_only, authenticated_any, paid_any, authenticated_or_paid]; default: owner_editor_only) |
| `topic_description` |  |  | opt | `string` (nullable) |
| `topic_keyword` |  |  | req | `string` |
| `topic_title` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "digest_threshold_pct": {
      "default": 70,
      "maximum": 100.0,
      "minimum": 0.0,
      "title": "Digest Threshold Pct",
      "type": "integer"
    },
    "feed_type": {
      "default": "batch",
      "enum": [
        "batch",
        "continuous"
      ],
      "title": "Feed Type",
      "type": "string"
    },
    "item_limit": {
      "anyOf": [
        {
          "maximum": 1000.0,
          "minimum": 5.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Item Limit"
    },
    "lookback_minutes": {
      "anyOf": [
        {
          "maximum": 43200.0,
          "minimum": 1440.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Lookback Minutes"
    },
    "providers": {
      "items": {
        "maxLength": 32,
        "minLength": 1,
        "pattern": "^[a-z0-9_]+$",
        "type": "string"
      },
      "minItems": 0,
      "title": "Providers",
      "type": "array"
    },
    "rubric": {
      "$ref": "#/components/schemas/InlineRubricRequestModel-Input"
    },
    "schedule": {
      "discriminator": {
        "mapping": {
          "manual": "#/components/schemas/FeedManualSchedule",
          "scheduled": "#/components/schemas/FeedScheduledSchedule"
        },
        "propertyName": "trigger_kind"
      },
      "oneOf": [
        {
          "$ref": "#/components/schemas/FeedManualSchedule"
        },
        {
          "$ref": "#/components/schemas/FeedScheduledSchedule"
        }
      ],
      "title": "Schedule"
    },
    "submission_access_policy": {
      "default": "owner_editor_only",
      "enum": [
        "owner_editor_only",
        "authenticated_any",
        "paid_any",
        "authenticated_or_paid"
      ],
      "title": "Submission Access Policy",
      "type": "string"
    },
    "topic_description": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Topic Description"
    },
    "topic_keyword": {
      "maxLength": 256,
      "minLength": 1,
      "title": "Topic Keyword",
      "type": "string"
    },
    "topic_title": {
      "maxLength": 256,
      "minLength": 1,
      "title": "Topic Title",
      "type": "string"
    }
  },
  "required": [
    "providers",
    "topic_keyword",
    "topic_title",
    "schedule",
    "rubric"
  ],
  "title": "FeedCreateRequestModel",
  "type": "object"
}
```

</details>

<a id="model-feedcreaterequestmodel-output"></a>
### Model: FeedCreateRequestModel-Output

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `digest_threshold_pct` |  |  | opt | `integer` (default: 70) |
| `feed_type` |  |  | opt | `string` (enum: [batch, continuous]; default: batch) |
| `item_limit` |  |  | opt | `integer` (nullable) |
| `lookback_minutes` |  |  | opt | `integer` (nullable) |
| `providers` |  |  | req | array[`string`] |
| `rubric` |  |  | req | [InlineRubricRequestModel-Output](#model-inlinerubricrequestmodel-output) |
|  | `criteria` |  | req | array[[InlineRubricCriterionModel](#model-inlinerubriccriterionmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `weight_pct` | req | `integer` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
| `schedule` |  |  | req | oneOf: [FeedManualSchedule](#model-feedmanualschedule) OR [FeedScheduledSchedule](#model-feedscheduledschedule) |
| `submission_access_policy` |  |  | opt | `string` (enum: [owner_editor_only, authenticated_any, paid_any, authenticated_or_paid]; default: owner_editor_only) |
| `topic_description` |  |  | opt | `string` (nullable) |
| `topic_keyword` |  |  | req | `string` |
| `topic_title` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "digest_threshold_pct": {
      "default": 70,
      "maximum": 100.0,
      "minimum": 0.0,
      "title": "Digest Threshold Pct",
      "type": "integer"
    },
    "feed_type": {
      "default": "batch",
      "enum": [
        "batch",
        "continuous"
      ],
      "title": "Feed Type",
      "type": "string"
    },
    "item_limit": {
      "anyOf": [
        {
          "maximum": 1000.0,
          "minimum": 5.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Item Limit"
    },
    "lookback_minutes": {
      "anyOf": [
        {
          "maximum": 43200.0,
          "minimum": 1440.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Lookback Minutes"
    },
    "providers": {
      "items": {
        "maxLength": 32,
        "minLength": 1,
        "pattern": "^[a-z0-9_]+$",
        "type": "string"
      },
      "minItems": 0,
      "title": "Providers",
      "type": "array"
    },
    "rubric": {
      "$ref": "#/components/schemas/InlineRubricRequestModel-Output"
    },
    "schedule": {
      "discriminator": {
        "mapping": {
          "manual": "#/components/schemas/FeedManualSchedule",
          "scheduled": "#/components/schemas/FeedScheduledSchedule"
        },
        "propertyName": "trigger_kind"
      },
      "oneOf": [
        {
          "$ref": "#/components/schemas/FeedManualSchedule"
        },
        {
          "$ref": "#/components/schemas/FeedScheduledSchedule"
        }
      ],
      "title": "Schedule"
    },
    "submission_access_policy": {
      "default": "owner_editor_only",
      "enum": [
        "owner_editor_only",
        "authenticated_any",
        "paid_any",
        "authenticated_or_paid"
      ],
      "title": "Submission Access Policy",
      "type": "string"
    },
    "topic_description": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Topic Description"
    },
    "topic_keyword": {
      "maxLength": 256,
      "minLength": 1,
      "title": "Topic Keyword",
      "type": "string"
    },
    "topic_title": {
      "maxLength": 256,
      "minLength": 1,
      "title": "Topic Title",
      "type": "string"
    }
  },
  "required": [
    "providers",
    "topic_keyword",
    "topic_title",
    "schedule",
    "rubric"
  ],
  "title": "FeedCreateRequestModel",
  "type": "object"
}
```

</details>

<a id="model-feedcreateresponsemodel"></a>
### Model: FeedCreateResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `feed_id` |  |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "feed_id": {
      "format": "uuid",
      "title": "Feed Id",
      "type": "string"
    }
  },
  "required": [
    "feed_id"
  ],
  "title": "FeedCreateResponseModel",
  "type": "object"
}
```

</details>

<a id="model-feedmanualschedule"></a>
### Model: FeedManualSchedule

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `trigger_kind` |  |  | opt | `string` (default: manual) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "trigger_kind": {
      "const": "manual",
      "default": "manual",
      "title": "Trigger Kind",
      "type": "string"
    }
  },
  "title": "FeedManualSchedule",
  "type": "object"
}
```

</details>

<a id="model-feedpauserequestmodel"></a>
### Model: FeedPauseRequestModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `is_paused` |  |  | req | `boolean` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "is_paused": {
      "title": "Is Paused",
      "type": "boolean"
    }
  },
  "required": [
    "is_paused"
  ],
  "title": "FeedPauseRequestModel",
  "type": "object"
}
```

</details>

<a id="model-feedprogressmodel"></a>
### Model: FeedProgressModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `accepted` |  |  | opt | `integer` (default: 0) |
| `content_review_complete` |  |  | opt | `boolean` (default: False) |
| `failed` |  |  | opt | `integer` (default: 0) |
| `queued` |  |  | opt | `integer` (default: 0) |
| `rejected` |  |  | opt | `integer` (default: 0) |
| `running` |  |  | opt | `integer` (default: 0) |
| `succeeded` |  |  | opt | `integer` (default: 0) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "accepted": {
      "default": 0,
      "title": "Accepted",
      "type": "integer"
    },
    "content_review_complete": {
      "default": false,
      "title": "Content Review Complete",
      "type": "boolean"
    },
    "failed": {
      "default": 0,
      "title": "Failed",
      "type": "integer"
    },
    "queued": {
      "default": 0,
      "title": "Queued",
      "type": "integer"
    },
    "rejected": {
      "default": 0,
      "title": "Rejected",
      "type": "integer"
    },
    "running": {
      "default": 0,
      "title": "Running",
      "type": "integer"
    },
    "succeeded": {
      "default": 0,
      "title": "Succeeded",
      "type": "integer"
    }
  },
  "title": "FeedProgressModel",
  "type": "object"
}
```

</details>

<a id="model-feedreadmodel"></a>
### Model: FeedReadModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `created_at` |  |  | req | `string` (format: date-time) |
| `feed_type` |  |  | req | `string` |
| `id` |  |  | req | `string` (format: uuid) |
| `interval_hours` |  |  | opt | `integer` (nullable) |
| `is_paused` |  |  | req | `boolean` |
| `last_error` |  |  | opt | `string` (nullable) |
| `latest_run` |  |  | opt | [FeedRunReadModel](#model-feedrunreadmodel) (nullable) |
|  | `attempt_number` |  | req | `integer` |
|  | `cancel_requested_at` |  | opt | `string` (format: date-time; nullable) |
|  | `completed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `error` |  | opt | `string` (nullable) |
|  | `id` |  | req | `string` (format: uuid) |
|  | `slot_at` |  | req | `string` (format: date-time) |
|  | `started_at` |  | req | `string` (format: date-time) |
|  | `status` |  | req | `string` |
|  | `trigger_kind` |  | req | `string` |
| `next_run_at` |  |  | opt | `string` (format: date-time; nullable) |
| `progress` |  |  | opt | [FeedProgressModel](#model-feedprogressmodel) |
|  | `accepted` |  | opt | `integer` (default: 0) |
|  | `content_review_complete` |  | opt | `boolean` (default: False) |
|  | `failed` |  | opt | `integer` (default: 0) |
|  | `queued` |  | opt | `integer` (default: 0) |
|  | `rejected` |  | opt | `integer` (default: 0) |
|  | `running` |  | opt | `integer` (default: 0) |
|  | `succeeded` |  | opt | `integer` (default: 0) |
| `rubric` |  |  | req | [FeedRubricReadModel](#model-feedrubricreadmodel) |
|  | `description` |  | req | `string` |
|  | `id` |  | req | `string` (format: uuid) |
|  | `slug` |  | req | `string` |
|  | `title` |  | req | `string` |
| `schedule_first_run_at` |  |  | opt | `string` (format: date-time; nullable) |
| `state` |  |  | req | `string` |
| `submission_access_policy` |  |  | opt | `string` (enum: [owner_editor_only, authenticated_any, paid_any, authenticated_or_paid]; default: owner_editor_only) |
| `topic` |  |  | req | [FeedTopicReadModel](#model-feedtopicreadmodel) |
|  | `criterion_id` |  | req | `string` |
|  | `description` |  | req | `string` |
|  | `id` |  | req | `string` (format: uuid) |
|  | `keyword` |  | req | `string` |
|  | `title` |  | req | `string` |
| `trigger_kind` |  |  | req | `string` |
| `updated_at` |  |  | req | `string` (format: date-time) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "created_at": {
      "format": "date-time",
      "title": "Created At",
      "type": "string"
    },
    "feed_type": {
      "title": "Feed Type",
      "type": "string"
    },
    "id": {
      "format": "uuid",
      "title": "Id",
      "type": "string"
    },
    "interval_hours": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Interval Hours"
    },
    "is_paused": {
      "title": "Is Paused",
      "type": "boolean"
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
    "latest_run": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/FeedRunReadModel"
        },
        {
          "type": "null"
        }
      ]
    },
    "next_run_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Next Run At"
    },
    "progress": {
      "$ref": "#/components/schemas/FeedProgressModel"
    },
    "rubric": {
      "$ref": "#/components/schemas/FeedRubricReadModel"
    },
    "schedule_first_run_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Schedule First Run At"
    },
    "state": {
      "title": "State",
      "type": "string"
    },
    "submission_access_policy": {
      "default": "owner_editor_only",
      "enum": [
        "owner_editor_only",
        "authenticated_any",
        "paid_any",
        "authenticated_or_paid"
      ],
      "title": "Submission Access Policy",
      "type": "string"
    },
    "topic": {
      "$ref": "#/components/schemas/FeedTopicReadModel"
    },
    "trigger_kind": {
      "title": "Trigger Kind",
      "type": "string"
    },
    "updated_at": {
      "format": "date-time",
      "title": "Updated At",
      "type": "string"
    }
  },
  "required": [
    "id",
    "state",
    "feed_type",
    "trigger_kind",
    "is_paused",
    "created_at",
    "updated_at",
    "topic",
    "rubric"
  ],
  "title": "FeedReadModel",
  "type": "object"
}
```

</details>

<a id="model-feedrubricreadmodel"></a>
### Model: FeedRubricReadModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `description` |  |  | req | `string` |
| `id` |  |  | req | `string` (format: uuid) |
| `slug` |  |  | req | `string` |
| `title` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "description": {
      "title": "Description",
      "type": "string"
    },
    "id": {
      "format": "uuid",
      "title": "Id",
      "type": "string"
    },
    "slug": {
      "title": "Slug",
      "type": "string"
    },
    "title": {
      "title": "Title",
      "type": "string"
    }
  },
  "required": [
    "id",
    "slug",
    "title",
    "description"
  ],
  "title": "FeedRubricReadModel",
  "type": "object"
}
```

</details>

<a id="model-feedrundateoverriderequestmodel"></a>
### Model: FeedRunDateOverrideRequestModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `run_id` |  |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "run_id": {
      "format": "uuid",
      "title": "Run Id",
      "type": "string"
    }
  },
  "required": [
    "run_id"
  ],
  "title": "FeedRunDateOverrideRequestModel",
  "type": "object"
}
```

</details>

<a id="model-feedruninprogressmodel"></a>
### Model: FeedRunInProgressModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `attempt_number` |  |  | req | `integer` |
| `cancel_requested_at` |  |  | opt | `string` (format: date-time; nullable) |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `run_id` |  |  | req | `string` (format: uuid) |
| `slot_at` |  |  | req | `string` (format: date-time) |
| `started_at` |  |  | req | `string` (format: date-time) |
| `status` |  |  | req | `string` |
| `trigger_kind` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "attempt_number": {
      "title": "Attempt Number",
      "type": "integer"
    },
    "cancel_requested_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Cancel Requested At"
    },
    "feed_id": {
      "format": "uuid",
      "title": "Feed Id",
      "type": "string"
    },
    "run_id": {
      "format": "uuid",
      "title": "Run Id",
      "type": "string"
    },
    "slot_at": {
      "format": "date-time",
      "title": "Slot At",
      "type": "string"
    },
    "started_at": {
      "format": "date-time",
      "title": "Started At",
      "type": "string"
    },
    "status": {
      "title": "Status",
      "type": "string"
    },
    "trigger_kind": {
      "title": "Trigger Kind",
      "type": "string"
    }
  },
  "required": [
    "run_id",
    "feed_id",
    "status",
    "started_at",
    "slot_at",
    "trigger_kind",
    "attempt_number"
  ],
  "title": "FeedRunInProgressModel",
  "type": "object"
}
```

</details>

<a id="model-feedrunitembucket"></a>
### Model: FeedRunItemBucket

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "enum": [
    "all",
    "succeeded",
    "rejected",
    "error",
    "pending"
  ],
  "title": "FeedRunItemBucket",
  "type": "string"
}
```

</details>

<a id="model-feedrunitemexcluderequestmodel"></a>
### Model: FeedRunItemExcludeRequestModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `is_excluded` |  |  | opt | `boolean` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
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
    }
  },
  "title": "FeedRunItemExcludeRequestModel",
  "type": "object"
}
```

</details>

<a id="model-feedrunitemexcluderesponsemodel"></a>
### Model: FeedRunItemExcludeResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `is_excluded` |  |  | req | `boolean` |
| `job_id` |  |  | req | `string` (format: uuid) |
| `run_id` |  |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "feed_id": {
      "format": "uuid",
      "title": "Feed Id",
      "type": "string"
    },
    "is_excluded": {
      "title": "Is Excluded",
      "type": "boolean"
    },
    "job_id": {
      "format": "uuid",
      "title": "Job Id",
      "type": "string"
    },
    "run_id": {
      "format": "uuid",
      "title": "Run Id",
      "type": "string"
    }
  },
  "required": [
    "feed_id",
    "run_id",
    "job_id",
    "is_excluded"
  ],
  "title": "FeedRunItemExcludeResponseModel",
  "type": "object"
}
```

</details>

<a id="model-feedrunitemreadmodel"></a>
### Model: FeedRunItemReadModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `author` |  |  | opt | `string` (nullable) |
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
| `content_review_topic_gate` |  |  | req | [TopicGateModel](#model-topicgatemodel) |
|  | `criteria` |  | opt | array[[CriterionAssessmentModel](#model-criterionassessmentmodel)] (default: []) |
|  |  | `aggregate_score` | req | `number` |
|  |  | `criterion_evaluations` | req | array[[CriterionEvaluationModel](#model-criterionevaluationmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  | `score` |  | opt | `number` (nullable) |
| `decision` |  |  | opt | `string` (nullable) |
| `evaluated_at` |  |  | opt | `string` (format: date-time; nullable) |
| `external_id` |  |  | req | `string` |
| `is_excluded` |  |  | opt | `boolean` (default: False) |
| `job_error_code` |  |  | opt | `string` (nullable) |
| `job_error_message` |  |  | opt | `string` (nullable) |
| `job_id` |  |  | req | `string` (format: uuid) |
| `job_status` |  |  | req | `string` |
| `provider` |  |  | req | `string` |
| `provider_context` |  |  | opt | `object` (nullable) |
| `requested_at` |  |  | req | `string` (format: date-time) |
| `source_created_at` |  |  | opt | `string` (format: date-time; nullable) |
| `text` |  |  | req | `string` |
| `title` |  |  | opt | `string` (nullable) |
| `url` |  |  | opt | `string` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "author": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Author"
    },
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
      "$ref": "#/components/schemas/TopicGateModel"
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
    "evaluated_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Evaluated At"
    },
    "external_id": {
      "title": "External Id",
      "type": "string"
    },
    "is_excluded": {
      "default": false,
      "title": "Is Excluded",
      "type": "boolean"
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
      "title": "Job Status",
      "type": "string"
    },
    "provider": {
      "title": "Provider",
      "type": "string"
    },
    "provider_context": {
      "anyOf": [
        {
          "additionalProperties": true,
          "type": "object"
        },
        {
          "type": "null"
        }
      ],
      "title": "Provider Context"
    },
    "requested_at": {
      "format": "date-time",
      "title": "Requested At",
      "type": "string"
    },
    "source_created_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Source Created At"
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
    "external_id",
    "provider",
    "job_status",
    "requested_at",
    "text",
    "content_review_topic_gate"
  ],
  "title": "FeedRunItemReadModel",
  "type": "object"
}
```

</details>

<a id="model-feedrunitemsort"></a>
### Model: FeedRunItemSort

(no documented fields)

<details>
<summary>JSON schema</summary>

```json
{
  "enum": [
    "score",
    "newest"
  ],
  "title": "FeedRunItemSort",
  "type": "string"
}
```

</details>

<a id="model-feedrunitemsresponsemodel"></a>
### Model: FeedRunItemsResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `cursor` |  |  | opt | `string` (nullable) |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `items` |  |  | req | array[[FeedRunItemReadModel](#model-feedrunitemreadmodel)] |
|  | `author` |  | opt | `string` (nullable) |
|  | `content_id` |  | req | `string` (format: uuid) |
|  | `content_review_rubric_result` |  | opt | [ExternalEvalResultModel](#model-externalevalresultmodel) (nullable) |
|  |  | `criteria` | req | array[[CriterionAssessmentModel](#model-criterionassessmentmodel)] |
|  |  | `overall_rationale` | opt | `string` (nullable) |
|  |  | `rubric_id` | req | `string` |
|  |  | `rubric_score` | req | `number` |
|  | `content_review_topic_gate` |  | req | [TopicGateModel](#model-topicgatemodel) |
|  |  | `criteria` | opt | array[[CriterionAssessmentModel](#model-criterionassessmentmodel)] (default: []) |
|  |  | `score` | opt | `number` (nullable) |
|  | `decision` |  | opt | `string` (nullable) |
|  | `evaluated_at` |  | opt | `string` (format: date-time; nullable) |
|  | `external_id` |  | req | `string` |
|  | `is_excluded` |  | opt | `boolean` (default: False) |
|  | `job_error_code` |  | opt | `string` (nullable) |
|  | `job_error_message` |  | opt | `string` (nullable) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `job_status` |  | req | `string` |
|  | `provider` |  | req | `string` |
|  | `provider_context` |  | opt | `object` (nullable) |
|  | `requested_at` |  | req | `string` (format: date-time) |
|  | `source_created_at` |  | opt | `string` (format: date-time; nullable) |
|  | `text` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (nullable) |
| `limit` |  |  | req | `integer` |
| `next_cursor` |  |  | opt | `string` (nullable) |
| `q` |  |  | opt | `string` (nullable) |
| `run_id` |  |  | req | `string` (format: uuid) |
| `sort` |  |  | req | `string` |
| `status` |  |  | req | `string` |
| `total_count` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "cursor": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Cursor"
    },
    "feed_id": {
      "format": "uuid",
      "title": "Feed Id",
      "type": "string"
    },
    "items": {
      "items": {
        "$ref": "#/components/schemas/FeedRunItemReadModel"
      },
      "title": "Items",
      "type": "array"
    },
    "limit": {
      "title": "Limit",
      "type": "integer"
    },
    "next_cursor": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Next Cursor"
    },
    "q": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Q"
    },
    "run_id": {
      "format": "uuid",
      "title": "Run Id",
      "type": "string"
    },
    "sort": {
      "title": "Sort",
      "type": "string"
    },
    "status": {
      "title": "Status",
      "type": "string"
    },
    "total_count": {
      "title": "Total Count",
      "type": "integer"
    }
  },
  "required": [
    "feed_id",
    "run_id",
    "items",
    "total_count",
    "limit",
    "sort",
    "status"
  ],
  "title": "FeedRunItemsResponseModel",
  "type": "object"
}
```

</details>

<a id="model-feedrunreadmodel"></a>
### Model: FeedRunReadModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `attempt_number` |  |  | req | `integer` |
| `cancel_requested_at` |  |  | opt | `string` (format: date-time; nullable) |
| `completed_at` |  |  | opt | `string` (format: date-time; nullable) |
| `error` |  |  | opt | `string` (nullable) |
| `id` |  |  | req | `string` (format: uuid) |
| `slot_at` |  |  | req | `string` (format: date-time) |
| `started_at` |  |  | req | `string` (format: date-time) |
| `status` |  |  | req | `string` |
| `trigger_kind` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "attempt_number": {
      "title": "Attempt Number",
      "type": "integer"
    },
    "cancel_requested_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Cancel Requested At"
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
    "error": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Error"
    },
    "id": {
      "format": "uuid",
      "title": "Id",
      "type": "string"
    },
    "slot_at": {
      "format": "date-time",
      "title": "Slot At",
      "type": "string"
    },
    "started_at": {
      "format": "date-time",
      "title": "Started At",
      "type": "string"
    },
    "status": {
      "title": "Status",
      "type": "string"
    },
    "trigger_kind": {
      "title": "Trigger Kind",
      "type": "string"
    }
  },
  "required": [
    "id",
    "trigger_kind",
    "slot_at",
    "attempt_number",
    "status",
    "started_at"
  ],
  "title": "FeedRunReadModel",
  "type": "object"
}
```

</details>

<a id="model-feedrunsresponsemodel"></a>
### Model: FeedRunsResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `runs` |  |  | req | array[[FeedRunReadModel](#model-feedrunreadmodel)] |
|  | `attempt_number` |  | req | `integer` |
|  | `cancel_requested_at` |  | opt | `string` (format: date-time; nullable) |
|  | `completed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `error` |  | opt | `string` (nullable) |
|  | `id` |  | req | `string` (format: uuid) |
|  | `slot_at` |  | req | `string` (format: date-time) |
|  | `started_at` |  | req | `string` (format: date-time) |
|  | `status` |  | req | `string` |
|  | `trigger_kind` |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "feed_id": {
      "format": "uuid",
      "title": "Feed Id",
      "type": "string"
    },
    "runs": {
      "items": {
        "$ref": "#/components/schemas/FeedRunReadModel"
      },
      "title": "Runs",
      "type": "array"
    }
  },
  "required": [
    "feed_id",
    "runs"
  ],
  "title": "FeedRunsResponseModel",
  "type": "object"
}
```

</details>

<a id="model-feedscheduledschedule"></a>
### Model: FeedScheduledSchedule

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `first_run_at` |  |  | req | `string` (format: date-time) |
| `interval_hours` |  |  | req | `integer` |
| `trigger_kind` |  |  | opt | `string` (default: scheduled) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "first_run_at": {
      "format": "date-time",
      "title": "First Run At",
      "type": "string"
    },
    "interval_hours": {
      "title": "Interval Hours",
      "type": "integer"
    },
    "trigger_kind": {
      "const": "scheduled",
      "default": "scheduled",
      "title": "Trigger Kind",
      "type": "string"
    }
  },
  "required": [
    "first_run_at",
    "interval_hours"
  ],
  "title": "FeedScheduledSchedule",
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

<a id="model-feedslistresponsemodel"></a>
### Model: FeedsListResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `feeds` |  |  | req | array[[FeedReadModel](#model-feedreadmodel)] |
|  | `created_at` |  | req | `string` (format: date-time) |
|  | `feed_type` |  | req | `string` |
|  | `id` |  | req | `string` (format: uuid) |
|  | `interval_hours` |  | opt | `integer` (nullable) |
|  | `is_paused` |  | req | `boolean` |
|  | `last_error` |  | opt | `string` (nullable) |
|  | `latest_run` |  | opt | [FeedRunReadModel](#model-feedrunreadmodel) (nullable) |
|  |  | `attempt_number` | req | `integer` |
|  |  | `cancel_requested_at` | opt | `string` (format: date-time; nullable) |
|  |  | `completed_at` | opt | `string` (format: date-time; nullable) |
|  |  | `error` | opt | `string` (nullable) |
|  |  | `id` | req | `string` (format: uuid) |
|  |  | `slot_at` | req | `string` (format: date-time) |
|  |  | `started_at` | req | `string` (format: date-time) |
|  |  | `status` | req | `string` |
|  |  | `trigger_kind` | req | `string` |
|  | `next_run_at` |  | opt | `string` (format: date-time; nullable) |
|  | `progress` |  | opt | [FeedProgressModel](#model-feedprogressmodel) |
|  |  | `accepted` | opt | `integer` (default: 0) |
|  |  | `content_review_complete` | opt | `boolean` (default: False) |
|  |  | `failed` | opt | `integer` (default: 0) |
|  |  | `queued` | opt | `integer` (default: 0) |
|  |  | `rejected` | opt | `integer` (default: 0) |
|  |  | `running` | opt | `integer` (default: 0) |
|  |  | `succeeded` | opt | `integer` (default: 0) |
|  | `rubric` |  | req | [FeedRubricReadModel](#model-feedrubricreadmodel) |
|  |  | `description` | req | `string` |
|  |  | `id` | req | `string` (format: uuid) |
|  |  | `slug` | req | `string` |
|  |  | `title` | req | `string` |
|  | `schedule_first_run_at` |  | opt | `string` (format: date-time; nullable) |
|  | `state` |  | req | `string` |
|  | `submission_access_policy` |  | opt | `string` (enum: [owner_editor_only, authenticated_any, paid_any, authenticated_or_paid]; default: owner_editor_only) |
|  | `topic` |  | req | [FeedTopicReadModel](#model-feedtopicreadmodel) |
|  |  | `criterion_id` | req | `string` |
|  |  | `description` | req | `string` |
|  |  | `id` | req | `string` (format: uuid) |
|  |  | `keyword` | req | `string` |
|  |  | `title` | req | `string` |
|  | `trigger_kind` |  | req | `string` |
|  | `updated_at` |  | req | `string` (format: date-time) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "feeds": {
      "items": {
        "$ref": "#/components/schemas/FeedReadModel"
      },
      "title": "Feeds",
      "type": "array"
    }
  },
  "required": [
    "feeds"
  ],
  "title": "FeedsListResponseModel",
  "type": "object"
}
```

</details>

<a id="model-feedsubmissioncreateresponsemodel"></a>
### Model: FeedSubmissionCreateResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `job_id` |  |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "job_id": {
      "format": "uuid",
      "title": "Job Id",
      "type": "string"
    }
  },
  "required": [
    "job_id"
  ],
  "title": "FeedSubmissionCreateResponseModel",
  "type": "object"
}
```

</details>

<a id="model-feedsubmissionoriginalmodel"></a>
### Model: FeedSubmissionOriginalModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `source_kind` |  |  | req | `string` |
| `text` |  |  | req | `string` |
| `title` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "source_kind": {
      "const": "original",
      "title": "Source Kind",
      "type": "string"
    },
    "text": {
      "minLength": 1,
      "title": "Text",
      "type": "string"
    },
    "title": {
      "minLength": 1,
      "title": "Title",
      "type": "string"
    }
  },
  "required": [
    "source_kind",
    "text",
    "title"
  ],
  "title": "FeedSubmissionOriginalModel",
  "type": "object"
}
```

</details>

<a id="model-feedsubmissionrepodiffmodel"></a>
### Model: FeedSubmissionRepoDiffModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `commit_sha` |  |  | req | `string` |
| `diff_unified` |  |  | req | `string` |
| `repo_url` |  |  | req | `string` |
| `source_kind` |  |  | opt | `string` (default: repo_diff) |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "commit_sha": {
      "minLength": 1,
      "title": "Commit Sha",
      "type": "string"
    },
    "diff_unified": {
      "minLength": 1,
      "title": "Diff Unified",
      "type": "string"
    },
    "repo_url": {
      "minLength": 1,
      "title": "Repo Url",
      "type": "string"
    },
    "source_kind": {
      "const": "repo_diff",
      "default": "repo_diff",
      "title": "Source Kind",
      "type": "string"
    }
  },
  "required": [
    "repo_url",
    "commit_sha",
    "diff_unified"
  ],
  "title": "FeedSubmissionRepoDiffModel",
  "type": "object"
}
```

</details>

<a id="model-feedsubmissionurlmodel"></a>
### Model: FeedSubmissionUrlModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `source_kind` |  |  | opt | `string` (default: url) |
| `url` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "source_kind": {
      "const": "url",
      "default": "url",
      "title": "Source Kind",
      "type": "string"
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
  "title": "FeedSubmissionUrlModel",
  "type": "object"
}
```

</details>

<a id="model-feedtopicreadmodel"></a>
### Model: FeedTopicReadModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `criterion_id` |  |  | req | `string` |
| `description` |  |  | req | `string` |
| `id` |  |  | req | `string` (format: uuid) |
| `keyword` |  |  | req | `string` |
| `title` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "criterion_id": {
      "title": "Criterion Id",
      "type": "string"
    },
    "description": {
      "title": "Description",
      "type": "string"
    },
    "id": {
      "format": "uuid",
      "title": "Id",
      "type": "string"
    },
    "keyword": {
      "title": "Keyword",
      "type": "string"
    },
    "title": {
      "title": "Title",
      "type": "string"
    }
  },
  "required": [
    "id",
    "keyword",
    "criterion_id",
    "title",
    "description"
  ],
  "title": "FeedTopicReadModel",
  "type": "object"
}
```

</details>

<a id="model-feedupdaterequestmodel"></a>
### Model: FeedUpdateRequestModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `digest_threshold_pct` |  |  | opt | `integer` (nullable) |
| `feed_type` |  |  | opt | `string` (enum: [batch, continuous]; nullable) |
| `item_limit` |  |  | opt | `integer` (nullable) |
| `lookback_minutes` |  |  | opt | `integer` (nullable) |
| `providers` |  |  | opt | array[`string`] (nullable) |
| `rubric` |  |  | opt | [InlineRubricRequestModel-Input](#model-inlinerubricrequestmodel-input) (nullable) |
|  | `criteria` |  | req | array[[InlineRubricCriterionModel](#model-inlinerubriccriterionmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `weight_pct` | req | `integer` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
| `schedule` |  |  | opt | oneOf: [FeedManualSchedule](#model-feedmanualschedule) OR [FeedScheduledSchedule](#model-feedscheduledschedule) (nullable) |
| `submission_access_policy` |  |  | opt | `string` (enum: [owner_editor_only, authenticated_any, paid_any, authenticated_or_paid]; nullable) |
| `topic_description` |  |  | opt | `string` (nullable) |
| `topic_keyword` |  |  | opt | `string` (nullable) |
| `topic_title` |  |  | opt | `string` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "digest_threshold_pct": {
      "anyOf": [
        {
          "maximum": 100.0,
          "minimum": 0.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Digest Threshold Pct"
    },
    "feed_type": {
      "anyOf": [
        {
          "enum": [
            "batch",
            "continuous"
          ],
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Feed Type"
    },
    "item_limit": {
      "anyOf": [
        {
          "maximum": 1000.0,
          "minimum": 5.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Item Limit"
    },
    "lookback_minutes": {
      "anyOf": [
        {
          "maximum": 43200.0,
          "minimum": 1440.0,
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Lookback Minutes"
    },
    "providers": {
      "anyOf": [
        {
          "items": {
            "maxLength": 32,
            "minLength": 1,
            "pattern": "^[a-z0-9_]+$",
            "type": "string"
          },
          "minItems": 0,
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "title": "Providers"
    },
    "rubric": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/InlineRubricRequestModel-Input"
        },
        {
          "type": "null"
        }
      ]
    },
    "schedule": {
      "anyOf": [
        {
          "discriminator": {
            "mapping": {
              "manual": "#/components/schemas/FeedManualSchedule",
              "scheduled": "#/components/schemas/FeedScheduledSchedule"
            },
            "propertyName": "trigger_kind"
          },
          "oneOf": [
            {
              "$ref": "#/components/schemas/FeedManualSchedule"
            },
            {
              "$ref": "#/components/schemas/FeedScheduledSchedule"
            }
          ]
        },
        {
          "type": "null"
        }
      ],
      "title": "Schedule"
    },
    "submission_access_policy": {
      "anyOf": [
        {
          "enum": [
            "owner_editor_only",
            "authenticated_any",
            "paid_any",
            "authenticated_or_paid"
          ],
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Submission Access Policy"
    },
    "topic_description": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Topic Description"
    },
    "topic_keyword": {
      "anyOf": [
        {
          "maxLength": 256,
          "minLength": 1,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Topic Keyword"
    },
    "topic_title": {
      "anyOf": [
        {
          "maxLength": 256,
          "minLength": 1,
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Topic Title"
    }
  },
  "title": "FeedUpdateRequestModel",
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

<a id="model-inlinerubriccriterionmodel"></a>
### Model: InlineRubricCriterionModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `criterion_id` |  |  | req | `string` |
| `description` |  |  | req | `string` |
| `title` |  |  | req | `string` |
| `verdict_options` |  |  | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  | `description` |  | req | `string` |
|  | `value` |  | req | `integer` |
| `weight_pct` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "criterion_id": {
      "title": "Criterion Id",
      "type": "string"
    },
    "description": {
      "title": "Description",
      "type": "string"
    },
    "title": {
      "title": "Title",
      "type": "string"
    },
    "verdict_options": {
      "items": {
        "$ref": "#/components/schemas/VerdictOptionModel"
      },
      "title": "Verdict Options",
      "type": "array"
    },
    "weight_pct": {
      "title": "Weight Pct",
      "type": "integer"
    }
  },
  "required": [
    "criterion_id",
    "title",
    "description",
    "weight_pct",
    "verdict_options"
  ],
  "title": "InlineRubricCriterionModel",
  "type": "object"
}
```

</details>

<a id="model-inlinerubricrequestmodel-input"></a>
### Model: InlineRubricRequestModel-Input

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `criteria` |  |  | req | array[[InlineRubricCriterionModel](#model-inlinerubriccriterionmodel)] |
|  | `criterion_id` |  | req | `string` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
|  | `verdict_options` |  | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `description` | req | `string` |
|  |  | `value` | req | `integer` |
|  | `weight_pct` |  | req | `integer` |
| `description` |  |  | req | `string` |
| `title` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "criteria": {
      "items": {
        "$ref": "#/components/schemas/InlineRubricCriterionModel"
      },
      "title": "Criteria",
      "type": "array"
    },
    "description": {
      "title": "Description",
      "type": "string"
    },
    "title": {
      "title": "Title",
      "type": "string"
    }
  },
  "required": [
    "title",
    "description",
    "criteria"
  ],
  "title": "InlineRubricRequestModel",
  "type": "object"
}
```

</details>

<a id="model-inlinerubricrequestmodel-output"></a>
### Model: InlineRubricRequestModel-Output

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `criteria` |  |  | req | array[[InlineRubricCriterionModel](#model-inlinerubriccriterionmodel)] |
|  | `criterion_id` |  | req | `string` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
|  | `verdict_options` |  | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `description` | req | `string` |
|  |  | `value` | req | `integer` |
|  | `weight_pct` |  | req | `integer` |
| `description` |  |  | req | `string` |
| `title` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "criteria": {
      "items": {
        "$ref": "#/components/schemas/InlineRubricCriterionModel"
      },
      "title": "Criteria",
      "type": "array"
    },
    "description": {
      "title": "Description",
      "type": "string"
    },
    "title": {
      "title": "Title",
      "type": "string"
    }
  },
  "required": [
    "title",
    "description",
    "criteria"
  ],
  "title": "InlineRubricRequestModel",
  "type": "object"
}
```

</details>

<a id="model-inprogressfeedrunsresponsemodel"></a>
### Model: InProgressFeedRunsResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `limit` |  |  | req | `integer` |
| `runs` |  |  | req | array[[FeedRunInProgressModel](#model-feedruninprogressmodel)] |
|  | `attempt_number` |  | req | `integer` |
|  | `cancel_requested_at` |  | opt | `string` (format: date-time; nullable) |
|  | `feed_id` |  | req | `string` (format: uuid) |
|  | `run_id` |  | req | `string` (format: uuid) |
|  | `slot_at` |  | req | `string` (format: date-time) |
|  | `started_at` |  | req | `string` (format: date-time) |
|  | `status` |  | req | `string` |
|  | `trigger_kind` |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "limit": {
      "title": "Limit",
      "type": "integer"
    },
    "runs": {
      "items": {
        "$ref": "#/components/schemas/FeedRunInProgressModel"
      },
      "title": "Runs",
      "type": "array"
    }
  },
  "required": [
    "runs",
    "limit"
  ],
  "title": "InProgressFeedRunsResponseModel",
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

<a id="model-manualevalcreaterequestmodel"></a>
### Model: ManualEvalCreateRequestModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `rubric` |  |  | req | [InlineRubricRequestModel-Input](#model-inlinerubricrequestmodel-input) |
|  | `criteria` |  | req | array[[InlineRubricCriterionModel](#model-inlinerubriccriterionmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `weight_pct` | req | `integer` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
| `url` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "rubric": {
      "$ref": "#/components/schemas/InlineRubricRequestModel-Input"
    },
    "url": {
      "minLength": 1,
      "title": "Url",
      "type": "string"
    }
  },
  "required": [
    "url",
    "rubric"
  ],
  "title": "ManualEvalCreateRequestModel",
  "type": "object"
}
```

</details>

<a id="model-manualevalcreateresponsemodel"></a>
### Model: ManualEvalCreateResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `job_id` |  |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "job_id": {
      "format": "uuid",
      "title": "Job Id",
      "type": "string"
    }
  },
  "required": [
    "job_id"
  ],
  "title": "ManualEvalCreateResponseModel",
  "type": "object"
}
```

</details>

<a id="model-manualevaljobsresponsemodel"></a>
### Model: ManualEvalJobsResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `jobs` |  |  | req | array[[ManualEvalJobSummaryModel](#model-manualevaljobsummarymodel)] |
|  | `author` |  | opt | `string` (nullable) |
|  | `evaluated_at` |  | opt | `string` (format: date-time; nullable) |
|  | `external_id` |  | req | `string` |
|  | `job_error_code` |  | opt | `string` (nullable) |
|  | `job_error_message` |  | opt | `string` (nullable) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `job_status` |  | req | `string` |
|  | `requested_at` |  | req | `string` (format: date-time) |
|  | `rubric_slug` |  | req | `string` |
|  | `rubric_title` |  | req | `string` |
|  | `source_created_at` |  | opt | `string` (format: date-time; nullable) |
|  | `text` |  | req | `string` |
|  | `url` |  | opt | `string` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "jobs": {
      "items": {
        "$ref": "#/components/schemas/ManualEvalJobSummaryModel"
      },
      "title": "Jobs",
      "type": "array"
    }
  },
  "required": [
    "jobs"
  ],
  "title": "ManualEvalJobsResponseModel",
  "type": "object"
}
```

</details>

<a id="model-manualevaljobsummarymodel"></a>
### Model: ManualEvalJobSummaryModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `author` |  |  | opt | `string` (nullable) |
| `evaluated_at` |  |  | opt | `string` (format: date-time; nullable) |
| `external_id` |  |  | req | `string` |
| `job_error_code` |  |  | opt | `string` (nullable) |
| `job_error_message` |  |  | opt | `string` (nullable) |
| `job_id` |  |  | req | `string` (format: uuid) |
| `job_status` |  |  | req | `string` |
| `requested_at` |  |  | req | `string` (format: date-time) |
| `rubric_slug` |  |  | req | `string` |
| `rubric_title` |  |  | req | `string` |
| `source_created_at` |  |  | opt | `string` (format: date-time; nullable) |
| `text` |  |  | req | `string` |
| `url` |  |  | opt | `string` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "author": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Author"
    },
    "evaluated_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Evaluated At"
    },
    "external_id": {
      "title": "External Id",
      "type": "string"
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
      "title": "Job Status",
      "type": "string"
    },
    "requested_at": {
      "format": "date-time",
      "title": "Requested At",
      "type": "string"
    },
    "rubric_slug": {
      "title": "Rubric Slug",
      "type": "string"
    },
    "rubric_title": {
      "title": "Rubric Title",
      "type": "string"
    },
    "source_created_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Source Created At"
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
    "job_status",
    "requested_at",
    "rubric_slug",
    "rubric_title",
    "external_id",
    "text"
  ],
  "title": "ManualEvalJobSummaryModel",
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
|  | `text` |  | req | `string` |
| `reference_answer` |  |  | req | [ReferenceAnswer](#model-referenceanswer) |
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

<a id="model-minertaskbatchmodel"></a>
### Model: MinerTaskBatchModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifacts` |  |  | req | array[[ScriptArtifactModel](#model-scriptartifactmodel)] |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `content_hash` |  | req | `string` |
|  | `size_bytes` |  | req | `integer` |
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
|  |  | `text` | req | `string` |
|  | `reference_answer` |  | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `text` | req | `string` |
|  | `task_id` |  | req | `string` (format: uuid) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "artifacts": {
      "items": {
        "$ref": "#/components/schemas/ScriptArtifactModel"
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

<a id="model-minertaskinputmodel"></a>
### Model: MinerTaskInputModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `budget_usd` |  |  | opt | `number` (default: 0.5) |
| `query` |  |  | req | [Query](#model-query) |
|  | `text` |  | req | `string` |
| `reference_answer` |  |  | req | [ReferenceAnswer](#model-referenceanswer) |
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

<a id="model-monitoringartifactaggregatemodel"></a>
### Model: MonitoringArtifactAggregateModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `avg_score` |  |  | req | `number` |
| `completed_run_count` |  |  | req | `integer` |
| `cost_totals` |  |  | req | [MonitoringCostTotalsModel](#model-monitoringcosttotalsmodel) |
|  | `llm_call_count` |  | req | `integer` |
|  | `llm_cost_usd` |  | req | `number` |
|  | `llm_total_tokens` |  | req | `integer` |
|  | `search_tool_call_count` |  | req | `integer` |
|  | `search_tool_cost_usd` |  | req | `number` |
|  | `total_cost_usd` |  | req | `number` |
| `error_counts` |  |  | opt | array[[MonitoringErrorCountModel](#model-monitoringerrorcountmodel)] (default: []) |
|  | `count` |  | req | `integer` |
|  | `error_code` |  | req | `string` |
| `expected_run_count` |  |  | req | `integer` |
| `llm_models` |  |  | opt | array[[MonitoringLlmModelUsageModel](#model-monitoringllmmodelusagemodel)] (default: []) |
|  | `call_count` |  | req | `integer` |
|  | `completion_tokens` |  | req | `integer` |
|  | `cost_usd` |  | req | `number` |
|  | `model` |  | req | `string` |
|  | `prompt_tokens` |  | req | `integer` |
|  | `total_tokens` |  | req | `integer` |
| `miner_uid` |  |  | req | `integer` |
| `total_score` |  |  | req | `number` |

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
    "avg_score": {
      "title": "Avg Score",
      "type": "number"
    },
    "completed_run_count": {
      "title": "Completed Run Count",
      "type": "integer"
    },
    "cost_totals": {
      "$ref": "#/components/schemas/MonitoringCostTotalsModel"
    },
    "error_counts": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/MonitoringErrorCountModel"
      },
      "title": "Error Counts",
      "type": "array"
    },
    "expected_run_count": {
      "title": "Expected Run Count",
      "type": "integer"
    },
    "llm_models": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/MonitoringLlmModelUsageModel"
      },
      "title": "Llm Models",
      "type": "array"
    },
    "miner_uid": {
      "title": "Miner Uid",
      "type": "integer"
    },
    "total_score": {
      "title": "Total Score",
      "type": "number"
    }
  },
  "required": [
    "miner_uid",
    "artifact_id",
    "expected_run_count",
    "completed_run_count",
    "total_score",
    "avg_score",
    "cost_totals"
  ],
  "title": "MonitoringArtifactAggregateModel",
  "type": "object"
}
```

</details>

<a id="model-monitoringbatchdetailmodel"></a>
### Model: MonitoringBatchDetailModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_aggregates` |  |  | req | array[[MonitoringArtifactAggregateModel](#model-monitoringartifactaggregatemodel)] |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `avg_score` |  | req | `number` |
|  | `completed_run_count` |  | req | `integer` |
|  | `cost_totals` |  | req | [MonitoringCostTotalsModel](#model-monitoringcosttotalsmodel) |
|  |  | `llm_call_count` | req | `integer` |
|  |  | `llm_cost_usd` | req | `number` |
|  |  | `llm_total_tokens` | req | `integer` |
|  |  | `search_tool_call_count` | req | `integer` |
|  |  | `search_tool_cost_usd` | req | `number` |
|  |  | `total_cost_usd` | req | `number` |
|  | `error_counts` |  | opt | array[[MonitoringErrorCountModel](#model-monitoringerrorcountmodel)] (default: []) |
|  |  | `count` | req | `integer` |
|  |  | `error_code` | req | `string` |
|  | `expected_run_count` |  | req | `integer` |
|  | `llm_models` |  | opt | array[[MonitoringLlmModelUsageModel](#model-monitoringllmmodelusagemodel)] (default: []) |
|  |  | `call_count` | req | `integer` |
|  |  | `completion_tokens` | req | `integer` |
|  |  | `cost_usd` | req | `number` |
|  |  | `model` | req | `string` |
|  |  | `prompt_tokens` | req | `integer` |
|  |  | `total_tokens` | req | `integer` |
|  | `miner_uid` |  | req | `integer` |
|  | `total_score` |  | req | `number` |
| `batch` |  |  | req | [MinerTaskBatchModel](#model-minertaskbatchmodel) |
|  | `artifacts` |  | req | array[[ScriptArtifactModel](#model-scriptartifactmodel)] |
|  |  | `artifact_id` | req | `string` (format: uuid) |
|  |  | `content_hash` | req | `string` |
|  |  | `size_bytes` | req | `integer` |
|  |  | `uid` | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `champion_artifact_id` |  | req | `string` (format: uuid; nullable) |
|  | `completed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `created_at` |  | req | `string` (format: date-time) |
|  | `cutoff_at` |  | req | `string` (format: date-time) |
|  | `failed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `tasks` |  | req | array[[MinerTask](#model-minertask)] |
|  |  | `budget_usd` | opt | `number` (default: 0.05) |
|  |  | `query` | req | [Query](#model-query) |
|  |  | `reference_answer` | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `task_id` | req | `string` (format: uuid) |
| `completed_validator_hotkeys` |  |  | req | array[`string`] |
| `cost_totals` |  |  | req | [MonitoringCostTotalsModel](#model-monitoringcosttotalsmodel) |
|  | `llm_call_count` |  | req | `integer` |
|  | `llm_cost_usd` |  | req | `number` |
|  | `llm_total_tokens` |  | req | `integer` |
|  | `search_tool_call_count` |  | req | `integer` |
|  | `search_tool_cost_usd` |  | req | `number` |
|  | `total_cost_usd` |  | req | `number` |
| `dispatch_failed_validator_hotkeys` |  |  | req | array[`string`] |
| `pending_validator_hotkeys` |  |  | req | array[`string`] |
| `summary` |  |  | req | [MonitoringBatchSummaryModel](#model-monitoringbatchsummarymodel) |
|  | `artifact_count` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `champion_artifact_id` |  | opt | `string` (format: uuid; nullable) |
|  | `completed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `created_at` |  | req | `string` (format: date-time) |
|  | `cutoff_at` |  | req | `string` (format: date-time) |
|  | `failed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `status` |  | req | `string` |
|  | `task_count` |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "artifact_aggregates": {
      "items": {
        "$ref": "#/components/schemas/MonitoringArtifactAggregateModel"
      },
      "title": "Artifact Aggregates",
      "type": "array"
    },
    "batch": {
      "$ref": "#/components/schemas/MinerTaskBatchModel"
    },
    "completed_validator_hotkeys": {
      "items": {
        "type": "string"
      },
      "title": "Completed Validator Hotkeys",
      "type": "array"
    },
    "cost_totals": {
      "$ref": "#/components/schemas/MonitoringCostTotalsModel"
    },
    "dispatch_failed_validator_hotkeys": {
      "items": {
        "type": "string"
      },
      "title": "Dispatch Failed Validator Hotkeys",
      "type": "array"
    },
    "pending_validator_hotkeys": {
      "items": {
        "type": "string"
      },
      "title": "Pending Validator Hotkeys",
      "type": "array"
    },
    "summary": {
      "$ref": "#/components/schemas/MonitoringBatchSummaryModel"
    }
  },
  "required": [
    "summary",
    "batch",
    "pending_validator_hotkeys",
    "completed_validator_hotkeys",
    "dispatch_failed_validator_hotkeys",
    "cost_totals",
    "artifact_aggregates"
  ],
  "title": "MonitoringBatchDetailModel",
  "type": "object"
}
```

</details>

<a id="model-monitoringbatchlistmodel"></a>
### Model: MonitoringBatchListModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batches` |  |  | req | array[[MonitoringBatchSummaryModel](#model-monitoringbatchsummarymodel)] |
|  | `artifact_count` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `champion_artifact_id` |  | opt | `string` (format: uuid; nullable) |
|  | `completed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `created_at` |  | req | `string` (format: date-time) |
|  | `cutoff_at` |  | req | `string` (format: date-time) |
|  | `failed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `status` |  | req | `string` |
|  | `task_count` |  | req | `integer` |
| `before` |  |  | opt | `string` (format: date-time; nullable) |
| `limit` |  |  | req | `integer` |
| `next_before` |  |  | opt | `string` (format: date-time; nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "batches": {
      "items": {
        "$ref": "#/components/schemas/MonitoringBatchSummaryModel"
      },
      "title": "Batches",
      "type": "array"
    },
    "before": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Before"
    },
    "limit": {
      "title": "Limit",
      "type": "integer"
    },
    "next_before": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Next Before"
    }
  },
  "required": [
    "batches",
    "limit"
  ],
  "title": "MonitoringBatchListModel",
  "type": "object"
}
```

</details>

<a id="model-monitoringbatchsummarymodel"></a>
### Model: MonitoringBatchSummaryModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_count` |  |  | req | `integer` |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `champion_artifact_id` |  |  | opt | `string` (format: uuid; nullable) |
| `completed_at` |  |  | opt | `string` (format: date-time; nullable) |
| `created_at` |  |  | req | `string` (format: date-time) |
| `cutoff_at` |  |  | req | `string` (format: date-time) |
| `failed_at` |  |  | opt | `string` (format: date-time; nullable) |
| `status` |  |  | req | `string` |
| `task_count` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "artifact_count": {
      "title": "Artifact Count",
      "type": "integer"
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
    "status": {
      "title": "Status",
      "type": "string"
    },
    "task_count": {
      "title": "Task Count",
      "type": "integer"
    }
  },
  "required": [
    "batch_id",
    "status",
    "created_at",
    "cutoff_at",
    "artifact_count",
    "task_count"
  ],
  "title": "MonitoringBatchSummaryModel",
  "type": "object"
}
```

</details>

<a id="model-monitoringchampionmodel"></a>
### Model: MonitoringChampionModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `miner_uid` |  |  | opt | `integer` (nullable) |
| `script_id` |  |  | opt | `string` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "miner_uid": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Miner Uid"
    },
    "script_id": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Script Id"
    }
  },
  "title": "MonitoringChampionModel",
  "type": "object"
}
```

</details>

<a id="model-monitoringcosttotalsmodel"></a>
### Model: MonitoringCostTotalsModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `llm_call_count` |  |  | req | `integer` |
| `llm_cost_usd` |  |  | req | `number` |
| `llm_total_tokens` |  |  | req | `integer` |
| `search_tool_call_count` |  |  | req | `integer` |
| `search_tool_cost_usd` |  |  | req | `number` |
| `total_cost_usd` |  |  | req | `number` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "llm_call_count": {
      "title": "Llm Call Count",
      "type": "integer"
    },
    "llm_cost_usd": {
      "title": "Llm Cost Usd",
      "type": "number"
    },
    "llm_total_tokens": {
      "title": "Llm Total Tokens",
      "type": "integer"
    },
    "search_tool_call_count": {
      "title": "Search Tool Call Count",
      "type": "integer"
    },
    "search_tool_cost_usd": {
      "title": "Search Tool Cost Usd",
      "type": "number"
    },
    "total_cost_usd": {
      "title": "Total Cost Usd",
      "type": "number"
    }
  },
  "required": [
    "llm_cost_usd",
    "search_tool_cost_usd",
    "total_cost_usd",
    "llm_total_tokens",
    "llm_call_count",
    "search_tool_call_count"
  ],
  "title": "MonitoringCostTotalsModel",
  "type": "object"
}
```

</details>

<a id="model-monitoringerrorcountmodel"></a>
### Model: MonitoringErrorCountModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `count` |  |  | req | `integer` |
| `error_code` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "count": {
      "title": "Count",
      "type": "integer"
    },
    "error_code": {
      "title": "Error Code",
      "type": "string"
    }
  },
  "required": [
    "error_code",
    "count"
  ],
  "title": "MonitoringErrorCountModel",
  "type": "object"
}
```

</details>

<a id="model-monitoringllmmodelusagemodel"></a>
### Model: MonitoringLlmModelUsageModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `call_count` |  |  | req | `integer` |
| `completion_tokens` |  |  | req | `integer` |
| `cost_usd` |  |  | req | `number` |
| `model` |  |  | req | `string` |
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
    "cost_usd": {
      "title": "Cost Usd",
      "type": "number"
    },
    "model": {
      "title": "Model",
      "type": "string"
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
    "model",
    "cost_usd",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "call_count"
  ],
  "title": "MonitoringLlmModelUsageModel",
  "type": "object"
}
```

</details>

<a id="model-monitoringoverviewmodel"></a>
### Model: MonitoringOverviewModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `champion` |  |  | req | [MonitoringChampionModel](#model-monitoringchampionmodel) |
|  | `miner_uid` |  | opt | `integer` (nullable) |
|  | `script_id` |  | opt | `string` (nullable) |
| `latest_batch` |  |  | opt | [MonitoringBatchSummaryModel](#model-monitoringbatchsummarymodel) (nullable) |
|  | `artifact_count` |  | req | `integer` |
|  | `batch_id` |  | req | `string` (format: uuid) |
|  | `champion_artifact_id` |  | opt | `string` (format: uuid; nullable) |
|  | `completed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `created_at` |  | req | `string` (format: date-time) |
|  | `cutoff_at` |  | req | `string` (format: date-time) |
|  | `failed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `status` |  | req | `string` |
|  | `task_count` |  | req | `integer` |
| `latest_batch_cost_totals` |  |  | opt | [MonitoringCostTotalsModel](#model-monitoringcosttotalsmodel) (nullable) |
|  | `llm_call_count` |  | req | `integer` |
|  | `llm_cost_usd` |  | req | `number` |
|  | `llm_total_tokens` |  | req | `integer` |
|  | `search_tool_call_count` |  | req | `integer` |
|  | `search_tool_cost_usd` |  | req | `number` |
|  | `total_cost_usd` |  | req | `number` |
| `runtime` |  |  | req | [MonitoringRuntimeConfigModel](#model-monitoringruntimeconfigmodel) |
|  | `miner_evaluation_timeout_seconds` |  | req | `number` |
|  | `miner_task_interval_minutes` |  | req | `integer` |
|  | `sandbox_image` |  | req | `string` |
| `validator_endpoints` |  |  | req | array[[MonitoringValidatorEndpointModel](#model-monitoringvalidatorendpointmodel)] |
|  | `base_url` |  | req | `string` |
|  | `first_registered_at` |  | req | `string` (format: date-time) |
|  | `health_checked_at` |  | opt | `string` (format: date-time; nullable) |
|  | `health_error` |  | opt | `string` (nullable) |
|  | `health_status` |  | req | `string` |
|  | `hotkey` |  | req | `string` |
|  | `last_eval_completed_at` |  | opt | `string` (format: date-time; nullable) |
|  | `last_registered_at` |  | req | `string` (format: date-time) |
| `validator_health` |  |  | req | [MonitoringValidatorHealthCountsModel](#model-monitoringvalidatorhealthcountsmodel) |
|  | `healthy` |  | req | `integer` |
|  | `unhealthy` |  | req | `integer` |
|  | `unknown` |  | req | `integer` |
| `weights` |  |  | opt | [MonitoringWeightsModel](#model-monitoringweightsmodel) (nullable) |
|  | `champion_uid` |  | opt | `integer` (nullable) |
|  | `weights` |  | req | `object` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "champion": {
      "$ref": "#/components/schemas/MonitoringChampionModel"
    },
    "latest_batch": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/MonitoringBatchSummaryModel"
        },
        {
          "type": "null"
        }
      ]
    },
    "latest_batch_cost_totals": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/MonitoringCostTotalsModel"
        },
        {
          "type": "null"
        }
      ]
    },
    "runtime": {
      "$ref": "#/components/schemas/MonitoringRuntimeConfigModel"
    },
    "validator_endpoints": {
      "items": {
        "$ref": "#/components/schemas/MonitoringValidatorEndpointModel"
      },
      "title": "Validator Endpoints",
      "type": "array"
    },
    "validator_health": {
      "$ref": "#/components/schemas/MonitoringValidatorHealthCountsModel"
    },
    "weights": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/MonitoringWeightsModel"
        },
        {
          "type": "null"
        }
      ]
    }
  },
  "required": [
    "champion",
    "validator_health",
    "validator_endpoints",
    "runtime"
  ],
  "title": "MonitoringOverviewModel",
  "type": "object"
}
```

</details>

<a id="model-monitoringresultrowmodel"></a>
### Model: MonitoringResultRowModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `cost_totals` |  |  | req | [MonitoringCostTotalsModel](#model-monitoringcosttotalsmodel) |
|  | `llm_call_count` |  | req | `integer` |
|  | `llm_cost_usd` |  | req | `number` |
|  | `llm_total_tokens` |  | req | `integer` |
|  | `search_tool_call_count` |  | req | `integer` |
|  | `search_tool_cost_usd` |  | req | `number` |
|  | `total_cost_usd` |  | req | `number` |
| `llm_models` |  |  | opt | array[[MonitoringLlmModelUsageModel](#model-monitoringllmmodelusagemodel)] (default: []) |
|  | `call_count` |  | req | `integer` |
|  | `completion_tokens` |  | req | `integer` |
|  | `cost_usd` |  | req | `number` |
|  | `model` |  | req | `string` |
|  | `prompt_tokens` |  | req | `integer` |
|  | `total_tokens` |  | req | `integer` |
| `miner_uid` |  |  | req | `integer` |
| `payload_json` |  |  | opt | `object` (nullable) |
| `received_at` |  |  | req | `string` (format: date-time) |
| `response` |  |  | opt | [Response](#model-response) (nullable) |
|  | `text` |  | req | `string` |
| `score` |  |  | req | `number` |
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
| `task_id` |  |  | req | `string` (format: uuid) |
| `validator_hotkey` |  |  | req | `string` |
| `validator_uid` |  |  | req | `integer` |

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
    "batch_id": {
      "format": "uuid",
      "title": "Batch Id",
      "type": "string"
    },
    "cost_totals": {
      "$ref": "#/components/schemas/MonitoringCostTotalsModel"
    },
    "llm_models": {
      "default": [],
      "items": {
        "$ref": "#/components/schemas/MonitoringLlmModelUsageModel"
      },
      "title": "Llm Models",
      "type": "array"
    },
    "miner_uid": {
      "title": "Miner Uid",
      "type": "integer"
    },
    "payload_json": {
      "anyOf": [
        {
          "additionalProperties": true,
          "type": "object"
        },
        {
          "type": "null"
        }
      ],
      "title": "Payload Json"
    },
    "received_at": {
      "format": "date-time",
      "title": "Received At",
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
    "score": {
      "title": "Score",
      "type": "number"
    },
    "specifics": {
      "$ref": "#/components/schemas/EvaluationDetails"
    },
    "task_id": {
      "format": "uuid",
      "title": "Task Id",
      "type": "string"
    },
    "validator_hotkey": {
      "title": "Validator Hotkey",
      "type": "string"
    },
    "validator_uid": {
      "title": "Validator Uid",
      "type": "integer"
    }
  },
  "required": [
    "batch_id",
    "validator_hotkey",
    "validator_uid",
    "miner_uid",
    "artifact_id",
    "task_id",
    "score",
    "received_at",
    "specifics",
    "cost_totals"
  ],
  "title": "MonitoringResultRowModel",
  "type": "object"
}
```

</details>

<a id="model-monitoringruntimeconfigmodel"></a>
### Model: MonitoringRuntimeConfigModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `miner_evaluation_timeout_seconds` |  |  | req | `number` |
| `miner_task_interval_minutes` |  |  | req | `integer` |
| `sandbox_image` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "miner_evaluation_timeout_seconds": {
      "title": "Miner Evaluation Timeout Seconds",
      "type": "number"
    },
    "miner_task_interval_minutes": {
      "title": "Miner Task Interval Minutes",
      "type": "integer"
    },
    "sandbox_image": {
      "title": "Sandbox Image",
      "type": "string"
    }
  },
  "required": [
    "miner_task_interval_minutes",
    "miner_evaluation_timeout_seconds",
    "sandbox_image"
  ],
  "title": "MonitoringRuntimeConfigModel",
  "type": "object"
}
```

</details>

<a id="model-monitoringscriptmodel"></a>
### Model: MonitoringScriptModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `content_b64` |  |  | opt | `string` (nullable) |
| `content_hash` |  |  | req | `string` |
| `revealed_at` |  |  | opt | `string` (format: date-time; nullable) |
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
    "content_b64": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Content B64"
    },
    "content_hash": {
      "title": "Content Hash",
      "type": "string"
    },
    "revealed_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Revealed At"
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
  "title": "MonitoringScriptModel",
  "type": "object"
}
```

</details>

<a id="model-monitoringvalidatorendpointmodel"></a>
### Model: MonitoringValidatorEndpointModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `base_url` |  |  | req | `string` |
| `first_registered_at` |  |  | req | `string` (format: date-time) |
| `health_checked_at` |  |  | opt | `string` (format: date-time; nullable) |
| `health_error` |  |  | opt | `string` (nullable) |
| `health_status` |  |  | req | `string` |
| `hotkey` |  |  | req | `string` |
| `last_eval_completed_at` |  |  | opt | `string` (format: date-time; nullable) |
| `last_registered_at` |  |  | req | `string` (format: date-time) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "base_url": {
      "title": "Base Url",
      "type": "string"
    },
    "first_registered_at": {
      "format": "date-time",
      "title": "First Registered At",
      "type": "string"
    },
    "health_checked_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Health Checked At"
    },
    "health_error": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Health Error"
    },
    "health_status": {
      "title": "Health Status",
      "type": "string"
    },
    "hotkey": {
      "title": "Hotkey",
      "type": "string"
    },
    "last_eval_completed_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Last Eval Completed At"
    },
    "last_registered_at": {
      "format": "date-time",
      "title": "Last Registered At",
      "type": "string"
    }
  },
  "required": [
    "hotkey",
    "base_url",
    "first_registered_at",
    "last_registered_at",
    "health_status"
  ],
  "title": "MonitoringValidatorEndpointModel",
  "type": "object"
}
```

</details>

<a id="model-monitoringvalidatorhealthcountsmodel"></a>
### Model: MonitoringValidatorHealthCountsModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `healthy` |  |  | req | `integer` |
| `unhealthy` |  |  | req | `integer` |
| `unknown` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "healthy": {
      "title": "Healthy",
      "type": "integer"
    },
    "unhealthy": {
      "title": "Unhealthy",
      "type": "integer"
    },
    "unknown": {
      "title": "Unknown",
      "type": "integer"
    }
  },
  "required": [
    "healthy",
    "unhealthy",
    "unknown"
  ],
  "title": "MonitoringValidatorHealthCountsModel",
  "type": "object"
}
```

</details>

<a id="model-monitoringweightsmodel"></a>
### Model: MonitoringWeightsModel

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
  "title": "MonitoringWeightsModel",
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
|  |  | `text` | req | `string` |
|  | `reference_answer` |  | req | [ReferenceAnswer](#model-referenceanswer) |
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

<a id="model-progresssnapshotresponse"></a>
### Model: ProgressSnapshotResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_count` |  |  | req | `integer` |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `created_at` |  |  | req | `string` (format: date-time) |
| `cutoff_at` |  |  | req | `string` (format: date-time) |
| `status` |  |  | req | `string` |
| `task_count` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "artifact_count": {
      "title": "Artifact Count",
      "type": "integer"
    },
    "batch_id": {
      "format": "uuid",
      "title": "Batch Id",
      "type": "string"
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
    "status": {
      "title": "Status",
      "type": "string"
    },
    "task_count": {
      "title": "Task Count",
      "type": "integer"
    }
  },
  "required": [
    "batch_id",
    "status",
    "created_at",
    "cutoff_at",
    "artifact_count",
    "task_count"
  ],
  "title": "ProgressSnapshotResponse",
  "type": "object"
}
```

</details>

<a id="model-publiccriterionassessmentmodel"></a>
### Model: PublicCriterionAssessmentModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `aggregate_score` |  |  | req | `number` |
| `criterion_evaluations` |  |  | req | array[[PublicCriterionEvaluationModel](#model-publiccriterionevaluationmodel)] |
|  | `citations` |  | opt | array[[CitationModel](#model-citationmodel)] (default: []) |
|  |  | `note` | req | `string` |
|  |  | `url` | req | `string` |
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
            "$ref": "#/components/schemas/PublicCriterionEvaluationModel"
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
  "title": "PublicCriterionAssessmentModel",
  "type": "object"
}
```

</details>

<a id="model-publiccriterionevaluationmodel"></a>
### Model: PublicCriterionEvaluationModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `citations` |  |  | opt | array[[CitationModel](#model-citationmodel)] (default: []) |
|  | `note` |  | req | `string` |
|  | `url` |  | req | `string` |
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
  "title": "PublicCriterionEvaluationModel",
  "type": "object"
}
```

</details>

<a id="model-publicexternalevalresultmodel"></a>
### Model: PublicExternalEvalResultModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `criteria` |  |  | req | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] |
|  | `aggregate_score` |  | req | `number` |
|  | `criterion_evaluations` |  | req | array[[PublicCriterionEvaluationModel](#model-publiccriterionevaluationmodel)] |
|  |  | `citations` | opt | array[[CitationModel](#model-citationmodel)] (default: []) |
|  |  | `justification` | req | `string` |
|  |  | `spans` | opt | array[[SpanModel](#model-spanmodel)] (default: []) |
|  |  | `verdict` | req | `integer` |
|  | `criterion_id` |  | req | `string` |
|  | `verdict_options` |  | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `description` | req | `string` |
|  |  | `value` | req | `integer` |
| `overall_rationale` |  |  | opt | `string` (nullable) |
| `rubric_score` |  |  | req | `number` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "criteria": {
      "items": {
        "$ref": "#/components/schemas/PublicCriterionAssessmentModel"
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
    "rubric_score": {
      "title": "Rubric Score",
      "type": "number"
    }
  },
  "required": [
    "criteria",
    "rubric_score"
  ],
  "title": "PublicExternalEvalResultModel",
  "type": "object"
}
```

</details>

<a id="model-publicfeedleaderboardresponsemodel"></a>
### Model: PublicFeedLeaderboardResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `cursor` |  |  | opt | `string` (nullable) |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `items` |  |  | req | array[[PublicFeedRunItemModel](#model-publicfeedrunitemmodel)] |
|  | `author` |  | opt | `string` (nullable) |
|  | `content_review_rubric_result` |  | opt | [PublicExternalEvalResultModel](#model-publicexternalevalresultmodel) (nullable) |
|  |  | `criteria` | req | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] |
|  |  | `overall_rationale` | opt | `string` (nullable) |
|  |  | `rubric_score` | req | `number` |
|  | `content_review_topic_gate` |  | req | [PublicTopicGateModel](#model-publictopicgatemodel) |
|  |  | `criteria` | opt | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] (default: []) |
|  |  | `score` | opt | `number` (nullable) |
|  | `external_id` |  | req | `string` |
|  | `is_excluded` |  | opt | `boolean` (default: False) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `provider_context` |  | opt | `object` (nullable) |
|  | `source_created_at` |  | opt | `string` (format: date-time; nullable) |
|  | `text` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (nullable) |
| `limit` |  |  | req | `integer` |
| `next_cursor` |  |  | opt | `string` (nullable) |
| `q` |  |  | opt | `string` (nullable) |
| `rubric` |  |  | req | [PublicRubricModel](#model-publicrubricmodel) |
|  | `criteria` |  | req | array[[PublicRubricCriterionModel](#model-publicrubriccriterionmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `weight_pct` | req | `integer` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
| `run_at` |  |  | opt | `string` (format: date-time; nullable) |
| `run_id` |  |  | opt | `string` (format: uuid; nullable) |
| `sort` |  |  | req | `string` |
| `status` |  |  | opt | `string` (nullable) |
| `topic` |  |  | req | [PublicFeedTopicModel](#model-publicfeedtopicmodel) |
|  | `description` |  | opt | `string` (nullable) |
|  | `keyword` |  | req | `string` |
|  | `title` |  | req | `string` |
| `total_count` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "cursor": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Cursor"
    },
    "feed_id": {
      "format": "uuid",
      "title": "Feed Id",
      "type": "string"
    },
    "items": {
      "items": {
        "$ref": "#/components/schemas/PublicFeedRunItemModel"
      },
      "title": "Items",
      "type": "array"
    },
    "limit": {
      "title": "Limit",
      "type": "integer"
    },
    "next_cursor": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Next Cursor"
    },
    "q": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Q"
    },
    "rubric": {
      "$ref": "#/components/schemas/PublicRubricModel"
    },
    "run_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Run At"
    },
    "run_id": {
      "anyOf": [
        {
          "format": "uuid",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Run Id"
    },
    "sort": {
      "title": "Sort",
      "type": "string"
    },
    "status": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Status"
    },
    "topic": {
      "$ref": "#/components/schemas/PublicFeedTopicModel"
    },
    "total_count": {
      "title": "Total Count",
      "type": "integer"
    }
  },
  "required": [
    "feed_id",
    "topic",
    "rubric",
    "items",
    "total_count",
    "limit",
    "sort"
  ],
  "title": "PublicFeedLeaderboardResponseModel",
  "type": "object"
}
```

</details>

<a id="model-publicfeedrundigestresponsemodel"></a>
### Model: PublicFeedRunDigestResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `curated_count` |  |  | req | `integer` |
| `digest_status` |  |  | req | `string` |
| `evaluated_count` |  |  | req | `integer` |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `items` |  |  | req | array[[PublicFeedRunItemModel](#model-publicfeedrunitemmodel)] |
|  | `author` |  | opt | `string` (nullable) |
|  | `content_review_rubric_result` |  | opt | [PublicExternalEvalResultModel](#model-publicexternalevalresultmodel) (nullable) |
|  |  | `criteria` | req | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] |
|  |  | `overall_rationale` | opt | `string` (nullable) |
|  |  | `rubric_score` | req | `number` |
|  | `content_review_topic_gate` |  | req | [PublicTopicGateModel](#model-publictopicgatemodel) |
|  |  | `criteria` | opt | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] (default: []) |
|  |  | `score` | opt | `number` (nullable) |
|  | `external_id` |  | req | `string` |
|  | `is_excluded` |  | opt | `boolean` (default: False) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `provider_context` |  | opt | `object` (nullable) |
|  | `source_created_at` |  | opt | `string` (format: date-time; nullable) |
|  | `text` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (nullable) |
| `run_at` |  |  | req | `string` (format: date-time) |
| `run_id` |  |  | req | `string` (format: uuid) |
| `settings` |  |  | req | [PublicFeedRunSettingsModel](#model-publicfeedrunsettingsmodel) |
|  | `digest_threshold_pct` |  | req | `integer` |
|  | `interval_hours` |  | opt | `integer` (nullable) |
|  | `item_limit` |  | req | `integer` |
|  | `lookback_minutes` |  | req | `integer` |
|  | `min_likes` |  | req | `integer` |
|  | `providers` |  | req | array[`string`] |
| `summary_markdown` |  |  | opt | `string` (nullable) |
| `threshold_pct` |  |  | req | `integer` |
| `topic` |  |  | req | [PublicFeedTopicModel](#model-publicfeedtopicmodel) |
|  | `description` |  | opt | `string` (nullable) |
|  | `keyword` |  | req | `string` |
|  | `title` |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "curated_count": {
      "title": "Curated Count",
      "type": "integer"
    },
    "digest_status": {
      "title": "Digest Status",
      "type": "string"
    },
    "evaluated_count": {
      "title": "Evaluated Count",
      "type": "integer"
    },
    "feed_id": {
      "format": "uuid",
      "title": "Feed Id",
      "type": "string"
    },
    "items": {
      "items": {
        "$ref": "#/components/schemas/PublicFeedRunItemModel"
      },
      "title": "Items",
      "type": "array"
    },
    "run_at": {
      "format": "date-time",
      "title": "Run At",
      "type": "string"
    },
    "run_id": {
      "format": "uuid",
      "title": "Run Id",
      "type": "string"
    },
    "settings": {
      "$ref": "#/components/schemas/PublicFeedRunSettingsModel"
    },
    "summary_markdown": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Summary Markdown"
    },
    "threshold_pct": {
      "title": "Threshold Pct",
      "type": "integer"
    },
    "topic": {
      "$ref": "#/components/schemas/PublicFeedTopicModel"
    }
  },
  "required": [
    "feed_id",
    "run_id",
    "run_at",
    "digest_status",
    "threshold_pct",
    "evaluated_count",
    "curated_count",
    "topic",
    "settings",
    "items"
  ],
  "title": "PublicFeedRunDigestResponseModel",
  "type": "object"
}
```

</details>

<a id="model-publicfeedrunitemmodel"></a>
### Model: PublicFeedRunItemModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `author` |  |  | opt | `string` (nullable) |
| `content_review_rubric_result` |  |  | opt | [PublicExternalEvalResultModel](#model-publicexternalevalresultmodel) (nullable) |
|  | `criteria` |  | req | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] |
|  |  | `aggregate_score` | req | `number` |
|  |  | `criterion_evaluations` | req | array[[PublicCriterionEvaluationModel](#model-publiccriterionevaluationmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  | `overall_rationale` |  | opt | `string` (nullable) |
|  | `rubric_score` |  | req | `number` |
| `content_review_topic_gate` |  |  | req | [PublicTopicGateModel](#model-publictopicgatemodel) |
|  | `criteria` |  | opt | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] (default: []) |
|  |  | `aggregate_score` | req | `number` |
|  |  | `criterion_evaluations` | req | array[[PublicCriterionEvaluationModel](#model-publiccriterionevaluationmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  | `score` |  | opt | `number` (nullable) |
| `external_id` |  |  | req | `string` |
| `is_excluded` |  |  | opt | `boolean` (default: False) |
| `job_id` |  |  | req | `string` (format: uuid) |
| `provider_context` |  |  | opt | `object` (nullable) |
| `source_created_at` |  |  | opt | `string` (format: date-time; nullable) |
| `text` |  |  | req | `string` |
| `title` |  |  | opt | `string` (nullable) |
| `url` |  |  | opt | `string` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "author": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Author"
    },
    "content_review_rubric_result": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/PublicExternalEvalResultModel"
        },
        {
          "type": "null"
        }
      ]
    },
    "content_review_topic_gate": {
      "$ref": "#/components/schemas/PublicTopicGateModel"
    },
    "external_id": {
      "title": "External Id",
      "type": "string"
    },
    "is_excluded": {
      "default": false,
      "title": "Is Excluded",
      "type": "boolean"
    },
    "job_id": {
      "format": "uuid",
      "title": "Job Id",
      "type": "string"
    },
    "provider_context": {
      "anyOf": [
        {
          "additionalProperties": true,
          "type": "object"
        },
        {
          "type": "null"
        }
      ],
      "title": "Provider Context"
    },
    "source_created_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Source Created At"
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
    "external_id",
    "text",
    "content_review_topic_gate"
  ],
  "title": "PublicFeedRunItemModel",
  "type": "object"
}
```

</details>

<a id="model-publicfeedrunreadmodel"></a>
### Model: PublicFeedRunReadModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `date` |  |  | req | `string` (format: date) |
| `is_overridden` |  |  | opt | `boolean` (default: False) |
| `run_at` |  |  | req | `string` (format: date-time) |
| `run_id` |  |  | req | `string` (format: uuid) |
| `status` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "date": {
      "format": "date",
      "title": "Date",
      "type": "string"
    },
    "is_overridden": {
      "default": false,
      "title": "Is Overridden",
      "type": "boolean"
    },
    "run_at": {
      "format": "date-time",
      "title": "Run At",
      "type": "string"
    },
    "run_id": {
      "format": "uuid",
      "title": "Run Id",
      "type": "string"
    },
    "status": {
      "title": "Status",
      "type": "string"
    }
  },
  "required": [
    "date",
    "run_id",
    "run_at",
    "status"
  ],
  "title": "PublicFeedRunReadModel",
  "type": "object"
}
```

</details>

<a id="model-publicfeedrunresponsemodel"></a>
### Model: PublicFeedRunResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `cursor` |  |  | opt | `string` (nullable) |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `items` |  |  | req | array[[PublicFeedRunItemModel](#model-publicfeedrunitemmodel)] |
|  | `author` |  | opt | `string` (nullable) |
|  | `content_review_rubric_result` |  | opt | [PublicExternalEvalResultModel](#model-publicexternalevalresultmodel) (nullable) |
|  |  | `criteria` | req | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] |
|  |  | `overall_rationale` | opt | `string` (nullable) |
|  |  | `rubric_score` | req | `number` |
|  | `content_review_topic_gate` |  | req | [PublicTopicGateModel](#model-publictopicgatemodel) |
|  |  | `criteria` | opt | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] (default: []) |
|  |  | `score` | opt | `number` (nullable) |
|  | `external_id` |  | req | `string` |
|  | `is_excluded` |  | opt | `boolean` (default: False) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `provider_context` |  | opt | `object` (nullable) |
|  | `source_created_at` |  | opt | `string` (format: date-time; nullable) |
|  | `text` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (nullable) |
| `limit` |  |  | req | `integer` |
| `next_cursor` |  |  | opt | `string` (nullable) |
| `q` |  |  | opt | `string` (nullable) |
| `rubric` |  |  | req | [PublicRubricModel](#model-publicrubricmodel) |
|  | `criteria` |  | req | array[[PublicRubricCriterionModel](#model-publicrubriccriterionmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `weight_pct` | req | `integer` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
| `run_at` |  |  | req | `string` (format: date-time) |
| `run_id` |  |  | req | `string` (format: uuid) |
| `settings` |  |  | req | [PublicFeedRunSettingsModel](#model-publicfeedrunsettingsmodel) |
|  | `digest_threshold_pct` |  | req | `integer` |
|  | `interval_hours` |  | opt | `integer` (nullable) |
|  | `item_limit` |  | req | `integer` |
|  | `lookback_minutes` |  | req | `integer` |
|  | `min_likes` |  | req | `integer` |
|  | `providers` |  | req | array[`string`] |
| `sort` |  |  | req | `string` |
| `status` |  |  | req | `string` |
| `topic` |  |  | req | [PublicFeedTopicModel](#model-publicfeedtopicmodel) |
|  | `description` |  | opt | `string` (nullable) |
|  | `keyword` |  | req | `string` |
|  | `title` |  | req | `string` |
| `total_count` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "cursor": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Cursor"
    },
    "feed_id": {
      "format": "uuid",
      "title": "Feed Id",
      "type": "string"
    },
    "items": {
      "items": {
        "$ref": "#/components/schemas/PublicFeedRunItemModel"
      },
      "title": "Items",
      "type": "array"
    },
    "limit": {
      "title": "Limit",
      "type": "integer"
    },
    "next_cursor": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Next Cursor"
    },
    "q": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Q"
    },
    "rubric": {
      "$ref": "#/components/schemas/PublicRubricModel"
    },
    "run_at": {
      "format": "date-time",
      "title": "Run At",
      "type": "string"
    },
    "run_id": {
      "format": "uuid",
      "title": "Run Id",
      "type": "string"
    },
    "settings": {
      "$ref": "#/components/schemas/PublicFeedRunSettingsModel"
    },
    "sort": {
      "title": "Sort",
      "type": "string"
    },
    "status": {
      "title": "Status",
      "type": "string"
    },
    "topic": {
      "$ref": "#/components/schemas/PublicFeedTopicModel"
    },
    "total_count": {
      "title": "Total Count",
      "type": "integer"
    }
  },
  "required": [
    "feed_id",
    "run_id",
    "run_at",
    "status",
    "topic",
    "rubric",
    "settings",
    "items",
    "total_count",
    "limit",
    "sort"
  ],
  "title": "PublicFeedRunResponseModel",
  "type": "object"
}
```

</details>

<a id="model-publicfeedrunsettingsmodel"></a>
### Model: PublicFeedRunSettingsModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `digest_threshold_pct` |  |  | req | `integer` |
| `interval_hours` |  |  | opt | `integer` (nullable) |
| `item_limit` |  |  | req | `integer` |
| `lookback_minutes` |  |  | req | `integer` |
| `min_likes` |  |  | req | `integer` |
| `providers` |  |  | req | array[`string`] |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "digest_threshold_pct": {
      "title": "Digest Threshold Pct",
      "type": "integer"
    },
    "interval_hours": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "title": "Interval Hours"
    },
    "item_limit": {
      "title": "Item Limit",
      "type": "integer"
    },
    "lookback_minutes": {
      "title": "Lookback Minutes",
      "type": "integer"
    },
    "min_likes": {
      "title": "Min Likes",
      "type": "integer"
    },
    "providers": {
      "items": {
        "type": "string"
      },
      "title": "Providers",
      "type": "array"
    }
  },
  "required": [
    "providers",
    "lookback_minutes",
    "item_limit",
    "min_likes",
    "digest_threshold_pct"
  ],
  "title": "PublicFeedRunSettingsModel",
  "type": "object"
}
```

</details>

<a id="model-publicfeedrunsresponsemodel"></a>
### Model: PublicFeedRunsResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `cursor` |  |  | opt | `string` (nullable) |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `limit` |  |  | req | `integer` |
| `next_cursor` |  |  | opt | `string` (nullable) |
| `runs` |  |  | req | array[[PublicFeedRunReadModel](#model-publicfeedrunreadmodel)] |
|  | `date` |  | req | `string` (format: date) |
|  | `is_overridden` |  | opt | `boolean` (default: False) |
|  | `run_at` |  | req | `string` (format: date-time) |
|  | `run_id` |  | req | `string` (format: uuid) |
|  | `status` |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "cursor": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Cursor"
    },
    "feed_id": {
      "format": "uuid",
      "title": "Feed Id",
      "type": "string"
    },
    "limit": {
      "title": "Limit",
      "type": "integer"
    },
    "next_cursor": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Next Cursor"
    },
    "runs": {
      "items": {
        "$ref": "#/components/schemas/PublicFeedRunReadModel"
      },
      "title": "Runs",
      "type": "array"
    }
  },
  "required": [
    "feed_id",
    "runs",
    "limit"
  ],
  "title": "PublicFeedRunsResponseModel",
  "type": "object"
}
```

</details>

<a id="model-publicfeedsubmissionresponsemodel"></a>
### Model: PublicFeedSubmissionResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `decision` |  |  | opt | `string` (nullable) |
| `evaluated_at` |  |  | opt | `string` (format: date-time; nullable) |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `item` |  |  | req | [PublicFeedRunItemModel](#model-publicfeedrunitemmodel) |
|  | `author` |  | opt | `string` (nullable) |
|  | `content_review_rubric_result` |  | opt | [PublicExternalEvalResultModel](#model-publicexternalevalresultmodel) (nullable) |
|  |  | `criteria` | req | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] |
|  |  | `overall_rationale` | opt | `string` (nullable) |
|  |  | `rubric_score` | req | `number` |
|  | `content_review_topic_gate` |  | req | [PublicTopicGateModel](#model-publictopicgatemodel) |
|  |  | `criteria` | opt | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] (default: []) |
|  |  | `score` | opt | `number` (nullable) |
|  | `external_id` |  | req | `string` |
|  | `is_excluded` |  | opt | `boolean` (default: False) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `provider_context` |  | opt | `object` (nullable) |
|  | `source_created_at` |  | opt | `string` (format: date-time; nullable) |
|  | `text` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (nullable) |
| `job_error_code` |  |  | opt | `string` (nullable) |
| `job_error_message` |  |  | opt | `string` (nullable) |
| `job_id` |  |  | req | `string` (format: uuid) |
| `job_status` |  |  | req | `string` |
| `requested_at` |  |  | req | `string` (format: date-time) |
| `rubric` |  |  | req | [PublicRubricModel](#model-publicrubricmodel) |
|  | `criteria` |  | req | array[[PublicRubricCriterionModel](#model-publicrubriccriterionmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `weight_pct` | req | `integer` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
| `topic` |  |  | req | [PublicFeedTopicModel](#model-publicfeedtopicmodel) |
|  | `description` |  | opt | `string` (nullable) |
|  | `keyword` |  | req | `string` |
|  | `title` |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
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
    "evaluated_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Evaluated At"
    },
    "feed_id": {
      "format": "uuid",
      "title": "Feed Id",
      "type": "string"
    },
    "item": {
      "$ref": "#/components/schemas/PublicFeedRunItemModel"
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
      "title": "Job Status",
      "type": "string"
    },
    "requested_at": {
      "format": "date-time",
      "title": "Requested At",
      "type": "string"
    },
    "rubric": {
      "$ref": "#/components/schemas/PublicRubricModel"
    },
    "topic": {
      "$ref": "#/components/schemas/PublicFeedTopicModel"
    }
  },
  "required": [
    "feed_id",
    "job_id",
    "job_status",
    "requested_at",
    "topic",
    "rubric",
    "item"
  ],
  "title": "PublicFeedSubmissionResponseModel",
  "type": "object"
}
```

</details>

<a id="model-publicfeedtopicmodel"></a>
### Model: PublicFeedTopicModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `description` |  |  | opt | `string` (nullable) |
| `keyword` |  |  | req | `string` |
| `title` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "description": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Description"
    },
    "keyword": {
      "title": "Keyword",
      "type": "string"
    },
    "title": {
      "title": "Title",
      "type": "string"
    }
  },
  "required": [
    "keyword",
    "title"
  ],
  "title": "PublicFeedTopicModel",
  "type": "object"
}
```

</details>

<a id="model-publicrubriccriterionmodel"></a>
### Model: PublicRubricCriterionModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `criterion_id` |  |  | req | `string` |
| `description` |  |  | req | `string` |
| `title` |  |  | req | `string` |
| `verdict_options` |  |  | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  | `description` |  | req | `string` |
|  | `value` |  | req | `integer` |
| `weight_pct` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "criterion_id": {
      "title": "Criterion Id",
      "type": "string"
    },
    "description": {
      "title": "Description",
      "type": "string"
    },
    "title": {
      "title": "Title",
      "type": "string"
    },
    "verdict_options": {
      "items": {
        "$ref": "#/components/schemas/VerdictOptionModel"
      },
      "title": "Verdict Options",
      "type": "array"
    },
    "weight_pct": {
      "title": "Weight Pct",
      "type": "integer"
    }
  },
  "required": [
    "criterion_id",
    "title",
    "description",
    "weight_pct",
    "verdict_options"
  ],
  "title": "PublicRubricCriterionModel",
  "type": "object"
}
```

</details>

<a id="model-publicrubricmodel"></a>
### Model: PublicRubricModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `criteria` |  |  | req | array[[PublicRubricCriterionModel](#model-publicrubriccriterionmodel)] |
|  | `criterion_id` |  | req | `string` |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
|  | `verdict_options` |  | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `description` | req | `string` |
|  |  | `value` | req | `integer` |
|  | `weight_pct` |  | req | `integer` |
| `description` |  |  | req | `string` |
| `title` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "criteria": {
      "items": {
        "$ref": "#/components/schemas/PublicRubricCriterionModel"
      },
      "title": "Criteria",
      "type": "array"
    },
    "description": {
      "title": "Description",
      "type": "string"
    },
    "title": {
      "title": "Title",
      "type": "string"
    }
  },
  "required": [
    "title",
    "description",
    "criteria"
  ],
  "title": "PublicRubricModel",
  "type": "object"
}
```

</details>

<a id="model-publictopicgatemodel"></a>
### Model: PublicTopicGateModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `criteria` |  |  | opt | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] (default: []) |
|  | `aggregate_score` |  | req | `number` |
|  | `criterion_evaluations` |  | req | array[[PublicCriterionEvaluationModel](#model-publiccriterionevaluationmodel)] |
|  |  | `citations` | opt | array[[CitationModel](#model-citationmodel)] (default: []) |
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
        "$ref": "#/components/schemas/PublicCriterionAssessmentModel"
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
  "title": "PublicTopicGateModel",
  "type": "object"
}
```

</details>

<a id="model-publictopscoringpostmodel"></a>
### Model: PublicTopScoringPostModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `author` |  |  | opt | `string` (nullable) |
| `content_review_rubric_result` |  |  | opt | [PublicExternalEvalResultModel](#model-publicexternalevalresultmodel) (nullable) |
|  | `criteria` |  | req | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] |
|  |  | `aggregate_score` | req | `number` |
|  |  | `criterion_evaluations` | req | array[[PublicCriterionEvaluationModel](#model-publiccriterionevaluationmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  | `overall_rationale` |  | opt | `string` (nullable) |
|  | `rubric_score` |  | req | `number` |
| `content_review_topic_gate` |  |  | req | [PublicTopicGateModel](#model-publictopicgatemodel) |
|  | `criteria` |  | opt | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] (default: []) |
|  |  | `aggregate_score` | req | `number` |
|  |  | `criterion_evaluations` | req | array[[PublicCriterionEvaluationModel](#model-publiccriterionevaluationmodel)] |
|  |  | `criterion_id` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  | `score` |  | opt | `number` (nullable) |
| `external_id` |  |  | req | `string` |
| `feed_id` |  |  | req | `string` (format: uuid) |
| `is_excluded` |  |  | opt | `boolean` (default: False) |
| `job_id` |  |  | req | `string` (format: uuid) |
| `provider_context` |  |  | opt | `object` (nullable) |
| `source_created_at` |  |  | opt | `string` (format: date-time; nullable) |
| `text` |  |  | req | `string` |
| `title` |  |  | opt | `string` (nullable) |
| `url` |  |  | opt | `string` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "author": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Author"
    },
    "content_review_rubric_result": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/PublicExternalEvalResultModel"
        },
        {
          "type": "null"
        }
      ]
    },
    "content_review_topic_gate": {
      "$ref": "#/components/schemas/PublicTopicGateModel"
    },
    "external_id": {
      "title": "External Id",
      "type": "string"
    },
    "feed_id": {
      "format": "uuid",
      "title": "Feed Id",
      "type": "string"
    },
    "is_excluded": {
      "default": false,
      "title": "Is Excluded",
      "type": "boolean"
    },
    "job_id": {
      "format": "uuid",
      "title": "Job Id",
      "type": "string"
    },
    "provider_context": {
      "anyOf": [
        {
          "additionalProperties": true,
          "type": "object"
        },
        {
          "type": "null"
        }
      ],
      "title": "Provider Context"
    },
    "source_created_at": {
      "anyOf": [
        {
          "format": "date-time",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Source Created At"
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
    "external_id",
    "text",
    "content_review_topic_gate",
    "feed_id"
  ],
  "title": "PublicTopScoringPostModel",
  "type": "object"
}
```

</details>

<a id="model-publictopscoringpostsresponsemodel"></a>
### Model: PublicTopScoringPostsResponseModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `cursor` |  |  | opt | `string` (nullable) |
| `feed_ids` |  |  | req | array[`string`] |
| `items` |  |  | req | array[[PublicTopScoringPostModel](#model-publictopscoringpostmodel)] |
|  | `author` |  | opt | `string` (nullable) |
|  | `content_review_rubric_result` |  | opt | [PublicExternalEvalResultModel](#model-publicexternalevalresultmodel) (nullable) |
|  |  | `criteria` | req | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] |
|  |  | `overall_rationale` | opt | `string` (nullable) |
|  |  | `rubric_score` | req | `number` |
|  | `content_review_topic_gate` |  | req | [PublicTopicGateModel](#model-publictopicgatemodel) |
|  |  | `criteria` | opt | array[[PublicCriterionAssessmentModel](#model-publiccriterionassessmentmodel)] (default: []) |
|  |  | `score` | opt | `number` (nullable) |
|  | `external_id` |  | req | `string` |
|  | `feed_id` |  | req | `string` (format: uuid) |
|  | `is_excluded` |  | opt | `boolean` (default: False) |
|  | `job_id` |  | req | `string` (format: uuid) |
|  | `provider_context` |  | opt | `object` (nullable) |
|  | `source_created_at` |  | opt | `string` (format: date-time; nullable) |
|  | `text` |  | req | `string` |
|  | `title` |  | opt | `string` (nullable) |
|  | `url` |  | opt | `string` (nullable) |
| `limit` |  |  | req | `integer` |
| `lookback_hours` |  |  | req | `integer` |
| `next_cursor` |  |  | opt | `string` (nullable) |
| `q` |  |  | opt | `string` (nullable) |
| `total_count` |  |  | req | `integer` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "cursor": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Cursor"
    },
    "feed_ids": {
      "items": {
        "format": "uuid",
        "type": "string"
      },
      "title": "Feed Ids",
      "type": "array"
    },
    "items": {
      "items": {
        "$ref": "#/components/schemas/PublicTopScoringPostModel"
      },
      "title": "Items",
      "type": "array"
    },
    "limit": {
      "title": "Limit",
      "type": "integer"
    },
    "lookback_hours": {
      "title": "Lookback Hours",
      "type": "integer"
    },
    "next_cursor": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Next Cursor"
    },
    "q": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Q"
    },
    "total_count": {
      "title": "Total Count",
      "type": "integer"
    }
  },
  "required": [
    "feed_ids",
    "lookback_hours",
    "items",
    "total_count",
    "limit"
  ],
  "title": "PublicTopScoringPostsResponseModel",
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

<a id="model-queryexecutionresultdto"></a>
### Model: QueryExecutionResultDTO

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `artifact_id` |  |  | req | `string` (format: uuid) |
| `response` |  |  | req | [Response](#model-response) |
|  | `text` |  | req | `string` |
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
    "response": {
      "$ref": "#/components/schemas/Response"
    },
    "uid": {
      "title": "Uid",
      "type": "integer"
    }
  },
  "required": [
    "artifact_id",
    "uid",
    "response"
  ],
  "title": "QueryExecutionResultDTO",
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

<a id="model-registervalidatorrequest"></a>
### Model: RegisterValidatorRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `base_url` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "base_url": {
      "title": "Base Url",
      "type": "string"
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

<a id="model-scriptartifactmodel"></a>
### Model: ScriptArtifactModel

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
