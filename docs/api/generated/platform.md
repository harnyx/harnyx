# Platform API reference (generated)

Generated from FastAPI OpenAPI.

## Domains
- [feeds](#feeds)
  - [POST /v1/feeds/search](#endpoint-post-v1-feeds-search)
  - [POST /v1/feeds/{feed_id}/tool/search](#endpoint-post-v1-feeds-feed_id-tool-search)
- [miner-task-batches](#miner-task-batches)
  - [POST /v1/miner-task-batches/batch](#endpoint-post-v1-miner-task-batches-batch)
  - [GET /v1/miner-task-batches/batch/{batch_id}](#endpoint-get-v1-miner-task-batches-batch-batch_id)
  - [GET /v1/miner-task-batches/progress/{batch_id}](#endpoint-get-v1-miner-task-batches-progress-batch_id)
  - [GET /v1/miner-task-batches/{batch_id}/artifacts/{artifact_id}](#endpoint-get-v1-miner-task-batches-batch_id-artifacts-artifact_id)
- [miners](#miners)
  - [POST /v1/miners/scripts](#endpoint-post-v1-miners-scripts)
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
|  |  | `budget_usd` | opt | `number` (default: 0.05) |
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
|  | `budget_usd` |  | opt | `number` (default: 0.05) |
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
|  | `budget_usd` |  | opt | `number` (default: 0.05) |
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

<a id="model-createbatchrequest"></a>
### Model: CreateBatchRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `champion_artifact_id` |  |  | opt | `string` (format: uuid; nullable) |
| `override_task_dataset` |  |  | opt | [OverrideMinerTaskDatasetModel](#model-overrideminertaskdatasetmodel) (nullable) |
|  | `tasks` |  | req | array[[MinerTaskInputModel](#model-minertaskinputmodel)] |
|  |  | `budget_usd` | opt | `number` (default: 0.05) |
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
  "description": "Query parameters for the `get_repo_file` tool.",
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
  "description": "Response payload for the `get_repo_file` tool.",
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

<a id="model-minertask"></a>
### Model: MinerTask

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `budget_usd` |  |  | opt | `number` (default: 0.05) |
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
      "default": 0.05,
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
|  | `budget_usd` |  | opt | `number` (default: 0.05) |
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
| `budget_usd` |  |  | opt | `number` (default: 0.05) |
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
      "default": 0.05,
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

<a id="model-overrideminertaskdatasetmodel"></a>
### Model: OverrideMinerTaskDatasetModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `tasks` |  |  | req | array[[MinerTaskInputModel](#model-minertaskinputmodel)] |
|  | `budget_usd` |  | opt | `number` (default: 0.05) |
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
  "description": "Query parameters for the `search_repo` tool.",
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
  "description": "Response payload for the `search_repo` tool.",
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
