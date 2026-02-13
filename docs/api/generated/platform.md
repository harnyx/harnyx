# Platform API reference (generated)

Generated from FastAPI OpenAPI.

## Domains
- [feeds](#feeds)
  - [POST /v1/feeds/search](#endpoint-post-v1-feeds-search)
- [miner-task-batches](#miner-task-batches)
  - [POST /v1/miner-task-batches/batch](#endpoint-post-v1-miner-task-batches-batch)
  - [GET /v1/miner-task-batches/batch/{batch_id}](#endpoint-get-v1-miner-task-batches-batch-batch_id)
  - [GET /v1/miner-task-batches/progress/{batch_id}](#endpoint-get-v1-miner-task-batches-progress-batch_id)
  - [GET /v1/miner-task-batches/{batch_id}/artifacts/{artifact_id}](#endpoint-get-v1-miner-task-batches-batch_id-artifacts-artifact_id)
- [miners](#miners)
  - [POST /v1/miners/scripts](#endpoint-post-v1-miners-scripts)
- [validators](#validators)
  - [POST /v1/validators/register](#endpoint-post-v1-validators-register)
- [weights](#weights)
  - [GET /v1/weights](#endpoint-get-v1-weights)

## feeds

### search

<a id="endpoint-post-v1-feeds-search"></a>
#### POST /v1/feeds/search

Search for similar feed items strictly prior to a given feed job.

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Request**
Content-Type: `application/json`
Body: [FeedSearchRequestModel](#model-feedsearchrequestmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `enqueue_seq` |  |  | req | `integer` |
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
|  | `enqueue_seq` |  | req | `integer` |
|  | `external_id` |  | req | `string` |
|  | `job_id` |  | req | `string` (format: uuid) |
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



## miner-task-batches

### batch

<a id="endpoint-post-v1-miner-task-batches-batch"></a>
#### POST /v1/miner-task-batches/batch

Create a miner task batch (claims + candidate artifacts).

**Auth**: Bittensor-signed (`Authorization: Bittensor ss58="...",sig="..."`)

**Request**
Content-Type: `application/json`
Body: [CreateBatchRequest](#model-createbatchrequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batch_id` |  |  | opt | `string` (format: uuid; nullable) |
| `champion_uid` |  |  | opt | `integer` (nullable) |
| `override_miner_task_dataset` |  |  | opt | [OverrideMinerTaskDatasetModel](#model-overrideminertaskdatasetmodel) (nullable) |
|  | `claims` |  | req | array[[MinerTaskClaim](#model-minertaskclaim)] |
|  |  | `budget_usd` | opt | `number` (default: 0.05) |
|  |  | `claim_id` | req | `string` (format: uuid) |
|  |  | `context` | opt | [FeedSearchContext](#model-feedsearchcontext) (nullable) |
|  |  | `reference_answer` | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `rubric` | req | [Rubric](#model-rubric) |
|  |  | `text` | req | `string` |
|  | `entrypoint` |  | req | `string` |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: [MinerTaskBatchModel](#model-minertaskbatchmodel)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `candidates` |  |  | req | array[[ScriptArtifactModel](#model-scriptartifactmodel)] |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `content_hash` |  | req | `string` |
|  | `size_bytes` |  | req | `integer` |
|  | `uid` |  | req | `integer` |
| `champion_uid` |  |  | req | `integer` (nullable) |
| `claims` |  |  | req | array[[MinerTaskClaimModel](#model-minertaskclaimmodel)] |
|  | `budget_usd` |  | req | `number` |
|  | `claim_id` |  | req | `string` (format: uuid) |
|  | `context` |  | opt | [FeedSearchContext](#model-feedsearchcontext) (nullable) |
|  |  | `enqueue_seq` | req | `integer` |
|  |  | `feed_id` | req | `string` (format: uuid) |
|  | `reference_answer` |  | req | [ReferenceAnswerModel](#model-referenceanswermodel) |
|  |  | `citations` | opt | array[[CitationModel](#model-citationmodel)] (default: []) |
|  |  | `justification` | req | `string` |
|  |  | `verdict` | req | `integer` |
|  | `rubric` |  | req | [RubricModel](#model-rubricmodel) |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  | `text` |  | req | `string` |
| `created_at` |  |  | req | `string` (format: date-time) |
| `cutoff_at` |  |  | req | `string` (format: date-time) |
| `entrypoint` |  |  | req | `string` |
| `status` |  |  | req | `string` |
| `status_message` |  |  | opt | `string` (nullable) |

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

<a id="endpoint-get-v1-miner-task-batches-batch-batch_id"></a>
##### GET /v1/miner-task-batches/batch/{batch_id}

Fetch a previously created miner task batch.

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
| `batch_id` |  |  | req | `string` (format: uuid) |
| `candidates` |  |  | req | array[[ScriptArtifactModel](#model-scriptartifactmodel)] |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `content_hash` |  | req | `string` |
|  | `size_bytes` |  | req | `integer` |
|  | `uid` |  | req | `integer` |
| `champion_uid` |  |  | req | `integer` (nullable) |
| `claims` |  |  | req | array[[MinerTaskClaimModel](#model-minertaskclaimmodel)] |
|  | `budget_usd` |  | req | `number` |
|  | `claim_id` |  | req | `string` (format: uuid) |
|  | `context` |  | opt | [FeedSearchContext](#model-feedsearchcontext) (nullable) |
|  |  | `enqueue_seq` | req | `integer` |
|  |  | `feed_id` | req | `string` (format: uuid) |
|  | `reference_answer` |  | req | [ReferenceAnswerModel](#model-referenceanswermodel) |
|  |  | `citations` | opt | array[[CitationModel](#model-citationmodel)] (default: []) |
|  |  | `justification` | req | `string` |
|  |  | `verdict` | req | `integer` |
|  | `rubric` |  | req | [RubricModel](#model-rubricmodel) |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  | `text` |  | req | `string` |
| `created_at` |  |  | req | `string` (format: date-time) |
| `cutoff_at` |  |  | req | `string` (format: date-time) |
| `entrypoint` |  |  | req | `string` |
| `status` |  |  | req | `string` |
| `status_message` |  |  | opt | `string` (nullable) |

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
| `batch_id` |  |  | req | `string` (format: uuid) |
| `candidate_count` |  |  | req | `integer` |
| `created_at` |  |  | req | `string` (format: date-time) |
| `cutoff_at` |  |  | req | `string` (format: date-time) |
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

#### artifacts

##### {artifact_id}

<a id="endpoint-get-v1-miner-task-batches-batch_id-artifacts-artifact_id"></a>
###### GET /v1/miner-task-batches/{batch_id}/artifacts/{artifact_id}

Download a stored script artifact for a batch candidate.

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
| `final_top` |  |  | opt | `array` (nullable) |
| `weights` |  |  | req | `object` |



## Models

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
| `batch_id` |  |  | opt | `string` (format: uuid; nullable) |
| `champion_uid` |  |  | opt | `integer` (nullable) |
| `override_miner_task_dataset` |  |  | opt | [OverrideMinerTaskDatasetModel](#model-overrideminertaskdatasetmodel) (nullable) |
|  | `claims` |  | req | array[[MinerTaskClaim](#model-minertaskclaim)] |
|  |  | `budget_usd` | opt | `number` (default: 0.05) |
|  |  | `claim_id` | req | `string` (format: uuid) |
|  |  | `context` | opt | [FeedSearchContext](#model-feedsearchcontext) (nullable) |
|  |  | `reference_answer` | req | [ReferenceAnswer](#model-referenceanswer) |
|  |  | `rubric` | req | [Rubric](#model-rubric) |
|  |  | `text` | req | `string` |
|  | `entrypoint` |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "batch_id": {
      "anyOf": [
        {
          "format": "uuid",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Batch Id"
    },
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
    "override_miner_task_dataset": {
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

<a id="model-feedsearchhitmodel"></a>
### Model: FeedSearchHitModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `content_id` |  |  | req | `string` (format: uuid) |
| `enqueue_seq` |  |  | req | `integer` |
| `external_id` |  |  | req | `string` |
| `job_id` |  |  | req | `string` (format: uuid) |
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
    "enqueue_seq": {
      "title": "Enqueue Seq",
      "type": "integer"
    },
    "external_id": {
      "title": "External Id",
      "type": "string"
    },
    "job_id": {
      "format": "uuid",
      "title": "Job Id",
      "type": "string"
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
| `enqueue_seq` |  |  | req | `integer` |
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
      "minimum": 0.0,
      "title": "Enqueue Seq",
      "type": "integer"
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
    "enqueue_seq",
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
|  | `enqueue_seq` |  | req | `integer` |
|  | `external_id` |  | req | `string` |
|  | `job_id` |  | req | `string` (format: uuid) |
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

<a id="model-minertaskbatchmodel"></a>
### Model: MinerTaskBatchModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `batch_id` |  |  | req | `string` (format: uuid) |
| `candidates` |  |  | req | array[[ScriptArtifactModel](#model-scriptartifactmodel)] |
|  | `artifact_id` |  | req | `string` (format: uuid) |
|  | `content_hash` |  | req | `string` |
|  | `size_bytes` |  | req | `integer` |
|  | `uid` |  | req | `integer` |
| `champion_uid` |  |  | req | `integer` (nullable) |
| `claims` |  |  | req | array[[MinerTaskClaimModel](#model-minertaskclaimmodel)] |
|  | `budget_usd` |  | req | `number` |
|  | `claim_id` |  | req | `string` (format: uuid) |
|  | `context` |  | opt | [FeedSearchContext](#model-feedsearchcontext) (nullable) |
|  |  | `enqueue_seq` | req | `integer` |
|  |  | `feed_id` | req | `string` (format: uuid) |
|  | `reference_answer` |  | req | [ReferenceAnswerModel](#model-referenceanswermodel) |
|  |  | `citations` | opt | array[[CitationModel](#model-citationmodel)] (default: []) |
|  |  | `justification` | req | `string` |
|  |  | `verdict` | req | `integer` |
|  | `rubric` |  | req | [RubricModel](#model-rubricmodel) |
|  |  | `description` | req | `string` |
|  |  | `title` | req | `string` |
|  |  | `verdict_options` | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  | `text` |  | req | `string` |
| `created_at` |  |  | req | `string` (format: date-time) |
| `cutoff_at` |  |  | req | `string` (format: date-time) |
| `entrypoint` |  |  | req | `string` |
| `status` |  |  | req | `string` |
| `status_message` |  |  | opt | `string` (nullable) |

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
        "$ref": "#/components/schemas/ScriptArtifactModel"
      },
      "title": "Candidates",
      "type": "array"
    },
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
    "claims": {
      "items": {
        "$ref": "#/components/schemas/MinerTaskClaimModel"
      },
      "title": "Claims",
      "type": "array"
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
    "entrypoint": {
      "title": "Entrypoint",
      "type": "string"
    },
    "status": {
      "title": "Status",
      "type": "string"
    },
    "status_message": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "title": "Status Message"
    }
  },
  "required": [
    "batch_id",
    "entrypoint",
    "cutoff_at",
    "created_at",
    "claims",
    "candidates",
    "champion_uid",
    "status"
  ],
  "title": "MinerTaskBatchModel",
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

<a id="model-minertaskclaimmodel"></a>
### Model: MinerTaskClaimModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `budget_usd` |  |  | req | `number` |
| `claim_id` |  |  | req | `string` (format: uuid) |
| `context` |  |  | opt | [FeedSearchContext](#model-feedsearchcontext) (nullable) |
|  | `enqueue_seq` |  | req | `integer` |
|  | `feed_id` |  | req | `string` (format: uuid) |
| `reference_answer` |  |  | req | [ReferenceAnswerModel](#model-referenceanswermodel) |
|  | `citations` |  | opt | array[[CitationModel](#model-citationmodel)] (default: []) |
|  |  | `note` | req | `string` |
|  |  | `url` | req | `string` |
|  | `justification` |  | req | `string` |
|  | `verdict` |  | req | `integer` |
| `rubric` |  |  | req | [RubricModel](#model-rubricmodel) |
|  | `description` |  | req | `string` |
|  | `title` |  | req | `string` |
|  | `verdict_options` |  | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  |  | `description` | req | `string` |
|  |  | `value` | req | `integer` |
| `text` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "budget_usd": {
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
      "$ref": "#/components/schemas/ReferenceAnswerModel"
    },
    "rubric": {
      "$ref": "#/components/schemas/RubricModel"
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
    "reference_answer",
    "budget_usd"
  ],
  "title": "MinerTaskClaimModel",
  "type": "object"
}
```

</details>

<a id="model-overrideminertaskdatasetmodel"></a>
### Model: OverrideMinerTaskDatasetModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
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
| `entrypoint` |  |  | req | `string` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "properties": {
    "claims": {
      "items": {
        "$ref": "#/components/schemas/MinerTaskClaim"
      },
      "title": "Claims",
      "type": "array"
    },
    "entrypoint": {
      "minLength": 1,
      "title": "Entrypoint",
      "type": "string"
    }
  },
  "required": [
    "entrypoint",
    "claims"
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
| `batch_id` |  |  | req | `string` (format: uuid) |
| `candidate_count` |  |  | req | `integer` |
| `created_at` |  |  | req | `string` (format: date-time) |
| `cutoff_at` |  |  | req | `string` (format: date-time) |
| `status` |  |  | req | `string` |

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
    "candidate_count": {
      "title": "Candidate Count",
      "type": "integer"
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
    }
  },
  "required": [
    "batch_id",
    "status",
    "created_at",
    "cutoff_at",
    "candidate_count"
  ],
  "title": "ProgressSnapshotResponse",
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

<a id="model-referenceanswermodel"></a>
### Model: ReferenceAnswerModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `citations` |  |  | opt | array[[CitationModel](#model-citationmodel)] (default: []) |
|  | `note` |  | req | `string` |
|  | `url` |  | req | `string` |
| `justification` |  |  | req | `string` |
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
    "verdict": {
      "title": "Verdict",
      "type": "integer"
    }
  },
  "required": [
    "verdict",
    "justification"
  ],
  "title": "ReferenceAnswerModel",
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

<a id="model-rubricmodel"></a>
### Model: RubricModel

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `description` |  |  | req | `string` |
| `title` |  |  | req | `string` |
| `verdict_options` |  |  | req | array[[VerdictOptionModel](#model-verdictoptionmodel)] |
|  | `description` |  | req | `string` |
|  | `value` |  | req | `integer` |

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
      "items": {
        "$ref": "#/components/schemas/VerdictOptionModel"
      },
      "title": "Verdict Options",
      "type": "array"
    }
  },
  "required": [
    "title",
    "description",
    "verdict_options"
  ],
  "title": "RubricModel",
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

<a id="model-weightsresponse"></a>
### Model: WeightsResponse

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `final_top` |  |  | opt | `array` (nullable) |
| `weights` |  |  | req | `object` |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "final_top": {
      "anyOf": [
        {
          "maxItems": 3,
          "minItems": 3,
          "prefixItems": [
            {
              "anyOf": [
                {
                  "type": "integer"
                },
                {
                  "type": "null"
                }
              ]
            },
            {
              "anyOf": [
                {
                  "type": "integer"
                },
                {
                  "type": "null"
                }
              ]
            },
            {
              "anyOf": [
                {
                  "type": "integer"
                },
                {
                  "type": "null"
                }
              ]
            }
          ],
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "title": "Final Top"
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
