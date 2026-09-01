# Sandbox API reference (generated)

Generated from FastAPI OpenAPI.

## Domains
- [{entrypoint_name}](#entrypoint_name)
  - [POST /entry/{entrypoint_name}](#endpoint-post-entry-entrypoint_name)
- [Misc](#misc)
  - [GET /healthz](#endpoint-get-healthz)

## {entrypoint_name}

<a id="endpoint-post-entry-entrypoint_name"></a>
### POST /entry/{entrypoint_name}

Invoke a registered entrypoint by name in a sandboxed worker process.

**Auth**: Tool token (`x-platform-token` header)

**Parameters**
| Param | In | Req | Notes |
| --- | --- | --- | --- |
| `entrypoint_name` | path | req | `string` |

**Request**
Content-Type: `application/json`
Body: [EntrypointRequest](#model-entrypointrequest)

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `context` |  |  | req | [_EntrypointContext](#model-_entrypointcontext) |
|  | `cost_budget` |  | opt | [ToolBudgetDTO](#model-toolbudgetdto) (nullable) |
|  |  | `session_budget_usd` | req | `number` |
|  |  | `session_hard_limit_usd` | req | `number` |
|  |  | `session_remaining_budget_usd` | req | `number` |
|  |  | `session_used_budget_usd` | req | `number` |
|  | `time_budget` |  | req | [ExecutionTimeBudgetDTO](#model-executiontimebudgetdto) |
|  |  | `limit_seconds` | req | `number` |
| `payload` |  |  | opt | `object` |
| `tool_config` |  |  | opt | `object` (nullable) |

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: `object`

(no documented fields)

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

Sandbox health check.

**Auth**: None.

**Responses**
`200` Successful Response
Content-Type: `application/json`
Body: `object`

(no documented fields)



## Models

<a id="model-_entrypointcontext"></a>
### Model: _EntrypointContext

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `cost_budget` |  |  | opt | [ToolBudgetDTO](#model-toolbudgetdto) (nullable) |
|  | `session_budget_usd` |  | req | `number` |
|  | `session_hard_limit_usd` |  | req | `number` |
|  | `session_remaining_budget_usd` |  | req | `number` |
|  | `session_used_budget_usd` |  | req | `number` |
| `time_budget` |  |  | req | [ExecutionTimeBudgetDTO](#model-executiontimebudgetdto) |
|  | `limit_seconds` |  | req | `number` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "description": "Validated sandbox-boundary context for every entrypoint invocation.",
  "properties": {
    "cost_budget": {
      "anyOf": [
        {
          "$ref": "#/components/schemas/ToolBudgetDTO"
        },
        {
          "type": "null"
        }
      ]
    },
    "time_budget": {
      "$ref": "#/components/schemas/ExecutionTimeBudgetDTO"
    }
  },
  "required": [
    "time_budget"
  ],
  "title": "_EntrypointContext",
  "type": "object"
}
```

</details>

<a id="model-entrypointrequest"></a>
### Model: EntrypointRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `context` |  |  | req | [_EntrypointContext](#model-_entrypointcontext) |
|  | `cost_budget` |  | opt | [ToolBudgetDTO](#model-toolbudgetdto) (nullable) |
|  |  | `session_budget_usd` | req | `number` |
|  |  | `session_hard_limit_usd` | req | `number` |
|  |  | `session_remaining_budget_usd` | req | `number` |
|  |  | `session_used_budget_usd` | req | `number` |
|  | `time_budget` |  | req | [ExecutionTimeBudgetDTO](#model-executiontimebudgetdto) |
|  |  | `limit_seconds` | req | `number` |
| `payload` |  |  | opt | `object` |
| `tool_config` |  |  | opt | `object` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "context": {
      "$ref": "#/components/schemas/_EntrypointContext"
    },
    "payload": {
      "additionalProperties": true,
      "title": "Payload",
      "type": "object"
    },
    "tool_config": {
      "anyOf": [
        {
          "additionalProperties": true,
          "type": "object"
        },
        {
          "type": "null"
        }
      ],
      "title": "Tool Config"
    }
  },
  "required": [
    "context"
  ],
  "title": "EntrypointRequest",
  "type": "object"
}
```

</details>

<a id="model-executiontimebudgetdto"></a>
### Model: ExecutionTimeBudgetDTO

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `limit_seconds` |  |  | req | `number` |

<details>
<summary>JSON schema</summary>

```json
{
  "additionalProperties": false,
  "description": "Configured full time limit for one miner invocation.",
  "properties": {
    "limit_seconds": {
      "exclusiveMinimum": 0.0,
      "title": "Limit Seconds",
      "type": "number"
    }
  },
  "required": [
    "limit_seconds"
  ],
  "title": "ExecutionTimeBudgetDTO",
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
