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
| `context` |  |  | opt | `object` |
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

<a id="model-entrypointrequest"></a>
### Model: EntrypointRequest

| 1st level | 2nd level | 3rd level | Req | Notes |
| --- | --- | --- | --- | --- |
| `context` |  |  | opt | `object` |
| `payload` |  |  | opt | `object` |
| `tool_config` |  |  | opt | `object` (nullable) |

<details>
<summary>JSON schema</summary>

```json
{
  "properties": {
    "context": {
      "additionalProperties": true,
      "title": "Context",
      "type": "object"
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
  "title": "EntrypointRequest",
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
