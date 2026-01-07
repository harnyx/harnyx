"""LLM adapter backed by the OpenAI Responses API."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from openai import APIError, APIStatusError, AsyncOpenAI
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from pydantic import BaseModel

from caster_commons.llm.provider import BaseLlmProvider
from caster_commons.llm.schema import (
    AbstractLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmInputImageData,
    LlmInputImagePart,
    LlmInputTextPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmResponse,
    LlmTool,
    LlmUsage,
)


class OpenAILlmProvider(BaseLlmProvider):
    """Calls the OpenAI Responses API and normalises results."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
        client: AsyncOpenAI | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key must be provided")
        super().__init__(provider_label="openai", max_concurrent=max_concurrent)
        self._api_key = api_key
        self._owns_client = client is None
        self._client: AsyncOpenAI = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            max_retries=0,  # we handle retries in _call_with_retry
        )
        self._logger = logging.getLogger("caster_commons.llm.calls")

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        payload = self._build_payload(request)
        return await self._call_with_retry(
            request,
            call_coro=lambda: self._request_openai(payload, timeout_seconds=request.timeout_seconds),
            verifier=self._verify_response,
            classify_exception=self._classify_exception,
        )

    async def _request_openai(self, payload: dict[str, Any], *, timeout_seconds: float | None) -> LlmResponse:
        client = (
            self._client.with_options(timeout=timeout_seconds)
            if timeout_seconds is not None
            else self._client
        )
        response_obj: Response = await client.responses.create(**payload)
        response = self._response_from_object(response_obj)
        metadata = dict(response.metadata or {})
        metadata.setdefault("raw_response", response_obj)
        return LlmResponse(
            id=response.id,
            choices=response.choices,
            usage=response.usage,
            metadata=metadata,
            finish_reason=response.finish_reason,
        )

    def _response_from_object(self, obj: Response) -> LlmResponse:
        output_items = tuple(obj.output or ())
        text = obj.output_text or _collect_output_text(output_items)
        tool_calls = _extract_message_tool_calls(output_items)
        web_calls = sum(
            1 for item in output_items if isinstance(item, ResponseFunctionWebSearch)
        )
        usage = _extract_usage(obj.usage) or LlmUsage()
        if web_calls:
            usage = usage + LlmUsage(web_search_calls=web_calls)
        return LlmResponse(
            id=obj.id,
            choices=_build_single_choice(text, tool_calls),
            usage=usage,
            finish_reason=self._finish_reason(obj),
            metadata={"web_search_calls": web_calls} if web_calls else None,
        )

    @staticmethod
    def _finish_reason(obj: Response) -> str:
        details = obj.incomplete_details
        if details and details.reason:
            return details.reason
        return "stop"

    @staticmethod
    def _verify_response(resp: LlmResponse) -> tuple[bool, bool, str | None]:
        if resp.finish_reason != "stop":
            return False, False, resp.finish_reason
        if not resp.choices:
            return False, True, "empty_choices"
        if not resp.raw_text:
            return False, True, "empty_output"
        return True, False, None

    @staticmethod
    def _classify_exception(
        exc: Exception,
        classify_exception: Callable[[Exception], tuple[bool, str]] | None = None,
    ) -> tuple[bool, str]:
        if isinstance(exc, APIStatusError):
            status = exc.status_code
            retryable = status in (408, 409, 429) or (status is not None and status >= 500)
            request_id = exc.response.headers.get("x-request-id") if exc.response else None
            return retryable, f"http_{status}_req_{request_id}"
        if isinstance(exc, APIError):
            return True, exc.__class__.__name__
        return False, str(exc)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    def _build_payload(self, request: AbstractLlmRequest) -> dict[str, Any]:
        if request.grounded:
            if request.output_mode != "text" or request.tools:
                raise ValueError("grounded OpenAI requests must use text output and do not support explicit tools")

        payload: dict[str, Any] = {
            "model": request.model,
            "input": [_serialize_message(message) for message in request.messages],
        }

        include = list(request.include) if request.include else None
        tools = None
        tool_choice: str | None = None
        if not request.grounded:
            if request.output_mode == "json_object":
                payload["text"] = {"format": {"type": "json_object"}}
            elif request.output_mode == "structured":
                if request.output_schema is None:
                    raise ValueError("structured output requires output_schema")
                schema = _json_schema_from_model(request.output_schema)
                payload["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": request.output_schema.__name__ or "response",
                        "schema": schema,
                        "strict": True,
                    }
                }
            tools = [_serialize_tool(tool) for tool in request.tools] if request.tools else None
            tool_choice = request.tool_choice
        else:
            tools = [{"type": "web_search"}]
            tool_choice = "auto"
            if include is None:
                include = ["web_search_call.action.sources"]

        optional_fields = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
            "include": include,
            "reasoning": {"effort": request.reasoning_effort}
            if request.reasoning_effort is not None
            else None,
        }
        payload |= {k: v for k, v in optional_fields.items() if v is not None}

        return payload


def _json_schema_from_model(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    if not isinstance(schema, Mapping):  # pragma: no cover - pydantic guarantees mapping
        raise TypeError("output_schema must produce a JSON object")
    # The Responses API expects a plain JSON schema object.
    return {
        **dict(schema),
        "additionalProperties": False,
    }


def _serialize_message(message: LlmMessage) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for part in message.content:
        match part:
            case LlmInputTextPart(text=text):
                content.append({"type": "input_text", "text": text})
            case LlmInputImagePart(data=LlmInputImageData(url=url)):
                content.append({"type": "input_image", "image_url": url})
            case _:
                raise ValueError(f"unsupported OpenAI request content part type: {part!r}")

    return {"role": message.role, "content": content}


def _serialize_tool(tool: LlmTool) -> dict[str, Any]:
    if tool.type == "function":
        if tool.function is None:
            raise ValueError("function tool requires 'function' metadata")
        return {
            "type": "function",
            "function": dict(tool.function),
        }
    payload: dict[str, Any] = {"type": tool.type}
    if tool.config:
        payload.update(dict(tool.config))
    return payload


def _collect_output_text(output_items: Sequence[ResponseOutputItem]) -> str:
    chunks: list[str] = []
    for item in output_items:
        if not isinstance(item, ResponseOutputMessage):
            continue
        for part in item.content:
            if isinstance(part, ResponseOutputText):
                chunks.append(part.text)
    return "".join(chunks)


def _extract_message_tool_calls(output_items: Sequence[ResponseOutputItem]) -> list[LlmMessageToolCall]:
    calls: list[LlmMessageToolCall] = []
    for item in output_items:
        if not isinstance(item, ResponseFunctionToolCall):
            continue
        call_id = item.id or f"toolcall-{len(calls)}"
        calls.append(
            LlmMessageToolCall(
                id=str(call_id),
                type=item.type,
                name=item.name,
                arguments=item.arguments,
            )
        )
    return calls


def _build_single_choice(
    text: str,
    tool_calls: Sequence[LlmMessageToolCall],
) -> tuple[LlmChoice, ...]:
    return (
        LlmChoice(
            index=0,
            message=LlmChoiceMessage(
                role="assistant",
                content=(LlmMessageContentPart(type="text", text=text or ""),),
                tool_calls=tuple(tool_calls) if tool_calls else None,
            ),
            finish_reason="stop",
        ),
    )


def _extract_usage(body: ResponseUsage | None) -> LlmUsage | None:
    if body is None:
        return None

    prompt_tokens = body.input_tokens
    completion_tokens = body.output_tokens
    total_tokens = body.total_tokens
    prompt_cached_tokens = body.input_tokens_details.cached_tokens
    reasoning_tokens = body.output_tokens_details.reasoning_tokens

    return LlmUsage(
        prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
        completion_tokens=int(completion_tokens) if completion_tokens is not None else None,
        total_tokens=int(total_tokens) if total_tokens is not None else None,
        prompt_cached_tokens=prompt_cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )

__all__ = ["OpenAILlmProvider"]
