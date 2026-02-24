"""LLM adapter backed by the Chutes HTTP API."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any, cast

import httpx

from caster_commons.llm.provider import BaseLlmProvider
from caster_commons.llm.schema import (
    AbstractLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmInputImagePart,
    LlmInputTextPart,
    LlmInputToolResultPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmResponse,
    LlmTool,
    LlmUsage,
)

logger = logging.getLogger(__name__)


class ChutesLlmProvider(BaseLlmProvider):
    """Wraps the Chutes chat completions endpoint as an LLM provider."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
        auth_header: str = "Authorization",
        max_concurrent: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Chutes API key must be provided for LLM usage")
        super().__init__(provider_label="chutes", max_concurrent=max_concurrent)
        normalized_base = base_url.rstrip("/")
        self._owns_client = client is None
        self._client: httpx.AsyncClient = client or httpx.AsyncClient(
            base_url=normalized_base,
            timeout=timeout,
        )
        self._api_key = api_key
        self._auth_header = auth_header

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        payload = self._build_payload(request)
        headers = self._auth_headers()

        return await self._call_with_retry(
            request,
            call_coro=lambda: self._request_chutes(payload, headers, timeout_seconds=request.timeout_seconds),
            verifier=self._verify_response,
            classify_exception=self._classify_exception,
        )

    def _build_payload(self, request: AbstractLlmRequest) -> dict[str, Any]:
        if request.grounded:
            raise ValueError("grounded mode is not supported for chutes provider")
        if request.output_mode != "text":
            raise ValueError("chutes provider supports text output only")
        payload: dict[str, Any] = {
            "provider": request.provider or "chutes",
            "model": request.model,
            "messages": [_serialize_message(message) for message in request.messages],
        }

        optional_fields = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
            "tools": [_serialize_tool(spec) for spec in request.tools] if request.tools else None,
            "tool_choice": request.tool_choice,
            "include": list(request.include) if request.include else None,
        }
        payload |= {k: v for k, v in optional_fields.items() if v is not None}
        if request.extra:
            payload.update(dict(request.extra))
        return payload

    def _auth_headers(self) -> dict[str, str]:
        return {self._auth_header: f"Bearer {self._api_key}"}

    async def _request_chutes(
        self,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        *,
        timeout_seconds: float | None,
    ) -> LlmResponse:
        request_kwargs: dict[str, Any] = {
            "json": payload,
            "headers": headers,
        }
        if timeout_seconds is not None:
            request_kwargs["timeout"] = timeout_seconds
        response = await self._client.post("v1/chat/completions", **request_kwargs)
        response.raise_for_status()
        body = await self._parse_body(response)
        llm_response = self._payload_to_response(body)
        metadata = dict(llm_response.metadata or {})
        metadata.setdefault("raw_response", body)
        return LlmResponse(
            id=llm_response.id,
            choices=llm_response.choices,
            usage=llm_response.usage,
            metadata=metadata,
            finish_reason=llm_response.finish_reason,
        )

    async def _parse_body(self, response: httpx.Response) -> Mapping[str, Any]:
        try:
            return cast(Mapping[str, Any], response.json())
        except ValueError as exc:  # pragma: no cover - network dependent
            raise RuntimeError("chutes chat completions returned non-JSON payload") from exc

    @staticmethod
    def _payload_to_response(payload: Mapping[str, Any]) -> LlmResponse:
        choices = _build_choices(payload)
        usage = _extract_usage(payload) or LlmUsage()
        response_id = _as_str(payload.get("id"))
        return LlmResponse(
            id=response_id,
            choices=choices,
            usage=usage,
        )

    @staticmethod
    def _verify_response(resp: LlmResponse) -> tuple[bool, bool, str | None]:
        if not resp.choices:
            return False, True, "empty_choices"
        if not resp.raw_text and not resp.tool_calls:
            return False, True, "empty_output"
        for call in _iter_tool_calls(resp):
            if not _is_valid_json(call.arguments):
                return False, True, "tool_call_args_invalid_json"
        return True, False, None

    @staticmethod
    def _classify_exception(
        exc: Exception,
        classify_exception: Callable[[Exception], tuple[bool, str]] | None = None,
    ) -> tuple[bool, str]:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code if exc.response else None
            retryable = status is not None and (status == 429 or status >= 500)
            return retryable, f"http_{status}"
        if isinstance(exc, httpx.HTTPError):
            return True, exc.__class__.__name__
        if classify_exception is not None:
            return classify_exception(exc)
        return False, str(exc)

    async def aclose(self) -> None:
        """Close the underlying HTTP client when this provider owns it."""
        if self._owns_client:
            await self._client.aclose()


def _serialize_tool(spec: LlmTool) -> dict[str, Any]:
    if spec.type == "function" and spec.function is not None:
        return {
            "type": "function",
            "function": dict(spec.function),
        }
    tool_payload: dict[str, Any] = {"type": spec.type}
    if spec.config:
        tool_payload.update(dict(spec.config))
    return tool_payload


def _serialize_message(message: LlmMessage) -> dict[str, Any]:
    fragments: list[str] = []
    tool_results: list[LlmInputToolResultPart] = []
    for part in message.content:
        match part:
            case LlmInputTextPart(text=text):
                fragments.append(text)
            case LlmInputImagePart():
                raise ValueError("chutes provider does not support image content parts")
            case LlmInputToolResultPart() as tool_result:
                tool_results.append(tool_result)
            case _:
                raise ValueError(f"unsupported Chutes request content part type: {part!r}")

    if tool_results:
        if fragments:
            raise ValueError("chutes input_tool_result messages cannot mix text parts")
        if len(tool_results) != 1:
            raise ValueError("chutes input_tool_result messages must include exactly one part")
        tool_result = tool_results[0]
        return {
            "role": "tool",
            "tool_call_id": tool_result.tool_call_id,
            "name": tool_result.name,
            "content": tool_result.output_json,
        }

    return {
        "role": message.role,
        "content": "\n".join(fragments),
    }


def _build_choices(payload: Mapping[str, Any]) -> tuple[LlmChoice, ...]:
    choices_payload = payload.get("choices")
    if not isinstance(choices_payload, list):
        return ()

    choices: list[LlmChoice] = []
    for idx, choice_payload in enumerate(choices_payload):
        choice = _choice_from_payload(choice_payload, index=idx)
        if choice is not None:
            choices.append(choice)
    return tuple(choices)


def _choice_from_payload(choice_payload: Any, *, index: int) -> LlmChoice | None:
    message = _message_payload(choice_payload)
    if message is None:
        return None

    parts = _message_parts(message)
    tool_calls = _message_tool_calls(message)
    reasoning = _message_reasoning(message)
    if not parts:
        parts = (LlmMessageContentPart(type="text", text=""),)

    return LlmChoice(
        index=index,
        message=LlmChoiceMessage(
            role="assistant",
            content=parts,
            tool_calls=tool_calls,
            reasoning=reasoning,
        ),
        finish_reason="stop",
    )


def _message_payload(choice_payload: Any) -> Mapping[str, Any] | None:
    if not isinstance(choice_payload, Mapping):
        return None
    message = choice_payload.get("message")
    return message if isinstance(message, Mapping) else None


def _message_parts(message: Mapping[str, Any]) -> tuple[LlmMessageContentPart, ...]:
    content_value = message.get("content")
    if isinstance(content_value, str):
        return (LlmMessageContentPart(type="text", text=content_value),)
    if isinstance(content_value, list):
        parts: list[LlmMessageContentPart] = []
        for part in content_value:
            if not isinstance(part, Mapping):
                continue
            text_value = part.get("text")
            if isinstance(text_value, str):
                parts.append(
                    LlmMessageContentPart(
                        type=str(part.get("type", "text")),
                        text=text_value,
                    ),
                )
        return tuple(parts)
    return ()


def _message_tool_calls(message: Mapping[str, Any]) -> tuple[LlmMessageToolCall, ...]:
    tool_calls_value = message.get("tool_calls")
    if not isinstance(tool_calls_value, list):
        return ()

    calls: list[LlmMessageToolCall] = []
    for call_payload in tool_calls_value:
        call = _tool_call_from_payload(call_payload, len(calls))
        if call is not None:
            calls.append(call)
    return tuple(calls)


def _message_reasoning(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    reasoning_value = message.get("reasoning")
    if isinstance(reasoning_value, Mapping):
        return dict(reasoning_value)
    return None


def _tool_call_from_payload(payload: Any, index: int) -> LlmMessageToolCall | None:
    if not isinstance(payload, Mapping):
        return None
    function = payload.get("function")
    if not isinstance(function, Mapping):
        return None

    name = function.get("name")
    arguments = function.get("arguments")
    if isinstance(arguments, Mapping):
        arguments = json.dumps(arguments)
    if not isinstance(name, str) or not isinstance(arguments, str):
        return None

    return LlmMessageToolCall(
        id=str(payload.get("id", f"toolcall-{index}")),
        type=str(payload.get("type", "function")),
        name=name,
        arguments=arguments,
    )


def _extract_usage(payload: Mapping[str, Any]) -> LlmUsage | None:
    usage_payload = payload.get("usage")
    if not isinstance(usage_payload, Mapping):
        return None
    prompt_tokens = usage_payload.get("prompt_tokens")
    completion_tokens = usage_payload.get("completion_tokens")
    total_tokens = usage_payload.get("total_tokens")
    return LlmUsage(
        prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
        completion_tokens=int(completion_tokens) if completion_tokens is not None else None,
        total_tokens=int(total_tokens) if total_tokens is not None else None,
    )


def _summarize_response(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        data = response.text
    if isinstance(data, Mapping) and "detail" in data:
        data = data["detail"]
    text = str(data)
    return text if len(text) <= 500 else text[:500] + "…"


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


def _iter_tool_calls(response: LlmResponse) -> tuple[LlmMessageToolCall, ...]:
    calls: list[LlmMessageToolCall] = []
    for choice in response.choices:
        calls.extend(choice.message.tool_calls or ())
    return tuple(calls)


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


__all__ = ["ChutesLlmProvider"]
