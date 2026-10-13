"""OpenAI Agents reference writer; shared model slots and client remain runner-owned."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx2
from agents import Agent, ModelSettings, OpenAIResponsesModel, RunConfig, Runner
from agents.exceptions import ModelBehaviorError, UserError
from agents.items import TResponseInputItem
from agents.run_config import ToolErrorFormatterArgs
from agents.run_error_handlers import RunErrorHandlerInput
from agents.stream_events import RawResponsesStreamEvent
from openai import AsyncOpenAI
from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from openai.types.shared import Reasoning
from pydantic import ValidationError

from harnyx_commons.errors import ToolProviderError
from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.task_ownership import await_owned_task

from .agent_runner import ACTIVE_CAPTURE, CallCapture, OpenAIStreamError, openai_stream
from .contracts import CandidateStageError, RegisteredReferenceOutput
from .prompts import REFERENCE_SYSTEM
from .reference_evidence import ReferenceSources

REFERENCE_MODEL = "gpt-6-luna"
MAX_FEEDBACK_RETRIES = 5
logger = logging.getLogger("harnyx_commons.llm.calls")


class ReferenceFeedback:
    """Allow five consecutive correction retries, resetting after a successful action."""

    def __init__(self, capture: CallCapture) -> None:
        self.capture = capture
        self.retries = 0

    def error(self, operation: str, error: str) -> str:
        exhausted = self.retries >= MAX_FEEDBACK_RETRIES
        if not exhausted:
            self.retries += 1
        event = {
            "type": "reference.model_feedback",
            "operation": operation,
            "error": error,
            "retry": self.retries,
            "max_retries": MAX_FEEDBACK_RETRIES,
            "exhausted": exhausted,
        }
        self.capture.event(event)
        logger.warning(
            "task_generation.reference_model_feedback",
            extra={"data": self.capture.identity() | {key: value for key, value in event.items() if key != "error"}},
        )
        if exhausted:
            raise CandidateStageError(
                "contract_invalid", "reference", f"Five model correction retries exhausted: {error}"
            )
        return json.dumps(
            event
            | {
                "instruction": (
                    "Correct the reported mistake using the same retained sources and registered evidence, "
                    "then try again. Do not repeat the invalid action unchanged."
                ),
            },
            ensure_ascii=False,
        )

    def success(self, operation: str) -> None:
        if self.retries:
            self.capture.event(
                {"type": "reference.model_feedback_reset", "operation": operation, "previous_retries": self.retries}
            )
            self.retries = 0


class _ReferenceResponseStream(httpx2.AsyncByteStream):
    """HTTPX marks a response closed before its transport close completes; retain that task."""

    def __init__(self, stream: httpx2.AsyncByteStream) -> None:
        self.stream = stream
        self.closing: asyncio.Task[None] | None = None

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self.stream:
            yield chunk

    async def aclose(self) -> None:
        if self.closing is None:
            self.closing = asyncio.create_task(self.stream.aclose())
        await await_owned_task(self.closing)


async def own_reference_response(response: httpx2.Response) -> None:
    capture = ACTIVE_CAPTURE.get()
    if capture is not None and capture.physical_accounting and response.request.url.path.endswith("/responses"):
        if not isinstance(response.stream, httpx2.AsyncByteStream):
            raise TypeError("Reference provider response requires an asynchronous byte stream")
        response.stream = _ReferenceResponseStream(response.stream)


class _ReferenceResponsesModel(OpenAIResponsesModel):
    """Join the pinned SDK's detached provider closes before releasing our slot."""

    def __init__(self, client: AsyncOpenAI) -> None:
        super().__init__(model=REFERENCE_MODEL, openai_client=client)
        self._stream_closes: list[asyncio.Future[None]] = []

    def _detach_stream_close(self, close_task: asyncio.Future[None]) -> None:
        self._stream_closes.append(close_task)
        super()._detach_stream_close(close_task)

    @asynccontextmanager
    async def owned_streams(self) -> AsyncIterator[None]:
        try:
            yield
        finally:
            failed = sys.exc_info()[0] is not None

            async def drain() -> None:
                results = await asyncio.gather(*self._stream_closes, return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException):
                        raise result

            try:
                await await_owned_task(asyncio.create_task(drain()))
            except Exception:
                if not failed:
                    raise
                logger.debug("task_generation.reference_stream_cleanup.failure.details", exc_info=True)


class ReferenceAgentRunner:
    def __init__(
        self,
        client: AsyncOpenAI,
        slots: asyncio.Semaphore,
        *,
        validate_output: Callable[[RegisteredReferenceOutput], object] | None = None,
    ) -> None:
        self.client, self.slots = client, slots
        self.validate_output = validate_output

    async def run(self, inputs: str, sources: ReferenceSources, capture: CallCapture) -> RegisteredReferenceOutput:
        feedback = ReferenceFeedback(capture)

        failed_output: RunErrorHandlerInput[Any] | None = None

        def invalid_output(data: RunErrorHandlerInput[Any]) -> None:
            nonlocal failed_output
            # Empty output makes the SDK run another turn instead of raising. Stop after
            # this response so every rejected final passes through our correction budget.
            failed_output = data
            result.cancel(mode="after_turn")

        def missing_tool(data: ToolErrorFormatterArgs[Any]) -> str | None:
            if data.kind == "tool_not_found":
                return feedback.error("unknown_tool", data.default_message)
            return None

        model = _ReferenceResponsesModel(self.client)
        agent = Agent(
            name="reference",
            instructions=REFERENCE_SYSTEM,
            model=model,
            tools=[*sources.tools(feedback)],
            output_type=RegisteredReferenceOutput,
            model_settings=ModelSettings(
                reasoning=Reasoning(effort="high", summary="detailed"),
                truncation="disabled",
                context_management=[{"type": "compaction", "compact_threshold": 200_000}],
            ),
        )
        conversation: str | list[TResponseInputItem] = inputs
        async with self.slots, model.owned_streams():
            while True:
                failed_output = None
                result = Runner.run_streamed(
                    agent,
                    conversation,
                    max_turns=None,
                    error_handlers={"invalid_final_output": invalid_output},
                    run_config=RunConfig(
                        tracing_disabled=True,
                        tool_not_found_behavior="return_error_to_model",
                        tool_error_formatter=missing_tool,
                    ),
                )
                try:
                    async with openai_stream(result) as events:
                        async for event in events:
                            if not isinstance(event, RawResponsesStreamEvent):
                                continue
                            data: dict[str, Any] = event.data.model_dump(mode="json", exclude_none=True)
                            capture.event(data)
                            if data["type"] in {"response.completed", "response.incomplete", "response.failed"}:
                                response = data["response"]
                                raw = response.get("usage")
                                if raw is not None:
                                    capture.account_response(
                                        response["id"],
                                        LlmUsage(
                                            prompt_tokens=raw["input_tokens"],
                                            completion_tokens=raw["output_tokens"],
                                            total_tokens=raw["total_tokens"],
                                            prompt_cached_tokens=raw.get("input_tokens_details", {}).get(
                                                "cached_tokens"
                                            ),
                                            prompt_cache_write_tokens=raw.get("input_tokens_details", {}).get(
                                                "cache_write_tokens"
                                            ),
                                            reasoning_tokens=raw.get("output_tokens_details", {}).get(
                                                "reasoning_tokens"
                                            ),
                                            web_search_calls=0,  # only explicit Parallel tools are available
                                        ),
                                    )
                            if data["type"] in {"response.incomplete", "response.failed", "error"}:
                                error = data.get("response", {}).get("error") or data
                                raise OpenAIStreamError(error.get("code"))
                except UserError as exc:
                    if isinstance(exc.__cause__, (ToolProviderError, CandidateStageError)):
                        raise exc.__cause__ from exc.__cause__.__cause__
                    raise
                except ModelBehaviorError as exc:
                    if failed_output is None:
                        raise CandidateStageError("contract_invalid", "reference", str(exc)) from exc
                if failed_output is not None:
                    # SDK diagnostic messages may be redacted. Validate only the public writer
                    # schema to obtain field errors without echoing its input or private state.
                    text = "".join(
                        part.text
                        for response in failed_output.run_data.raw_responses
                        for item in response.output
                        if isinstance(item, ResponseOutputMessage)
                        for part in item.content
                        if isinstance(part, ResponseOutputText)
                    )
                    reason = "Return a complete reference matching the writer output schema"
                    try:
                        RegisteredReferenceOutput.model_validate_json(text, strict=True)
                    except ValidationError as invalid:
                        reason = json.dumps(
                            invalid.errors(include_input=False, include_url=False, include_context=False)
                        )
                    message = feedback.error("final_output", reason)
                    conversation = [*failed_output.run_data.history, {"role": "user", "content": message}]
                    continue
                output = result.final_output
                if not isinstance(output, RegisteredReferenceOutput):
                    raise TypeError("Reference SDK returned an unexpected output type")
                try:
                    if self.validate_output is not None:
                        self.validate_output(output)
                except (ValueError, RecursionError) as exc:
                    message = feedback.error("final_output", str(exc))
                    conversation = [*result.to_input_list(mode="normalized"), {"role": "user", "content": message}]
                    continue
                feedback.success("final_output")
                return output
