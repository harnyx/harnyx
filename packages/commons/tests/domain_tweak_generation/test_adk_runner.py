from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from google.genai.errors import ClientError

from harnyx_commons.domain_tweak_generation import adk_runner as runner_module
from harnyx_commons.domain_tweak_generation.adk_runner import (
    DomainTweakAdkRunner,
    DomainTweakAdkTurn,
    _LiveAdkContext,
    _redact_source_tool_content,
)
from harnyx_commons.domain_tweak_generation.types import (
    DomainTweakAdkEventSummary,
    DomainTweakAdkRunConfig,
)
from harnyx_commons.domain_tweak_generation.validation import validate_form_blueprint_output
from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.miner_task_generation import DomainTweakFormBlueprint

pytestmark = pytest.mark.anyio("asyncio")


@dataclass
class _Executor:
    responses: list[str]
    calls: list[dict[str, object]] = field(default_factory=list)

    async def __call__(self, **kwargs: object) -> DomainTweakAdkTurn:
        self.calls.append(kwargs)
        return DomainTweakAdkTurn(final_text=self.responses.pop(0), events=())


async def test_runner_passes_explicit_tools_search_flag_and_native_schema() -> None:
    executor = _Executor([_blueprint_json()])
    runner = DomainTweakAdkRunner(turn_executor=executor)

    async def acquire_sources() -> dict[str, object]:
        return {"status": "completed"}

    result = await runner.run_phase(
        phase="form_blueprint",
        prompt="analyze",
        config=DomainTweakAdkRunConfig(model="gemini-test", max_retries=0),
        agent_instruction="instruction",
        search_enabled=True,
        function_tools=(acquire_sources,),
        output_schema=DomainTweakFormBlueprint,
        validate=validate_form_blueprint_output,
    )

    assert result.terminal_status == "validated"
    assert executor.calls[0]["search_enabled"] is True
    assert executor.calls[0]["function_tool_names"] == ("acquire_sources",)
    assert executor.calls[0]["output_schema"] is DomainTweakFormBlueprint


async def test_runner_retries_only_structural_validation_with_feedback() -> None:
    executor = _Executor(["not-json", _blueprint_json()])
    runner = DomainTweakAdkRunner(turn_executor=executor)

    result = await runner.run_phase(
        phase="form_blueprint",
        prompt="original work order",
        config=DomainTweakAdkRunConfig(model="gemini-test", max_retries=1),
        agent_instruction="instruction",
        search_enabled=False,
        function_tools=(),
        output_schema=DomainTweakFormBlueprint,
        validate=validate_form_blueprint_output,
    )

    assert result.terminal_status == "validated"
    assert [attempt.prompt_kind for attempt in result.attempts] == ["initial", "feedback"]
    assert executor.calls[0]["prompt"] == "original work order"
    assert "failed deterministic schema" in str(executor.calls[1]["prompt"])


async def test_runner_uses_one_hard_stage_timeout_without_soft_timeout_turns() -> None:
    async def blocked_executor(**kwargs: object) -> DomainTweakAdkTurn:
        _ = kwargs
        await asyncio.sleep(60)
        raise AssertionError("unreachable")

    runner = DomainTweakAdkRunner(turn_executor=blocked_executor)
    result = await runner.run_phase(
        phase="form_blueprint",
        prompt="work order",
        config=DomainTweakAdkRunConfig(
            model="gemini-test",
            max_retries=2,
            phase_timeout_seconds=0.01,
        ),
        agent_instruction="instruction",
        search_enabled=False,
        function_tools=(),
        output_schema=DomainTweakFormBlueprint,
        validate=validate_form_blueprint_output,
    )

    assert result.terminal_status == "timeout"
    assert len(result.attempts) == 1
    assert result.attempts[0].prompt_kind == "initial"
    assert DomainTweakAdkRunConfig(model="gemini-test").phase_timeout_seconds == 600.0


async def test_live_failure_before_first_event_returns_typed_invocation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FailingLiveContext()
    monkeypatch.setattr(
        runner_module._LiveAdkContext,
        "create",
        _async_value(context),
    )

    result = await DomainTweakAdkRunner().run_phase(
        phase="form_blueprint",
        prompt="work order",
        config=DomainTweakAdkRunConfig(model="gemini-test", max_retries=0),
        agent_instruction="instruction",
        search_enabled=False,
        function_tools=(),
        output_schema=DomainTweakFormBlueprint,
        validate=validate_form_blueprint_output,
    )

    assert result.terminal_status == "invocation_error"
    assert result.error_type == "RuntimeError"
    assert context.closed


async def test_transient_live_provider_failure_remains_candidate_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FailingLiveContext(
        exc=ClientError(
            429,
            {"error": {"code": 429, "message": "rate limited", "status": "RESOURCE_EXHAUSTED"}},
        )
    )
    monkeypatch.setattr(
        runner_module._LiveAdkContext,
        "create",
        _async_value(context),
    )

    result = await DomainTweakAdkRunner().run_phase(
        phase="form_blueprint",
        prompt="work order",
        config=DomainTweakAdkRunConfig(model="gemini-test", max_retries=0),
        agent_instruction="instruction",
        search_enabled=False,
        function_tools=(),
        output_schema=DomainTweakFormBlueprint,
        validate=validate_form_blueprint_output,
    )

    assert result.terminal_status == "invocation_error"
    assert result.error_type == "ClientError"
    assert context.closed


@pytest.mark.parametrize("status_code", (401, 403))
async def test_live_authentication_failure_aborts_the_batch(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    context = _FailingLiveContext(
        exc=ClientError(
            status_code,
            {
                "error": {
                    "code": status_code,
                    "message": "authentication failed",
                    "status": "UNAUTHENTICATED",
                }
            },
        )
    )
    monkeypatch.setattr(
        runner_module._LiveAdkContext,
        "create",
        _async_value(context),
    )

    with pytest.raises(ClientError) as exc_info:
        await DomainTweakAdkRunner().run_phase(
            phase="form_blueprint",
            prompt="work order",
            config=DomainTweakAdkRunConfig(model="gemini-test", max_retries=0),
            agent_instruction="instruction",
            search_enabled=False,
            function_tools=(),
            output_schema=DomainTweakFormBlueprint,
            validate=validate_form_blueprint_output,
        )

    assert exc_info.value.code == status_code
    assert context.closed


@pytest.mark.parametrize(
    "failure_code",
    (
        ToolProviderFailureCode.CREDENTIAL_UNAVAILABLE,
        ToolProviderFailureCode.AUTHENTICATION_FAILED,
    ),
)
async def test_source_tool_credential_failure_aborts_the_batch(
    monkeypatch: pytest.MonkeyPatch,
    failure_code: ToolProviderFailureCode,
) -> None:
    context = _FailingLiveContext(
        exc=ToolProviderError(
            "Parallel credential rejected",
            provider="parallel",
            failure_code=failure_code,
        )
    )
    monkeypatch.setattr(
        runner_module._LiveAdkContext,
        "create",
        _async_value(context),
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await DomainTweakAdkRunner().run_phase(
            phase="reference_answer_generation",
            prompt="work order",
            config=DomainTweakAdkRunConfig(model="gemini-test", max_retries=0),
            agent_instruction="instruction",
            search_enabled=True,
            function_tools=(),
            output_schema=DomainTweakFormBlueprint,
            validate=validate_form_blueprint_output,
        )

    assert exc_info.value.failure_code is failure_code
    assert context.closed


async def test_live_context_construction_failure_aborts_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_create(**kwargs: object) -> _LiveAdkContext:
        _ = kwargs
        raise ValueError("unsupported ADK agent configuration")

    monkeypatch.setattr(runner_module._LiveAdkContext, "create", failing_create)

    with pytest.raises(ValueError, match="unsupported ADK agent configuration"):
        await DomainTweakAdkRunner().run_phase(
            phase="form_blueprint",
            prompt="work order",
            config=DomainTweakAdkRunConfig(model="gemini-test", max_retries=0),
            agent_instruction="instruction",
            search_enabled=False,
            function_tools=(),
            output_schema=DomainTweakFormBlueprint,
            validate=validate_form_blueprint_output,
        )


async def test_live_context_setup_is_inside_the_hard_stage_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def blocked_create(**kwargs: object) -> _LiveAdkContext:
        _ = kwargs
        await asyncio.sleep(60)
        raise AssertionError("unreachable")

    monkeypatch.setattr(runner_module._LiveAdkContext, "create", blocked_create)

    result = await asyncio.wait_for(
        DomainTweakAdkRunner().run_phase(
            phase="form_blueprint",
            prompt="work order",
            config=DomainTweakAdkRunConfig(
                model="gemini-test",
                max_retries=0,
                phase_timeout_seconds=0.01,
            ),
            agent_instruction="instruction",
            search_enabled=False,
            function_tools=(),
            output_schema=DomainTweakFormBlueprint,
            validate=validate_form_blueprint_output,
        ),
        timeout=0.2,
    )

    assert result.terminal_status == "timeout"
    assert result.attempts == ()
    assert result.error_type == "TimeoutError"


async def test_live_context_cleanup_cannot_extend_the_hard_stage_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _SlowClosingLiveContext()
    monkeypatch.setattr(
        runner_module._LiveAdkContext,
        "create",
        _async_value(context),
    )

    result = await asyncio.wait_for(
        DomainTweakAdkRunner().run_phase(
            phase="form_blueprint",
            prompt="work order",
            config=DomainTweakAdkRunConfig(
                model="gemini-test",
                max_retries=0,
                phase_timeout_seconds=0.01,
            ),
            agent_instruction="instruction",
            search_enabled=False,
            function_tools=(),
            output_schema=DomainTweakFormBlueprint,
            validate=validate_form_blueprint_output,
        ),
        timeout=0.2,
    )

    assert result.terminal_status == "validated"
    assert context.close_cancelled


async def test_live_context_closes_cancelled_event_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iterator = _BlockingIterator()
    context = _LiveAdkContext(
        runner=_FakeLiveRunner(iterator),
        user_id="user",
        session_id="session",
    )
    monkeypatch.setattr(
        runner_module,
        "summarize_adk_event",
        lambda event: DomainTweakAdkEventSummary(content_text_preview=str(event)),
    )

    task = asyncio.create_task(context.run_turn("prompt", event_summaries=[]))
    await iterator.wait_until_blocked()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert iterator.closed


async def test_event_iterator_cleanup_cannot_extend_the_hard_stage_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iterator = _SlowClosingIterator()
    context = _LiveAdkContext(
        runner=_FakeLiveRunner(iterator),
        user_id="user",
        session_id="session",
    )
    monkeypatch.setattr(
        runner_module._LiveAdkContext,
        "create",
        _async_value(context),
    )
    monkeypatch.setattr(
        runner_module,
        "summarize_adk_event",
        lambda event: DomainTweakAdkEventSummary(content_text_preview=str(event)),
    )

    result = await asyncio.wait_for(
        DomainTweakAdkRunner().run_phase(
            phase="form_blueprint",
            prompt="work order",
            config=DomainTweakAdkRunConfig(
                model="gemini-test",
                max_retries=0,
                phase_timeout_seconds=0.01,
            ),
            agent_instruction="instruction",
            search_enabled=False,
            function_tools=(),
            output_schema=DomainTweakFormBlueprint,
            validate=validate_form_blueprint_output,
        ),
        timeout=0.2,
    )

    assert result.terminal_status == "timeout"
    await asyncio.sleep(0)
    assert iterator.close_called
    assert context._runner.close_called


async def test_event_iterator_cleanup_failure_does_not_replace_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iterator = _FailingCloseIterator()
    context = _LiveAdkContext(
        runner=_FakeLiveRunner(iterator),
        user_id="user",
        session_id="session",
    )
    monkeypatch.setattr(
        runner_module._LiveAdkContext,
        "create",
        _async_value(context),
    )
    monkeypatch.setattr(
        runner_module,
        "summarize_adk_event",
        lambda event: DomainTweakAdkEventSummary(content_text_preview=str(event)),
    )

    result = await asyncio.wait_for(
        DomainTweakAdkRunner().run_phase(
            phase="form_blueprint",
            prompt="work order",
            config=DomainTweakAdkRunConfig(
                model="gemini-test",
                max_retries=0,
                phase_timeout_seconds=0.01,
            ),
            agent_instruction="instruction",
            search_enabled=False,
            function_tools=(),
            output_schema=DomainTweakFormBlueprint,
            validate=validate_form_blueprint_output,
        ),
        timeout=0.2,
    )

    assert iterator.close_called
    assert result.terminal_status == "timeout"


async def test_source_tool_event_preview_is_redacted() -> None:
    summary = DomainTweakAdkEventSummary(
        function_response_names=("read_cached_source",),
        content_text_preview="private source body",
        content_text_length=19,
    )

    redacted = _redact_source_tool_content(summary)

    assert redacted.content_text_preview is None
    assert redacted.content_text_length == 19


class _BlockingIterator:
    def __init__(self) -> None:
        self._yielded = False
        self._blocked = asyncio.Event()
        self._release = asyncio.Event()
        self.closed = False

    def __aiter__(self) -> _BlockingIterator:
        return self

    async def __anext__(self) -> object:
        if not self._yielded:
            self._yielded = True
            return "event"
        self._blocked.set()
        await self._release.wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True
        self._release.set()

    async def wait_until_blocked(self) -> None:
        await asyncio.wait_for(self._blocked.wait(), timeout=1)


class _SlowClosingIterator(_BlockingIterator):
    def __init__(self) -> None:
        super().__init__()
        self.close_called = False

    async def aclose(self) -> None:
        self.close_called = True
        await asyncio.sleep(60)


class _FailingCloseIterator(_BlockingIterator):
    def __init__(self) -> None:
        super().__init__()
        self.close_called = False

    async def aclose(self) -> None:
        self.close_called = True
        raise RuntimeError("iterator cleanup failed")


@dataclass
class _FakeLiveRunner:
    iterator: _BlockingIterator
    close_called: bool = False

    def run_async(self, **kwargs: object) -> _BlockingIterator:
        _ = kwargs
        return self.iterator

    async def close(self) -> None:
        self.close_called = True


@dataclass
class _FailingLiveContext:
    exc: Exception = field(
        default_factory=lambda: RuntimeError("provider failed before the first event")
    )
    closed: bool = False

    async def run_turn(
        self,
        prompt: str,
        *,
        event_summaries: list[object],
        deadline: float,
    ) -> DomainTweakAdkTurn:
        _ = prompt, event_summaries, deadline
        raise self.exc

    async def close(self) -> None:
        self.closed = True


class _SlowClosingLiveContext:
    close_cancelled = False

    async def run_turn(
        self,
        prompt: str,
        *,
        event_summaries: list[object],
        deadline: float,
    ) -> DomainTweakAdkTurn:
        _ = prompt, event_summaries, deadline
        return DomainTweakAdkTurn(final_text=_blueprint_json(), events=())

    async def close(self) -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.close_cancelled = True
            raise


def _async_value(value: object):
    async def create(**kwargs: object) -> object:
        _ = kwargs
        return value

    return create


def _blueprint_json() -> str:
    return DomainTweakFormBlueprint(
        status="proceed",
        operation="Filter a closed candidate set.",
        load_bearing_invariants=("closed candidate set",),
        non_load_bearing_surface_features=(),
        retrieval_boundary="Sources supply predicate values.",
        answer_shape="Exhaustive list.",
        semantic_ambiguities=(),
        no_generate_reason=None,
    ).model_dump_json()
