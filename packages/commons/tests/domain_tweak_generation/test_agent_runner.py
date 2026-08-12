from contextlib import nullcontext
from typing import Any, cast, get_args

import pytest
from claude_agent_sdk import CLIConnectionError, CLINotFoundError, ProcessError, ResultMessage
from pydantic import BaseModel, Field

from harnyx_commons.domain.session import LlmUsageTotals
from harnyx_commons.domain.tool_usage import LlmModelUsageCost, LlmUsageSummary, ToolUsageSummary
from harnyx_commons.domain_tweak_generation import (
    BatchTerminalGenerationError,
    CandidateStageError,
    StageRunResult,
)
from harnyx_commons.domain_tweak_generation import agent_runner as agent_runner_module
from harnyx_commons.domain_tweak_generation.agent_runner import (
    EFFORT,
    MODEL,
    DomainTweakAgentRunner,
    _AgentSDKResultContractError,
    _looks_batch_terminal,
    _raise_classified_exception,
    _raise_for_provider_result,
    _usage_from_result,
    _web_search_capture_hooks,
    _WebSearchCaptureState,
)
from harnyx_commons.domain_tweak_generation.contracts import StageName


class _StageOutput(BaseModel):
    value: str = Field(min_length=1)


def _usage(*, cost: float, call_count: int = 1) -> ToolUsageSummary:
    totals = LlmUsageTotals(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        call_count=call_count,
    )
    return ToolUsageSummary(
        llm=LlmUsageSummary(
            call_count=call_count,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            actual_cost=cost,
            providers={"vertex": {MODEL: LlmModelUsageCost(usage=totals, actual_cost=cost)}},
        ),
        actual_total_cost_usd=cost,
        actual_cost_by_provider={"vertex": cost},
    )


def test_agent_runtime_is_frozen_without_model_fallback() -> None:
    """Future failure: the evaluated Opus-low runtime must not silently drift."""
    assert MODEL == "claude-opus-5"
    assert EFFORT == "low"
    assert _looks_batch_terminal("model not found")
    assert _looks_batch_terminal("permission denied")
    assert not _looks_batch_terminal("HTTP 503")


@pytest.mark.anyio
async def test_web_search_results_are_registered_by_the_sdk_hook() -> None:
    """Future failure: native search URLs must become host-owned opaque fetch candidates."""
    responses: list[object] = []

    def registrar(response: object) -> str:
        responses.append(response)
        return "source_candidate_id=SC1 title=Report"

    state = _WebSearchCaptureState()
    matcher = _web_search_capture_hooks(registrar, state)["PostToolUse"][0]
    hook = matcher.hooks[0]
    tool_response = {
        "type": "search_result",
        "source": "https://example.com/report",
        "title": "Report",
        "content": [{"type": "text", "text": "Summary"}],
    }
    hook_input = {
        "session_id": "session",
        "transcript_path": "/workspace/transcript",
        "cwd": "/workspace",
        "agent_id": "agent",
        "agent_type": "main",
        "hook_event_name": "PostToolUse",
        "tool_name": "WebSearch",
        "tool_input": {"query": "example"},
        "tool_response": tool_response,
        "tool_use_id": "tool",
    }

    captured = await hook(hook_input, "tool", {})  # type: ignore[arg-type]

    assert responses == [tool_response]
    assert state.contract_error is None
    assert captured["hookSpecificOutput"]["additionalContext"] == "source_candidate_id=SC1 title=Report"


@pytest.mark.anyio
async def test_web_search_hook_records_registrar_failures_for_terminal_propagation() -> None:
    """Future failure: an SDK-swallowed hook exception must not let the model invent a candidate ID."""

    def registrar(_response: object) -> str:
        raise ValueError("unexpected result shape")

    state = _WebSearchCaptureState()
    matcher = _web_search_capture_hooks(registrar, state)["PostToolUse"][0]
    hook = matcher.hooks[0]
    hook_input = {
        "session_id": "session",
        "transcript_path": "/workspace/transcript",
        "cwd": "/workspace",
        "agent_id": "agent",
        "agent_type": "main",
        "hook_event_name": "PostToolUse",
        "tool_name": "WebSearch",
        "tool_input": {"query": "example"},
        "tool_response": {"results": []},
        "tool_use_id": "tool",
    }

    captured = await hook(hook_input, "tool", {})  # type: ignore[arg-type]

    assert state.contract_error == "ValueError: unexpected result shape"
    assert "Do not invent" in captured["hookSpecificOutput"]["additionalContext"]


@pytest.mark.parametrize("status", [401, 403, 404])
def test_auth_and_model_access_results_stop_the_batch(status: int) -> None:
    """Future failure: a broken shared provider configuration must not burn fresh candidates forever."""
    result = ResultMessage(
        subtype="error",
        duration_ms=1,
        duration_api_ms=1,
        is_error=True,
        num_turns=1,
        session_id="session",
        result="provider failure",
        api_error_status=status,
    )

    with pytest.raises(BatchTerminalGenerationError):
        _raise_for_provider_result("question_generation", result, tool_usage=ToolUsageSummary.zero())


def test_invalid_provider_request_stops_the_batch() -> None:
    """Future failure: a shared invalid request must not trigger an endless stream of fresh candidates."""
    result = ResultMessage(
        subtype="error",
        duration_ms=1,
        duration_api_ms=1,
        is_error=True,
        num_turns=1,
        session_id="session",
        result="invalid provider request",
        api_error_status=400,
    )

    with pytest.raises(BatchTerminalGenerationError):
        _raise_for_provider_result("question_generation", result, tool_usage=ToolUsageSummary.zero())


@pytest.mark.parametrize("status", [429, 500, 503])
def test_rate_limit_and_server_results_end_only_the_candidate(status: int) -> None:
    """Future failure: transient provider pressure must leave the slot open for a fresh next-round attempt."""
    result = ResultMessage(
        subtype="error",
        duration_ms=1,
        duration_api_ms=1,
        is_error=True,
        num_turns=1,
        session_id="session",
        result="temporary provider failure",
        api_error_status=status,
    )

    with pytest.raises(CandidateStageError, match="temporary provider failure") as captured:
        _raise_for_provider_result("question_generation", result, tool_usage=ToolUsageSummary.zero())

    assert captured.value.failure_class == "transient_provider"


def test_structured_output_retry_exhaustion_is_a_candidate_contract_failure() -> None:
    """Future failure: malformed structured output must not be reported as provider pressure."""
    result = ResultMessage(
        subtype="error_max_structured_output_retries",
        duration_ms=1,
        duration_api_ms=1,
        is_error=True,
        num_turns=3,
        session_id="session",
        result="Failed to produce valid structured output",
    )
    usage = _usage(cost=0.75, call_count=3)

    with pytest.raises(CandidateStageError) as captured:
        _raise_for_provider_result(
            "reference",
            result,
            tool_usage=usage,
            actual_llm_cost_usd=0.75,
        )

    assert captured.value.failure_class == "contract_invalid"
    assert captured.value.tool_usage == usage
    assert captured.value.actual_llm_cost_usd == 0.75


@pytest.mark.parametrize(
    "error",
    [CLINotFoundError("missing CLI"), CLIConnectionError("CLI startup failed")],
)
def test_agent_sdk_startup_errors_stop_the_batch(error: Exception) -> None:
    """Future failure: a broken shared SDK runtime must not refill forever as if candidates were bad."""
    with pytest.raises(BatchTerminalGenerationError) as captured:
        _raise_classified_exception("portfolio", error, client_initialized=False, elapsed_ms=12.0)

    assert captured.value.failure_class == "sdk_or_provider_configuration"
    assert captured.value.stage == "portfolio"


def test_unknown_sdk_exception_stops_the_batch_instead_of_refilling() -> None:
    """Future failure: an unknown shared runtime bug must not be misreported as candidate pressure."""
    with pytest.raises(BatchTerminalGenerationError) as captured:
        _raise_classified_exception(
            "question_generation",
            RuntimeError("unexpected SDK failure"),
            client_initialized=True,
            elapsed_ms=12.0,
        )

    assert captured.value.failure_class == "unexpected_sdk_failure"
    assert captured.value.stage == "question_generation"


@pytest.mark.parametrize(
    "error",
    [CLIConnectionError("stream disconnected"), ProcessError("CLI exited", exit_code=1)],
)
def test_initialized_sdk_transport_failure_ends_only_the_candidate(error: Exception) -> None:
    """Future failure: one interrupted SDK process must refill its slot rather than abort every sibling."""
    usage = ToolUsageSummary(actual_total_cost_usd=1.25, actual_cost_by_provider={"vertex": 1.25})

    with pytest.raises(CandidateStageError) as captured:
        _raise_classified_exception(
            "reference",
            error,
            client_initialized=True,
            elapsed_ms=12.0,
            tool_usage=usage,
        )

    assert captured.value.failure_class == "transient_provider"
    assert captured.value.tool_usage.actual_total_cost_usd == 1.25


def test_result_usage_counts_every_agent_turn() -> None:
    """Future failure: agentic tool turns must not be reported as one provider call."""
    result = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=4,
        session_id="session",
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    usage = _usage_from_result(result, search_calls=2)

    assert usage.llm.call_count == 4
    assert usage.llm.providers["vertex"][MODEL].usage.call_count == 4
    assert usage.actual_total_cost_usd is None


def test_result_usage_accepts_additive_sdk_usage_fields() -> None:
    """Future failure: additive provider accounting must not break the pinned required token contract."""
    result = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=2,
        session_id="session",
        total_cost_usd=0.25,
        usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 7,
            "server_tool_use": {"web_search_requests": 1},
        },
    )

    usage = _usage_from_result(result, search_calls=1)

    assert usage.llm.prompt_tokens == 10
    assert usage.llm.completion_tokens == 5
    assert usage.llm.total_tokens == 15
    assert usage.actual_total_cost_usd == 0.25


@pytest.mark.parametrize(
    "usage_payload",
    [
        None,
        {"prompt_tokens": 10, "completion_tokens": 5},
        {"input_tokens": "10", "output_tokens": 5},
        {"input_tokens": True, "output_tokens": 5},
        {"input_tokens": -1, "output_tokens": 5},
        {"input_tokens": 10},
    ],
    ids=("missing", "renamed", "string", "boolean", "negative", "partial"),
)
def test_success_result_usage_rejects_drifted_sdk_accounting(usage_payload: object) -> None:
    """Future failure: SDK accounting drift must not become a successful zero-usage stage."""
    result = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="session",
        usage=cast(Any, usage_payload),
    )

    with pytest.raises(_AgentSDKResultContractError, match="accounting contract invalid"):
        _usage_from_result(result, search_calls=0)


@pytest.mark.parametrize(
    ("num_turns", "total_cost_usd"),
    [
        (cast(Any, True), 0.25),
        (-1, 0.25),
        (1, cast(Any, True)),
        (1, cast(Any, "0.25")),
        (1, -0.25),
    ],
    ids=("boolean-turns", "negative-turns", "boolean-cost", "string-cost", "negative-cost"),
)
def test_result_usage_rejects_invalid_turn_and_cost_accounting(
    num_turns: int,
    total_cost_usd: float | None,
) -> None:
    """Future failure: invalid SDK counts and dollars must not enter aggregate observability."""
    result = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=num_turns,
        session_id="session",
        total_cost_usd=total_cost_usd,
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    with pytest.raises(_AgentSDKResultContractError, match="accounting contract invalid"):
        _usage_from_result(result, search_calls=0)


def test_error_result_without_usage_preserves_provider_error_accounting() -> None:
    """Future failure: provider error results may omit usage without masking their provider classification."""
    result = ResultMessage(
        subtype="error",
        duration_ms=1,
        duration_api_ms=1,
        is_error=True,
        num_turns=1,
        session_id="session",
        api_error_status=401,
        result="authentication failed",
    )

    usage = _usage_from_result(result, search_calls=0)

    assert usage.llm.call_count == 1
    assert usage.llm.total_tokens == 0
    with pytest.raises(BatchTerminalGenerationError) as captured:
        _raise_for_provider_result("question_generation", result, tool_usage=usage)
    assert captured.value.failure_class == "provider_auth"


def test_sdk_accounting_contract_error_is_batch_terminal_configuration_failure() -> None:
    """Future failure: shared SDK result drift must not be refilled as candidate-specific pressure."""
    error = _AgentSDKResultContractError("invalid accounting")

    with pytest.raises(BatchTerminalGenerationError) as captured:
        _raise_classified_exception(
            "question_generation",
            error,
            client_initialized=True,
            elapsed_ms=1.0,
        )

    assert captured.value.failure_class == "sdk_or_provider_configuration"


def test_result_usage_preserves_exact_reported_zero_cost() -> None:
    result = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="session",
        total_cost_usd=0.0,
        usage={"input_tokens": 1, "output_tokens": 1},
    )

    usage = _usage_from_result(result, search_calls=0)

    assert usage.actual_total_cost_usd == 0.0
    assert usage.llm.actual_cost == 0.0
    assert usage.actual_cost_by_provider == {"vertex": 0.0}


@pytest.mark.anyio
async def test_each_stage_records_one_named_generation_with_explicit_known_or_unknown_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: stage cost attribution must not collapse into one opaque batch total."""
    starts: list[dict[str, object]] = []
    updates: list[tuple[object, dict[str, object]]] = []
    stage_costs: dict[StageName, float | None] = {
        "portfolio": 1.0,
        "question_generation": 2.0,
        "reference": 0.0,
        "reference_repair": 4.0,
        "audit": None,
    }

    class ObservationScope:
        def __init__(self, observation: object) -> None:
            self.observation = observation

        def __enter__(self) -> object:
            return self.observation

        def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            return False

    def start(name: str, as_type: str, **kwargs: object) -> ObservationScope:
        observation = object()
        starts.append({"name": name, "as_type": as_type, **kwargs, "observation": observation})
        return ObservationScope(observation)

    def update(observation: object, **kwargs: object) -> None:
        updates.append((observation, kwargs))

    async def execute(**kwargs: object) -> StageRunResult:
        stage = cast(StageName, kwargs["stage"])
        cost = stage_costs[stage]
        return StageRunResult(
            _StageOutput(value=stage),
            12.0,
            _usage(cost=cost or 0.0, call_count=2),
            validation_repaired=stage == "reference_repair",
            actual_llm_cost_usd=cost,
        )

    monkeypatch.setattr(agent_runner_module, "start_metadata_only_observation", start)
    monkeypatch.setattr(agent_runner_module, "update_generation_best_effort", update)
    runner = DomainTweakAgentRunner(executor=execute)

    for stage in get_args(StageName):
        await runner.run_stage(
            stage=stage,
            system_prompt="system",
            prompt="prompt",
            output_model=_StageOutput,
            timeout_seconds=30,
        )

    assert [item["name"] for item in starts] == [f"miner_task_generation.{stage}" for stage in get_args(StageName)]
    assert all(item["as_type"] == "generation" for item in starts)
    for stage, (observation, update_kwargs) in zip(get_args(StageName), updates, strict=True):
        start_item = starts[get_args(StageName).index(stage)]
        assert observation is start_item["observation"]
        expected_cost = stage_costs[stage]
        assert update_kwargs["cost_details"] == (None if expected_cost is None else {"total": expected_cost})
        assert update_kwargs["usage_details"] == {"input": 10, "output": 5, "total": 15}
        assert cast(dict[str, object], update_kwargs["metadata"])["cost_status"] == (
            "unavailable" if expected_cost is None else "reported"
        )


@pytest.mark.anyio
async def test_typed_stage_failure_records_billable_usage_and_sanitized_failure_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: rejected paid work must remain attributable without exporting exception content."""
    updates: list[dict[str, object]] = []

    class ObservationScope:
        def __enter__(self) -> object:
            return object()

        def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            return False

    async def execute(**_kwargs: object) -> StageRunResult:
        raise CandidateStageError(
            "transient_provider",
            "audit",
            "private-provider-error-sentinel",
            tool_usage=_usage(cost=2.5, call_count=3),
            elapsed_ms=42,
            actual_llm_cost_usd=2.5,
        )

    monkeypatch.setattr(
        agent_runner_module,
        "start_metadata_only_observation",
        lambda *args, **kwargs: ObservationScope(),
    )
    monkeypatch.setattr(
        agent_runner_module,
        "update_generation_best_effort",
        lambda observation, **kwargs: updates.append(kwargs),
    )

    with pytest.raises(CandidateStageError, match="private-provider-error-sentinel"):
        await DomainTweakAgentRunner(executor=execute).run_stage(
            stage="audit",
            system_prompt="system",
            prompt="prompt",
            output_model=_StageOutput,
            timeout_seconds=30,
        )

    assert updates == [
        {
            "usage_details": {"input": 10, "output": 5, "total": 15},
            "cost_details": {"total": 2.5},
            "metadata": {
                "failure_class": "transient_provider",
                "elapsed_ms": 42,
                "agent_turn_count": 3,
                "cost_status": "reported",
            },
            "level": "ERROR",
            "status_message": "transient_provider",
        }
    ]
    assert "private-provider-error-sentinel" not in repr(updates)


@pytest.mark.anyio
async def test_query_failure_before_any_result_marks_stage_cost_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: startup and first-query timeouts must not be reported as a known zero cost."""

    class QueryTimeoutClient:
        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            return False

        async def query(self, _prompt: str) -> None:
            raise TimeoutError("query failed before result")

    monkeypatch.setattr(agent_runner_module, "ClaudeSDKClient", lambda **kwargs: QueryTimeoutClient())
    monkeypatch.setattr(
        agent_runner_module,
        "start_metadata_only_observation",
        lambda *args, **kwargs: nullcontext(None),
    )

    with pytest.raises(CandidateStageError) as captured:
        await DomainTweakAgentRunner().run_stage(
            stage="question_generation",
            system_prompt="system",
            prompt="prompt",
            output_model=_StageOutput,
            timeout_seconds=30,
        )

    assert captured.value.tool_usage.actual_total_cost_usd is None
    assert captured.value.tool_usage.llm.actual_cost is None
    assert captured.value.actual_llm_cost_usd is None


@pytest.mark.anyio
async def test_receive_response_process_failure_ends_only_the_initialized_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: a CLI process lost during a turn must not abort the whole refill batch."""

    class InterruptedClient:
        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            return False

        async def query(self, _prompt: str) -> None:
            return None

        async def receive_response(self):
            raise ProcessError("CLI stream interrupted", exit_code=1)
            yield  # pragma: no cover

    monkeypatch.setattr(agent_runner_module, "ClaudeSDKClient", lambda **kwargs: InterruptedClient())
    monkeypatch.setattr(
        agent_runner_module,
        "start_metadata_only_observation",
        lambda *args, **kwargs: nullcontext(None),
    )

    with pytest.raises(CandidateStageError) as captured:
        await DomainTweakAgentRunner().run_stage(
            stage="question_generation",
            system_prompt="system",
            prompt="prompt",
            output_model=_StageOutput,
            timeout_seconds=30,
        )

    assert captured.value.failure_class == "transient_provider"


@pytest.mark.anyio
async def test_receive_response_without_result_ends_only_the_initialized_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: an incomplete SDK stream must refill its slot rather than abort every sibling."""

    class IncompleteClient:
        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            return False

        async def query(self, _prompt: str) -> None:
            return None

        async def receive_response(self):
            if False:
                yield None

    monkeypatch.setattr(agent_runner_module, "ClaudeSDKClient", lambda **kwargs: IncompleteClient())
    monkeypatch.setattr(
        agent_runner_module,
        "start_metadata_only_observation",
        lambda *args, **kwargs: nullcontext(None),
    )

    with pytest.raises(CandidateStageError) as captured:
        await DomainTweakAgentRunner().run_stage(
            stage="question_generation",
            system_prompt="system",
            prompt="prompt",
            output_model=_StageOutput,
            timeout_seconds=30,
        )

    assert captured.value.failure_class == "transient_provider"


@pytest.mark.anyio
async def test_stage_contract_feedback_repairs_in_same_client_and_aggregates_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: a mechanical output defect must not discard paid research or open a fresh client."""
    results = [
        ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=2,
            session_id="session",
            total_cost_usd=0.5,
            usage={"input_tokens": 10, "output_tokens": 5},
            structured_output={"value": "wrong"},
        ),
        ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="session",
            total_cost_usd=0.75,
            usage={"input_tokens": 7, "output_tokens": 3},
            structured_output={"value": "right"},
        ),
    ]

    class FeedbackClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            return False

        async def query(self, prompt: str) -> None:
            self.queries.append(prompt)

        async def receive_response(self):
            yield results.pop(0)

    clients: list[FeedbackClient] = []

    def client_factory(**_kwargs: object) -> FeedbackClient:
        client = FeedbackClient()
        clients.append(client)
        return client

    monkeypatch.setattr(agent_runner_module, "ClaudeSDKClient", client_factory)
    monkeypatch.setattr(
        agent_runner_module,
        "start_metadata_only_observation",
        lambda *args, **kwargs: nullcontext(None),
    )

    result = await DomainTweakAgentRunner().run_stage(
        stage="reference",
        system_prompt="system",
        prompt="initial",
        output_model=_StageOutput,
        output_validator=lambda output: ("value must be right",) if cast(_StageOutput, output).value != "right" else (),
        timeout_seconds=30,
    )

    assert cast(_StageOutput, result.output).value == "right"
    assert result.validation_repaired
    assert result.tool_usage.llm.call_count == 3
    assert result.tool_usage.llm.prompt_tokens == 17
    assert result.actual_llm_cost_usd == 1.25
    assert len(clients) == 1
    assert clients[0].queries[0] == "initial"
    assert "value must be right" in clients[0].queries[1]


@pytest.mark.anyio
async def test_persistent_packet_size_defect_uses_feedback_then_contract_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: an unfit reference packet must use bounded feedback before candidate loss."""
    results = [
        ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="session",
            total_cost_usd=0.25,
            usage={"input_tokens": 5, "output_tokens": 3},
            structured_output={"value": "still too large"},
        )
        for _ in range(2)
    ]

    class FeedbackClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            return False

        async def query(self, prompt: str) -> None:
            self.queries.append(prompt)

        async def receive_response(self):
            yield results.pop(0)

    client = FeedbackClient()
    monkeypatch.setattr(agent_runner_module, "ClaudeSDKClient", lambda **_kwargs: client)
    monkeypatch.setattr(
        agent_runner_module,
        "start_metadata_only_observation",
        lambda *args, **kwargs: nullcontext(None),
    )

    with pytest.raises(CandidateStageError) as captured:
        await DomainTweakAgentRunner().run_stage(
            stage="reference",
            system_prompt="system",
            prompt="initial",
            output_model=_StageOutput,
            output_validator=lambda _output: ("required proof packet envelope exceeds 128000 characters",),
            timeout_seconds=30,
        )

    assert captured.value.failure_class == "contract_invalid"
    assert len(client.queries) == 2
    assert "required proof packet envelope" in client.queries[1]


@pytest.mark.anyio
async def test_feedback_query_failure_keeps_known_prefix_usage_but_marks_total_cost_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: a paid first result must not make an unfinished feedback query look fully costed."""
    first_result = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=2,
        session_id="session",
        total_cost_usd=1.25,
        usage={"input_tokens": 10, "output_tokens": 5},
        structured_output={},
    )

    class FeedbackTimeoutClient:
        query_count = 0

        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            return False

        async def query(self, _prompt: str) -> None:
            self.query_count += 1
            if self.query_count == 2:
                raise TimeoutError("feedback query failed before result")

        async def receive_response(self):
            yield first_result

    monkeypatch.setattr(agent_runner_module, "ClaudeSDKClient", lambda **kwargs: FeedbackTimeoutClient())
    monkeypatch.setattr(
        agent_runner_module,
        "start_metadata_only_observation",
        lambda *args, **kwargs: nullcontext(None),
    )

    with pytest.raises(CandidateStageError) as captured:
        await DomainTweakAgentRunner().run_stage(
            stage="question_generation",
            system_prompt="system",
            prompt="prompt",
            output_model=_StageOutput,
            timeout_seconds=30,
        )

    assert captured.value.tool_usage.llm.prompt_tokens == 10
    assert captured.value.tool_usage.llm.actual_cost is None
    assert captured.value.tool_usage.actual_total_cost_usd is None
    assert captured.value.tool_usage.actual_cost_by_provider == {}
    assert captured.value.actual_llm_cost_usd is None
