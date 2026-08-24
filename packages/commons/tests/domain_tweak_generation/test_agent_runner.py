from contextlib import nullcontext
from typing import Any, cast

import pytest
from claude_agent_sdk import CLIConnectionError, CLINotFoundError, ResultMessage
from pydantic import BaseModel, Field

from harnyx_commons.domain.session import LlmUsageTotals
from harnyx_commons.domain.tool_usage import LlmModelUsageCost, LlmUsageSummary, ToolUsageSummary
from harnyx_commons.domain_tweak_generation import (
    BatchTerminalGenerationError,
    CandidateStageError,
    SourceWorkspace,
)
from harnyx_commons.domain_tweak_generation import agent_runner as agent_runner_module
from harnyx_commons.domain_tweak_generation.agent_runner import (
    MODEL,
    DomainTweakAgentRunner,
    _AgentSDKResultContractError,
    _raise_classified_exception,
    _raise_for_provider_result,
    _usage_from_result,
    _web_search_capture_hooks,
    _WebSearchCaptureState,
)


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
async def test_web_search_hook_accepts_titleless_result_without_retaining_contract_error() -> None:
    """Future failure: a usable titleless result must not poison an otherwise successful agent stage."""
    workspace = SourceWorkspace()
    state = _WebSearchCaptureState()
    matcher = _web_search_capture_hooks(workspace.register_web_search_results, state)["PostToolUse"][0]
    hook = matcher.hooks[0]
    hook_input = {
        "session_id": "session",
        "transcript_path": "/workspace/transcript",
        "cwd": "/workspace",
        "agent_id": "agent",
        "agent_type": "main",
        "hook_event_name": "PostToolUse",
        "tool_name": "WebSearch",
        "tool_input": {"query": "NIST SP 800-37 revision 1"},
        "tool_response": {
            "query": "NIST SP 800-37 revision 1",
            "results": [
                {
                    "tool_use_id": "server-tool-1",
                    "content": [
                        {"title": "NIST publication", "url": "https://example.com/nist"},
                        {
                            "title": "",
                            "url": "https://beta.csrc.nist.gov/publications/detail/sp/800-37/rev-1/final",
                        },
                    ],
                },
                "Search results for query `NIST SP 800-37 revision 1`",
            ],
            "durationSeconds": 0.2,
        },
        "tool_use_id": "tool",
    }

    captured = await hook(hook_input, "tool", {})  # type: ignore[arg-type]

    assert state.contract_error is None
    context = captured["hookSpecificOutput"]["additionalContext"]
    assert "source_candidate:1" in context
    assert "source_candidate:2" in context
    titleless = workspace.get_source_candidate("source_candidate:2")
    assert titleless.url == "https://beta.csrc.nist.gov/publications/detail/sp/800-37/rev-1/final"
    assert titleless.title == ""


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


@pytest.mark.parametrize(
    "status",
    [
        401,
        404,
    ],
)
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


@pytest.mark.parametrize(
    "status",
    [
        429,
    ],
)
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
    [
        CLINotFoundError("missing CLI"),
    ],
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
    [
        CLIConnectionError("stream disconnected"),
    ],
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


@pytest.mark.parametrize(
    "usage_payload",
    [
        None,
        {"input_tokens": 10},
    ],
    ids=("missing", "renamed"),
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
    ],
    ids=("boolean-turns",),
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
