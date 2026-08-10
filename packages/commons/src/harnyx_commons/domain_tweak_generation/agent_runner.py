"""Claude Agent SDK boundary with typed provider failures and one feedback repair."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ClaudeSDKError,
    CLINotFoundError,
    HookMatcher,
    ResultMessage,
    ServerToolUseBlock,
    ToolUseBlock,
)
from claude_agent_sdk.types import HookContext, HookEvent, HookInput, SyncHookJSONOutput
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from harnyx_commons.domain.session import LlmUsageTotals
from harnyx_commons.domain.tool_usage import (
    LlmModelUsageCost,
    LlmUsageSummary,
    SearchToolUsageSummary,
    ToolUsageSummary,
)
from harnyx_commons.domain.tool_usage_accounting import (
    known_zero_actual_cost_tool_usage,
    merge_complete_actual_cost_usage,
)
from harnyx_commons.domain_tweak_generation.contracts import (
    AgentToolSet,
    BatchTerminalGenerationError,
    CandidateStageError,
    StageName,
    StageRunResult,
)
from harnyx_commons.observability.langfuse import (
    LangfuseGeneration,
    start_metadata_only_observation,
    update_generation_best_effort,
)

MODEL = "claude-opus-5"
EFFORT = "low"
_PROVIDER = "vertex"
_DISALLOWED_TOOLS = ["WebFetch", "Read", "Write", "Edit", "Bash", "Task", "Skill"]
_T = TypeVar("_T", bound=BaseModel)


class _ResultStreamEndedError(RuntimeError):
    """The initialized SDK stream ended before publishing its required result."""


class _WebSearchContractError(RuntimeError):
    """Native WebSearch returned a non-error shape outside the documented contract."""


class _AgentSDKResultContractError(RuntimeError):
    """Agent SDK result accounting did not match the pinned boundary contract."""


class _AgentSDKUsagePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)


class _AgentSDKResultAccountingPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    is_error: bool
    num_turns: int = Field(ge=0)
    total_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    usage: _AgentSDKUsagePayload | None

    @model_validator(mode="after")
    def require_success_usage(self) -> _AgentSDKResultAccountingPayload:
        if not self.is_error and self.usage is None:
            raise ValueError("usage is required for a successful Agent SDK result")
        return self


@dataclass(slots=True)
class _WebSearchCaptureState:
    contract_error: str | None = None


StageOutputValidator = Callable[[Any], Sequence[str]]


class StageExecutor(Protocol):
    async def __call__(
        self,
        *,
        stage: StageName,
        system_prompt: str,
        prompt: str,
        output_model: type[BaseModel],
        timeout_seconds: float,
        web_search: bool,
        tool_set: AgentToolSet,
        output_validator: StageOutputValidator | None,
    ) -> StageRunResult: ...


class DomainTweakAgentRunner:
    def __init__(
        self,
        *,
        project_id: str | None = None,
        region: str = "global",
        executor: StageExecutor | None = None,
    ) -> None:
        self._project_id = project_id
        self._region = region
        self._executor = executor

    async def run_stage(
        self,
        *,
        stage: StageName,
        system_prompt: str,
        prompt: str,
        output_model: type[_T],
        timeout_seconds: float,
        web_search: bool = False,
        tool_set: AgentToolSet | None = None,
        output_validator: StageOutputValidator | None = None,
    ) -> StageRunResult:
        resolved_tool_set = tool_set or AgentToolSet()
        with start_metadata_only_observation(
            f"miner_task_generation.{stage}",
            "generation",
            model=MODEL,
            model_parameters={"provider": _PROVIDER, "effort": EFFORT},
            metadata={
                "stage": stage,
                "timeout_seconds": timeout_seconds,
                "web_search": web_search,
                "allowed_tool_count": len(resolved_tool_set.allowed_tools),
            },
        ) as generation:
            try:
                result = await self._execute_stage(
                    stage=stage,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    output_model=output_model,
                    timeout_seconds=timeout_seconds,
                    web_search=web_search,
                    tool_set=resolved_tool_set,
                    output_validator=output_validator,
                )
            except (CandidateStageError, BatchTerminalGenerationError) as exc:
                _update_failed_stage_generation(generation, exc)
                raise
            except BaseException as exc:
                update_generation_best_effort(
                    generation,
                    metadata={"failure_type": type(exc).__name__, "cost_status": "unavailable"},
                    level="ERROR",
                    status_message=type(exc).__name__,
                )
                raise
            _update_successful_stage_generation(generation, result)
            return result

    async def _execute_stage(
        self,
        *,
        stage: StageName,
        system_prompt: str,
        prompt: str,
        output_model: type[_T],
        timeout_seconds: float,
        web_search: bool,
        tool_set: AgentToolSet,
        output_validator: StageOutputValidator | None,
    ) -> StageRunResult:
        if self._executor is not None:
            return await self._executor(
                stage=stage,
                system_prompt=system_prompt,
                prompt=prompt,
                output_model=output_model,
                timeout_seconds=timeout_seconds,
                web_search=web_search,
                tool_set=tool_set,
                output_validator=output_validator,
            )
        return await self._run_live(
            stage=stage,
            system_prompt=system_prompt,
            prompt=prompt,
            output_model=output_model,
            timeout_seconds=timeout_seconds,
            web_search=web_search,
            tool_set=tool_set,
            output_validator=output_validator,
        )

    async def _run_live(
        self,
        *,
        stage: StageName,
        system_prompt: str,
        prompt: str,
        output_model: type[_T],
        timeout_seconds: float,
        web_search: bool,
        tool_set: AgentToolSet,
        output_validator: StageOutputValidator | None,
    ) -> StageRunResult:
        started = time.perf_counter()
        total_usage = known_zero_actual_cost_tool_usage()
        env = {
            "API_TIMEOUT_MS": str(int(timeout_seconds * 1000)),
            "CLAUDE_CODE_MAX_RETRIES": "0",
            "CLAUDE_CODE_USE_VERTEX": "1",
            "CLOUD_ML_REGION": self._region,
            "DISABLE_TELEMETRY": "1",
            "ENABLE_TOOL_SEARCH": "false",
        }
        if self._project_id:
            env["ANTHROPIC_VERTEX_PROJECT_ID"] = self._project_id
        tools = ["WebSearch"] if web_search else []
        allowed_tools = [*tools, *tool_set.allowed_tools]
        if web_search and tool_set.search_result_registrar is None:
            raise BatchTerminalGenerationError(
                "sdk_or_provider_configuration",
                "WebSearch requires a candidate-local search result registrar",
                stage=stage,
                tool_usage=total_usage,
                actual_llm_cost_usd=0.0,
            )
        search_capture_state = _WebSearchCaptureState()
        hooks = (
            _web_search_capture_hooks(tool_set.search_result_registrar, search_capture_state)
            if web_search and tool_set.search_result_registrar is not None
            else None
        )
        options = ClaudeAgentOptions(
            tools=tools,
            allowed_tools=allowed_tools,
            disallowed_tools=_DISALLOWED_TOOLS,
            system_prompt=system_prompt,
            mcp_servers=tool_set.mcp_servers,
            strict_mcp_config=True,
            permission_mode="dontAsk",
            model=MODEL,
            fallback_model=None,
            cwd=Path.cwd(),
            env=env,
            include_partial_messages=False,
            setting_sources=[],
            skills=[],
            thinking={"type": "adaptive", "display": "summarized"},
            effort=EFFORT,
            output_format={"type": "json_schema", "schema": output_model.model_json_schema()},
            hooks=hooks,
        )
        validation_repaired = False
        attempted_query_costs: list[float | None] = []
        client_initialized = False
        try:
            async with asyncio.timeout(timeout_seconds):
                async with ClaudeSDKClient(options=options) as client:
                    client_initialized = True
                    turn_prompt = prompt
                    for attempt_index in range(2):
                        attempted_query_costs.append(None)
                        await client.query(turn_prompt)
                        result, search_calls = await _receive_result(client)
                        usage = _usage_from_result(result, search_calls=search_calls)
                        attempted_query_costs[-1] = usage.actual_total_cost_usd
                        total_usage = merge_complete_actual_cost_usage(total_usage, usage)
                        if search_capture_state.contract_error is not None:
                            raise _WebSearchContractError(search_capture_state.contract_error)
                        actual_llm_cost_usd = _strict_sum_attempted_query_costs(attempted_query_costs)
                        _raise_for_provider_result(
                            stage,
                            result,
                            tool_usage=total_usage,
                            actual_llm_cost_usd=actual_llm_cost_usd,
                        )
                        try:
                            parsed = output_model.model_validate(result.structured_output)
                        except ValidationError as exc:
                            defects = _pydantic_defects(exc)
                        else:
                            defects = tuple(output_validator(parsed)) if output_validator is not None else ()
                        if defects:
                            if attempt_index == 1:
                                raise CandidateStageError(
                                    "contract_invalid",
                                    stage,
                                    "structured output remained invalid after feedback: " + "; ".join(defects),
                                    tool_usage=total_usage,
                                    elapsed_ms=(time.perf_counter() - started) * 1000,
                                    actual_llm_cost_usd=actual_llm_cost_usd,
                                )
                            validation_repaired = True
                            turn_prompt = _validation_feedback(defects, output_model)
                            continue
                        return StageRunResult(
                            output=parsed,
                            elapsed_ms=(time.perf_counter() - started) * 1000,
                            tool_usage=total_usage,
                            retry_after_seconds=_retry_after_seconds(result),
                            validation_repaired=validation_repaired,
                            actual_llm_cost_usd=actual_llm_cost_usd,
                        )
        except CandidateStageError:
            raise
        except BatchTerminalGenerationError:
            raise
        except TimeoutError as exc:
            failure_usage = _usage_with_attempted_query_costs(total_usage, attempted_query_costs)
            raise CandidateStageError(
                "transient_provider",
                stage,
                f"{stage} timed out",
                tool_usage=failure_usage,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                actual_llm_cost_usd=_strict_sum_attempted_query_costs(attempted_query_costs),
            ) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure_usage = _usage_with_attempted_query_costs(total_usage, attempted_query_costs)
            _raise_classified_exception(
                stage,
                exc,
                client_initialized=client_initialized,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                tool_usage=failure_usage,
                actual_llm_cost_usd=_strict_sum_attempted_query_costs(attempted_query_costs),
            )
        raise AssertionError("stage execution must return or raise")


def _update_successful_stage_generation(
    generation: LangfuseGeneration | None,
    result: StageRunResult,
) -> None:
    update_generation_best_effort(
        generation,
        usage_details=_stage_usage_details(result.tool_usage),
        cost_details=_stage_cost_details(result.actual_llm_cost_usd),
        metadata={
            "elapsed_ms": result.elapsed_ms,
            "agent_turn_count": result.tool_usage.llm.call_count,
            "validation_repaired": result.validation_repaired,
            "cost_status": _cost_status(result.actual_llm_cost_usd),
        },
    )


def _update_failed_stage_generation(
    generation: LangfuseGeneration | None,
    error: CandidateStageError | BatchTerminalGenerationError,
) -> None:
    update_generation_best_effort(
        generation,
        usage_details=_stage_usage_details(error.tool_usage),
        cost_details=_stage_cost_details(error.actual_llm_cost_usd),
        metadata={
            "failure_class": error.failure_class,
            "elapsed_ms": error.elapsed_ms,
            "agent_turn_count": error.tool_usage.llm.call_count,
            "cost_status": _cost_status(error.actual_llm_cost_usd),
        },
        level="ERROR",
        status_message=error.failure_class,
    )


def _stage_usage_details(usage: ToolUsageSummary) -> dict[str, int]:
    return {
        "input": usage.llm.prompt_tokens,
        "output": usage.llm.completion_tokens,
        "total": usage.llm.total_tokens,
    }


def _stage_cost_details(actual_llm_cost_usd: float | None) -> dict[str, float] | None:
    if actual_llm_cost_usd is None:
        return None
    return {"total": actual_llm_cost_usd}


def _cost_status(actual_llm_cost_usd: float | None) -> str:
    return "unavailable" if actual_llm_cost_usd is None else "reported"


def _web_search_capture_hooks(
    registrar: Callable[[object], str],
    state: _WebSearchCaptureState,
) -> dict[HookEvent, list[HookMatcher]]:
    async def capture_results(
        hook_input: HookInput,
        _tool_use_id: str | None,
        _context: HookContext,
    ) -> SyncHookJSONOutput:
        if hook_input["hook_event_name"] != "PostToolUse" or hook_input["tool_name"] != "WebSearch":
            return {}
        try:
            additional_context = registrar(hook_input["tool_response"])
        except Exception as exc:
            state.contract_error = f"{type(exc).__name__}: {exc}"[:1_000]
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": (
                        "The host rejected this WebSearch result because its response contract was invalid. "
                        "Do not invent or use a source_candidate_id."
                    ),
                }
            }
        if not additional_context:
            raise _WebSearchContractError("WebSearch result registrar returned empty context")
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": additional_context,
            }
        }

    return {"PostToolUse": [HookMatcher(matcher="WebSearch", hooks=[capture_results])]}


async def _receive_result(client: ClaudeSDKClient) -> tuple[ResultMessage, int]:
    result: ResultMessage | None = None
    search_calls = 0
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            search_calls += sum(
                1
                for block in message.content
                if (isinstance(block, ToolUseBlock) and block.name == "WebSearch")
                or (isinstance(block, ServerToolUseBlock) and block.name == "web_search")
            )
        if isinstance(message, ResultMessage):
            result = message
    if result is None:
        raise _ResultStreamEndedError("Agent SDK stream ended without ResultMessage")
    return result, search_calls


def _pydantic_defects(error: ValidationError) -> tuple[str, ...]:
    defects = tuple(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors(include_url=False)
    )
    return defects


def _validation_feedback(defects: Sequence[str], output_model: type[BaseModel]) -> str:
    bounded_defects = [
        {
            "message": item[:500],
        }
        for item in defects[:20]
    ]
    return (
        "Your previous structured output violated the required contract. Correct only these concrete defects and "
        "return a complete replacement object. Do not restart research or change supported facts.\n"
        + json.dumps({"defects": bounded_defects, "schema": output_model.model_json_schema()}, ensure_ascii=False)
    )


def _usage_from_result(result: ResultMessage, *, search_calls: int) -> ToolUsageSummary:
    try:
        accounting = _AgentSDKResultAccountingPayload.model_validate(
            {
                "is_error": result.is_error,
                "num_turns": result.num_turns,
                "total_cost_usd": result.total_cost_usd,
                "usage": result.usage,
            }
        )
    except ValidationError as exc:
        raise _AgentSDKResultContractError(
            "Agent SDK result accounting contract invalid: " + "; ".join(_pydantic_defects(exc))
        ) from exc

    raw_usage = accounting.usage
    prompt_tokens = raw_usage.input_tokens if raw_usage is not None else 0
    completion_tokens = raw_usage.output_tokens if raw_usage is not None else 0
    reported_total_tokens = raw_usage.total_tokens if raw_usage is not None else None
    total_tokens = reported_total_tokens or prompt_tokens + completion_tokens
    reasoning_tokens = (raw_usage.reasoning_tokens if raw_usage is not None else None) or 0
    actual_cost = accounting.total_cost_usd
    call_count = accounting.num_turns
    totals = LlmUsageTotals(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        call_count=call_count,
    )
    return ToolUsageSummary(
        search_tool=SearchToolUsageSummary(call_count=search_calls, actual_cost=0.0),
        llm=LlmUsageSummary(
            call_count=call_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
            actual_cost=actual_cost,
            providers={
                _PROVIDER: {
                    MODEL: LlmModelUsageCost(usage=totals, actual_cost=actual_cost),
                }
            },
        ),
        actual_total_cost_usd=actual_cost,
        actual_cost_by_provider={} if actual_cost is None else {_PROVIDER: actual_cost},
    )


def _strict_sum_attempted_query_costs(costs: list[float | None]) -> float | None:
    if not costs or any(cost is None for cost in costs):
        return None
    return sum(cost for cost in costs if cost is not None)


def _usage_with_attempted_query_costs(
    usage: ToolUsageSummary,
    attempted_query_costs: Sequence[float | None],
) -> ToolUsageSummary:
    if not attempted_query_costs or all(cost is not None for cost in attempted_query_costs):
        return usage
    return merge_complete_actual_cost_usage(usage, ToolUsageSummary.zero())


def _raise_for_provider_result(
    stage: StageName,
    result: ResultMessage,
    *,
    tool_usage: ToolUsageSummary,
    actual_llm_cost_usd: float | None = None,
) -> None:
    if not result.is_error:
        return
    message = (
        " ".join(
            part
            for part in (
                result.result,
                result.terminal_reason,
                *(result.errors or []),
            )
            if part
        )
        or "Agent SDK returned an error result"
    )
    status = result.api_error_status
    if result.subtype == "error_max_structured_output_retries":
        raise CandidateStageError(
            "contract_invalid",
            stage,
            message,
            tool_usage=tool_usage,
            elapsed_ms=float(result.duration_ms),
            actual_llm_cost_usd=actual_llm_cost_usd,
        )
    if status in {401, 403}:
        raise BatchTerminalGenerationError(
            "provider_auth",
            message,
            stage=stage,
            tool_usage=tool_usage,
            elapsed_ms=float(result.duration_ms),
            actual_llm_cost_usd=actual_llm_cost_usd,
        )
    if status == 404 or _looks_batch_terminal(message):
        raise BatchTerminalGenerationError(
            "provider_configuration",
            message,
            stage=stage,
            tool_usage=tool_usage,
            elapsed_ms=float(result.duration_ms),
            actual_llm_cost_usd=actual_llm_cost_usd,
        )
    if status == 429 or (status is not None and status >= 500):
        raise CandidateStageError(
            "transient_provider",
            stage,
            message,
            retry_after_seconds=_retry_after_seconds(result),
            tool_usage=tool_usage,
            elapsed_ms=float(result.duration_ms),
            actual_llm_cost_usd=actual_llm_cost_usd,
        )
    if status is not None and 400 <= status < 500 and status != 408:
        raise BatchTerminalGenerationError(
            "provider_configuration",
            message,
            stage=stage,
            tool_usage=tool_usage,
            elapsed_ms=float(result.duration_ms),
            actual_llm_cost_usd=actual_llm_cost_usd,
        )
    raise CandidateStageError(
        "transient_provider",
        stage,
        message,
        tool_usage=tool_usage,
        elapsed_ms=float(result.duration_ms),
        actual_llm_cost_usd=actual_llm_cost_usd,
    )


def _raise_classified_exception(
    stage: StageName,
    exc: Exception,
    *,
    client_initialized: bool,
    elapsed_ms: float,
    tool_usage: ToolUsageSummary | None = None,
    actual_llm_cost_usd: float | None = None,
) -> None:
    message = f"{type(exc).__name__}: {exc}"
    is_initialized_transport_failure = isinstance(exc, _ResultStreamEndedError) or (
        isinstance(exc, ClaudeSDKError) and not isinstance(exc, CLINotFoundError)
    )
    if client_initialized and is_initialized_transport_failure:
        if not _looks_batch_terminal(message):
            raise CandidateStageError(
                "transient_provider",
                stage,
                message,
                tool_usage=tool_usage,
                elapsed_ms=elapsed_ms,
                actual_llm_cost_usd=actual_llm_cost_usd,
            ) from exc
    failure_class = (
        "sdk_or_provider_configuration"
        if isinstance(
            exc,
            (ClaudeSDKError, ImportError, TypeError, _AgentSDKResultContractError, _WebSearchContractError),
        )
        or _looks_batch_terminal(message)
        else "unexpected_sdk_failure"
    )
    raise BatchTerminalGenerationError(
        failure_class,
        message,
        stage=stage,
        tool_usage=tool_usage,
        elapsed_ms=elapsed_ms,
        actual_llm_cost_usd=actual_llm_cost_usd,
    ) from exc


def _looks_batch_terminal(message: str) -> bool:
    lowered = message.casefold()
    return any(
        marker in lowered
        for marker in (
            "authentication",
            "unauthorized",
            "permission denied",
            "access denied",
            "invalid model",
            "model not found",
            "model is not available",
            "not entitled",
            "not enabled",
            "invalid mcp",
            "tool schema",
        )
    )


def _retry_after_seconds(value: object) -> float | None:
    for name in ("retry_after", "retry_after_seconds"):
        raw = getattr(value, name, None)
        if isinstance(raw, int | float) and raw >= 0:
            return float(raw)
    text = str(value)
    match = re.search(r"retry[- ]after[:= ]+(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    return float(match.group(1)) if match else None


__all__ = ["DomainTweakAgentRunner", "EFFORT", "MODEL", "StageExecutor"]
