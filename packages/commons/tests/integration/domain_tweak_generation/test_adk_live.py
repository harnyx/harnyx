from __future__ import annotations

import os

import pytest

from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.domain_tweak_generation import (
    DomainTweakAdkPhaseResult,
    DomainTweakAdkRunConfig,
    DomainTweakAdkRunner,
)
from harnyx_commons.domain_tweak_generation.prompts import phase_instruction
from harnyx_commons.domain_tweak_generation.validation import (
    validate_form_blueprint_output,
    validate_form_review_output,
    validate_reference_answer_output,
    validate_semantic_support_output,
    validate_structured_output_materialization,
)
from harnyx_commons.llm.providers.vertex.credentials import cleanup_credentials_file, prepare_credentials
from harnyx_commons.miner_task_generation import (
    DOMAIN_TWEAK_REQUIREMENT_CATEGORIES,
    DomainTweakFormBlueprint,
    DomainTweakFormReview,
    DomainTweakQuestionRequirement,
    DomainTweakReferenceAnswerOutput,
    DomainTweakReferenceClaim,
    DomainTweakRequirementCategoryAudit,
    DomainTweakSemanticSupportReview,
    DomainTweakStructuredOutputMaterialization,
    DomainTweakStructuredOutputMaterializationWire,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.expensive,
    pytest.mark.anyio("asyncio"),
    pytest.mark.flaky(
        reruns=1,
        only_rerun=[r"terminal_status=(?:timeout|invocation_error|validation_failed)"],
    ),
]


async def test_adk_live_native_schema_no_tool_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials_path = _configure_adk_vertex_environment(monkeypatch)
    runner = DomainTweakAdkRunner()
    config = _config()
    try:
        blueprint = await runner.run_phase(
            phase="form_blueprint",
            prompt=(
                "Analyze this form: Which entries in a closed list satisfy two retrieved predicates? "
                "Return proceed with a filter operation, at least one invariant, a retrieval boundary, "
                "an exhaustive-list answer shape, empty optional arrays where appropriate, and null "
                "no_generate_reason."
            ),
            config=config,
            agent_instruction=phase_instruction("form_blueprint"),
            search_enabled=False,
            function_tools=(),
            output_schema=DomainTweakFormBlueprint,
            validate=validate_form_blueprint_output,
        )
        form_review = await runner.run_phase(
            phase="form_review",
            prompt=(
                "Return form_match true. Emit one question requirement with ID metric, category "
                "metric_or_field_relation, relation derived_calculation, and a complete nine-row category "
                "audit covering candidate_universe, metric_or_field_relation, scope, time_qualifier, "
                "cardinality, completeness, ranking, absence, and other exactly once. Only "
                "metric_or_field_relation is present."
            ),
            config=config,
            agent_instruction=phase_instruction("form_review"),
            search_enabled=False,
            function_tools=(),
            output_schema=DomainTweakFormReview,
            validate=validate_form_review_output,
        )
        semantic = await runner.run_phase(
            phase="semantic_support_gate",
            prompt=(
                "Return pass. Emit one supported requirement finding for requirement ID metric and one "
                "supported claim finding for claim ID answer. Both use evidence window window-1. Return "
                "no unmanifested material claims and null abandon_reason."
            ),
            config=config,
            agent_instruction=phase_instruction("semantic_support_gate"),
            search_enabled=False,
            function_tools=(),
            output_schema=DomainTweakSemanticSupportReview,
            validate=validate_semantic_support_output,
        )
    finally:
        cleanup_credentials_file(credentials_path)

    for result in (blueprint, form_review, semantic):
        assert result.terminal_status == "validated", _phase_result_debug(result)
        assert result.attempts
        assert all(not attempt.event_summaries or not _tool_names(attempt) for attempt in result.attempts)


async def test_adk_live_reference_stage_combines_search_and_source_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials_path = _configure_adk_vertex_environment(monkeypatch)
    runner = DomainTweakAdkRunner()
    tool_calls: list[str] = []

    async def read_cached_source(
        source_id: str,
        offset: int = 0,
        limit: int = 10_000,
    ) -> dict[str, object]:
        """Read a previously acquired source range."""
        tool_calls.append("read_cached_source")
        return {
            "status": "acquired",
            "source_id": source_id,
            "window_id": "window-1",
            "offset": offset,
            "limit": limit,
            "url": "https://example.com/",
            "title": "Example Domain",
            "content": "Example Domain is a domain for use in illustrative examples.",
        }

    async def acquire_sources(
        claim_id: str,
        urls: list[str],
        offset: int = 0,
        limit: int = 10_000,
    ) -> dict[str, object]:
        """Acquire direct public URLs for a known claim."""
        tool_calls.append("acquire_sources")
        return {
            "status": "completed",
            "claim_id": claim_id,
            "results": [
                {
                    "status": "acquired",
                    "window_id": "window-2",
                    "url": urls[0],
                    "offset": offset,
                    "limit": limit,
                    "content": "IANA reserves example domains for documentation.",
                }
            ],
        }

    try:
        result = await runner.run_phase(
            phase="reference_answer_generation",
            prompt=(
                "Use Google Search once for the official purpose of example.com. Then call "
                "read_cached_source with source_id source-1 and call acquire_sources with claim_id answer "
                "and URL https://www.iana.org/help/example-domains. Return a finalized native response: "
                "unchanged disposition; proposed short answer Example Domain; a concise reader-facing answer; "
                "one answer-determining claim with claim_id answer supported by window-1 and window-2; both "
                "window IDs selected as citations; and null abandon_reason."
            ),
            config=_config(),
            agent_instruction=phase_instruction("reference_answer_generation"),
            search_enabled=True,
            function_tools=(read_cached_source, acquire_sources),
            output_schema=DomainTweakReferenceAnswerOutput,
            validate=validate_reference_answer_output,
        )
    finally:
        cleanup_credentials_file(credentials_path)

    assert result.terminal_status == "validated", _phase_result_debug(result)
    assert set(tool_calls) == {"read_cached_source", "acquire_sources"}
    event_summaries = tuple(summary for attempt in result.attempts for summary in attempt.event_summaries)
    assert sum(summary.web_search_query_count for summary in event_summaries) >= 1
    assert {name for summary in event_summaries for name in summary.function_call_names} >= {
        "read_cached_source",
        "acquire_sources",
    }


async def test_adk_live_materializes_json_string_schema_and_value_without_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials_path = _configure_adk_vertex_environment(monkeypatch)
    form_review = DomainTweakFormReview(
        form_match=True,
        reviewer_feedback="The exhaustive filtering form is preserved.",
        question_requirements=(
            DomainTweakQuestionRequirement(
                requirement_id="metric",
                category="metric_or_field_relation",
                requirement="Return every candidate that satisfies both predicates.",
                required_relation="derived_calculation",
            ),
        ),
        requirement_category_audit=tuple(
            DomainTweakRequirementCategoryAudit(
                category=category,
                present=category == "metric_or_field_relation",
                explanation=(
                    "The question filters by two predicates."
                    if category == "metric_or_field_relation"
                    else "This category is absent."
                ),
            )
            for category in DOMAIN_TWEAK_REQUIREMENT_CATEGORIES
        ),
    )
    reference_output = DomainTweakReferenceAnswerOutput(
        status="finalized",
        answer_disposition="unchanged",
        proposed_short_answer="A",
        reference_answer_text="A is the only candidate that satisfies both predicates.",
        claims=(
            DomainTweakReferenceClaim(
                claim_id="answer",
                claim="A is the complete qualifying candidate set.",
                role="answer_determining",
                evidence_window_ids=("window-1",),
                support_explanation="The acquired table establishes both predicates.",
            ),
        ),
        citation_window_ids=("window-1",),
        abandon_reason=None,
    )
    try:
        result = await DomainTweakAdkRunner().run_phase(
            phase="structured_output_materialization",
            prompt=(
                "Return materialized. The question is: Which candidates satisfy both predicates? "
                "Return only one candidates field as an array of strings containing A. Encode a fixed "
                "closed object JSON Schema in output_schema_json and the matching value in "
                "structured_output_json. Emit one field binding at candidates[] using requirement ID "
                "metric, claim ID answer, and evidence window ID window-1. Every object must require all "
                "properties and include additionalProperties false. Add no schema annotations."
            ),
            config=_config().model_copy(update={"max_retries": 0}),
            agent_instruction=phase_instruction("structured_output_materialization"),
            search_enabled=False,
            function_tools=(),
            output_schema=DomainTweakStructuredOutputMaterializationWire,
            validate=lambda text: validate_structured_output_materialization(
                text,
                question="Which candidates satisfy both predicates?",
                form_review=form_review,
                reference_output=reference_output,
            ),
        )
    finally:
        cleanup_credentials_file(credentials_path)

    assert result.terminal_status == "validated", _phase_result_debug(result)
    assert isinstance(
        result.parsed_output,
        DomainTweakStructuredOutputMaterialization,
    )
    assert result.parsed_output.structured_output == {"candidates": ["A"]}
    assert len(result.attempts) == 1
    assert all(not _tool_names(attempt) for attempt in result.attempts)


def _config() -> DomainTweakAdkRunConfig:
    return DomainTweakAdkRunConfig(
        model=os.environ.get("DOMAIN_TWEAK_ADK_LIVE_MODEL", "gemini-3.1-flash-lite"),
        max_retries=1,
        phase_timeout_seconds=600,
    )


def _tool_names(attempt: object) -> set[str]:
    event_summaries = getattr(attempt, "event_summaries", ())
    return {
        name for summary in event_summaries for name in (*summary.function_call_names, *summary.function_response_names)
    }


def _phase_result_debug(result: DomainTweakAdkPhaseResult) -> str:
    lines = [f"terminal_status={result.terminal_status}"]
    if result.error_type or result.error:
        lines.append(f"error={result.error_type}: {result.error}")
    for attempt in result.attempts:
        lines.append(
            "attempt "
            f"{attempt.attempt_index} "
            f"prompt_kind={attempt.prompt_kind} "
            f"validation_ok={attempt.validation_ok} "
            f"feedback={list(attempt.validation_feedback)} "
            f"preview={attempt.final_text_preview!r}"
        )
    return "\n".join(lines)


def _configure_adk_vertex_environment(monkeypatch: pytest.MonkeyPatch) -> str | None:
    vertex = VertexSettings()
    project = _required_env_value(vertex.gcp_project_id, "GCP_PROJECT_ID")
    location = _required_env_value(vertex.gcp_location, "GCP_LOCATION")
    credentials_b64 = _required_env_value(
        vertex.gcp_sa_credential_b64_value,
        "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64",
    )

    _credentials, credentials_path = prepare_credentials(None, credentials_b64)
    if credentials_path is not None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", credentials_path)
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", project)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", location)
    return credentials_path


def _required_env_value(value: str | None, name: str) -> str:
    if not value:
        raise AssertionError(f"{name} must be configured for the ADK live smoke")
    return value
