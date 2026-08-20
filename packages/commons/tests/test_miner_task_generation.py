from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest

from harnyx_commons.llm.provider import BaseLlmProvider, LlmProviderPort
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import (
    AbstractLlmRequest,
    GroundedLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmInputTextPart,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.miner_task_generation import (
    MinerTaskDatasetBuilder,
    MinerTaskDatasetRequest,
    MinerTaskModelSpec,
)

pytestmark = pytest.mark.anyio("asyncio")


def test_miner_task_dataset_request_validates_optional_plain_text_probability() -> None:
    """Future failure: caller-owned probabilities must be bounded without affecting legacy callers."""
    spec = MinerTaskModelSpec(provider="vertex", model="unused", temperature=None, max_output_tokens=None)
    common = {
        "batch_id": uuid4(),
        "minimum_task_total": 1,
        "generation_task_buffer": 0,
        "generation_spec": spec,
        "reference_spec": spec,
    }

    assert MinerTaskDatasetRequest(**common).plain_text_probability is None
    assert MinerTaskDatasetRequest(**common, plain_text_probability=0.7).plain_text_probability == 0.7
    with pytest.raises(ValueError):
        MinerTaskDatasetRequest(**common, plain_text_probability=-0.01)
    with pytest.raises(ValueError):
        MinerTaskDatasetRequest(**common, plain_text_probability=1.01)


class StubLlmProvider(LlmProviderPort):
    def __init__(self, generated_query_texts: tuple[str, ...]) -> None:
        self.calls: list[GroundedLlmRequest | LlmRequest] = []
        self._generated_query_texts = generated_query_texts

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:  # pragma: no cover - interface
        typed_request = cast(GroundedLlmRequest | LlmRequest, request)
        self.calls.append(typed_request)
        use_case = typed_request.use_case

        if use_case == "miner_task_dataset_generation":
            response = _response_from_text(
                response_id="dataset-1",
                text=_generation_payload_json(self._generated_query_texts),
                usage=LlmUsage(prompt_tokens=11, completion_tokens=17, total_tokens=28),
            )
            return _apply_postprocessor(request=typed_request, response=response)

        query_text = "\n".join(_text_parts(typed_request.messages[-1].content))
        response = _response_from_text(
            response_id=f"reference-{len(self.calls)}",
            text=json.dumps(
                {
                    "text": f"Reference answer for: {query_text}",
                    "citations": [
                        {
                            "url": "https://example.com/reference",
                            "note": "Grounded source",
                            "title": "Example reference",
                        }
                    ],
                }
            ),
            usage=LlmUsage(prompt_tokens=5, completion_tokens=9, total_tokens=14),
        )
        return _apply_postprocessor(request=typed_request, response=response)


class RetryingStubLlmProvider(BaseLlmProvider):
    def __init__(self, generation_payloads: tuple[tuple[str, ...], ...]) -> None:
        super().__init__(provider_label="vertex")
        self._generation_payloads = generation_payloads
        self._retry_policy = RetryPolicy(
            attempts=len(generation_payloads),
            initial_ms=0,
            max_ms=0,
            jitter=0.0,
        )
        self.requests: list[AbstractLlmRequest] = []
        self.generation_attempts = 0

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        self.requests.append(request)
        use_case = request.use_case

        if use_case == "miner_task_dataset_generation":
            async def _call(_: AbstractLlmRequest) -> LlmResponse:
                if self.generation_attempts >= len(self._generation_payloads):
                    raise AssertionError("unexpected extra generation attempt")
                payload = self._generation_payloads[self.generation_attempts]
                self.generation_attempts += 1
                return _response_from_text(
                    response_id=f"dataset-{self.generation_attempts}",
                    text=_generation_payload_json(payload),
                    usage=LlmUsage(prompt_tokens=11, completion_tokens=17, total_tokens=28),
                )

            def _always_pass_verifier(_: LlmResponse) -> tuple[bool, bool, str | None]:
                return True, False, None

            return await self._call_with_retry(
                request,
                call_coro=_call,
                verifier=_always_pass_verifier,
                policy=self._retry_policy,
            )

        query_text = "\n".join(_text_parts(request.messages[-1].content))
        response = _response_from_text(
            response_id=f"reference-{len(self.requests)}",
            text=json.dumps({"text": f"Reference answer for: {query_text}"}),
            usage=LlmUsage(prompt_tokens=5, completion_tokens=9, total_tokens=14),
        )
        return _apply_postprocessor(
            request=cast(GroundedLlmRequest | LlmRequest, request),
            response=response,
        )


async def test_miner_task_dataset_builder_returns_generic_tasks() -> None:
    provider = StubLlmProvider(
        (
            "Summarize the latest SEC guidance on stablecoin reserves.",
            "Compare Claude 4.5 Sonnet and Gemini 2.5 Flash for coding tasks.",
            "Explain how the Artemis program differs from Apollo.",
        )
    )
    builder = MinerTaskDatasetBuilder(
        generation_llm=provider,
        reference_llm=provider,
        clock=lambda: datetime(2026, 3, 6, tzinfo=UTC),
    )

    tasks, generation_usage, reference_usage = await builder.build_with_usage(
        MinerTaskDatasetRequest(
            batch_id=uuid4(),
            minimum_task_total=2,
            generation_task_buffer=1,
            generation_spec=MinerTaskModelSpec(
                provider="vertex",
                model="gemini-test",
                temperature=0.2,
                max_output_tokens=256,
            ),
            reference_spec=MinerTaskModelSpec(
                provider="vertex",
                model="gemini-reference",
                temperature=0.0,
                max_output_tokens=512,
            ),
        )
    )

    assert len(tasks) == 2
    assert len(generation_usage) == 1
    assert len(reference_usage) == 2
    assert tasks[0].query.text == "Summarize the latest SEC guidance on stablecoin reserves."
    assert tasks[1].query.text == "Compare Claude 4.5 Sonnet and Gemini 2.5 Flash for coding tasks."
    assert tasks[0].reference_answer.text.startswith("Reference answer for:")
    assert tasks[1].reference_answer.text.startswith("Reference answer for:")
    assert tasks[0].reference_answer.citations is not None
    assert tasks[0].reference_answer.citations[0].url == "https://example.com/reference"
    assert tasks[0].reference_answer.citations[0].title == "Example reference"
    assert tasks[0].reference_answer.citations[0].note == "Grounded source"

    generation_request = provider.calls[0]
    assert generation_request.provider == "vertex"
    assert generation_request.model == "gemini-test"
    assert generation_request.grounded is False
    assert generation_request.output_mode == "structured"

    generation_system_prompt = "\n".join(_text_parts(generation_request.messages[0].content))
    generation_prompt = "\n".join(_text_parts(generation_request.messages[-1].content))
    assert "at least one independent-source synthesis move" in generation_system_prompt
    assert "no single evidence item can answer completely" in generation_system_prompt
    assert "simple single-entity lookups" in generation_system_prompt
    assert "top search result" in generation_system_prompt
    assert "Generate exactly 3 user queries." in generation_prompt
    assert "definite affirmative answer already documented" in generation_prompt
    assert "independent-source synthesis" in generation_prompt
    assert "answered shallowly from one evidence item" in generation_prompt
    assert "top result can provide every requested field" in generation_prompt
    assert "At most one task may be a premise-correction task." in generation_prompt
    assert "Before including each query" in generation_prompt
    assert "fragile status words" in generation_prompt
    assert "old-memory 'most recent' questions" in generation_prompt
    assert "rubric" not in generation_prompt.lower()

    reference_request = provider.calls[1]
    reference_prompt = "\n".join(_text_parts(reference_request.messages[0].content))
    reference_user_prompt = "\n".join(_text_parts(reference_request.messages[-1].content))
    assert "Return exactly one JSON object" in reference_prompt
    assert "citation note is scorer-visible evidence" in reference_prompt
    assert "claim-bearing note" in reference_prompt
    assert "Every load-bearing claim" in reference_prompt
    assert "query-required subclaims" in reference_prompt
    assert "independent sources" in reference_prompt
    assert "one broad citation" in reference_prompt
    assert "same reporting period and basis" in reference_prompt
    assert "primary or official sources" in reference_prompt
    assert "grounds the rejection itself" in reference_prompt
    assert "Never return a negative" in reference_prompt
    assert "False-premise examples" in reference_prompt
    assert "Current timestamp: 2026-03-06T00:00:00+00:00" in reference_user_prompt
    assert "Treat this as a deep-research reference answer." in reference_user_prompt
    assert "Verify the named event" in reference_user_prompt
    assert "Every factual sentence in text" in reference_user_prompt
    assert "null or empty citations" in reference_user_prompt


async def test_miner_task_dataset_builder_dismisses_incomplete_reference_citations() -> None:
    class IncompleteCitationStubLlmProvider(LlmProviderPort):
        def __init__(self) -> None:
            self.calls: list[GroundedLlmRequest | LlmRequest] = []

        async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
            typed_request = cast(GroundedLlmRequest | LlmRequest, request)
            self.calls.append(typed_request)
            use_case = typed_request.use_case
            if use_case == "miner_task_dataset_generation":
                response = _response_from_text(
                    response_id="dataset-1",
                    text=_generation_payload_json(("What happened?",)),
                    usage=LlmUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )
                return _apply_postprocessor(request=typed_request, response=response)

            response = _response_from_text(
                response_id="reference-1",
                text=json.dumps(
                    {
                        "text": "Reference answer for: What happened?",
                        "citations": [
                            {
                                "url": "https://example.com/complete",
                                "title": "Complete source",
                                "note": "Complete support",
                            },
                            {
                                "url": "https://example.com/missing-note",
                                "title": "Missing note source",
                            },
                            {
                                "url": "https://example.com/missing-title",
                                "note": "Missing title support",
                            },
                        ],
                    }
                ),
                usage=LlmUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
            return _apply_postprocessor(request=typed_request, response=response)

        async def aclose(self) -> None:
            return None

    provider = IncompleteCitationStubLlmProvider()
    builder = MinerTaskDatasetBuilder(
        generation_llm=provider,
        reference_llm=provider,
        clock=lambda: datetime(2026, 3, 6, tzinfo=UTC),
    )

    tasks = await builder.build(
        MinerTaskDatasetRequest(
            batch_id=uuid4(),
            created_at=datetime(2026, 3, 6, tzinfo=UTC),
            minimum_task_total=1,
            generation_task_buffer=0,
            generation_spec=MinerTaskModelSpec(
                provider="vertex",
                model="gemini-test",
                temperature=0.2,
                max_output_tokens=256,
            ),
            reference_spec=MinerTaskModelSpec(
                provider="vertex",
                model="gemini-reference",
                temperature=0.0,
                max_output_tokens=512,
            ),
        )
    )

    citations = tasks[0].reference_answer.citations
    assert citations is not None
    assert len(citations) == 1
    assert citations[0].url == "https://example.com/complete"
    assert citations[0].title == "Complete source"
    assert citations[0].note == "Complete support"


async def test_miner_task_dataset_builder_rejects_bedrock_reference_generation() -> None:
    provider = StubLlmProvider(("What changed at AWS Bedrock this quarter?",))
    builder = MinerTaskDatasetBuilder(
        generation_llm=provider,
        reference_llm=provider,
        clock=lambda: datetime(2026, 3, 6, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="grounded mode not supported for provider 'bedrock'"):
        await builder.build(
            MinerTaskDatasetRequest(
                batch_id=uuid4(),
                created_at=datetime(2026, 3, 6, tzinfo=UTC),
                minimum_task_total=1,
                generation_task_buffer=0,
                generation_spec=MinerTaskModelSpec(
                    provider="vertex",
                    model="gemini-test",
                    temperature=0.2,
                    max_output_tokens=256,
                ),
                reference_spec=MinerTaskModelSpec(
                    provider="bedrock",
                    model="openai.gpt-oss-20b-1:0",
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )
        )


async def test_miner_task_dataset_builder_accepts_buffered_underfill_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = StubLlmProvider(
        (
            "Summarize the latest SEC guidance on stablecoin reserves.",
            "Compare Claude 4.5 Sonnet and Gemini 2.5 Flash for coding tasks.",
        )
    )
    builder = MinerTaskDatasetBuilder(
        generation_llm=provider,
        reference_llm=provider,
        clock=lambda: datetime(2026, 3, 6, tzinfo=UTC),
    )
    caplog.set_level(logging.WARNING, logger="harnyx_commons.miner_task_generation")

    tasks = await builder.build(
        MinerTaskDatasetRequest(
            batch_id=uuid4(),
            created_at=datetime(2026, 3, 6, tzinfo=UTC),
            minimum_task_total=2,
            generation_task_buffer=1,
            generation_spec=MinerTaskModelSpec(
                provider="vertex",
                model="gemini-test",
                temperature=0.2,
                max_output_tokens=256,
            ),
            reference_spec=MinerTaskModelSpec(
                provider="vertex",
                model="gemini-reference",
                temperature=0.0,
                max_output_tokens=512,
            ),
        )
    )

    assert len(tasks) == 2
    assert "miner-task dataset generated fewer tasks than requested" in caplog.text


async def test_miner_task_dataset_builder_accepts_buffered_duplicates_after_deduplication(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = StubLlmProvider(
        (
            "Summarize the latest SEC guidance on stablecoin reserves.",
            "Compare Claude 4.5 Sonnet and Gemini 2.5 Flash for coding tasks.",
            "Compare Claude 4.5 Sonnet and Gemini 2.5 Flash for coding tasks.",
            "Explain how the Artemis program differs from Apollo.",
        )
    )
    builder = MinerTaskDatasetBuilder(
        generation_llm=provider,
        reference_llm=provider,
        clock=lambda: datetime(2026, 3, 6, tzinfo=UTC),
    )
    caplog.set_level(logging.WARNING, logger="harnyx_commons.miner_task_generation")

    tasks = await builder.build(
        MinerTaskDatasetRequest(
            batch_id=uuid4(),
            created_at=datetime(2026, 3, 6, tzinfo=UTC),
            minimum_task_total=3,
            generation_task_buffer=1,
            generation_spec=MinerTaskModelSpec(
                provider="vertex",
                model="gemini-test",
                temperature=0.2,
                max_output_tokens=256,
            ),
            reference_spec=MinerTaskModelSpec(
                provider="vertex",
                model="gemini-reference",
                temperature=0.0,
                max_output_tokens=512,
            ),
        )
    )

    assert tuple(task.query.text for task in tasks) == (
        "Summarize the latest SEC guidance on stablecoin reserves.",
        "Compare Claude 4.5 Sonnet and Gemini 2.5 Flash for coding tasks.",
        "Explain how the Artemis program differs from Apollo.",
    )
    assert "miner-task dataset dropped duplicate generated tasks" in caplog.text
    assert "miner-task dataset generated fewer tasks than requested" in caplog.text


async def test_miner_task_dataset_builder_retries_duplicate_heavy_under_minimum() -> None:
    provider = RetryingStubLlmProvider(
        (
            (
                "Summarize the latest SEC guidance on stablecoin reserves.",
                "Summarize the latest SEC guidance on stablecoin reserves.",
                "Compare Claude 4.5 Sonnet and Gemini 2.5 Flash for coding tasks.",
                "Compare Claude 4.5 Sonnet and Gemini 2.5 Flash for coding tasks.",
            ),
            (
                "Summarize the latest SEC guidance on stablecoin reserves.",
                "Compare Claude 4.5 Sonnet and Gemini 2.5 Flash for coding tasks.",
                "Explain how the Artemis program differs from Apollo.",
                "What changed in the latest SEC guidance on stablecoin reserves?",
            ),
        )
    )
    builder = MinerTaskDatasetBuilder(
        generation_llm=provider,
        reference_llm=provider,
        clock=lambda: datetime(2026, 3, 6, tzinfo=UTC),
    )

    tasks = await builder.build(
        MinerTaskDatasetRequest(
            batch_id=uuid4(),
            created_at=datetime(2026, 3, 6, tzinfo=UTC),
            minimum_task_total=3,
            generation_task_buffer=1,
            generation_spec=MinerTaskModelSpec(
                provider="vertex",
                model="gemini-test",
                temperature=0.2,
                max_output_tokens=256,
            ),
            reference_spec=MinerTaskModelSpec(
                provider="vertex",
                model="gemini-reference",
                temperature=0.0,
                max_output_tokens=512,
            ),
        )
    )

    assert provider.generation_attempts == 2
    assert tuple(task.query.text for task in tasks) == (
        "Summarize the latest SEC guidance on stablecoin reserves.",
        "Compare Claude 4.5 Sonnet and Gemini 2.5 Flash for coding tasks.",
        "Explain how the Artemis program differs from Apollo.",
    )


async def test_miner_task_dataset_builder_rejects_below_minimum_task_total() -> None:
    provider = RetryingStubLlmProvider(
        (
            (
                "Summarize the latest SEC guidance on stablecoin reserves.",
                "Summarize the latest SEC guidance on stablecoin reserves.",
                "Summarize the latest SEC guidance on stablecoin reserves.",
            ),
        )
    )
    builder = MinerTaskDatasetBuilder(
        generation_llm=provider,
        reference_llm=provider,
        clock=lambda: datetime(2026, 3, 6, tzinfo=UTC),
    )

    with pytest.raises(RuntimeError, match="generated unique task count below minimum_task_total"):
        await builder.build(
            MinerTaskDatasetRequest(
                batch_id=uuid4(),
                created_at=datetime(2026, 3, 6, tzinfo=UTC),
                minimum_task_total=2,
                generation_task_buffer=1,
                generation_spec=MinerTaskModelSpec(
                    provider="vertex",
                    model="gemini-test",
                    temperature=0.2,
                    max_output_tokens=256,
                ),
                reference_spec=MinerTaskModelSpec(
                    provider="vertex",
                    model="gemini-reference",
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )
        )


class BlankReferenceStubLlmProvider(LlmProviderPort):
    def __init__(self) -> None:
        self.calls: list[GroundedLlmRequest | LlmRequest] = []

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        typed_request = cast(GroundedLlmRequest | LlmRequest, request)
        self.calls.append(typed_request)
        use_case = typed_request.use_case

        if use_case == "miner_task_dataset_generation":
            response = _response_from_text(
                response_id="dataset-1",
                text=_generation_payload_json(("What happened?",)),
                usage=LlmUsage(prompt_tokens=11, completion_tokens=17, total_tokens=28),
            )
            return _apply_postprocessor(request=typed_request, response=response)

        response = _response_from_text(
            response_id="reference-1",
            text=json.dumps({"text": "   "}),
            usage=LlmUsage(prompt_tokens=5, completion_tokens=9, total_tokens=14),
        )
        return _apply_postprocessor(request=typed_request, response=response)


async def test_miner_task_dataset_builder_rejects_blank_non_vertex_reference_answers() -> None:
    provider = BlankReferenceStubLlmProvider()
    builder = MinerTaskDatasetBuilder(
        generation_llm=provider,
        reference_llm=provider,
        clock=lambda: datetime(2026, 3, 6, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="grounded mode not supported for provider 'chutes'"):
        await builder.build(
            MinerTaskDatasetRequest(
                batch_id=uuid4(),
                created_at=datetime(2026, 3, 6, tzinfo=UTC),
                minimum_task_total=1,
                generation_task_buffer=0,
                generation_spec=MinerTaskModelSpec(
                    provider="vertex",
                    model="gemini-test",
                    temperature=0.2,
                    max_output_tokens=256,
                ),
                reference_spec=MinerTaskModelSpec(
                    provider="chutes",
                    model="non-vertex-reference",
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )
        )

    assert len(provider.calls) == 1


def _generation_payload_json(texts: tuple[str, ...]) -> str:
    return json.dumps({"tasks": [{"text": text} for text in texts]})


def _response_from_text(
    *,
    response_id: str,
    text: str,
    usage: LlmUsage,
) -> LlmResponse:
    return LlmResponse(
        id=response_id,
        choices=(
            LlmChoice(
                index=0,
                message=LlmChoiceMessage(
                    role="assistant",
                    content=(LlmMessageContentPart(type="text", text=text),),
                ),
            ),
        ),
        usage=usage,
    )


def _apply_postprocessor(
    *,
    request: GroundedLlmRequest | LlmRequest,
    response: LlmResponse,
) -> LlmResponse:
    if request.postprocessor is None:
        return response
    result = request.postprocessor(response)
    if not result.ok:
        raise RuntimeError(result.reason or "stub postprocessor failed")
    return LlmResponse(
        id=response.id,
        choices=response.choices,
        usage=response.usage,
        metadata=response.metadata,
        postprocessed=result.processed,
        finish_reason=response.finish_reason,
    )


def _text_parts(parts: Sequence[object]) -> list[str]:
    fragments: list[str] = []
    for part in parts:
        if isinstance(part, LlmInputTextPart):
            fragments.append(part.text or "")
    return fragments
