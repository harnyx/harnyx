"""Protect input identity and reject unusable or unrelated ranked output."""

import pytest
from pydantic import ValidationError

from harnyx_miner_sdk.tools.rerank_models import RerankRequest, RerankResponse, validate_rerank_results


def request(**changes: object) -> RerankRequest:
    return RerankRequest.model_validate(
        {
            "provider": "openrouter",
            "model": "cohere/rerank-v3.5",
            "query": "capital?",
            "documents": [" Berlin ", "Paris"],
            **changes,
        }
    )


def test_rerank_preserves_document_text_and_accepts_top_n_above_input_count() -> None:
    invocation = request(top_n=10)
    assert invocation.documents == [" Berlin ", "Paris"]
    response = RerankResponse.model_validate(
        {
            "provider": "openrouter",
            "model": invocation.model,
            "results": [
                {"index": 1, "relevance_score": 0.9, "document": "Paris"},
                {"index": 0, "relevance_score": 0.1, "document": " Berlin "},
            ],
        }
    )
    validate_rerank_results(invocation, response)


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "nvidia/rerank:free"},
        {"provider": "chutes"},
        {"documents": []},
        {"documents": ["  "]},
        {"query": "  "},
        {"top_n": 0},
        {"top_n": True},
        {"timeout": float("inf")},
        {"timeout": -1.0},
        {"documents": [{"image": "https://example.com"}]},
        {"provider_extra": {"provider": {"unexpected": True}}},
    ],
)
def test_rerank_rejects_invalid_or_free_requests(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        request(**changes)


@pytest.mark.parametrize(
    "results",
    [
        [{"index": 0, "relevance_score": 0.9, "document": " Berlin "}] * 2,
        [{"index": 2, "relevance_score": 0.9, "document": "Paris"}],
        [{"index": 1, "relevance_score": 0.9, "document": "altered"}],
        [{"index": 1, "relevance_score": float("nan"), "document": "Paris"}],
        [
            {"index": 1, "relevance_score": 0.1, "document": "Paris"},
            {"index": 0, "relevance_score": 0.9, "document": " Berlin "},
        ],
    ],
)
def test_rerank_rejects_duplicate_unrelated_or_unsorted_output(results: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError):
        response = RerankResponse.model_validate(
            {"provider": "openrouter", "model": request().model, "results": results}
        )
        validate_rerank_results(request(), response)
