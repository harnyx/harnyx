"""Typed text reranking contracts shared by miner and hosted runtimes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from harnyx_miner_sdk.tools.llm_provider_extra import OpenRouterExtra
from harnyx_miner_sdk.tools.openrouter_models import validate_paid_openrouter_model
from harnyx_miner_sdk.tools.types import ToolInvocationTimeout


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    provider: Literal["openrouter"]
    model: str = Field(min_length=1)
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1)
    top_n: int | None = Field(default=None, ge=1)
    provider_extra: OpenRouterExtra | None = None
    timeout: ToolInvocationTimeout | None = None

    @field_validator("model")
    @classmethod
    def _model(cls, value: str) -> str:
        return validate_paid_openrouter_model(value)

    @field_validator("query")
    @classmethod
    def _query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rerank query must not be blank")
        return value

    @field_validator("documents")
    @classmethod
    def _documents(cls, value: list[str]) -> list[str]:
        if any(not document.strip() for document in value):
            raise ValueError("rerank documents must not be blank")
        return value


class RerankResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    index: int = Field(ge=0)
    relevance_score: float = Field(allow_inf_nan=False)
    document: str


class RerankUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    search_units: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class RerankResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    provider: Literal["openrouter"]
    model: str
    results: list[RerankResult] = Field(min_length=1)
    usage: RerankUsage | None = None


def validate_rerank_results(request: RerankRequest, response: RerankResponse) -> None:
    expected_count = min(request.top_n or len(request.documents), len(request.documents))
    if response.model != request.model or len(response.results) != expected_count:
        raise ValueError("rerank response model or result count does not match request")
    indices = [result.index for result in response.results]
    if len(set(indices)) != len(indices):
        raise ValueError("rerank response contains duplicate document indices")
    previous_score = float("inf")
    for result in response.results:
        if result.index >= len(request.documents) or result.document != request.documents[result.index]:
            raise ValueError("rerank response document does not match its original index")
        if result.relevance_score > previous_score:
            raise ValueError("rerank results are not in descending relevance order")
        previous_score = result.relevance_score


__all__ = ["RerankRequest", "RerankResponse", "RerankResult", "RerankUsage", "validate_rerank_results"]
