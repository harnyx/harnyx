"""Provider-aware scoring embedding client construction."""

from __future__ import annotations

from harnyx_commons.llm.provider_types import BEDROCK_PROVIDER, CHUTES_PROVIDER, VERTEX_MAAS_PROVIDER, LlmProviderName
from harnyx_commons.llm.providers.chutes import ChutesTextEmbeddingClient
from harnyx_validator.infrastructure.scoring.vertex_embedding import LazyVertexTextEmbeddingClient

ScoringEmbeddingClient = ChutesTextEmbeddingClient | LazyVertexTextEmbeddingClient


def create_scoring_embedding_client(
    *,
    provider_name: LlmProviderName,
    vertex_model: str,
    chutes_model: str,
    chutes_api_key: str | None,
    scoring_timeout_seconds: float,
    vertex_project: str | None,
    vertex_location: str | None,
    vertex_maas_location: str | None,
    vertex_service_account_b64: str | None,
    vertex_timeout_seconds: float,
    chutes_base_url: str | None = None,
) -> ScoringEmbeddingClient:
    if provider_name == BEDROCK_PROVIDER:
        raise ValueError("SCORING_LLM_PROVIDER='bedrock' is not supported")
    if provider_name == CHUTES_PROVIDER:
        if not chutes_api_key:
            raise RuntimeError("CHUTES_API_KEY must be configured for chutes scoring embeddings")
        return ChutesTextEmbeddingClient(
            model=chutes_model,
            api_key=chutes_api_key,
            base_url=chutes_base_url,
            timeout_seconds=scoring_timeout_seconds,
        )

    location = vertex_location
    if provider_name == VERTEX_MAAS_PROVIDER:
        location = vertex_maas_location
    return LazyVertexTextEmbeddingClient(
        project=vertex_project,
        location=location,
        service_account_b64=vertex_service_account_b64,
        model=vertex_model,
        timeout_seconds=vertex_timeout_seconds,
    )


__all__ = ["ScoringEmbeddingClient", "create_scoring_embedding_client"]
