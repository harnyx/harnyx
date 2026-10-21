"""Live coverage of the OpenRouter embeddings API."""

import pytest

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.providers.openrouter import OpenRouterEmbeddingClient
from harnyx_commons.tools.embedding_models import QWEN3_OPENROUTER_EMBEDDING_MODEL

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


async def test_openrouter_embedding_client_invokes_qwen3_8b_live() -> None:
    settings = LlmSettings()
    assert settings.openrouter_api_key_value, "OPENROUTER_API_KEY must be configured"

    client = OpenRouterEmbeddingClient(
        model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
        api_key=settings.openrouter_api_key,
        dimensions=32,
        timeout_seconds=180.0,
    )
    try:
        response = await client.embed_many(("hello",))
    finally:
        await client.aclose()

    assert len(response.vectors) == 1
    assert len(response.vectors[0]) == 32
    assert response.usage is not None
    assert response.usage.prompt_tokens is not None
    assert response.usage.prompt_tokens > 0
    assert response.model
    if response.usage.cost is not None:
        assert isinstance(response.usage.cost, int | float)
        assert response.usage.cost >= 0.0
    if response.id is not None:
        assert response.id

