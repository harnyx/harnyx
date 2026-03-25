from __future__ import annotations

import pytest

from harnyx_commons.clients import CHUTES
from harnyx_commons.llm.providers.chutes import ChutesTextEmbeddingClient
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.settings import Settings

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


async def test_chutes_embedding_live_returns_non_empty_float_vector() -> None:
    settings = Settings.load()
    api_key = settings.llm.chutes_api_key_value

    assert api_key, "CHUTES_API_KEY must be configured"

    client = ChutesTextEmbeddingClient(
        model=bootstrap._SCORING_CHUTES_EMBEDDING_MODEL,
        api_key=api_key,
        timeout_seconds=float(CHUTES.timeout_seconds),
    )

    try:
        vector = await client.embed("hello world")
    finally:
        await client.aclose()

    assert vector
    assert all(isinstance(value, float) for value in vector)
