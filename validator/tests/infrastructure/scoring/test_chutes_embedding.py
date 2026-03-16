from __future__ import annotations

import httpx
import pytest

from harnyx_validator.infrastructure.scoring.chutes_embedding import ChutesTextEmbeddingClient

pytestmark = pytest.mark.anyio("asyncio")


async def test_chutes_text_embedding_client_posts_openai_compatible_embeddings_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["headers"] = dict(request.headers)
        captured["json"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "embedding": [0.25, 0.5, 0.75],
                        "index": 0,
                        "object": "embedding",
                    }
                ]
            },
        )

    client = ChutesTextEmbeddingClient(
        model="text-embedding-3-small",
        client=httpx.AsyncClient(base_url="https://llm.chutes.ai", transport=httpx.MockTransport(handler)),
        api_key="test-key",
        dimensions=3,
    )

    vector = await client.embed("hello world")

    assert vector == (0.25, 0.5, 0.75)
    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/embeddings"
    assert '"model":"text-embedding-3-small"' in str(captured["json"])
    assert '"input":"hello world"' in str(captured["json"])
