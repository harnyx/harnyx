from __future__ import annotations

import pytest

from harnyx_commons.llm.providers.chutes import ChutesEmbeddingUsage, ChutesTextEmbeddingResponse
from harnyx_commons.llm.providers.openrouter import OpenRouterEmbeddingResponse, OpenRouterEmbeddingUsage
from harnyx_commons.tools.embedding_models import (
    QWEN3_CHUTES_EMBEDDING_MODEL,
    QWEN3_OPENROUTER_EMBEDDING_MODEL,
    EmbedTextRequest,
    parse_miner_selected_embedding_provider_model,
)
from harnyx_commons.tools.invocation_clients import ChutesEmbeddingProvider, OpenRouterEmbeddingProvider

pytestmark = pytest.mark.anyio("asyncio")


def test_miner_selected_embedding_provider_model_sets_are_provider_namespaces() -> None:
    assert parse_miner_selected_embedding_provider_model(
        provider="chutes",
        model=QWEN3_CHUTES_EMBEDDING_MODEL,
    ).model == QWEN3_CHUTES_EMBEDDING_MODEL
    assert parse_miner_selected_embedding_provider_model(
        provider="openrouter",
        model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
    ).model == QWEN3_OPENROUTER_EMBEDDING_MODEL
    with pytest.raises(ValueError, match="not supported"):
        parse_miner_selected_embedding_provider_model(
            provider="chutes",
            model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
        )
    with pytest.raises(ValueError, match="not supported"):
        parse_miner_selected_embedding_provider_model(
            provider="openrouter",
            model=QWEN3_CHUTES_EMBEDDING_MODEL,
        )
async def test_chutes_embedding_provider_formats_query_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        async def embed_many(self, texts: tuple[str, ...], **_: object) -> ChutesTextEmbeddingResponse:
            captured["texts"] = texts
            return ChutesTextEmbeddingResponse(
                vectors=((0.1, 0.2, 0.3),),
                usage=ChutesEmbeddingUsage(prompt_tokens=8, total_tokens=8),
            )

    provider = ChutesEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    result = await provider.embed_text(
        EmbedTextRequest(
            provider="chutes",
            model=QWEN3_CHUTES_EMBEDDING_MODEL,
            texts=("find subnet incentives",),
            input_type="query",
            instruction="Given a web search query, retrieve relevant passages that answer the query",
        )
    )

    assert captured["texts"] == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query:find subnet incentives",
    )
    assert result.response.data[0].embedding == [0.1, 0.2, 0.3]
    assert result.actual_cost_provider == "chutes"
    assert result.actual_cost_evidence["usd_per_second"] == pytest.approx(0.0005)


@pytest.mark.parametrize(
    ("requested_timeout", "expected_provider_timeout"),
    [(400.0, 410.0), (5.0, 300.0)],
)
async def test_chutes_embedding_provider_applies_effective_request_timeout_to_model_client(
    monkeypatch: pytest.MonkeyPatch,
    requested_timeout: float,
    expected_provider_timeout: float,
) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        async def embed_many(
            self,
            texts: tuple[str, ...],
            **kwargs: object,
        ) -> ChutesTextEmbeddingResponse:
            _ = texts
            captured.update(kwargs)
            return ChutesTextEmbeddingResponse(vectors=((0.1, 0.2, 0.3),))

    provider = ChutesEmbeddingProvider(api_key="test-key", timeout_seconds=300.0)

    def client_for(**kwargs: object) -> _FakeClient:
        _ = kwargs
        return _FakeClient()

    monkeypatch.setattr(provider, "_client_for", client_for)

    await provider.embed_text(
        EmbedTextRequest(
            provider="chutes",
            model=QWEN3_CHUTES_EMBEDDING_MODEL,
            texts=("timeout parity",),
            input_type="document",
            timeout=requested_timeout,
        )
    )

    assert captured["timeout_seconds"] == expected_provider_timeout


async def test_chutes_embedding_provider_leaves_document_text_unformatted(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        async def embed_many(self, texts: tuple[str, ...], **_: object) -> ChutesTextEmbeddingResponse:
            captured["texts"] = texts
            return ChutesTextEmbeddingResponse(
                vectors=((0.4, 0.5, 0.6),),
                usage=ChutesEmbeddingUsage(prompt_tokens=6, total_tokens=6),
            )

    provider = ChutesEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    await provider.embed_text(
        EmbedTextRequest(
            provider="chutes",
            model=QWEN3_CHUTES_EMBEDDING_MODEL,
            texts=("The subnet rewards useful miner answers.",),
            input_type="document",
        )
    )

    assert captured["texts"] == ("The subnet rewards useful miner answers.",)


async def test_chutes_embedding_provider_allows_missing_usage_tokens_when_elapsed_second_priced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        async def embed_many(self, texts: tuple[str, ...], **_: object) -> ChutesTextEmbeddingResponse:
            _ = texts
            return ChutesTextEmbeddingResponse(vectors=((0.4, 0.5, 0.6),))

    provider = ChutesEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    result = await provider.embed_text(
        EmbedTextRequest(
            provider="chutes",
            model=QWEN3_CHUTES_EMBEDDING_MODEL,
            texts=("The subnet rewards useful miner answers.",),
            input_type="document",
        )
    )

    assert result.actual_cost_provider == "chutes"
    assert result.actual_cost_evidence["usd_per_second"] == pytest.approx(0.0005)
    assert "input_tokens" not in result.actual_cost_evidence


async def test_openrouter_embedding_provider_posts_native_model_and_settles_static_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        async def embed_many(self, texts: tuple[str, ...], **_: object) -> OpenRouterEmbeddingResponse:
            captured["texts"] = texts
            return OpenRouterEmbeddingResponse(
                vectors=((0.7, 0.8, 0.9),),
                model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
                usage=OpenRouterEmbeddingUsage(prompt_tokens=12, total_tokens=12),
            )

    provider = OpenRouterEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    result = await provider.embed_text(
        EmbedTextRequest(
            provider="openrouter",
            model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
            texts=("find subnet incentives",),
            input_type="query",
        )
    )

    assert captured["texts"] == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query:find subnet incentives",
    )
    assert result.actual_cost_provider == "openrouter"
    assert result.actual_cost_evidence["input_tokens"] == 12
    assert result.actual_cost_evidence["input_per_million"] == pytest.approx(0.01)
    assert result.actual_cost_evidence["provider_cost_status"] == "missing"


async def test_openrouter_embedding_provider_applies_request_timeout_to_model_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        async def embed_many(
            self,
            texts: tuple[str, ...],
            **kwargs: object,
        ) -> OpenRouterEmbeddingResponse:
            _ = texts
            captured.update(kwargs)
            return OpenRouterEmbeddingResponse(
                vectors=((0.7, 0.8, 0.9),),
                model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
            )

    provider = OpenRouterEmbeddingProvider(api_key="test-key", timeout_seconds=300.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    await provider.embed_text(
        EmbedTextRequest(
            provider="openrouter",
            model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
            texts=("timeout parity",),
            input_type="document",
            timeout=400.0,
        )
    )

    assert captured["timeout_seconds"] == 410.0


async def test_openrouter_embedding_provider_prefers_provider_returned_cost_and_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        async def embed_many(self, texts: tuple[str, ...], **_: object) -> OpenRouterEmbeddingResponse:
            _ = texts
            return OpenRouterEmbeddingResponse(
                vectors=((0.7, 0.8, 0.9),),
                usage=OpenRouterEmbeddingUsage(
                    prompt_tokens=12,
                    total_tokens=12,
                    cost=0.0042,
                    cost_details={"upstream_inference_cost": 0.004},
                ),
                id="gen-emb-1",
                model="Qwen/Qwen3-Embedding-8B",
            )

    provider = OpenRouterEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    result = await provider.embed_text(
        EmbedTextRequest(
            provider="openrouter",
            model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
            texts=("find subnet incentives",),
            input_type="query",
        )
    )

    assert result.actual_cost_usd == pytest.approx(0.0042)
    assert result.actual_cost_provider == "openrouter"
    assert result.actual_cost_evidence == {
        "settlement_source": "provider_returned",
        "pricing_origin": "openrouter_embedding_usage_cost",
        "provider_cost_details": {"upstream_inference_cost": 0.004},
        "upstream_model": "Qwen/Qwen3-Embedding-8B",
        "provider_request_id": "gen-emb-1",
        "provider": "openrouter",
        "model": QWEN3_OPENROUTER_EMBEDDING_MODEL,
        "input_type": "query",
        "text_count": 1,
    }


async def test_openrouter_embedding_provider_falls_back_for_malformed_provider_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        async def embed_many(self, texts: tuple[str, ...], **_: object) -> OpenRouterEmbeddingResponse:
            _ = texts
            return OpenRouterEmbeddingResponse(
                vectors=((0.7, 0.8, 0.9),),
                model="Qwen/Qwen3-Embedding-8B",
                usage=OpenRouterEmbeddingUsage(prompt_tokens=12, total_tokens=12, cost="invalid"),
            )

    provider = OpenRouterEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    result = await provider.embed_text(
        EmbedTextRequest(
            provider="openrouter",
            model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
            texts=("find subnet incentives",),
            input_type="query",
        )
    )

    assert result.actual_cost_evidence["settlement_source"] == "static_pricing"
    assert result.actual_cost_evidence["provider_cost_status"] == "malformed"
    assert result.actual_cost_evidence["upstream_model"] == "Qwen/Qwen3-Embedding-8B"


async def test_openrouter_embedding_provider_forwards_provider_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        async def embed_many(
            self,
            texts: tuple[str, ...],
            *,
            extra: dict[str, object] | None = None,
            **_: object,
        ) -> OpenRouterEmbeddingResponse:
            captured["texts"] = texts
            captured["extra"] = extra
            return OpenRouterEmbeddingResponse(
                vectors=((0.7, 0.8, 0.9),),
                model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
                usage=OpenRouterEmbeddingUsage(prompt_tokens=12, total_tokens=12),
            )

    provider = OpenRouterEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    result = await provider.embed_text(
        EmbedTextRequest(
            provider="openrouter",
            model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
            texts=("find subnet incentives",),
            input_type="query",
            provider_extra={"provider": {"only": ["nebius"], "allow_fallbacks": False}},
        )
    )

    assert captured["texts"] == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query:find subnet incentives",
    )
    assert captured["extra"] == {"provider": {"only": ["nebius"], "allow_fallbacks": False}}
    assert result.actual_cost_provider == "openrouter"


async def test_openrouter_embedding_provider_settles_zero_token_cache_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        async def embed_many(self, texts: tuple[str, ...], **_: object) -> OpenRouterEmbeddingResponse:
            _ = texts
            return OpenRouterEmbeddingResponse(
                vectors=((0.7, 0.8, 0.9),),
                model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
                usage=OpenRouterEmbeddingUsage(prompt_tokens=0, total_tokens=0),
            )

    provider = OpenRouterEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    result = await provider.embed_text(
        EmbedTextRequest(
            provider="openrouter",
            model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
            texts=("find subnet incentives",),
            input_type="query",
        )
    )

    assert result.actual_cost_provider == "openrouter"
    assert result.actual_cost_usd == 0.0
    assert result.actual_cost_evidence["input_tokens"] == 0
    assert result.response.usage is not None
    assert result.response.usage.prompt_tokens == 0
    assert result.response.usage.total_tokens == 0


async def test_openrouter_embedding_provider_returns_vectors_when_usage_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        async def embed_many(self, texts: tuple[str, ...], **_: object) -> OpenRouterEmbeddingResponse:
            _ = texts
            return OpenRouterEmbeddingResponse(
                vectors=((0.7, 0.8, 0.9),),
                model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
                id="gen-emb-unavailable",
            )

    provider = OpenRouterEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    result = await provider.embed_text(
        EmbedTextRequest(
            provider="openrouter",
            model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
            texts=("find subnet incentives",),
            input_type="query",
        )
    )

    assert result.response.data[0].embedding == [0.7, 0.8, 0.9]
    assert result.response.usage is None
    assert result.actual_cost_usd is None
    assert result.actual_cost_provider == "openrouter"
    assert result.actual_cost_evidence == {
        "settlement_source": "unavailable",
        "pricing_origin": "unavailable",
        "provider_cost_status": "missing",
        "usage_status": "missing",
        "upstream_model": QWEN3_OPENROUTER_EMBEDDING_MODEL,
        "provider_request_id": "gen-emb-unavailable",
        "provider": "openrouter",
        "model": QWEN3_OPENROUTER_EMBEDDING_MODEL,
        "input_type": "query",
        "text_count": 1,
    }


async def test_openrouter_embedding_provider_uses_provider_cost_without_usage_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        async def embed_many(self, texts: tuple[str, ...], **_: object) -> OpenRouterEmbeddingResponse:
            _ = texts
            return OpenRouterEmbeddingResponse(
                vectors=((0.7, 0.8, 0.9),),
                model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
                usage=OpenRouterEmbeddingUsage(cost=0.0042),
            )

    provider = OpenRouterEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    result = await provider.embed_text(
        EmbedTextRequest(
            provider="openrouter",
            model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
            texts=("find subnet incentives",),
            input_type="query",
        )
    )

    assert result.actual_cost_usd == pytest.approx(0.0042)
    assert result.actual_cost_evidence["settlement_source"] == "provider_returned"


@pytest.mark.parametrize("provider_cost", [None, "invalid"])
async def test_openrouter_embedding_provider_marks_cost_unavailable_without_usable_tokens(
    monkeypatch: pytest.MonkeyPatch,
    provider_cost: float | str | None,
) -> None:
    class _FakeClient:
        async def embed_many(self, texts: tuple[str, ...], **_: object) -> OpenRouterEmbeddingResponse:
            _ = texts
            return OpenRouterEmbeddingResponse(
                vectors=((0.7, 0.8, 0.9),),
                model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
                usage=OpenRouterEmbeddingUsage(cost=provider_cost),
            )

    provider = OpenRouterEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    result = await provider.embed_text(
        EmbedTextRequest(
            provider="openrouter",
            model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
            texts=("find subnet incentives",),
            input_type="query",
        )
    )

    assert result.actual_cost_usd is None
    assert result.actual_cost_evidence["settlement_source"] == "unavailable"
    assert result.actual_cost_evidence["usage_status"] == "tokens_missing"
    assert result.actual_cost_evidence["provider_cost_status"] == (
        "missing" if provider_cost is None else "malformed"
    )
