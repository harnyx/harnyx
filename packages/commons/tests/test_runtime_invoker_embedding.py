from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from harnyx_commons.domain.session import ProviderCredentialSource
from harnyx_commons.errors import ProviderCredentialUnavailableError, ToolProviderError, ToolProviderFailureCode
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.tools.embedding_models import EmbedTextRequest, EmbedTextResponse
from harnyx_commons.tools.executor import ToolInvocationContext, ToolInvocationOutput
from harnyx_commons.tools.ports import EmbeddingProviderResult
from harnyx_commons.tools.runtime_invoker import DEFAULT_EMBEDDING_TOOL_TIMEOUT_SECONDS, RuntimeToolInvoker

pytestmark = pytest.mark.anyio("asyncio")


class _CapturingEmbeddingProvider:
    def __init__(self) -> None:
        self.requests: list[EmbedTextRequest] = []

    async def embed_text(self, request: EmbedTextRequest) -> EmbeddingProviderResult:
        self.requests.append(request)
        return EmbeddingProviderResult(
            response=EmbedTextResponse.model_validate(
                {
                    "provider": request.provider,
                    "model": request.model,
                    "input_type": request.input_type,
                    "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
                    "dimensions": 3,
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                }
            ),
            actual_cost_usd=0.00000008,
            actual_cost_provider=request.provider,
            actual_cost_evidence={"settlement_source": "test"},
        )

    async def aclose(self) -> None:
        return None


class _UnavailableCostEmbeddingProvider(_CapturingEmbeddingProvider):
    async def embed_text(self, request: EmbedTextRequest) -> EmbeddingProviderResult:
        result = await super().embed_text(request)
        return EmbeddingProviderResult(
            response=result.response,
            actual_cost_usd=None,
            actual_cost_provider="openrouter",
            actual_cost_evidence={
                "settlement_source": "unavailable",
                "upstream_model": "Qwen/Qwen3-Embedding-8B",
                "provider_request_id": "gen-emb-unavailable",
            },
        )


def _context(source: ProviderCredentialSource) -> ToolInvocationContext:
    return ToolInvocationContext(
        receipt_id=f"receipt-{source.value}-embedding",
        session_id=uuid4(),
        active_attempt=0,
        uid=1,
        provider_credential_source=source,
    )


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("chutes", "Qwen/Qwen3-Embedding-8B-TEE"),
        ("openrouter", "qwen/qwen3-embedding-8b"),
    ],
)
async def test_platform_credential_session_resolves_requested_embedding_without_miner_fallback(
    provider: str,
    model: str,
) -> None:
    platform_provider = _CapturingEmbeddingProvider()
    miner_resolver_calls: list[str] = []
    platform_resolver_calls: list[str] = []

    def miner_resolver(provider: str, _context: ToolInvocationContext | None) -> _CapturingEmbeddingProvider:
        miner_resolver_calls.append(provider)
        return _CapturingEmbeddingProvider()

    def platform_resolver(provider: str, _context: ToolInvocationContext | None) -> _CapturingEmbeddingProvider:
        platform_resolver_calls.append(provider)
        return platform_provider

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        embedding_provider_resolver=miner_resolver,
        platform_embedding_provider_resolver=platform_resolver,
    )

    await invoker.invoke(
        "embed_text",
        args=(),
        kwargs={
            "provider": provider,
            "model": model,
            "texts": ["What is Harnyx?"],
            "input_type": "query",
        },
        context=_context(ProviderCredentialSource.PLATFORM),
    )

    assert platform_resolver_calls == [provider]
    assert miner_resolver_calls == []
    assert len(platform_provider.requests) == 1
    assert platform_provider.requests[0].timeout == DEFAULT_EMBEDDING_TOOL_TIMEOUT_SECONDS


async def test_context_free_embedding_uses_matching_direct_provider_without_resolver() -> None:
    direct_provider = _CapturingEmbeddingProvider()
    resolver_calls: list[str] = []

    def resolver(provider: str, _context: ToolInvocationContext | None) -> _CapturingEmbeddingProvider:
        resolver_calls.append(provider)
        return _CapturingEmbeddingProvider()

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        embedding_provider=direct_provider,
        embedding_provider_name="chutes",
        embedding_provider_resolver=resolver,
    )

    await invoker.invoke(
        "embed_text",
        args=(),
        kwargs={
            "provider": "chutes",
            "model": "Qwen/Qwen3-Embedding-8B-TEE",
            "texts": ["What is Harnyx?"],
            "input_type": "query",
        },
    )

    assert len(direct_provider.requests) == 1
    assert resolver_calls == []


async def test_miner_credential_embedding_uses_miner_resolver_without_direct_fallback() -> None:
    direct_provider = _CapturingEmbeddingProvider()
    miner_provider = _CapturingEmbeddingProvider()
    resolver_calls: list[str] = []

    def resolver(provider: str, _context: ToolInvocationContext | None) -> _CapturingEmbeddingProvider:
        resolver_calls.append(provider)
        return miner_provider

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        embedding_provider=direct_provider,
        embedding_provider_name="chutes",
        embedding_provider_resolver=resolver,
    )

    await invoker.invoke(
        "embed_text",
        args=(),
        kwargs={
            "provider": "openrouter",
            "model": "qwen/qwen3-embedding-8b",
            "texts": ["What is Harnyx?"],
            "input_type": "query",
        },
        context=_context(ProviderCredentialSource.MINER),
    )

    assert resolver_calls == ["openrouter"]
    assert len(miner_provider.requests) == 1
    assert direct_provider.requests == []


async def test_platform_embedding_credential_source_error_is_mapped_at_invoker_boundary() -> None:
    def resolver(provider: str, _context: ToolInvocationContext | None) -> _CapturingEmbeddingProvider:
        raise ProviderCredentialUnavailableError(provider)

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        platform_embedding_provider_resolver=resolver,
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await invoker.invoke(
            "embed_text",
            args=(),
            kwargs={
                "provider": "openrouter",
                "model": "qwen/qwen3-embedding-8b",
                "texts": ["What is Harnyx?"],
                "input_type": "query",
            },
            context=_context(ProviderCredentialSource.PLATFORM),
        )

    assert exc_info.value.failure_code is ToolProviderFailureCode.CREDENTIAL_UNAVAILABLE
    assert exc_info.value.provider == "openrouter"


async def test_runtime_invoker_lowers_openrouter_embedding_provider_extra() -> None:
    embedding_provider = _CapturingEmbeddingProvider()
    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        embedding_provider=embedding_provider,
        embedding_provider_name="openrouter",
    )

    output = await invoker.invoke(
        "embed_text",
        args=(),
        kwargs={
            "provider": "openrouter",
            "model": "qwen/qwen3-embedding-8b",
            "texts": ["What is Harnyx?"],
            "input_type": "query",
            "provider_extra": {"provider": {"only": ["nebius"], "allow_fallbacks": False}},
        },
    )

    assert isinstance(output, ToolInvocationOutput)
    assert len(embedding_provider.requests) == 1
    request = embedding_provider.requests[0]
    assert request.provider_extra is not None
    assert request.provider_extra.to_request_extra() == {
        "provider": {"only": ["nebius"], "allow_fallbacks": False}
    }
    assert output.actual_cost_provider == "openrouter"


async def test_runtime_invoker_rejects_chutes_embedding_provider_extra() -> None:
    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        embedding_provider=_CapturingEmbeddingProvider(),
        embedding_provider_name="chutes",
    )

    with pytest.raises(ValidationError):
        await invoker.invoke(
            "embed_text",
            args=(),
            kwargs={
                "provider": "chutes",
                "model": "Qwen/Qwen3-Embedding-8B-TEE",
                "texts": ["What is Harnyx?"],
                "input_type": "query",
                "provider_extra": {"provider": {"only": ["nebius"]}},
            },
        )


async def test_runtime_invoker_preserves_openrouter_embedding_when_cost_is_unavailable() -> None:
    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        embedding_provider=_UnavailableCostEmbeddingProvider(),
        embedding_provider_name="openrouter",
    )

    output = await invoker.invoke(
        "embed_text",
        args=(),
        kwargs={
            "provider": "openrouter",
            "model": "qwen/qwen3-embedding-8b",
            "texts": ["What is Harnyx?"],
            "input_type": "query",
        },
    )

    assert isinstance(output, ToolInvocationOutput)
    assert output.public_payload["data"] == [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]
    assert output.actual_cost_usd is None
    assert output.actual_cost_provider == "openrouter"
    assert output.actual_cost_evidence == {
        "settlement_source": "unavailable",
        "upstream_model": "Qwen/Qwen3-Embedding-8B",
        "provider_request_id": "gen-emb-unavailable",
    }
