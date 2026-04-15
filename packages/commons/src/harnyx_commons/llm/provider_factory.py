"""Shared cached LLM provider construction for platform and validator."""

from __future__ import annotations

from collections.abc import Callable

from harnyx_commons.clients import CHUTES
from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm.adapter import LlmProviderAdapter
from harnyx_commons.llm.provider import LlmProviderName, LlmProviderPort, parse_provider_name
from harnyx_commons.llm.providers.bedrock import BedrockLlmProvider
from harnyx_commons.llm.providers.chutes import ChutesLlmProvider
from harnyx_commons.llm.providers.vertex.provider import VertexLlmProvider


def build_cached_llm_provider_resolver(
    *,
    llm_settings: LlmSettings,
    bedrock_settings: BedrockSettings,
    vertex_settings: VertexSettings,
) -> Callable[[str], LlmProviderPort]:
    """Return a cached provider resolver that applies shared concurrency settings."""

    cache: dict[LlmProviderName, LlmProviderPort] = {}

    def resolve(name: str) -> LlmProviderPort:
        provider_name = parse_provider_name(name, component="shared")
        provider = cache.get(provider_name)
        if provider is None:
            provider = _build_provider(
                provider_name=provider_name,
                llm_settings=llm_settings,
                bedrock_settings=bedrock_settings,
                vertex_settings=vertex_settings,
            )
            cache[provider_name] = provider
        return provider

    return resolve


def _build_provider(
    *,
    provider_name: LlmProviderName,
    llm_settings: LlmSettings,
    bedrock_settings: BedrockSettings,
    vertex_settings: VertexSettings,
) -> LlmProviderPort:
    max_concurrent = _max_concurrent_for_provider(provider_name, llm_settings)

    if provider_name == "bedrock":
        return LlmProviderAdapter(
            provider_name=provider_name,
            delegate=BedrockLlmProvider(
                region=bedrock_settings.region_value,
                connect_timeout_seconds=bedrock_settings.connect_timeout_seconds,
                read_timeout_seconds=bedrock_settings.read_timeout_seconds,
                max_concurrent=max_concurrent,
            ),
        )

    if provider_name == "chutes":
        return LlmProviderAdapter(
            provider_name=provider_name,
            delegate=ChutesLlmProvider(
                base_url=CHUTES.base_url,
                api_key=llm_settings.chutes_api_key_value,
                timeout=CHUTES.timeout_seconds,
                max_concurrent=max_concurrent,
            ),
        )

    if provider_name == "vertex":
        return LlmProviderAdapter(
            provider_name=provider_name,
            delegate=VertexLlmProvider(
                project=vertex_settings.gcp_project_id,
                location=vertex_settings.gcp_location,
                timeout=vertex_settings.vertex_timeout_seconds,
                service_account_b64=vertex_settings.gcp_sa_credential_b64_value,
                max_concurrent=max_concurrent,
            ),
        )

    if provider_name == "vertex-maas":
        return LlmProviderAdapter(
            provider_name=provider_name,
            delegate=VertexLlmProvider(
                project=vertex_settings.gcp_project_id,
                location=vertex_settings.vertex_maas_gcp_location,
                timeout=vertex_settings.vertex_timeout_seconds,
                service_account_b64=vertex_settings.gcp_sa_credential_b64_value,
                max_concurrent=max_concurrent,
            ),
        )

    raise ValueError(f"unsupported llm provider: {provider_name}")


def _max_concurrent_for_provider(
    provider_name: LlmProviderName,
    llm_settings: LlmSettings,
) -> int:
    if provider_name == "bedrock":
        return llm_settings.bedrock_max_concurrent
    if provider_name in {"vertex", "vertex-maas"}:
        return llm_settings.vertex_max_concurrent
    if provider_name == "chutes":
        return llm_settings.chutes_max_concurrent
    raise ValueError(f"unsupported llm provider: {provider_name}")


__all__ = ["build_cached_llm_provider_resolver"]
