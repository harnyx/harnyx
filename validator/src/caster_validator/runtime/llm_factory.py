"""LLM provider factory for validator runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from caster_commons.llm.adapter import LlmProviderAdapter
from caster_commons.llm.provider import LlmProviderName, LlmProviderPort, parse_provider_name
from caster_commons.llm.providers.chutes import ChutesLlmProvider
from caster_commons.llm.providers.openai import OpenAILlmProvider
from caster_commons.llm.providers.vertex.provider import VertexLlmProvider


def create_llm_provider_factory(
    *,
    openai_api_key: str,
    openai_base_url: str,
    openai_timeout: float,
    chutes_api_key: str,
    chutes_base_url: str,
    chutes_timeout: float,
    gcp_project_id: str | None,
    gcp_location: str | None,
    vertex_maas_gcp_location: str,
    vertex_timeout: float,
    gcp_service_account_b64: str,
) -> Callable[[str], LlmProviderPort]:
    """Create a factory function for resolving LLM providers by name."""
    cfg = ProviderConfig(
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_timeout=openai_timeout,
        chutes_api_key=chutes_api_key,
        chutes_base_url=chutes_base_url,
        chutes_timeout=chutes_timeout,
        gcp_project_id=gcp_project_id,
        gcp_location=gcp_location,
        vertex_maas_gcp_location=vertex_maas_gcp_location,
        vertex_timeout=vertex_timeout,
        gcp_service_account_b64=gcp_service_account_b64,
    )
    cache: dict[LlmProviderName, LlmProviderPort] = {}
    providers = _provider_registry()

    def resolve(name: str) -> LlmProviderPort:
        provider_name = parse_provider_name(name, component="validator llm provider")
        if provider_name in cache:
            return cache[provider_name]

        spec = providers.get(provider_name)
        if spec is None:
            raise RuntimeError(f"unknown llm provider {provider_name!r}")

        base_provider = spec.build(cfg)
        provider = _wrap_adapter(provider_name, base_provider)
        cache[provider_name] = provider
        return provider

    return resolve


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    openai_api_key: str
    openai_base_url: str
    openai_timeout: float
    chutes_api_key: str
    chutes_base_url: str
    chutes_timeout: float
    gcp_project_id: str | None
    gcp_location: str | None
    vertex_maas_gcp_location: str
    vertex_timeout: float
    gcp_service_account_b64: str


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    name: LlmProviderName
    build: Callable[[ProviderConfig], LlmProviderPort]


def _provider_registry() -> dict[LlmProviderName, ProviderSpec]:
    specs = (
        ProviderSpec("openai", _build_openai),
        ProviderSpec("chutes", _build_chutes),
        ProviderSpec("vertex", _build_vertex),
        ProviderSpec("vertex-maas", _build_vertex_maas),
    )
    return {spec.name: spec for spec in specs}


def _build_openai(cfg: ProviderConfig) -> LlmProviderPort:
    if not cfg.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY must be configured for the openai provider")
    return OpenAILlmProvider(
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
        timeout=cfg.openai_timeout,
    )


def _build_chutes(cfg: ProviderConfig) -> LlmProviderPort:
    if not cfg.chutes_api_key:
        raise RuntimeError("CHUTES_API_KEY must be configured for the chutes provider")
    return ChutesLlmProvider(
        base_url=cfg.chutes_base_url,
        api_key=cfg.chutes_api_key,
        timeout=cfg.chutes_timeout,
    )


def _build_vertex(cfg: ProviderConfig) -> LlmProviderPort:
    return VertexLlmProvider(
        project=cfg.gcp_project_id,
        location=cfg.gcp_location,
        timeout=cfg.vertex_timeout,
        credentials_path=None,
        service_account_b64=cfg.gcp_service_account_b64,
    )


def _build_vertex_maas(cfg: ProviderConfig) -> LlmProviderPort:
    return VertexLlmProvider(
        project=cfg.gcp_project_id,
        location=cfg.vertex_maas_gcp_location,
        timeout=cfg.vertex_timeout,
        credentials_path=None,
        service_account_b64=cfg.gcp_service_account_b64,
    )


def _wrap_adapter(name: str, base_provider: LlmProviderPort) -> LlmProviderPort:
    return LlmProviderAdapter(provider_name=name, delegate=base_provider)


__all__ = ["create_llm_provider_factory"]
