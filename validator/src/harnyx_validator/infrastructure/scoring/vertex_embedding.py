"""Vertex-backed text embeddings for validator run scoring."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

import httpx
from google import genai
from google.genai import errors, types

from harnyx_commons.config.external_client import ExternalClientRetrySettings
from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_commons.llm.providers.vertex.credentials import prepare_credentials
from harnyx_commons.llm.retry_utils import RetryPolicy, backoff_ms

_VERTEX_API_VERSION = "v1"


class _EmbeddingValues(Protocol):
    values: Sequence[float] | None


class _EmbeddingResponse(Protocol):
    embeddings: Sequence[_EmbeddingValues] | None


class VertexEmbeddingRetryExhaustedError(LlmRetryExhaustedError):
    """Retry flow failed after exhausting embedding attempts."""


@dataclass(frozen=True, slots=True)
class VertexTextEmbeddingClient:
    client: genai.Client
    model: str
    dimensions: int
    retry_policy: RetryPolicy

    @classmethod
    def from_vertex_settings(
        cls,
        *,
        project: str | None,
        location: str | None,
        service_account_b64: str | None,
        model: str,
        timeout_seconds: float,
        dimensions: int = 768,
    ) -> VertexTextEmbeddingClient:
        if project is None or not project.strip():
            raise RuntimeError("GCP_PROJECT_ID must be configured for validator run scoring embeddings")
        if location is None or not location.strip():
            raise RuntimeError("GCP_LOCATION must be configured for validator run scoring embeddings")
        normalized_model = model.strip()
        if not normalized_model:
            raise RuntimeError("validator run scoring embedding model must be configured")
        timeout_ms = math.ceil(timeout_seconds * 1000) if timeout_seconds > 0 else None
        credentials, _ = prepare_credentials(None, service_account_b64)
        client = genai.Client(
            vertexai=True,
            project=project.strip(),
            location=location.strip(),
            credentials=credentials,
            http_options=types.HttpOptions(
                api_version=_VERTEX_API_VERSION,
                timeout=int(timeout_ms) if timeout_ms is not None else None,
            ),
        )
        return cls(
            client=client,
            model=normalized_model,
            dimensions=dimensions,
            retry_policy=ExternalClientRetrySettings().retry_policy,
        )

    async def embed(self, text: str) -> tuple[float, ...]:
        normalized = text.strip()
        if not normalized:
            raise ValueError("embedding input text must not be empty")
        for attempt in range(self.retry_policy.attempts):
            try:
                response = await self.client.aio.models.embed_content(
                    model=self.model,
                    contents=normalized,
                    config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
                )
                return _extract_vector(
                    cast(_EmbeddingResponse, response),
                    expected_dimensions=self.dimensions,
                )
            except Exception as exc:
                retryable, reason = _classify_embedding_exception(exc)
                if not retryable:
                    raise
                if (attempt + 1) >= self.retry_policy.attempts:
                    raise VertexEmbeddingRetryExhaustedError(reason) from exc
                await asyncio.sleep(backoff_ms(attempt, self.retry_policy) / 1000)
        raise AssertionError("embedding retry loop exited unexpectedly")

    async def aclose(self) -> None:
        await self.client.aio.aclose()
        self.client.close()


@dataclass(frozen=True, slots=True)
class MissingTextEmbeddingClient:
    reason: str

    async def embed(self, text: str) -> tuple[float, ...]:
        _ = text
        raise RuntimeError(self.reason)

    async def aclose(self) -> None:
        return None


@dataclass(slots=True)
class LazyVertexTextEmbeddingClient:
    project: str | None
    location: str | None
    service_account_b64: str | None
    model: str
    timeout_seconds: float
    dimensions: int = 768
    _client: VertexTextEmbeddingClient | None = field(default=None, init=False, repr=False)

    async def embed(self, text: str) -> tuple[float, ...]:
        client = self._client
        if client is None:
            client = VertexTextEmbeddingClient.from_vertex_settings(
                project=self.project,
                location=self.location,
                service_account_b64=self.service_account_b64,
                model=self.model,
                timeout_seconds=self.timeout_seconds,
                dimensions=self.dimensions,
            )
            self._client = client
        return await client.embed(text)

    async def aclose(self) -> None:
        client = self._client
        if client is None:
            return None
        await client.aclose()
        self._client = None


def _classify_embedding_exception(exc: Exception) -> tuple[bool, str]:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else None
        retryable = status is not None and (status == 429 or status >= 500)
        return retryable, f"http_{status}"
    if isinstance(exc, httpx.HTTPError):
        return True, exc.__class__.__name__
    if isinstance(exc, errors.APIError):
        code = exc.code
        message = str(exc)
        retryable = code in {429, 503, 529}
        return retryable, f"api_error:{code}:{message}"
    return False, str(exc)


def _extract_vector(response: _EmbeddingResponse, *, expected_dimensions: int) -> tuple[float, ...]:
    embeddings = response.embeddings
    if embeddings is None or not embeddings:
        raise RuntimeError("embedding response missing embeddings")
    values = embeddings[0].values
    if values is None or not values:
        raise RuntimeError("embedding response missing vector values")
    vector = tuple(float(value) for value in values)
    if len(vector) != expected_dimensions:
        raise RuntimeError(
            f"embedding dimensions mismatch: expected={expected_dimensions} actual={len(vector)}"
        )
    return vector


__all__ = [
    "LazyVertexTextEmbeddingClient",
    "MissingTextEmbeddingClient",
    "VertexEmbeddingRetryExhaustedError",
    "VertexTextEmbeddingClient",
]
