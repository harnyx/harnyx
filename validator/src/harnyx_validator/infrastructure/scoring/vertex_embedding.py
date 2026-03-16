"""Vertex-backed text embeddings for validator run scoring."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from harnyx_commons.llm.providers.vertex.credentials import prepare_credentials

_VERTEX_API_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class VertexTextEmbeddingClient:
    client: genai.Client
    model: str
    dimensions: int

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
        )

    async def embed(self, text: str) -> tuple[float, ...]:
        normalized = text.strip()
        if not normalized:
            raise ValueError("embedding input text must not be empty")
        return await asyncio.to_thread(self._embed_sync, normalized)

    def _embed_sync(self, text: str) -> tuple[float, ...]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
        )
        embeddings = response.embeddings
        if embeddings is None or not embeddings:
            raise RuntimeError("embedding response missing embeddings")
        values = embeddings[0].values
        if values is None or not values:
            raise RuntimeError("embedding response missing vector values")
        vector = tuple(float(value) for value in values)
        if len(vector) != self.dimensions:
            raise RuntimeError(
                f"embedding dimensions mismatch: expected={self.dimensions} actual={len(vector)}"
            )
        return vector


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
        return None


__all__ = [
    "LazyVertexTextEmbeddingClient",
    "MissingTextEmbeddingClient",
    "VertexTextEmbeddingClient",
]
