"""Chutes-backed text embeddings for validator run scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from harnyx_commons.clients import CHUTES


class _EmbeddingDatum(BaseModel):
    # Chutes can return provider-specific metadata alongside the embedding vector.
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    embedding: list[float] = Field(min_length=1)


class _EmbeddingResponse(BaseModel):
    # Keep the boundary tolerant to extra response metadata while still validating the data array shape.
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    data: list[_EmbeddingDatum] = Field(min_length=1)


@dataclass(slots=True)
class ChutesTextEmbeddingClient:
    model: str
    api_key: str
    base_url: str = CHUTES.base_url
    timeout_seconds: float = 30.0
    dimensions: int | None = None
    client: httpx.AsyncClient | None = None
    _owns_client: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        normalized_model = self.model.strip()
        if not normalized_model:
            raise ValueError("chutes embedding model must be configured")
        if not self.api_key:
            raise ValueError("Chutes API key must be provided for embeddings")
        self.model = normalized_model
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
                timeout=self.timeout_seconds,
            )
            self._owns_client = True
        else:
            self._owns_client = False

    async def embed(self, text: str) -> tuple[float, ...]:
        normalized = text.strip()
        if not normalized:
            raise ValueError("embedding input text must not be empty")
        client = self._require_client()
        response = await client.post(
            "v1/embeddings",
            json=self._request_body(normalized),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        payload = _EmbeddingResponse.model_validate(response.json())
        vector = tuple(float(value) for value in payload.data[0].embedding)
        if self.dimensions is not None and len(vector) != self.dimensions:
            raise RuntimeError(
                f"embedding dimensions mismatch: expected={self.dimensions} actual={len(vector)}"
            )
        return vector

    async def aclose(self) -> None:
        if self._owns_client:
            await self._require_client().aclose()

    def _request_body(self, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": text,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        return payload

    def _require_client(self) -> httpx.AsyncClient:
        if self.client is None:
            raise RuntimeError("chutes embedding client is not initialized")
        return self.client


__all__ = ["ChutesTextEmbeddingClient"]
