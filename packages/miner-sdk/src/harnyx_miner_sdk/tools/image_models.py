"""Image upload identity and limits shared by the SDK and trusted hosts."""

from __future__ import annotations

import hashlib
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
IMAGE_UPLOAD_HEADER = "x-harnyx-image-upload"


class ImageUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_id: UUID
    size_bytes: int = Field(strict=True, gt=0, le=MAX_IMAGE_BYTES)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def from_bytes(cls, data: bytes, upload_id: UUID) -> ImageUploadRequest:
        if not isinstance(data, bytes):
            raise TypeError("image_upload requires bytes")
        if not 0 < len(data) <= MAX_IMAGE_BYTES:
            raise ValueError("image must contain between 1 byte and 20 MiB")
        return cls(upload_id=upload_id, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


class ImageReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    image_id: UUID
    mime_type: Literal["image/png", "image/jpeg", "image/webp"]
    size_bytes: int = Field(gt=0, le=MAX_IMAGE_BYTES)
