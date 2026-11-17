"""Bounded validator storage-probe report carried by signed registration."""
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class StorageProbeReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    probe_id: UUID
    checked_at: datetime
    status: Literal["succeeded", "failed"]
    stage: str = Field(max_length=160)
    cleanup: Literal["succeeded", "not_needed", "unconfirmed"]
    image_bytes: int = Field(gt=0)
    iops: int = Field(gt=0)
    bytes_per_second: int = Field(gt=0)
    details: str = Field(default="", max_length=12000)
    cleanup_details: str = Field(default="", max_length=12000)
