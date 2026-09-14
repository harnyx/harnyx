"""Strict wire models shared by the sandbox runtime and its HTTP client."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AdmissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    limit_seconds: float = Field(gt=0, allow_inf_nan=False)
    include_wait_in_budget: bool = False


class SandboxAdmission(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    reservation_id: str = Field(min_length=1)
    generation: str = Field(min_length=1)
    deadline_monotonic_ns: int = Field(gt=0)
