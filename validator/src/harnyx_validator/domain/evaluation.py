"""Validator miner-task run domain models."""

from __future__ import annotations

from datetime import datetime
from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    EvaluationError,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_validator.domain.shared_config import VALIDATOR_STRICT_CONFIG


class MinerTaskRun(BaseModel):
    model_config = VALIDATOR_STRICT_CONFIG

    session_id: UUID
    uid: int = Field(ge=0)
    artifact_id: UUID
    task_id: UUID
    response: Response | None = None
    details: EvaluationDetails
    completed_at: datetime

    @model_validator(mode="after")
    def _validate_state(self) -> Self:
        if self.details.score_breakdown is not None:
            if self.response is None:
                raise ValueError("successful runs must include a response")
            return self
        if self.details.error is None:
            raise ValueError("failed runs must include an evaluation error")
        if self.response is not None:
            raise ValueError("failed runs must not include a response")
        return self


__all__ = [
    "EvaluationDetails",
    "EvaluationError",
    "MinerTask",
    "MinerTaskRun",
    "Query",
    "ReferenceAnswer",
    "Response",
    "ScoreBreakdown",
]
