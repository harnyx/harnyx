"""Private Platform-to-validator reference judging contract."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict

from harnyx_commons.domain.miner_task import MinerTask, ScoreBreakdown
from harnyx_commons.endpoint_answer import EndpointAnswer


class ReferenceSelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    batch_id: UUID
    task: MinerTask
    candidate: EndpointAnswer


def validate_reference_result(result: ScoreBreakdown) -> ScoreBreakdown:
    if result.comparison_score not in (0.0, 0.5, 1.0) or result.fast_score_evidence is not None:
        raise ValueError("reference judging requires a complete two-order score")
    return result
