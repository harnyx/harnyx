"""Validator-owned comparison of an endpoint answer to the dataset reference."""

import asyncio

from harnyx_commons.application.endpoint_answer import verified_endpoint_answer
from harnyx_commons.domain.miner_task import ScoreBreakdown
from harnyx_commons.miner_task_scoring import EvaluationScoringService
from harnyx_commons.reference_selection import ReferenceSelectionRequest, validate_reference_result


class ReferenceJudge:
    def __init__(self, scoring: EvaluationScoringService) -> None:
        self._scoring = scoring

    async def judge(self, request: ReferenceSelectionRequest) -> ScoreBreakdown:
        response = await asyncio.to_thread(verified_endpoint_answer, request.candidate, request.task.query)
        return validate_reference_result(await self._scoring.score_reference(task=request.task, response=response))
