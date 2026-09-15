"""Verify original miner evidence before identity-blind quality judging."""

from harnyx_commons.application.endpoint_answer import verified_endpoint_answer as verified_rating_answer
from harnyx_commons.miner_task_scoring import EvaluationScoringService
from harnyx_commons.rating_competition import RatingJudgment, RatingWork


class RatingCompetitionService:
    def __init__(self, scoring: EvaluationScoringService) -> None:
        self._scoring = scoring

    async def judge(self, work: RatingWork) -> RatingJudgment:
        first = verified_rating_answer(work.first, work.query)
        second = verified_rating_answer(work.second, work.query)
        evidence = await self._scoring.compare_quality(query=work.query, first=first, second=second)
        return RatingJudgment(
            comparison_id=work.comparison_id, quality_result=evidence.quality_result, two_order_evidence=evidence
        )
