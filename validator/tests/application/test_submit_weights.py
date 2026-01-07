from __future__ import annotations

from datetime import UTC, datetime

import pytest

from caster_validator.application.ports.platform import ChampionWeights
from caster_validator.application.ports.subtensor import ValidatorNodeInfo
from caster_validator.application.submit_weights import WeightSubmissionService
from validator.tests.fixtures.subtensor import FakeSubtensorClient


def fixed_clock() -> datetime:
    return datetime(2025, 10, 17, 13, tzinfo=UTC)


class StubPlatform:
    def __init__(
        self,
        weights: dict[int, float],
        final_top: tuple[int | None, int | None, int | None],
    ):
        self._weights = weights
        self._final_top = final_top

    def get_champion_weights(self) -> ChampionWeights:
        return ChampionWeights(final_top=self._final_top, weights=self._weights)


def test_submission_service_submits_platform_weights() -> None:
    fake = FakeSubtensorClient()
    fake.validator_metadata = ValidatorNodeInfo(uid=7, version_key=None)
    fake.current_block_height = 1_234
    netuid = 1
    fake.tempo_by_netuid[netuid] = 360
    platform = StubPlatform(weights={5: 0.6, 1: 0.4}, final_top=(5, 1, None))
    service = WeightSubmissionService(
        subtensor=fake,
        netuid=netuid,
        clock=fixed_clock,
        platform=platform,
    )

    result = service.submit()

    assert result.final_top == (5, 1, None)
    assert fake.weight_updates[-1] == result.weights
    assert pytest.approx(result.weights[5], rel=1e-6) == 0.6
    assert pytest.approx(result.weights[1], rel=1e-6) == 0.4


def test_submission_service_raises_on_empty_weights() -> None:
    fake = FakeSubtensorClient()
    platform = StubPlatform(weights={}, final_top=(None, None, None))
    service = WeightSubmissionService(
        subtensor=fake,
        netuid=1,
        clock=fixed_clock,
        platform=platform,
    )
    with pytest.raises(RuntimeError):
        service.submit()
