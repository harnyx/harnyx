from __future__ import annotations

from datetime import UTC, datetime

import pytest

from harnyx_validator.application.ports.platform import ChampionWeights
from harnyx_validator.application.ports.subtensor import ValidatorNodeInfo, WeightSubmissionTooEarlyError
from harnyx_validator.application.submit_weights import WeightSubmissionService
from validator.tests.fixtures.subtensor import FakeSubtensorClient


def fixed_clock() -> datetime:
    return datetime(2025, 10, 17, 13, tzinfo=UTC)


class StubPlatform:
    def __init__(
        self,
        weights: dict[int, float],
        champion_uid: int | None,
    ):
        self._weights = weights
        self._champion_uid = champion_uid
        self.calls = 0

    def get_champion_weights(self) -> ChampionWeights:
        self.calls += 1
        return ChampionWeights(champion_uid=self._champion_uid, weights=self._weights)


def test_submission_service_submits_platform_weights() -> None:
    fake = FakeSubtensorClient()
    fake.validator_metadata = ValidatorNodeInfo(uid=7, version_key=None)
    fake.current_block_height = 1_234
    netuid = 1
    fake.tempo_by_netuid[netuid] = 360
    platform = StubPlatform(weights={5: 0.6, 1: 0.4}, champion_uid=5)
    service = WeightSubmissionService(
        subtensor=fake,
        netuid=netuid,
        clock=fixed_clock,
        platform=platform,
    )

    result = service.submit()

    assert result.champion_uid == 5
    assert fake.weight_updates[-1] == result.weights
    assert pytest.approx(result.weights[5], rel=1e-6) == 0.6
    assert pytest.approx(result.weights[1], rel=1e-6) == 0.4


def test_submission_service_raises_on_empty_weights() -> None:
    fake = FakeSubtensorClient()
    platform = StubPlatform(weights={}, champion_uid=None)
    service = WeightSubmissionService(
        subtensor=fake,
        netuid=1,
        clock=fixed_clock,
        platform=platform,
    )
    with pytest.raises(RuntimeError):
        service.submit()


def test_try_submit_attempts_submission_without_prechecking_cadence() -> None:
    fake = FakeSubtensorClient()
    fake.validator_metadata = ValidatorNodeInfo(uid=-1, version_key=None)
    platform = StubPlatform(weights={5: 1.0}, champion_uid=5)
    service = WeightSubmissionService(
        subtensor=fake,
        netuid=1,
        clock=fixed_clock,
        platform=platform,
    )

    result = service.try_submit()

    assert result is not None
    assert platform.calls == 1
    assert fake.weight_updates == [{5: 1.0}]
    assert fake.weight_updates == [{5: 1.0}]


def test_try_submit_returns_none_when_chain_reports_too_early() -> None:
    fake = FakeSubtensorClient()
    fake.validator_metadata = ValidatorNodeInfo(uid=7, version_key=None)
    fake.submit_weights_exception = WeightSubmissionTooEarlyError("SettingWeightsTooFast")
    platform = StubPlatform(weights={5: 1.0}, champion_uid=5)
    service = WeightSubmissionService(
        subtensor=fake,
        netuid=1,
        clock=fixed_clock,
        platform=platform,
    )

    result = service.try_submit()

    assert result is None
    assert platform.calls == 1
    assert fake.weight_updates == []


def test_try_submit_propagates_non_too_early_submission_errors() -> None:
    fake = FakeSubtensorClient()
    fake.validator_metadata = ValidatorNodeInfo(uid=7, version_key=None)
    fake.submit_weights_exception = RuntimeError("metadata unavailable")
    platform = StubPlatform(weights={5: 1.0}, champion_uid=5)
    service = WeightSubmissionService(
        subtensor=fake,
        netuid=1,
        clock=fixed_clock,
        platform=platform,
    )

    with pytest.raises(RuntimeError, match="metadata unavailable"):
        service.try_submit()

    assert platform.calls == 1
    assert fake.weight_updates == []


def test_try_submit_submits_platform_weights() -> None:
    fake = FakeSubtensorClient()
    fake.validator_metadata = ValidatorNodeInfo(uid=7, version_key=None)
    netuid = 1
    fake.current_block_height = 200
    fake.last_update_by_uid[7] = 100
    fake.weights_rate_limit_by_netuid[netuid] = 100
    fake.commit_reveal_enabled_by_netuid[netuid] = False
    platform = StubPlatform(weights={5: 1.0}, champion_uid=5)
    service = WeightSubmissionService(
        subtensor=fake,
        netuid=netuid,
        clock=fixed_clock,
        platform=platform,
    )

    result = service.try_submit()

    assert result is not None
    assert platform.calls == 1
