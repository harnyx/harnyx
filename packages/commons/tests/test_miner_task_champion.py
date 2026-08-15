from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from harnyx_commons.miner_task_champion import (
    ChampionArtifactInput,
    ChampionRunInput,
    ChampionSelection,
    CurrentChampionInput,
    SubmittedArtifactInput,
    artifact_batch_scores,
    filter_successful_validator_runs,
    select_batch_artifacts,
    select_champion,
    selection_from_stored_champion_weights,
    validate_champion_run_inputs,
)
from harnyx_commons.miner_task_ranking import ArtifactAggregateBundle, RankingCascade


def test_selection_from_stored_champion_weights_reads_single_champion_score() -> None:
    artifact_id = uuid4()

    selection = selection_from_stored_champion_weights(
        final_top=(7,),
        weights={"7": 0.5},
        champion_artifact_id=artifact_id,
        champion_hotkey_ss58="hotkey-7",
    )

    assert selection == ChampionSelection(
        champion_uid=7,
        weights={7: 1.0},
        score=0.5,
        champion_artifact_id=artifact_id,
        champion_hotkey_ss58="hotkey-7",
    )


def test_selection_from_stored_champion_weights_keeps_legacy_split_weights_at_full_score() -> None:
    selection = selection_from_stored_champion_weights(
        final_top=(7, 8),
        weights={"7": 0.6, "8": 0.4},
        champion_artifact_id=None,
    )

    assert selection == ChampionSelection(
        champion_uid=7,
        weights={7: 1.0},
        score=1.0,
        champion_artifact_id=None,
    )


def test_select_batch_artifacts_keeps_incumbent_and_new_challengers() -> None:
    cutoff = datetime(2026, 4, 28, tzinfo=UTC)
    incumbent = SubmittedArtifactInput(
        uid=1,
        artifact_id=uuid4(),
        submitted_at=cutoff - timedelta(days=1),
        miner_hotkey_ss58="hotkey-1",
    )
    stale = SubmittedArtifactInput(
        uid=2,
        artifact_id=uuid4(),
        submitted_at=cutoff,
        miner_hotkey_ss58="hotkey-2",
    )
    challenger = SubmittedArtifactInput(
        uid=3,
        artifact_id=uuid4(),
        submitted_at=cutoff + timedelta(seconds=1),
        miner_hotkey_ss58="hotkey-3",
    )

    selected = select_batch_artifacts(
        latest_by_hotkey={"hotkey-1": incumbent, "hotkey-2": stale, "hotkey-3": challenger},
        previous_completed_cutoff=cutoff,
        current_champion=CurrentChampionInput(
            uid=1,
            artifact_id=incumbent.artifact_id,
            miner_hotkey_ss58="hotkey-1",
        ),
        incumbent=incumbent,
    )

    assert selected == (incumbent, challenger)


def test_select_batch_artifacts_prioritizes_incumbent_hotkey_new_challenger() -> None:
    cutoff = datetime(2026, 4, 28, tzinfo=UTC)
    incumbent = SubmittedArtifactInput(
        uid=1,
        artifact_id=uuid4(),
        submitted_at=cutoff - timedelta(days=1),
        miner_hotkey_ss58="hotkey-1",
    )
    earlier_copycat = SubmittedArtifactInput(
        uid=2,
        artifact_id=uuid4(),
        submitted_at=cutoff + timedelta(seconds=1),
        miner_hotkey_ss58="hotkey-2",
    )
    higher_artifact_non_incumbent = SubmittedArtifactInput(
        uid=3,
        artifact_id=UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"),
        submitted_at=cutoff + timedelta(seconds=3),
        miner_hotkey_ss58="hotkey-3",
    )
    lower_artifact_non_incumbent = SubmittedArtifactInput(
        uid=3,
        artifact_id=UUID("00000000-0000-0000-0000-000000000001"),
        submitted_at=cutoff + timedelta(seconds=3),
        miner_hotkey_ss58="hotkey-4",
    )
    incumbent_challenger = SubmittedArtifactInput(
        uid=1,
        artifact_id=uuid4(),
        submitted_at=cutoff + timedelta(seconds=2),
        miner_hotkey_ss58="hotkey-1",
    )

    selected = select_batch_artifacts(
        latest_by_hotkey={
            "hotkey-1": incumbent_challenger,
            "hotkey-2": earlier_copycat,
            "hotkey-3": higher_artifact_non_incumbent,
            "hotkey-4": lower_artifact_non_incumbent,
        },
        previous_completed_cutoff=cutoff,
        current_champion=CurrentChampionInput(
            uid=incumbent.uid,
            artifact_id=incumbent.artifact_id,
            miner_hotkey_ss58="hotkey-1",
        ),
        incumbent=incumbent,
    )

    assert selected == (
        incumbent,
        incumbent_challenger,
        earlier_copycat,
        lower_artifact_non_incumbent,
        higher_artifact_non_incumbent,
    )


def test_select_batch_artifacts_fails_when_incumbent_record_does_not_match_champion() -> None:
    cutoff = datetime(2026, 4, 28, tzinfo=UTC)
    incumbent = SubmittedArtifactInput(
        uid=2,
        artifact_id=uuid4(),
        submitted_at=cutoff,
        miner_hotkey_ss58="hotkey-2",
    )

    with pytest.raises(RuntimeError, match="incumbent script hotkey mismatch"):
        select_batch_artifacts(
            latest_by_hotkey={"hotkey-2": incumbent},
            previous_completed_cutoff=cutoff,
            current_champion=CurrentChampionInput(
                uid=1,
                artifact_id=incumbent.artifact_id,
                miner_hotkey_ss58="hotkey-1",
            ),
            incumbent=incumbent,
        )


def test_filter_successful_validator_runs_keeps_only_successful_validators() -> None:
    validator_a = uuid4()
    validator_b = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    runs = (
        ChampionRunInput(validator_a, artifact_id, task_id, 1.0, 0.1),
        ChampionRunInput(validator_b, artifact_id, task_id, 1.0, 0.1),
    )

    assert filter_successful_validator_runs(runs, successful_validator_ids=(validator_b,)) == (runs[1],)


def test_artifact_batch_scores_normalize_aggregate_totals_by_task_count() -> None:
    artifact_a = uuid4()
    artifact_b = uuid4()

    scores = artifact_batch_scores(
        artifact_ids=(artifact_a, artifact_b),
        task_count=2,
        aggregates=ArtifactAggregateBundle(
            vectors={},
            totals={artifact_a: 1.2, artifact_b: 0.8},
            costs={},
        ),
    )

    assert scores == {
        artifact_a: pytest.approx(0.6),
        artifact_b: pytest.approx(0.4),
    }


def test_artifact_batch_scores_reject_invalid_normalized_score() -> None:
    artifact_id = uuid4()

    with pytest.raises(ValueError, match="artifact batch score must be between 0.0 and 1.0"):
        artifact_batch_scores(
            artifact_ids=(artifact_id,),
            task_count=2,
            aggregates=ArtifactAggregateBundle(vectors={}, totals={artifact_id: 2.2}, costs={}),
        )


def test_validate_champion_run_inputs_rejects_incomplete_validator_coverage() -> None:
    validator_id = uuid4()
    task_a = uuid4()
    task_b = uuid4()
    artifact_id = uuid4()

    with pytest.raises(ValueError, match="validator has incomplete run coverage for batch"):
        validate_champion_run_inputs(
            task_ids=(task_a, task_b),
            artifacts=(ChampionArtifactInput(artifact_id=artifact_id, uid=7, miner_hotkey_ss58="hotkey-7"),),
            runs=(ChampionRunInput(validator_id, artifact_id, task_a, 1.0, 0.1),),
        )


def test_select_champion_returns_winner_take_all_selection() -> None:
    validator_id = uuid4()
    task_id = uuid4()
    incumbent = uuid4()
    challenger = uuid4()

    selection = select_champion(
        task_ids=(task_id,),
        artifacts=(
            ChampionArtifactInput(artifact_id=incumbent, uid=7, miner_hotkey_ss58="hotkey-7"),
            ChampionArtifactInput(artifact_id=challenger, uid=8, miner_hotkey_ss58="hotkey-8"),
        ),
        runs=(
            ChampionRunInput(validator_id, incumbent, task_id, 0.1, 1.0),
            ChampionRunInput(validator_id, challenger, task_id, 1.0, 1.0),
        ),
        current_champion_artifact_id=incumbent,
        cascade=RankingCascade(),
    )

    assert selection == ChampionSelection(
        champion_uid=8,
        weights={8: 1.0},
        score=1.0,
        champion_artifact_id=challenger,
        champion_hotkey_ss58="hotkey-8",
    )
    assert selection is not None
    assert selection.incumbent_artifact_id == incumbent
    assert selection.similarity_fallback_artifact_ids == (challenger,)
    assert selection.ranking_trace is not None
    assert selection.ranking_trace.successful_dethroner_artifact_ids() == (challenger,)


def test_select_champion_similarity_candidates_walk_backward_through_dethrone_sequence() -> None:
    validator_id = uuid4()
    task_id = uuid4()
    incumbent = uuid4()
    challenger_a = uuid4()
    challenger_b = uuid4()

    selection = select_champion(
        task_ids=(task_id,),
        artifacts=(
            ChampionArtifactInput(artifact_id=incumbent, uid=7, miner_hotkey_ss58="hotkey-7"),
            ChampionArtifactInput(artifact_id=challenger_a, uid=8, miner_hotkey_ss58="hotkey-8"),
            ChampionArtifactInput(artifact_id=challenger_b, uid=9, miner_hotkey_ss58="hotkey-9"),
        ),
        runs=(
            ChampionRunInput(validator_id, incumbent, task_id, 0.50, 10.0, elapsed_ms=5_000.0),
            ChampionRunInput(validator_id, challenger_a, task_id, 0.60, 10.0, elapsed_ms=5_000.0),
            ChampionRunInput(validator_id, challenger_b, task_id, 0.60, 7.0, elapsed_ms=5_000.0),
        ),
        current_champion_artifact_id=incumbent,
        cascade=RankingCascade(),
    )

    assert selection is not None
    assert selection.champion_artifact_id == challenger_b
    assert selection.similarity_fallback_artifact_ids == (challenger_b, challenger_a)


def test_select_champion_similarity_candidates_include_replacement_of_zero_incumbent() -> None:
    validator_id = uuid4()
    task_id = uuid4()
    incumbent = uuid4()
    challenger = uuid4()

    selection = select_champion(
        task_ids=(task_id,),
        artifacts=(
            ChampionArtifactInput(artifact_id=incumbent, uid=7, miner_hotkey_ss58="hotkey-7"),
            ChampionArtifactInput(artifact_id=challenger, uid=8, miner_hotkey_ss58="hotkey-8"),
        ),
        runs=(
            ChampionRunInput(validator_id, incumbent, task_id, 0.0, 1.0),
            ChampionRunInput(validator_id, challenger, task_id, 0.8, 1.0),
        ),
        current_champion_artifact_id=incumbent,
        cascade=RankingCascade(),
    )

    assert selection is not None
    assert selection.champion_artifact_id == challenger
    assert selection.similarity_fallback_artifact_ids == (challenger,)
