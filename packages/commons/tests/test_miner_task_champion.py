from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from harnyx_commons.miner_task_champion import (
    ChampionArtifactInput,
    ChampionRunInput,
    ChampionSelection,
    CurrentChampionInput,
    SubmittedArtifactInput,
    filter_successful_validator_runs,
    select_batch_artifacts,
    select_champion,
    selection_from_stored_champion_weights,
    validate_champion_run_inputs,
)
from harnyx_commons.miner_task_ranking import CascadeConfig, RankingCascade


def test_selection_from_stored_champion_weights_reads_single_champion_score() -> None:
    artifact_id = uuid4()

    selection = selection_from_stored_champion_weights(
        final_top=(7,),
        weights={"7": 0.5},
        champion_artifact_id=artifact_id,
    )

    assert selection == ChampionSelection(
        champion_uid=7,
        weights={7: 1.0},
        score=0.5,
        champion_artifact_id=artifact_id,
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
    incumbent = SubmittedArtifactInput(uid=1, artifact_id=uuid4(), submitted_at=cutoff - timedelta(days=1))
    stale = SubmittedArtifactInput(uid=2, artifact_id=uuid4(), submitted_at=cutoff)
    challenger = SubmittedArtifactInput(uid=3, artifact_id=uuid4(), submitted_at=cutoff + timedelta(seconds=1))

    selected = select_batch_artifacts(
        latest_by_uid={1: incumbent, 2: stale, 3: challenger},
        previous_completed_cutoff=cutoff,
        current_champion=CurrentChampionInput(uid=1, artifact_id=incumbent.artifact_id),
        incumbent=incumbent,
    )

    assert selected == (incumbent, challenger)


def test_select_batch_artifacts_fails_when_incumbent_record_does_not_match_champion() -> None:
    cutoff = datetime(2026, 4, 28, tzinfo=UTC)
    incumbent = SubmittedArtifactInput(uid=2, artifact_id=uuid4(), submitted_at=cutoff)

    with pytest.raises(RuntimeError, match="incumbent script uid mismatch"):
        select_batch_artifacts(
            latest_by_uid={2: incumbent},
            previous_completed_cutoff=cutoff,
            current_champion=CurrentChampionInput(uid=1, artifact_id=incumbent.artifact_id),
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


def test_validate_champion_run_inputs_rejects_incomplete_validator_coverage() -> None:
    validator_id = uuid4()
    task_a = uuid4()
    task_b = uuid4()
    artifact_id = uuid4()

    with pytest.raises(ValueError, match="validator has incomplete run coverage for batch"):
        validate_champion_run_inputs(
            task_ids=(task_a, task_b),
            artifacts=(ChampionArtifactInput(artifact_id=artifact_id, uid=7),),
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
            ChampionArtifactInput(artifact_id=incumbent, uid=7),
            ChampionArtifactInput(artifact_id=challenger, uid=8),
        ),
        runs=(
            ChampionRunInput(validator_id, incumbent, task_id, 0.1, 1.0),
            ChampionRunInput(validator_id, challenger, task_id, 1.0, 1.0),
        ),
        current_champion_artifact_id=incumbent,
        cascade=RankingCascade(CascadeConfig(score_margin_required=0.2)),
    )

    assert selection == ChampionSelection(
        champion_uid=8,
        weights={8: 1.0},
        score=1.0,
        champion_artifact_id=challenger,
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
            ChampionArtifactInput(artifact_id=incumbent, uid=7),
            ChampionArtifactInput(artifact_id=challenger_a, uid=8),
            ChampionArtifactInput(artifact_id=challenger_b, uid=9),
        ),
        runs=(
            ChampionRunInput(validator_id, incumbent, task_id, 0.50, 10.0),
            ChampionRunInput(validator_id, challenger_a, task_id, 0.60, 10.0),
            ChampionRunInput(validator_id, challenger_b, task_id, 0.60, 7.0),
        ),
        current_champion_artifact_id=incumbent,
        cascade=RankingCascade(CascadeConfig(score_margin_required=0.2)),
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
            ChampionArtifactInput(artifact_id=incumbent, uid=7),
            ChampionArtifactInput(artifact_id=challenger, uid=8),
        ),
        runs=(
            ChampionRunInput(validator_id, incumbent, task_id, 0.0, 1.0),
            ChampionRunInput(validator_id, challenger, task_id, 0.8, 1.0),
        ),
        current_champion_artifact_id=incumbent,
        cascade=RankingCascade(CascadeConfig(score_margin_required=0.2)),
    )

    assert selection is not None
    assert selection.champion_artifact_id == challenger
    assert selection.similarity_fallback_artifact_ids == (challenger,)
