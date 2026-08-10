import pytest
from pydantic import ValidationError

from harnyx_commons.domain_tweak_generation import DomainTweakBatchGenerationResult, SourceDossier


def test_batch_result_rejects_partial_success_state() -> None:
    """Future failure: a partial refill must not escape as a successful batch result."""
    with pytest.raises(ValidationError, match="finalized task count must equal target_count"):
        DomainTweakBatchGenerationResult(
            target_count=1,
            portfolio_call_count=10_000,
            slot_attempt_count=10_000,
            round_count=10_000,
            failure_counts={"reasoning_no_generate": 10_000},
        )

    assert "discarded_candidates" not in DomainTweakBatchGenerationResult.model_fields
    assert "rejected_attempts" not in DomainTweakBatchGenerationResult.model_fields
    assert "completed" not in DomainTweakBatchGenerationResult.model_fields


def test_no_generate_dossier_requires_typed_terminal_cause() -> None:
    """Future failure: dossier attribution must not be reconstructed from unrelated workspace history."""
    with pytest.raises(ValidationError, match="requires failure_class"):
        SourceDossier(
            status="no_generate",
            no_generate_reason="route was not viable",
        )

    dossier = SourceDossier(
        status="no_generate",
        no_generate_reason="route was not viable",
        failure_class="reasoning_no_generate",
    )
    assert dossier.failure_class == "reasoning_no_generate"


def test_source_no_generate_requires_the_exact_failed_fetch_id() -> None:
    """Future failure: a prior failure class alone must not identify the dossier's terminal blocker."""
    with pytest.raises(ValidationError, match="requires source_failure_id"):
        SourceDossier(
            status="no_generate",
            no_generate_reason="the required public document could not be fetched",
            failure_class="source_unavailable",
        )

    dossier = SourceDossier(
        status="no_generate",
        no_generate_reason="the required public document could not be fetched",
        failure_class="source_unavailable",
        source_failure_id="source_failure:3",
    )
    assert dossier.source_failure_id == "source_failure:3"


def test_reasoning_no_generate_forbids_a_source_failure_id() -> None:
    """Future failure: a model-decided dead end must not be attributed to an incidental fetch attempt."""
    with pytest.raises(ValidationError, match="reasoning_no_generate cannot contain source_failure_id"):
        SourceDossier(
            status="no_generate",
            no_generate_reason="the explored route cannot support the requested relationship",
            failure_class="reasoning_no_generate",
            source_failure_id="source_failure:1",
        )
