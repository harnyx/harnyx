from __future__ import annotations

import pytest
from pydantic import ValidationError

from harnyx_commons.domain.miner_task import EvaluationError, MinerTaskErrorCode


def test_evaluation_error_normalizes_string_code_to_enum() -> None:
    error = EvaluationError(code="timeout_inconclusive", message="terminal timeout")

    assert error.code is MinerTaskErrorCode.TIMEOUT_INCONCLUSIVE
    assert error.model_dump(mode="json") == {
        "code": "timeout_inconclusive",
        "message": "terminal timeout",
    }


def test_evaluation_error_accepts_persisted_script_validation_failed_code() -> None:
    error = EvaluationError(code="script_validation_failed", message="invalid script")

    assert error.code is MinerTaskErrorCode.SCRIPT_VALIDATION_FAILED
    assert error.model_dump(mode="json") == {
        "code": "script_validation_failed",
        "message": "invalid script",
    }


def test_evaluation_error_accepts_enum_member_directly() -> None:
    error = EvaluationError(
        code=MinerTaskErrorCode.SCORING_LLM_RETRY_EXHAUSTED,
        message="scoring exhausted",
    )

    assert error.code is MinerTaskErrorCode.SCORING_LLM_RETRY_EXHAUSTED


def test_evaluation_error_rejects_unknown_code() -> None:
    with pytest.raises(ValidationError, match="not_a_real_code"):
        EvaluationError(code="not_a_real_code", message="boom")
