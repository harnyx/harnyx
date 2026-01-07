"""Shared parsing helpers for validator infrastructure."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, TypeAdapter

from caster_commons.domain.claim import EvaluationClaim
from caster_validator.application.dto.evaluation import EvaluationBatchSpec, ScriptArtifactSpec


class _ScriptArtifactPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uid: int
    digest: str
    size_bytes: int
    artifact_id: str | None = None


class _BatchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    entrypoint: str
    cutoff_at_iso: str = Field(validation_alias=AliasChoices("cutoff_at_iso", "cutoff_at"))
    created_at_iso: str = Field(validation_alias=AliasChoices("created_at_iso", "created_at"))
    uids: tuple[int, ...]
    claims: tuple[EvaluationClaim, ...]
    artifacts: tuple[_ScriptArtifactPayload, ...]


_BATCH_PAYLOAD_ADAPTER = TypeAdapter(_BatchPayload)


def parse_batch(payload: Mapping[str, object]) -> EvaluationBatchSpec:
    """Normalize raw batch payloads into EvaluationBatchSpec."""
    parsed = _BATCH_PAYLOAD_ADAPTER.validate_python(payload)
    artifacts = tuple(
        ScriptArtifactSpec(
            uid=item.uid,
            digest=item.digest,
            size_bytes=item.size_bytes,
            artifact_id=item.artifact_id,
        )
        for item in parsed.artifacts
    )

    return EvaluationBatchSpec(
        run_id=parsed.run_id,
        entrypoint=parsed.entrypoint,
        cutoff_at_iso=parsed.cutoff_at_iso,
        created_at_iso=parsed.created_at_iso,
        uids=parsed.uids,
        claims=parsed.claims,
        artifacts=artifacts,
    )


__all__ = ["parse_batch"]
