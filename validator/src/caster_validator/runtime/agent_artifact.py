"""Agent artifact resolution and validation utilities (platform-provided only)."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from caster_commons.sandbox.agent_staging import MAX_AGENT_BYTES, AgentArtifact, stage_agent_source
from caster_validator.application.dto.evaluation import MinerTaskBatchSpec, ScriptArtifactSpec
from caster_validator.application.ports.platform import PlatformPort

logger = logging.getLogger("caster_validator.agent_artifact")


def resolve_platform_agent_specs(
    *,
    batch_id: UUID,
    candidates: tuple[ScriptArtifactSpec, ...],
    platform_client: PlatformPort,
    state_dir: Path,
    container_root: str,
) -> dict[UUID, AgentArtifact]:
    """Resolve agent artifacts from platform-provided specs."""

    specs: dict[UUID, AgentArtifact] = {}
    artifact_ids = [candidate.artifact_id for candidate in candidates]
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("batch candidates must be unique by artifact_id")

    for spec in candidates:
        try:
            data = platform_client.fetch_artifact(batch_id, spec.artifact_id)
        except Exception as exc:
            logger.error(
                "Failed to fetch platform agent",
                extra={"batch_id": str(batch_id), "uid": spec.uid, "artifact_id": str(spec.artifact_id)},
                exc_info=exc,
            )
            continue
        if len(data) > MAX_AGENT_BYTES:
            logger.error(
                "Platform agent exceeds size limit",
                extra={
                    "batch_id": str(batch_id),
                    "uid": spec.uid,
                    "artifact_id": str(spec.artifact_id),
                    "size_bytes": len(data),
                },
            )
            continue
        content_hash = hashlib.sha256(data).hexdigest()
        if content_hash != spec.content_hash:
            logger.error(
                "Platform agent sha256 mismatch",
                extra={
                    "batch_id": str(batch_id),
                    "uid": spec.uid,
                    "artifact_id": str(spec.artifact_id),
                    "expected_sha256": spec.content_hash,
                    "actual_sha256": content_hash,
                },
            )
            continue
        artifact = _stage_platform_agent(
            content_hash=content_hash,
            data=data,
            state_dir=state_dir,
            container_root=container_root,
        )
        specs[spec.artifact_id] = artifact
        logger.info(
            "Staged platform agent",
            extra={
                "uid": spec.uid,
                "artifact_id": str(spec.artifact_id),
                "content_hash": content_hash,
                "host_path": str(artifact.host_path),
                "container_path": artifact.container_path,
            },
        )
    return specs


def _stage_platform_agent(
    *,
    content_hash: str,
    data: bytes,
    state_dir: Path,
    container_root: str,
) -> AgentArtifact:
    artifact = stage_agent_source(
        state_dir=state_dir,
        container_root=container_root,
        namespace="platform_agents",
        key=content_hash,
        data=data,
        max_bytes=MAX_AGENT_BYTES,
    )
    if artifact.content_hash != content_hash:
        raise RuntimeError(
            "platform agent sha256 mismatch after staging "
            f"(expected={content_hash} actual={artifact.content_hash})"
        )
    return artifact


def create_platform_agent_resolver(
    platform_client: PlatformPort,
) -> Callable[[UUID, MinerTaskBatchSpec, Path, str], dict[UUID, AgentArtifact]]:
    """Create a resolver function for platform agent artifacts."""

    def resolver(
        batch_id: UUID,
        batch: MinerTaskBatchSpec,
        state_dir: Path,
        container_root: str,
    ) -> dict[UUID, AgentArtifact]:
        return resolve_platform_agent_specs(
            batch_id=batch_id,
            candidates=batch.candidates,
            platform_client=platform_client,
            state_dir=state_dir,
            container_root=container_root,
        )

    return resolver


__all__ = [
    "AgentArtifact",
    "MAX_AGENT_BYTES",
    "create_platform_agent_resolver",
    "resolve_platform_agent_specs",
]
