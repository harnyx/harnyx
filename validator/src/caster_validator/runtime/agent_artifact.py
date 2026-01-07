"""Agent artifact resolution and validation utilities (platform-provided only)."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID

from caster_validator.application.dto.evaluation import EvaluationBatchSpec, ScriptArtifactSpec
from caster_validator.application.ports.platform import PlatformPort

logger = logging.getLogger("caster_validator.agent_artifact")

MAX_AGENT_BYTES = 256_000
_LOG_SNIPPET_LIMIT = 512


@dataclass(frozen=True, slots=True)
class AgentArtifact:
    """Represents a resolved agent script artifact."""

    digest: str
    host_path: Path
    container_path: str


def resolve_platform_agent_specs(
    *,
    run_id: UUID,
    uids: tuple[int, ...],
    artifacts: tuple[ScriptArtifactSpec, ...],
    platform_client: PlatformPort,
    state_dir: Path,
    container_root: str,
) -> dict[int, AgentArtifact]:
    """Resolve agent artifacts from platform-provided specs."""

    cache_root = state_dir / "platform_agents"
    cache_root.mkdir(parents=True, exist_ok=True)
    artifacts_by_uid = {artifact.uid: artifact for artifact in artifacts}
    specs: dict[int, AgentArtifact] = {}

    for uid in uids:
        spec = artifacts_by_uid.get(uid)
        if spec is None:
            continue
        data = platform_client.fetch_artifact(run_id, uid)
        if len(data) > MAX_AGENT_BYTES:
            raise ValueError("platform agent exceeds size limit")
        digest = hashlib.sha256(data).hexdigest()
        if digest != spec.digest:
            raise ValueError("platform agent sha256 mismatch")
        artifact = _stage_platform_agent(
            cache_root=cache_root,
            digest=digest,
            data=data,
            state_dir=state_dir,
            container_root=container_root,
        )
        specs[uid] = artifact
        logger.info(
            "Staged platform agent",
            extra={
                "uid": uid,
                "digest": digest,
                "host_path": str(artifact.host_path),
                "container_path": artifact.container_path,
            },
        )
    return specs


def _stage_platform_agent(
    *,
    cache_root: Path,
    digest: str,
    data: bytes,
    state_dir: Path,
    container_root: str,
) -> AgentArtifact:
    agent_dir = cache_root / digest
    agent_path = agent_dir / "agent.py"
    if agent_path.exists():
        container_path = _container_path_for(
            staged_path=agent_path,
            state_dir=state_dir,
            container_root=container_root,
        )
        return AgentArtifact(digest=digest, host_path=agent_path, container_path=container_path)

    agent_dir.mkdir(parents=True, exist_ok=True)
    temp_path = agent_dir / "agent.py.tmp"
    temp_path.write_bytes(data)
    try:
        _validate_agent_source(temp_path)
        temp_path.replace(agent_path)
    except Exception:
        with suppress(FileNotFoundError):
            temp_path.unlink()
        raise
    (agent_dir / "agent.sha256").write_text(digest, encoding="utf-8")
    container_path = _container_path_for(
        staged_path=agent_path,
        state_dir=state_dir,
        container_root=container_root,
    )
    return AgentArtifact(digest=digest, host_path=agent_path, container_path=container_path)


def _container_path_for(*, staged_path: Path, state_dir: Path, container_root: str) -> str:
    rel = staged_path.relative_to(state_dir)
    rel_path = PurePosixPath("/".join(rel.parts))
    root = PurePosixPath(container_root or "/")
    combined = root / rel_path
    return combined.as_posix()


def _validate_agent_source(path: Path) -> None:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("platform agent must be UTF-8 encoded") from exc
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        raise ValueError("platform agent failed bytecode compilation") from exc


def create_platform_agent_resolver(
    platform_client: PlatformPort,
) -> Callable[[UUID, EvaluationBatchSpec, Path, str], dict[int, AgentArtifact]]:
    """Create a resolver function for platform agent artifacts."""

    def resolver(
        run_id: UUID,
        batch: EvaluationBatchSpec,
        state_dir: Path,
        container_root: str,
    ) -> dict[int, AgentArtifact]:
        return resolve_platform_agent_specs(
            run_id=run_id,
            uids=batch.uids,
            artifacts=batch.artifacts,
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
