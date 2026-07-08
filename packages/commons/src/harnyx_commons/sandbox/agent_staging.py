"""Helpers for staging agent scripts into a shared state directory."""

from __future__ import annotations

import hashlib
import os
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DEFAULT_AGENT_FILENAME = "agent.py"
MAX_AGENT_BYTES = 1_000_000
_LOG_SNIPPET_LIMIT = 512
_DIRECTORY_MODE = 0o755
_FILE_MODE = 0o644


@dataclass(frozen=True, slots=True)
class AgentArtifact:
    """Represents a staged agent script artifact."""

    content_hash: str
    host_path: Path
    container_path: str


class AgentSourceValidationError(ValueError):
    """Raised when miner-provided source bytes violate the script contract."""


def _container_path_for(*, staged_path: Path, state_dir: Path, container_root: str) -> str:
    rel = staged_path.relative_to(state_dir)
    rel_path = PurePosixPath("/".join(rel.parts))
    root = PurePosixPath(container_root or "/")
    combined = root / rel_path
    return combined.as_posix()


def agent_paths(
    *,
    state_dir: Path,
    container_root: str,
    namespace: str,
    key: str,
    filename: str = DEFAULT_AGENT_FILENAME,
) -> tuple[Path, str]:
    """Return (host_path, container_path) for a staged agent."""

    host_path = state_dir / namespace / key / filename
    container_path = _container_path_for(
        staged_path=host_path,
        state_dir=state_dir,
        container_root=container_root,
    )
    return host_path, container_path


def stage_agent_source(
    *,
    state_dir: Path,
    container_root: str,
    namespace: str,
    key: str,
    data: bytes,
    filename: str = DEFAULT_AGENT_FILENAME,
    max_bytes: int = MAX_AGENT_BYTES,
) -> AgentArtifact:
    """Stage an agent script into `state_dir` and return its resolved paths.

    The sandbox container must bind-mount the same `state_dir` at `container_root`.
    """

    if not namespace:
        raise ValueError("namespace must be provided")
    if not key:
        raise ValueError("key must be provided")
    if not data:
        raise AgentSourceValidationError("agent source is empty")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    if len(data) > max_bytes:
        raise AgentSourceValidationError(f"agent exceeds size limit (size_bytes={len(data)} max_bytes={max_bytes})")

    content_hash = hashlib.sha256(data).hexdigest()
    agent_path, container_path = agent_paths(
        state_dir=state_dir,
        container_root=container_root,
        namespace=namespace,
        key=key,
        filename=filename,
    )
    checksum_path = agent_path.parent / "agent.sha256"

    if agent_path.exists():
        _verify_existing_agent(agent_path=agent_path, content_hash=content_hash)
        _normalize_staged_permissions(
            state_dir=state_dir,
            agent_path=agent_path,
            extra_files=(checksum_path,),
        )
        return AgentArtifact(
            content_hash=content_hash,
            host_path=agent_path,
            container_path=container_path,
        )

    agent_dir = agent_path.parent
    agent_dir.mkdir(parents=True, exist_ok=True)
    temp_path = agent_dir / f".{filename}.{uuid.uuid4().hex}.tmp"
    temp_path.write_bytes(data)
    try:
        _validate_agent_source(temp_path)
        try:
            os.link(temp_path, agent_path)
        except FileExistsError:
            _verify_existing_agent(agent_path=agent_path, content_hash=content_hash)
    except Exception:
        with suppress(FileNotFoundError):
            temp_path.unlink()
        raise
    else:
        temp_path.unlink()

    checksum_path.write_text(content_hash, encoding="utf-8")
    _normalize_staged_permissions(
        state_dir=state_dir,
        agent_path=agent_path,
        extra_files=(checksum_path,),
    )

    return AgentArtifact(
        content_hash=content_hash,
        host_path=agent_path,
        container_path=container_path,
    )


def _normalize_staged_permissions(
    *,
    state_dir: Path,
    agent_path: Path,
    extra_files: tuple[Path, ...] = (),
) -> None:
    current = state_dir
    current.chmod(_DIRECTORY_MODE)
    relative_dir = agent_path.parent.relative_to(state_dir)
    for part in relative_dir.parts:
        current = current / part
        current.chmod(_DIRECTORY_MODE)

    agent_path.chmod(_FILE_MODE)
    for extra_file in extra_files:
        if extra_file.exists():
            extra_file.chmod(_FILE_MODE)


def _verify_existing_agent(*, agent_path: Path, content_hash: str) -> None:
    existing_hash = hashlib.sha256(agent_path.read_bytes()).hexdigest()
    if existing_hash != content_hash:
        raise RuntimeError(
            f"staged agent content hash mismatch for {agent_path}: expected={content_hash} actual={existing_hash}"
        )


def _validate_agent_source(path: Path) -> None:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise AgentSourceValidationError("agent must be UTF-8 encoded") from exc
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        snippet = source[:_LOG_SNIPPET_LIMIT].replace("\n", "\\n")
        raise AgentSourceValidationError(f"agent failed bytecode compilation: snippet={snippet!r}") from exc


__all__ = [
    "AgentArtifact",
    "AgentSourceValidationError",
    "DEFAULT_AGENT_FILENAME",
    "MAX_AGENT_BYTES",
    "agent_paths",
    "stage_agent_source",
]
