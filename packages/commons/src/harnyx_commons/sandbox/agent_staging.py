"""Helpers for staging agent scripts into a shared state directory."""

from __future__ import annotations

import hashlib
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DEFAULT_AGENT_FILENAME = "agent.py"
MAX_AGENT_BYTES = 256_000
_LOG_SNIPPET_LIMIT = 512


@dataclass(frozen=True, slots=True)
class AgentArtifact:
    """Represents a staged agent script artifact."""

    content_hash: str
    host_path: Path
    container_path: str


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
        raise ValueError("agent source is empty")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    if len(data) > max_bytes:
        raise ValueError(f"agent exceeds size limit (size_bytes={len(data)} max_bytes={max_bytes})")

    content_hash = hashlib.sha256(data).hexdigest()
    agent_path, container_path = agent_paths(
        state_dir=state_dir,
        container_root=container_root,
        namespace=namespace,
        key=key,
        filename=filename,
    )

    if agent_path.exists():
        existing_hash = hashlib.sha256(agent_path.read_bytes()).hexdigest()
        if existing_hash != content_hash:
            raise RuntimeError(
                f"staged agent content hash mismatch for {agent_path}: expected={content_hash} actual={existing_hash}"
            )
        return AgentArtifact(
            content_hash=content_hash,
            host_path=agent_path,
            container_path=container_path,
        )

    agent_dir = agent_path.parent
    agent_dir.mkdir(parents=True, exist_ok=True)
    temp_path = agent_dir / f"{filename}.tmp"
    temp_path.write_bytes(data)
    try:
        _validate_agent_source(temp_path)
        temp_path.replace(agent_path)
    except Exception:
        with suppress(FileNotFoundError):
            temp_path.unlink()
        raise

    agent_path.chmod(0o644)
    (agent_dir / "agent.sha256").write_text(content_hash, encoding="utf-8")

    return AgentArtifact(
        content_hash=content_hash,
        host_path=agent_path,
        container_path=container_path,
    )


def _validate_agent_source(path: Path) -> None:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("agent must be UTF-8 encoded") from exc
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        snippet = source[:_LOG_SNIPPET_LIMIT].replace("\n", "\\n")
        raise ValueError(f"agent failed bytecode compilation: snippet={snippet!r}") from exc


__all__ = [
    "AgentArtifact",
    "DEFAULT_AGENT_FILENAME",
    "MAX_AGENT_BYTES",
    "agent_paths",
    "stage_agent_source",
]
