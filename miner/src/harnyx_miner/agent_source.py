from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

from harnyx_commons.sandbox.agent_staging import MAX_AGENT_BYTES
from harnyx_miner_sdk.decorators import clear_entrypoints, entrypoint_exists


def require_existing_agent_path(raw_path: str, *, label: str = "agent path") -> Path:
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    return path


def load_agent_bytes(agent_path: Path) -> bytes:
    content = agent_path.read_bytes()
    return validate_agent_bytes(content, label=str(agent_path))


def load_agent_query_entrypoint(agent_path: Path) -> None:
    clear_entrypoints()
    try:
        runpy.run_path(str(agent_path))
    except Exception:
        clear_entrypoints()
        raise
    if not entrypoint_exists("query"):
        clear_entrypoints()
        raise RuntimeError("agent did not register entrypoint 'query'")


def validate_agent_query_entrypoint(agent_path: Path) -> None:
    try:
        load_agent_query_entrypoint(agent_path)
    finally:
        clear_entrypoints()


def load_submittable_agent_bytes(agent_path: Path) -> bytes:
    content = load_agent_bytes(agent_path)
    validate_agent_query_entrypoint(agent_path)
    return content


def validate_agent_bytes(agent_bytes: bytes, *, label: str = "agent script") -> bytes:
    if len(agent_bytes) > MAX_AGENT_BYTES:
        raise ValueError(f"{label} exceeds {MAX_AGENT_BYTES} bytes ({len(agent_bytes)} bytes)")
    try:
        source = agent_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8 encoded") from exc
    try:
        compile(source, label, "exec")
    except SyntaxError as exc:
        raise ValueError(f"{label} failed bytecode compilation") from exc
    return agent_bytes


def agent_sha256(agent_bytes: bytes) -> str:
    return hashlib.sha256(agent_bytes).hexdigest()


__all__ = [
    "MAX_AGENT_BYTES",
    "agent_sha256",
    "load_agent_query_entrypoint",
    "load_agent_bytes",
    "load_submittable_agent_bytes",
    "require_existing_agent_path",
    "validate_agent_query_entrypoint",
    "validate_agent_bytes",
]
