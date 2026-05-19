from __future__ import annotations

from pathlib import Path

import pytest

from harnyx_commons.sandbox.agent_staging import MAX_AGENT_BYTES, AgentSourceValidationError, stage_agent_source


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_stage_agent_source_normalizes_bind_mount_permissions(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o700)
    data = b"print('hello')\n"

    artifact = stage_agent_source(
        state_dir=state_dir,
        container_root="/workspace/.harnyx_state",
        namespace="local_eval_agents",
        key="artifact-1",
        data=data,
    )

    checksum_path = artifact.host_path.parent / "agent.sha256"

    state_dir.chmod(0o700)
    artifact.host_path.parent.parent.chmod(0o700)
    artifact.host_path.parent.chmod(0o700)
    artifact.host_path.chmod(0o600)
    checksum_path.chmod(0o600)

    reused = stage_agent_source(
        state_dir=state_dir,
        container_root="/workspace/.harnyx_state",
        namespace="local_eval_agents",
        key="artifact-1",
        data=data,
    )

    assert reused == artifact
    assert _mode(state_dir) == 0o755
    assert _mode(artifact.host_path.parent.parent) == 0o755
    assert _mode(artifact.host_path.parent) == 0o755
    assert _mode(artifact.host_path) == 0o644
    assert _mode(checksum_path) == 0o644


def test_stage_agent_source_rejects_empty_source_as_script_validation(tmp_path: Path) -> None:
    with pytest.raises(AgentSourceValidationError, match="agent source is empty"):
        stage_agent_source(
            state_dir=tmp_path,
            container_root="/workspace/.harnyx_state",
            namespace="local_eval_agents",
            key="artifact-1",
            data=b"",
        )


def test_stage_agent_source_rejects_oversized_source_as_script_validation(tmp_path: Path) -> None:
    with pytest.raises(AgentSourceValidationError, match="agent exceeds size limit"):
        stage_agent_source(
            state_dir=tmp_path,
            container_root="/workspace/.harnyx_state",
            namespace="local_eval_agents",
            key="artifact-1",
            data=b"x" * 5,
            max_bytes=4,
        )


def test_default_max_agent_bytes_is_one_mb() -> None:
    assert MAX_AGENT_BYTES == 1_000_000


def test_stage_agent_source_accepts_default_max_size(tmp_path: Path) -> None:
    source = b"#" * (MAX_AGENT_BYTES - 1) + b"\n"

    artifact = stage_agent_source(
        state_dir=tmp_path,
        container_root="/workspace/.harnyx_state",
        namespace="local_eval_agents",
        key="exact-limit",
        data=source,
    )

    assert len(source) == MAX_AGENT_BYTES
    assert artifact.host_path.read_bytes() == source


def test_stage_agent_source_rejects_default_max_size_plus_one(tmp_path: Path) -> None:
    source = b"#" * MAX_AGENT_BYTES + b"\n"

    with pytest.raises(AgentSourceValidationError, match="agent exceeds size limit"):
        stage_agent_source(
            state_dir=tmp_path,
            container_root="/workspace/.harnyx_state",
            namespace="local_eval_agents",
            key="over-limit",
            data=source,
        )


def test_stage_agent_source_rejects_non_utf8_source_as_script_validation(tmp_path: Path) -> None:
    with pytest.raises(AgentSourceValidationError, match="agent must be UTF-8 encoded"):
        stage_agent_source(
            state_dir=tmp_path,
            container_root="/workspace/.harnyx_state",
            namespace="local_eval_agents",
            key="artifact-1",
            data=b"\xff",
        )


def test_stage_agent_source_rejects_syntax_error_as_script_validation(tmp_path: Path) -> None:
    with pytest.raises(AgentSourceValidationError, match="agent failed bytecode compilation"):
        stage_agent_source(
            state_dir=tmp_path,
            container_root="/workspace/.harnyx_state",
            namespace="local_eval_agents",
            key="artifact-1",
            data=b"def broken(:\n",
        )
