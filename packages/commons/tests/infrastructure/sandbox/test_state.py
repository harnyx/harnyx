from __future__ import annotations

from pathlib import Path

import pytest

from harnyx_commons.sandbox.options import ReadonlyVolumeSubpathMount
from harnyx_commons.sandbox.state import resolve_artifact_state_mount


def test_artifact_mount_uses_only_selected_directory_for_path_backed_state(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    artifact_path = state_dir / "agents" / "selected" / "agent.py"

    mount = resolve_artifact_state_mount(
        artifact_host_path=artifact_path,
        state_dir=state_dir,
        artifact_container_path="/workspace/.harnyx_state/agents/selected/agent.py",
        mount_source=str(state_dir),
    )

    assert mount.bind_volumes == (
        (str(state_dir / "agents" / "selected"), "/workspace/.harnyx_state/agents/selected", "ro"),
    )
    assert mount.volume_subpaths == ()


def test_artifact_mount_uses_only_selected_subpath_for_named_volume() -> None:
    state_dir = Path("/workspace/.harnyx_state")
    artifact_path = state_dir / "agents" / "selected" / "agent.py"

    mount = resolve_artifact_state_mount(
        artifact_host_path=artifact_path,
        state_dir=state_dir,
        artifact_container_path="/workspace/.harnyx_state/agents/selected/agent.py",
        mount_source="validator-state",
    )

    assert mount.bind_volumes == ()
    assert mount.volume_subpaths == (
        ReadonlyVolumeSubpathMount(
            source="validator-state",
            subpath="agents/selected",
            target="/workspace/.harnyx_state/agents/selected",
        ),
    )


def test_artifact_mount_rejects_artifact_outside_state(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_artifact_state_mount(
            artifact_host_path=tmp_path / "outside" / "agent.py",
            state_dir=tmp_path / "state",
            artifact_container_path="/sandbox/agent.py",
            mount_source=str(tmp_path / "state"),
        )


@pytest.mark.parametrize(
    ("source", "subpath", "target"),
    [
        ("/host/path", "agents/selected", "/sandbox"),
        ("state", "../peer", "/sandbox"),
        ("state", "agents/selected", "relative"),
    ],
)
def test_volume_subpath_mount_rejects_unsafe_fields(source: str, subpath: str, target: str) -> None:
    with pytest.raises(ValueError):
        ReadonlyVolumeSubpathMount(source=source, subpath=subpath, target=target)
