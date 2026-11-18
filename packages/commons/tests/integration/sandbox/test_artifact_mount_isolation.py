from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from harnyx_commons.sandbox.docker import DockerSandboxManager
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_commons.sandbox.state import resolve_artifact_state_mount

pytestmark = pytest.mark.integration

_IMAGE = os.getenv("SANDBOX_IMAGE", "local/harnyx-sandbox:0.1.0-dev")
_CONTAINER_ROOT = "/workspace/.harnyx_state"


def _run_isolation_probe(options: SandboxOptions) -> dict[str, object]:
    docker = shutil.which("docker") or "docker"
    manager = DockerSandboxManager(docker_binary=docker)
    args = [docker, "run", "--rm"]
    manager._add_volumes(args, options)
    args.extend(
        [
            "--entrypoint",
            "python",
            _IMAGE,
            "-c",
            (
                "import json,pathlib; root=pathlib.Path('/workspace/.harnyx_state'); "
                "selected=root/'selected/agent.py'; writable=True; "
                "\ntry: selected.write_text('changed')"
                "\nexcept OSError: writable=False"
                "\nprint(json.dumps({'selected': selected.read_text(), "
                "'peer_visible': (root/'peer/agent.py').exists(), "
                "'result_visible': (root/'results/output.json').exists(), 'writable': writable}))"
            ),
        ]
    )
    result = subprocess.run(args, check=True, capture_output=True, text=True)  # noqa: S603
    return json.loads(result.stdout)


def _assert_isolated(result: dict[str, object]) -> None:
    assert result == {
        "selected": "selected-source",
        "peer_visible": False,
        "result_visible": False,
        "writable": False,
    }


def test_path_backed_mount_exposes_only_selected_artifact_directory(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    selected = state_dir / "selected" / "agent.py"
    peer = state_dir / "peer" / "agent.py"
    result = state_dir / "results" / "output.json"
    for path, content in ((selected, "selected-source"), (peer, "peer-source"), (result, "synthetic-result")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    mount = resolve_artifact_state_mount(
        artifact_host_path=selected,
        state_dir=state_dir,
        artifact_container_path=f"{_CONTAINER_ROOT}/selected/agent.py",
        mount_source=str(state_dir),
    )

    _assert_isolated(
        _run_isolation_probe(
            SandboxOptions(
                image=_IMAGE,
                container_name="artifact-isolation-path",
                volumes=mount.bind_volumes,
                volume_subpath_mounts=mount.volume_subpaths,
            )
        )
    )


def test_named_volume_mount_exposes_only_selected_artifact_directory() -> None:
    docker = shutil.which("docker") or "docker"
    volume = f"harnyx-artifact-isolation-{uuid4().hex}"
    subprocess.run([docker, "volume", "create", volume], check=True, capture_output=True, text=True)  # noqa: S603
    try:
        subprocess.run(  # noqa: S603
            [
                docker,
                "run",
                "--rm",
                "--user",
                "0",
                "-v",
                f"{volume}:/state",
                "--entrypoint",
                "sh",
                _IMAGE,
                "-c",
                (
                    "mkdir -p /state/selected /state/peer /state/results && "
                    "printf selected-source >/state/selected/agent.py && "
                    "printf peer-source >/state/peer/agent.py && "
                    "printf synthetic-result >/state/results/output.json && chown -R 1000:1000 /state"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        state_dir = Path(_CONTAINER_ROOT)
        mount = resolve_artifact_state_mount(
            artifact_host_path=state_dir / "selected" / "agent.py",
            state_dir=state_dir,
            artifact_container_path=f"{_CONTAINER_ROOT}/selected/agent.py",
            mount_source=volume,
        )

        _assert_isolated(
            _run_isolation_probe(
                SandboxOptions(
                    image=_IMAGE,
                    container_name="artifact-isolation-volume",
                    volumes=mount.bind_volumes,
                    volume_subpath_mounts=mount.volume_subpaths,
                )
            )
        )
    finally:
        subprocess.run([docker, "volume", "rm", volume], check=True, capture_output=True, text=True)  # noqa: S603
