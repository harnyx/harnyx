"""Real image and kernel acceptance for bounded miner workspaces."""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from harnyx_validator.infrastructure.artifact_storage_helper import Request, RuntimeOwner

pytestmark = [pytest.mark.integration, pytest.mark.anyio("asyncio"), pytest.mark.artifact_storage]

_AGENT = b'''
import os
import tempfile
from pathlib import Path
from harnyx_miner_sdk.decorators import entrypoint

Path("module.txt").write_text("initialized")

@entrypoint("storage")
async def storage(request: dict[str, object]) -> dict[str, object]:
    if request.get("io_probe"):
        import mmap
        import time
        with open("io-data", "wb") as stream:
            started = time.monotonic()
            stream.write(b"x" * (8 * 1024**2))
            stream.flush()
            os.fsync(stream.fileno())
            write_seconds = time.monotonic() - started
        fd = os.open("io-data", os.O_RDONLY | os.O_DIRECT)
        buffer = mmap.mmap(-1, 4096)
        try:
            started = time.monotonic()
            for _ in range(128):
                os.readv(fd, [buffer])
            read_seconds = time.monotonic() - started
        finally:
            buffer.close()
            os.close(fd)
        return {"write_seconds": write_seconds, "read_seconds": read_seconds}
    full = False
    written = 0
    if request.get("fill"):
        try:
            with open("large", "wb", buffering=0) as stream:
                while True:
                    written += stream.write(b"x" * (1024 * 1024))
        except OSError as exc:
            if exc.errno != 28:
                raise
            full = True
        Path("large").unlink()
    with tempfile.NamedTemporaryFile() as stream:
        stream.write(b"restored")
        stream.flush()
        parent = str(Path(stream.name).parent)
    Path("small").write_text("ok")
    return {"cwd": os.getcwd(), "temp": parent, "full": full, "written": written,
            "module": Path("module.txt").read_text(), "uid": os.getuid()}
'''


def launch(sandbox_launcher):
    return sandbox_launcher("validator.tests.integration.sandbox.recovery_agent", source=_AGENT, evaluation_policy=True)


async def invoke(deployment, **payload):
    return await deployment.client.invoke(
        "storage", payload=payload, context={"time_budget": {"limit_seconds": 120.0}},
        token=uuid4().hex, session_id=uuid4(),
    )


async def test_artifact_capacity_isolated_and_deletion_restores_writes(sandbox_launcher, artifact_storage_manager):
    first, second = launch(sandbox_launcher), launch(sandbox_launcher)
    results = await asyncio.gather(invoke(first, fill=True), invoke(second))
    assert results[0]["full"]
    assert 0 < results[0]["written"] < 128 * 1024**2
    assert not results[1]["full"]
    assert results[0]["cwd"] != results[1]["cwd"]
    for result in results:
        assert result["cwd"] == result["temp"]
        assert result["module"] == "initialized" and result["uid"] == 1000
    assert len(artifact_storage_manager._request(Request(action="list")).slots) == 2


async def test_helper_loss_keeps_consumer_io_and_blocks_new_allocation(sandbox_launcher, artifact_storage_manager):
    deployment = launch(sandbox_launcher)
    manager = artifact_storage_manager
    slot = manager._request(Request(action="list")).slots[0]
    manager._run("kill", manager._helper)
    try:
        result = await invoke(deployment)
        assert result["module"] == "initialized"
        with pytest.raises(RuntimeError):
            manager._prepare("must-not-allocate")
        # Bypass storage cleanup: the kernel must detach even with its helper dead.
        assert manager._docker_manager.stop(deployment)
    finally:
        # Explicit test recovery of the same trusted helper; production requires
        # validator restart and incarnation fencing instead of silently retrying.
        manager._run("start", manager._helper)
        for _ in range(100):
            try:
                manager._request(Request(action="owner"))
                manager._request(Request(action="claim", owner=RuntimeOwner(
                    container_id=manager._identity.container_id, started_at=manager._identity.started_at,
                )))
                break
            except RuntimeError:
                await asyncio.sleep(0.05)

    import json

    remaining = manager._run(
        "exec", manager._helper, "/workspace/.venv/bin/python", "-c",
        "import json,sys; from harnyx_validator.infrastructure.artifact_storage_helper import DiskPool,Slot; "
        "p=DiskPool(); s=Slot.model_validate_json(sys.argv[1]); "
        "print(json.dumps({'loops':p._matching_loops(s), 'image':p._path(s.index,'img').exists()}))",
        slot.model_dump_json(),
    )
    assert json.loads(remaining) == {"loops": [], "image": False}
    # Kernel reclamation leaves accounting for ownership-fenced normal cleanup.
    assert manager.stop(deployment)
    assert manager._request(Request(action="list")).slots == []


async def test_same_container_restart_reclaims_attached_allocation(artifact_storage_manager):
    import json

    from harnyx_validator.infrastructure.artifact_storage import ValidatorRuntimeIdentity

    manager = artifact_storage_manager
    manager._prepare(f"{manager._pool}-sandbox-never-launched")
    old_identity = manager._identity
    manager._run("kill", manager._helper)
    manager._run("restart", old_identity.container_id)
    inspected = json.loads(manager._run("inspect", old_identity.container_id))[0]
    assert inspected["State"]["StartedAt"] != old_identity.started_at
    manager._identity = ValidatorRuntimeIdentity(
        container_id=old_identity.container_id, started_at=inspected["State"]["StartedAt"],
        image_id=old_identity.image_id,
    )
    manager.recover(sandbox_labels={"storage-test": manager._pool}, sandbox_name_prefix=f"{manager._pool}-sandbox-")
    assert manager._request(Request(action="list")).slots == []


@pytest.mark.parametrize("artifact_storage_manager", [(32, 4), (64, 8)], indirect=True)
async def test_loop_io_limits_throttle_miner_buffered_writes_and_direct_reads(
    sandbox_launcher, artifact_storage_manager,
):
    manager = artifact_storage_manager
    deployment = launch(sandbox_launcher)
    result = await invoke(deployment, io_probe=True)
    # Broad lower bounds catch missing throttles without imposing a host-speed upper bound.
    assert result["write_seconds"] >= 0.6 * (8 * 1024**2 / manager._bytes_per_second)
    assert result["read_seconds"] >= 0.6 * (128 / manager._iops)
    slot = manager._request(Request(action="list")).slots[0]
    device = manager._io_limits(slot).device
    # The timings above execute in the real miner. Docker inspect also confirms all four launch limits.
    import json

    host = json.loads(manager._run("inspect", deployment.identifier))[0]["HostConfig"]
    for direction in ("Read", "Write"):
        assert host[f"BlkioDevice{direction}IOps"] == [{"Path": device, "Rate": manager._iops}]
        assert host[f"BlkioDevice{direction}Bps"] == [{"Path": device, "Rate": manager._bytes_per_second}]


async def test_background_probe_reports_real_cleanup(artifact_storage_manager, monkeypatch):
    from harnyx_commons.sandbox.docker import DockerSandboxManager
    from harnyx_validator.infrastructure.artifact_storage import ArtifactStorageSandboxManager
    from harnyx_validator.runtime.storage_probe import StorageProbeWorker

    owner = artifact_storage_manager
    probe = ArtifactStorageSandboxManager(
        DockerSandboxManager(), owner._identity, iops=32, bytes_per_second=4 * 1024**2,
        pool_name=f"{owner._pool}-probe", image_bytes=128 * 1024**2,
    )
    from harnyx_validator.runtime import storage_probe

    monkeypatch.setattr(storage_probe, "PROBE_POOL", probe._pool)
    monkeypatch.setattr(storage_probe, "PROBE_LABELS", {"storage-test": probe._pool})
    reports = []
    worker = StorageProbeWorker(lambda: probe, reports.append, iops=32, bytes_per_second=4 * 1024**2)
    try:
        await asyncio.to_thread(worker._run)
        assert len(reports) == 1
        assert reports[0].status == "succeeded", reports[0].model_dump_json()
        assert reports[0].cleanup == "succeeded"
        assert probe._container(probe._helper) is None
        assert probe._volume(f"{probe._pool}-slot-0") is None
        owner._run("run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
                   "--mount", f"type=volume,src={probe._backing},dst=/check,readonly",
                   "--entrypoint", "/bin/sh", owner._identity.image_id,
                   "-c", "test -z \"$(find /check -name '*.img' -o -name '*.node' -o -name 'slot-*')\"")
    finally:
        owner._run("volume", "rm", probe._backing)
