"""Protect bounded accounting and destructive recovery ownership."""
from __future__ import annotations

import os
import struct
from pathlib import Path
from unittest.mock import Mock

import pytest

from harnyx_commons.sandbox.manager import SandboxDeployment, SandboxStartError
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_validator.infrastructure.artifact_storage import (
    ArtifactStorageSandboxManager,
    ValidatorRuntimeIdentity,
    _Config,
    _Container,
    _State,
)
from harnyx_validator.infrastructure.artifact_storage_helper import DiskPool, Reply, Request, Slot


def test_all_slots_including_failed_preparations_count_toward_bound(tmp_path):
    pool = DiskPool(tmp_path, 4096)
    for index in range(15):
        pool._save(Slot(index=index, consumer=f"consumer-{index}"))
    with pytest.raises(RuntimeError, match="all 15 slots"):
        pool.prepare("sixteenth")
    assert len(pool.slots()) == 15
    assert not list(tmp_path.glob("*.img"))


@pytest.mark.parametrize("suffix", ["img", "node", "pending"])
def test_orphan_files_are_not_treated_as_available_slots(tmp_path, suffix):
    pool = DiskPool(tmp_path)
    (tmp_path / f"slot-0.{suffix}").touch()
    with pytest.raises(RuntimeError, match="unrecognized resources"):
        pool.prepare("new")


def test_preallocation_failure_keeps_owned_image_and_slot(tmp_path, monkeypatch):
    pool = DiskPool(tmp_path, 4096)

    def full(*args):
        raise OSError("disk full")

    monkeypatch.setattr(os, "posix_fallocate", full)
    with pytest.raises(OSError, match="disk full"):
        pool.prepare("consumer")
    slot = pool.slots()[0]
    assert slot.phase == "reserved"
    assert (slot.device, slot.inode) == ((tmp_path / "slot-0.img").stat().st_dev,
                                       (tmp_path / "slot-0.img").stat().st_ino)


def test_reclaim_refuses_replaced_backing_identity(tmp_path):
    pool = DiskPool(tmp_path)
    image = tmp_path / "slot-0.img"
    image.touch()
    slot = Slot(index=0, consumer="consumer", device=image.stat().st_dev, inode=image.stat().st_ino + 1)
    pool._save(slot)
    with pytest.raises(RuntimeError, match="backing identity uncertain"):
        pool.reclaim(slot)
    assert image.exists()
    assert pool.slots()[0].phase == "reclaiming"


def test_activation_does_not_arm_automatic_detachment_on_another_backing(tmp_path, monkeypatch):
    pool = DiskPool(tmp_path)
    slot = Slot(index=0, consumer="consumer", device=1, inode=2, phase="attached")
    pool._save(slot)
    (tmp_path / "slot-0.node").touch()
    status = bytearray(232)
    struct.pack_into("=QQ", status, 0, 1, 3)
    ioctl = Mock(return_value=bytes(status))
    monkeypatch.setattr("harnyx_validator.infrastructure.artifact_storage_helper.fcntl.ioctl", ioctl)

    with pytest.raises(RuntimeError, match="mapping changed"):
        pool.activate(slot)

    assert pool.slots()[0].phase == "attached"
    assert [call.args[1] for call in ioctl.call_args_list] == [0x4C05]  # Read status only; never mutate the peer.


def test_partial_reclaim_can_finish_without_reallocating_image(tmp_path, monkeypatch):
    pool = DiskPool(tmp_path)
    slot = Slot(index=0, consumer="consumer", device=1, inode=2, phase="reclaiming")
    pool._save(slot)
    monkeypatch.setattr(pool, "_matching_loops", lambda _: [])
    pool.reclaim(slot)
    assert pool.slots() == []


@pytest.mark.parametrize("failure", ["autoclear", "identity"])
def test_activation_preserves_image_when_safe_unlink_is_unconfirmed(tmp_path, monkeypatch, failure):
    pool = DiskPool(tmp_path)
    image = tmp_path / "slot-0.img"
    image.write_bytes(b"retained")
    info = image.stat()
    slot = Slot(index=0, consumer="consumer", device=info.st_dev,
                inode=info.st_ino + (failure == "identity"), phase="attached")
    pool._save(slot)
    (tmp_path / "slot-0.node").touch()
    status = bytearray(232)
    struct.pack_into("=QQ", status, 0, slot.device, slot.inode)
    struct.pack_into("=I", status, 52, 4 if failure == "identity" else 0)
    monkeypatch.setattr("harnyx_validator.infrastructure.artifact_storage_helper.fcntl.ioctl",
                        Mock(return_value=bytes(status)))

    with pytest.raises(RuntimeError, match="backing identity uncertain|automatic loop detachment unconfirmed"):
        pool.activate(slot)

    assert image.read_bytes() == b"retained"
    assert pool.slots()[0].phase == "attached"


def test_missing_image_with_live_mapping_is_retained(tmp_path, monkeypatch):
    pool = DiskPool(tmp_path)
    slot = Slot(index=0, consumer="consumer", device=1, inode=2)
    pool._save(slot)
    monkeypatch.setattr(pool, "_matching_loops", lambda _: [4])
    with pytest.raises(RuntimeError, match="still attached"):
        pool.reclaim(slot)
    assert len(pool.slots()) == 1


def manager():
    return ArtifactStorageSandboxManager(
        Mock(), ValidatorRuntimeIdentity(container_id="owner", started_at="now", image_id="sha256:image"),
        iops=32, bytes_per_second=4 * 1024**2,
    )


def test_live_incarnation_is_not_reclaimed(monkeypatch):
    storage = manager()
    labels = {"ai.harnyx.artifact-storage": "harnyx-artifact-storage",
              "ai.harnyx.artifact-storage.container": "owner", "ai.harnyx.artifact-storage.started": "now"}
    helper = _Container(Id="helper", State=_State(Running=True, StartedAt="now"), Config=_Config(Labels=labels))
    monkeypatch.setattr(storage, "_container", lambda _: helper)
    commands = []

    def run(*args, **kwargs):
        commands.append(args)
        if args[0] == "inspect":
            return '[{"Id":"owner","State":{"Running":true,"StartedAt":"now"},"Config":{"Labels":{}}}]'
        return "owner"

    monkeypatch.setattr(storage, "_run", run)
    with pytest.raises(RuntimeError, match="live validator incarnation"):
        storage.recover(sandbox_labels={}, sandbox_name_prefix="sandbox-")
    assert not any(cmd[0] == "rm" for cmd in commands)
    storage._docker_manager.cleanup_stale_sandbox_containers.assert_not_called()


def test_storage_cleanup_failure_does_not_override_container_removal(monkeypatch, caplog):
    storage = manager()
    storage._docker_manager.stop.return_value = True
    storage._deployments["container-id"] = "consumer"
    request = Mock(side_effect=RuntimeError("helper died"))
    monkeypatch.setattr(storage, "_request", request)
    assert storage.stop(SandboxDeployment(client=Mock(), identifier="container-id"))
    request.assert_called_once()


def test_unremoved_start_failure_preserves_consumer_for_later_stop(monkeypatch):
    storage = manager()
    storage._ready = True
    storage._mountpoint = "/backing"
    slot = Slot(index=0, consumer="consumer")
    monkeypatch.setattr(storage, "_prepare", lambda _: slot)
    monkeypatch.setattr(storage, "_loop_number", lambda _: 7)
    deployment = SandboxDeployment(client=Mock(), identifier="id")
    storage._docker_manager.start.side_effect = SandboxStartError("failed", unremoved_deployment=deployment)
    with pytest.raises(SandboxStartError):
        storage.start(SandboxOptions(image="image", container_name="consumer"))
    assert storage._deployments == {"id": "consumer"}


def test_unconfirmed_container_removal_never_reclaims_storage(monkeypatch):
    storage = manager()
    storage._docker_manager.stop.return_value = False
    reclaim = Mock()
    monkeypatch.setattr(storage, "_cleanup_consumer", reclaim)
    assert not storage.stop(SandboxDeployment(client=Mock(), identifier="id"))
    reclaim.assert_not_called()


def test_shutdown_waits_until_consumer_launch_is_established(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    storage = manager()
    storage._ready = True
    storage._mountpoint = "/backing"
    slot = Slot(index=0, consumer="consumer")
    monkeypatch.setattr(storage, "_prepare", lambda _: slot)
    monkeypatch.setattr(storage, "_loop_number", lambda _: 7)
    entered = threading.Event()
    release = threading.Event()
    launched = threading.Event()
    closing = threading.Event()
    reclaimed = threading.Event()

    def launch(_):
        entered.set()
        assert release.wait(5)
        launched.set()
        return SandboxDeployment(client=Mock(), identifier="id")

    def reclaim(_):
        assert launched.is_set()
        reclaimed.set()
        raise RuntimeError("consumer still exists")

    storage._docker_manager.start.side_effect = launch
    monkeypatch.setattr(storage, "_request", lambda request: Reply(slots=[slot]))
    monkeypatch.setattr(storage, "_reclaim", reclaim)

    def close():
        closing.set()
        storage.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        started = executor.submit(storage.start, SandboxOptions(image="image", container_name="consumer"))
        assert entered.wait(5)
        closed = executor.submit(close)
        assert closing.wait(5)
        assert not reclaimed.wait(0.1)
        release.set()
        assert started.result(timeout=5).identifier == "id"
        with pytest.raises(RuntimeError, match="consumer still exists"):
            closed.result(timeout=5)


def test_disconnected_client_does_not_terminate_helper_request_loop(tmp_path):
    import socket
    import threading

    from harnyx_validator.infrastructure.artifact_storage_helper import _serve_connection

    pool = DiskPool(tmp_path)
    for disconnect in (True, False):
        server, client = socket.socketpair()
        client.sendall(Request(action="owner").model_dump_json().encode() + b"\n")
        if disconnect:
            client.close()
        worker = threading.Thread(target=_serve_connection, args=(pool, server))
        worker.start()
        if not disconnect:
            with client, client.makefile("rb") as stream:
                assert Reply.model_validate_json(stream.readline()).error is None
        worker.join(timeout=5)
        assert not worker.is_alive()


@pytest.mark.parametrize("started_at, running", [("new-start", True), ("now", False)])
def test_ended_incarnation_can_be_recovered(monkeypatch, started_at, running):
    import json

    from harnyx_validator.infrastructure.artifact_storage_helper import RuntimeOwner

    storage = manager()
    inspected = [{"Id": "owner", "State": {"Running": running, "StartedAt": started_at}, "Config": {"Labels": None}}]
    monkeypatch.setattr(storage, "_run", lambda *args: json.dumps(inspected) if args[0] == "inspect" else "owner")
    storage._confirm_owner_ended(RuntimeOwner(container_id="owner", started_at="now"))


def test_owner_record_survives_helper_restart_and_requires_claim(tmp_path):
    from harnyx_validator.infrastructure.artifact_storage_helper import RuntimeOwner

    owner = RuntimeOwner(container_id="owner", started_at="now")
    pool = DiskPool(tmp_path)
    pool.dispatch(Request(action="claim", owner=owner))
    restarted = DiskPool(tmp_path)
    assert restarted.dispatch(Request(action="owner")).owner == owner
    with pytest.raises(RuntimeError, match="has not been claimed"):
        restarted.dispatch(Request(action="prepare", consumer="new"))


def test_interrupted_owner_claim_preserves_incarnation_until_reclaimed(tmp_path):
    from harnyx_validator.infrastructure.artifact_storage_helper import RuntimeOwner

    previous = RuntimeOwner(container_id="old", started_at="before")
    (tmp_path / "runtime.pending").write_text(previous.model_dump_json())
    restarted = DiskPool(tmp_path)
    assert restarted.dispatch(Request(action="owner")).owner == previous
    with pytest.raises(RuntimeError, match="has not been claimed"):
        restarted.dispatch(Request(action="prepare", consumer="new"))
    current = RuntimeOwner(container_id="new", started_at="after")
    restarted.dispatch(Request(action="claim", owner=current))
    assert DiskPool(tmp_path).dispatch(Request(action="owner")).owner == current
    assert not (tmp_path / "runtime.pending").exists()


def test_malformed_pending_owner_is_not_replaced(tmp_path):
    from pydantic import ValidationError

    (tmp_path / "runtime.pending").write_text('{"container_id":')
    with pytest.raises(ValidationError):
        DiskPool(tmp_path).dispatch(Request(action="owner"))
    assert (tmp_path / "runtime.pending").exists()


def test_creation_interrupted_before_identity_save_can_be_reclaimed(tmp_path, monkeypatch):
    pool = DiskPool(tmp_path, 4096)
    save = pool._save

    def interrupted(slot):
        if slot.inode is not None:
            raise RuntimeError("interrupted before identity save")
        save(slot)

    monkeypatch.setattr(pool, "_save", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        pool.prepare("consumer")
    restarted = DiskPool(tmp_path, 4096)
    monkeypatch.setattr(restarted, "_matching_loops", lambda _: [])
    restarted.reclaim(restarted.slots()[0])
    assert restarted.slots() == []
    assert not (tmp_path / "slot-0.img").exists()


@pytest.mark.parametrize("unexpected", ["data", "symlink", "hardlink", "node", "attached"])
def test_missing_identity_does_not_adopt_unexpected_backing(tmp_path, monkeypatch, unexpected):
    pool = DiskPool(tmp_path)
    slot = Slot(index=0, consumer="consumer")
    pool._save(slot)
    image = tmp_path / "slot-0.img"
    image.touch()
    if unexpected == "data":
        image.write_bytes(b"unexpected")
    elif unexpected == "symlink":
        image.unlink()
        (tmp_path / "other").touch()
        image.symlink_to(tmp_path / "other")
    elif unexpected == "hardlink":
        os.link(image, tmp_path / "other")
    elif unexpected == "node":
        (tmp_path / "slot-0.node").touch()
    monkeypatch.setattr(pool, "_matching_loops", lambda _: [1] if unexpected == "attached" else [])
    with pytest.raises(RuntimeError, match="retained"):
        pool.reclaim(slot)
    assert image.exists()
    assert pool.slots()[0] == slot


def test_interrupted_initial_reservation_can_be_recovered(tmp_path, monkeypatch):
    pool = DiskPool(tmp_path, 4096)
    with monkeypatch.context() as patch:
        def interrupted_rename(*args):
            raise OSError("interrupted rename")

        patch.setattr(Path, "replace", interrupted_rename)
        with pytest.raises(OSError, match="interrupted rename"):
            pool.prepare("consumer")
    assert not (tmp_path / "slot-0.img").exists()
    restarted = DiskPool(tmp_path, 4096)
    slots = restarted.slots()
    assert slots == [Slot(index=0, consumer="consumer")]
    restarted.reclaim(slots[0])
    assert restarted.slots() == []
    assert not (tmp_path / "slot-0.pending").exists()


@pytest.mark.parametrize("updates, side_effect", [
    ({"index": 1}, None),
    ({"phase": "active"}, None),
    ({"device": 1, "inode": 2}, None),
    ({}, "img"),
    ({}, "node"),
])
def test_ambiguous_pending_reservation_is_preserved(tmp_path, updates, side_effect):
    slot = Slot(index=0, consumer="consumer").model_copy(update=updates)
    pending = tmp_path / "slot-0.pending"
    pending.write_text(slot.model_dump_json())
    if side_effect:
        (tmp_path / f"slot-0.{side_effect}").touch()
    with pytest.raises(RuntimeError, match="incomplete reservation uncertain"):
        DiskPool(tmp_path).slots()
    assert pending.read_text() == slot.model_dump_json()
    assert not (tmp_path / "slot-0.json").exists()


def test_artifact_start_retains_security_args_and_applies_both_io_limits(monkeypatch):
    storage = manager()
    storage._mountpoint = "/backing"
    storage._iops = 64
    storage._bytes_per_second = 8 * 1024**2
    storage._ready = True
    storage._mountpoint = "/backing"
    slot = Slot(index=0, consumer="consumer")
    monkeypatch.setattr(storage, "_prepare", lambda _: slot)
    monkeypatch.setattr(storage, "_request", lambda _: Reply(loop_number=7))
    storage._docker_manager.start.return_value = SandboxDeployment(client=Mock(), identifier="id")
    storage.start(SandboxOptions(image="image", container_name="consumer", extra_args=("--read-only",)))
    options = storage._docker_manager.start.call_args.args[0]
    assert options.extra_args == ("--read-only",)
    assert options.block_io_limits.device == "/backing/slot-0.node"
    assert options.block_io_limits.iops == 64
    assert options.block_io_limits.bytes_per_second == 8 * 1024**2
    assert options.volumes == (("harnyx-artifact-storage-slot-0", "/sandbox/queries", "rw"),)


def test_missing_loop_identity_cleans_preparation_without_launching(monkeypatch):
    storage = manager()
    storage._ready = True
    storage._mountpoint = "/backing"
    monkeypatch.setattr(storage, "_prepare", lambda _: Slot(index=0, consumer="consumer"))
    monkeypatch.setattr(storage, "_request", lambda _: Reply())
    cleanup = Mock()
    monkeypatch.setattr(storage, "_cleanup_consumer", cleanup)
    with pytest.raises(RuntimeError, match="device identity unavailable"):
        storage.start(SandboxOptions(image="image", container_name="consumer"))
    storage._docker_manager.start.assert_not_called()
    cleanup.assert_called_once_with("consumer")


def test_loop_device_lookup_refuses_replaced_backing(tmp_path, monkeypatch):
    from harnyx_validator.infrastructure.artifact_storage_helper import Request

    pool = DiskPool(tmp_path)
    pool.claimed = True
    pool._save(Slot(index=0, consumer="consumer", device=1, inode=2))
    monkeypatch.setattr(pool, "_mapping", lambda _: (1, 99))
    with pytest.raises(RuntimeError, match="loop identity changed"):
        pool.dispatch(Request(action="device", index=0))


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("missing", [None, "riops", "wiops", "rbps", "wbps"])
def test_io_probe_rejects_missing_or_unlimited_kernel_throttle(tmp_path, version, missing):
    from harnyx_validator.infrastructure.artifact_storage_helper import verify_io_limits

    values = {"riops": "32", "wiops": "32", "rbps": "4194304", "wbps": "4194304"}
    if missing:
        values[missing] = "max" if version == 2 else "0"
    if version == 2:
        (tmp_path / "io.max").write_text("7:8 rbps=99\n7:7 " + " ".join(f"{k}={v}" for k, v in values.items()))
    else:
        (tmp_path / "blkio").mkdir()
        for key, value in values.items():
            direction = "read" if key.startswith("r") else "write"
            (tmp_path / "blkio" / f"blkio.throttle.{direction}_{key[1:]}_device").write_text(f"7:7 {value}\n")
    if missing:
        with pytest.raises(RuntimeError, match="not enforced"):
            verify_io_limits(7, 32, 4194304, tmp_path)
    else:
        verify_io_limits(7, 32, 4194304, tmp_path)


def test_probe_cleanup_does_not_remove_unacquired_helper():
    storage = ArtifactStorageSandboxManager(
        Mock(), ValidatorRuntimeIdentity(container_id="owner", started_at="now", image_id="image"),
        iops=32, bytes_per_second=4194304,
    )
    storage._run = Mock()
    with pytest.raises(RuntimeError, match="acquisition unconfirmed"):
        storage.finish_probe()
    storage._run.assert_not_called()


def test_probe_cleanup_retains_backing_on_incomplete_claim():
    storage = ArtifactStorageSandboxManager(
        Mock(), ValidatorRuntimeIdentity(container_id="owner", started_at="now", image_id="image"),
        iops=32, bytes_per_second=4194304,
    )
    storage._helper_started = True
    storage._run = Mock()
    with pytest.raises(RuntimeError, match="claim incomplete"):
        storage.finish_probe()
    storage._run.assert_called_once_with("rm", "-f", storage._helper)


def test_helper_failure_reply_preserves_trace_and_command_stderr():
    import socket
    import subprocess
    import threading

    from harnyx_validator.infrastructure.artifact_storage_helper import MAX_MESSAGE, _serve_connection

    pool = Mock()
    pool.dispatch.side_effect = subprocess.CalledProcessError(1, ["mkfs.ext4"], stderr=b"format rejected")
    server, client = socket.socketpair()
    thread = threading.Thread(target=_serve_connection, args=(pool, server))
    thread.start()
    try:
        with client, client.makefile("rwb") as stream:
            stream.write(Request(action="list").model_dump_json().encode() + b"\n")
            stream.flush()
            reply = Reply.model_validate_json(stream.readline(MAX_MESSAGE))
        assert "CalledProcessError" in reply.error_trace
        assert "format rejected" in reply.error_trace
    finally:
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_uncertain_probe_creation_retains_storage_through_finish(monkeypatch):
    from harnyx_validator.infrastructure.artifact_storage import _Volume

    storage = manager()
    monkeypatch.setattr(storage, "_container", lambda _: None)
    monkeypatch.setattr(storage, "_volume", lambda _: _Volume(
        Name="backing", Mountpoint="/owned", Labels={"ai.harnyx.artifact-storage": "harnyx-artifact-storage"},
    ))
    commands = Mock(return_value="template")
    monkeypatch.setattr(storage, "_run", commands)
    requests = Mock(return_value=Reply())
    monkeypatch.setattr(storage, "_request", requests)
    monkeypatch.setattr(storage, "_prepare", lambda name: Slot(index=0, consumer=name))
    monkeypatch.setattr(storage, "_loop_number", lambda _: 7)
    reclaim = Mock()
    monkeypatch.setattr(storage, "_reclaim", reclaim)
    storage._docker_manager.recreate_with_block_io.side_effect = SandboxStartError(
        "response lost", unremoved_deployment=SandboxDeployment(client=Mock(), identifier="sandbox-storage-probe"),
    )
    with pytest.raises(SandboxStartError):
        storage.recover(sandbox_labels={}, sandbox_name_prefix="sandbox-")
    with pytest.raises(RuntimeError, match="creation outcome uncertain"):
        storage.finish_probe()
    reclaim.assert_not_called()
    assert not any(call.args[0].action == "release" for call in requests.call_args_list)
    assert not any(call.args[0] == "rm" for call in commands.call_args_list)
