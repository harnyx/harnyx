"""Trusted, internal disk-image allocator. Never imported by the validator server.

The Docker manager owns mounts and consumers; this process owns backing files
and loop mappings. Its root-only Unix socket is accessible only in the helper.
"""
from __future__ import annotations

import errno
import fcntl
import os
import socket
import stat
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SLOT_COUNT = 15
IMAGE_BYTES = 3 * 1024**3
ROOT = Path("/backing")
SOCKET = ROOT / "control.sock"
MAX_MESSAGE = 65536
_LOOP_GET_STATUS64 = 0x4C05
_LOOP_SET_STATUS64 = 0x4C04
_LO_FLAGS_AUTOCLEAR = 4
_LO_FLAGS_OFFSET = 52  # offsetof(struct loop_info64, lo_flags), Linux loop.h


class Slot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    index: int = Field(ge=0, lt=SLOT_COUNT)
    consumer: str = Field(min_length=1, max_length=200)
    device: int | None = None
    inode: int | None = None
    phase: Literal["reserved", "attached", "active", "reclaiming"] = "reserved"


class RuntimeOwner(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    container_id: str = Field(min_length=1, max_length=200)
    started_at: str = Field(min_length=1, max_length=100)


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["owner", "claim", "release", "list", "prepare", "activate", "reclaim", "device"]
    owner: RuntimeOwner | None = None
    consumer: str = Field(default="", max_length=200)
    index: int = Field(default=0, ge=0, lt=SLOT_COUNT)


class Reply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slots: list[Slot] = Field(default_factory=list, max_length=SLOT_COUNT)
    error: str | None = None
    error_trace: str | None = Field(default=None, max_length=12000)
    owner: RuntimeOwner | None = None
    loop_number: int | None = Field(default=None, ge=0)


def _run(*args: str) -> None:
    subprocess.run(args, check=True, capture_output=True, timeout=120)  # noqa: S603


def _sync_directory(root: Path) -> None:
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class DiskPool:
    """Serialized operations, fixed filenames and write-ahead ownership records."""

    def __init__(self, root: Path = ROOT, image_bytes: int = IMAGE_BYTES) -> None:
        self.root = root
        if not 0 < image_bytes <= IMAGE_BYTES:
            raise ValueError("artifact image size must be positive and at most 3 GiB")
        self.image_bytes = image_bytes
        self.claimed = False

    def _path(self, index: int, suffix: str) -> Path:
        return self.root / f"slot-{index}.{suffix}"

    def _save(self, slot: Slot) -> None:
        temporary = self._path(slot.index, "pending")
        with temporary.open("w") as stream:
            stream.write(slot.model_dump_json())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self._path(slot.index, "json"))
        _sync_directory(self.root)

    def slots(self) -> list[Slot]:
        slots = []
        for index in range(SLOT_COUNT):
            manifest = self._path(index, "json")
            if manifest.exists():
                slot = Slot.model_validate_json(manifest.read_bytes())
                if slot.index != index:
                    raise RuntimeError(f"slot {index}: manifest identity mismatch")
                slots.append(slot)
            elif self._path(index, "pending").exists():
                pending = self._path(index, "pending")
                try:
                    slot = Slot.model_validate_json(pending.read_bytes())
                except ValueError as exc:
                    raise RuntimeError(f"slot {index}: unrecognized resources; operator inspection required") from exc
                if (
                    slot.index != index or slot.phase != "reserved"
                    or slot.device is not None or slot.inode is not None
                    or any(self._path(index, suffix).exists() for suffix in ("img", "node"))
                ):
                    raise RuntimeError(f"slot {index}: incomplete reservation uncertain; retained")
                # The first reservation precedes all image/device side effects.
                # Commit it so normal ownership-fenced recovery can reclaim it.
                pending.replace(manifest)
                _sync_directory(self.root)
                slots.append(slot)
            elif any(self._path(index, suffix).exists() for suffix in ("img", "node")):
                raise RuntimeError(f"slot {index}: unrecognized resources; operator inspection required")
        return slots

    def _node(self, path: Path, number: int) -> None:
        if path.exists():
            info = path.lstat()
            if not stat.S_ISBLK(info.st_mode):
                raise RuntimeError(f"unexpected non-device at {path}")
            path.unlink()
        os.mknod(path, stat.S_IFBLK | 0o600, os.makedev(7, number))

    def _mapping(self, node: Path) -> tuple[int, int] | None:
        fd = os.open(node, os.O_RDONLY)
        try:
            try:
                status = fcntl.ioctl(fd, _LOOP_GET_STATUS64, bytes(232))
            except OSError as exc:
                if exc.errno == errno.ENXIO:
                    return None
                raise
            device, inode = struct.unpack_from("=QQ", status)
            return device, inode
        finally:
            os.close(fd)

    def _matching_loops(self, slot: Slot) -> list[int]:
        matches = []
        probe = self.root / "inspect.node"
        try:
            for device in Path("/sys/block").glob("loop[0-9]*"):
                number = int(device.name[4:])
                self._node(probe, number)
                try:
                    mapping = self._mapping(probe)
                except OSError as exc:
                    if exc.errno in (errno.ENODEV, errno.ENOENT):
                        continue
                    raise
                if mapping == (slot.device, slot.inode):
                    matches.append(number)
        finally:
            probe.unlink(missing_ok=True)
        return matches

    def prepare(self, consumer: str) -> Slot:
        if not consumer:
            raise ValueError("consumer identity is required")
        occupied = {slot.index for slot in self.slots()}
        available = next((i for i in range(SLOT_COUNT) if i not in occupied), None)
        if available is None:
            raise RuntimeError("artifact storage exhausted: all 15 slots are occupied")
        slot = Slot(index=available, consumer=consumer)
        self._save(slot)
        image = self._path(slot.index, "img")
        fd = os.open(image, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            info = os.fstat(fd)
            slot = slot.model_copy(update={"device": info.st_dev, "inode": info.st_ino})
            self._save(slot)
            os.posix_fallocate(fd, 0, self.image_bytes)
            os.fsync(fd)
        finally:
            os.close(fd)
        _run("mkfs.ext4", "-q", "-F", "-m", "0", "-E", "nodiscard,root_owner=1000:1000", str(image))
        control = os.open("/dev/loop-control", os.O_RDWR)
        try:
            for _ in range(16):
                number = fcntl.ioctl(control, 0x4C82)  # LOOP_CTL_GET_FREE
                node = self._path(slot.index, "node")
                self._node(node, number)
                try:
                    _run("losetup", str(node), str(image))
                except subprocess.CalledProcessError:
                    if self._mapping(node) is not None:
                        continue  # another allocator won; never detach its mapping
                    raise
                if self._mapping(node) != (slot.device, slot.inode):
                    raise RuntimeError(f"slot {slot.index}: loop identity changed during attachment")
                slot = slot.model_copy(update={"phase": "attached"})
                self._save(slot)
                return slot
        finally:
            os.close(control)
        raise RuntimeError("loop allocation remained busy after 16 attempts")

    def activate(self, slot: Slot) -> None:
        """Arm last-close detachment only after Docker has mounted the device."""
        fd = os.open(self._path(slot.index, "node"), os.O_RDWR)
        try:
            status = bytearray(fcntl.ioctl(fd, _LOOP_GET_STATUS64, bytes(232)))
            if struct.unpack_from("=QQ", status) != (slot.device, slot.inode):
                raise RuntimeError(f"slot {slot.index}: mapping changed; activation refused")
            flags = struct.unpack_from("=I", status, _LO_FLAGS_OFFSET)[0]
            struct.pack_into("=I", status, _LO_FLAGS_OFFSET, flags | _LO_FLAGS_AUTOCLEAR)
            fcntl.ioctl(fd, _LOOP_SET_STATUS64, status)
            confirmed = fcntl.ioctl(fd, _LOOP_GET_STATUS64, bytes(232))
            if not struct.unpack_from("=I", confirmed, _LO_FLAGS_OFFSET)[0] & _LO_FLAGS_AUTOCLEAR:
                raise RuntimeError(f"slot {slot.index}: automatic loop detachment unconfirmed")
            image = self._path(slot.index, "img")
            info = image.lstat()
            if (
                not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or (info.st_dev, info.st_ino) != (slot.device, slot.inode)
            ):
                raise RuntimeError(f"slot {slot.index}: backing identity uncertain; retained")
            # The mounted loop retains the inode. Removing its only name makes
            # last-close detachment release the image even if this helper dies.
            image.unlink()
            _sync_directory(self.root)
        finally:
            os.close(fd)
        # Retain slot accounting until consumer and volume removal are confirmed.
        self._save(slot.model_copy(update={"phase": "active"}))

    def reclaim(self, slot: Slot) -> None:
        """Caller has confirmed consumer and filesystem-volume absence."""
        image = self._path(slot.index, "img")
        if slot.phase == "reserved" and slot.device is None and slot.inode is None and image.exists():
            # Creation precedes identity persistence, but allocation/attachment
            # follow it. Only an empty, unlinked-to-peers regular file can be
            # the interrupted creation; do not adopt arbitrary backing data.
            info = image.lstat()
            if (
                not stat.S_ISREG(info.st_mode) or info.st_size != 0 or info.st_nlink != 1
                or self._path(slot.index, "node").exists()
            ):
                raise RuntimeError(f"slot {slot.index}: incomplete backing identity uncertain; retained")
            candidate = slot.model_copy(update={"device": info.st_dev, "inode": info.st_ino})
            if self._matching_loops(candidate):
                raise RuntimeError(f"slot {slot.index}: unidentified backing already attached; retained")
            slot = candidate
            self._save(slot)
        slot = slot.model_copy(update={"phase": "reclaiming"})
        self._save(slot)
        if image.exists():
            info = image.stat()
            if (info.st_dev, info.st_ino) != (slot.device, slot.inode):
                raise RuntimeError(f"slot {slot.index}: backing identity uncertain; retained")
            for number in self._matching_loops(slot):
                node = self._path(slot.index, "node")
                self._node(node, number)
                if self._mapping(node) != (slot.device, slot.inode):
                    raise RuntimeError(f"slot {slot.index}: mapping changed; retained")
                _run("losetup", "-d", str(node))
            deadline = time.monotonic() + 5
            while self._matching_loops(slot):
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"slot {slot.index}: loop detach unconfirmed; retained")
                time.sleep(0.05)
            image.unlink()
        elif slot.device is not None and self._matching_loops(slot):
            raise RuntimeError(f"slot {slot.index}: missing backing file still attached; retained")
        self._path(slot.index, "node").unlink(missing_ok=True)
        self._path(slot.index, "pending").unlink(missing_ok=True)
        self._path(slot.index, "json").unlink()
        _sync_directory(self.root)

    def dispatch(self, request: Request) -> Reply:
        if request.action == "owner":
            path = self.root / "runtime.json"
            if path.exists():
                return Reply(owner=RuntimeOwner.model_validate_json(path.read_bytes()))
            pending = self.root / "runtime.pending"
            if pending.exists():
                # Return the interrupted claim to the manager for the same
                # live-incarnation fence used by committed ownership records.
                return Reply(owner=RuntimeOwner.model_validate_json(pending.read_bytes()))
            if self.slots():
                raise RuntimeError("pool owner missing with retained resources; inspection required")
            return Reply()
        if request.action == "claim":
            if request.owner is None:
                raise ValueError("runtime owner required")
            temporary = self.root / "runtime.pending"
            with temporary.open("w") as stream:
                stream.write(request.owner.model_dump_json())
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.root / "runtime.json")
            _sync_directory(self.root)
            self.claimed = True
            return Reply()
        if not self.claimed:
            raise RuntimeError("storage helper runtime ownership has not been claimed")
        if request.action == "release":
            if self.slots():
                raise RuntimeError("cannot release a pool with occupied slots")
            (self.root / "runtime.json").unlink()
            _sync_directory(self.root)
            self.claimed = False
            return Reply()
        if request.action == "prepare":
            return Reply(slots=[self.prepare(request.consumer)])
        slots = self.slots()
        if request.action == "list":
            return Reply(slots=slots)
        slot = next(s for s in slots if s.index == request.index)
        if request.action == "device":
            node = self._path(slot.index, "node")
            if self._mapping(node) != (slot.device, slot.inode):
                raise RuntimeError(f"slot {slot.index}: loop identity changed; I/O limits refused")
            return Reply(loop_number=os.minor(node.stat().st_rdev))
        if request.action == "activate":
            self.activate(slot)
        else:
            self.reclaim(slot)
        return Reply()


def serve(image_bytes: int = IMAGE_BYTES) -> None:
    os.umask(0o077)
    with (ROOT / "pool.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        SOCKET.unlink(missing_ok=True)
        with socket.socket(socket.AF_UNIX) as server:
            server.bind(str(SOCKET))
            server.listen(1)
            pool = DiskPool(image_bytes=image_bytes)
            while True:
                connection, _ = server.accept()
                _serve_connection(pool, connection)


def _serve_connection(pool: DiskPool, connection: socket.socket) -> None:
    # A cancelled Docker exec must not kill the owner of unrelated allocations.
    try:
        with connection:
            connection.settimeout(130)
            with connection.makefile("rwb") as stream:
                try:
                    request = Request.model_validate_json(stream.readline(MAX_MESSAGE))
                    reply = pool.dispatch(request)
                except Exception as exc:
                    detail = "".join(traceback.format_exception(exc))
                    if isinstance(exc, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
                        detail += "\nstderr=" + (exc.stderr or b"").decode(errors="replace")[-4000:]
                    reply = Reply(
                        error=f"{type(exc).__name__}: {str(exc)[:4000]}",
                        error_trace=detail[-12000:],
                    )
                stream.write(reply.model_dump_json().encode() + b"\n")
                stream.flush()
    except OSError:
        # The operation's durable records remain authoritative if delivery fails.
        return


def call(request: Request) -> Reply:
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(130)
        client.connect(str(SOCKET))
        with client.makefile("rwb") as stream:
            stream.write(request.model_dump_json().encode() + b"\n")
            stream.flush()
            return Reply.model_validate_json(stream.readline(MAX_MESSAGE))


def verify_io_limits(number: int, iops: int, bps: int, root: Path = Path("/sys/fs/cgroup")) -> None:
    """Reject daemons which silently ignore throttle flags, before admitting work."""
    device = f"7:{number}"
    if (root / "io.max").exists():
        entries = [line.split() for line in (root / "io.max").read_text().splitlines()]
        values = next((dict(item.split("=") for item in row[1:]) for row in entries if row[0] == device), {})
        expected = {"riops": str(iops), "wiops": str(iops), "rbps": str(bps), "wbps": str(bps)}
        if any(values.get(key) != value for key, value in expected.items()):
            raise RuntimeError("artifact I/O limits not enforced in cgroup io.max")
    else:
        for direction in ("read", "write"):
            for unit, value in (("iops", iops), ("bps", bps)):
                path = root / "blkio" / f"blkio.throttle.{direction}_{unit}_device"
                values = dict(line.split() for line in path.read_text().splitlines())
                if values.get(device) != str(value):
                    raise RuntimeError(f"artifact I/O limit not enforced in {path.name}")


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "serve":
        serve(int(sys.argv[2]) if len(sys.argv) == 3 else IMAGE_BYTES)
    elif sys.argv[1:] and sys.argv[1] == "probe":
        verify_io_limits(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
        probe = Path("/probe/check")
        with probe.open("wb") as stream:
            stream.write(b"ok")
            stream.flush()
            os.fsync(stream.fileno())
        if probe.read_bytes() != b"ok":
            raise RuntimeError("storage probe readback mismatch")
        probe.unlink()
    else:
        print(call(Request.model_validate_json(sys.stdin.readline(MAX_MESSAGE))).model_dump_json())
