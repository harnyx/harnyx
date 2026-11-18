"""Validator-owned bounded storage around the shared sandbox lifecycle."""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from harnyx_commons.sandbox.docker import DockerSandboxManager
from harnyx_commons.sandbox.manager import SandboxDeployment, SandboxStartError
from harnyx_commons.sandbox.options import BlockIOLimits, SandboxOptions
from harnyx_validator.infrastructure.artifact_storage_helper import IMAGE_BYTES, Reply, Request, RuntimeOwner, Slot

logger = logging.getLogger(__name__)
_LABEL = "ai.harnyx.artifact-storage"
_MODULE = "harnyx_validator.infrastructure.artifact_storage_helper"
_PYTHON = "/workspace/.venv/bin/python"


class _State(BaseModel):
    running: bool = Field(alias="Running")
    started_at: str = Field(alias="StartedAt")


class _Config(BaseModel):
    labels: dict[str, str] | None = Field(default=None, alias="Labels")


class _Container(BaseModel):
    identifier: str = Field(alias="Id")
    state: _State = Field(alias="State")
    config: _Config = Field(alias="Config")


class _Volume(BaseModel):
    name: str = Field(alias="Name")
    mountpoint: str = Field(alias="Mountpoint")
    labels: dict[str, str] | None = Field(default=None, alias="Labels")
    options: dict[str, str] | None = Field(default=None, alias="Options")


class ValidatorRuntimeIdentity(BaseModel):
    """Strict Docker identity used to fence recovery across container restarts."""

    model_config = ConfigDict(frozen=True)
    container_id: str = Field(min_length=1)
    started_at: str = Field(min_length=1)
    image_id: str = Field(min_length=1)


class DockerServerVersion(BaseModel):
    engine: str = Field(alias="Version", min_length=1, max_length=128)
    api: str = Field(alias="ApiVersion", min_length=1, max_length=128)


def read_docker_server_version() -> DockerServerVersion:
    result = subprocess.run(
        ["docker", "version", "--format", "{{json .Server}}"],  # noqa: S607
        capture_output=True, text=True, check=True, timeout=10,
    )
    return DockerServerVersion.model_validate_json(result.stdout)


class ArtifactStorageSandboxManager:
    def __init__(
        self,
        docker_manager: DockerSandboxManager,
        identity: ValidatorRuntimeIdentity,
        *,
        iops: int,
        bytes_per_second: int,
        docker_binary: str = "docker",
        pool_name: str = "harnyx-artifact-storage",
        image_bytes: int = IMAGE_BYTES,
    ) -> None:
        if iops <= 0 or bytes_per_second <= 0:
            raise ValueError("artifact I/O limits must be positive")
        self._iops = iops
        self._bytes_per_second = bytes_per_second
        self._docker_manager = docker_manager
        self._identity = identity
        self._docker = docker_binary
        self._pool = pool_name
        self._helper = f"{pool_name}-helper"
        self._backing = f"{pool_name}-backing"
        self._image_bytes = image_bytes
        self._mountpoint = ""
        self._lock = threading.RLock()
        self._ready = False
        self._helper_started = False
        self._claimed = False
        self._uncertain_creation = False
        self.stage = "initialization"
        self._deployments: dict[str, str] = {}

    def _run(self, *args: str, data: str | None = None) -> str:
        self.stage = "docker " + " ".join(args[:2])
        try:
            result = subprocess.run(  # noqa: S603
                [self._docker, *args], input=data, capture_output=True, text=True, check=True, timeout=145,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"artifact storage {args[:2]} exit {exc.returncode}: {exc.stderr[-4000:]}") from exc
        except subprocess.TimeoutExpired as exc:
            stderr = exc.stderr or b""
            message = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
            raise RuntimeError(f"artifact storage {args[:2]} timed out: {message[-4000:]}") from exc
        return result.stdout.strip()

    def _container(self, name: str) -> _Container | None:
        ids = self._run("container", "ls", "-aq", "--no-trunc", "--filter", f"name=^/{name}$")
        if not ids:
            return None
        containers = TypeAdapter(list[_Container]).validate_json(self._run("inspect", name))
        return containers[0]

    def _volume(self, name: str) -> _Volume | None:
        names = self._run("volume", "ls", "--format", "{{.Name}}", "--filter", f"name=^{name}$")
        if name not in names.splitlines():
            return None
        return TypeAdapter(list[_Volume]).validate_json(self._run("volume", "inspect", name))[0]

    def _owned(self, labels: dict[str, str] | None) -> None:
        if labels is None or labels.get(_LABEL) != self._pool:
            raise RuntimeError("artifact storage resource ownership uncertain; retained")

    def _request(self, request: Request) -> Reply:
        try:
            raw = self._run(
                "exec", "-i", self._helper, _PYTHON, "-m", _MODULE,
                data=request.model_dump_json() + "\n",
            )
        finally:
            self.stage = f"helper {request.action}"
        reply = Reply.model_validate_json(raw)
        if reply.error is not None:
            raise RuntimeError(f"artifact storage {request.action}: {reply.error}\n{reply.error_trace or ''}")
        return reply

    def recover(self, *, sandbox_labels: dict[str, str], sandbox_name_prefix: str) -> None:
        """Claim ownership before stale sandbox removal; never recover a live owner."""
        with self._lock:
            self._ready = False
            previous = self._container(self._helper)
            if previous is not None:
                labels = previous.config.labels or {}
                self._owned(labels)
                self._confirm_owner_ended(RuntimeOwner(
                    container_id=labels[f"{_LABEL}.container"], started_at=labels[f"{_LABEL}.started"],
                ))
                self._run("rm", "-f", previous.identifier)
            backing = self._volume(self._backing)
            if backing is None:
                self._run("volume", "create", "--label", f"{_LABEL}={self._pool}", self._backing)
                backing = self._volume(self._backing)
            if backing is None:
                raise RuntimeError("artifact storage backing volume creation unconfirmed")
            self._owned(backing.labels)
            self._mountpoint = backing.mountpoint
            # Docker atomically reserves the helper name. A concurrent loser must
            # not clean up the winner or any of its sandboxes.
            self._run(
                "run", "-d", "--name", self._helper,
                "--label", f"{_LABEL}={self._pool}",
                "--label", f"{_LABEL}.container={self._identity.container_id}",
                "--label", f"{_LABEL}.started={self._identity.started_at}",
                "--user", "0", "--network", "none", "--read-only",
                "--cap-drop", "ALL", "--cap-add", "MKNOD", "--security-opt", "no-new-privileges",
                "--device", "/dev/loop-control:/dev/loop-control:rw",
                "--device-cgroup-rule", "b 7:* rwm",
                "--mount", f"type=volume,src={self._backing},dst=/backing",
                "--entrypoint", _PYTHON, self._identity.image_id, "-m", _MODULE, "serve", str(self._image_bytes),
            )
            self._helper_started = True
            deadline = time.monotonic() + 15
            while True:
                try:
                    old_owner = self._request(Request(action="owner")).owner
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)
            if old_owner is not None:
                self._confirm_owner_ended(old_owner)
            self._request(Request(action="claim", owner=RuntimeOwner(
                container_id=self._identity.container_id, started_at=self._identity.started_at,
            )))
            self._claimed = True
            if not self._docker_manager.cleanup_stale_sandbox_containers(
                labels=sandbox_labels, name_prefix=sandbox_name_prefix,
            ):
                raise RuntimeError("stale sandbox removal unconfirmed; storage recovery blocked")
            for slot in self._request(Request(action="list")).slots:
                self._reclaim(slot)
            # A real prepare/mount/cleanup proves permissions and utilities at
            # recovery, before marking this manager ready. It uses one fixed slot.
            probe = self._prepare(f"{sandbox_name_prefix}storage-probe")
            failed_stage = "probe complete"
            try:
                loop_number = self._loop_number(probe)
                self._run(
                    "create", "--name", probe.consumer,
                    *(arg for key, value in sandbox_labels.items() for arg in ("--label", f"{key}={value}")),
                    "--network", "none", "--read-only",
                    "--cap-drop", "ALL", "--user", "1000:1000", "--security-opt", "no-new-privileges",
                    "--mount", f"type=volume,src={self._filesystem(probe)},dst=/probe,volume-nocopy",
                    "--entrypoint", _PYTHON, self._identity.image_id, "-m", _MODULE, "probe",
                    str(loop_number), str(self._iops), str(self._bytes_per_second),
                )
                limits = self._io_limits(probe)
                self.stage = "create with block I/O"
                self._docker_manager.recreate_with_block_io(probe.consumer, limits)
                self._run("start", "--attach", probe.consumer)
            except SandboxStartError:
                self._uncertain_creation = True
                failed_stage = self.stage
                raise
            except Exception:
                failed_stage = self.stage
                raise
            finally:
                if not self._uncertain_creation:
                    if self._container(probe.consumer) is not None:
                        self._run("rm", "-f", probe.consumer)
                    self._reclaim(probe)
                self.stage = failed_stage
            self._ready = True

    def _confirm_owner_ended(self, owner: RuntimeOwner) -> None:
        ids = self._run("container", "ls", "-aq", "--no-trunc", "--filter", f"id={owner.container_id}")
        if ids:
            inspected = TypeAdapter(list[_Container]).validate_json(self._run("inspect", owner.container_id))[0]
            if inspected.state.running and inspected.state.started_at == owner.started_at:
                raise RuntimeError("artifact storage already belongs to a live validator incarnation")

    def _filesystem(self, slot: Slot) -> str:
        return f"{self._pool}-slot-{slot.index}"

    def _prepare(self, consumer: str) -> Slot:
        slot = self._request(Request(action="prepare", consumer=consumer)).slots[0]
        name = self._filesystem(slot)
        if self._volume(name) is not None:
            raise RuntimeError(f"unexpected existing filesystem volume {name}; allocation retained")
        device = str(PurePosixPath(self._mountpoint) / f"slot-{slot.index}.node")
        self._run(
            "volume", "create", "--label", f"{_LABEL}={self._pool}",
            "--label", f"{_LABEL}.consumer={consumer}", "--driver", "local",
            "--opt", "type=ext4", "--opt", f"device={device}",
            "--opt", "o=rw,nosuid,nodev,noexec", name,
        )
        return slot

    def _loop_number(self, slot: Slot) -> int:
        number = self._request(Request(action="device", index=slot.index)).loop_number
        if number is None:
            raise RuntimeError("artifact loop device identity unavailable")
        return number

    def _io_limits(self, slot: Slot) -> BlockIOLimits:
        # Use the exact owned node already used by the filesystem volume.
        return BlockIOLimits(
            device=str(PurePosixPath(self._mountpoint) / f"slot-{slot.index}.node"),
            iops=self._iops, bytes_per_second=self._bytes_per_second,
        )

    def _reclaim(self, slot: Slot) -> None:
        if self._container(slot.consumer) is not None:
            raise RuntimeError(f"slot {slot.index}: consumer remains; storage retained")
        name = self._filesystem(slot)
        volume = self._volume(name)
        if volume is not None:
            self._owned(volume.labels)
            expected = str(PurePosixPath(self._mountpoint) / f"slot-{slot.index}.node")
            if (
                (volume.labels or {}).get(f"{_LABEL}.consumer") != slot.consumer
                or (volume.options or {}).get("device") != expected
            ):
                raise RuntimeError(f"slot {slot.index}: filesystem identity uncertain; retained")
            self._run("volume", "rm", name)  # no force; Docker refuses remaining references
        if self._volume(name) is not None:
            raise RuntimeError(f"slot {slot.index}: volume removal unconfirmed")
        self._request(Request(action="reclaim", index=slot.index))

    def start(self, options: SandboxOptions) -> SandboxDeployment:
        # Keep launch and activation inside the lifecycle lock: otherwise close
        # could delete the volume before Docker uses it (Docker would recreate
        # an unlimited ordinary named volume).
        with self._lock:
            if not self._ready:
                raise RuntimeError("artifact storage not ready")
            try:
                slot = self._prepare(options.container_name)
            except Exception:
                self._cleanup_consumer(options.container_name)
                raise
            try:
                self._loop_number(slot)
                deployment = self._docker_manager.start(replace(
                    options, volumes=(*options.volumes, (self._filesystem(slot), "/sandbox/queries", "rw")),
                    block_io_limits=self._io_limits(slot),
                ))
            except SandboxStartError as exc:
                self._uncertain_creation = True
                if exc.unremoved_deployment.identifier:
                    self._deployments[exc.unremoved_deployment.identifier] = options.container_name
                raise
            except Exception:
                self._cleanup_consumer(options.container_name)
                raise
            if deployment.identifier:
                self._deployments[deployment.identifier] = options.container_name
            try:
                self._request(Request(action="activate", index=slot.index))
            except Exception as exc:
                raise SandboxStartError(
                    "artifact storage activation unconfirmed", unremoved_deployment=deployment,
                ) from exc
            return deployment

    def _cleanup_consumer(self, consumer: str) -> None:
        try:
            for slot in self._request(Request(action="list")).slots:
                if slot.consumer == consumer:
                    self._reclaim(slot)
        except Exception:
            logger.exception("artifact storage cleanup failed; allocation retained", extra={"consumer": consumer})

    def stop(self, deployment: SandboxDeployment) -> bool:
        removed = self._docker_manager.stop(deployment)
        if removed and deployment.identifier:
            with self._lock:
                consumer = self._deployments.pop(deployment.identifier, deployment.identifier)
                self._cleanup_consumer(consumer)
        return removed

    def finish_probe(self) -> None:
        """Clean only a helper acquired by this manager; retain uncertain resources."""
        if not self._helper_started:
            raise RuntimeError("probe helper acquisition unconfirmed; resources retained for recovery")
        if self._claimed:
            self.close()
        else:
            self._run("rm", "-f", self._helper)
            raise RuntimeError("probe ownership claim incomplete; backing resources retained for recovery")

    def close(self) -> None:
        """Release helper only after all slots have been reclaimed."""
        with self._lock:
            self._ready = False
            if self._uncertain_creation:
                raise RuntimeError("container creation outcome uncertain; storage retained for recovery")
            for slot in self._request(Request(action="list")).slots:
                self._reclaim(slot)
            self._request(Request(action="release"))
            self._run("rm", "-f", self._helper)
