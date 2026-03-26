"""Resolve validator runtime metadata for platform registration."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path

from harnyx_validator.application.dto.registration import ValidatorRegistrationMetadata
from harnyx_validator.version import VALIDATOR_RELEASE_VERSION

_DOCKER_BINARY = "docker"
_MOUNTINFO_CONTAINER_ID_PATTERN = re.compile(
    r"/containers/([0-9a-f]{12,64})/(?:hostname|hosts|resolv\.conf)(?:\s|$)"
)
logger = logging.getLogger("harnyx_validator.runtime.registration")


def _run_docker_command(args: list[str], *, error_context: str) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)  # noqa: S603
    except OSError as exc:
        raise RuntimeError(f"{error_context}: failed to execute docker CLI: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"{error_context}: stderr={stderr}") from exc

    return (result.stdout or "").strip()


def _resolve_current_container_id_from_mountinfo() -> str | None:
    try:
        mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError:
        return None
    match = _MOUNTINFO_CONTAINER_ID_PATTERN.search(mountinfo)
    if match is None:
        return None
    return match.group(1)


def _resolve_current_container_id() -> str:
    mountinfo_container = _resolve_current_container_id_from_mountinfo()
    if mountinfo_container is not None:
        return mountinfo_container

    container = (os.getenv("HOSTNAME") or "").strip()
    if container:
        return container
    raise RuntimeError("failed to resolve current validator container id via /proc/self/mountinfo or HOSTNAME")


def _inspect_current_image_id() -> str:
    return _inspect_container_image_id(_resolve_current_container_id())


def _inspect_container_image_id(container: str) -> str:
    image_id = _run_docker_command(
        [_DOCKER_BINARY, "inspect", "--format", "{{.Image}}", container],
        error_context=f"docker inspect failed for container={container}",
    )
    if not image_id:
        raise RuntimeError(f"docker inspect returned empty image id for container={container}")
    return image_id


def _inspect_registry_digest(local_image_id: str) -> str | None:
    output = _run_docker_command(
        [_DOCKER_BINARY, "image", "inspect", "--format", "{{json .RepoDigests}}", local_image_id],
        error_context=f"docker image inspect failed for image_id={local_image_id}",
    )
    if not output:
        raise RuntimeError(f"docker image inspect returned empty repo digests for image_id={local_image_id}")

    repo_digests = json.loads(output)
    if repo_digests is None:
        return None
    if not isinstance(repo_digests, list):
        raise TypeError("docker image inspect repo digests must be a JSON list or null")
    if not repo_digests:
        return None

    repo_digest = repo_digests[0]
    if not isinstance(repo_digest, str) or not repo_digest:
        raise RuntimeError(f"docker image inspect returned invalid repo digest for image_id={local_image_id}")
    _, separator, digest = repo_digest.partition("@")
    if separator != "@" or not digest:
        raise RuntimeError(f"docker image inspect returned invalid repo digest entry: {repo_digest}")
    return digest


def _optional_env(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _resolve_image_identity() -> tuple[str | None, str | None]:
    try:
        local_image_id = _inspect_current_image_id()
    except RuntimeError as exc:
        logger.warning(
            "validator registration image inspection unavailable",
            extra={
                "data": {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            },
        )
        return None, None

    try:
        registry_digest = _inspect_registry_digest(local_image_id)
    except (RuntimeError, TypeError, json.JSONDecodeError) as exc:
        logger.warning(
            "validator registration registry digest inspection unavailable",
            extra={
                "data": {
                    "local_image_id": local_image_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            },
        )
        return local_image_id, None

    return local_image_id, registry_digest


def resolve_validator_registration_metadata() -> ValidatorRegistrationMetadata:
    local_image_id, registry_digest = _resolve_image_identity()
    return ValidatorRegistrationMetadata(
        validator_version=VALIDATOR_RELEASE_VERSION,
        source_revision=_optional_env("SOURCE_REVISION"),
        registry_digest=registry_digest,
        local_image_id=local_image_id,
    )


__all__ = ["resolve_validator_registration_metadata"]
