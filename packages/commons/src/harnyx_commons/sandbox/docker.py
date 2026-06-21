"""Docker-backed sandbox manager implementation shared across services."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx

from harnyx_commons.json_types import JsonValue
from harnyx_commons.protocol_headers import (
    HOST_CONTAINER_URL_HEADER,
    SESSION_ID_HEADER,
)
from harnyx_commons.sandbox.client import SandboxClient, SandboxInvokeError
from harnyx_commons.sandbox.diagnostic_files import (
    ensure_private_diagnostic_dir,
    write_private_json,
    write_private_text,
)
from harnyx_commons.sandbox.manager import SandboxDeployment, SandboxManager
from harnyx_commons.sandbox.options import DEFAULT_TOKEN_HEADER, SandboxOptions

logger = logging.getLogger(__name__)

_MOUNTINFO_CONTAINER_ID_PATTERN = re.compile(
    r"/containers/([0-9a-f]{12,64})/(?:hostname|hosts|resolv\.conf)(?:\s|$)"
)
_NON_SENSITIVE_DIAGNOSTIC_ENV_KEYS = frozenset({"SANDBOX_HOST", "SANDBOX_PORT"})
_STALE_LEGACY_CONTAINER_STATUSES = ("created", "exited", "dead")
_SANDBOX_ENTRYPOINT_CONNECT_ATTEMPTS = 2
_CONTAINER_IP_READY_TIMEOUT_SECONDS = 5.0
_CONTAINER_IP_READY_POLL_INTERVAL_SECONDS = 0.1
_IMAGE_PULL_ATTEMPTS = 3
_IMAGE_PULL_INITIAL_BACKOFF_SECONDS = 1.0
_RETRYABLE_SANDBOX_ENTRYPOINT_CONNECTION_NOT_ESTABLISHED_ERRORS = (
    httpx.ConnectError,
)


class _ContainerIpNotReadyError(RuntimeError):
    """Docker has not published a usable container IP for the configured network yet."""


class HttpSandboxClient(SandboxClient):
    """Sandbox client backed by an HTTP endpoint exposed by the sandbox container."""

    def __init__(
        self,
        base_url: str,
        *,
        host_container_url: str | None = None,
        timeout: float = 310.0,
        client: httpx.AsyncClient | None = None,
        connect_attempts: int = _SANDBOX_ENTRYPOINT_CONNECT_ATTEMPTS,
    ) -> None:
        if connect_attempts < 1:
            raise ValueError("sandbox connect_attempts must be at least 1")
        self._token_header = DEFAULT_TOKEN_HEADER
        self._host_container_url = host_container_url
        self._connect_attempts = connect_attempts
        self._owns_client = client is None
        self._client: httpx.AsyncClient = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=0),
        )

    def configure(
        self,
        *,
        host_container_url: str | None = None,
    ) -> None:
        if host_container_url is not None:
            self._host_container_url = host_container_url

    async def invoke(
        self,
        entrypoint: str,
        *,
        payload: Mapping[str, JsonValue],
        context: Mapping[str, JsonValue],
        token: str,
        session_id: UUID,
    ) -> Mapping[str, JsonValue]:
        headers: dict[str, str] = {
            self._token_header: token,
            SESSION_ID_HEADER: str(session_id),
            HOST_CONTAINER_URL_HEADER: self._host_container_url or "",
            "Connection": "close",
        }
        try:
            response = await self._post_entrypoint_with_connect_retry(
                entrypoint=entrypoint,
                payload=payload,
                context=context,
                headers=headers,
                session_id=session_id,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            logger.error(
                "sandbox entrypoint request timed out: entrypoint=%s session_id=%s",
                entrypoint,
                session_id,
                exc_info=exc,
                extra={
                    "entrypoint": entrypoint,
                    "session_id": str(session_id),
                },
            )
            raise _sandbox_invoke_error(
                status_code=504,
                detail={"exception": "TimeoutException", "error": str(exc)},
                message=(
                    f"sandbox entrypoint request timed out: "
                    f"entrypoint={entrypoint} session_id={session_id}"
                ),
            ) from exc
        except httpx.RequestError as exc:  # pragma: no cover - network errors
            logger.error(
                "sandbox entrypoint request failed: entrypoint=%s session_id=%s error=%s",
                entrypoint,
                session_id,
                str(exc),
                exc_info=exc,
                extra={
                    "entrypoint": entrypoint,
                    "session_id": str(session_id),
                },
            )
            raise _sandbox_invoke_error(
                status_code=0,
                detail={"exception": exc.__class__.__name__, "error": str(exc)},
                message=(
                    f"sandbox entrypoint request failed: "
                    f"entrypoint={entrypoint} session_id={session_id} error={exc}"
                ),
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail_payload = _unwrap_response_detail(_response_json_or_text(exc.response))
            detail = _parse_sandbox_response_detail(detail_payload)
            status = exc.response.status_code
            logger.error(
                (
                    "sandbox entrypoint request failed: entrypoint=%s status=%s "
                    "session_id=%s detail=%s"
                ),
                entrypoint,
                status,
                session_id,
                detail,
                exc_info=exc,
                extra={
                    "entrypoint": entrypoint,
                    "status": status,
                    "detail": detail,
                    "session_id": str(session_id),
                },
            )
            raise SandboxInvokeError(
                f"sandbox entrypoint request failed with status {status}: {detail.raw}",
                status_code=status,
                detail_code=detail.code,
                detail_exception=detail.exception,
                detail_error=detail.error,
            ) from exc
        body = response.json()
        return _parse_sandbox_invoke_result(body)

    async def _post_entrypoint_with_connect_retry(
        self,
        *,
        entrypoint: str,
        payload: Mapping[str, JsonValue],
        context: Mapping[str, JsonValue],
        headers: Mapping[str, str],
        session_id: UUID,
    ) -> httpx.Response:
        for attempt_number in range(1, self._connect_attempts + 1):
            try:
                return await self._client.post(
                    f"/entry/{entrypoint}",
                    json={
                        "payload": dict(payload),
                        "context": dict(context),
                    },
                    headers=headers,
                )
            except _RETRYABLE_SANDBOX_ENTRYPOINT_CONNECTION_NOT_ESTABLISHED_ERRORS as exc:
                if attempt_number == self._connect_attempts:
                    raise
                logger.warning(
                    "retrying sandbox entrypoint connect failure",
                    extra={
                        "entrypoint": entrypoint,
                        "session_id": str(session_id),
                        "attempt": attempt_number,
                        "max_attempts": self._connect_attempts,
                        "error_type": exc.__class__.__name__,
                    },
                )
        raise RuntimeError("sandbox entrypoint connect retry loop exhausted without response")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def close(self) -> None:
        if not self._owns_client:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop: best-effort close; swallow loop/transport teardown errors.
            try:
                asyncio.run(self._client.aclose())
            except RuntimeError:
                pass
        else:
            loop.create_task(self._client.aclose())


def _summarize_response(response: httpx.Response) -> str:
    detail = _parse_sandbox_response_detail(_unwrap_response_detail(_response_json_or_text(response)))
    text = str(detail.raw)
    return text if len(text) <= 500 else text[:500] + "…"


@dataclass(frozen=True, slots=True)
class _SandboxResponseDetail:
    raw: object
    code: str | None
    exception: str | None
    error: str | None


def _parse_sandbox_invoke_result(value: object) -> dict[str, JsonValue]:
    body = _require_object_mapping(value, label="sandbox response must be a JSON object")
    result = body.get("result", body)
    return _require_json_object(result, label="sandbox response result must be a JSON object")


def _parse_sandbox_response_detail(value: object) -> _SandboxResponseDetail:
    mapping = _object_mapping_or_none(value)
    if mapping is None:
        return _SandboxResponseDetail(raw=value, code=None, exception=None, error=None)
    return _SandboxResponseDetail(
        raw=value,
        code=_as_optional_str(mapping.get("code")),
        exception=_as_optional_str(mapping.get("exception")),
        error=_as_optional_str(mapping.get("error")),
    )


def _unwrap_response_detail(value: object) -> object:
    mapping = _object_mapping_or_none(value)
    if mapping is None:
        return value
    return mapping.get("detail", value)


def _response_json_or_text(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return response.text


def _object_mapping_or_none(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        result[key] = item
    return result


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    return value


def _sandbox_invoke_error(
    *,
    status_code: int,
    detail: object,
    message: str,
) -> SandboxInvokeError:
    parsed = _parse_sandbox_response_detail(detail)
    return SandboxInvokeError(
        message,
        status_code=status_code,
        detail_code=parsed.code,
        detail_exception=parsed.exception,
        detail_error=parsed.error,
    )


def _published_port_spec(bind_host: str | None, options: SandboxOptions) -> str:
    if options.host_port is None:
        raise ValueError("sandbox host_port must be configured to publish a port")
    if options.host_port == 0:
        if bind_host is None:
            return str(options.container_port)
        return f"{bind_host}::{options.container_port}"
    if bind_host is None:
        return f"{options.host_port}:{options.container_port}"
    return f"{bind_host}:{options.host_port}:{options.container_port}"


def _require_object_mapping(value: object, *, label: str) -> dict[str, object]:
    mapping = _object_mapping_or_none(value)
    if mapping is None:
        raise ValueError(label)
    return mapping


def _require_json_object(value: object, *, label: str) -> dict[str, JsonValue]:
    mapping = _require_object_mapping(value, label=label)
    result: dict[str, JsonValue] = {}
    for key, item in mapping.items():
        result[key] = _to_json_value(item, label=f"{label}.{key}")
    return result


def _to_json_value(value: object, *, label: str) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_to_json_value(item, label=f"{label}[]") for item in value]
    if isinstance(value, Mapping):
        return _require_json_object(value, label=label)
    raise TypeError(f"{label} must be JSON-compatible")


class DockerSandboxManager(SandboxManager):
    """Launches sandbox containers using the Docker CLI."""

    _popen: Callable[..., subprocess.Popen[str]]

    def __init__(
        self,
        *,
        docker_binary: str = "docker",
        host: str = "127.0.0.1",
        published_port_bind_host: str | None = None,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        client_factory: Callable[[str, str | None], SandboxClient] | None = None,
        log_consumer: Callable[[str], None] | None = None,
        log_runner: Callable[..., subprocess.Popen[str]] | None = None,
        command_timeout_seconds: float = 120.0,
    ) -> None:
        self._docker = docker_binary
        self._host = host
        self._published_port_bind_host = published_port_bind_host
        self._run = command_runner or self._default_run
        self._command_timeout_seconds = command_timeout_seconds
        self._client_factory = client_factory or (
            lambda base_url, host_container_url: HttpSandboxClient(
                base_url,
                host_container_url=host_container_url,
            )
        )
        self._log_consumer = log_consumer
        self._popen: Callable[..., subprocess.Popen[str]] | None = (
            log_runner or self._default_popen
        )
        self._log_streams: dict[str, tuple[subprocess.Popen[str], threading.Thread]] = {}

    def start(self, options: SandboxOptions) -> SandboxDeployment:
        self._validate_options(options)

        container_id = self._launch_container(options)
        base_url: str | None = None
        client: SandboxClient | None = None
        try:
            base_url, client = self._ready_client(options, container_id=container_id)
            self._post_launch_steps(options, base_url, container_id)
        except Exception as exc:
            self._write_failure_diagnostics(
                options=options,
                container_id=container_id,
                error=exc,
            )
            if client is not None:
                client.close()
            self._stop_log_stream(container_id)
            self._best_effort_stop_and_remove(container_id, stop_timeout_seconds=options.stop_timeout_seconds)
            raise
        assert base_url is not None
        assert client is not None

        return SandboxDeployment(
            client=client,
            identifier=container_id,
            base_url=base_url,
            stop_timeout_seconds=options.stop_timeout_seconds,
        )

    @staticmethod
    def _validate_options(options: SandboxOptions) -> None:
        if options.host_port is None and not options.network:
            raise ValueError("sandbox network must be provided when host_port is not published")
        if not options.host_container_url:
            raise ValueError("sandbox host_container_url must be configured for tool routing")

    def _build_run_args(self, options: SandboxOptions) -> list[str]:
        args = self._base_args(options)
        self._add_ports_and_network(args, options)
        self._add_workdir(args, options)
        self._add_volumes(args, options)
        self._add_env(args, options)
        self._add_hosts(args, options)
        self._add_security(args, options)
        self._add_entrypoint_and_cmd(args, options)
        return args

    def _base_args(self, options: SandboxOptions) -> list[str]:
        args = [
            self._docker,
            "run",
            "--pull",
            self._docker_run_pull_policy(options),
            "-d",
            "--name",
            options.container_name,
        ]
        for key, value in sorted(options.labels.items()):
            args.extend(["--label", f"{key}={value}"])
        if options.user:
            args.extend(["--user", options.user])
        return args

    @staticmethod
    def _docker_run_pull_policy(options: SandboxOptions) -> str:
        if options.pull_policy == "always":
            return "never"
        return options.pull_policy

    def _add_ports_and_network(self, args: list[str], options: SandboxOptions) -> None:
        if options.host_port is not None:
            args.extend(["-p", _published_port_spec(self._published_port_bind_host, options)])
        if options.network:
            args.extend(["--network", options.network])

    def _add_workdir(self, args: list[str], options: SandboxOptions) -> None:
        if options.working_dir:
            args.extend(["-w", options.working_dir])

    def _add_volumes(self, args: list[str], options: SandboxOptions) -> None:
        for host_path, container_path, mode in options.volumes:
            mount = f"{host_path}:{container_path}"
            if mode:
                mount = f"{mount}:{mode}"
            args.extend(["-v", mount])

    def _add_env(self, args: list[str], options: SandboxOptions) -> None:
        for key, value in options.env.items():
            args.extend(["-e", f"{key}={value}"])

    def _add_hosts(self, args: list[str], options: SandboxOptions) -> None:
        for hostname, address in options.extra_hosts:
            args.extend(["--add-host", f"{hostname}:{address}"])

    def _add_security(self, args: list[str], options: SandboxOptions) -> None:
        if options.seccomp_profile:
            args.extend(["--security-opt", f"seccomp={options.seccomp_profile}"])
        for ulimit in options.ulimits:
            args.extend(["--ulimit", ulimit])

    def _add_entrypoint_and_cmd(self, args: list[str], options: SandboxOptions) -> None:
        if options.entrypoint:
            args.extend(["--entrypoint", options.entrypoint])
        if options.extra_args:
            args.extend(options.extra_args)
        args.append(options.image)
        if options.command:
            args.extend(options.command)

    def _launch_container(self, options: SandboxOptions) -> str:
        args = self._build_run_args(options)
        return self._run_container(args, options)

    def _ready_client(self, options: SandboxOptions, *, container_id: str) -> tuple[str, SandboxClient]:
        base_url, client = self._build_client(options, container_id=container_id)
        try:
            self._maybe_wait_for_health(base_url, options)
        except Exception:
            client.close()
            raise
        return base_url, client

    def _post_launch_steps(self, options: SandboxOptions, base_url: str, container_id: str) -> None:
        self._maybe_start_logs(container_id)
        if options.startup_delay_seconds > 0:
            time.sleep(options.startup_delay_seconds)

    def _run_container(self, args: list[str], options: SandboxOptions) -> str:
        self._pull_image_if_required(options)
        logger.info(
            "launching sandbox container",
            extra={
                "image": options.image,
                "container_name": options.container_name,
                "pull_policy": options.pull_policy,
                "host_port": options.host_port,
                "container_port": options.container_port,
                "network": options.network,
            },
        )
        try:
            result = self._run_command(args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - exercised in integration
            self._write_failure_diagnostics(
                options=options,
                container_id=None,
                error=exc,
                docker_run_args=args,
                docker_run_result=exc,
            )
            self._raise_run_error(exc, args, options)
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - integration only
            self._write_failure_diagnostics(
                options=options,
                container_id=None,
                error=exc,
                docker_run_args=args,
                docker_run_result=None,
            )
            redacted_command = _shell_join(_redact_docker_run_args(args))
            raise RuntimeError(
                f"docker run timed out after {self._command_timeout_seconds:g}s: {redacted_command}"
            ) from exc
        container_id = result.stdout.strip()
        if not container_id:
            raise RuntimeError("docker run did not return a container identifier")
        return container_id

    def _pull_image_if_required(self, options: SandboxOptions) -> None:
        if options.pull_policy != "always":
            return

        args = [self._docker, "pull", options.image]
        last_error: subprocess.CalledProcessError | subprocess.TimeoutExpired | None = None
        for attempt_number in range(1, _IMAGE_PULL_ATTEMPTS + 1):
            try:
                self._run_command(args, capture_output=True, text=True, check=True)
                return
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                last_error = exc
                if attempt_number == _IMAGE_PULL_ATTEMPTS:
                    break
                logger.warning(
                    "retrying sandbox image pull failure",
                    extra={
                        "image": options.image,
                        "attempt": attempt_number,
                        "max_attempts": _IMAGE_PULL_ATTEMPTS,
                        "error_type": exc.__class__.__name__,
                    },
                )
                time.sleep(_IMAGE_PULL_INITIAL_BACKOFF_SECONDS * (2 ** (attempt_number - 1)))

        assert last_error is not None
        error = RuntimeError(f"docker pull failed after {_IMAGE_PULL_ATTEMPTS} attempts: image={options.image}")
        self._write_pull_failure_diagnostics(
            options=options,
            error=error,
            docker_pull_args=args,
            docker_pull_result=last_error if isinstance(last_error, subprocess.CalledProcessError) else None,
        )
        raise error from last_error

    def _raise_run_error(
        self,
        exc: subprocess.CalledProcessError,
        args: list[str],
        options: SandboxOptions,
    ) -> None:
        cmd_str = _shell_join(_redact_docker_run_args(args))
        stdout = _redact_sensitive_text((exc.stdout or "").strip(), options)
        stderr = _redact_sensitive_text((exc.stderr or "").strip(), options)
        context_bits = []
        if stdout:
            context_bits.append(f"stdout={stdout}")
        if stderr:
            context_bits.append(f"stderr={stderr}")
        details = f" {' | '.join(context_bits)}" if context_bits else ""
        logger.exception(
            "docker run failed (returncode=%s)%s",
            exc.returncode,
            details,
            extra={
                "container": options.container_name,
                "image": options.image,
                "docker_cmd": cmd_str,
            },
        )
        raise RuntimeError(
            f"docker run failed (returncode={exc.returncode}) cmd={cmd_str} stderr={stderr}"
        ) from exc

    def _write_failure_diagnostics(
        self,
        *,
        options: SandboxOptions,
        container_id: str | None,
        error: BaseException,
        docker_run_args: list[str] | None = None,
        docker_run_result: subprocess.CalledProcessError | None = None,
    ) -> None:
        if options.failure_diagnostics_dir is None:
            return
        diagnostics_dir = Path(options.failure_diagnostics_dir)
        try:
            ensure_private_diagnostic_dir(diagnostics_dir)
            run_args = docker_run_args or self._build_run_args(options)
            write_private_json(
                diagnostics_dir / "sandbox-options.json",
                _diagnostic_options_snapshot(options),
            )
            write_private_text(
                diagnostics_dir / "docker-run.txt",
                _shell_join(_redact_docker_run_args(run_args)),
            )
            write_private_text(diagnostics_dir / "error.txt", _diagnostic_error_text(error, options))
            if docker_run_result is not None:
                write_private_json(
                    diagnostics_dir / "docker-run-result.json",
                    {
                        "returncode": docker_run_result.returncode,
                        "stdout": _redact_sensitive_text(docker_run_result.stdout or "", options),
                        "stderr": _redact_sensitive_text(docker_run_result.stderr or "", options),
                    },
                )
            if container_id is not None:
                self._write_docker_command_output(
                    diagnostics_dir / "docker-inspect.json",
                    [self._docker, "inspect", container_id],
                    options=options,
                )
                self._write_docker_command_output(
                    diagnostics_dir / "docker-logs.txt",
                    [self._docker, "logs", container_id],
                    options=options,
                )
        except Exception as exc:  # pragma: no cover - diagnostic path must not mask failures
            logger.warning(
                "sandbox failure diagnostics could not be written: diagnostics_dir=%s error=%s",
                diagnostics_dir,
                exc,
                exc_info=exc,
            )

    def _write_pull_failure_diagnostics(
        self,
        *,
        options: SandboxOptions,
        error: BaseException,
        docker_pull_args: list[str],
        docker_pull_result: subprocess.CalledProcessError | None,
    ) -> None:
        if options.failure_diagnostics_dir is None:
            return
        diagnostics_dir = Path(options.failure_diagnostics_dir)
        try:
            ensure_private_diagnostic_dir(diagnostics_dir)
            write_private_json(
                diagnostics_dir / "sandbox-options.json",
                _diagnostic_options_snapshot(options),
            )
            write_private_text(diagnostics_dir / "docker-pull.txt", _shell_join(docker_pull_args))
            write_private_text(diagnostics_dir / "error.txt", _diagnostic_error_text(error, options))
            if docker_pull_result is not None:
                write_private_json(
                    diagnostics_dir / "docker-pull-result.json",
                    {
                        "returncode": docker_pull_result.returncode,
                        "stdout": _redact_sensitive_text(docker_pull_result.stdout or "", options),
                        "stderr": _redact_sensitive_text(docker_pull_result.stderr or "", options),
                    },
                )
        except Exception as exc:  # pragma: no cover - diagnostic path must not mask failures
            logger.warning(
                "sandbox pull failure diagnostics could not be written: diagnostics_dir=%s error=%s",
                diagnostics_dir,
                exc,
                exc_info=exc,
            )

    def _write_docker_command_output(
        self,
        path: Path,
        args: list[str],
        *,
        options: SandboxOptions,
    ) -> None:
        try:
            result = self._run_command(args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            write_private_text(
                path.with_suffix(f"{path.suffix}.error.txt"),
                (
                    f"command={_shell_join(args)}\n"
                    f"returncode={exc.returncode}\n"
                    f"stdout={_redact_sensitive_text(exc.stdout or '', options)}\n"
                    f"stderr={_redact_sensitive_text(exc.stderr or '', options)}\n"
                ),
            )
            return
        except subprocess.TimeoutExpired as exc:
            write_private_text(
                path.with_suffix(f"{path.suffix}.error.txt"),
                (
                    f"command={_shell_join(args)}\n"
                    f"timeout={exc.timeout}\n"
                    f"stdout={_redact_sensitive_text(str(exc.stdout or ''), options)}\n"
                    f"stderr={_redact_sensitive_text(str(exc.stderr or ''), options)}\n"
                ),
            )
            return
        write_private_text(path, _redact_sensitive_text(result.stdout or "", options))

    def _build_client(self, options: SandboxOptions, *, container_id: str) -> tuple[str, SandboxClient]:
        if options.host_port is not None:
            base_host = self._host
            if options.host_port == 0:
                published_port = self._resolve_published_port(options, container_id=container_id)
            else:
                published_port = options.host_port
        else:
            base_host = self._resolve_container_ip(options, container_id=container_id)
            published_port = options.container_port

        base_url = f"http://{base_host}:{published_port}"
        client = self._client_factory(base_url, options.host_container_url)
        return base_url, client

    def _resolve_published_port(self, options: SandboxOptions, *, container_id: str) -> int:
        args = [
            self._docker,
            "port",
            container_id,
            str(options.container_port),
        ]
        try:
            result = self._run_command(args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - integration only
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(f"docker port failed: stderr={stderr}") from exc
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - integration only
            raise RuntimeError(f"docker port timed out after {self._command_timeout_seconds:g}s") from exc

        output = (result.stdout or "").strip()
        if not output:
            raise RuntimeError("docker port returned an empty mapping")

        for line in output.splitlines():
            mapping = line
            if "->" in line:
                _, _, mapping = line.partition("->")
                mapping = mapping.strip()

            port_bits = mapping.rsplit(":", 1)
            if len(port_bits) != 2:
                continue
            published_port = port_bits[1].strip()
            if published_port.isdigit():
                return int(published_port)

        raise RuntimeError(f"docker port returned an unexpected mapping: {output}")

    def _resolve_container_ip(self, options: SandboxOptions, *, container_id: str) -> str:
        network = options.network
        if not network:
            raise ValueError("sandbox network must be provided when host_port is not published")

        deadline = time.monotonic() + self._container_ip_ready_timeout(options)
        last_error: _ContainerIpNotReadyError | None = None
        while True:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                if last_error is None:
                    raise RuntimeError(f"docker inspect returned invalid IP address for network: {network}")
                raise RuntimeError(str(last_error)) from last_error
            try:
                return self._resolve_container_ip_once(
                    options,
                    container_id=container_id,
                    network=network,
                    command_timeout_seconds=min(remaining_seconds, self._command_timeout_seconds),
                )
            except _ContainerIpNotReadyError as exc:
                last_error = exc
                sleep_seconds = min(
                    _CONTAINER_IP_READY_POLL_INTERVAL_SECONDS,
                    max(deadline - time.monotonic(), 0.0),
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

    def _resolve_container_ip_once(
        self,
        options: SandboxOptions,
        *,
        container_id: str,
        network: str,
        command_timeout_seconds: float,
    ) -> str:
        args = [
            self._docker,
            "inspect",
            "--format",
            "{{json .}}",
            container_id,
        ]
        try:
            result = self._run_command(
                args,
                capture_output=True,
                text=True,
                check=True,
                timeout=command_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise _ContainerIpNotReadyError(
                f"docker inspect timed out while waiting for network: {network}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            stdout = _redact_sensitive_text((exc.stdout or "").strip(), options)
            stderr = _redact_sensitive_text((exc.stderr or "").strip(), options)
            details = f"stdout={stdout} stderr={stderr}".strip()
            raise RuntimeError(
                f"docker inspect failed while resolving sandbox network: container={container_id} {details}"
            ) from exc
        output = (result.stdout or "").strip()
        if not output:
            raise _ContainerIpNotReadyError("docker inspect returned empty container details")

        container = json.loads(output)
        if not isinstance(container, dict):
            raise RuntimeError("docker inspect container details must be a JSON object")

        state = container.get("State")
        if not isinstance(state, dict):
            raise RuntimeError("docker inspect did not include container state")
        status = state.get("Status")
        if status in {"dead", "exited", "removing"}:
            exit_code = state.get("ExitCode")
            error = state.get("Error")
            raise RuntimeError(
                "sandbox container exited before readiness: "
                f"container={container_id} status={status} exit_code={exit_code} error={error}"
            )

        network_settings = container.get("NetworkSettings")
        if not isinstance(network_settings, dict):
            raise _ContainerIpNotReadyError("docker inspect did not include network settings")
        networks = network_settings.get("Networks")
        if not isinstance(networks, dict):
            raise _ContainerIpNotReadyError("docker inspect network settings must be a JSON object")

        network_details = networks.get(network)
        if not isinstance(network_details, dict):
            raise _ContainerIpNotReadyError(f"docker inspect did not include network: {network}")

        ip_address = network_details.get("IPAddress")
        if not isinstance(ip_address, str) or not ip_address:
            raise _ContainerIpNotReadyError(f"docker inspect returned invalid IP address for network: {network}")

        return ip_address

    @staticmethod
    def _container_ip_ready_timeout(options: SandboxOptions) -> float:
        if options.wait_for_healthz:
            return max(_CONTAINER_IP_READY_TIMEOUT_SECONDS, options.healthz_timeout)
        return _CONTAINER_IP_READY_TIMEOUT_SECONDS

    def _maybe_wait_for_health(self, base_url: str, options: SandboxOptions) -> None:
        if not options.wait_for_healthz:
            return
        self._wait_for_healthz(
            base_url,
            path=options.healthz_path,
            timeout_seconds=options.healthz_timeout,
        )

    def _maybe_start_logs(self, container_id: str) -> None:
        if self._log_consumer:
            self._start_log_stream(container_id)

    def stop(self, deployment: SandboxDeployment) -> None:
        identifier = deployment.identifier
        if not identifier:
            return
        args = [self._docker, "stop"]
        if deployment.stop_timeout_seconds is not None:
            args.extend(["-t", str(deployment.stop_timeout_seconds)])
        args.append(identifier)
        logger.info("stopping sandbox container", extra={"container": identifier})
        try:
            self._run_command(args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - exercised in integration
            logger.warning(
                "docker stop failed (ignored): returncode=%s stderr=%s",
                exc.returncode,
                exc.stderr,
                extra={"container": identifier, "stderr": exc.stderr},
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - integration only
            logger.warning(
                "docker stop timed out (ignored): timeout=%s",
                exc.timeout,
                extra={"container": identifier, "timeout": exc.timeout},
            )
        self._best_effort_remove(identifier)
        deployment.client.close()
        self._stop_log_stream(identifier)

    def cleanup_stale_sandbox_containers(
        self,
        *,
        labels: Mapping[str, str],
        name_prefix: str,
    ) -> None:
        """Best-effort remove stale sandbox containers selected by labels or name prefix."""
        if not name_prefix:
            logger.warning("skipping stale sandbox cleanup because name_prefix is empty")
            return
        try:
            labeled_ids: tuple[str, ...] = ()
            if labels:
                labeled_ids = self._list_container_ids_all_states(
                    tuple(("label", f"{key}={value}") for key, value in sorted(labels.items()))
                )
            prefixed_ids = tuple(
                container_id
                for status in _STALE_LEGACY_CONTAINER_STATUSES
                for container_id in self._list_container_ids_all_states(
                    (("name", f"^/{name_prefix}"), ("status", status))
                )
            )
            container_ids = sorted(set(labeled_ids) | set(prefixed_ids))
            if not container_ids:
                return
            self._run_command(
                [self._docker, "rm", "-f", *container_ids],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(
                "removed stale sandbox containers",
                extra={"container_count": len(container_ids), "name_prefix": name_prefix},
            )
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "failed to remove stale sandbox containers",
                extra={"returncode": exc.returncode, "stderr": exc.stderr, "name_prefix": name_prefix},
                exc_info=exc,
            )
        except subprocess.TimeoutExpired as exc:
            logger.warning(
                "failed to remove stale sandbox containers",
                extra={"timeout": exc.timeout, "name_prefix": name_prefix},
                exc_info=exc,
            )
        except Exception as exc:  # pragma: no cover - defensive best-effort cleanup
            logger.warning(
                "failed to remove stale sandbox containers",
                extra={"name_prefix": name_prefix},
                exc_info=exc,
            )

    def _list_container_ids_all_states(
        self,
        filters: Sequence[tuple[str, str]],
    ) -> tuple[str, ...]:
        args = [self._docker, "ps", "-aq"]
        for filter_name, filter_value in filters:
            args.extend(["--filter", f"{filter_name}={filter_value}"])
        result = self._run_command(args, capture_output=True, text=True, check=True)
        return tuple(line.strip() for line in (result.stdout or "").splitlines() if line.strip())

    def _best_effort_stop_and_remove(self, container_id: str, *, stop_timeout_seconds: int | None) -> None:
        args = [self._docker, "stop"]
        if stop_timeout_seconds is not None:
            args.extend(["-t", str(stop_timeout_seconds)])
        args.append(container_id)
        try:
            self._run_command(args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - integration only
            logger.warning(
                "docker stop failed (ignored): returncode=%s stderr=%s",
                exc.returncode,
                exc.stderr,
                extra={"container": container_id, "stderr": exc.stderr},
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - integration only
            logger.warning(
                "docker stop timed out (ignored): timeout=%s",
                exc.timeout,
                extra={"container": container_id, "timeout": exc.timeout},
            )
        self._best_effort_remove(container_id)

    def _best_effort_remove(self, container_id: str) -> None:
        args = [self._docker, "rm", "-f", container_id]
        try:
            self._run_command(args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - integration only
            logger.warning(
                "docker rm failed (ignored): returncode=%s stderr=%s",
                exc.returncode,
                exc.stderr,
                extra={"container": container_id, "stderr": exc.stderr},
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - integration only
            logger.warning(
                "docker rm timed out (ignored): timeout=%s",
                exc.timeout,
                extra={"container": container_id, "timeout": exc.timeout},
            )

    def _wait_for_healthz(self, base_url: str, *, path: str, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        url = f"{base_url}{path}"
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                response = httpx.get(url, timeout=2.0)
                if response.status_code == 200:
                    return
            except Exception as exc:  # pragma: no cover - integration only
                last_error = exc
                time.sleep(0.5)
        raise RuntimeError(f"sandbox healthz did not succeed: last_error={last_error}")

    def _start_log_stream(self, container_id: str) -> None:
        if not self._log_consumer:
            return
        args = [self._docker, "logs", "-f", container_id]
        if self._popen is None:
            raise RuntimeError("log runner is not configured")
        process = self._popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if process.stdout is None:
            return

        def _consume() -> None:
            assert process.stdout is not None
            assert self._log_consumer is not None
            for line in process.stdout:
                self._log_consumer(line.rstrip())

        thread = threading.Thread(target=_consume, daemon=True)
        thread.start()
        self._log_streams[container_id] = (process, thread)

    def _stop_log_stream(self, container_id: str) -> None:
        stream = self._log_streams.pop(container_id, None)
        if not stream:
            return
        process, thread = stream
        process.terminate()
        thread.join(timeout=5)

    @staticmethod
    def _default_run(
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:  # pragma: no cover - thin wrapper
        return subprocess.run(*args, **kwargs)  # noqa: S603

    def _run_command(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        if "timeout" not in kwargs:
            kwargs["timeout"] = self._command_timeout_seconds
        return self._run(*args, **kwargs)

    @staticmethod
    def _default_popen(
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.Popen[str]:  # pragma: no cover - thin wrapper
        return subprocess.Popen(*args, **kwargs)  # noqa: S603


def _shell_join(args: Sequence[str]) -> str:
    return shlex.join(list(args))


def _redact_docker_run_args(args: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for arg in args:
        if redact_next:
            key, separator, _ = arg.partition("=")
            redacted.append(f"{key}{separator}<redacted>" if separator else "<redacted>")
            redact_next = False
            continue
        if arg in {"-e", "--env"}:
            redacted.append(arg)
            redact_next = True
            continue
        if arg.startswith("--env="):
            prefix, _, env_value = arg.partition("=")
            key, separator, _ = env_value.partition("=")
            redacted.append(f"{prefix}={key}{separator}<redacted>" if separator else f"{prefix}=<redacted>")
            continue
        redacted.append(arg)
    return redacted


def _diagnostic_error_text(error: BaseException, options: SandboxOptions) -> str:
    return f"{error.__class__.__name__}: {_redact_sensitive_text(str(error), options)}\n"


def _redact_sensitive_text(text: str, options: SandboxOptions) -> str:
    redacted = text
    for value in _sensitive_env_values(options):
        if value:
            redacted = redacted.replace(value, "<redacted>")
    return redacted


def _sensitive_env_values(options: SandboxOptions) -> tuple[str, ...]:
    return tuple(
        value
        for key, value in options.env.items()
        if key not in _NON_SENSITIVE_DIAGNOSTIC_ENV_KEYS
    )


def _diagnostic_env_snapshot(options: SandboxOptions) -> dict[str, str]:
    return {
        key: value if key in _NON_SENSITIVE_DIAGNOSTIC_ENV_KEYS else "<redacted>"
        for key, value in sorted(options.env.items())
    }


def _diagnostic_options_snapshot(options: SandboxOptions) -> dict[str, object]:
    return {
        "image": options.image,
        "container_name": options.container_name,
        "pull_policy": options.pull_policy,
        "host_port": options.host_port,
        "container_port": options.container_port,
        "env": _diagnostic_env_snapshot(options),
        "entrypoint": options.entrypoint,
        "command": list(options.command) if options.command is not None else None,
        "network": options.network,
        "host_container_url": options.host_container_url,
        "volumes": [list(volume) for volume in options.volumes],
        "working_dir": options.working_dir,
        "extra_hosts": [list(extra_host) for extra_host in options.extra_hosts],
        "startup_delay_seconds": options.startup_delay_seconds,
        "wait_for_healthz": options.wait_for_healthz,
        "healthz_path": options.healthz_path,
        "healthz_timeout": options.healthz_timeout,
        "stop_timeout_seconds": options.stop_timeout_seconds,
        "extra_args": list(options.extra_args),
        "user": options.user,
        "seccomp_profile": options.seccomp_profile,
        "ulimits": list(options.ulimits),
        "labels": dict(sorted(options.labels.items())),
    }


def resolve_network_gateway(*, docker_binary: str, network: str) -> str:
    args = [docker_binary, "network", "inspect", network, "--format", "{{json .IPAM.Config}}"]
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:  # pragma: no cover - integration only
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            f"docker network inspect failed for network={network}: stderr={stderr}"
        ) from exc

    output = (result.stdout or "").strip()
    if not output:
        raise RuntimeError(f"docker network inspect returned empty output for network={network}")

    config = json.loads(output)
    if not isinstance(config, list) or not config:
        raise TypeError("docker network inspect IPAM config must be a non-empty JSON list")

    first = config[0]
    if not isinstance(first, dict):
        raise TypeError("docker network inspect IPAM config entry must be a JSON object")

    gateway = first.get("Gateway")
    if not isinstance(gateway, str) or not gateway:
        raise RuntimeError(f"docker network inspect did not include a Gateway for network={network}")

    return gateway


def resolve_container_ip(*, docker_binary: str, container: str, network: str) -> str:
    args = [docker_binary, "inspect", "--format", "{{json .NetworkSettings.Networks}}", container]
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:  # pragma: no cover - integration only
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            f"docker inspect failed for container={container}: stderr={stderr}"
        ) from exc

    output = (result.stdout or "").strip()
    if not output:
        raise RuntimeError(f"docker inspect returned empty network settings for container={container}")

    networks = json.loads(output)
    if not isinstance(networks, dict):
        raise TypeError("docker inspect network settings must be a JSON object")

    network_details = networks.get(network)
    if not isinstance(network_details, dict):
        raise RuntimeError(f"docker inspect did not include network={network} for container={container}")

    ip_address = network_details.get("IPAddress")
    if not isinstance(ip_address, str) or not ip_address:
        raise RuntimeError(
            f"docker inspect returned invalid IP address for container={container} network={network}"
        )

    return ip_address


def _resolve_current_container_id_from_mountinfo() -> str | None:
    try:
        mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError:
        return None
    match = _MOUNTINFO_CONTAINER_ID_PATTERN.search(mountinfo)
    if match is None:
        return None
    return match.group(1)


def _resolve_runtime_container_ip(*, docker_binary: str, network: str) -> str:
    container = os.getenv("HOSTNAME")
    if not container:
        raise RuntimeError("HOSTNAME must be set to derive HOST_CONTAINER_URL in containerized runs")
    try:
        return resolve_container_ip(docker_binary=docker_binary, container=container, network=network)
    except RuntimeError as hostname_error:
        mountinfo_container = _resolve_current_container_id_from_mountinfo()
        if mountinfo_container is None or mountinfo_container == container:
            raise
        logger.warning(
            "falling back to mountinfo-derived container id for sandbox host resolution: "
            "hostname_container=%s mountinfo_container=%s network=%s error=%s",
            container,
            mountinfo_container,
            network,
            hostname_error,
        )
        try:
            return resolve_container_ip(
                docker_binary=docker_binary,
                container=mountinfo_container,
                network=network,
            )
        except RuntimeError as mountinfo_error:
            raise RuntimeError(
                "failed to resolve current container IP "
                f"via HOSTNAME={container} and mountinfo_container={mountinfo_container}"
            ) from mountinfo_error


def resolve_sandbox_host_container_url(
    *, docker_binary: str, sandbox_network: str | None, rpc_port: int
) -> str:
    if not sandbox_network:
        raise RuntimeError("SANDBOX_NETWORK must be configured to derive HOST_CONTAINER_URL")

    if os.getenv("KUBERNETES_SERVICE_HOST"):
        host = resolve_network_gateway(docker_binary=docker_binary, network=sandbox_network)
    elif Path("/.dockerenv").exists():
        host = _resolve_runtime_container_ip(docker_binary=docker_binary, network=sandbox_network)
    else:
        host = resolve_network_gateway(docker_binary=docker_binary, network=sandbox_network)

    return f"http://{host}:{rpc_port}"


__all__ = [
    "DockerSandboxManager",
    "SandboxOptions",
    "HttpSandboxClient",
    "resolve_container_ip",
    "resolve_network_gateway",
    "resolve_sandbox_host_container_url",
]
