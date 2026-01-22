"""Docker-backed sandbox manager implementation shared across services."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any
from uuid import UUID

import httpx

from caster_commons.json_types import JsonValue
from caster_commons.sandbox.client import SandboxClient
from caster_commons.sandbox.manager import SandboxDeployment, SandboxManager, default_token_header
from caster_commons.sandbox.options import SandboxOptions

logger = logging.getLogger(__name__)


class HttpSandboxClient(SandboxClient):
    """Sandbox client backed by an HTTP endpoint exposed by the sandbox container."""

    def __init__(
        self,
        base_url: str,
        *,
        token_header: str | None = None,
        timeout: float = 45.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._token_header = token_header or default_token_header()
        self._owns_client = client is None
        self._client: httpx.AsyncClient = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
        )

    async def invoke(
        self,
        entrypoint: str,
        *,
        payload: Mapping[str, JsonValue],
        context: Mapping[str, JsonValue],
        token: str,
        session_id: UUID,
    ) -> Mapping[str, JsonValue]:
        try:
            response = await self._client.post(
                f"/entry/{entrypoint}",
                json={
                    "payload": dict(payload),
                    "context": dict(context),
                },
                headers={
                    self._token_header: token,
                    "x-caster-session-id": str(session_id),
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _summarize_response(exc.response)
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
            raise RuntimeError(
                f"sandbox entrypoint request failed with status {status}: {detail}",
            ) from exc
        body = response.json()
        result = body.get("result", body)
        if not isinstance(result, Mapping):
            raise ValueError("sandbox response must be a mapping")
        return dict(result)

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
    try:
        data = response.json()
    except ValueError:
        data = response.text
    if isinstance(data, Mapping) and "detail" in data:
        data = data["detail"]
    text = str(data)
    return text if len(text) <= 500 else text[:500] + "…"


class DockerSandboxManager(SandboxManager):
    """Launches sandbox containers using the Docker CLI."""

    _popen: Callable[..., subprocess.Popen[str]]

    def __init__(
        self,
        *,
        docker_binary: str = "docker",
        host: str = "127.0.0.1",
        command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        client_factory: Callable[[str, str], SandboxClient] | None = None,
        log_consumer: Callable[[str], None] | None = None,
        log_runner: Callable[..., subprocess.Popen[str]] | None = None,
    ) -> None:
        self._docker = docker_binary
        self._host = host
        self._run = command_runner or self._default_run
        self._client_factory = client_factory or (
            lambda base_url, token_header: HttpSandboxClient(
                base_url,
                token_header=token_header,
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
            base_url, client = self._ready_client(options)
            self._post_launch_steps(options, base_url, container_id)
        except Exception:
            if client is not None:
                client.close()
            self._stop_log_stream(container_id)
            self._best_effort_stop(container_id, stop_timeout_seconds=options.stop_timeout_seconds)
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
            options.pull_policy,
            "-d",
            "--rm",
            "--name",
            options.container_name,
        ]
        if options.user:
            args.extend(["--user", options.user])
        return args

    def _add_ports_and_network(self, args: list[str], options: SandboxOptions) -> None:
        if options.host_port is not None:
            if options.host_port == 0:
                args.extend(["-p", str(options.container_port)])
            else:
                args.extend(["-p", f"{options.host_port}:{options.container_port}"])
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

    def _ready_client(self, options: SandboxOptions) -> tuple[str, SandboxClient]:
        base_url, client = self._build_client(options)
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
            result = self._run(args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - exercised in integration
            self._raise_run_error(exc, args, options)
        container_id = result.stdout.strip()
        if not container_id:
            raise RuntimeError("docker run did not return a container identifier")
        return container_id

    def _raise_run_error(
        self,
        exc: subprocess.CalledProcessError,
        args: list[str],
        options: SandboxOptions,
    ) -> None:
        cmd_str = " ".join(str(part) for part in (exc.cmd or args))
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
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

    def _build_client(self, options: SandboxOptions) -> tuple[str, SandboxClient]:
        if options.host_port is not None:
            base_host = self._host
            if options.host_port == 0:
                published_port = self._resolve_published_port(options)
            else:
                published_port = options.host_port
        else:
            base_host = options.container_name
            published_port = options.container_port

        base_url = f"http://{base_host}:{published_port}"
        client = self._client_factory(base_url, options.token_header)
        return base_url, client

    def _resolve_published_port(self, options: SandboxOptions) -> int:
        args = [
            self._docker,
            "port",
            options.container_name,
            str(options.container_port),
        ]
        try:
            result = self._run(args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - integration only
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(f"docker port failed: stderr={stderr}") from exc

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
            self._run(args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - exercised in integration
            logger.warning(
                "docker stop failed (ignored): returncode=%s stderr=%s",
                exc.returncode,
                exc.stderr,
                extra={"container": identifier, "stderr": exc.stderr},
            )
        deployment.client.close()
        self._stop_log_stream(identifier)

    def _best_effort_stop(self, container_id: str, *, stop_timeout_seconds: int | None) -> None:
        args = [self._docker, "stop"]
        if stop_timeout_seconds is not None:
            args.extend(["-t", str(stop_timeout_seconds)])
        args.append(container_id)
        try:
            self._run(args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - integration only
            logger.warning(
                "docker stop failed (ignored): returncode=%s stderr=%s",
                exc.returncode,
                exc.stderr,
                extra={"container": container_id, "stderr": exc.stderr},
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

    @staticmethod
    def _default_popen(
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.Popen[str]:  # pragma: no cover - thin wrapper
        return subprocess.Popen(*args, **kwargs)  # noqa: S603


__all__ = [
    "DockerSandboxManager",
    "SandboxOptions",
    "HttpSandboxClient",
]
