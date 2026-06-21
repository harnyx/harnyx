from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.integration

_AGENT_MODULE = "commons.tests.integration.sandbox.seccomp_agent"


def test_docker_sandbox_manager_starts_sandbox_with_explicit_host_port(sandbox_launcher) -> None:
    deployment = sandbox_launcher(agent_module=_AGENT_MODULE)

    assert deployment.base_url.startswith("http://127.0.0.1:")
    response = httpx.get(f"{deployment.base_url}/healthz", timeout=2.0)
    assert response.status_code == 200


def test_docker_sandbox_manager_starts_sandbox_with_dynamic_host_port(sandbox_launcher) -> None:
    deployment = sandbox_launcher(agent_module=_AGENT_MODULE, host_port=0)

    assert deployment.base_url.startswith("http://127.0.0.1:")
    response = httpx.get(f"{deployment.base_url}/healthz", timeout=2.0)
    assert response.status_code == 200
