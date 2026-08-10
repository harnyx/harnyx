from __future__ import annotations

import logging

import pytest

import harnyx_commons.sandbox.runtime as runtime_module


def test_create_sandbox_manager_uses_default_host_probe_address() -> None:
    manager = runtime_module.create_sandbox_manager(logger_name="test.runtime")

    assert manager._host == runtime_module.HOST_PROBE_ADDRESS
    assert manager._published_port_bind_host is None


def test_create_sandbox_manager_accepts_host_override() -> None:
    manager = runtime_module.create_sandbox_manager(
        logger_name="test.runtime",
        host="127.0.0.1",
    )

    assert manager._host == "127.0.0.1"
    assert manager._published_port_bind_host is None


def test_create_sandbox_manager_accepts_published_port_bind_host_override() -> None:
    manager = runtime_module.create_sandbox_manager(
        logger_name="test.runtime",
        host="127.0.0.1",
        published_port_bind_host="127.0.0.1",
    )

    assert manager._host == "127.0.0.1"
    assert manager._published_port_bind_host == "127.0.0.1"


def test_sandbox_manager_maps_owned_invocation_output_and_preserves_other_lines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = runtime_module.create_sandbox_manager(logger_name="test.runtime")
    caplog.set_level(logging.INFO, logger="test.runtime")

    manager._log_consumer(
        'HARNYX_SANDBOX_INVOCATION_OUTPUT {"session_id":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",'
        '"entrypoint":"query",'
        '"stream":"stdout","message":"received"}'
    )
    manager._log_consumer("ordinary container lifecycle line")

    assert caplog.records[0].message == "sandbox_invocation.output"
    assert caplog.records[0].data == {
        "session_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "entrypoint": "query",
        "stream": "stdout",
        "message": "received",
    }
    assert caplog.records[1].message == "ordinary container lifecycle line"


@pytest.mark.parametrize(
    "record",
    (
        '{"session_id":"not-a-uuid","entrypoint":"query","stream":"stdout","message":"received"}',
        '{"session_id":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa","entrypoint":"query",'
        '"stream":"side-channel","message":"received"}',
        '{"session_id":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa","entrypoint":"query",'
        '"stream":"stdout","message":"received","extra":"unexpected"}',
    ),
)
def test_sandbox_manager_preserves_malformed_invocation_output_as_raw_log(
    caplog: pytest.LogCaptureFixture,
    record: str,
) -> None:
    manager = runtime_module.create_sandbox_manager(logger_name="test.runtime")
    caplog.set_level(logging.INFO, logger="test.runtime")
    line = runtime_module.SANDBOX_INVOCATION_OUTPUT_PREFIX + record

    manager._log_consumer(line)

    assert caplog.records[-1].message == line


def test_build_sandbox_options_accepts_explicit_host_container_url_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "resolve_sandbox_host_container_url",
        lambda **_: pytest.fail("explicit host_container_url should bypass resolver"),
    )

    options = runtime_module.build_sandbox_options(
        image="harnyx/sandbox:demo",
        network=None,
        pull_policy="missing",
        rpc_port=8100,
        container_name="sandbox-demo",
        host_container_url="http://host.docker.internal:39100",
    )

    assert options.host_container_url == "http://host.docker.internal:39100"
    assert options.host_port == 0
    assert options.network is None
