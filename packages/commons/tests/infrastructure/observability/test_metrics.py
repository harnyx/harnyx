from __future__ import annotations

import re

import pytest
from prometheus_client import generate_latest

from harnyx_commons.observability import metrics as metrics_mod


@pytest.fixture(autouse=True)
def reset_metrics_bootstrap() -> None:
    metrics_mod._reset_metrics_for_tests()
    yield
    metrics_mod._reset_metrics_for_tests()


def test_get_meter_provider_requires_bootstrap() -> None:
    with pytest.raises(RuntimeError, match="Metrics bootstrap has not run"):
        metrics_mod.get_meter_provider()


def test_configure_metrics_is_idempotent() -> None:
    metrics_mod.configure_metrics(service_name="test-service")
    first_provider = metrics_mod.get_meter_provider()

    metrics_mod.configure_metrics(service_name="other-service")

    assert metrics_mod.get_meter_provider() is first_provider


def test_prometheus_registry_exposes_metrics_after_bootstrap() -> None:
    metrics_mod.configure_metrics(service_name="test-service")
    meter = metrics_mod.get_meter_provider().get_meter("commons-metrics-test")
    counter = meter.create_counter("test_counter")

    counter.add(1, {"kind": "demo"})

    payload = generate_latest().decode()

    assert "test_counter" in payload
    assert 'kind="demo"' in payload


def test_http_server_duration_histogram_has_long_running_buckets() -> None:
    metrics_mod.configure_metrics(service_name="test-service")
    meter = metrics_mod.get_meter_provider().get_meter("commons-metrics-test")
    duration = meter.create_histogram("http.server.duration", unit="ms")

    duration.record(20000, {"http.target": "/probe", "http.status_code": 200})

    payload = generate_latest().decode()

    assert "http_server_duration_milliseconds_bucket" in payload
    assert _bucket_value(payload, "10000", "10000.0") == 0.0
    assert _bucket_value(payload, "30000", "30000.0") == 1.0
    assert _bucket_value(payload, "900000", "900000.0") == 1.0
    assert _bucket_value(payload, "+Inf") == 1.0


def _bucket_value(payload: str, *les: str) -> float:
    for le in les:
        match = re.search(
            rf'^http_server_duration_milliseconds_bucket{{[^}}]*le="{re.escape(le)}"[^}}]*}} (?P<value>[0-9.]+)$',
            payload,
            re.MULTILINE,
        )
        if match is not None:
            return float(match.group("value"))
    raise AssertionError(f"missing http.server.duration bucket le in {les!r}")
