"""Shared OpenTelemetry metrics bootstrap helpers."""

from __future__ import annotations

import os

from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

_METRICS_CONFIGURED = False
_METER_PROVIDER: MeterProvider | None = None
_PROMETHEUS_READER: PrometheusMetricReader | None = None
_HTTP_SERVER_DURATION_MILLISECONDS_BUCKETS = (
    0,
    5,
    10,
    25,
    50,
    75,
    100,
    250,
    500,
    750,
    1000,
    2500,
    5000,
    7500,
    10000,
    30000,
    60000,
    120000,
    300000,
    600000,
    900000,
)


def _metrics_views() -> tuple[View, ...]:
    return (
        View(
            instrument_name="http.server.duration",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=_HTTP_SERVER_DURATION_MILLISECONDS_BUCKETS,
            ),
        ),
    )


def configure_metrics(*, service_name: str) -> None:
    """Configure OpenTelemetry metrics for Prometheus scrape exposure."""

    global _METRICS_CONFIGURED, _METER_PROVIDER, _PROMETHEUS_READER
    if _METRICS_CONFIGURED:
        return

    resolved_service_name = (os.getenv("OTEL_SERVICE_NAME") or service_name).strip()
    if not resolved_service_name:
        raise RuntimeError("service_name must be a non-empty string")

    resource = Resource.create({SERVICE_NAME: resolved_service_name})
    reader = PrometheusMetricReader()
    provider = MeterProvider(
        resource=resource,
        metric_readers=[reader],
        views=_metrics_views(),
    )

    _PROMETHEUS_READER = reader
    _METER_PROVIDER = provider
    _METRICS_CONFIGURED = True


def get_meter_provider() -> MeterProvider:
    """Return the configured meter provider or fail loudly."""

    if _METER_PROVIDER is None:
        raise RuntimeError("Metrics bootstrap has not run")
    return _METER_PROVIDER


def _reset_metrics_for_tests() -> None:
    """Reset the shared metrics bootstrap for test isolation."""

    global _METRICS_CONFIGURED, _METER_PROVIDER, _PROMETHEUS_READER

    if _METER_PROVIDER is not None:
        _METER_PROVIDER.shutdown()

    _METER_PROVIDER = None
    _PROMETHEUS_READER = None
    _METRICS_CONFIGURED = False


__all__ = [
    "configure_metrics",
    "get_meter_provider",
]
