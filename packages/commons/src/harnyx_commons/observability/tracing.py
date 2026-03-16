"""Shared tracing helpers (OpenTelemetry bootstrap + baggage attach helper)."""

from __future__ import annotations

import os
from contextvars import Token

from opentelemetry import baggage, context, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_TRACING_CONFIGURED = False


def configure_tracing(*, service_name: str) -> None:
    """Configure OpenTelemetry tracing when an OTLP endpoint is provided.

    This is intentionally opt-in: if no OTLP endpoint is configured via env, this
    function is a no-op (operators don't need to set anything up).

    If tracing is explicitly enabled but missing exporter config, it fails loudly.
    """

    global _TRACING_CONFIGURED
    if _TRACING_CONFIGURED:
        return

    traces_exporter = (os.getenv("OTEL_TRACES_EXPORTER") or "").strip().lower()
    if traces_exporter == "none":
        _TRACING_CONFIGURED = True
        return

    otlp_endpoint = (
        os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    if not otlp_endpoint:
        if traces_exporter and traces_exporter != "none":
            raise RuntimeError(
                "Tracing enabled but OTLP endpoint missing: set OTEL_EXPORTER_OTLP_ENDPOINT "
                "(or OTEL_EXPORTER_OTLP_TRACES_ENDPOINT), or set OTEL_TRACES_EXPORTER=none."
            )
        _TRACING_CONFIGURED = True
        return

    resolved_service_name = (os.getenv("OTEL_SERVICE_NAME") or service_name).strip()
    if not resolved_service_name:
        raise RuntimeError("service_name must be a non-empty string")

    resource = Resource.create({"service.name": resolved_service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _TRACING_CONFIGURED = True


def attach_baggage(values: dict[str, str]) -> Token[context.Context]:
    """Attach the supplied baggage values to the current context; returns a detach token."""

    ctx = None
    for key, value in values.items():
        ctx = baggage.set_baggage(key, value, context=ctx)
    return context.attach(ctx) if ctx is not None else context.attach(context.get_current())


__all__ = [
    "attach_baggage",
    "configure_tracing",
]
