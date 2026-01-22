"""Shared logging helpers (formatter + base config builder)."""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
import types
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from logging.config import dictConfig
from typing import Any

from google.cloud import logging as gcp_logging
from opentelemetry import baggage, trace


def _level(env_var: str, default: str) -> str:
    return os.getenv(env_var, default).upper()


def _should_emit_json_payload() -> bool:
    # Cloud Run and Kubernetes log ingestion can parse JSON log lines into structured payloads.
    # Outside managed runtimes we keep logs human-readable by default.
    return bool(os.getenv("K_SERVICE") or os.getenv("KUBERNETES_SERVICE_HOST"))


def _compact_json(value: Any, *, limit: int = 512) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if len(encoded) <= limit:
        return encoded
    return encoded[:limit] + "... (truncated)"


def _structured_payload(record: logging.LogRecord) -> dict[str, Any]:
    record_dict = record.__dict__
    record_data = record_dict.get("data")
    record_json_fields = record_dict.get("json_fields")

    message = record.getMessage()
    sanitized_data: Any | None = None
    if record_data:
        sanitized_data = _sanitize_for_json(record_data)
        message = f"{message} | data={_compact_json(sanitized_data)}"

    payload: dict[str, Any] = {
        "message": message,
        "severity": record.levelname,
        "logger": record.name,
        "timestamp": (
            f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(record.created))}"
            f".{int(record.msecs):03d}Z"
        ),
    }
    if record.exc_info:
        payload["exception"] = "".join(traceback.format_exception(*record.exc_info)).rstrip("\n")
    if record.stack_info:
        payload["stack_info"] = str(record.stack_info)
    if sanitized_data is not None:
        payload["data"] = sanitized_data

    if record_json_fields:
        json_fields = _sanitize_for_json(record_json_fields)
        if isinstance(json_fields, Mapping):
            for key, value in json_fields.items():
                if key in payload:
                    payload.setdefault("json_fields", {})[key] = value
                else:
                    payload[key] = value
        else:
            payload["json_fields"] = json_fields

    return payload


class ExtrasFormatter(logging.Formatter):
    """Append structured `data` payloads when present."""

    def format(self, record: logging.LogRecord) -> str:
        record_dict = record.__dict__
        record_data = record_dict.get("data")

        if _should_emit_json_payload():
            return json.dumps(_structured_payload(record), sort_keys=True, separators=(",", ":"))

        formatted = super().format(record)
        if record_data:
            try:
                encoded = json.dumps(record_data, sort_keys=True, separators=(",", ":"))
            except TypeError:
                encoded = str(record_data)
            return f"{formatted} | data={encoded}"
        return formatted


class CloudJsonSanitizer(logging.Filter):
    """Make json_fields/data JSON-serializable before Cloud Logging ships them."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - thin wrapper
        record_dict = record.__dict__
        sanitized_data: Any | None = None
        if "data" in record_dict:
            sanitized_data = _sanitize_for_json(record_dict["data"])
            record_dict["data"] = sanitized_data
        if "json_fields" in record_dict:
            json_fields = _sanitize_for_json(record_dict["json_fields"])
            if not isinstance(json_fields, Mapping):
                json_fields = {"json_fields": json_fields}
            if sanitized_data is not None and "data" not in json_fields:
                json_fields["data"] = sanitized_data
            record_dict["json_fields"] = json_fields
        return True


class OtelContextLogFilter(logging.Filter):
    """Inject OpenTelemetry trace context + baggage into json_fields."""

    def __init__(self, *, gcp_project_id: str | None = None) -> None:
        super().__init__()
        self._gcp_project_id = gcp_project_id.strip() if gcp_project_id else None

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - thin wrapper
        record_dict = record.__dict__
        json_fields = record_dict.get("json_fields")

        if json_fields is None:
            json_fields_map: dict[str, Any] = {}
        elif isinstance(json_fields, Mapping):
            json_fields_map = dict(json_fields)
        else:
            json_fields_map = {"json_fields": json_fields}

        otel_value = json_fields_map.get("otel")
        otel: dict[str, Any]
        if isinstance(otel_value, Mapping):
            otel = dict(otel_value)
        else:
            otel = {}

        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            trace_id = f"{span_context.trace_id:032x}"
            span_id = f"{span_context.span_id:016x}"
            otel["trace_id"] = trace_id
            otel["span_id"] = span_id

            if self._gcp_project_id:
                json_fields_map.setdefault(
                    "logging.googleapis.com/trace",
                    f"projects/{self._gcp_project_id}/traces/{trace_id}",
                )
                json_fields_map.setdefault("logging.googleapis.com/spanId", span_id)
                json_fields_map.setdefault(
                    "logging.googleapis.com/trace_sampled",
                    bool(span_context.trace_flags.sampled),
                )

        baggage_values = baggage.get_all()
        if baggage_values:
            otel["baggage"] = {key: str(value) for key, value in baggage_values.items()}

        if not otel:
            return True

        json_fields_map["otel"] = otel
        record_dict["json_fields"] = json_fields_map
        return True


def build_log_config(
    *,
    root_level_env: str,
    root_default: str,
    extra_loggers: Mapping[str, dict[str, Any]] | None = None,
    cloud_logging_enabled: bool = False,
    gcp_project: str | None = None,
    cloud_log_name: str = "llm-traffic",
    cloud_log_labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a dictConfig-compatible logging configuration."""

    logger = logging.getLogger("caster_commons.observability.logging")
    start = time.monotonic()

    cloud_handler_name = None
    cloud_handler = None
    if cloud_logging_enabled:
        if not gcp_project:
            raise RuntimeError("GCP project required when cloud logging is enabled")
        cloud_handler_name = "cloud_logging"
        logger.debug(
            "building cloud logging handler",
            extra={"data": {"project": gcp_project, "log_name": cloud_log_name}},
        )
        cloud_handler = _cloud_logging_handler(gcp_project, cloud_log_name, cloud_log_labels)

    loggers = _logger_definitions(extra_loggers, cloud_handler_name)

    handlers = _handlers(cloud_handler_name, cloud_handler)

    logger.debug(
        "built log config",
        extra={
            "data": {
                "cloud_logging_enabled": cloud_logging_enabled,
                "elapsed_s": round(time.monotonic() - start, 3),
            }
        },
    )
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": _formatters(),
        "filters": _filters(gcp_project_id=gcp_project),
        "handlers": handlers,
        "root": _root_logger(root_level_env, root_default, cloud_handler_name),
        "loggers": loggers,
    }


def _logger_definitions(
    extra_loggers: Mapping[str, dict[str, Any]] | None,
    cloud_handler_name: str | None,
) -> dict[str, dict[str, Any]]:
    loggers: dict[str, dict[str, Any]] = {
        "uvicorn": {
            "level": _level("UVICORN_LOG_LEVEL", "INFO"),
            "handlers": _handler_list(cloud_handler_name),
            "propagate": False,
        },
        "uvicorn.error": {
            "level": _level("UVICORN_LOG_LEVEL", "INFO"),
            "handlers": _handler_list(cloud_handler_name),
            "propagate": False,
        },
        "uvicorn.access": {
            "level": _level("UVICORN_ACCESS_LOG_LEVEL", "WARNING"),
            "handlers": _handler_list(cloud_handler_name),
            "propagate": False,
        },
        "httpx": {
            "level": _level("HTTPX_LOG_LEVEL", "WARNING"),
            "handlers": _handler_list(cloud_handler_name),
            "propagate": False,
        },
        "httpcore": {
            "level": _level("HTTPX_LOG_LEVEL", "WARNING"),
            "handlers": _handler_list(cloud_handler_name),
            "propagate": False,
        },
        "caster_commons.llm.calls": {
            "level": _level("LLM_LOG_LEVEL", "WARNING"),
            "handlers": _handler_list(cloud_handler_name),
            "propagate": False,
        },
        "websockets.client": {
            "level": _level("WEBSOCKETS_LOG_LEVEL", "WARNING"),
            "handlers": _handler_list(cloud_handler_name),
            "propagate": False,
        },
    }
    if extra_loggers:
        loggers.update(extra_loggers)
    if cloud_handler_name:
        for config in loggers.values():
            handlers = config.get("handlers")
            if isinstance(handlers, list) and cloud_handler_name not in handlers:
                handlers.append(cloud_handler_name)
    return loggers


def _formatters() -> dict[str, Any]:
    return {
        "console": {
            "()": ExtrasFormatter,
            "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        }
    }


def _handlers(cloud_handler_name: str | None, cloud_handler: dict[str, Any] | None) -> dict[str, Any]:
    handlers: dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "stream": "ext://sys.stdout",
            "filters": ["otel_context"],
        }
    }
    if cloud_handler_name and cloud_handler:
        cloud_handler = dict(cloud_handler)
        cloud_handler["filters"] = ["otel_context", "cloud_json_sanitizer"]
        handlers[cloud_handler_name] = cloud_handler
    return handlers


def _root_logger(root_level_env: str, root_default: str, cloud_handler_name: str | None) -> dict[str, Any]:
    return {
        "level": _level(root_level_env, root_default),
        "handlers": _handler_list(cloud_handler_name),
    }


def _handler_list(cloud_handler_name: str | None) -> list[str]:
    handlers = ["console"]
    if cloud_handler_name:
        handlers.append(cloud_handler_name)
    return handlers


def _cloud_logging_handler(
    project: str,
    log_name: str,
    labels: Mapping[str, str] | None,
) -> dict[str, Any]:
    logger = logging.getLogger("caster_commons.observability.logging")
    start = time.monotonic()
    from google.cloud.logging_v2.resource import Resource

    from caster_commons.gcp.credentials import service_account_credentials_from_b64

    service_account_b64 = os.getenv("GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64")
    logger.debug(
        "initializing cloud logging client",
        extra={
            "data": {
                "project": project,
                "log_name": log_name,
                "service_account_b64_present": bool(service_account_b64 and service_account_b64.strip()),
                "service_account_b64_len": len(service_account_b64) if service_account_b64 else 0,
            }
        },
    )
    credentials = (
        service_account_credentials_from_b64(
            service_account_b64.strip(),
            source="GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64",
        )
        if service_account_b64
        else None
    )
    logger.debug(
        "creating google cloud logging client",
        extra={"data": {"project": project}},
    )
    client: gcp_logging.Client = gcp_logging.Client(  # type: ignore[no-untyped-call]
        project=project,
        credentials=credentials,
    )
    logger.debug(
        "created google cloud logging client",
        extra={"data": {"project": project, "elapsed_s": round(time.monotonic() - start, 3)}},
    )
    return {
        "level": "INFO",
        "class": "google.cloud.logging_v2.handlers.handlers.CloudLoggingHandler",
        "client": client,
        "name": log_name,
        "resource": Resource("global", {"project_id": project}),
        "labels": dict(labels or {}),
        "formatter": "console",
    }


def _filters(*, gcp_project_id: str | None) -> dict[str, Any]:
    return {
        "otel_context": {
            "()": OtelContextLogFilter,
            "gcp_project_id": gcp_project_id,
        },
        "cloud_json_sanitizer": {
            "()": CloudJsonSanitizer,
        }
    }


def _sanitize_for_json(value: Any, depth: int = 10, max_items: int = 200) -> Any:
    """Return a JSON-serializable copy; fallback to string for unknowns."""

    if depth <= 0:
        return "<depth_exceeded>"

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (bytes, bytearray)):
        return f"<bytes len={len(value)}>"

    if isinstance(value, (types.BuiltinFunctionType, types.FunctionType, types.MethodType)):
        return f"<callable {value.__name__}>"
    if callable(value):
        return f"<callable {value.__class__.__name__}>"

    if is_dataclass(value) and not isinstance(value, type):
        return _sanitize_for_json(asdict(value), depth - 1, max_items)

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= max_items:
                result["<truncated>"] = f"...{len(value) - idx} more"
                break
            result[str(k)] = _sanitize_for_json(v, depth - 1, max_items)
        return result

    if isinstance(value, (list, tuple, set)):
        out = []
        iterable = list(value)
        for idx, item in enumerate(iterable):
            if idx >= max_items:
                out.append(f"... {len(iterable) - idx} more")
                break
            out.append(_sanitize_for_json(item, depth - 1, max_items))
        return out

    try:
        obj_dict = vars(value)
    except TypeError:
        obj_dict = None
    if obj_dict is not None:
        return _sanitize_for_json(obj_dict, depth - 1, max_items)

    try:
        return str(value)
    except Exception:  # pragma: no cover - very rare
        return "<unrepresentable>"


def shutdown_logging() -> None:
    """Flush/close CloudLoggingHandler instances proactively."""

    try:
        from google.cloud.logging_v2.handlers.handlers import CloudLoggingHandler
    except Exception:  # pragma: no cover - optional dependency
        return

    seen: set[int] = set()
    root = logging.getLogger()
    loggers = [root]
    loggers.extend(
        logger
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    )
    for logger in loggers:
        for handler in logger.handlers:
            hid = id(handler)
            if hid in seen:
                continue
            seen.add(hid)
            if isinstance(handler, CloudLoggingHandler):
                try:
                    handler.flush()  # type: ignore[no-untyped-call]
                except Exception:  # noqa: S110
                    pass
                try:
                    handler.close()  # type: ignore[no-untyped-call]
                except Exception:  # noqa: S110
                    pass


def configure_logging(
    *,
    root_level_env: str,
    root_default: str,
    extra_loggers: Mapping[str, dict[str, Any]] | None = None,
    cloud_logging_enabled: bool = False,
    gcp_project: str | None = None,
    cloud_log_name: str = "llm-traffic",
    cloud_log_labels: Mapping[str, str] | None = None,
) -> None:
    """Apply the shared logging config."""
    logger = logging.getLogger("caster_commons.observability.logging")
    start = time.monotonic()
    logger.debug(
        "configuring logging",
        extra={
            "data": {
                "cloud_logging_enabled": cloud_logging_enabled,
                "gcp_project": gcp_project,
                "cloud_log_name": cloud_log_name,
            }
        },
    )
    config = build_log_config(
        root_level_env=root_level_env,
        root_default=root_default,
        extra_loggers=extra_loggers or {},
        cloud_logging_enabled=cloud_logging_enabled,
        gcp_project=gcp_project,
        cloud_log_name=cloud_log_name,
        cloud_log_labels=cloud_log_labels,
    )
    dictConfig(config)
    logger.debug(
        "configured logging",
        extra={"data": {"elapsed_s": round(time.monotonic() - start, 3)}},
    )
    _reset_package_logger_levels(
        root_level=logging.getLogger().level,
        explicit_loggers=set(config.get("loggers", {})),
    )


_PACKAGE_LOGGER_ROOTS: tuple[str, ...] = (
    "caster_platform",
    "caster_validator",
    "caster_miner",
    "caster_commons",
)


def _reset_package_logger_levels(*, root_level: int, explicit_loggers: set[str]) -> None:
    """Restore our package loggers after bittensor silences third parties."""

    for root_name in _PACKAGE_LOGGER_ROOTS:
        logger = logging.getLogger(root_name)
        logger.setLevel(root_level)
        logger.propagate = True

    for name, entry in logging.Logger.manager.loggerDict.items():
        if not isinstance(entry, logging.Logger):
            continue
        for root_name in _PACKAGE_LOGGER_ROOTS:
            if name.startswith(f"{root_name}.") and name not in explicit_loggers:
                entry.setLevel(logging.NOTSET)
                break


__all__ = ["ExtrasFormatter", "build_log_config", "configure_logging", "shutdown_logging"]
