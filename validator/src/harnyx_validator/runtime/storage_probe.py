"""One background compatibility trial, independent of miner work and registration."""
from __future__ import annotations

import logging
import os
import threading
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from harnyx_commons.storage_diagnostics import StorageProbeReport
from harnyx_validator.infrastructure.artifact_storage import ArtifactStorageSandboxManager

logger = logging.getLogger(__name__)
PROBE_POOL = "harnyx-storage-probe"
PROBE_IMAGE_BYTES = 128 * 1024**2
PROBE_LABELS = {"ai.harnyx.storage-probe": "true"}


def _details(exc: Exception) -> str:
    # No frame locals, environment dump or Docker inspect payload is collected.
    return _redact("".join(traceback.format_exception(exc)))[-12000:]


def _redact(text: str) -> str:
    values: list[str] = [value for value in os.environ.values() if len(value) >= 8]
    for value in sorted(values, key=lambda value: len(value), reverse=True):
        text = text.replace(value, "<redacted>")
    return text


class StorageProbeWorker:
    def __init__(
        self,
        create_manager: Callable[[], ArtifactStorageSandboxManager],
        report: Callable[[StorageProbeReport], None],
        *,
        iops: int,
        bytes_per_second: int,
    ) -> None:
        self._create_manager = create_manager
        self._report = report
        self._iops = iops
        self._bytes_per_second = bytes_per_second
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="storage-probe", daemon=True)
        try:
            self._thread.start()
        except RuntimeError:
            self._thread = None
            logger.warning("storage probe thread unavailable; validator continues", exc_info=True)

    def stop(self) -> None:
        # Do not spend the service's shutdown grace waiting for optional Docker
        # diagnostics. In-flight cleanup can finish while execution workers stop;
        # interrupted resources remain owned for recovery on the next startup.
        self._stop.set()

    def _probe(self) -> StorageProbeReport:
        manager = None
        stage = "validator runtime identity"
        error = ""
        cleanup_error = ""
        cleanup = "not_needed"
        try:
            manager = self._create_manager()
            manager.recover(sandbox_labels=PROBE_LABELS, sandbox_name_prefix=f"{PROBE_POOL}-consumer-")
            stage = "complete"
        except Exception as exc:
            stage = manager.stage if manager is not None else stage
            error = _details(exc)
        finally:
            if manager is not None:
                try:
                    manager.finish_probe()
                    cleanup = "succeeded"
                except Exception as exc:
                    cleanup = "unconfirmed"
                    cleanup_error = _details(exc)
        return StorageProbeReport(
            probe_id=uuid4(), checked_at=datetime.now(UTC),
            status="failed" if error or cleanup_error else "succeeded",
            stage=stage, cleanup=cleanup,
            image_bytes=PROBE_IMAGE_BYTES, iops=self._iops, bytes_per_second=self._bytes_per_second,
            details=error, cleanup_details=cleanup_error,
        )

    def _run(self) -> None:
        if self._stop.is_set():
            return
        try:
            result = self._probe()
            logger.info("validator storage probe", extra={"data": result.model_dump(mode="json")})
            # Retry delivery only, never rerun allocation. Normal registration is independent.
            for attempt in range(3):
                if self._stop.is_set():
                    return
                try:
                    self._report(result)
                    return
                except Exception:
                    logger.warning("storage probe report delivery failed", extra={"attempt": attempt + 1})
                if self._stop.wait(30):
                    return
        except Exception:
            # A diagnostics bug must not become a validator startup dependency.
            logger.warning("storage probe worker failed; validator continues", exc_info=True)
