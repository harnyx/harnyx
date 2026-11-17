"""Temporary query directories and bounded, best-effort filesystem cleanup."""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore
from types import TracebackType

QUERY_WORKSPACE_ROOT = Path("/sandbox/queries")
logger = logging.getLogger(__name__)


class QueryWorkspaces:
    """Own directory cleanup independently of worker termination confirmation."""

    def __init__(self, root: Path = QUERY_WORKSPACE_ROOT) -> None:
        self.root = root
        self._capacity = BoundedSemaphore(10)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="query-files")

    def prepare(self) -> str:
        # Never create the root: a missing mount must fail, not fall back to rootfs.
        path = tempfile.mkdtemp(prefix="query-", dir=self.root)
        try:
            with tempfile.TemporaryFile(dir=path):
                pass
        except OSError:
            self.remove_later(path)
            raise
        return path

    def remove_later(self, path: str) -> None:
        if not self._capacity.acquire(blocking=False):
            logger.warning("query file cleanup deferred to volume removal", extra={"workspace": path})
            return
        try:
            future = self._executor.submit(self._remove, path)
        except RuntimeError:
            self._capacity.release()
            logger.warning("query file cleanup unavailable; retaining files", extra={"workspace": path})
            return
        future.add_done_callback(self._finished)

    def _finished(self, future: Future[None]) -> None:
        self._capacity.release()
        if not future.cancelled() and future.exception() is not None:
            logger.warning("query file cleanup failed", exc_info=future.exception())

    @staticmethod
    def _remove(path: str) -> None:
        def onerror(
            function: Callable[..., object],
            failed_path: str,
            error: tuple[type[BaseException], BaseException, TracebackType | None],
        ) -> None:
            if not isinstance(error[1], FileNotFoundError):
                logger.warning(
                    "query file removal failed", extra={"workspace": path, "path": failed_path}, exc_info=error,
                )

        # Python's fd-based rmtree does not traverse substituted symlinks on Linux.
        shutil.rmtree(path, onerror=onerror)

    def close(self) -> None:
        # Container removal owns leftovers; do not hold shutdown for pending files.
        self._executor.shutdown(wait=False, cancel_futures=True)
