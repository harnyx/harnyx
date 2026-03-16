"""Filesystem-backed tracking for weight submission backoff."""

from __future__ import annotations

import os
from pathlib import Path


class FileBackoff:
    """Persist and query the last block used for weight submission."""

    def __init__(self, path: Path) -> None:
        self._path = path

    # ------------------------------------------------------------------
    # public API

    def read_last_block(self) -> int | None:
        """Return the canonical last submission block when present."""

        return self._read_block(self._path)

    def should_skip(self, now_block: int, min_blocks: int) -> tuple[bool, int]:
        """Return whether to skip and the remaining blocks in the backoff."""

        last = self.read_last_block()
        if last is None:
            return False, 0
        elapsed = max(0, now_block - last)
        remaining = max(0, min_blocks - elapsed)
        return remaining > 0, remaining

    def write_last_block(self, block: int) -> None:
        """Persist the last observed submission block."""

        self._write_block(self._path, block)

    # ------------------------------------------------------------------
    # helpers

    def _read_block(self, path: Path) -> int | None:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"backoff file {path} is empty")
        try:
            return int(text, 10)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"backoff file {path} contains invalid block value: {text!r}") from exc

    def _write_block(self, path: Path, block: int) -> None:
        if block < 0:
            raise ValueError("block must be non-negative")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(f"{path}.tmp")
        tmp_path.write_text(str(block), encoding="utf-8")
        os.replace(tmp_path, path)


__all__ = ["FileBackoff"]
