from __future__ import annotations

from pathlib import Path

import pytest

from caster_validator.infrastructure.state.backoff_file import FileBackoff


def test_read_missing_returns_none(tmp_path: Path) -> None:
    target = tmp_path / "backoff.txt"
    backoff = FileBackoff(target)

    assert backoff.read_last_block() is None


def test_write_and_read_last_block(tmp_path: Path) -> None:
    target = tmp_path / "backoff.txt"
    backoff = FileBackoff(target)

    backoff.write_last_block(1_234)
    assert backoff.read_last_block() == 1_234


def test_should_skip_respects_min_blocks(tmp_path: Path) -> None:
    target = tmp_path / "backoff.txt"
    backoff = FileBackoff(target)
    backoff.write_last_block(1_000)

    skip, remaining = backoff.should_skip(1_050, 100)
    assert skip is True
    assert remaining == 50

    skip, remaining = backoff.should_skip(1_200, 100)
    assert skip is False
    assert remaining == 0


def test_invalid_file_contents_raise(tmp_path: Path) -> None:
    target = tmp_path / "backoff.txt"
    target.write_text("not-a-number", encoding="utf-8")
    backoff = FileBackoff(target)

    with pytest.raises(ValueError, match="invalid block value"):
        backoff.read_last_block()


def test_write_negative_block_raises(tmp_path: Path) -> None:
    target = tmp_path / "backoff.txt"
    backoff = FileBackoff(target)

    with pytest.raises(ValueError, match="non-negative"):
        backoff.write_last_block(-1)
