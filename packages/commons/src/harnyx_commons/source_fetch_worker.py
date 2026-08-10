"""Killable complete-source HTTP worker with a bounded binary stdout protocol."""

from __future__ import annotations

import json
import os
from typing import cast


def main() -> None:
    from harnyx_commons.domain_tweak_generation.source_fetch import (
        DocumentKind,
        SourceFetchError,
        _fetch_complete_body,
    )

    try:
        import sys

        fetched = _fetch_complete_body(sys.argv[1], cast(DocumentKind, sys.argv[2]))
        header = json.dumps(
            {
                "final_url": fetched.final_url,
                "media_type": fetched.media_type,
                "content_encoding": fetched.content_encoding,
                "body_length": len(fetched.body),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        _write(b"O" + len(header).to_bytes(4, "big") + header + fetched.body)
    except SourceFetchError as exc:
        _write_error(exc.code, str(exc))
    except BaseException as exc:
        _write_error("source_unavailable", f"source fetch failed: {type(exc).__name__}: {exc}")


def _write(payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        offset += os.write(1, payload[offset:])


def _write_error(code: str, message: str) -> None:
    try:
        payload = json.dumps({"code": code[:64], "message": message[:2_000]}).encode("utf-8")
    except BaseException:
        payload = b'{"code":"source_unavailable","message":"source fetch failed"}'
    _write(b"E" + payload)


if __name__ == "__main__":
    main()
