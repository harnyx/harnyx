from __future__ import annotations

import asyncio
import json
import resource
import subprocess
import sys
from pathlib import Path

import pytest

from harnyx_commons.domain_tweak_generation import SourceDocument, SourceWorkspace

pytestmark = [pytest.mark.integration, pytest.mark.expensive]

_MAX_SOURCE_BYTES = 5_000_000
_MAX_CONCURRENT_SCAN_RSS_GROWTH_BYTES = 64 * 1024 * 1024


async def _measure_concurrent_scan_rss() -> dict[str, object]:
    workspace = SourceWorkspace()
    contents = (
        (("needle filler value\n" * 250_000) + "needle target A")[: _MAX_SOURCE_BYTES - 1],
        (("other filler value\n" * 260_000) + "needle target B")[: _MAX_SOURCE_BYTES - 1],
    )
    sources = tuple(
        workspace.store(
            SourceDocument(
                requested_url=f"https://example.com/report-{index}",
                final_url=f"https://example.com/report-{index}",
                media_type="text/plain",
                content=content,
                fetched_bytes=len(content),
            )
        )
        for index, content in enumerate(contents)
    )
    peak_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    results = await asyncio.gather(
        *(workspace._run_similarity_result(source, "needle target", 3) for source in sources)
    )
    peak_after_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "rss_growth_bytes": max(0, peak_after_kib - peak_before_kib) * 1024,
        "chunk_counts": [len(result["chunks"]) for result in results],
    }


def test_concurrent_near_limit_similarity_scans_stay_within_rss_budget() -> None:
    """Future failure: two legal concurrent scans must not materialize full-token corpora and exhaust workers."""
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and repository-owned test path
        [sys.executable, str(Path(__file__).resolve()), "--measure-rss"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    measurement = json.loads(completed.stdout)

    assert measurement["chunk_counts"] == [3, 3]
    assert measurement["rss_growth_bytes"] < _MAX_CONCURRENT_SCAN_RSS_GROWTH_BYTES


if __name__ == "__main__":
    if sys.argv[1:] != ["--measure-rss"]:
        raise SystemExit("expected --measure-rss")
    print(json.dumps(asyncio.run(_measure_concurrent_scan_rss())))
