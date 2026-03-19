from __future__ import annotations

import pytest

from harnyx_commons.tools import local_dev_host

pytestmark = pytest.mark.anyio("asyncio")


class _StubSettings:
    desearch_api_key_value = "desearch-key"
    chutes_api_key_value = "chutes-key"
    desearch_max_concurrent = 1
    chutes_max_concurrent = 1


class _StubSearchClient:
    def __init__(self, **_: object) -> None:
        pass

    async def aclose(self) -> None:
        return None


class _StubLlmProvider:
    def __init__(self, **_: object) -> None:
        pass

    async def aclose(self) -> None:
        return None


async def test_create_local_tool_host_tooling_info_matches_miner_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_dev_host, "LlmSettings", lambda: _StubSettings())
    monkeypatch.setattr(local_dev_host, "DeSearchClient", _StubSearchClient)
    monkeypatch.setattr(local_dev_host, "ChutesLlmProvider", _StubLlmProvider)

    host = local_dev_host.create_local_tool_host()
    try:
        payload = await host.invoke("tooling_info")
    finally:
        await host.aclose()

    assert "search_repo" not in payload["response"]["tool_names"]
    assert "get_repo_file" not in payload["response"]["tool_names"]
    assert "search_repo" not in payload["response"]["pricing"]
    assert "get_repo_file" not in payload["response"]["pricing"]
