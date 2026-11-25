"""Check saygm liveness with its cheapest listed chat model."""

from dataclasses import replace

import httpx
import pytest

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.provider import LlmProviderError
from harnyx_commons.llm.provider_factory import build_miner_paid_llm_provider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest, LlmThinkingConfig

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


async def test_saygm_streaming_live() -> None:
    settings = LlmSettings()
    provider = build_miner_paid_llm_provider(
        provider="saygm",
        api_key=settings.saygm_api_key,
        llm_settings=settings,
    )
    chunks = []

    async def capture_output(chunk):
        chunks.append(chunk)

    request = LlmRequest(
        on_output=capture_output,
        provider="saygm",
        model="gpt-oss-20b",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("Say hello."),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
        extra={"X-GM-Routing": "cheapest"},
        thinking=LlmThinkingConfig(enabled=False),
        timeout=60.0,
    )
    try:
        response = await provider.invoke(request)
        restricted = replace(request, extra={"X-GM-Providers": "harnyx-unlisted-provider"})
        with pytest.raises(LlmProviderError) as rejected:
            await provider.invoke(restricted)
        cause = rejected.value.__cause__
        assert isinstance(cause, httpx.HTTPStatusError)
        assert cause.response.status_code == 400
        assert cause.response.json()["error"]["code"] == "no_allowed_provider"
    finally:
        await provider.aclose()

    assert chunks
    assert response.usage.total_tokens > 0
    assert response.raw_text.strip()
