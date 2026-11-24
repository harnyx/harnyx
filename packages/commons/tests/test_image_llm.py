"""Images reach provider encoding; signed capabilities must not reach output or billing."""

from uuid import uuid4

import pytest

from harnyx_commons.errors import ToolProviderError
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.llm.providers.openai_chat_codec import OpenAiChatRequestParts
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.tools.executor import ToolInvocationContext
from harnyx_commons.tools.image_redaction import ImageUrlRedactor
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata
from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker
from harnyx_miner_sdk.llm import LlmInputImageData, LlmInputImagePart, LlmInputTextPart, LlmMessage
from harnyx_miner_sdk.tools.llm_chat_stream_models import LlmChatChunk, TextUpdate, ToolUpdate

pytestmark = pytest.mark.anyio("asyncio")
URL = "https://private.example/image.png?X-Amz-Signature=do-not-expose"


@pytest.mark.parametrize("provider", ["chutes", "openrouter", "ai_gateway", "saygm", "cloudflare"])
def test_chat_encoder_preserves_image_and_text_order(provider):
    request = LlmRequest(
        provider=provider,
        model="vision",
        temperature=None,
        max_output_tokens=None,
        messages=(
            LlmMessage(
                role="user",
                content=(
                    LlmInputTextPart("before"),
                    LlmInputImagePart(LlmInputImageData(url=URL, mime_type="image/png")),
                    LlmInputTextPart("after"),
                ),
            ),
        ),
    )
    payload = OpenAiChatRequestParts.from_request(
        request,
        provider_name=provider,
        image_error_message="images unavailable",
        tool_mix_error_message="mixed tools",
        tool_count_error_message="tool count",
    ).model_dump()
    assert payload["messages"][0]["content"] == [
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": URL}},
        {"type": "text", "text": "after"},
    ]


@pytest.mark.parametrize("field", ["text", "reasoning", "arguments"])
async def test_redaction_spans_every_chunk_boundary(field):
    for split in range(1, len(URL)):
        chunks = []

        async def sink(chunk, received=chunks):
            received.append(chunk)

        emit = ImageUrlRedactor([URL]).wrap(sink)
        for text, final in [("hello " + URL[:split], False), (URL[split:] + " world", True)]:
            update = TextUpdate(operation="append", value=text)
            kwargs = (
                {field: update} if field != "arguments" else {"tool_calls": [ToolUpdate(index=0, arguments=update)]}
            )
            await emit(LlmChatChunk(choice_index=0, finish_reason="stop" if final else None, **kwargs))
        result = "".join(
            (chunk.tool_calls[0].arguments if field == "arguments" else getattr(chunk, field)).value for chunk in chunks
        )
        assert result == "hello [image] world"


async def test_replacement_discards_pending_suffix():
    chunks = []

    async def sink(chunk):
        chunks.append(chunk)

    emit = ImageUrlRedactor([URL]).wrap(sink)
    await emit(LlmChatChunk(choice_index=0, text=TextUpdate(operation="append", value=URL[:12])))
    await emit(LlmChatChunk(choice_index=0, text=TextUpdate(operation="replace", value="safe"), finish_reason="stop"))
    assert chunks[-1].text == TextUpdate(operation="replace", value="safe")


@pytest.mark.parametrize("failure", [False, True])
async def test_runtime_resolves_once_sanitizes_stream_result_and_billing(failure):
    image_id = uuid4()
    resolutions, chunks, requests = [], [], []

    async def resolve(ids, context):
        resolutions.append(tuple(ids))
        return {image_id: LlmInputImagePart(LlmInputImageData(url=URL, mime_type="image/png"))}

    async def sink(chunk):
        chunks.append(chunk)

    class Provider:
        async def invoke(self, request):
            requests.append(request)
            assert not request.include_payloads_in_observability
            for value, finish in [(URL[:20], None), (URL[20:], "stop")]:
                await request.on_output(
                    LlmChatChunk(
                        choice_index=0,
                        text=TextUpdate(operation="append", value=value),
                        finish_reason=finish,
                    )
                )
            if failure:
                raise ToolProviderError(
                    URL,
                    provider="chutes",
                    http_status=400,
                    billing=ProviderBillingMetadata(
                        "chutes",
                        "response_body",
                        actual_cost_usd=0.01,
                        provider_usage={"detail": URL},
                    ),
                )
            return LlmResponse(
                id="reply",
                choices=(
                    LlmChoice(
                        index=0,
                        message=LlmChoiceMessage(
                            role="assistant",
                            content=(LlmMessageContentPart(type="text", text=URL),),
                        ),
                    ),
                ),
                usage=LlmUsage(prompt_tokens=10, completion_tokens=2),
                metadata={
                    "actual_cost_usd": 0.01,
                    "actual_cost_provider": "chutes",
                    "actual_cost_evidence": {"detail": URL},
                    "_provider_evidence": {"response": URL},
                },
            )

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        llm_provider=Provider(),
        llm_provider_name="chutes",
        llm_provider_resolver=lambda *_: Provider(),
        image_resolver=resolve,
    )
    kwargs = {
        "provider": "chutes",
        "model": "deepseek-ai/DeepSeek-V3.2-TEE",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image_id": str(image_id)},
                    {"type": "image", "image_id": str(image_id)},
                ],
            }
        ],
    }
    context = ToolInvocationContext(receipt_id="r", session_id=uuid4(), active_attempt=1, uid=1, on_output=sink)
    if failure:
        with pytest.raises(ToolProviderError) as caught:
            await invoker.invoke("llm_chat", args=(), kwargs=kwargs, context=context)
        assert caught.value.billing.actual_cost_usd == 0.01
        assert URL not in repr(caught.value.billing) and URL not in str(caught.value)
    else:
        result = await invoker.invoke("llm_chat", args=(), kwargs=kwargs, context=context)
        assert result.actual_cost_usd == 0.01 and URL not in repr(result)
    assert resolutions == [(image_id,)]
    assert "".join(chunk.text.value for chunk in chunks) == "[image]"
    assert requests[0].messages[0].content[0].data.url == URL
    assert kwargs["messages"][0]["content"][0]["image_id"] == str(image_id)


@pytest.mark.parametrize("provider_name", ["openrouter", "saygm", "cloudflare"])
async def test_real_provider_builders_encode_images(provider_name):
    from pydantic import SecretStr

    from harnyx_commons.llm.providers.cloudflare import CloudflareLlmProvider
    from harnyx_commons.llm.providers.openrouter import build_openrouter_chat_provider
    from harnyx_commons.llm.providers.saygm import SayGmLlmProvider
    from harnyx_miner_sdk.tools.cloudflare import CloudflareCredential

    if provider_name == "openrouter":
        provider, client = build_openrouter_chat_provider("test-key")
    elif provider_name == "saygm":
        provider = SayGmLlmProvider(api_key=SecretStr("test-key"))
        client = None
    else:
        provider = CloudflareLlmProvider(credential=CloudflareCredential.model_validate(
            {"account_id": "a" * 32, "api_token": "test-key"}
        ))
        client = None
    try:
        request = LlmRequest(provider=provider_name, model="vision", temperature=None, max_output_tokens=None,
                             messages=(LlmMessage(role="user", content=(
                                 LlmInputImagePart(LlmInputImageData(url=URL, mime_type="image/png")),
                             )),))
        payload = provider._build_request(request).model_dump()
        assert payload["messages"][0]["content"] == [{"type": "image_url", "image_url": {"url": URL}}]
    finally:
        await provider.aclose()
        if client is not None:
            await client.aclose()


async def test_reasoning_detail_fragments_redacted_without_changing_replay_shape():
    details = [{"type": "reasoning.text", "index": 0, "text": "before " + URL[:25]},
               {"type": "reasoning.text", "index": 0, "text": URL[25:] + " after"}]
    chunks = []

    async def sink(chunk):
        chunks.append(chunk)

    redactor = ImageUrlRedactor([URL])
    emit = redactor.wrap(sink)
    for detail in details:
        await emit(LlmChatChunk(choice_index=0, reasoning_details=[detail]))
    assert all(chunk.reasoning_details is None for chunk in chunks)
    await redactor.finish()
    cleaned = [detail for chunk in chunks for detail in chunk.reasoning_details or []]
    assert "".join(detail["text"] for detail in cleaned) == "before [image] after"
    assert [(d["type"], d["index"]) for d in cleaned] == [("reasoning.text", 0)] * 2
    assert redactor.value({"reasoning_details": details})["reasoning_details"] == cleaned
    assert details[0]["text"].endswith(URL[:25])  # Provider evidence is not mutated in place.


async def test_finish_flushes_harmless_suffix_without_provider_finish_chunk():
    chunks = []

    async def sink(chunk):
        chunks.append(chunk)

    redactor = ImageUrlRedactor([URL])
    await redactor.wrap(sink)(LlmChatChunk(choice_index=0, text=TextUpdate(operation="append", value="say h")))
    await redactor.finish()
    assert "".join(chunk.text.value for chunk in chunks) == "say h"
