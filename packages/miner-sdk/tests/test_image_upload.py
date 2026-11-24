"""Upload identity and the public image-ID contract must survive the SDK/host hop."""

import hashlib
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from harnyx_miner_sdk._internal.tool_invoker import bind_tool_invoker
from harnyx_miner_sdk.api import image_upload
from harnyx_miner_sdk.llm import LlmInputImageData, LlmInputImagePart
from harnyx_miner_sdk.tools.image_models import IMAGE_UPLOAD_HEADER, ImageUploadRequest
from harnyx_miner_sdk.tools.llm_chat_models import LlmChatRequest
from harnyx_miner_sdk.tools.proxy import ToolProxy

pytestmark = pytest.mark.anyio("asyncio")


async def test_upload_sends_exact_bytes_and_reuses_explicit_identity():
    data, upload_id, image_id, session_id = bytes(range(256)), uuid4(), uuid4(), uuid4()
    requests = []
    token = str(uuid4())

    def handler(request):
        assert request.url.path == "/v1/tools/images"
        assert request.headers["x-session-id"] == str(session_id)
        assert request.headers["x-platform-token"] == token
        metadata = ImageUploadRequest.model_validate_json(request.headers[IMAGE_UPLOAD_HEADER])
        assert metadata.upload_id == upload_id
        assert metadata.size_bytes == len(data) and metadata.sha256 == hashlib.sha256(data).hexdigest()
        assert request.content == data
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "image_id": str(image_id),
                "mime_type": "image/png",
                "size_bytes": len(data),
            },
        )

    async with httpx.AsyncClient(base_url="http://host", transport=httpx.MockTransport(handler)) as client:
        proxy = ToolProxy("http://host", token=token, session_id=str(session_id), client=client)
        with bind_tool_invoker(proxy):
            first = await image_upload(data, upload_id)
            second = await image_upload(data, upload_id)
    assert first == second and first.image_id == image_id
    assert len(requests) == 2


@pytest.mark.parametrize("data", [b"", bytearray(b"abc"), b"x" * (20 * 1024 * 1024 + 1)])
async def test_invalid_bytes_rejected_before_host_call(data):
    with pytest.raises((ValueError, TypeError)):
        await image_upload(data)


def test_user_parts_preserve_order_and_require_scoped_resolution():
    image_id = uuid4()
    payload = {
        "provider": "openrouter",
        "model": "openai/gpt-oss-20b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image", "image_id": str(image_id)},
                    {"type": "text", "text": "after"},
                ],
            }
        ],
    }
    request = LlmChatRequest.model_validate(payload)
    with pytest.raises(ValueError):
        request.to_tool_request()
    image = LlmInputImagePart(LlmInputImageData(url="https://private/image", mime_type="image/png"))
    content = request.to_tool_request({image_id: image}).messages[0].content
    assert content[0].text == "before" and content[1] == image and content[2].text == "after"
    assert request.model_dump(mode="json")["messages"] == payload["messages"]


@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "image_url", "image_url": {"url": "https://elsewhere"}}],
        [{"type": "image", "image_id": str(uuid4()), "url": "https://elsewhere"}],
    ],
)
def test_external_urls_and_empty_content_rejected(content):
    with pytest.raises(ValidationError):
        LlmChatRequest.model_validate(
            {
                "provider": "openrouter",
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": content}],
            }
        )
