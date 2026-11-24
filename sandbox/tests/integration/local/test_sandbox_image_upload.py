"""Binary uploads share the worker connection without corrupting parallel calls."""

import asyncio
import hashlib
from uuid import uuid4

import pytest
from harnyx_sandbox.sandbox.harness import EntrypointRequest, SandboxHarness, _EntrypointContext

from harnyx_miner_sdk.tools.image_models import ImageReference

pytestmark = [pytest.mark.integration, pytest.mark.local_integration, pytest.mark.anyio("asyncio")]


@pytest.mark.parametrize("cancel", [False, True])
async def test_binary_upload_parallel_calls_and_cancellation_keep_framing(tmp_path, cancel):
    path = tmp_path / "agent.py"
    path.write_text("""
import asyncio
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.api import image_upload
from harnyx_miner_sdk._internal.tool_invoker import _current_tool_invoker
@entrypoint("probe")
async def probe(request: dict[str, object]) -> dict[str, object]:
    upload = asyncio.create_task(image_upload(bytes(range(256)) * 4096))
    await asyncio.sleep(.02)
    if request["cancel"]:
        upload.cancel()
        try:
            await upload
        except asyncio.CancelledError:
            pass
    else:
        await upload
    results = await asyncio.gather(*(_current_tool_invoker().invoke("echo", kwargs={"i": i}) for i in range(8)))
    return {"results": results}
""")
    received = []

    class Proxy:
        async def upload_image(self, data, request):
            await asyncio.sleep(0.05)
            assert request.size_bytes == len(data) == 1024 * 1024
            assert request.sha256 == hashlib.sha256(data).hexdigest()
            received.append(data)
            return ImageReference(image_id=uuid4(), mime_type="image/png", size_bytes=len(data))

        async def invoke(self, method, *, args, kwargs):
            return kwargs["i"]

        async def aclose(self):
            pass

    runtime = SandboxHarness(artifact_path=str(path), tool_factory=lambda *_: Proxy())
    try:
        result = await runtime.invoke(
            "probe",
            EntrypointRequest(
                payload={"cancel": cancel},
                context=_EntrypointContext.model_validate({"time_budget": {"limit_seconds": 10}}),
            ),
        )
        assert result == {"results": list(range(8))}
        assert len(received) == 1
    finally:
        await runtime.close()


@pytest.mark.parametrize("failure_write", [1, 2, 3])
async def test_partial_image_write_closes_connection(failure_write):
    from harnyx_sandbox.sandbox.compiler import ConnectionInvoker

    from harnyx_miner_sdk.tools.image_models import ImageUploadRequest

    class Writer:
        writes = 0
        closed = False

        def write(self, data):
            self.writes += 1
            if self.writes == failure_write:
                raise ConnectionResetError("partial upload")

        async def drain(self):
            pass

        def close(self):
            self.closed = True

    writer = Writer()
    invoker = ConnectionInvoker(asyncio.StreamReader(), writer)
    with pytest.raises(ConnectionResetError):
        await invoker.upload_image(b"image", ImageUploadRequest.from_bytes(b"image", uuid4()))
    assert writer.closed and not invoker.calls and not invoker.uploads


async def test_cancel_during_drain_retains_slot_and_complete_frame():
    from harnyx_sandbox.sandbox.compiler import ConnectionInvoker
    from harnyx_sandbox.sandbox.worker_protocol import decode_worker_message, read_frame, read_image_payload

    from harnyx_miner_sdk.tools.image_models import ImageUploadRequest

    draining, release = asyncio.Event(), asyncio.Event()
    reader = asyncio.StreamReader()

    class Writer:
        def write(self, data):
            reader.feed_data(data)

        async def drain(self):
            draining.set()
            await release.wait()

        def close(self):
            reader.feed_eof()

    invoker = ConnectionInvoker(asyncio.StreamReader(), Writer())
    data = b"raw image" * 100
    task = asyncio.create_task(invoker.upload_image(data, ImageUploadRequest.from_bytes(data, uuid4())))
    await draining.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert invoker.upload_slot.locked() and invoker.write_lock.locked()
    header = decode_worker_message(await read_frame(reader))
    assert await read_image_payload(reader, header.request.size_bytes) == data
    invoker.calls[header.call_id].set_result(ImageReference(
        image_id=uuid4(), mime_type="image/png", size_bytes=len(data),
    ).model_dump(mode="json"))
    retained = tuple(invoker.uploads)
    release.set()
    await asyncio.gather(*retained)
    assert not invoker.upload_slot.locked() and not invoker.write_lock.locked()


@pytest.mark.parametrize("frame", [b"J", b"I" + (20 * 1024 * 1024 + 1).to_bytes(8, "big"),
                                   b"I" + (4).to_bytes(8, "big")])
async def test_bad_image_frame_rejected_before_reading_payload(frame):
    from harnyx_sandbox.sandbox.worker_protocol import WorkerMessageViolationError, read_image_payload

    reader = asyncio.StreamReader()
    reader.feed_data(frame)  # No payload or EOF: an attempted body read would hang.
    async with asyncio.timeout(1):
        with pytest.raises(WorkerMessageViolationError):
            await read_image_payload(reader, 5)


async def test_active_llm_stream_progresses_while_upload_waits(tmp_path):
    from harnyx_miner_sdk.tools.llm_chat_stream_models import LlmChatChunk, LlmChatOutputFrame, TextUpdate

    path = tmp_path / "agent.py"
    path.write_text('''
import asyncio
from harnyx_miner_sdk.api import image_upload, llm_chat
from harnyx_miner_sdk.decorators import entrypoint
@entrypoint("probe")
async def probe(request: dict[str, object]) -> dict[str, object]:
    stream = await llm_chat(provider="chutes", model="test", stream=True,
        messages=[{"role":"user","content":"hello"}])
    upload = asyncio.create_task(image_upload(b"pixels"))
    chunk = await anext(stream)
    await stream.aclose()
    await upload
    return {"text": chunk.text.value}
''')
    uploaded, closed = asyncio.Event(), asyncio.Event()

    class Frames:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await uploaded.wait()
            return LlmChatOutputFrame(chunk=LlmChatChunk(choice_index=0,
                text=TextUpdate(operation="append", value="still streaming")))

        async def aclose(self):
            closed.set()

    class Proxy:
        async def invoke_llm_chat_stream(self, *, kwargs):
            return Frames()

        async def upload_image(self, data, request):
            uploaded.set()
            await closed.wait()
            return ImageReference(image_id=uuid4(), mime_type="image/png", size_bytes=len(data))

        async def aclose(self):
            pass

    runtime = SandboxHarness(artifact_path=str(path), tool_factory=lambda *_: Proxy())
    try:
        async with asyncio.timeout(5):
            result = await runtime.invoke(
                "probe", EntrypointRequest(payload={}, context=_EntrypointContext.model_validate(
                    {"time_budget": {"limit_seconds": 4}}
                ))
            )
        assert result == {"text": "still streaming"} and closed.is_set()
    finally:
        await runtime.close()
