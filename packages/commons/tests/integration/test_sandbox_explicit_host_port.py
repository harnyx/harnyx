from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.integration

_AGENT_MODULE = "commons.tests.integration.sandbox.seccomp_agent"


def test_docker_sandbox_manager_starts_sandbox_with_explicit_host_port(sandbox_launcher) -> None:
    deployment = sandbox_launcher(agent_module=_AGENT_MODULE)

    assert deployment.base_url.startswith("http://127.0.0.1:")
    response = httpx.get(f"{deployment.base_url}/healthz", timeout=2.0)
    assert response.status_code == 200


def test_docker_sandbox_manager_starts_sandbox_with_dynamic_host_port(sandbox_launcher) -> None:
    deployment = sandbox_launcher(agent_module=_AGENT_MODULE, host_port=0)

    assert deployment.base_url.startswith("http://127.0.0.1:")
    response = httpx.get(f"{deployment.base_url}/healthz", timeout=2.0)
    assert response.status_code == 200


@pytest.mark.anyio("asyncio")
async def test_fresh_sandbox_delivers_stream_before_host_completion(sandbox_launcher) -> None:
    """Exercise the packaged SDK, private worker IPC and HTTP stream through the built image."""
    import asyncio
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from uuid import uuid4

    observed = threading.Event()
    calls = []
    token, session_id = uuid4().hex, uuid4()
    budget = {"session_budget_usd": 1.0, "session_hard_limit_usd": 1.0,
              "session_used_budget_usd": 0.0, "session_remaining_budget_usd": 1.0}

    def result(response):
        return {"receipt_id": "settled-image", "response": response, "results": [],
                "result_policy": "log_only", "budget": budget}

    class Host(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
            calls.append((payload, self.headers["x-platform-token"], self.headers["x-session-id"]))
            streaming = payload["tool"] == "llm_chat"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
            self.end_headers()
            if streaming:
                self.wfile.write(b'data: {"type":"output","chunk":{"choice_index":0,'
                                 b'"text":{"operation":"append","value":"early"}}}\n\n')
                self.wfile.flush()
                if not observed.wait(10):
                    return
                frame = {"type": "completed", "result": result({"id": "response-1", "choices": [], "usage": {}})}
                self.wfile.write(("data: " + json.dumps(frame) + "\n\n").encode())
            else:
                observed.set()
                self.wfile.write(json.dumps(result({"status": "ok", "echo": "observed"})).encode())
            self.wfile.flush()

    source = b'''
from harnyx_miner_sdk.api import llm_chat, test_tool
from harnyx_miner_sdk.decorators import entrypoint
@entrypoint("streaming")
async def streaming(request: dict[str, object]) -> dict[str, object]:
    stream = await llm_chat(provider="chutes", model="test", stream=True,
        messages=[{"role":"user", "content":"hello"}], timeout=5.0)
    async with stream:
        first = await anext(stream)
        await test_tool("observed")
        remaining = [chunk async for chunk in stream]
    return {"text": first.text.value, "remaining": len(remaining), "receipt": stream.result.receipt_id}
'''
    server = ThreadingHTTPServer(("0.0.0.0", 0), Host)  # noqa: S104 - container must reach the controlled test host
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        deployment = sandbox_launcher(agent_module=_AGENT_MODULE, source=source, tool_host_port=server.server_port)
        async with asyncio.timeout(15):
            response = await deployment.client.invoke(
                "streaming", payload={}, context={"time_budget": {"limit_seconds": 10.0}},
                token=token, session_id=session_id,
            )
        assert response == {"text": "early", "remaining": 0, "receipt": "settled-image"}
        assert [call[0]["tool"] for call in calls] == ["llm_chat", "test_tool"]
        assert calls[0][0]["kwargs"]["stream"] is True
        assert all(call[1:] == (token, str(session_id)) for call in calls)
    finally:
        observed.set()
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(timeout=2)
