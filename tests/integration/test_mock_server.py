import asyncio
import json
import socket
from collections.abc import AsyncIterator

import pytest
import uvicorn
from httpx import ASGITransport, AsyncClient

from inference_lab.benchmark import mock_server

pytestmark = pytest.mark.integration

PAYLOAD = {
    "model": "mock-model",
    "messages": [{"role": "user", "content": "synthetic input"}],
    "stream": True,
}


@pytest.mark.parametrize("change", [{"stream": False}, {"messages": []}, {"model": "real-model"}])
async def test_mock_rejects_unsupported_requests(change: dict[str, object]) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=mock_server.app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/chat/completions", json=PAYLOAD | change)
    assert response.status_code == 422


async def test_mock_streams_before_generation_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    release = asyncio.Event()
    original_stream = mock_server._stream

    async def gated_stream() -> AsyncIterator[str]:
        async for event in original_stream():
            yield event
            await release.wait()

    monkeypatch.setattr(mock_server, "_stream", gated_stream)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(mock_server.app, log_level="error", lifespan="off"))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if task.done():
                        await task
                    await asyncio.sleep(0.01)
            async with AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=5) as client:
                async with client.stream("POST", "/v1/chat/completions", json=PAYLOAD) as response:
                    assert response.status_code == 200
                    assert response.headers["content-type"].startswith("text/event-stream")
                    lines = response.aiter_lines()
                    first = json.loads((await anext(lines)).removeprefix("data: "))
                    assert first["choices"][0]["delta"] == {"role": "assistant"}
                    assert not release.is_set()
                    release.set()
                    events = [line[6:] async for line in lines if line.startswith("data: ")]
                    assert events[-1] == "[DONE]"
                    chunks = [json.loads(event) for event in events[:-1]]
                    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
                    assert (
                        "".join(
                            choice["delta"].get("content", "")
                            for chunk in chunks
                            for choice in chunk["choices"]
                        )
                        == "Hello world"
                    )
                    assert chunks[-1]["usage"]["completion_tokens"] == 5
                    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
        finally:
            release.set()
            server.should_exit = True
            await asyncio.wait_for(task, timeout=5)
