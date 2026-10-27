"""Real loopback sockets: ASGITransport would buffer and hide stream/cancel bugs."""

import asyncio
import json
import socket
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from inference_lab.serving.client import StreamError, StreamingClient, readiness

pytestmark = pytest.mark.integration
PAYLOAD = {
    "model": "synthetic",
    "messages": [{"role": "user", "content": "fixture"}],
    "stream": True,
    "n": 1,
}


def event(content: str | None = None, finish: str | None = None) -> bytes:
    delta = {"role": "assistant"} if content is None else {"content": content}
    data = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return ("data: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode()


USAGE = (
    b'data: {"choices": [], "usage": {"prompt_tokens": 3, '
    b'"completion_tokens": 5, "total_tokens": 8}}\n\n'
)
DONE = b"data: [DONE]\n\n"


@asynccontextmanager
async def server(
    source: Callable[[], AsyncIterator[bytes]],
    *,
    status: int = 200,
    content_type: str = "text/event-stream",
    captured: list[dict] | None = None,
    metrics: Callable[[], str] | None = None,
) -> AsyncIterator[str]:
    app = FastAPI()

    if metrics is not None:

        @app.get("/metrics", response_class=PlainTextResponse)
        async def scheduler() -> str:
            return metrics()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def completion(request: Request) -> Response:
        if captured is not None:
            captured.append(await request.json())
        return StreamingResponse(source(), status_code=status, media_type=content_type)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        runner = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        task = asyncio.create_task(runner.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(5):
                while not runner.started:
                    if task.done():
                        await task
                    await asyncio.sleep(0.005)
            yield f"http://127.0.0.1:{port}/v1/chat/completions"
        finally:
            runner.should_exit = True
            await asyncio.wait_for(task, 5)


async def collect(client: StreamingClient, url: str, **kwargs: object) -> list:
    async with client.stream(url, PAYLOAD, **kwargs) as events:
        return [item async for item in events]


async def test_incremental_role_fragmented_utf8_multiline_and_usage() -> None:
    release = asyncio.Event()
    captured: list[dict] = []

    async def source() -> AsyncIterator[bytes]:
        yield b"\xef\xbb\xbf: heartbeat\r\n" + event().replace(b"\n", b"\r\n")
        await release.wait()
        # Split a multibyte code point and CRLF across socket writes.
        text = event("two tokens: é").replace(b"\n", b"\r\n")
        split = text.index(b"\xc3") + 1
        for part in (text[:split], text[split:-1], text[-1:]):
            yield part
            await asyncio.sleep(0.005)
        yield (
            b'event: message\rdata: {"choices": [\rdata: {"index": 0, '
            b'"delta": {"content": " tail"}, "finish_reason": "stop"}]}\r\r'
        )
        yield USAGE + DONE

    async with server(source, captured=captured) as url, StreamingClient() as client:
        async with client.stream(
            url, PAYLOAD | {"seed": 42, "stream_options": {"include_usage": True}}
        ) as events:
            first = await asyncio.wait_for(anext(events), 2)
            assert first.data["choices"][0]["delta"] == {"role": "assistant"}
            assert first.content == "" and not release.is_set()
            release.set()
            remaining = [item async for item in events]
    assert [item.content for item in remaining] == ["two tokens: é", " tail", "", ""]
    assert remaining[-1].done
    assert remaining[-2].data["usage"]["completion_tokens"] == 5
    assert sum(bool(item.content) for item in remaining) == 2  # Events are not tokens.
    assert remaining[0].received_ns >= first.received_ns
    assert captured[0]["seed"] == 42 and captured[0]["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize(
    ("wire", "code"),
    [
        (event("partial"), "premature_eof"),
        (event("partial", "stop") + USAGE, "premature_eof"),
        (event("partial", "stop") + DONE, "incomplete_stream"),
        (USAGE + DONE, "unexpected_usage"),
        (DONE, "incomplete_stream"),
        (b'data: {"error": {"message": "SECRET"}}\n\n', "upstream_error"),
        (b"event: error\ndata: SECRET\n\n", "upstream_error"),
        (b"data: invalid SECRET\n\n", "invalid_event"),
        (b"data: []\n\n", "invalid_event"),
        (b'data: {"choices": [{"index": 1, "delta": {}}]}\n\n', "invalid_choices"),
        (event("x", "tool_calls"), "unexpected_finish_reason"),
        (event("x", "stop") + event("extra"), "event_after_finish"),
        (event("x", "stop") + USAGE + USAGE, "unexpected_usage"),
        (
            event("x", "stop")
            + USAGE.replace(b'"completion_tokens": 5', b'"completion_tokens": true'),
            "invalid_usage",
        ),
        (b"data: " + b"x" * 1_048_577, "event_too_large"),
    ],
)
async def test_failures_after_http_200_are_not_success(wire: bytes, code: str) -> None:
    async def source() -> AsyncIterator[bytes]:
        yield wire

    async with server(source) as url, StreamingClient() as client:
        with pytest.raises(StreamError) as error:
            await collect(client, url)
    assert error.value.code == code
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize(
    ("status", "mime", "code"),
    [
        (429, "application/json", "http_error"),
        (307, "text/event-stream", "http_error"),
        (200, "application/json", "invalid_content_type"),
    ],
)
async def test_http_failures_and_no_body_leak_or_retry(status: int, mime: str, code: str) -> None:
    calls: list[dict] = []

    async def source() -> AsyncIterator[bytes]:
        yield b"SECRET UPSTREAM BODY"

    async with (
        server(source, status=status, content_type=mime, captured=calls) as url,
        StreamingClient() as client,
    ):
        with pytest.raises(StreamError) as error:
            await collect(client, url)
    assert error.value.code == code and len(calls) == 1
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize("mode", ["cancel", "break", "idle", "global"])
async def test_timeout_and_consumer_exit_close_upstream(mode: str) -> None:
    closed = asyncio.Event()
    received = asyncio.Event()

    async def source() -> AsyncIterator[bytes]:
        try:
            yield event()
            if mode == "global":
                while True:
                    await asyncio.sleep(0.02)
                    yield b": heartbeat\n\n"
            else:
                await asyncio.Event().wait()
        finally:
            closed.set()

    async with (
        server(source) as url,
        StreamingClient(
            total_timeout=0.12 if mode == "global" else 3,
            idle_timeout=0.06 if mode == "idle" else 1,
        ) as client,
    ):

        async def consume() -> None:
            async with client.stream(url, PAYLOAD) as events:
                async for _ in events:
                    received.set()
                    if mode == "break":
                        break

        task = asyncio.create_task(consume())
        await asyncio.wait_for(received.wait(), 2)
        if mode == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif mode == "break":
            await task
        else:
            with pytest.raises(StreamError) as error:
                await task
            assert error.value.code == ("idle_timeout" if mode == "idle" else "global_timeout")
        await asyncio.wait_for(closed.wait(), 2)


async def test_optional_usage_and_pool_survives_a_request_failure() -> None:
    calls = 0

    async def source() -> AsyncIterator[bytes]:
        nonlocal calls
        calls += 1
        if calls == 1:
            yield event("partial")
        else:
            yield event("ok", "stop") + DONE

    async with server(source) as url, StreamingClient() as client:
        with pytest.raises(StreamError, match="premature_eof"):
            await collect(client, url)
        result = await collect(client, url, require_usage=False)
        assert result[-1].done and result[0].data.get("usage") is None


async def test_readiness_requires_real_generation_not_just_health() -> None:
    async def bad_source() -> AsyncIterator[bytes]:
        yield event("partial")

    async with server(bad_source) as url:
        with pytest.raises(StreamError, match="premature_eof"):
            await readiness(url, PAYLOAD, startup_timeout=2)

    async def good_source() -> AsyncIterator[bytes]:
        yield event("synthetic", "length") + USAGE + DONE

    async with server(good_source) as url:
        result = await readiness(url, PAYLOAD, startup_timeout=2)
        assert result == {
            "status": "VALIDATED",
            "content_events": 1,
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            "finish_reason": "length",
        }


async def test_network_error_is_safe() -> None:
    async def fail(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("SECRET endpoint and credentials")

    async with StreamingClient(transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(StreamError, match="^network_error$"):
            await collect(client, "http://127.0.0.1:1/v1/chat/completions")


@pytest.mark.parametrize("change", [{"stream": False}, {"n": True}, {"n": 2}])
async def test_reject_invalid_payload_before_sending(change: dict) -> None:
    async with StreamingClient() as client:
        with pytest.raises(ValueError):
            async with client.stream("http://127.0.0.1:1/v1/chat/completions", PAYLOAD | change):
                pytest.fail("Invalid payload sent")


@pytest.mark.parametrize("success", [True, False])
async def test_cli_probe_writes_manifest_even_when_stream_fails(
    tmp_path, monkeypatch, success: bool
) -> None:
    import os
    from argparse import Namespace
    from pathlib import Path

    from inference_lab.config.io import load_experiment
    from inference_lab.config.schemas import ExperimentConfig
    from inference_lab.serving.__main__ import _probe, _write
    from inference_lab.serving.runtime import VERSIONS, Observation, launch_plan, process_identity

    async def source() -> AsyncIterator[bytes]:
        yield event("synthetic", "length")
        if success:
            yield USAGE + DONE

    async with server(source) as url:
        config = load_experiment(
            Path(__file__).resolve().parents[2] / "experiments/qwen3b-fp16.yaml"
        ).model_dump(mode="json")
        config["target"]["url"] = url
        plan = launch_plan(ExperimentConfig.model_validate(config))
        identity = process_identity(os.getpid())
        _write(tmp_path / "launch.json", plan.model_dump(mode="json"))
        _write(tmp_path / "server-process.json", {"pid": os.getpid(), "start_ticks": identity[1]})
        _write(tmp_path / "versions.json", VERSIONS)
        (tmp_path / "server.log").write_text(
            "Synthetic CPU fixture: no real runtime observations.\nApplication startup complete.\n"
        )
        monkeypatch.setattr(
            "inference_lab.serving.__main__.gpu_observations",
            lambda *_: {
                "gpu.present": Observation(
                    value=None, source="synthetic", reason="No GPU in CPU test"
                ),
                "gpu.used": Observation(
                    value=None, source="synthetic", reason="No GPU in CPU test"
                ),
            },
        )
        args = Namespace(output=tmp_path, startup_timeout=2)
        assert await _probe(args) == (2 if success else 1)
        manifest = json.loads((tmp_path / "runtime_manifest.json").read_text())
        assert manifest["readiness"]["status"] == ("VALIDATED" if success else "FAILED")
        assert manifest["status"] == ("UNAVAILABLE" if success else "FAILED")
        assert manifest["observed"]["runtime.dtype"]["value"] is None
        assert "synthetic" not in json.dumps(manifest["readiness"])  # No response content stored.
        with pytest.raises(ValueError, match="already exists"):
            await _probe(args)
