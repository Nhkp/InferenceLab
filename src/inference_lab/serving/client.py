"""One persistent HTTP pool, explicit stream lifetime, no retries or content logging."""

import asyncio
import json
import math
import re
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import TypeAdapter

from inference_lab.config.schemas import Endpoint


class StreamError(Exception):
    """Safe metadata only: never embed upstream bodies, URLs, or exception text."""

    def __init__(self, code: str, *, status_code: int | None = None) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(code)


@dataclass(frozen=True)
class StreamEvent:
    data: dict[str, Any] | None
    received_ns: int
    done: bool = False

    @property
    def content(self) -> str:
        if self.data is None or not self.data["choices"]:
            return ""
        return self.data["choices"][0]["delta"].get("content") or ""


async def _lines(response: httpx.Response) -> AsyncIterator[str]:
    # ponytail: bound one SSE line/event to 1 MiB; raise this explicit limit if a
    # future API needs larger events rather than buffering unbounded upstream data.
    pending = b""
    first = True
    async for chunk in response.aiter_bytes():
        pending += chunk
        while match := re.search(rb"[\r\n]", pending):
            end = match.start()
            if pending[end:] == b"\r":
                break  # A CRLF may straddle reads.
            if end > 1_048_576:
                raise StreamError("event_too_large")
            line = pending[:end].decode("utf-8", errors="replace")
            if first:
                line = line.removeprefix("\ufeff")
                first = False
            width = 2 if pending[end : end + 2] == b"\r\n" else 1
            pending = pending[end + width :]
            yield line
        if len(pending) > 1_048_576:
            raise StreamError("event_too_large")
    if pending:
        # A final CR terminates a line; other unterminated data is not dispatched.
        if pending.endswith(b"\r"):
            yield pending[:-1].decode("utf-8", errors="replace")
        else:
            yield pending.decode("utf-8", errors="replace")


def _invalid_constant(_value: str) -> None:
    raise ValueError("Non-JSON numeric constant")


def _decode(data: str, event_type: str) -> StreamEvent:
    now = time.perf_counter_ns()
    if event_type == "error":
        raise StreamError("upstream_error")
    if data == "[DONE]":
        return StreamEvent(None, now, done=True)
    try:
        event = json.loads(data, parse_constant=_invalid_constant)
    except (ValueError, RecursionError):
        raise StreamError("invalid_event") from None
    if not isinstance(event, dict):
        raise StreamError("invalid_event")
    if "error" in event:
        raise StreamError("upstream_error")
    choices = event.get("choices")
    if not isinstance(choices, list) or len(choices) > 1:
        raise StreamError("invalid_choices")
    if choices:
        choice = choices[0]
        if (
            not isinstance(choice, dict)
            or type(choice.get("index")) is not int
            or choice["index"] != 0
        ):
            raise StreamError("invalid_choices")
        delta = choice.get("delta")
        if not isinstance(delta, dict) or not isinstance(
            delta.get("content", ""), str | type(None)
        ):
            raise StreamError("invalid_delta")
        if choice.get("finish_reason") not in (None, "stop", "length"):
            raise StreamError("unexpected_finish_reason")
    usage = event.get("usage")
    if usage is not None:
        if not isinstance(usage, dict) or any(
            type(usage.get(key)) is not int or usage[key] < 0
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            raise StreamError("invalid_usage")
        if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
            raise StreamError("invalid_usage")
    return StreamEvent(event, now)


async def _events(response: httpx.Response, *, require_usage: bool) -> AsyncIterator[StreamEvent]:
    data: list[str] = []
    size = 0
    event_type = "message"
    finished = False
    usage_seen = False
    async for line in _lines(response):
        if line:
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "data":
                size += len(value.encode("utf-8")) + 1
                if size > 1_048_576:
                    raise StreamError("event_too_large")
                data.append(value)
            elif field == "event":
                event_type = value
            continue
        if not data:
            event_type = "message"
            continue
        event = _decode("\n".join(data), event_type)
        data, size, event_type = [], 0, "message"
        if event.done:
            if not finished or (require_usage and not usage_seen):
                raise StreamError("incomplete_stream")
            yield event
            return
        assert event.data is not None
        choices = event.data["choices"]
        if choices:
            if finished or usage_seen:
                raise StreamError("event_after_finish")
            finished = choices[0].get("finish_reason") is not None
        if event.data.get("usage") is not None:
            if not finished or usage_seen:
                raise StreamError("unexpected_usage")
            usage_seen = True
        yield event
    raise StreamError("premature_eof")


class StreamingClient:
    """Use once per campaign/proxy lifespan; consume each request inside `async with`.

    Closing that context, including after a break, exception, or cancellation, closes
    upstream. The global deadline includes pool admission and consumer time. Idle
    timeout measures socket inactivity, not time since the last content event.
    """

    def __init__(
        self,
        *,
        total_timeout: float = 180,
        idle_timeout: float = 30,
        concurrency: int = 1,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if any(not math.isfinite(t) or t <= 0 for t in (total_timeout, idle_timeout)):
            raise ValueError("Timeouts must be finite and positive")
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("Concurrency must be a positive integer")
        self.total_timeout = total_timeout
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(total_timeout, read=idle_timeout),
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    async def __aenter__(self) -> "StreamingClient":
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self._client.aclose()

    @asynccontextmanager
    async def stream(
        self,
        url: str,
        payload: Mapping[str, Any],
        *,
        require_usage: bool = True,
    ) -> AsyncIterator[AsyncIterator[StreamEvent]]:
        # Validate locally before opening the connection; never follow redirects to
        # a different runtime. Generation options otherwise pass through unchanged.
        TypeAdapter(Endpoint).validate_python(url)
        if (
            payload.get("stream") is not True
            or type(payload.get("n", 1)) is not int
            or payload.get("n", 1) != 1
        ):
            raise ValueError("Only stream=true and n=1 are supported")
        try:
            async with asyncio.timeout(self.total_timeout):
                async with self._client.stream("POST", url, json=dict(payload)) as response:
                    if response.status_code != 200:
                        raise StreamError("http_error", status_code=response.status_code)
                    if (
                        response.headers.get("content-type", "").split(";")[0].strip().lower()
                        != "text/event-stream"
                    ):
                        raise StreamError("invalid_content_type")
                    yield _events(response, require_usage=require_usage)
        except TimeoutError:
            raise StreamError("global_timeout") from None
        except httpx.ReadTimeout:
            raise StreamError("idle_timeout") from None
        except httpx.TimeoutException:
            raise StreamError("http_timeout") from None
        except httpx.HTTPError:
            raise StreamError("network_error") from None


async def readiness(
    url: str,
    payload: Mapping[str, Any],
    *,
    startup_timeout: float = 600,
    total_timeout: float = 180,
    idle_timeout: float = 30,
) -> dict[str, Any]:
    """Health polling is readiness only. Run one generation smoke, never retry it."""
    endpoint = TypeAdapter(Endpoint).validate_python(url)
    if not math.isfinite(startup_timeout) or startup_timeout <= 0:
        raise ValueError("Startup timeout must be finite and positive")
    health_url = str(endpoint).removesuffix("/v1/chat/completions") + "/health"
    try:
        async with asyncio.timeout(startup_timeout):
            async with httpx.AsyncClient(trust_env=False, timeout=2) as health:
                while True:
                    try:
                        if (await health.get(health_url)).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(1)
    except TimeoutError:
        raise StreamError("startup_timeout") from None
    content_events = 0
    usage = None
    finish_reason = None
    async with StreamingClient(total_timeout=total_timeout, idle_timeout=idle_timeout) as client:
        async with client.stream(url, payload) as events:
            async for event in events:
                if event.content:
                    content_events += 1
                if event.data is not None:
                    if event.data.get("usage") is not None:
                        usage = event.data["usage"]
                    if event.data["choices"]:
                        finish_reason = (
                            event.data["choices"][0].get("finish_reason") or finish_reason
                        )
    if not content_events or usage is None or usage["completion_tokens"] < 1:
        raise StreamError("empty_generation")
    return {
        "status": "VALIDATED",
        "content_events": content_events,
        "usage": usage,
        "finish_reason": finish_reason,
    }
