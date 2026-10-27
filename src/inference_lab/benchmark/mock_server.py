"""Synthetic streaming upstream for CPU development; never loads a model."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated, Literal

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from inference_lab.gateway.app import health

app = FastAPI(title="InferenceLab synthetic upstream")
app.add_api_route("/health", health, methods=["GET"])
# ponytail: process-local accounting for the single-worker mock; a multi-worker
# mock would need shared counters. This never observes real GPU scheduler state.
_active_streams = 0


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=10_000)


class MockCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["mock-model"] = "mock-model"
    messages: list[Message] = Field(min_length=1, max_length=10)
    stream: Literal[True]
    n: Annotated[int, Field(strict=True, ge=1, le=1)] = 1
    temperature: Annotated[float, Field(strict=True, ge=0, le=0)] = 0.0
    max_tokens: Literal[5] = 5
    ignore_eos: Literal[True] = True
    seed: int = Field(default=42, ge=0)
    stream_options: dict[Literal["include_usage"], Literal[True]] = Field(default_factory=dict)


async def _stream() -> AsyncIterator[str]:
    for delta, finish in (
        ({"role": "assistant"}, None),
        ({"content": "Hello"}, None),
        ({"content": " world"}, "stop"),
    ):
        event = {
            "id": "synthetic-completion",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        yield f"data: {json.dumps(event)}\n\n"
        await asyncio.sleep(0.05)
    # ponytail: fixed synthetic input=3/output=5; real workloads use vLLM usage
    # and tokenizer-verified plans, not this mock's counters.
    usage = {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    yield f"data: {json.dumps(event | {'choices': [], 'usage': usage})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def completions(_request: MockCompletionRequest) -> StreamingResponse:
    async def tracked() -> AsyncIterator[str]:
        global _active_streams
        _active_streams += 1
        try:
            async for event in _stream():
                yield event
        finally:
            _active_streams -= 1

    return StreamingResponse(
        tracked(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def synthetic_scheduler() -> str:
    return (
        "# Synthetic CPU scheduler gauges, not GPU telemetry\n"
        f'vllm:num_requests_running{{model_name="mock-model"}} {_active_streams}\n'
        'vllm:num_requests_waiting{model_name="mock-model"} 0\n'
    )
