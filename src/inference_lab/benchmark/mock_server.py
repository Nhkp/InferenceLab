"""Synthetic streaming upstream for CPU development; never loads a model."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from inference_lab.gateway.app import health

app = FastAPI(title="InferenceLab synthetic upstream")
app.add_api_route("/health", health, methods=["GET"])


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=10_000)


class MockCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["mock-model"] = "mock-model"
    messages: list[Message] = Field(min_length=1, max_length=10)
    stream: Literal[True]


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
    # ponytail: fixed synthetic counters, not tokenization or measured GPU usage;
    # replace with explicit fixtures when workload/usage semantics are introduced.
    usage = {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    yield f"data: {json.dumps(event | {'choices': [], 'usage': usage})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def completions(_request: MockCompletionRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
