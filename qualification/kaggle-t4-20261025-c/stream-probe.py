
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

q = Path(sys.argv[1])

async def main():
    result = {
        "status": "FAILED",
        "request_start_ns": None,
        "first_content_ns": None,
        "last_content_ns": None,
        "terminal_ns": None,
        "content_events": 0,
        "usage": None,
        "token_count_source": "server_usage",
        "finish_reason": None,
        "done_received": False,
    }
    payload = {
        "model": "qualification-qwen3b",
        "messages": [{
            "role": "user",
            "content": "Count from one to one hundred, one integer per line."
        }],
        "stream": True,
        "stream_options": {"include_usage": True},
        "n": 1,
        "temperature": 0,
        "seed": 42,
        "max_tokens": 128,
        "ignore_eos": True,
    }
    timeout = aiohttp.ClientTimeout(total=180, sock_read=30)
    try:
        with (q / "stream-sample.txt").open("a") as sample:
            async with aiohttp.ClientSession(timeout=timeout) as client:
                result["request_start_ns"] = time.perf_counter_ns()
                async with client.post(
                    "http://127.0.0.1:8001/v1/chat/completions",
                    json=payload,
                ) as response:
                    assert response.status == 200, (
                        f"HTTP status {response.status}"
                    )
                    assert response.headers.get(
                        "Content-Type", ""
                    ).startswith("text/event-stream")

                    data = []
                    async for raw in response.content:
                        line = raw.decode("utf-8").rstrip("\r\n")
                        if line.startswith("data:"):
                            data.append(line[5:].removeprefix(" "))
                            continue
                        if line or not data:
                            continue

                        received_ns = time.perf_counter_ns()
                        event = "\n".join(data)
                        data.clear()

                        if event == "[DONE]":
                            result["done_received"] = True
                            result["terminal_ns"] = received_ns
                            sample.write(json.dumps({
                                "received_ns": received_ns, "done": True
                            }) + "\n")
                            break

                        chunk = json.loads(event)
                        assert "error" not in chunk, "Server sent an SSE error"
                        assert chunk.get("model") == "qualification-qwen3b"
                        record = {"received_ns": received_ns,
                                  "content_chars": 0}
                        for choice in chunk.get("choices", []):
                            assert choice["index"] == 0
                            content = choice.get("delta", {}).get("content")
                            if content:
                                assert isinstance(content, str)
                                result["content_events"] += 1
                                if result["first_content_ns"] is None:
                                    result["first_content_ns"] = received_ns
                                result["last_content_ns"] = received_ns
                                record["content_chars"] += len(content)
                            if choice.get("finish_reason") is not None:
                                result["finish_reason"] = choice["finish_reason"]
                                record["finish_reason"] = choice["finish_reason"]
                        if chunk.get("usage") is not None:
                            result["usage"] = chunk["usage"]
                            record["usage"] = chunk["usage"]
                        sample.write(json.dumps(record) + "\n")
                        sample.flush()

        assert result["done_received"], "Stream ended without [DONE]"
        assert result["first_content_ns"] is not None, "No generated content"
        assert result["finish_reason"] == "length", "Unexpected finish reason"
        assert result["usage"] is not None, "Missing server token usage"
        assert result["usage"]["completion_tokens"] == 128, (
            "Controlled output length was not respected"
        )
        assert result["usage"]["prompt_tokens"] > 0
        result["status"] = "VALIDATED"

    except Exception as exc:
        if result["terminal_ns"] is None:
            result["terminal_ns"] = time.perf_counter_ns()
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["metrics"] = {}
    for name in ("client_ttft_ms", "client_e2e_ms",
                 "client_tpot_estimate_ms"):
        result["metrics"][name] = {
            "value": None, "unit": "ms",
            "reason": "Successful stream with required data not established."
        }

    if result["status"] == "VALIDATED":
        start = result["request_start_ns"]
        first = result["first_content_ns"]
        last = result["last_content_ns"]
        end = result["terminal_ns"]
        tokens = result["usage"]["completion_tokens"]
        result["metrics"]["client_ttft_ms"].update(
            value=(first - start) / 1e6, reason=None)
        result["metrics"]["client_e2e_ms"].update(
            value=(end - start) / 1e6, reason=None)
        tpot = result["metrics"]["client_tpot_estimate_ms"]
        if tokens > 1 and result["content_events"] >= 2:
            tpot.update(value=(last - first) / 1e6 / (tokens - 1),
                        reason=None)
        else:
            tpot["reason"] = "Need >1 output token and >=2 content events."

    # Append attempts so rerunning never erases previous evidence.
    with (q / "stream-results.jsonl").open("a") as out:
        out.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "VALIDATED" else 1

sys.exit(asyncio.run(main()))
