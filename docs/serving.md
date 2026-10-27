# Direct serving — stage 03

`inference_lab.serving` runs vLLM as an external process. FastAPI still exposes
only health; the gateway and benchmark runner are subsequent stages. No GPU import,
model download, retry, or database write occurs in the HTTP client or CPU tests.

## Qualified launch profile

The launch builder reuses [qualification c](../qualification/kaggle-t4-20261025-c/qualification-report.md):
Python 3.12.12, torch 2.8.0 / CUDA runtime 12.8, vLLM 0.10.2 **V0/XFORMERS**,
Triton 3.4.0, Transformers 4.55.2, xformers 0.0.32.post1. It checks the selected
GPU interpreter and reference Tesla T4 devices before launching. A version or
hardware mismatch stops execution; it does not install replacements or fall back.

The model/tokenizer commits, FP16, KV setting `auto`, eager execution, disabled
prefix caching, and disabled chunked prefill are explicit. Scheduler/context/memory
values come from the experiment and are recorded; changed capacities still need
GPU validation. Single GPU uses index 0 / TP=1. FP16 TP=2 uses indices 0,1 and the
`mp` executor. AWQ is limited to its qualified publisher checkpoint/revision and
TP=1. Its original source commit remains unknown; no causal performance claim follows.
Unsupported profile changes fail; extending qualification is a separate operation.

Preview the exact command without GPU dependencies or downloads:

```sh
uv run --locked python -m inference_lab.serving command experiments/qwen3b-fp16.yaml \
  --python /path/to/qualified/.venv/bin/python
```

`--served-model-name` explicitly uses the model repo ID, replacing the historical
qualification-only alias. Both launch and smoke use that same name. Sensitive
credentials, full prompts/responses, and arbitrary environment variables are
excluded from launch manifests. Request/output logging is disabled in vLLM.
Use a clean launch shell: conflicting CUDA visibility or unexpected `VLLM_*`
environment overrides are rejected. Access tokens may remain in the environment;
they are not copied into artifacts.

## Kaggle: processes and loopback, without Docker

Make this repository and its locked CPU control environment available in the same
session as the **existing qualified GPU environment**. `uv sync --locked` installs
only CPU control dependencies; it does not recreate a lost GPU environment. Stop
the mock or an earlier serving process using port 8001 first. Never kill unrelated
GPU processes. From the repository root:

```sh
GPU_PYTHON=/kaggle/working/qualification/kaggle-t4-20261025-b/.venv/bin/python
test -x "$GPU_PYTHON" || exit 1
mkdir -p results/serving
attempt="results/serving/$(date -u +%Y%m%dT%H%M%SZ)"
uv run --locked python -m inference_lab.serving serve experiments/qwen3b-fp16.yaml \
  --python "$GPU_PYTHON" --output "$attempt" > "$attempt-launch.log" 2>&1 &
```

In the same shell, run readiness (it waits for launch artifacts and startup):

```sh
uv run --locked python -m inference_lab.serving probe --output "$attempt" \
  --startup-timeout 600
```

`serve` requires a fresh directory and becomes the vLLM process via `exec`.
`probe` waits for its own startup log, verifies the recorded process identity,
polls `/health`, then performs one
real streaming generation with the experiment's generation settings, seed and final
usage enabled. Health polling is readiness, never a benchmark retry. Generation is
not retried. Both final usage and a valid finish reason plus `[DONE]` are required;
a healthy port or `/v1/models` alone does not pass. This synthetic prompt is only a
functional smoke, without a token-length workload or performance claim.

A probe reads `VLLM_TIMEOUT_SECONDS` (global request deadline, default 180) and
`VLLM_STREAM_IDLE_TIMEOUT_SECONDS` (socket inactivity, default 30). Startup polling
has its own deadline. Increase a timeout explicitly if needed; retain failed
artifacts. Each manifest is exclusive: a repeated attempt needs a new launch directory.
A launch preflight failure retains available launch/version evidence and returns
nonzero; it cannot produce a successful readiness manifest.

For TP=2, copy the YAML and explicitly set `runtime.tensor_parallel_size: 2`.
For AWQ, copy it, set `runtime.quantization: awq`, both model/tokenizer repo IDs to
`Qwen/Qwen2.5-3B-Instruct-AWQ`, and both revisions to
`3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd`; keep FP16 and TP=1. These launch
variants are supported because 00B qualified them, not because this stage reran them.

Stop a foreground launch with Ctrl+C. For the background launch above, signal the
recorded server rather than the `uv` wrapper, after verifying its process identity:

```sh
uv run --locked python - "$attempt" <<'PY'
import json, os, signal, sys
from pathlib import Path
from inference_lab.serving.runtime import process_identity
process = json.loads((Path(sys.argv[1]) / "server-process.json").read_text())
handle = os.pidfd_open(process["pid"])
identity = process_identity(process["pid"])
assert identity is not None and identity[1] == process["start_ticks"], "Server exited or PID changed"
signal.pidfd_send_signal(handle, signal.SIGINT)
os.close(handle)
PY
```

Check `nvidia-smi` afterward: vLLM workers can outlive the API process. If work or
processes remain, preserve evidence and clean up only that launch's workers or
restart the isolated session before another check. There is no redeployment orchestrator.

## Artifacts and interpretation

A launch retains `launch.json`, `versions.json`, `gpu-before.json`,
`server-process.json` and `server.log`. After a probe, `runtime_manifest.json`
contains the expected configuration, allowlisted command/environment, observations
with per-value sources, mismatches, and the content-free readiness result. Stream
failures retain their normalized error and manifest, rather than losing evidence.
Preflight failures also retain `launch-failure.json`. A busy port is rejected
before GPU launch; the probe does not mistake an older healthy endpoint for this server.

The parser recognizes the qualified 0.10.2 startup format. Engine initialization
proves its reported model/revisions, dtype, quantization, TP and cache/eager settings;
startup argument records expose scheduler/memory settings when present. Unknown
formats/omitted values remain `null` with reasons. Conflicting log values are
unknown. `runtime.quantization="none"` denotes observed unquantized execution;
it differs from an unknown observation. `auto` is never presented as a resolved KV
dtype, and AWQ metadata never identifies a particular linear kernel.

GPU inventory includes index/UUID/name/memory. Used GPU UUIDs require observed
compute processes belonging to the launch PID or descendants, guarded by Linux
process start ticks. Present/selected GPUs are not automatically labeled used.
A visible-device launch setting is configuration, not a separate CUDA-enumeration
observation. GPU inventory failure stays unknown. This is a point-in-time process
observation, not per-request tracing.

Top-level `status` combines the generation smoke, expected-field observations and
GPU ownership/placement. Missing evidence produces `UNAVAILABLE`, even when
`readiness.status` is `VALIDATED`; mismatches or stream failures produce `FAILED`.
CLI exits are 0 for validated, 1 for failed, and 2 for unavailable/not tested.
Resolved KV dtype and linear-kernel identity remain explicitly unknown and outside
this smoke status: inspect limitations before publication. The manifest is serving
evidence, not a benchmark
`ExperimentResult`. KV dtype/linear-kernel introspection, calibrated performance,
and production readiness are not claimed.

## Reusable streaming client

```python
from inference_lab.serving.client import StreamingClient

async with StreamingClient(concurrency=4, total_timeout=180, idle_timeout=30) as client:
    async with client.stream(url, payload) as events:  # stream=true, n=1
        async for event in events:
            if event.content:
                consume(event.content)  # Ephemeral; do not log complete responses.
            if event.done:
                break
```

The caller supplies generation options, including `stream_options.include_usage`.
Usage is required by default; `require_usage=False` is an explicit mode for an API
that does not provide it, without inventing token counts. Reuse one client for its
lifespan; the pool is sized to concurrency. Consume within the request's `async with`:
leaving it closes upstream even after early break, exceptions or task cancellation.
The global deadline includes pool admission and time inside that request context.

The client dispatches SSE at blank-line boundaries, handles comments, CR/LF/CRLF,
fragmented UTF-8 and multiline data, and forwards parsed events/usage without
retokenization. Role-only/usage/DONE events have no content. One event may contain
multiple tokens. Lines and event data are bounded to 1 MiB. Transport errors,
non-200 responses, invalid events, error frames, unexpected finish reasons, missing
usage and truncated streams raise `StreamError` with safe codes/status only.
Redirects and retries are disabled. No partial response becomes a successful stream.

## Optional Linux GPU Compose host

This is separate from Kaggle and the CPU Compose stack. It requires an accessible
NVIDIA Container Toolkit GPU host and builds an opt-in image from Python 3.12.12
with the locked GPU extra. Building installs GPU packages, not model weights;
starting the service may download the explicitly pinned public checkpoint.
No image build or hardware compatibility is assumed from the Kaggle qualification.

```sh
export SERVING_RUN_ID="stage03-$(date -u +%Y%m%dT%H%M%SZ)"
docker compose -f compose.gpu.yaml --profile gpu config --quiet
docker compose -f compose.gpu.yaml --profile gpu up --build -d vllm
docker compose -f compose.gpu.yaml --profile gpu exec vllm \
  python -m inference_lab.serving probe --output "/app/results/$SERVING_RUN_ID"
docker compose -f compose.gpu.yaml --profile gpu cp \
  "vllm:/app/results/$SERVING_RUN_ID" results/
docker compose -f compose.gpu.yaml --profile gpu down
```

The profile reserves GPU 0 and uses Linux host networking/PIDs so loopback and
GPU-process attribution work together. The API binds only `127.0.0.1:8001`. Model
cache/artifacts use persistent named volumes; `down` retains them. It does not
start the mock/API services. Inspect startup errors if the exact version preflight
fails; do not loosen pins to make an unqualified image run. TP=2/AWQ Compose
variants require explicit configuration/device changes and separate validation.

## Verification and remaining GPU checks

`make lint typecheck test` covers the client over controlled real loopback sockets,
CLI probe success/failure artifacts, pin preservation, startup-log parsing,
conflicting/unknown evidence, UUID ownership and process-reuse protection. Parsing
old qualification logs is an offline test, not a new GPU execution. CPU tests never
load GPU packages or download weights. The optional profile can be syntax-checked
without a GPU via `docker compose ... config`.

Run the launch/probe above on the qualified T4 session for the separate GPU smoke.
For GPU cancellation, use a single isolated stream, receive content, then leave its
request context early. Preserve the log and available runtime running/waiting
counters before/after; wait for updates and verify that the request is aborted and
the engine drains. A closed client connection or an initially stale zero counter
alone is insufficient evidence. Missing counters/abort evidence means **NOT_TESTED**
or **UNAVAILABLE**, never validated cancellation. Do not start another measurement
with unresolved residual work. Errors after HTTP 200 and cancellation are validated
on the controlled CPU upstream; actual vLLM drain and new deployment smokes remain
**NOT_TESTED** until executed on the target hardware.

Implementation references: [HTTPX async stream lifetime](https://www.python-httpx.org/async/),
[WHATWG SSE framing](https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation),
[vLLM 0.10.2 arguments](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/engine/arg_utils.py),
and [runtime scheduler counters](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/engine/metrics.py).
