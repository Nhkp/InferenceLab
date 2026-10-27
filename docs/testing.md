# Testing

`make install` installs the default CPU dependencies from `uv.lock`. GPU extras are
opt-in. `make lint typecheck test` runs Ruff, strict source typing, and CPU tests.

Tests use pytest/pytest-asyncio and `test_*.py` names. Markers are `unit`,
`integration`, `gpu`, and `slow`; unknown markers fail collection. No coverage
percentage is mandated.

```sh
uv run --locked pytest -m "not gpu and not slow"
uv run --locked pytest -m integration
```

The health test checks the response without importing torch/vLLM/Triton. The mock
checks reject unsupported inputs. A real loopback HTTP test gates the generator
after the role frame and verifies that the client receives that frame before the
generator is released, then checks content, synthetic usage, and termination.
This avoids treating an in-process transport's buffered response as proof of
incremental HTTP delivery.

Configuration tests cover YAML/JSON round trips, duplicate/unsafe YAML rejection,
immutable publication revisions, unsupported fields/versions, non-finite values,
strict booleans/counts, workload bounds, used-GPU/TP consistency, missing reasons,
units/populations, policy calibration and comparison dimensions, verdict exit codes,
native JSON Schema generation, and exclusion of database credentials. Fixture
data is explicitly synthetic.

Direct-serving tests use real loopback sockets to exercise fragmented UTF-8/SSE,
multiline/CRLF framing, role-only events, multi-token content, final usage and DONE,
HTTP/stream/network errors, premature termination, global/idle timeouts, early break,
and cancellation closing upstream. Probe tests preserve success/failure manifests.
Launch tests enforce qualified pins/profile, reject changed assumptions, retain
preflight failures, and protect existing artifacts. Offline log parsing checks
expected/effective differences, unknown/conflicting values and GPU UUID/process
ownership; it does not execute GPU work. See [serving](serving.md) for separate GPU checks.

The mock performs no tokenization, model download, or inference. Its counters are
explicit synthetic fixtures. Benchmark statistics, campaign artifact validation, gateway
forwarding, actual GPU engine drain, database behavior, and performance gates are
not implemented or tested at this stage.

There are currently no pytest GPU tests. A future `pytest -m gpu` run with no
selected checks is not validation. The real GPU evidence is in
`qualification/kaggle-t4-20261025-c/`; the original failed attempt is preserved.
Do not run qualification scripts as part of the CPU application tests.
