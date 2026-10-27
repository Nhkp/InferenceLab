# InferenceLab

Reproducible LLM serving experiments and GPU performance evidence.

## Current state

The repository contains an installable Python 3.12 package, a health endpoint,
a synthetic streaming upstream, versioned schemas, environment settings,
a reusable HTTP streaming client, explicit vLLM launch/probe commands, runtime
manifests, CPU tests, Docker Compose, and CPU CI.
Benchmark runners, gateway forwarding, database imports, telemetry, dashboards,
regression evaluation, and the portfolio Kernel Lab are later implementation stages.

The authoritative [design](docs/PROJECT_DESIGN_DOCUMENT.md) and
[benchmark contract](docs/BENCHMARK_CONTRACT.md) define their behavior.
The [implementation pack](inference_lab_codex_prompts_v5_1/README.md)
defines stage order. The documentation links reuse those canonical files.

## Quickstart: CPU only

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Make.
The project selects **Python 3.12.12**, matching qualification; this explicitly
replaces the initial scaffold's Python 3.14 setting. No model weights or GPU
packages are needed for the default installation.

```sh
make install
make lint typecheck test
make run
```

In another terminal:

```sh
curl --fail http://127.0.0.1:8000/health
```

Expected response: `{"status":"ok"}`. Stop the server with Ctrl+C.
Run `make mock` in a separate terminal to start the synthetic upstream on port 8001:

```sh
curl --fail --no-buffer http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock-model","messages":[{"role":"user","content":"synthetic input"}],"stream":true}'
```

The mock emits a role event, two content events, a finish reason, fixed synthetic
usage, and `[DONE]`. Its counters are fixtures: two content events deliberately
carry a synthetic count of five output tokens. They are not model measurements.
Only the documented request subset is accepted; non-streaming requests are rejected.
The health-only API does not forward to this upstream yet.

## Commands and containers

| Command | Purpose |
| --- | --- |
| `make install` | Install the locked CPU package and development tools |
| `make lint` / `make format` | Check / apply Ruff linting and formatting |
| `make typecheck` | Run strict mypy on application source |
| `make test` | Run CPU unit and integration tests, excluding GPU/slow markers |
| `make run` / `make mock` | Run the health API / synthetic upstream |
| `make build` | Build the Python wheel and source distribution |
| `make up` / `make down` | Build/start / stop the CPU Compose services |

Compose exposes the API at `127.0.0.1:8000` and mock at `127.0.0.1:8001`.
It starts no database or GPU service. Docker is for local development;
Kaggle qualification uses processes and loopback HTTP in a single session.

## Repository layout

- `src/inference_lab/`: gateway, mock, config schemas/settings, and direct serving; benchmark,
  registry, telemetry, and storage namespaces for subsequent stages.
- `tests/unit/`, `tests/integration/`: CPU health, validation, and HTTP streaming checks.
- `qualification/`: preserved Kaggle evidence, failure history, and qualification scripts.
- `kernel_lab/rmsnorm/`, `experiments/datasets/`, `dashboards/`, `results/`,
  `deploy/{prometheus,grafana}/`, `kaggle/`, `scripts/`: reserved stage directories.
- `.github/workflows/ci.yaml`: locked CPU installation, lint, typing, tests, package
  build, and CPU image build; no GPU execution or weight downloads.

## Configuration and architecture

See [architecture and configuration](docs/architecture.md) for canonical schema 1.0,
environment variables, publication checks, and deployment profiles.
`experiments/qwen3b-fp16.yaml` is the experiment example. Its dataset and the
uncalibrated policy in `tests/data/` are synthetic schema fixtures.

```sh
uv run --locked python -c 'from pathlib import Path; from inference_lab.config.io import load_experiment; print(load_experiment(Path("experiments/qwen3b-fp16.yaml")).name)'
```

This validates configuration without starting a model or campaign. Numerical gates
and schema-based validation of future campaigns remain later stages.

## Qualified GPU environment

[Qualification `kaggle-t4-20261025-c`](qualification/kaggle-t4-20261025-c/qualification-report.md)
preserves actual user-executed Kaggle checks: FP16 streaming, `torch.compile`,
minimal Triton/RMSNorm correctness, TP=2, and same-source AWQ all passed.
The baseline uses one Tesla T4, FP16, TP=1, vLLM **0.10.2 V0**, **XFORMERS**,
eager mode, prefix caching disabled, and **chunked prefill disabled**.
Chunked prefill enabled failed in attempt b and remains unqualified.

Qualified core versions: Python **3.12.12**, torch **2.8.0** (CUDA runtime **12.8**),
Triton **3.4.0**, Transformers **4.55.2**, xformers **0.0.32.post1**.
Model/tokenizer: `Qwen/Qwen2.5-3B-Instruct` at
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. AWQ revision and provenance limits
are recorded in the report and its artifacts.

The optional `gpu` extra pins these core versions for isolated Linux x86_64 GPU
environments. It does not freeze the entire historical Kaggle image: consult its
preserved package inventory and qualify any changed effective environment.
CPU `uv sync` does not enable this extra. See [direct serving](docs/serving.md) for
Kaggle CLI commands, an optional Linux GPU Compose profile, runtime evidence,
timeouts, and cancellation semantics. The new deployment path still needs its own
T4 smoke; the preserved 00B evidence has not been rerun by stage 03.

## Measurement and limits

Future benchmarks use a closed-loop external HTTP runner, monotone client timing,
server usage tokens, and canonical JSON artifacts. SSE events are not tokens;
`client_tpot_estimate_ms` is a client reception estimate. PostgreSQL will index
artifacts without becoming a campaign dependency. Policy evaluation and schema-based artifact
validation of campaign artifacts are introduced with their implementation stages;
canonical schema definitions and synthetic-fixture validation are available now.

Current GPU evidence consists of correctness and functional smokes, without SLO,
p99, throughput, TP speedup, or quantization speedup certification. Runtime KV dtype
introspection and upstream cancellation/error-path coverage remain open.
Prometheus/Grafana configuration and Kernel Lab performance experiments are roadmap.

See [testing](docs/testing.md) for check scope. Keep secrets in environment variables;
never log authorization headers or complete prompts/responses.
