# Direct measurement — stage 08

The pack's execution order is **03 → 08 → 09 → 10 → 04**. This stage implements
one concurrency point and repetition. Stage 09 adds [campaign orchestration](campaigns.md)
on top of this runner; gates and gateway forwarding remain later steps. The runner is an external HTTP process and has no DB,
model-loading, or GPU-package dependency.

## CPU smoke

Start `make mock` in another terminal. Then, from the repository root:

```sh
smoke_dir=$(mktemp -d /tmp/inferencelab-smoke.XXXXXX)
uv run --locked python -m inference_lab.benchmark.plan \
  --experiment experiments/mock-stream.yaml --synthetic --output "$smoke_dir/plan.json"
uv run --locked python -m inference_lab.benchmark.workload \
  --experiment experiments/mock-stream.yaml --plan "$smoke_dir/plan.json" \
  --point 1 --output "$smoke_dir/run"
```

This measures the **CPU mock**, using fixed synthetic token counts. Its report and
result explicitly set `synthetic=true`, `published=false`, and no qualification ID.
The mock accepts the runner's generation subset: stream=true, n=1, temperature=0,
max_tokens=5, ignore_eos=true, seed, and include_usage=true. Its `/metrics` scheduler
gauges are process-local synthetic counters for the default single-worker mock.
No weight download or tokenization occurs on this path.

## Tokenizer-verified public workloads

`experiments/datasets/` contains four versioned, non-sensitive datasets: short
128/128, long 512/128, long 2048/128, and generation-heavy 128/512. Inputs were
verified **after chat templating**, using the pinned Qwen tokenizer, Transformers
4.55.2, tokenizers 0.21.4, and Jinja2 3.1.6. `tokenizer-evidence.json` records hashes,
shapes, revision, and versions. This is CPU tokenization evidence, not GPU inference.
The `synthetic` dataset flag identifies mock fixtures; these public reference inputs
are synthetic text intended for real serving and therefore use `synthetic=false`.

The Qwen example now references `short-chat-v1.jsonl`; model/runtime pins are
unchanged. For another shape, copy the canonical YAML and change `workload.dataset`
and `generation.max_tokens` together. No CLI generation/runtime overrides exist.
Dataset rows must share a name/version/shape and have unique IDs.

Prepare a plan **before measurement** in the existing qualified environment:

```sh
GPU_PYTHON=/kaggle/working/qualification/kaggle-t4-20261025-b/.venv/bin/python
PYTHONPATH=src "$GPU_PYTHON" -m inference_lab.benchmark.plan \
  --experiment experiments/qwen3b-fp16.yaml --output /kaggle/working/short-plan.json
```

Only the tokenizer is loaded; weights are not needed. Missing or different tokenizer
dependencies fail rather than being installed automatically. Tokenizer files may
be fetched at the explicit immutable revision. The CPU runner consumes the saved
plan without importing Transformers. `--generate-datasets` writes reference JSONL
files and tokenizer provenance into a fresh output location instead of a plan.
Template hashing uses 00B's UTF-8 JSON-string convention, not raw template bytes.

Plans fix the dataset SHA-256, tokenizer/model revisions, template/hash/parameters,
post-template input lengths, seeded request order, payloads, and warmup requests.
Sampling is deterministic with replacement. One prepared plan can be reused for
paired runs; run-specific request IDs are separate from its hash. Inputs plus
requested outputs must fit the configured context. Only versioned public inputs
belong in these artifacts: plan/dataset snapshots contain their prompt payloads;
responses, authorization headers, and secrets are never recorded.

## One real point/repetition

Launch/probe vLLM with the [serving runbook](serving.md), in the same Kaggle session
as the runner. A fresh, validated runtime manifest and live server/GPU ownership
are required. Model, runtime, seed, and direct target must match. Remote targets are
rejected so laptop metadata cannot be presented as server hardware.

```sh
uv run --locked python -m inference_lab.benchmark.workload \
  --experiment experiments/qwen3b-fp16.yaml --plan /kaggle/working/short-plan.json \
  --runtime-manifest results/serving/ATTEMPT/runtime_manifest.json \
  --point 0 --repetition 0
```

Point/repetition are zero-based selections from the YAML, not overrides. Default
output is `results/<campaign_id>/<run_id>/`; IDs are generated unless supplied.
`--output` requires a fresh directory. Manually grouped runs can share
`--campaign-id`, but each needs its own run ID. This command does not iterate the
concurrency list, repeat runs, alternate configurations, or restart vLLM.

Use an isolated target. The runner checks exposed running/waiting gauges before
warmup, waits for all warmup streams to finish, checks drain, then measures exactly
N attempts with C fixed workers and a C-sized persistent HTTP pool. There is no
think time, N-task admission queue, or retry. Payloads/tokenization are prepared
outside the window. Warmup records and post-run drain checks are outside measurement.
Available runtime gauges are required; missing/invalid counters stop the run.
Gauges alone, especially stale zeros after cancellation, do not prove engine abort.

## Artifacts, metrics, and interruption

Each run retains experiment, plan, dataset, environment, runtime manifest, available
server/package/NVIDIA evidence, `result.json`, `report.md`, `raw/requests.jsonl`,
and separate `raw/warmup.jsonl`. `raw/gpu.jsonl` is empty with an explicit limitation;
continuous GPU telemetry belongs to its later stage. Original qualification evidence
is preserved. Metadata uses observed GPU UUIDs and distinguishes present/visible/used
GPUs, driver-supported CUDA, PyTorch CUDA runtime, and an available toolkit.

Every issued attempt records monotone start/content/terminal timestamps, normalized
errors, server usage/source, finish reason, and separate transport/workload status.
Role/empty/usage/DONE events do not count as content. A terminal success requires
valid finish/usage/DONE. Wrong input/output lengths retain their evidence and mark
the workload nonconformant. This stage always leaves `published=false`.

All aggregates can be recalculated from raw request records. Percentiles are linear
and conditional on successful transport with available observations; coverage is
reported separately. Rates share `[min(start), max(terminal)]` over **all** attempts,
including errors and drain of the finite request batch. Partial-response tokens are
excluded from successful throughput. Empty/zero-duration windows have null rates
with reasons. TPOT divides the reception interval by output_tokens−1, never chunk
count; it is null for <=1 token or fewer than two content events.

```python
import json
from pathlib import Path
from inference_lab.benchmark.metrics import aggregate
from inference_lab.config.schemas import RequestMeasurement

folder = Path("results/CAMPAIGN/RUN")
raw = [RequestMeasurement.model_validate_json(line)
       for line in (folder / "raw/requests.jsonl").read_text().splitlines()]
recomputed = {name: metric.model_dump(mode="json") for name, metric in aggregate(raw).items()}
assert recomputed == json.loads((folder / "result.json").read_text())["metrics"]
```

Client timings include HTTP/pool work; local raw-record bookkeeping can influence
closed-loop pacing. Finite-batch throughput is not stationary capacity, and p99 from
small samples has limited tail support. No SLO, speedup, or regression verdict follows.

SIGINT/task cancellation closes upstream, records cancelled issued attempts, and
retains an incomplete result. Unissued requests remain only in the plan. Initial
incomplete snapshots and flushed raw lines also survive abrupt termination; after
SIGKILL/power loss there may be no terminal record for an active request, and the
initial summary can be stale. Recalculate available raw data without promoting it
to a complete run. Check residual server work before another measurement; real T4
cancellation/drain and this new GPU measurement path remain **NOT_TESTED** locally.
