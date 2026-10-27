# Architecture and configuration — stage 02

The [PDD](PROJECT_DESIGN_DOCUMENT.md) defines scope; the
[benchmark contract](BENCHMARK_CONTRACT.md) defines metrics and artifact semantics.
`inference_lab.config.schemas` implements contract **1.0**. Unknown fields and
versions are rejected. It validates records, without executing benchmarks or gates.

## Components and profiles

The intended path is an external HTTP runner → vLLM, or runner → separate FastAPI
gateway → vLLM. The gateway never loads weights. JSON artifacts are authoritative;
PostgreSQL later imports a rebuildable index. Client, runtime and GPU metrics stay
distinct. Kernel Lab runs independently from serving measurements.

Local/CI currently runs the health API and synthetic SSE upstream without GPU
packages or weights. Compose starts only those services. Kaggle uses processes in
one session, loopback HTTP, and autonomous artifacts, without a database, Compose,
persistent daemon, or tunnel to a laptop.

## Settings

`Settings` reads environment variables and optional `.env`; `.env.example` lists
APP_ENV, LOG_LEVEL, HOST, PORT, VLLM_BASE_URL, VLLM_MODEL, VLLM_TIMEOUT_SECONDS,
VLLM_STREAM_IDLE_TIMEOUT_SECONDS, PROMETHEUS_ENABLED, and STORE_PROMPT_CONTENT
(false by default). DATABASE_URL is optional, masked, and excluded from serialization.
HTTP endpoints reject embedded credentials, queries and fragments. Result schemas
have no prompt/response, authorization-header, or database-credential fields.

The serving readiness probe uses the configured global and stream-idle timeouts;
see [direct serving](serving.md) for launch profiles and runtime-manifest evidence.
Other settings define the contract for subsequent components. Existing Make targets
still start bootstrap services on their documented fixed ports; telemetry and
prompt storage are not activated by declaring their settings.

## Canonical YAML

`experiments/qwen3b-fp16.yaml` is the single experiment example. It preserves the
canonical fields and pinned model/tokenizer commits, with FP16 and explicitly
**enable_chunked_prefill=false**, following qualification c. The dataset is a
synthetic schema fixture; concurrency and measurement values are templates, not
qualified load capacities or performance claims.

```python
from pathlib import Path
from inference_lab.config.io import load_experiment

experiment = load_experiment(Path("experiments/qwen3b-fp16.yaml"))
experiment.validate_input_length(128)  # Supply a measured, post-template length.
```

Safe YAML loading rejects duplicate/non-string mapping keys, unsafe object tags,
and merge keys. Dataset/policy references are relative to `project_root` (default:
current directory). Dataset loading and tokenization are later runner work; a
filename is never evidence of input length.

Draft revisions may be mobile. `load_experiment(..., published=True)` and published
results require immutable model/tokenizer commits. Published real GPU results
also require actual used-GPU identities and qualification_id. Synthetic results
cannot claim GPU execution. Structural validation does not certify compatibility.

## Results and evidence

ExperimentRun identifies a campaign/run, zero-based point/repetition, and one
concurrency. ExperimentResult contains result_schema_version and
metric_contract_version, resolved config, environment, workload/runtime evidence,
counters, metrics, limitations, artifact references, and optional policy snapshot.
Expected launch configuration remains in experiment.yaml. Observations must not
be inferred from that expected configuration; unverified runtime values require
null and reasons. Present, visible and used GPUs are distinct; TP uses the used
count. T4 execution cannot claim BF16.

RequestMeasurement records IDs, plan order, separate transport/workload status,
ordered monotone timestamps, content-event count, server usage, finish reason,
normalized error, and explicit missing reasons. Invalid observed workload lengths
can be preserved with conformant=false and a limitation, rather than discarded.

All canonical aggregate metrics must appear. Missing values are null with reasons,
never default zero. Each metric records unit, source, sample_count and population.
Canonical client latencies use milliseconds and successful requests; rates use all
attempts and error_rate is a ratio. Percentiles use linear interpolation. SSE
events are not tokens; `client_tpot_estimate_ms` remains a reception estimate.

Native `model_json_schema()`, `model_dump_json()`, and `model_validate_json()` provide
schema export and round trips, avoiding a parallel schema/serializer implementation.

## Policies and comparisons

SLO/regression clauses live only in the policy referenced by policy_file. Policies
require finite thresholds, sample minima, explicit comparison dimensions, and a
nonnegative noise margin no larger than the regression threshold. TTFT/TPOT
policies require streaming. Published gates need calibration evidence and at least
three repetitions; these conditions do not prove statistical precision.

ComparisonResult records run IDs, structured reasons and changed paths, with
PASS/FAIL/INCONCLUSIVE/INVALID mapping to exit_code 0/1/2/2. Unauthorized differences
require INVALID. Exact allowed paths such as runtime.version are accepted;
wildcards are rejected. Numerical gate evaluation and evidence sufficiency are
later comparator work, not implied by a valid ComparisonResult object.

All `tests/data/` records are explicitly synthetic, including invented timestamps
and lengths. Their uncalibrated policy demonstrates structure only. Preserved 00B
artifacts remain GPU qualification evidence and are not converted into benchmark
ExperimentResult documents.
