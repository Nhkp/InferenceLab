# Sequential campaigns — stage 09

The campaign CLI reuses the [one-point runner](benchmarking.md). It visits only the
configured concurrency list and repetitions: no generated model/dtype/workload grid,
no automatic runtime deployment, and no database dependency. Each run keeps its own
raw records, window, metrics and warmup; campaign reports never average percentiles.

## Try a CPU campaign

Start `make mock` in another terminal. From the repository root:

```sh
campaign_dir=$(mktemp -d /tmp/inferencelab-campaign.XXXXXX)
uv run --locked python -m inference_lab.benchmark.plan \
  --experiment experiments/mock-stream.yaml --synthetic --output "$campaign_dir/plan.json"
uv run --locked python -m inference_lab.benchmark.runner experiments/mock-stream.yaml \
  --plan "$campaign_dir/plan.json" --output "$campaign_dir/output"
cat "$campaign_dir/output/report.md"
```

This executes C=1 and C=4, one repetition each. Both root and point artifacts label
the mock as synthetic. The public request counts are fixtures; timings describe the
CPU HTTP path, not GPU inference. No tokenizer, weights or GPU imports are needed.

## Real co-located campaign

Prepare a verified plan and launch/probe the serving runtime as described in
[benchmarking](benchmarking.md) and [serving](serving.md). Then, in that same Kaggle
session, run:

```sh
uv run --locked python -m inference_lab.benchmark.runner experiments/qwen3b-fp16.yaml \
  --plan /kaggle/working/short-plan.json \
  --runtime-manifest results/serving/ATTEMPT/runtime_manifest.json
```

The default directory is `results/<generated_campaign_id>/`; `--campaign-id` and
`--output` can supply explicit identities/locations. Output must be fresh. The
reference config schedules 15 runs: five configured concurrencies × three
repetitions. Each run has a globally distinct ID containing its campaign ID,
condition, point index and repetition index. No percentile mixes these runs.

All inputs are validated before measurement. Every run rechecks frozen inputs,
live runtime identity/configuration/GPU ownership, performs its own warmup, and
checks sampled scheduler drain. A failed/incomplete/nonconformant point stops the
campaign; remaining runs stay pending. Missing idle counters stop execution.
Unexpected configuration/runtime mismatches are never authorized by comparison
`allowed_changes`: those apply only between declared conditions.

Use an isolated target. Campaigns run sequentially within one invocation; concurrent
traffic from other campaigns or standalone tools is not scheduled or isolated here.
The output lock prevents simultaneous invocations from overwriting the same campaign.

## Paired baseline/candidate protocol

One optional `--candidate` selects exactly two canonical experiments. Both need
prepared plans and the **same policy snapshot** through their canonical `policy_file`.
Comparison kind/allowed changes come only from that policy; no inline SLO syntax is
introduced. Names and policy-file locations are labels/references; behavioral config
differences must be explicitly allowed and are enumerated in `campaign.json`.

Paired runs require the same points/repetitions and matched dataset bytes, tokenizer
chat-template hash/parameters, input lengths, generation, seed and request order.
The pairing hash excludes model routing so same-source AWQ can use its served name;
ordinary plan hashes remain unchanged and preserve complete payloads. Different
context/workload experiments use separate single campaigns rather than being
misrepresented as matched requests.

Within each point, repetition order is **AB, BA, AB, ...**. AWQ/TP comparisons must
be exploratory: AWQ keeps TP/resources and activation dtype fixed; TP keeps the
checkpoint/tokenizer and dtype fixed. Only the qualified deployment profiles are
available; another runtime/model/hardware profile needs its own qualification.
Real regression pairs require at least three repetitions and an existing nonempty
calibration reference captured before any candidate run. Calibration files are
hashed/preserved; their statistical adequacy and gate verdict belong to stage 10.
Synthetic policy fixtures remain explicitly uncalibrated and unpublished.

This copyable mock example demonstrates handoffs without claiming a useful candidate:

```sh
paired_dir=$(mktemp -d /tmp/inferencelab-paired.XXXXXX)
uv run --locked python -m inference_lab.benchmark.plan \
  --experiment experiments/mock-paired.yaml --synthetic --output "$paired_dir/plan.json"
uv run --locked python -m inference_lab.benchmark.runner experiments/mock-paired.yaml \
  --plan "$paired_dir/plan.json" --candidate experiments/mock-paired.yaml \
  --candidate-plan "$paired_dir/plan.json" --output "$paired_dir/output"
```

The command returns **2** at a planned condition switch, with `awaiting_runtime`.
For real GPU runs, stop/drain the previous runtime using the serving runbook, launch
and probe the next condition into a **new** serving directory, then supply that
fresh manifest. There is no automated restart and no interactive approval prompt:

```sh
uv run --locked python -m inference_lab.benchmark.runner \
  --resume PATH_TO_CAMPAIGN --runtime-manifest PATH_TO_NEW_SERVING/runtime_manifest.json
```

For the mock, no GPU restart/manifest is needed; repeat this until complete:

```sh
uv run --locked python -m inference_lab.benchmark.runner --resume "$paired_dir/output"
```

Each resume executes the next contiguous block for one condition, then pauses on
the next switch. Resume accepts only frozen inputs plus a fresh runtime manifest;
configuration/plan/output overrides are rejected. Session evidence hashes the Linux
boot ID, PID namespace and hostname, so exports cannot resume in another boot or
notebook session. Request deadlines must remain unchanged. Unavailable Linux session
evidence fails explicitly; it is not a remote-campaign fallback.

## Evidence and failure behavior

The root retains `campaign.json`, `report.md`, a lock file, and `inputs/<condition>/`
containing frozen experiment, plan, dataset, policy and available calibration files.
Every completed run's files are hashed; input and completed-result integrity are
rechecked before continuation. Point directories remain autonomous measurement
artifacts; copy the **whole campaign** to retain pairing and calibration provenance.
The source YAML/plan/policy/calibration files are not needed for a planned resume.

Point `drain.json` records before-warmup, after-warmup and after-measurement gauge
checks with explicit NOT_TESTED/VALIDATED/FAILED states. These are sampled idle
observations, not proof of GPU request abort. Real run artifacts include process
identity and the final server log, plus the existing environment/usage/raw evidence.
No prompts/responses or authorization headers are added to campaign reports.

SIGINT/task cancellation closes active HTTP streams and preserves incomplete root
and child artifacts. Failed/interrupted campaigns do not retry or resume. After an
abrupt kill, the last root checkpoint may still say running; it stays incomplete
and cannot silently execute pending work. Verify residual runtime work before
starting a fresh campaign. Existing artifacts are never overwritten by a new run.

CLI exits are **0** for complete, **1** for refusal/failure, **2** for a planned runtime
handoff, and **130** for SIGINT. These are execution statuses, not SLO gate verdicts.
All artifacts remain `published=false`; no performance certification, p99 precision,
TP/AWQ speedup or regression PASS follows. GPU campaigns and real cancellation/drain
remain **NOT_TESTED** locally; the CPU tests exercise orchestration only.
