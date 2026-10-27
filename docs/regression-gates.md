# Offline campaign regression gates — stage 10

Evaluate complete native campaign exports without starting a server, importing GPU
packages, downloading models, or accessing PostgreSQL. Keep the entire campaign
folder: individual result JSON files lack the frozen protocol and raw evidence.

## Run a comparison

Complete the [CPU paired campaign](campaigns.md) first, including its manual
handoffs. Then use its output directory for both conditions:

```sh
uv run --locked python -m inference_lab.benchmark.policy \
  --baseline "$paired_dir/output" --candidate "$paired_dir/output" \
  --output "$paired_dir/comparison.json"
```

The CLI selects `baseline`/`candidate` from a paired campaign, or `experiment` from
each of two single-condition campaign folders. A separate `--policy FILE` is
optional when a frozen policy exists; an explicit policy must match every recorded
policy snapshot. Thresholds come exclusively from canonical `RegressionPolicy`.

JSON goes to stdout; `--output` creates a new report outside both inputs. Existing
reports are never overwritten. Exit codes: **0 PASS**, **1 FAIL**, **2 INCONCLUSIVE
or INVALID**. Read `verdict` and structured `reasons` to distinguish exit code 2.
Inputs remain unchanged, and reports always retain `published=false`.

## Integrity and comparability

The evaluator checks schemas, hashes, frozen configuration/dataset/plan/policy,
run/point/repetition identities, observed token ranges, warmup, drain evidence,
and canonical metrics replayed from every raw request. Incomplete exports are
INVALID. Paired observations must be adjacent, sequential, and in the same
recorded session; offline evaluation does not require that session to remain live.

Differences include used GPUs/topology, effective runtime/backends, model revisions,
tokenizer/template, dataset/plan, target, generation, client/environment, dependencies,
and timing protocol. Every difference requires an exact `comparison.allowed_changes`
path. For a runtime upgrade, declare `runtime.version` and induced package changes
such as `dependencies.torch`; package names use lowercase and hyphens. Wildcards
are rejected. Metadata must remain stable within each condition's repetitions.
Unused GPUs do not participate in hardware comparability.

TP/AWQ changes require `kind: exploratory`. TP requires the same checkpoint and
dtype; isolated AWQ requires equal dtype/TP/used GPU count and hashed
`provenance.json` matching the qualification provenance structure (`source_model`,
`quantized_model`, architecture/vocabulary/template checks). Missing same-source
proof invalidates attribution. This does not imply a TP or AWQ speedup.

## Decisions and evidence limits

Each clause retains per-repetition values, sample counts, policy minima, paired
relative changes, observed min/max, and candidate run IDs that violate an SLO.
No percentiles are averaged. Relative degradation is `(candidate-baseline)/baseline`
for latency/error rate, `(baseline-candidate)/baseline` for throughput. A zero or
nonfinite relative baseline is INVALID.

For threshold T and noise margin B: all changes <= T-B give PASS; all > T+B give
FAIL; otherwise INCONCLUSIVE. Decimal boundary arithmetic avoids binary rounding
changing an exact threshold decision. SLOs apply to each candidate repetition:
all satisfy PASS, all violate FAIL, mixed INCONCLUSIVE. Clause priority is
INVALID, FAIL, INCONCLUSIVE, PASS. An entirely absent required metric is INVALID;
partial coverage, insufficient samples, or too few repetitions are INCONCLUSIVE.
Optional absent metrics do not invalidate other clauses.

Uncalibrated synthetic CPU campaigns demonstrate the mechanism only. Real gates
require at least three pairs per point, pre-measurement policy snapshots, verified
used hardware/runtime, and identical frozen calibration evidence. Unknown effective
linear backend or KV dtype prevents a real gate; current qualification does not
supply those observations. GPU gate execution remains **NOT_TESTED**.

A supported calibration artifact is a JSON attestation containing `schema_version`
`1.0`, `synthetic: false`, `policy_sha256` (SHA256 of canonical `encoded(policy)`),
`workload_id`, `baseline_run_sha256` (at least two independent baseline run IDs to
result hashes), `sample_counts` by metric, and nonempty `protocol`,
`volume_justification`, and `noise_justification`. Reference it through the canonical
policy's `calibration_ref` before capturing campaign inputs. Its baselines must be
distinct from comparison runs, and counts must meet policy minima. Missing or
unsupported justification is INCONCLUSIVE; changed identity/evidence is INVALID.
These are recorded attestations requiring reviewer inspection of the original
calibration runs, not automatic statistical certification. Empirical SLOs are not
population guarantees, and paired min/max is not a confidence interval.
