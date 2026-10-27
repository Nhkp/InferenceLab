"""Contract 1.0 comparability and conservative empirical decisions; no GPU execution."""

import math
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Literal, cast

from pydantic import JsonValue

from inference_lab.benchmark.plan import digest, encoded
from inference_lab.config.schemas import (
    ComparisonClause,
    ComparisonDifference,
    ComparisonReason,
    ComparisonResult,
    ExperimentResult,
    RegressionPolicy,
    Verdict,
)

PRIORITY: dict[Verdict, int] = {"PASS": 0, "INCONCLUSIVE": 1, "FAIL": 2, "INVALID": 3}


def combined(verdicts: Sequence[Verdict]) -> Verdict:
    return max(verdicts, key=PRIORITY.__getitem__, default="INVALID")


def used_topology(result: ExperimentResult) -> JsonValue:
    text = result.environment.topology
    if text is None:
        return None
    lines = re.sub(r"\x1b\[[0-9;]*m", "", text).splitlines()
    header = next(
        (line.split() for line in lines if line.strip().startswith("GPU0") and "CPU" in line), []
    )
    columns = [token for token in header if re.fullmatch(r"GPU\d+", token)]
    used = sorted(
        (g for g in result.environment.gpu_present if g.uuid in result.environment.gpu_used_uuids),
        key=lambda g: g.uuid,
    )
    rows = {
        line.split()[0]: line.split()[1:]
        for line in lines
        if line.split() and re.fullmatch(r"GPU\d+", line.split()[0]) and "CPU" not in line
    }
    try:
        return (
            [
                {
                    "links": [
                        rows[f"GPU{gpu.index}"][columns.index(f"GPU{other.index}")]
                        for other in used
                    ],
                    "cpu_numa_affinity": cast(JsonValue, rows[f"GPU{gpu.index}"][len(columns) :]),
                }
                for gpu in used
            ]
            if columns and used
            else None
        )
    except (KeyError, ValueError, IndexError):
        return None


def fingerprint(
    result: ExperimentResult, extra: Mapping[str, JsonValue] | None = None
) -> dict[str, JsonValue]:
    config = result.config.model_dump(mode="json")
    fields = {
        f"{section}.{key}": value
        for section in (
            "model",
            "runtime",
            "target",
            "workload",
            "generation",
            "warmup",
            "measurement",
        )
        for key, value in config[section].items()
    }
    env, workload, runtime = result.environment, result.workload, result.runtime_manifest
    used = sorted(
        (g for g in env.gpu_present if g.uuid in env.gpu_used_uuids), key=lambda g: g.uuid
    )
    fields.update(
        {
            "metric_contract_version": result.metric_contract_version,
            "runtime.version": env.vllm_version,
            "runtime.attention_backend": runtime.attention_backend,
            "runtime.linear_backend": runtime.linear_backend,
            "runtime.kv_cache_dtype_resolved": runtime.kv_cache_dtype_resolved,
            "dependencies.torch": env.torch_version,
            "dependencies.triton": env.triton_version,
            "environment.python": env.python,
            "environment.os": env.os,
            "environment.driver_supported_cuda": env.driver_supported_cuda,
            "environment.pytorch_cuda_runtime": env.pytorch_cuda_runtime,
            "environment.cuda_toolkit": env.cuda_toolkit,
            "client.git_sha": env.git_sha,
            "client.git_dirty": env.git_dirty,
            "hardware.cpu": env.cpu,
            "hardware.ram_bytes": env.ram_bytes,
            "hardware.driver": env.nvidia_driver,
            "hardware.gpu_used_uuids": sorted(env.gpu_used_uuids),
            "hardware.gpu_used_count": len(env.gpu_used_uuids),
            "hardware.gpu_names": [g.name for g in used],
            "hardware.gpu_indices": [g.index for g in used],
            "hardware.gpu_compute_capability": [list(g.compute_capability) for g in used],
            "hardware.gpu_vram_bytes": [g.vram_bytes for g in used],
            "hardware.topology": used_topology(result),
            "workload.dataset_sha256": workload.dataset_sha256,
            "workload.dataset_version": workload.dataset_version,
            "workload.plan_sha256": workload.plan_sha256,
            "workload.chat_template_sha256": workload.chat_template_sha256,
            "workload.template_parameters": workload.template_parameters,
            "workload.input_tokens": [workload.input_tokens_min, workload.input_tokens_max],
            "workload.output_tokens": [workload.output_tokens_min, workload.output_tokens_max],
            "protocol.concurrency": result.concurrency,
        }
    )
    if extra:
        fields.update(extra)
    return cast(dict[str, JsonValue], fields)


def evaluate_runs(
    baseline: Sequence[ExperimentResult],
    candidate: Sequence[ExperimentResult],
    policy: RegressionPolicy,
    *,
    metadata: Mapping[str, Mapping[str, JsonValue]] | None = None,
) -> ComparisonResult:
    """Each input is a complete condition; artifact replay is an additional CLI boundary."""
    baseline = sorted(baseline, key=lambda r: (r.point, r.repetition))
    candidate = sorted(candidate, key=lambda r: (r.point, r.repetition))
    reasons: list[ComparisonReason] = []
    verdicts: list[Verdict] = []
    differences: list[ComparisonDifference] = []
    clauses: list[ComparisonClause] = []
    all_runs = list(baseline) + list(candidate)
    modes = {r.synthetic for r in all_runs}

    def issue(verdict: Verdict, code: str, message: str, field: str | None = None) -> None:
        verdicts.append(verdict)
        reasons.append(ComparisonReason(code=code, message=message, field=field))

    def finish() -> ComparisonResult:
        return ComparisonResult(
            schema_version="1.0",
            metric_contract_version="1.0",
            baseline_run_ids=[r.run_id for r in baseline],
            candidate_run_ids=[r.run_id for r in candidate],
            comparison=policy.comparison,
            verdict=combined(verdicts),
            reasons=reasons,
            differences=differences,
            clauses=clauses,
            policy_sha256=digest(encoded(policy)),
            synthetic=next(iter(modes)) if len(modes) == 1 else None,
            limitations=[
                "Empirical SLOs and observed paired min/max are not population guarantees "
                "or confidence intervals.",
                "Client percentiles remain conditional on successes; attempts/errors "
                "and metric coverage are preserved.",
                "This evaluation does not publish or mutate its input benchmark artifacts.",
            ]
            + (
                ["Synthetic CPU fixtures; no GPU performance certification."]
                if modes == {True}
                else []
            ),
        )

    if not baseline or not candidate:
        issue("INVALID", "missing_condition", "Both conditions require observed runs")
        return finish()
    if len({r.run_id for r in all_runs}) != len(all_runs) or len(modes) != 1:
        issue(
            "INVALID",
            "identity_or_mode_mismatch",
            "Run IDs must be distinct and synthetic/real modes must match",
        )
        return finish()
    for runs in (baseline, candidate):
        config = runs[0].config
        wanted = {
            (p, r)
            for p in range(len(config.workload.concurrency))
            for r in range(config.measurement.repetitions)
        }
        actual = {(r.point, r.repetition) for r in runs}
        if actual != wanted or len(actual) != len(runs):
            issue(
                "INVALID",
                "incomplete_condition",
                "Condition does not cover its declared points and repetitions",
            )
        if any(r.config != config for r in runs):
            issue(
                "INVALID", "mixed_configuration", "One condition contains multiple configurations"
            )
    if {(r.point, r.repetition) for r in baseline} != {(r.point, r.repetition) for r in candidate}:
        issue(
            "INVALID",
            "unpaired_repetitions",
            "Conditions have different point/repetition identities",
        )
    for r in all_runs:
        if any(
            m.value is not None and (not math.isfinite(m.value) or m.value < 0)
            for m in r.metrics.values()
        ):
            issue("INVALID", "nonfinite_metric", "Metrics must be finite and nonnegative", r.run_id)
        if (
            not r.complete
            or not r.workload.conformant
            or r.counters.attempts != r.config.measurement.requests_per_run
        ):
            issue(
                "INVALID",
                "incomplete_or_nonconformant",
                "A run is incomplete or its workload is unverified",
                r.run_id,
            )
        if r.workload.dataset_name != policy.workload_id:
            issue(
                "INVALID",
                "policy_workload_mismatch",
                "Policy is bound to another workload",
                r.run_id,
            )
        if r.policy is not None and r.policy != policy:
            issue(
                "INVALID",
                "policy_snapshot_mismatch",
                "Policy differs from its pre-measurement snapshot",
                r.run_id,
            )
        if not r.synthetic:
            critical = (
                r.runtime_manifest.resolved,
                r.runtime_manifest.attention_backend,
                r.runtime_manifest.linear_backend,
                r.runtime_manifest.kv_cache_dtype_resolved,
                used_topology(r),
                r.environment.nvidia_driver,
                r.environment.vllm_version,
                r.environment.pytorch_cuda_runtime,
                r.environment.torch_version,
                r.environment.triton_version,
                r.environment.git_sha,
            )
            if (
                len(r.environment.gpu_used_uuids) != r.config.runtime.tensor_parallel_size
                or r.qualification_id is None
                or any(value is None for value in critical)
            ):
                issue(
                    "INVALID",
                    "unverified_runtime",
                    "Real comparisons require observed used hardware and critical runtime metadata",
                    r.run_id,
                )
    if "INVALID" in verdicts:
        return finish()
    allowed = set(policy.comparison.allowed_changes)
    seen = set()
    for runs in (baseline, candidate):
        for point in {r.point for r in runs}:
            group = [r for r in runs if r.point == point]
            fingerprints = [fingerprint(r, (metadata or {}).get(r.run_id)) for r in group]
            if any(value != fingerprints[0] for value in fingerprints[1:]):
                issue(
                    "INVALID",
                    "unstable_condition",
                    "Effective metadata changed between repetitions",
                    f"point.{point}",
                )
    for left, right in zip(baseline, candidate, strict=True):
        a = fingerprint(left, (metadata or {}).get(left.run_id))
        b = fingerprint(right, (metadata or {}).get(right.run_id))
        for path in sorted(a.keys() | b.keys()):
            before, after = a.get(path), b.get(path)
            if before != after:
                key = (path, json_key(before), json_key(after))
                if key not in seen:
                    differences.append(
                        ComparisonDifference(path=path, baseline=before, candidate=after)
                    )
                    seen.add(key)
                if path not in allowed:
                    issue(
                        "INVALID",
                        "undeclared_difference",
                        "Critical difference is not explicitly allowed",
                        path,
                    )
        if policy.comparison.kind == "regression":
            if (
                left.config.runtime.quantization != right.config.runtime.quantization
                or left.config.runtime.tensor_parallel_size
                != right.config.runtime.tensor_parallel_size
            ):
                issue(
                    "INVALID",
                    "exploratory_required",
                    "AWQ/TP comparisons cannot be standard regression gates",
                )
            if a.get("protocol.request_plan_sha256", left.workload.plan_sha256) != b.get(
                "protocol.request_plan_sha256", right.workload.plan_sha256
            ):
                issue(
                    "INVALID",
                    "unmatched_request_plans",
                    "Regression requires matched request plans",
                )
        else:
            quant = left.config.runtime.quantization != right.config.runtime.quantization
            tp = (
                left.config.runtime.tensor_parallel_size
                != right.config.runtime.tensor_parallel_size
            )
            if quant and (
                tp
                or left.config.runtime.dtype != right.config.runtime.dtype
                or len(left.environment.gpu_used_uuids) != len(right.environment.gpu_used_uuids)
            ):
                issue(
                    "INVALID",
                    "awq_resource_mismatch",
                    "Isolated AWQ comparison requires equal dtype, TP and used GPU count",
                )
            if quant and (
                a.get("model.source_repo_id") is None
                or a.get("model.source_repo_id") != b.get("model.source_repo_id")
                or a.get("model.source_revision") is None
                or a.get("model.source_revision") != b.get("model.source_revision")
            ):
                issue(
                    "INVALID",
                    "awq_source_unverified",
                    "Same-source AWQ provenance must be observed",
                )
            if tp and (
                left.config.model != right.config.model
                or left.config.runtime.dtype != right.config.runtime.dtype
            ):
                issue(
                    "INVALID",
                    "tp_checkpoint_mismatch",
                    "TP comparison requires the same checkpoint/tokenizer and dtype",
                )
    if "INVALID" in verdicts:
        return finish()
    for point in sorted({r.point for r in baseline}):
        point_baseline = [r for r in baseline if r.point == point]
        point_candidate = [r for r in candidate if r.point == point]
        sufficient: dict[str, tuple[Verdict | None, str]] = {}
        for metric in policy.required_metrics:
            baseline_metrics = [r.metrics[metric] for r in point_baseline]
            candidate_metrics = [r.metrics[metric] for r in point_candidate]
            missing = all(m.value is None for m in baseline_metrics) or all(
                m.value is None for m in candidate_metrics
            )
            low = any(
                m.value is None or m.sample_count < policy.min_samples_per_metric[metric]
                for m in baseline_metrics + candidate_metrics
            )
            status: Verdict | None = None
            reason = "Available paired repetitions satisfy sample minima"
            if missing:
                status, reason = (
                    "INVALID",
                    "Required metric is entirely absent in a condition",
                )
            elif low or len(point_baseline) < policy.min_repetitions:
                status, reason = (
                    "INCONCLUSIVE",
                    "Metric coverage or repetition count is below policy minima",
                )
            sufficient[metric] = (status, reason)
            if status:
                issue(
                    status,
                    "missing_required_metric"
                    if status == "INVALID"
                    else "insufficient_observations",
                    reason,
                    f"point.{point}.{metric}",
                )
        for kind, limits in (("slo", policy.slo), ("regression", policy.regression)):
            for metric in limits:
                base_values = [r.metrics[metric].value for r in point_baseline]
                candidate_values = [r.metrics[metric].value for r in point_candidate]
                status, reason = sufficient[metric]
                changes = []
                violations = []
                threshold = (
                    policy.slo[metric].max
                    if kind == "slo"
                    else policy.regression[metric].max_relative_degradation
                )
                margin = None if kind == "slo" else policy.regression[metric].noise_margin
                if kind == "slo":
                    violations = [
                        r.run_id
                        for r in point_candidate
                        if (value := r.metrics[metric].value) is not None and value > threshold
                    ]
                    if status is None:
                        status = (
                            "PASS"
                            if not violations
                            else "FAIL"
                            if len(violations) == len(point_candidate)
                            else "INCONCLUSIVE"
                        )
                        reason = (
                            "All candidate repetitions satisfy the empirical SLO"
                            if status == "PASS"
                            else "Candidate SLO violations are preserved per repetition"
                        )
                else:
                    for baseline_value, candidate_value in zip(
                        base_values, candidate_values, strict=True
                    ):
                        if baseline_value is None or candidate_value is None:
                            continue
                        if baseline_value <= 0:
                            status, reason = (
                                "INVALID",
                                "Relative comparison needs positive finite baseline values",
                            )
                            break
                        aa, bb = Decimal(str(baseline_value)), Decimal(str(candidate_value))
                        change = (aa - bb) / aa if "per_second" in metric else (bb - aa) / aa
                        value = float(change)
                        if not math.isfinite(value):
                            status, reason = (
                                "INVALID",
                                "Relative change is not a finite number",
                            )
                            break
                        changes.append(change)
                    if status is None:
                        lower = Decimal(str(threshold)) - Decimal(str(margin))
                        upper = Decimal(str(threshold)) + Decimal(str(margin))
                        status = (
                            "PASS"
                            if all(change <= lower for change in changes)
                            else "FAIL"
                            if all(change > upper for change in changes)
                            else "INCONCLUSIVE"
                        )
                        reason = (
                            "Paired changes use the noise-band rule; min/max is observed dispersion"
                        )
                assert status is not None
                clause = ComparisonClause(
                    point=point,
                    metric=metric,
                    kind=cast(Literal["slo", "regression"], kind),
                    verdict=status,
                    threshold=threshold,
                    noise_margin=margin,
                    repetitions=[r.repetition for r in point_baseline],
                    required_repetitions=policy.min_repetitions,
                    min_samples=policy.min_samples_per_metric[metric],
                    baseline_values=base_values,
                    candidate_values=candidate_values,
                    baseline_sample_counts=[r.metrics[metric].sample_count for r in point_baseline],
                    candidate_sample_counts=[
                        r.metrics[metric].sample_count for r in point_candidate
                    ],
                    relative_changes=[float(v) for v in changes],
                    min_relative_change=float(min(changes)) if changes else None,
                    max_relative_change=float(max(changes)) if changes else None,
                    violating_run_ids=violations,
                    reason=reason,
                )
                clauses.append(clause)
                issue(status, f"{kind}_{status.lower()}", reason, f"point.{point}.{metric}")
    return finish()


def json_key(value: JsonValue) -> str:
    import json

    return json.dumps(value, sort_keys=True)
