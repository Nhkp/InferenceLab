"""Invented CPU values exercise the contract; none are hardware measurements."""

import json
from pathlib import Path

import pytest

from inference_lab.benchmark.gates import combined, evaluate_runs, fingerprint, used_topology
from inference_lab.config.schemas import (
    ComparisonConfig,
    ExperimentResult,
    RegressionPolicy,
)

pytestmark = pytest.mark.unit
DATA = Path(__file__).resolve().parents[1] / "data"
METRIC = "client_ttft_p95_ms"


def policy(**changes):
    data = json.loads((DATA / "synthetic-policy.json").read_text())
    data.update(changes)
    return RegressionPolicy.model_validate(data)


def runs(values, *, role, metric=METRIC, samples=1):
    data = json.loads((DATA / "synthetic-result.json").read_text())
    data["config"]["measurement"]["repetitions"] = len(values)
    results = []
    for index, value in enumerate(values):
        current = json.loads(json.dumps(data))
        current.update(run_id=f"{role}-{index}", repetition=index)
        current["metrics"][metric].update(
            value=value,
            sample_count=samples if value is not None else 0,
            reason=None if value is not None else "Synthetic missing observation",
        )
        results.append(ExperimentResult.model_validate(current))
    return results


@pytest.mark.parametrize(
    ("values", "verdict"),
    [
        ([108, 108, 108], "PASS"),
        ([112, 112, 112], "INCONCLUSIVE"),
        ([112.01] * 3, "FAIL"),
        ([105, 120, 105], "INCONCLUSIVE"),
        ([90] * 3, "PASS"),
    ],
)
def test_noise_band_exact_boundaries_and_mixed_repetitions(values, verdict):
    result = evaluate_runs(runs([100] * 3, role="b"), runs(values, role="c"), policy())
    assert result.verdict == verdict
    assert result.synthetic and not result.published
    clause = next(c for c in result.clauses if c.kind == "regression")
    assert clause.relative_changes == pytest.approx([(v - 100) / 100 for v in values])
    assert clause.min_relative_change == min(clause.relative_changes)
    assert clause.max_relative_change == max(clause.relative_changes)
    assert clause.baseline_sample_counts == [1, 1, 1]


@pytest.mark.parametrize(
    ("values", "verdict", "violations"),
    [
        ([1000] * 3, "PASS", []),
        ([1001] * 3, "FAIL", ["c-0", "c-1", "c-2"]),
        ([1000, 1001, 1000], "INCONCLUSIVE", ["c-1"]),
    ],
)
def test_slo_candidate_repetitions_and_raw_violation_ids(values, verdict, violations):
    result = evaluate_runs(runs([100] * 3, role="b"), runs(values, role="c"), policy(regression={}))
    assert result.verdict == verdict
    assert result.clauses[0].violating_run_ids == violations


@pytest.mark.parametrize(
    ("baseline", "candidate", "verdict"),
    [
        ([None] * 3, [100] * 3, "INVALID"),
        ([100] * 3, [None] * 3, "INVALID"),
        ([100, None, 100], [105] * 3, "INCONCLUSIVE"),
        ([0] * 3, [1] * 3, "INVALID"),
        ([100] * 2, [105] * 2, "INCONCLUSIVE"),
    ],
)
def test_absence_partial_coverage_zero_and_short_runs(baseline, candidate, verdict):
    assert (
        evaluate_runs(runs(baseline, role="b"), runs(candidate, role="c"), policy()).verdict
        == verdict
    )


def test_small_n_and_nonfinite_are_never_pass():
    baseline, candidate = runs([100] * 3, role="b"), runs([105] * 3, role="c")
    assert (
        evaluate_runs(baseline, candidate, policy(min_samples_per_metric={METRIC: 2})).verdict
        == "INCONCLUSIVE"
    )
    baseline[0].metrics[METRIC].value = float("nan")
    assert evaluate_runs(baseline, candidate, policy()).verdict == "INVALID"


def test_throughput_direction_and_decimal_boundary():
    metric = "successful_requests_per_second"
    p = policy(
        required_metrics=[metric],
        min_samples_per_metric={metric: 1},
        slo={},
        regression={metric: {"max_relative_degradation": 0.1, "noise_margin": 0.02}},
    )
    result = evaluate_runs(
        runs([100] * 3, role="b", metric=metric), runs([92] * 3, role="c", metric=metric), p
    )
    assert result.verdict == "PASS" and result.clauses[0].relative_changes == [0.08] * 3
    assert (
        evaluate_runs(
            runs([100] * 3, role="b", metric=metric), runs([87] * 3, role="c", metric=metric), p
        ).verdict
        == "FAIL"
    )
    p = policy(regression={METRIC: {"max_relative_degradation": 0.3, "noise_margin": 0.1}})
    assert evaluate_runs(runs([100] * 3, role="b"), runs([120] * 3, role="c"), p).verdict == "PASS"


def test_explicit_runtime_version_and_induced_dependencies():
    baseline, candidate = runs([100] * 3, role="b"), runs([105] * 3, role="c")
    for run in baseline + candidate:
        run.environment.vllm_version = (
            "synthetic-a" if run.run_id.startswith("b") else "synthetic-b"
        )
    result = evaluate_runs(baseline, candidate, policy())
    assert result.verdict == "PASS"
    assert [d.path for d in result.differences] == ["runtime.version"]
    metadata = {
        r.run_id: {"dependencies.torch": "a" if r.run_id.startswith("b") else "b"}
        for r in baseline + candidate
    }
    assert evaluate_runs(baseline, candidate, policy(), metadata=metadata).verdict == "INVALID"
    p = policy(
        comparison={
            "kind": "regression",
            "allowed_changes": ["runtime.version", "dependencies.torch"],
        }
    )
    assert evaluate_runs(baseline, candidate, p, metadata=metadata).verdict == "PASS"
    for invalid in ("dependencies.*", "dependencies.Torch", "runtime.*"):
        with pytest.raises(ValueError):
            ComparisonConfig(kind="regression", allowed_changes=[invalid])


@pytest.mark.parametrize(
    "change", ["missing_pair", "duplicate", "incomplete", "plan", "dtype", "tp", "unstable"]
)
def test_unmatched_or_unverified_data_are_invalid(change):
    baseline, candidate = runs([100] * 3, role="b"), runs([105] * 3, role="c")
    if change == "missing_pair":
        candidate.pop()
    elif change == "duplicate":
        candidate[1].run_id = candidate[0].run_id
    elif change == "incomplete":
        candidate[0].complete = False
    elif change == "plan":
        for r in candidate:
            r.workload.plan_sha256 = "0" * 64
    elif change == "dtype":
        for r in candidate:
            r.config.runtime.dtype = "bfloat16"
    elif change == "tp":
        for r in candidate:
            r.config.runtime.tensor_parallel_size = 2
    else:
        candidate[0].environment.cpu = "different"
    assert evaluate_runs(baseline, candidate, policy()).verdict == "INVALID"


def test_precedence_optional_absence_and_required_only_coverage():
    assert combined(["PASS", "INCONCLUSIVE", "FAIL", "INVALID"]) == "INVALID"
    assert combined(["INCONCLUSIVE", "FAIL"]) == "FAIL"
    baseline, candidate = runs([100] * 3, role="b"), runs([105] * 3, role="c")
    assert evaluate_runs(baseline, candidate, policy()).verdict == "PASS"  # Optional p99 absent.
    p = policy(
        required_metrics=[METRIC, "client_e2e_p95_ms"],
        min_samples_per_metric={METRIC: 1, "client_e2e_p95_ms": 1},
    )
    assert evaluate_runs(baseline, candidate, p).verdict == "INVALID"


def test_only_used_gpu_topology_participates():
    run = runs([100], role="b")[0]
    # Invented metadata tests projection only; these are not validated hardware runs.
    from inference_lab.config.schemas import GPUDevice

    run.environment.gpu_present = [
        GPUDevice(
            index=i, uuid=f"fake-{i}", name="synthetic", vram_bytes=1, compute_capability=(7, 5)
        )
        for i in (0, 1)
    ]
    run.environment.gpu_used_uuids = ["fake-0"]
    run.environment.topology = (
        "GPU0 GPU1 CPU Affinity NUMA Affinity\nGPU0 X PHB 0-3 0\nGPU1 PHB X 0-3 0\n"
    )
    before = fingerprint(run)
    run.environment.gpu_present[1].vram_bytes = 1000
    run.environment.topology = run.environment.topology.replace("PHB", "SYS")
    assert fingerprint(run) == before
    assert used_topology(run) == [{"links": ["X"], "cpu_numa_affinity": ["0-3", "0"]}]
    run.environment.gpu_used_uuids = ["fake-1"]
    assert fingerprint(run) != before


def test_real_unknown_runtime_and_exploratory_attribution_guards():
    baseline, candidate = runs([100] * 3, role="b"), runs([105] * 3, role="c")
    for r in baseline + candidate:
        r.synthetic = False  # Guard exercise only, with deliberately missing hardware evidence.
    assert evaluate_runs(baseline, candidate, policy()).verdict == "INVALID"
    baseline, candidate = runs([100] * 3, role="b"), runs([105] * 3, role="c")
    for r in candidate:
        r.config.runtime.quantization = "awq"
    p = policy(comparison={"kind": "regression", "allowed_changes": ["runtime.quantization"]})
    assert evaluate_runs(baseline, candidate, p).verdict == "INVALID"
    p = policy(comparison={"kind": "exploratory", "allowed_changes": ["runtime.quantization"]})
    assert evaluate_runs(baseline, candidate, p).verdict == "INVALID"  # Missing source proof.
    metadata = {
        r.run_id: {"model.source_repo_id": "synthetic-source", "model.source_revision": "a" * 40}
        for r in baseline + candidate
    }
    assert evaluate_runs(baseline, candidate, p, metadata=metadata).verdict == "PASS"
    for r in candidate:
        r.config.runtime.tensor_parallel_size = 2
    p = policy(
        comparison={
            "kind": "exploratory",
            "allowed_changes": ["runtime.quantization", "runtime.tensor_parallel_size"],
        }
    )
    assert evaluate_runs(baseline, candidate, p, metadata=metadata).verdict == "INVALID"
