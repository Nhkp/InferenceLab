"""Read-only offline campaign gates: 0 PASS, 1 FAIL, 2 INCONCLUSIVE/INVALID."""

import argparse
import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from inference_lab.benchmark.comparison_artifacts import (
    CalibrationEvidence,
    ConditionArtifacts,
    read_condition,
)
from inference_lab.benchmark.gates import combined, evaluate_runs
from inference_lab.benchmark.plan import digest, encoded
from inference_lab.config.io import load_policy
from inference_lab.config.schemas import (
    ComparisonReason,
    ComparisonResult,
    RegressionPolicy,
    Verdict,
)


def evidence_checks(
    baseline: ConditionArtifacts, candidate: ConditionArtifacts, policy: RegressionPolicy
) -> list[tuple[Verdict, ComparisonReason]]:
    issues: list[tuple[Verdict, ComparisonReason]] = []

    def issue(verdict: Verdict, code: str, message: str) -> None:
        issues.append((verdict, ComparisonReason(code=code, message=message)))

    if baseline.campaign.session_id != candidate.campaign.session_id:
        issue("INVALID", "different_sessions", "Paired campaigns require the same recorded session")
    runs = [(r, baseline) for r in baseline.results] + [(r, candidate) for r in candidate.results]
    ordered = sorted(runs, key=lambda item: item[1].intervals[item[0].run_id][0])
    for (previous_run, previous), (run, current) in zip(ordered, ordered[1:], strict=False):
        if previous.intervals[previous_run.run_id][1] > current.intervals[run.run_id][0]:
            issue("INVALID", "overlapping_runs", "Paired runs must execute sequentially")
    identities = [(r.point, r.repetition) for r, _ in ordered]
    if any(identities[i] != identities[i + 1] for i in range(0, len(identities) - 1, 2)):
        issue("INVALID", "unpaired_order", "Baseline/candidate repetitions must be adjacent pairs")
    if baseline.campaign.synthetic and candidate.campaign.synthetic:
        return issues
    if (
        any(
            sum(r.point == point for r in side.results) < 3
            for side in (baseline, candidate)
            for point in {r.point for r in side.results}
        )
        or policy.min_repetitions < 3
    ):
        issue(
            "INCONCLUSIVE",
            "insufficient_real_repetitions",
            "Real gates require at least three pairs per point",
        )
    if any(r.policy is None for r, _ in runs):
        issue(
            "INCONCLUSIVE", "unfrozen_policy", "Real gates require pre-measurement policy snapshots"
        )
    if policy.calibration_ref is None:
        issue(
            "INCONCLUSIVE",
            "unreferenced_calibration",
            "Real policies must reference their frozen calibration artifact",
        )
    if baseline.calibration is None or candidate.calibration is None:
        issue(
            "INCONCLUSIVE",
            "missing_calibration",
            "Volume and noise require frozen independent baseline calibration",
        )
        return issues
    if baseline.calibration != candidate.calibration:
        issue("INVALID", "changed_calibration", "Conditions contain different calibration evidence")
        return issues
    try:
        calibration = CalibrationEvidence.model_validate_json(baseline.calibration)
    except ValidationError:
        issue(
            "INCONCLUSIVE",
            "unverified_calibration",
            "Calibration lacks supported volume/noise justification and baseline provenance",
        )
        return issues
    if (
        calibration.synthetic
        or calibration.policy_sha256 != digest(encoded(policy))
        or calibration.workload_id != policy.workload_id
    ):
        issue(
            "INVALID", "calibration_identity", "Calibration must bind this real policy and workload"
        )
    ids = {r.run_id for r, _ in runs}
    if ids & calibration.baseline_run_sha256.keys() or (
        baseline.result_hashes | candidate.result_hashes
    ) & set(calibration.baseline_run_sha256.values()):
        issue(
            "INVALID",
            "nonindependent_calibration",
            "Calibration baselines must be independent of the comparison runs",
        )
    if any(
        calibration.sample_counts.get(metric, 0) < minimum
        for metric, minimum in policy.min_samples_per_metric.items()
    ):
        issue(
            "INCONCLUSIVE",
            "calibration_volume",
            "Calibration support is below policy sample minima",
        )
    return issues


def compare_campaigns(
    baseline_root: Path, candidate_root: Path, policy_path: Path | None = None
) -> ComparisonResult:
    policy = None
    try:
        if policy_path is not None:
            policy = load_policy(policy_path)
        baseline = read_condition(baseline_root, role="baseline")
        candidate = read_condition(candidate_root, role="candidate")
        if policy is None:
            policy = baseline.results[0].policy
        if policy is None:
            raise ValueError("A canonical policy is required")
        result = evaluate_runs(
            baseline.results,
            candidate.results,
            policy,
            metadata=baseline.metadata | candidate.metadata,
        )
        issues = evidence_checks(baseline, candidate, policy)
        result.verdict = combined([result.verdict] + [verdict for verdict, _ in issues])
        result.reasons.extend(reason for _, reason in issues)
        if result.synthetic is False:
            result.limitations.append(
                "Calibration justifications and baseline hashes are recorded attestations; "
                "this evaluator does not establish their statistical adequacy."
            )
        return ComparisonResult.model_validate(result.model_dump())
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        yaml.YAMLError,
    ) as error:
        # Validation exceptions can contain payloads; expose their category, never raw inputs.
        return ComparisonResult(
            schema_version="1.0",
            metric_contract_version="1.0",
            baseline_run_ids=[],
            candidate_run_ids=[],
            comparison=policy.comparison if policy else None,
            verdict="INVALID",
            reasons=[
                ComparisonReason(
                    code="invalid_artifacts",
                    message=f"Artifact integrity check failed ({type(error).__name__})",
                )
            ],
            differences=[],
            policy_sha256=digest(encoded(policy)) if policy else None,
            limitations=["No gate decision is possible from invalid campaign evidence."],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Create a new JSON report outside both input campaigns"
    )
    args = parser.parse_args()
    if args.output is not None and any(
        args.output.resolve().is_relative_to(root.resolve())
        for root in (args.baseline, args.candidate)
    ):
        parser.error("Report output must be outside the input campaigns")
    result = compare_campaigns(args.baseline, args.candidate, args.policy)
    rendered = json.dumps(result.model_dump(mode="json"), indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        try:
            with args.output.open("x") as output:
                output.write(rendered)
        except OSError:
            parser.error("Cannot create report; existing files are never overwritten")
    print(rendered, end="")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
