import json
import subprocess
import sys

import pytest
from test_comparison_artifacts import complete_campaign, fixture_timings

from inference_lab.benchmark.comparison_artifacts import read_condition
from inference_lab.benchmark.plan import digest, encoded
from inference_lab.benchmark.policy import compare_campaigns, evidence_checks
from inference_lab.config.schemas import ComparisonResult

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("candidate", "verdict", "exit_code"),
    [
        (105, "PASS", 0),
        (110, "INCONCLUSIVE", 2),
        (120, "FAIL", 1),
    ],
)
async def test_native_campaign_cli_decisions_and_read_only_inputs(
    tmp_path, candidate, verdict, exit_code
):
    root = await complete_campaign(tmp_path)
    fixture_timings(root, candidate=candidate)
    before = {p.relative_to(root): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "comparison.json"
    command = [
        sys.executable,
        "-m",
        "inference_lab.benchmark.policy",
        "--baseline",
        str(root),
        "--candidate",
        str(root),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == exit_code, completed.stderr
    result = ComparisonResult.model_validate_json(completed.stdout)
    assert result.verdict == verdict and result.synthetic and not result.published
    assert result.clauses and result.policy_sha256
    assert json.loads(output.read_text()) == json.loads(completed.stdout)
    assert before == {
        p.relative_to(root): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()
    }
    # Existing evidence/reports are never overwritten, even by a second evaluation.
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode == 2 and "never overwritten" in repeated.stderr


async def test_invalid_files_and_explicit_changed_policy_are_structured(tmp_path):
    root = await complete_campaign(tmp_path)
    fixture_timings(root)
    frozen = root / "inputs/baseline/policy.json"
    policy = json.loads(frozen.read_text())
    policy["slo"]["client_ttft_p95_ms"]["max"] = 2
    path = tmp_path / "changed-policy.json"
    path.write_text(json.dumps(policy))
    result = compare_campaigns(root, root, path)
    assert result.verdict == "INVALID"
    assert any(r.code == "policy_snapshot_mismatch" for r in result.reasons)
    path.write_text("slo: [bad")
    assert compare_campaigns(root, root, path).verdict == "INVALID"
    path.write_text("api_key: private-fixture-text")
    assert "private-fixture-text" not in compare_campaigns(root, root, path).model_dump_json()
    (root / "campaign.json").write_text("{}")
    command = [
        sys.executable,
        "-m",
        "inference_lab.benchmark.policy",
        "--baseline",
        str(root),
        "--candidate",
        str(root),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 2
    assert json.loads(completed.stdout)["verdict"] == "INVALID"
    assert "Traceback" not in completed.stderr


async def test_session_order_and_output_inside_input_refused(tmp_path):
    root = await complete_campaign(tmp_path)
    fixture_timings(root)
    b, c = read_condition(root, role="baseline"), read_condition(root, role="candidate")
    p = b.results[0].policy
    assert evidence_checks(b, c, p) == []
    c.campaign.session_id = "0" * 64
    assert any(reason.code == "different_sessions" for _, reason in evidence_checks(b, c, p))
    c.campaign.session_id = b.campaign.session_id
    first = c.results[0].run_id
    c.intervals[first] = b.intervals[b.results[0].run_id]
    assert any(reason.code == "overlapping_runs" for _, reason in evidence_checks(b, c, p))
    command = [
        sys.executable,
        "-m",
        "inference_lab.benchmark.policy",
        "--baseline",
        str(root),
        "--candidate",
        str(root),
        "--output",
        str(root / "comparison.json"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 2 and not (root / "comparison.json").exists()


async def test_calibration_is_required_for_real_gate_and_bound_to_frozen_policy(tmp_path):
    root = await complete_campaign(tmp_path)
    fixture_timings(root)
    b, c = read_condition(root, role="baseline"), read_condition(root, role="candidate")
    p = b.results[0].policy
    # Exercise the evidence guard only: this does not construct or certify real GPU runs.
    b.campaign.synthetic = c.campaign.synthetic = False
    assert any(
        v == "INCONCLUSIVE" and r.code == "missing_calibration" for v, r in evidence_checks(b, c, p)
    )
    p = p.model_copy(update={"calibration_ref": "synthetic-test-attestation.json"})
    calibration = {
        "schema_version": "1.0",
        "synthetic": False,
        "policy_sha256": digest(encoded(p)),
        "workload_id": p.workload_id,
        "baseline_run_sha256": {"calibration-a": "a" * 64, "calibration-b": "b" * 64},
        "sample_counts": {"client_ttft_p95_ms": 1},
        "protocol": "Invented baseline CPU protocol",
        "volume_justification": "Invented test justification, never statistical evidence",
        "noise_justification": "Invented test budget justification",
    }
    b.calibration = c.calibration = json.dumps(calibration).encode()
    assert evidence_checks(b, c, p) == []
    calibration["baseline_run_sha256"][b.results[0].run_id] = "c" * 64
    b.calibration = c.calibration = json.dumps(calibration).encode()
    assert any(
        v == "INVALID" and r.code == "nonindependent_calibration"
        for v, r in evidence_checks(b, c, p)
    )
    b.calibration = c.calibration = b"unstructured notes"
    assert any(v == "INCONCLUSIVE" for v, _ in evidence_checks(b, c, p))
