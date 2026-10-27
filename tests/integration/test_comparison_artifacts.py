"""Native CPU campaign exports; rewritten timings below are invented fixture data."""

import json

import pytest
from test_campaigns import inputs
from test_streaming_client import DONE, USAGE, event, server
from test_workload import scheduler

from inference_lab.benchmark.campaign import Campaign
from inference_lab.benchmark.comparison_artifacts import dependencies, read_condition
from inference_lab.benchmark.metrics import aggregate
from inference_lab.benchmark.plan import encoded
from inference_lab.benchmark.runner import capture_run, resume_campaign, run_campaign
from inference_lab.config.schemas import ExperimentResult, RequestMeasurement

pytestmark = pytest.mark.integration


async def complete_campaign(tmp_path):
    async def source():
        yield event("synthetic")
        yield event(" tail", "length") + USAGE + DONE

    root = tmp_path / "campaign"
    async with server(source, metrics=lambda: scheduler()) as url:
        exp, plan = inputs(tmp_path, url, paired=True, repetitions=3, points=(1,), count=1)
        state = await run_campaign(
            exp, plan, root, campaign_id="gate-fixture", candidate=exp, candidate_plan=plan
        )
        while state.status == "awaiting_runtime":
            state = await resume_campaign(root)
    assert state.complete
    return root


def fixture_timings(root, *, baseline=100, candidate=105):
    """Freeze deterministic synthetic times, then rebuild canonical metrics and hashes."""
    state = Campaign.model_validate_json((root / "campaign.json").read_bytes())
    for slot, entry in enumerate(state.schedule):
        folder = root / entry.run_id
        for filename, offset, delay in (
            ("raw/warmup.jsonl", 0, 1),
            (
                "raw/requests.jsonl",
                50_000_000,
                baseline if entry.condition == "baseline" else candidate,
            ),
        ):
            records = []
            for line in (folder / filename).read_text().splitlines():
                record = RequestMeasurement.model_validate_json(line)
                start = 1_000_000_000 * (slot + 1) + offset
                record.request_start_ns = start
                record.first_content_ns = start + int(delay * 1_000_000)
                record.last_content_ns = record.first_content_ns + 10_000_000
                record.terminal_ns = record.last_content_ns + 10_000_000
                records.append(record)
            (folder / filename).write_bytes(b"".join(encoded(record) for record in records))
        result = ExperimentResult.model_validate_json((folder / "result.json").read_bytes())
        result.metrics = aggregate(records)
        (folder / "result.json").write_bytes(encoded(result))
        capture_run(root, entry)
    (root / "campaign.json").write_bytes(encoded(state))


async def test_replay_export_offline_without_live_server_or_source_files(tmp_path):
    root = await complete_campaign(tmp_path)
    fixture_timings(root)
    (tmp_path / "experiment.yaml").unlink()
    (tmp_path / "plan.json").unlink()
    (tmp_path / "policy.yaml").unlink()
    baseline = read_condition(root, role="baseline")
    candidate = read_condition(root, role="candidate")
    assert len(baseline.results) == len(candidate.results) == 3
    assert all(r.metrics["client_ttft_p95_ms"].value == 100 for r in baseline.results)
    assert all(r.metrics["client_ttft_p95_ms"].value == 105 for r in candidate.results)


@pytest.mark.parametrize(
    "corruption",
    [
        "hash",
        "metrics",
        "raw_identity",
        "schema",
        "warmup",
        "drain",
        "incomplete",
        "environment",
        "workload",
    ],
)
async def test_reject_corrupt_or_semantically_inconsistent_exports(tmp_path, corruption):
    root = await complete_campaign(tmp_path)
    fixture_timings(root)
    state = Campaign.model_validate_json((root / "campaign.json").read_bytes())
    entry = state.schedule[0]
    folder = root / entry.run_id
    if corruption == "hash":
        (folder / "raw/requests.jsonl").write_text("changed")
    elif corruption in ("metrics", "schema", "workload"):
        data = json.loads((folder / "result.json").read_text())
        if corruption == "metrics":
            data["metrics"]["client_ttft_p95_ms"]["value"] = 1
        elif corruption == "workload":
            data["workload"]["dataset_version"] = "wrong-version"
        else:
            data["metric_contract_version"] = "2.0"
        (folder / "result.json").write_text(json.dumps(data))
    elif corruption in ("raw_identity", "warmup"):
        path = folder / ("raw/warmup.jsonl" if corruption == "warmup" else "raw/requests.jsonl")
        data = json.loads(path.read_text())
        data["request_id"] = "wrong-identity"
        path.write_text(json.dumps(data) + "\n")
    elif corruption == "drain":
        (folder / "drain.json").write_text("{}")
    elif corruption == "environment":
        data = json.loads((folder / "environment.json").read_text())
        data["cpu"] = "different"
        (folder / "environment.json").write_text(json.dumps(data))
    else:
        state.complete = False
        state.status = "failed"
    if corruption != "hash":
        capture_run(
            root, entry
        )  # Even internally consistent new hashes cannot hide semantic changes.
        (root / "campaign.json").write_bytes(encoded(state))
    with pytest.raises(ValueError):
        read_condition(root, role="baseline")


def test_dependency_normalization_and_no_wildcard_package_inventory(tmp_path):
    path = tmp_path / "pip.txt"
    path.write_text("Torch==2.8.0\nvllm==0.10.2\nfoo_bar==1\n")
    assert dependencies(path) == {
        "dependencies.torch": "2.8.0",
        "dependencies.foo-bar": "1",
        "runtime.version": "0.10.2",
    }
    path.write_text("foo.bar==1\nfoo_bar==2\n")
    with pytest.raises(ValueError):
        dependencies(path)
    path.write_text("UNAVAILABLE\n")
    assert dependencies(path) == {}
