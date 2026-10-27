import asyncio
import json
import sys
from pathlib import Path

import pytest
import yaml
from test_streaming_client import DONE, USAGE, event, server
from test_workload import configuration, planned, scheduler

from inference_lab.benchmark.campaign import Campaign
from inference_lab.benchmark.metrics import aggregate
from inference_lab.benchmark.plan import encoded
from inference_lab.benchmark.runner import resume_campaign, run_campaign
from inference_lab.config.schemas import ExperimentResult, RequestMeasurement
from inference_lab.config.settings import Settings

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def inputs(tmp_path, url, *, paired=False, repetitions=2, points=(1, 2), count=2):
    data = configuration(url, count=count, warmup=1, concurrency=1).model_dump(mode="json")
    data["workload"]["concurrency"] = list(points)
    data["measurement"]["repetitions"] = repetitions
    data["workload"]["dataset"] = str(ROOT / "experiments/datasets/mock-v1.jsonl")
    if paired:
        policy = json.loads((ROOT / "tests/data/synthetic-policy.json").read_text())
        policy.update(workload_id="mock-v1", min_repetitions=repetitions)
        policy["comparison"]["allowed_changes"] = []
        policy_path = tmp_path / "policy.yaml"
        policy_path.write_text(yaml.safe_dump(policy))
        data["policy_file"] = str(policy_path)
    exp = tmp_path / "experiment.yaml"
    exp.write_text(yaml.safe_dump(data))
    from inference_lab.config.io import load_experiment

    config = load_experiment(exp)
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(encoded(planned(config)))
    return exp, plan_path


async def test_sequential_campaign_identity_warmups_artifacts_and_replay(tmp_path) -> None:
    active = peak = calls = 0

    async def source():
        nonlocal active, peak, calls
        calls += 1
        active += 1
        peak = max(peak, active)
        try:
            yield event("one")
            await asyncio.sleep(0.005)
            yield event(" two", "length") + USAGE + DONE
        finally:
            active -= 1

    output = tmp_path / "campaign"
    async with server(source, metrics=lambda: scheduler(active)) as url:
        exp, plan = inputs(tmp_path, url)
        result = await run_campaign(exp, plan, output, campaign_id="cpu-campaign")
    assert result.complete and result.status == "complete" and result.synthetic
    assert calls == 12 and peak <= 2 and active == 0
    assert [(e.point, e.repetition, e.concurrency) for e in result.schedule] == [
        (0, 0, 1),
        (0, 1, 1),
        (1, 0, 2),
        (1, 1, 2),
    ]
    assert len({e.run_id for e in result.schedule}) == 4
    assert Campaign.model_validate_json((output / "campaign.json").read_bytes()) == result
    for entry in result.schedule:
        folder = output / entry.run_id
        point = ExperimentResult.model_validate_json((folder / "result.json").read_bytes())
        assert point.campaign_id == result.campaign_id and point.run_id == entry.run_id
        assert point.config.workload.concurrency[point.point] == point.concurrency
        assert point.counters.attempts == 2 and point.published is False
        raw = [
            RequestMeasurement.model_validate_json(line)
            for line in (folder / "raw/requests.jsonl").read_text().splitlines()
        ]
        assert aggregate(raw) == point.metrics
        assert json.loads((folder / "drain.json").read_text())["after_measurement"] == "VALIDATED"
        assert len((folder / "raw/warmup.jsonl").read_text().splitlines()) == 1
    report = (output / "report.md").read_text()
    assert "percentiles are never averaged" in report and "synthetic: True" in report
    with pytest.raises(ValueError, match="Only a planned"):
        await resume_campaign(output)


async def test_paired_order_handoffs_and_self_contained_resume(tmp_path) -> None:
    active = 0

    async def source():
        nonlocal active
        active += 1
        try:
            yield event("synthetic", "length") + USAGE + DONE
        finally:
            active -= 1

    output = tmp_path / "paired"
    async with server(source, metrics=lambda: scheduler(active)) as url:
        exp, plan = inputs(tmp_path, url, paired=True, repetitions=3, points=(1,), count=1)
        candidate = tmp_path / "candidate.yaml"
        data = yaml.safe_load(exp.read_text())
        data["name"] = "candidate-label"
        candidate.write_text(yaml.safe_dump(data))
        campaign = await run_campaign(
            exp, plan, output, campaign_id="paired", candidate=candidate, candidate_plan=plan
        )
        assert campaign.status == "awaiting_runtime"
        assert [e.condition for e in campaign.schedule] == [
            "baseline",
            "candidate",
            "candidate",
            "baseline",
            "baseline",
            "candidate",
        ]
        assert [e.status for e in campaign.schedule].count("completed") == 1
        # Original YAML, plan and policy are unnecessary after inputs are frozen.
        exp.unlink()
        candidate.unlink()
        plan.unlink()
        (tmp_path / "policy.yaml").unlink()
        for expected_completed in (3, 5, 6):
            campaign = await resume_campaign(output)
            assert sum(e.status == "completed" for e in campaign.schedule) == expected_completed
        assert campaign.complete and campaign.published is False
        assert len(set(campaign.request_plan_sha256.values())) == 1
        assert all((output / e.run_id / "policy.json").exists() for e in campaign.schedule)


@pytest.mark.parametrize("corrupt", ["input", "result", "raw", "order", "session", "timeout"])
async def test_resume_refuses_changed_evidence_before_any_request(
    tmp_path, monkeypatch, corrupt
) -> None:
    calls = 0

    async def source():
        nonlocal calls
        calls += 1
        yield event("synthetic", "length") + USAGE + DONE

    output = tmp_path / "paired"
    async with server(source, metrics=lambda: scheduler()) as url:
        exp, plan = inputs(tmp_path, url, paired=True, repetitions=1, points=(1,), count=1)
        campaign = await run_campaign(
            exp, plan, output, campaign_id="paired", candidate=exp, candidate_plan=plan
        )
        assert campaign.status == "awaiting_runtime"
        before = calls
        if corrupt == "input":
            path = output / campaign.conditions["candidate"].files["dataset"]
            path.write_bytes(path.read_bytes() + b"\n")
        elif corrupt in ("result", "raw"):
            name = "result.json" if corrupt == "result" else "raw/requests.jsonl"
            path = output / campaign.schedule[0].run_id / name
            path.write_bytes(path.read_bytes() + b"\n")
        elif corrupt == "order":
            state = json.loads((output / "campaign.json").read_text())
            state["schedule"].reverse()
            (output / "campaign.json").write_text(json.dumps(state))
        elif corrupt == "session":
            monkeypatch.setattr("inference_lab.benchmark.runner.session_id", lambda: "f" * 64)
        kwargs = {"settings": Settings(vllm_timeout_seconds=181)} if corrupt == "timeout" else {}
        with pytest.raises(ValueError):
            await resume_campaign(output, **kwargs)
        assert calls == before


@pytest.mark.parametrize("mode", ["warmup", "measurement", "nonconformant", "busy_drain"])
async def test_failed_point_never_advances_campaign(tmp_path, mode) -> None:
    calls = 0

    async def source():
        nonlocal calls
        calls += 1
        if (mode == "warmup") or (mode == "measurement" and calls > 1):
            yield event("partial")
        else:
            usage = USAGE
            if mode == "nonconformant" and calls > 1:
                usage = USAGE.replace(b'"completion_tokens": 5', b'"completion_tokens": 4').replace(
                    b'"total_tokens": 8', b'"total_tokens": 7'
                )
            yield event("synthetic", "length") + usage + DONE

    def metrics():
        return scheduler(1 if mode == "busy_drain" and calls >= 3 else 0)

    output = tmp_path / "failed"
    async with server(source, metrics=metrics) as url:
        exp, plan = inputs(tmp_path, url, count=2)
        campaign = await run_campaign(
            exp,
            plan,
            output,
            campaign_id="failed",
            settings=Settings(vllm_stream_idle_timeout_seconds=0.05),
        )
    assert campaign.status == "failed" and not campaign.complete
    assert campaign.schedule[0].status == "failed"
    assert all(e.status == "pending" for e in campaign.schedule[1:])
    assert calls == (1 if mode == "warmup" else 3)
    assert (output / campaign.schedule[0].run_id / "result.json").exists()
    with pytest.raises(ValueError, match="no point retries"):
        await resume_campaign(output)


async def test_interruption_preserves_active_run_and_no_later_point(tmp_path) -> None:
    started = asyncio.Event()
    closed = asyncio.Event()
    calls = 0

    async def source():
        nonlocal calls
        calls += 1
        try:
            if calls == 1:
                yield event("warmup", "length") + USAGE + DONE
            else:
                yield event("partial")
                started.set()
                await asyncio.Event().wait()
        finally:
            closed.set()

    output = tmp_path / "interrupted"
    async with server(source, metrics=lambda: scheduler()) as url:
        exp, plan = inputs(tmp_path, url, count=1)
        task = asyncio.create_task(run_campaign(exp, plan, output, campaign_id="interrupted"))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    campaign = Campaign.model_validate_json((output / "campaign.json").read_bytes())
    assert campaign.status == "interrupted" and not campaign.complete
    assert campaign.schedule[0].status == "interrupted"
    assert all(e.status == "pending" for e in campaign.schedule[1:])
    point = ExperimentResult.model_validate_json(
        (output / campaign.schedule[0].run_id / "result.json").read_bytes()
    )
    assert not point.complete and point.counters.attempts == 1
    assert calls == 2


async def test_cli_campaign_without_gpu_or_database(tmp_path) -> None:
    async def source():
        yield event("synthetic", "length") + USAGE + DONE

    async with server(source, metrics=lambda: scheduler()) as url:
        exp, plan = inputs(tmp_path, url, repetitions=1, points=(1, 2), count=1)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "inference_lab.benchmark.runner",
            str(exp),
            "--plan",
            str(plan),
            "--output",
            str(tmp_path / "cli"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        assert process.returncode == 0, stderr.decode()
        assert b"complete" in stdout
    campaign = Campaign.model_validate_json((tmp_path / "cli/campaign.json").read_bytes())
    assert campaign.complete and len(campaign.schedule) == 2


async def test_bad_workload_is_refused_before_creating_campaign(tmp_path) -> None:
    exp, plan = inputs(tmp_path, "http://127.0.0.1:1/v1/chat/completions")
    data = json.loads(plan.read_text())
    data["requests"][0]["input_tokens"] = 4
    plan.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        await run_campaign(exp, plan, tmp_path / "invalid", campaign_id="invalid")
    assert not (tmp_path / "invalid").exists()


async def test_runtime_mismatch_stops_before_gpu_or_http(tmp_path, monkeypatch) -> None:
    from inference_lab.benchmark.plan import prepare_plan
    from inference_lab.config.io import load_experiment
    from inference_lab.config.schemas import ExperimentConfig

    config = load_experiment(ROOT / "experiments/qwen3b-fp16.yaml").model_dump(mode="json")
    config["workload"]["dataset"] = str(ROOT / "experiments/datasets/short-chat-v1.jsonl")
    config["workload"]["concurrency"] = [1]
    config["measurement"].update(requests_per_run=1, repetitions=1)
    config = ExperimentConfig.model_validate(config)
    exp = tmp_path / "experiment.yaml"
    exp.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    plan = prepare_plan(
        config,
        Path(config.workload.dataset),
        tokenize=lambda _: 128,
        template="CPU preflight fixture",
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(encoded(plan))

    def mismatch(*_):
        raise ValueError("SECRET upstream diagnostics must not leak")

    monkeypatch.setattr("inference_lab.benchmark.workload.check_runtime", mismatch)
    output = tmp_path / "refused"
    campaign = await run_campaign(
        exp,
        plan_path,
        output,
        campaign_id="refused",
        manifest_path=tmp_path / "runtime_manifest.json",
    )
    assert campaign.status == "failed" and campaign.schedule[0].status == "failed"
    assert not (output / campaign.schedule[0].run_id).exists()
    assert "SECRET" not in (output / "campaign.json").read_text()


async def test_concurrent_resume_cannot_overwrite_campaign_state(tmp_path) -> None:
    from inference_lab.benchmark.runner import campaign_lock

    calls = 0

    async def source():
        nonlocal calls
        calls += 1
        yield event("synthetic", "length") + USAGE + DONE

    output = tmp_path / "paired"
    async with server(source, metrics=lambda: scheduler()) as url:
        exp, plan = inputs(tmp_path, url, paired=True, repetitions=1, points=(1,), count=1)
        await run_campaign(
            exp, plan, output, campaign_id="locked", candidate=exp, candidate_plan=plan
        )
        original = (output / "campaign.json").read_bytes()
        before = calls
        with campaign_lock(output):
            with pytest.raises(ValueError, match="already running"):
                await resume_campaign(output)
        assert (output / "campaign.json").read_bytes() == original and calls == before


@pytest.mark.parametrize("tamper", [False, True])
async def test_calibration_reference_is_frozen_before_candidate(
    tmp_path, monkeypatch, tamper
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = 0

    async def source():
        nonlocal calls
        calls += 1
        yield event("synthetic", "length") + USAGE + DONE

    output = tmp_path / "paired"
    async with server(source, metrics=lambda: scheduler()) as url:
        exp, plan = inputs(tmp_path, url, paired=True, repetitions=1, points=(1,), count=1)
        policy_path = tmp_path / "policy.yaml"
        policy = yaml.safe_load(policy_path.read_text())
        policy["calibration_ref"] = "calibration.json"
        policy_path.write_text(yaml.safe_dump(policy))
        calibration = tmp_path / "calibration.json"
        evidence = b'{"synthetic":true,"scope":"CPU calibration fixture, no GPU evidence"}\n'
        calibration.write_bytes(evidence)
        state = await run_campaign(
            exp, plan, output, campaign_id="calibration", candidate=exp, candidate_plan=plan
        )
        snapshot = output / state.conditions["candidate"].files["calibration"]
        assert snapshot.read_bytes() == evidence
        calibration.unlink()
        policy_path.unlink()
        if tamper:
            snapshot.write_bytes(b"changed")
            before = calls
            with pytest.raises(ValueError, match="changed"):
                await resume_campaign(output)
            assert calls == before
        else:
            assert (await resume_campaign(output)).complete


async def test_cli_manual_handoff_exit_codes(tmp_path) -> None:
    async def source():
        yield event("synthetic", "length") + USAGE + DONE

    async with server(source, metrics=lambda: scheduler()) as url:
        exp, plan = inputs(tmp_path, url, paired=True, repetitions=1, points=(1,), count=1)
        output = tmp_path / "cli-paired"
        args = [
            str(exp),
            "--plan",
            str(plan),
            "--candidate",
            str(exp),
            "--candidate-plan",
            str(plan),
            "--output",
            str(output),
        ]
        for extra, wanted in [(args, 2), (["--resume", str(output)], 0)]:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "inference_lab.benchmark.runner",
                *extra,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
            assert process.returncode == wanted, stderr.decode()
            assert b"awaiting_runtime" in stdout if wanted == 2 else b"complete" in stdout
