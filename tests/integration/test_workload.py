import asyncio
import json
from pathlib import Path

import pytest
from test_streaming_client import DONE, USAGE, event, server

from inference_lab.benchmark.metrics import aggregate
from inference_lab.benchmark.plan import encoded, prepare_plan
from inference_lab.benchmark.workload import closed_loop, drain, run_point
from inference_lab.config.io import load_experiment
from inference_lab.config.schemas import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRun,
    RequestMeasurement,
)
from inference_lab.config.settings import Settings
from inference_lab.serving.client import StreamingClient

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def configuration(url: str, *, count=8, warmup=4, concurrency=4):
    data = load_experiment(ROOT / "experiments/mock-stream.yaml").model_dump(mode="json")
    data["target"]["url"] = url
    data["measurement"]["requests_per_run"] = count
    data["warmup"]["requests_per_point"] = warmup
    data["workload"]["concurrency"] = [concurrency]
    return ExperimentConfig.model_validate(data)


def planned(config):
    return prepare_plan(
        config, ROOT / config.workload.dataset, tokenize=lambda _: 3, template="synthetic/mock-v1"
    )


def scheduler(active=0):
    return (
        f'vllm:num_requests_running{{model_name="mock-model"}} {active}\n'
        'vllm:num_requests_waiting{model_name="mock-model"} 0\n'
    )


async def test_closed_loop_exact_attempts_concurrency_ceiling_and_no_retry() -> None:
    active = peak = calls = 0
    entered = asyncio.Event()
    release = asyncio.Event()
    records = []

    async def source():
        nonlocal active, peak, calls
        calls += 1
        active += 1
        peak = max(peak, active)
        try:
            if active == 3:
                entered.set()
            yield event()
            await release.wait()
            yield event("multiple tokens", "stop") + USAGE + DONE
        finally:
            active -= 1

    async with server(source) as url, StreamingClient(concurrency=3) as client:
        config = configuration(url, count=7, warmup=0, concurrency=3)
        plan = planned(config)
        task = asyncio.create_task(
            closed_loop(
                client, config, plan.requests, concurrency=3, run_id="test", save=records.append
            )
        )
        await asyncio.wait_for(entered.wait(), 2)
        assert calls == 3 and not records
        release.set()
        await task
    assert calls == 7 and peak == 3 and active == 0
    assert sorted(r.plan_index for r in records) == list(range(7))
    assert all(r.content_events == 1 and r.output_tokens == 5 for r in records)
    assert all(
        aggregate(records)[f"client_tpot_estimate_{q}_ms"].value is None
        for q in ("p50", "p95", "p99")
    )


async def test_one_run_artifacts_warmup_exclusion_and_recalculation(tmp_path) -> None:
    active = calls = 0

    async def source():
        nonlocal active, calls
        active += 1
        calls += 1
        try:
            yield event()
            await asyncio.sleep(0.01)
            yield event("one")
            yield event(" tail", "length") + USAGE + DONE
        finally:
            active -= 1

    async with server(source, metrics=lambda: scheduler(active)) as url:
        config = configuration(url)
        run = ExperimentRun(
            campaign_id="cpu-campaign",
            run_id="cpu-run",
            point=0,
            repetition=0,
            concurrency=4,
            config=config,
        )
        result = await run_point(run, planned(config), tmp_path / "run")
    assert result.complete and result.synthetic and result.workload.conformant
    assert result.published is False and result.qualification_id is None
    assert calls == 12 and result.counters.attempts == 8
    folder = tmp_path / "run"
    raw = [
        RequestMeasurement.model_validate_json(line)
        for line in (folder / "raw/requests.jsonl").read_text().splitlines()
    ]
    assert len((folder / "raw/warmup.jsonl").read_text().splitlines()) == 4
    assert aggregate(raw) == result.metrics
    assert ExperimentResult.model_validate_json((folder / "result.json").read_text()) == result
    assert (folder / "pip-freeze.txt").read_text().strip() != "UNAVAILABLE"
    assert (folder / "plan.json").read_bytes() == encoded(planned(config))
    assert not (folder / "raw/gpu.jsonl").read_bytes()
    assert "multiple tokens" not in (folder / "result.json").read_text()
    assert set(result.metrics) == set(aggregate(raw))
    assert "successful_requests_per_second" in (folder / "report.md").read_text()
    with pytest.raises(FileExistsError):
        await run_point(run, planned(config), folder)


@pytest.mark.parametrize("mode", ["eof", "error", "timeout", "nonconformant"])
async def test_partial_stream_errors_and_lengths_are_not_hidden(tmp_path, mode) -> None:
    calls = 0

    async def source():
        nonlocal calls
        calls += 1
        yield event("partial")
        if mode == "error":
            yield b'data: {"error":{"message":"SECRET"}}\n\n'
        elif mode == "timeout":
            await asyncio.sleep(1)
        elif mode == "nonconformant":
            yield (
                event("end", "stop")
                + USAGE.replace(b'"completion_tokens": 5', b'"completion_tokens": 4').replace(
                    b'"total_tokens": 8', b'"total_tokens": 7'
                )
                + DONE
            )

    async with server(source, metrics=lambda: scheduler()) as url:
        config = configuration(url, count=1, warmup=0, concurrency=1)
        run = ExperimentRun(
            campaign_id="fail-campaign",
            run_id="fail-run",
            point=0,
            repetition=0,
            concurrency=1,
            config=config,
        )
        result = await run_point(
            run,
            planned(config),
            tmp_path / "run",
            settings=Settings(vllm_stream_idle_timeout_seconds=0.05),
        )
    assert calls == 1 and result.complete and not result.workload.conformant
    raw = json.loads((tmp_path / "run/raw/requests.jsonl").read_text())
    assert raw["content_events"] == 1 or mode == "nonconformant"
    if mode == "nonconformant":
        assert raw["transport_status"] == "success" and raw["workload_status"] == "nonconformant"
    else:
        assert result.counters.successes == 0
        assert result.metrics["successful_output_tokens_per_second"].value == 0
        assert result.metrics["client_ttft_p50_ms"].value is None
        assert result.workload.input_tokens_min is None
    assert "SECRET" not in (tmp_path / "run/result.json").read_text()


async def test_cancellation_preserves_issued_attempts_and_incomplete_result(tmp_path) -> None:
    active = 0
    started = asyncio.Event()
    closed = asyncio.Event()

    async def source():
        nonlocal active
        active += 1
        try:
            yield event("partial")
            started.set()
            await asyncio.Event().wait()
        finally:
            active -= 1
            closed.set()

    async with server(source, metrics=lambda: scheduler(active)) as url:
        config = configuration(url, count=9, warmup=0, concurrency=1)
        run = ExperimentRun(
            campaign_id="cancel-campaign",
            run_id="cancel-run",
            point=0,
            repetition=0,
            concurrency=1,
            config=config,
        )
        task = asyncio.create_task(run_point(run, planned(config), tmp_path / "run"))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(closed.wait(), 2)
    result = ExperimentResult.model_validate_json((tmp_path / "run/result.json").read_text())
    assert not result.complete and result.counters.attempts == 1
    raw = json.loads((tmp_path / "run/raw/requests.jsonl").read_text())
    assert raw["transport_status"] == "cancelled" and raw["terminal_ns"] is not None


async def test_failed_warmup_and_unavailable_drain_prevent_measurement(tmp_path) -> None:
    calls = 0

    async def source():
        nonlocal calls
        calls += 1
        yield event("partial")

    async with server(source, metrics=lambda: scheduler()) as url:
        config = configuration(url, count=3, warmup=1, concurrency=1)
        run = ExperimentRun(
            campaign_id="warm-campaign",
            run_id="warm-run",
            point=0,
            repetition=0,
            concurrency=1,
            config=config,
        )
        result = await run_point(run, planned(config), tmp_path / "run")
    assert calls == 1 and not result.complete and result.counters.attempts == 0
    assert (tmp_path / "run/raw/requests.jsonl").read_bytes() == b""
    assert all(metric.value is None for metric in result.metrics.values())
    async with server(source, metrics=lambda: "# no counters\n") as url:
        from inference_lab.serving.client import StreamError

        with pytest.raises(StreamError, match="drain_unavailable"):
            await drain(configuration(url), timeout=1)


async def test_cli_executes_a_saved_plan_without_gpu_packages(tmp_path) -> None:
    import sys

    import yaml

    async def source():
        yield event("synthetic", "length") + USAGE + DONE

    async with server(source, metrics=lambda: scheduler()) as url:
        config = configuration(url, count=2, warmup=1, concurrency=1)
        config_path = tmp_path / "experiment.yaml"
        config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
        plan_path = tmp_path / "plan.json"
        plan_path.write_bytes(encoded(planned(config)))
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "inference_lab.benchmark.workload",
            "--experiment",
            str(config_path),
            "--plan",
            str(plan_path),
            "--output",
            str(tmp_path / "run"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        assert process.returncode == 0, stderr.decode()
        assert b"complete=True" in stdout
    result = ExperimentResult.model_validate_json((tmp_path / "run/result.json").read_text())
    assert result.counters.attempts == 2 and result.synthetic


async def test_terminal_time_includes_final_usage_and_done() -> None:
    import time

    from inference_lab.benchmark.workload import measure_request

    content = asyncio.Event()
    release = asyncio.Event()
    usage_sent = 0

    async def source():
        nonlocal usage_sent
        yield event()
        yield event("text", "length")
        content.set()
        await release.wait()
        usage_sent = time.perf_counter_ns()
        yield USAGE + DONE

    async with server(source) as url, StreamingClient() as client:
        config = configuration(url, count=1, warmup=0, concurrency=1)
        task = asyncio.create_task(
            measure_request(
                client, url, planned(config).requests[0], request_id="end", index=0, config=config
            )
        )
        await asyncio.wait_for(content.wait(), 2)
        assert not task.done()
        release.set()
        record = await task
    assert record.terminal_ns >= usage_sent
    assert record.first_content_ns is not None and record.content_events == 1


def test_runtime_guards_and_environment_sources_without_gpu(monkeypatch, tmp_path) -> None:
    from inference_lab.benchmark.evidence import environment
    from inference_lab.benchmark.workload import check_runtime
    from inference_lab.serving.runtime import VERSIONS, Observation, launch_plan, runtime_manifest

    config = load_experiment(ROOT / "experiments/qwen3b-fp16.yaml")
    gpu = {
        "gpu.present": Observation(
            value=[{"index": 0, "uuid": "GPU-fixture", "name": "Tesla T4"}], source="CPU fixture"
        ),
        "gpu.used": Observation(
            value=[{"pid": 123, "uuid": "GPU-fixture"}, {"pid": 124, "uuid": "GPU-fixture"}],
            source="CPU fixture",
        ),
    }
    manifest = runtime_manifest(
        launch_plan(config),
        log=(ROOT / "qualification/kaggle-t4-20261025-c/server-log.txt").read_text(),
        gpu=gpu,
        smoke={"status": "VALIDATED"},
        versions=VERSIONS,
    )
    path = tmp_path / "runtime_manifest.json"
    path.write_text(manifest.model_dump_json())
    (tmp_path / "server-process.json").write_text(
        json.dumps({"pid": 123, "start_ticks": "fixture-start"})
    )
    monkeypatch.setattr(
        "inference_lab.benchmark.workload.process_identity", lambda _: (1, "fixture-start")
    )
    monkeypatch.setattr("inference_lab.benchmark.workload.gpu_observations", lambda *_: gpu)
    assert check_runtime(config, path) == manifest
    reversed_gpu = dict(gpu)
    reversed_gpu["gpu.used"] = Observation(
        value=list(reversed(gpu["gpu.used"].value)), source="CPU fixture"
    )
    monkeypatch.setattr(
        "inference_lab.benchmark.workload.gpu_observations", lambda *_: reversed_gpu
    )
    assert check_runtime(config, path) == manifest
    changed = config.model_copy(
        update={"workload": config.workload.model_copy(update={"seed": 43})}
    )
    with pytest.raises(ValueError, match="does not match"):
        check_runtime(changed, path)
    monkeypatch.setattr("inference_lab.benchmark.workload.process_identity", lambda _: None)
    with pytest.raises(ValueError, match="exited"):
        check_runtime(config, path)

    def capture(args):
        if args == ["nvidia-smi"]:
            return "Driver Version: 580.178.04 CUDA Version: 13.0"
        if args == ["nvidia-smi", "topo", "-m"]:
            return "GPU0 X"
        if any("--query-gpu=" in arg for arg in args):
            return "0, GPU-fixture, Tesla T4, 15360, 7.5, 580.178.04"
        if args == ["git", "rev-parse", "HEAD"]:
            return "a" * 40
        if args == ["git", "status", "--porcelain"]:
            return ""
        if args[:2] == ["nvcc", "--version"]:
            return None
        return "synthetic-package==1\n"

    monkeypatch.setattr("inference_lab.benchmark.evidence.command", capture)
    env = environment(config, manifest, tmp_path)
    assert env.gpu_used_uuids == ["GPU-fixture"]
    assert env.gpu_visible_uuids == ["GPU-fixture"]
    assert env.gpu_present[0].compute_capability == (7, 5)
    assert env.driver_supported_cuda == "13.0" and env.pytorch_cuda_runtime == "12.8"
    assert env.cuda_toolkit is None and env.missing_reasons["cuda_toolkit"]
    assert env.gpu_present[0].vram_bytes == 15360 * 1024**2
    assert env.git_dirty is False


async def test_real_run_requires_manifest_before_issuing_requests(tmp_path) -> None:
    config = load_experiment(ROOT / "experiments/qwen3b-fp16.yaml")
    plan = prepare_plan(
        config,
        ROOT / config.workload.dataset,
        tokenize=lambda _: 128,
        template="CPU fixture, no inference",
    )
    run = ExperimentRun(
        campaign_id="fixture", run_id="fixture", point=0, repetition=0, concurrency=1, config=config
    )
    with pytest.raises(ValueError, match="requires an observed"):
        await run_point(run, plan, tmp_path / "run")
    assert not (tmp_path / "run").exists()
