"""One closed-loop point/repetition, shared by standalone and campaign CLIs."""

import argparse
import asyncio
import json
import math
import shutil
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from prometheus_client.parser import text_string_to_metric_families

from inference_lab.benchmark.evidence import environment, runtime_evidence
from inference_lab.benchmark.metrics import aggregate, error_breakdown
from inference_lab.benchmark.plan import (
    PlannedRequest,
    RequestPlan,
    digest,
    encoded,
    validate_plan,
)
from inference_lab.config.io import load_experiment, load_policy
from inference_lab.config.schemas import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRun,
    RegressionPolicy,
    RequestMeasurement,
    WorkloadEvidence,
)
from inference_lab.config.settings import Settings
from inference_lab.serving.client import StreamError, StreamingClient
from inference_lab.serving.runtime import RuntimeManifest, gpu_observations, process_identity


def write_json(path: Path, value: Any) -> None:
    # Result snapshots are replaced atomically; raw records are never replaced.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


async def measure_request(
    client: StreamingClient,
    url: str,
    planned: PlannedRequest,
    *,
    request_id: str,
    index: int,
    config: ExperimentConfig,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> RequestMeasurement:
    first = last = terminal = None
    content_events = 0
    usage = None
    finish = error = None
    status: Literal["success", "error", "timeout", "cancelled", "incomplete"] = "incomplete"
    start = clock()
    try:
        async with client.stream(url, planned.payload) as events:
            async for event in events:
                if event.content:
                    first = first if first is not None else event.received_ns
                    last = event.received_ns
                    content_events += 1
                if event.data is not None:
                    usage = event.data.get("usage") or usage
                    if event.data["choices"]:
                        finish = event.data["choices"][0].get("finish_reason") or finish
                if event.done:
                    terminal = event.received_ns
                    status = "success"
    except StreamError as exc:
        error = exc.code + (f"_{exc.status_code}" if exc.status_code is not None else "")
        status = "timeout" if "timeout" in exc.code else "error"
    except asyncio.CancelledError:
        # Capture this issued attempt, then let the closed-loop worker stop. Parent
        # cancellation is re-raised after the enclosing run saves incomplete artifacts.
        status, error = "cancelled", "cancelled"
    terminal = terminal if terminal is not None else clock()
    input_tokens = usage["prompt_tokens"] if usage is not None else None
    output_tokens = usage["completion_tokens"] if usage is not None else None
    conformant = (
        input_tokens == planned.input_tokens
        and output_tokens is not None
        and output_tokens == config.generation.max_tokens
    )
    nullable = {
        "first_content_ns": first,
        "last_content_ns": last,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_count_source": "server_usage" if usage is not None else None,
        "finish_reason": finish,
    }
    return RequestMeasurement.model_validate(
        dict(
            request_id=request_id,
            dataset_request_id=planned.dataset_request_id,
            plan_index=index,
            transport_status=status,
            workload_status=(
                "not_checked" if usage is None else "conformant" if conformant else "nonconformant"
            ),
            request_start_ns=start,
            terminal_ns=terminal,
            content_events=content_events,
            error=error,
            missing_reasons={
                key: "Not observed before stream termination"
                for key, value in nullable.items()
                if value is None
            },
            **nullable,
        )
    )


async def closed_loop(
    client: StreamingClient,
    config: ExperimentConfig,
    planned: Sequence[PlannedRequest],
    *,
    concurrency: int,
    run_id: str,
    save: Callable[[RequestMeasurement], None],
) -> None:
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("Concurrency must be a positive integer")
    pending = iter(enumerate(planned))

    async def worker() -> None:
        # Claim synchronously between awaits: C workers, not N queued request tasks.
        for index, request in pending:
            result = await measure_request(
                client,
                str(config.target.url),
                request,
                request_id=f"{run_id}-{index:06d}",
                index=index,
                config=config,
            )
            save(result)
            if result.transport_status == "cancelled":
                return

    async with asyncio.TaskGroup() as group:
        for _ in range(min(concurrency, len(planned))):
            group.create_task(worker())


async def drain(config: ExperimentConfig, *, timeout: float) -> None:
    """Outside measurement: require available, finite runtime running/waiting gauges."""
    names = {"vllm:num_requests_running", "vllm:num_requests_waiting"}
    url = httpx.URL(str(config.target.url)).copy_with(path="/metrics")
    try:
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(trust_env=False, timeout=min(timeout, 5)) as client:
                while True:
                    response = await client.get(url)
                    if response.status_code != 200:
                        raise StreamError("drain_unavailable")
                    values: dict[str, list[float]] = {name: [] for name in names}
                    try:
                        for family in text_string_to_metric_families(response.text):
                            for sample in family.samples:
                                if (
                                    sample.name in names
                                    and sample.labels.get("model_name") == config.model.repo_id
                                ):
                                    values[sample.name].append(sample.value)
                    except ValueError:
                        raise StreamError("drain_invalid_metrics") from None
                    if any(not readings for readings in values.values()) or any(
                        not math.isfinite(value) or value < 0
                        for readings in values.values()
                        for value in readings
                    ):
                        raise StreamError("drain_unavailable")
                    if all(value == 0 for readings in values.values() for value in readings):
                        return
                    await asyncio.sleep(0.05)
    except (TimeoutError, httpx.HTTPError):
        raise StreamError("drain_timeout_or_network_error") from None


def check_runtime(config: ExperimentConfig, path: Path) -> RuntimeManifest:
    manifest = RuntimeManifest.model_validate_json(path.read_text())
    expected = manifest.expected.experiment
    if (
        manifest.status != "VALIDATED"
        or manifest.mismatches
        or manifest.readiness.get("status") != "VALIDATED"
        or config.model != expected.model
        or config.runtime != expected.runtime
        or config.workload.seed != expected.workload.seed
        or config.target != expected.target
    ):
        raise ValueError("Active runtime manifest does not match the experiment")
    process = json.loads((path.parent / "server-process.json").read_text())
    identity = process_identity(process["pid"])
    if identity is None or identity[1] != process["start_ticks"]:
        raise ValueError("Runtime process exited or its PID was replaced; recapture the manifest")
    current = gpu_observations(process["pid"], process["start_ticks"])["gpu.used"]
    previous = manifest.observed["gpu.used"]
    if (
        current.reason
        or not isinstance(current.value, list)
        or not isinstance(previous.value, list)
    ):
        raise ValueError("Server GPU ownership unavailable; recapture the runtime manifest")
    if sorted(json.dumps(row, sort_keys=True) for row in current.value) != sorted(
        json.dumps(row, sort_keys=True) for row in previous.value
    ):
        raise ValueError("Server GPU ownership changed; recapture the runtime manifest")
    return manifest


def result_for(
    run: ExperimentRun,
    plan: RequestPlan,
    requests: Sequence[RequestMeasurement],
    *,
    env: Any,
    runtime: Any,
    complete: bool,
    limitations: list[str],
    qualification_id: str | None,
    policy: Any = None,
) -> ExperimentResult:
    lengths: dict[str, Any] = {}
    reasons = {}
    for kind in ("input", "output"):
        values = [value for r in requests if (value := getattr(r, kind + "_tokens")) is not None]
        for suffix, operation in (("min", min), ("max", max)):
            field = kind + "_tokens_" + suffix
            lengths[field] = operation(values) if values else None
            if not values:
                reasons[field] = "No server usage token counts observed"
    conformant = bool(requests) and all(
        r.transport_status == "success" and r.workload_status == "conformant" for r in requests
    )
    if not conformant:
        limitations = limitations + [
            "Workload is nonconformant or unverified; publication is blocked."
        ]
    return ExperimentResult.model_validate(
        dict(
            **run.model_dump(),
            result_schema_version="1.0",
            metric_contract_version="1.0",
            complete=complete,
            published=False,
            synthetic=plan.synthetic,
            qualification_id=qualification_id,
            environment=env,
            runtime_manifest=runtime,
            workload=WorkloadEvidence(
                dataset_name=plan.dataset_name,
                dataset_version=plan.dataset_version,
                dataset_sha256=plan.dataset_sha256,
                plan_sha256=digest(encoded(plan)),
                tokenizer_repo_id=plan.model.tokenizer_repo_id,
                tokenizer_revision=plan.model.tokenizer_revision,
                chat_template_sha256=plan.chat_template_sha256,
                template_parameters={key: value for key, value in plan.template_parameters.items()},
                conformant=conformant,
                missing_reasons=reasons,
                **lengths,
            ),
            counters={
                "attempts": len(requests),
                "successes": sum(r.transport_status == "success" for r in requests),
                "errors": sum(r.transport_status != "success" for r in requests),
            },
            metrics=aggregate(requests),
            percentile_method="linear",
            limitations=limitations,
            policy=policy,
            artifacts={
                "requests": "raw/requests.jsonl",
                "gpu": "raw/gpu.jsonl",
                "plan": "plan.json",
                "dataset": "dataset.jsonl",
                "drain": "drain.json",
            },
        )
    )


def report(result: ExperimentResult, requests: Sequence[RequestMeasurement]) -> str:
    lines = [
        f"# Run {result.run_id}",
        "",
        f"Campaign: {result.campaign_id}; point: {result.point}; "
        f"repetition: {result.repetition}; C={result.concurrency}.",
        f"Complete: {result.complete}; synthetic: {result.synthetic}; published: false.",
        f"Attempts: {result.counters.attempts}; successes: {result.counters.successes}; "
        f"errors: {result.counters.errors}.",
        f"Workload conformant: {result.workload.conformant}. "
        "Percentiles: linear, conditional on successful responses.",
        "",
        "| Metric | Value | Unit | Observations | Missing reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, metric in result.metrics.items():
        lines.append(
            f"| {name} | {metric.value} | {metric.unit} | "
            f"{metric.sample_count} | {metric.reason or ''} |"
        )
    lines += [
        "",
        f"Normalized errors: {json.dumps(error_breakdown(requests), sort_keys=True)}",
        "",
        "## Limits",
        "",
    ]
    lines += ["- " + limitation for limitation in result.limitations]
    return "\n".join(lines) + "\n"


def validate_workload(
    config: ExperimentConfig,
    plan: RequestPlan,
    *,
    dataset_path: Path | None = None,
    policy_path: Path | None = None,
) -> RegressionPolicy | None:
    """Validate frozen input copies without changing their canonical configuration."""
    validate_plan(plan, config, dataset_path or Path(config.workload.dataset))
    if not plan.synthetic:
        config.model.require_immutable()
    if policy_path is not None and config.policy_file is None:
        raise ValueError("Policy snapshots require the canonical policy_file reference")
    policy = (
        load_policy(policy_path or Path(config.policy_file))
        if config.policy_file is not None
        else None
    )
    if policy is not None:
        policy.validate_experiment(config)
    if policy is not None and policy.workload_id != plan.dataset_name:
        raise ValueError("Policy workload_id differs from the prepared dataset")
    if not config.generation.stream or config.target.url.host != "127.0.0.1":
        raise ValueError("This runner requires streaming and co-located loopback HTTP")
    if config.runtime.enable_prefix_caching:
        raise ValueError("Reference measurement requires prefix caching disabled")
    return policy


async def run_point(
    run: ExperimentRun,
    plan: RequestPlan,
    output: Path,
    *,
    manifest_path: Path | None = None,
    settings: Settings | None = None,
    dataset_path: Path | None = None,
    policy_path: Path | None = None,
) -> ExperimentResult:
    config = run.config
    policy = validate_workload(config, plan, dataset_path=dataset_path, policy_path=policy_path)
    if plan.synthetic and manifest_path is not None:
        raise ValueError("Synthetic fixtures cannot use real GPU runtime evidence")
    if not plan.synthetic and manifest_path is None:
        raise ValueError("Real serving requires an observed runtime manifest")
    manifest = check_runtime(config, manifest_path) if manifest_path is not None else None
    settings = settings or Settings()
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    (output / "experiment.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )
    (output / "plan.json").write_bytes(encoded(plan))
    shutil.copyfile(dataset_path or Path(config.workload.dataset), output / "dataset.jsonl")
    if policy is not None:
        (output / "policy.json").write_bytes(encoded(policy))
    if manifest_path is not None:
        shutil.copyfile(manifest_path, output / "runtime_manifest.json")
        shutil.copyfile(
            manifest_path.parent / "server-process.json", output / "server-process.json"
        )
        shutil.copyfile(manifest_path.parent / "server.log", output / "server.log")
    else:
        write_json(
            output / "runtime_manifest.json",
            {"synthetic": True, "reason": "CPU mock, no GPU runtime"},
        )
        (output / "server.log").write_text("NOT_COLLECTED: controlled synthetic upstream\n")
    (output / "raw/gpu.jsonl").touch()
    try:
        env = environment(config, manifest, output)
    except Exception as error:
        write_json(
            output / "failure.json",
            {"phase": "environment_preflight", "error_type": type(error).__name__},
        )
        raise
    write_json(output / "environment.json", env.model_dump(mode="json"))
    runtime = runtime_evidence(config, manifest)
    requests: list[RequestMeasurement] = []
    warmup: list[RequestMeasurement] = []
    interrupted = False
    stop_error = None
    limitations = [
        "One point/repetition; no SLO, p99 certification, speedup or gate verdict.",
        "Client TPOT is a reception estimate; content-event counts are not tokens.",
        "raw/gpu.jsonl is empty: continuous GPU telemetry belongs to its later stage.",
        "Runtime KV dtype and linear kernel remain unknown when not exposed by the manifest.",
        "Gauges are sampled outside measurement; stale gauges alone cannot validate cancellation.",
    ]
    if plan.synthetic:
        limitations.append(
            "Synthetic CPU upstream and token counts; client timings are not GPU performance."
        )
    drain_checks: dict[str, Any] = {
        "schema_version": "1.0",
        "source": "runtime_scheduler_gauges",
        "before_warmup": "NOT_TESTED",
        "after_warmup": "NOT_TESTED",
        "after_measurement": "NOT_TESTED",
        "limitation": "Sampled scheduler gauges alone do not prove GPU request cancellation.",
    }
    write_json(output / "drain.json", drain_checks)
    phase = "before_warmup"
    with (
        (output / "raw/requests.jsonl").open("x", buffering=1) as raw,
        (output / "raw/warmup.jsonl").open("x", buffering=1) as warm_raw,
    ):

        def save(request: RequestMeasurement) -> None:
            raw.write(request.model_dump_json() + "\n")
            requests.append(request)

        def save_warm(request: RequestMeasurement) -> None:
            warm_raw.write(request.model_dump_json() + "\n")
            warmup.append(request)

        def snapshot(complete: bool) -> ExperimentResult:
            result = result_for(
                run,
                plan,
                requests,
                env=env,
                runtime=runtime,
                complete=complete,
                limitations=limitations,
                qualification_id=manifest.qualification_id if manifest else None,
                policy=policy,
            )
            write_json(output / "result.json", result.model_dump(mode="json"))
            (output / "report.md").write_text(report(result, requests))
            return result

        snapshot(False)
        try:
            async with StreamingClient(
                concurrency=run.concurrency,
                total_timeout=settings.vllm_timeout_seconds,
                idle_timeout=settings.vllm_stream_idle_timeout_seconds,
            ) as client:
                await drain(config, timeout=settings.vllm_stream_idle_timeout_seconds)
                drain_checks[phase] = "VALIDATED"
                await closed_loop(
                    client,
                    config,
                    plan.warmup,
                    concurrency=run.concurrency,
                    run_id=run.run_id + "-warmup",
                    save=save_warm,
                )
                if any(
                    r.transport_status != "success" or r.workload_status != "conformant"
                    for r in warmup
                ):
                    raise StreamError("warmup_failed_or_nonconformant")
                phase = "after_warmup"
                await drain(config, timeout=settings.vllm_stream_idle_timeout_seconds)
                drain_checks[phase] = "VALIDATED"
                await closed_loop(
                    client,
                    config,
                    plan.requests,
                    concurrency=run.concurrency,
                    run_id=run.run_id,
                    save=save,
                )
                phase = "after_measurement"
                await drain(config, timeout=settings.vllm_stream_idle_timeout_seconds)
                drain_checks[phase] = "VALIDATED"
        except asyncio.CancelledError:
            interrupted = True
            limitations.append(
                "Interrupted: attempts retained; GPU drain unverified. Check before target reuse."
            )
        except StreamError as exc:
            stop_error = exc.code
            drain_checks["error"] = exc.code
            if exc.code.startswith("drain_"):
                drain_checks[phase] = "FAILED"
            limitations.append(
                f"Stopped: {exc.code}; inspect residual work before reusing the target."
            )
        except Exception:
            stop_error = "runner_internal_error"
            limitations.append("Runner stopped unexpectedly; residual work is unverified.")
            raise
        finally:
            write_json(output / "drain.json", drain_checks)
            try:
                if manifest_path is not None:
                    shutil.copyfile(manifest_path.parent / "server.log", output / "server.log")
            except OSError:
                stop_error = "server_log_collection_failed"
                limitations.append("Final server log collection failed; evidence is incomplete.")
            finally:
                result = snapshot(
                    not interrupted and stop_error is None and len(requests) == len(plan.requests)
                )
    if interrupted:
        raise asyncio.CancelledError
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--output", type=Path, help="Fresh run artifact directory")
    parser.add_argument("--point", type=int, default=0)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--campaign-id", default="campaign-" + uuid.uuid4().hex)
    parser.add_argument("--run-id", default="run-" + uuid.uuid4().hex)
    args = parser.parse_args()
    try:
        config = load_experiment(args.experiment)
        if not 0 <= args.point < len(config.workload.concurrency):
            raise ValueError("Point is outside the configured concurrency list")
        run = ExperimentRun(
            campaign_id=args.campaign_id,
            run_id=args.run_id,
            point=args.point,
            repetition=args.repetition,
            concurrency=config.workload.concurrency[args.point],
            config=config,
        )
        plan = RequestPlan.model_validate_json(args.plan.read_text())
        output = args.output or Path("results") / run.campaign_id / run.run_id
        result = asyncio.run(run_point(run, plan, output, manifest_path=args.runtime_manifest))
        print(
            f"Artifacts: {output}; complete={result.complete}; "
            f"workload_conformant={result.workload.conformant}"
        )
        return (
            0
            if result.complete and result.workload.conformant and result.counters.errors == 0
            else 1
        )
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, ExceptionGroup, yaml.YAMLError):
        print(
            "Runner failed: check config, prepared plan, runtime evidence and output. "
            "Available artifacts are preserved."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
