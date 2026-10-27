"""Sequential campaigns with frozen inputs and explicit manual runtime handoffs."""

import argparse
import asyncio
import fcntl
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml

from inference_lab.benchmark.campaign import (
    Campaign,
    CampaignEntry,
    Condition,
    ConditionName,
    capture_condition,
    load_inputs,
    pair_protocol,
    paired_plan_hash,
    schedule,
    session_id,
    validate_completed,
)
from inference_lab.benchmark.plan import RequestPlan, digest
from inference_lab.benchmark.workload import check_runtime, run_point, validate_workload, write_json
from inference_lab.config.io import load_experiment, load_policy
from inference_lab.config.schemas import ExperimentConfig, ExperimentResult, ExperimentRun
from inference_lab.config.settings import Settings


@contextmanager
def campaign_lock(root: Path) -> Iterator[None]:
    """Protect checkpoint state against two concurrent resume invocations."""
    with (root / ".campaign.lock").open("r+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Campaign is already running") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def timeouts(settings: Settings) -> dict[str, float]:
    return {
        "global_seconds": settings.vllm_timeout_seconds,
        "idle_seconds": settings.vllm_stream_idle_timeout_seconds,
    }


def point_result(root: Path, entry: CampaignEntry) -> ExperimentResult | None:
    path = root / entry.run_id / "result.json"
    try:
        return ExperimentResult.model_validate_json(path.read_bytes())
    except (OSError, ValueError):
        return None


def report(root: Path, campaign: Campaign) -> str:
    lines = [
        f"# Campaign {campaign.campaign_id}",
        "",
        f"Status: {campaign.status}; complete: {campaign.complete}; "
        f"synthetic: {campaign.synthetic}; published: false.",
        "Each row is one condition/point/repetition; percentiles are never averaged.",
        "",
        "| Condition | Point | Rep | C | Status | Attempts / success / errors | TTFT p95 ms (n) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in campaign.schedule:
        result = point_result(root, entry)
        counts, metric = "unavailable", "null"
        if result is not None:
            c = result.counters
            counts = f"{c.attempts} / {c.successes} / {c.errors}"
            observed = result.metrics["client_ttft_p95_ms"]
            metric = f"{observed.value} (n={observed.sample_count})"
        lines.append(
            f"| {entry.condition} | {entry.point} | {entry.repetition} | "
            f"{entry.concurrency} | {entry.status} | {counts} | {metric} |"
        )
    lines += ["", "## Declared configuration differences", ""]
    lines += [
        f"- `{d.path}`: {json.dumps(d.baseline)} → {json.dumps(d.candidate)}"
        for d in campaign.differences
    ] or ["- None."]
    lines += ["", "## Limits", ""] + ["- " + value for value in campaign.limitations]
    if campaign.error:
        lines += ["", f"Stopped: {campaign.error}."]
    return "\n".join(lines) + "\n"


def checkpoint(root: Path, campaign: Campaign) -> None:
    Campaign.model_validate(campaign.model_dump())
    write_json(root / "campaign.json", campaign.model_dump(mode="json"))
    (root / "report.md").write_text(report(root, campaign))


def capture_run(root: Path, entry: CampaignEntry) -> None:
    directory = root / entry.run_id
    if directory.exists():
        entry.artifacts_sha256 = {
            path.relative_to(directory).as_posix(): digest(path.read_bytes())
            for path in sorted(directory.rglob("*"))
            if path.is_file() and not path.is_symlink()
        }


def validate_frozen(root: Path, campaign: Campaign) -> None:
    # ponytail: rehash completed artifacts at each step (quadratic in run count).
    # The matrix is small; use an append-only integrity ledger for larger campaigns.
    plans, policies = {}, {}
    for name, condition in campaign.conditions.items():
        plan, _, policy_path = load_inputs(root, condition)
        plans[name] = plan
        if plan.synthetic != campaign.synthetic:
            raise ValueError("Campaign synthetic/real identity changed")
        policies[name] = load_policy(policy_path) if policy_path is not None else None
        if paired_plan_hash(plan) != campaign.request_plan_sha256[name]:
            raise ValueError("Frozen request pairing changed")
    comparison, differences = pair_protocol(
        {name: condition.config for name, condition in campaign.conditions.items()},
        plans,
        policies,
    )
    if comparison != campaign.comparison or differences != campaign.differences:
        raise ValueError("Frozen comparison protocol changed")
    validate_completed(root, campaign)


async def execute(
    campaign: Campaign,
    output: Path,
    *,
    manifest_path: Path | None,
    settings: Settings,
) -> Campaign:
    active_condition: ConditionName | None = None
    for entry in campaign.schedule:
        if entry.status == "completed":
            continue
        if active_condition is not None and entry.condition != active_condition:
            campaign.status = "awaiting_runtime"
            checkpoint(output, campaign)
            return campaign
        active_condition = entry.condition
        try:
            validate_frozen(output, campaign)
            condition = campaign.conditions[entry.condition]
            plan, dataset_path, policy_path = load_inputs(output, condition)
            entry.status = "running"
            checkpoint(output, campaign)
            run = ExperimentRun(
                campaign_id=campaign.campaign_id,
                run_id=entry.run_id,
                config=condition.config,
                point=entry.point,
                repetition=entry.repetition,
                concurrency=entry.concurrency,
            )
            print(
                f"Running {entry.condition}: point={entry.point}, "
                f"repetition={entry.repetition}, C={entry.concurrency}",
                flush=True,
            )
            result = await run_point(
                run,
                plan,
                output / entry.run_id,
                manifest_path=manifest_path,
                settings=settings,
                dataset_path=dataset_path,
                policy_path=policy_path,
            )
            capture_run(output, entry)
            if not result.complete or not result.workload.conformant or result.counters.errors:
                entry.status, campaign.status = "failed", "failed"
                campaign.error = "incomplete_or_nonconformant_point"
                checkpoint(output, campaign)
                return campaign
            drain = json.loads((output / entry.run_id / "drain.json").read_text())
            if drain.get("after_measurement") != "VALIDATED":
                raise ValueError("No recorded scheduler drain after measurement")
            entry.status = "completed"
            checkpoint(output, campaign)
        except asyncio.CancelledError:
            capture_run(output, entry)
            entry.status, campaign.status = "interrupted", "interrupted"
            campaign.error = "interrupted_residual_work_unverified"
            checkpoint(output, campaign)
            raise
        except Exception as error:
            capture_run(output, entry)
            entry.status, campaign.status = "failed", "failed"
            campaign.error = f"point_execution_{type(error).__name__}"
            checkpoint(output, campaign)
            return campaign
    campaign.status, campaign.complete = "complete", True
    checkpoint(output, campaign)
    return campaign


async def run_campaign(
    experiment: Path,
    plan_path: Path,
    output: Path,
    *,
    campaign_id: str,
    candidate: Path | None = None,
    candidate_plan: Path | None = None,
    manifest_path: Path | None = None,
    settings: Settings | None = None,
) -> Campaign:
    configs: dict[ConditionName, ExperimentConfig] = {
        "experiment" if candidate is None else "baseline": load_experiment(experiment)
    }
    plans: dict[ConditionName, RequestPlan] = {
        next(iter(configs)): RequestPlan.model_validate_json(plan_path.read_bytes())
    }
    if candidate is not None:
        if candidate_plan is None:
            raise ValueError("Candidate requires its prepared plan")
        configs["candidate"] = load_experiment(candidate)
        plans["candidate"] = RequestPlan.model_validate_json(candidate_plan.read_bytes())
    elif candidate_plan is not None:
        raise ValueError("Candidate plan requires a candidate experiment")
    policies = {name: validate_workload(config, plans[name]) for name, config in configs.items()}
    comparison, differences = pair_protocol(configs, plans, policies)
    for policy in policies.values():
        if policy is not None and policy.calibration_ref is not None:
            if not Path(policy.calibration_ref).read_bytes():
                raise ValueError("Calibration reference must exist before measurement")
    settings = settings or Settings()
    campaign = Campaign(
        campaign_id=campaign_id,
        session_id=session_id(),
        synthetic=all(plan.synthetic for plan in plans.values()),
        conditions={
            name: Condition(config=config, files={}, sha256={}) for name, config in configs.items()
        },
        schedule=schedule(campaign_id, configs),
        timeouts=timeouts(settings),
        comparison=comparison,
        differences=differences,
        request_plan_sha256={name: paired_plan_hash(plan) for name, plan in plans.items()},
        limitations=[
            "Unpublished campaign; no SLO/regression verdict or performance certification.",
            "Per-run metrics retain their own populations/windows; no averaged percentiles.",
            "Runtime switches are manual; interrupted/failed campaigns cannot retry or resume.",
            "Calibration references are frozen evidence, not an assertion of statistical adequacy.",
            "Sampled idle gauges alone do not validate GPU cancellation/abort.",
        ],
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / ".campaign.lock").touch(exist_ok=False)
    with campaign_lock(output):
        try:
            campaign.conditions = {
                name: capture_condition(output, name, config, plans[name], policies[name])
                for name, config in configs.items()
            }
            validate_frozen(output, campaign)
        except Exception as error:
            write_json(
                output / "failure.json",
                {"phase": "input_capture", "error_type": type(error).__name__},
            )
            raise
        checkpoint(output, campaign)
        return await execute(campaign, output, manifest_path=manifest_path, settings=settings)


async def resume_campaign(
    output: Path,
    *,
    manifest_path: Path | None = None,
    settings: Settings | None = None,
) -> Campaign:
    with campaign_lock(output):
        campaign = Campaign.model_validate_json((output / "campaign.json").read_bytes())
        if campaign.status != "awaiting_runtime":
            raise ValueError("Only a planned manual runtime handoff can resume; no point retries")
        if session_id() != campaign.session_id:
            raise ValueError("Campaign belongs to a different boot/notebook session")
        settings = settings or Settings()
        if timeouts(settings) != campaign.timeouts:
            raise ValueError("Request deadlines changed during the campaign")
        validate_frozen(output, campaign)
        entry = next(entry for entry in campaign.schedule if entry.status == "pending")
        plan, _, _ = load_inputs(output, campaign.conditions[entry.condition])
        if plan.synthetic:
            if manifest_path is not None:
                raise ValueError("Synthetic campaigns cannot use GPU evidence")
        else:
            if manifest_path is None:
                raise ValueError("Supply a fresh manifest for the next manual runtime condition")
            check_runtime(campaign.conditions[entry.condition].config, manifest_path)
        campaign.status = "running"
        checkpoint(output, campaign)
        return await execute(campaign, output, manifest_path=manifest_path, settings=settings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, nargs="?")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-plan", type=Path)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    try:
        if args.resume is not None:
            if any(
                value is not None
                for value in (
                    args.experiment,
                    args.plan,
                    args.candidate,
                    args.candidate_plan,
                    args.output,
                    args.campaign_id,
                )
            ):
                parser.error("Resume uses frozen inputs; only --runtime-manifest may be supplied")
            output = args.resume
            campaign = asyncio.run(resume_campaign(output, manifest_path=args.runtime_manifest))
        else:
            if args.experiment is None or args.plan is None:
                parser.error("Supply one canonical experiment and its --plan")
            campaign_id = args.campaign_id or "campaign-" + uuid.uuid4().hex
            output = args.output or Path("results") / campaign_id
            campaign = asyncio.run(
                run_campaign(
                    args.experiment,
                    args.plan,
                    output,
                    campaign_id=campaign_id,
                    candidate=args.candidate,
                    candidate_plan=args.candidate_plan,
                    manifest_path=args.runtime_manifest,
                )
            )
        print(f"Campaign {campaign.campaign_id}: {campaign.status}; artifacts: {output}")
        if campaign.status == "awaiting_runtime":
            next_entry = next(entry for entry in campaign.schedule if entry.status == "pending")
            print(
                f"Next: {next_entry.condition}. Restart/probe its runtime manually, "
                "then --resume this directory."
            )
            return 2
        return 0 if campaign.complete else 1
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, yaml.YAMLError):
        print(
            "Campaign refused: check inputs, session, runtime and artifact integrity. "
            "Existing artifacts were preserved."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
