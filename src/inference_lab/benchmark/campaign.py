"""Frozen campaign inputs and schedules; point results remain authoritative."""

import json
import os
import platform
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, StrictBool, model_validator

from inference_lab.benchmark.plan import RequestPlan, digest, encoded
from inference_lab.benchmark.workload import validate_workload
from inference_lab.config.io import _read_yaml
from inference_lab.config.schemas import (
    ArtifactPath,
    ComparisonConfig,
    ComparisonDifference,
    ContractModel,
    Count,
    ExperimentConfig,
    ExperimentResult,
    Identifier,
    RegressionPolicy,
    Sha256,
    Text,
)

ConditionName = Literal["experiment", "baseline", "candidate"]


class Condition(ContractModel):
    config: ExperimentConfig
    files: dict[Text, ArtifactPath]
    sha256: dict[Text, Sha256]


class CampaignEntry(ContractModel):
    condition: ConditionName
    run_id: Identifier
    point: Count
    repetition: Count
    concurrency: Annotated[int, Field(strict=True, gt=0)]
    status: Literal["pending", "running", "completed", "failed", "interrupted"] = "pending"
    artifacts_sha256: dict[ArtifactPath, Sha256] = Field(default_factory=dict)


class Campaign(ContractModel):
    campaign_schema_version: Literal["1.0"] = "1.0"
    metric_contract_version: Literal["1.0"] = "1.0"
    campaign_id: Identifier
    session_id: Sha256
    synthetic: StrictBool
    conditions: dict[ConditionName, Condition]
    schedule: Annotated[list[CampaignEntry], Field(min_length=1)]
    timeouts: dict[Text, float]
    comparison: ComparisonConfig | None
    differences: list[ComparisonDifference]
    request_plan_sha256: dict[ConditionName, Sha256]
    status: Literal["running", "awaiting_runtime", "complete", "failed", "interrupted"] = "running"
    complete: StrictBool = False
    published: Literal[False] = False
    error: Text | None = None
    limitations: list[Text]

    @model_validator(mode="after")
    def coherent_schedule(self) -> "Campaign":
        paired = self.comparison is not None
        expected = {"baseline", "candidate"} if paired else {"experiment"}
        if set(self.conditions) != expected or set(self.request_plan_sha256) != expected:
            raise ValueError("Campaign conditions do not match its single/paired protocol")
        if len({entry.run_id for entry in self.schedule}) != len(self.schedule):
            raise ValueError("Campaign run IDs must be unique")
        seen = set()
        for entry in self.schedule:
            if entry.condition not in self.conditions:
                raise ValueError("Schedule references an unknown campaign condition")
            condition = self.conditions[entry.condition]
            config = condition.config
            if (
                entry.point >= len(config.workload.concurrency)
                or entry.concurrency != config.workload.concurrency[entry.point]
                or entry.repetition >= config.measurement.repetitions
            ):
                raise ValueError("Schedule differs from canonical experiment points/repetitions")
            identity = (entry.condition, entry.point, entry.repetition)
            if identity in seen:
                raise ValueError("Duplicate campaign point/repetition")
            seen.add(identity)
            if entry.status == "completed" and not entry.artifacts_sha256:
                raise ValueError("Completed entries require their artifact hashes")
        wanted = {
            (name, point, repetition)
            for name, condition in self.conditions.items()
            for point in range(len(condition.config.workload.concurrency))
            for repetition in range(condition.config.measurement.repetitions)
        }
        if seen != wanted:
            raise ValueError("Campaign schedule must cover every configured point/repetition")
        actual_order = [(e.condition, e.point, e.repetition, e.run_id) for e in self.schedule]
        expected_order = [
            (e.condition, e.point, e.repetition, e.run_id)
            for e in schedule(
                self.campaign_id, {name: c.config for name, c in self.conditions.items()}
            )
        ]
        if actual_order != expected_order:
            raise ValueError("Campaign execution order differs from its frozen protocol")
        incomplete_seen = False
        for entry in self.schedule:
            if entry.status != "completed":
                incomplete_seen = True
            elif incomplete_seen:
                raise ValueError("Completed runs must form a sequential prefix")
        if set(self.timeouts) != {"global_seconds", "idle_seconds"} or any(
            value <= 0 for value in self.timeouts.values()
        ):
            raise ValueError("Campaign needs finite positive global/idle deadlines")
        if self.complete != (self.status == "complete"):
            raise ValueError("Campaign status and completeness disagree")
        if self.complete and any(entry.status != "completed" for entry in self.schedule):
            raise ValueError("Complete campaigns require every run to finish successfully")
        if self.status == "awaiting_runtime" and (
            not any(entry.status == "completed" for entry in self.schedule)
            or not any(entry.status == "pending" for entry in self.schedule)
            or any(entry.status not in ("pending", "completed") for entry in self.schedule)
        ):
            raise ValueError("Runtime handoffs require completed runs and unstarted pending runs")
        return self


def session_id() -> str:
    """Same Linux boot/container evidence, independent of the CLI process PID."""
    try:
        identity = {
            "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "pid_namespace": os.readlink("/proc/self/ns/pid"),
            "hostname": platform.node(),
        }
    except OSError:
        raise ValueError("Campaigns require verifiable co-located Linux session evidence") from None
    return digest(json.dumps(identity, sort_keys=True).encode())


def paired_plan_hash(plan: RequestPlan) -> str:
    # Model routing differs for same-source AWQ. Dataset, template, lengths,
    # generation and request order must still match for these paired runs.
    data = plan.model_dump(mode="json")
    data.pop("model")
    for request in data["warmup"] + data["requests"]:
        request["payload"].pop("model")
    return digest(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())


def pair_protocol(
    configs: dict[ConditionName, ExperimentConfig],
    plans: dict[ConditionName, RequestPlan],
    policies: dict[ConditionName, RegressionPolicy | None],
) -> tuple[ComparisonConfig | None, list[ComparisonDifference]]:
    if set(configs) == {"experiment"}:
        return None, []
    baseline, candidate = configs["baseline"], configs["candidate"]
    policy = policies["baseline"]
    if policy is None or policy != policies["candidate"]:
        raise ValueError("Paired campaigns require the same canonical policy snapshot")
    if (
        baseline.workload.concurrency != candidate.workload.concurrency
        or baseline.measurement.repetitions != candidate.measurement.repetitions
    ):
        raise ValueError("Paired configurations require the same points and repetitions")
    if paired_plan_hash(plans["baseline"]) != paired_plan_hash(plans["candidate"]):
        raise ValueError("Paired campaigns require matched dataset/template/request plans")
    differences = []
    left, right = baseline.model_dump(mode="json"), candidate.model_dump(mode="json")
    for section in (
        "model",
        "runtime",
        "target",
        "workload",
        "generation",
        "warmup",
        "measurement",
    ):
        for key, value in left[section].items():
            if value != right[section][key]:
                differences.append(
                    ComparisonDifference(
                        path=f"{section}.{key}", baseline=value, candidate=right[section][key]
                    )
                )
    if {difference.path for difference in differences} - set(policy.comparison.allowed_changes):
        raise ValueError("Paired configuration differences were not explicitly allowed")
    quantization_changed = baseline.runtime.quantization != candidate.runtime.quantization
    tp_changed = baseline.runtime.tensor_parallel_size != candidate.runtime.tensor_parallel_size
    if quantization_changed or tp_changed:
        if policy.comparison.kind != "exploratory":
            raise ValueError("AWQ/TP comparisons must be exploratory")
        if baseline.runtime.dtype != candidate.runtime.dtype:
            raise ValueError("AWQ/TP experiments require the same activation dtype")
        if quantization_changed and tp_changed:
            raise ValueError("AWQ experiments require the same TP/GPU resources")
        if tp_changed and baseline.model != candidate.model:
            raise ValueError("TP experiments require the same checkpoint/tokenizer")
    for name, config in configs.items():
        policy.validate_experiment(
            config, published=not plans[name].synthetic and policy.comparison.kind == "regression"
        )
    return policy.comparison, differences


def schedule(
    campaign_id: str, configs: dict[ConditionName, ExperimentConfig]
) -> list[CampaignEntry]:
    first = next(iter(configs.values()))
    entries = []
    for point, concurrency in enumerate(first.workload.concurrency):
        for repetition in range(first.measurement.repetitions):
            order: tuple[ConditionName, ...] = ("experiment",)
            if "baseline" in configs:
                order = (
                    ("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")
                )
            for name in order:
                entries.append(
                    CampaignEntry(
                        condition=name,
                        run_id=f"{campaign_id}-{name}-p{point:03d}-r{repetition:03d}",
                        point=point,
                        repetition=repetition,
                        concurrency=concurrency,
                    )
                )
    return entries


def frozen_file(root: Path, reference: str, expected_hash: str) -> Path:
    path = root / reference
    if (
        not path.resolve().is_relative_to(root.resolve())
        or digest(path.read_bytes()) != expected_hash
    ):
        raise ValueError("Campaign artifact changed or escapes its output directory")
    return path


def load_inputs(root: Path, condition: Condition) -> tuple[RequestPlan, Path, Path | None]:
    if set(condition.files) != set(condition.sha256) or not {
        "experiment",
        "plan",
        "dataset",
    } <= set(condition.files):
        raise ValueError("Campaign input references are incomplete")
    files = {
        key: frozen_file(root, path, condition.sha256[key]) for key, path in condition.files.items()
    }
    config = ExperimentConfig.model_validate(_read_yaml(files["experiment"]))
    if config != condition.config:
        raise ValueError("Campaign index differs from the frozen canonical experiment")
    plan = RequestPlan.model_validate_json(files["plan"].read_bytes())
    validate_workload(config, plan, dataset_path=files["dataset"], policy_path=files.get("policy"))
    return plan, files["dataset"], files.get("policy")


def capture_condition(
    root: Path,
    name: ConditionName,
    config: ExperimentConfig,
    plan: RequestPlan,
    policy: RegressionPolicy | None,
) -> Condition:
    directory = root / "inputs" / name
    directory.mkdir(parents=True)
    data = {
        "experiment": yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False).encode(),
        "plan": encoded(plan),
        "dataset": Path(config.workload.dataset).read_bytes(),
    }
    if policy is not None:
        data["policy"] = encoded(policy)
        if policy.calibration_ref is not None:
            data["calibration"] = Path(policy.calibration_ref).read_bytes()
            if not data["calibration"]:
                raise ValueError("Calibration reference is empty")
    files, hashes = {}, {}
    for key, content in data.items():
        filename = {
            "experiment": "experiment.yaml",
            "plan": "plan.json",
            "dataset": "dataset.jsonl",
            "policy": "policy.json",
            "calibration": "calibration.artifact",
        }[key]
        path = directory / filename
        path.write_bytes(content)
        files[key] = path.relative_to(root).as_posix()
        hashes[key] = digest(content)
    return Condition(config=config, files=files, sha256=hashes)


def validate_completed(root: Path, campaign: Campaign) -> None:
    for entry in campaign.schedule:
        if entry.status != "completed":
            continue
        if (
            "result.json" not in entry.artifacts_sha256
            or "raw/requests.jsonl" not in entry.artifacts_sha256
        ):
            raise ValueError("Completed run has incomplete artifact references")
        for reference, sha256 in entry.artifacts_sha256.items():
            frozen_file(root, f"{entry.run_id}/{reference}", sha256)
        result = ExperimentResult.model_validate_json(
            (root / entry.run_id / "result.json").read_bytes()
        )
        if (
            result.campaign_id != campaign.campaign_id
            or result.run_id != entry.run_id
            or result.point != entry.point
            or result.repetition != entry.repetition
            or result.concurrency != entry.concurrency
            or result.config != campaign.conditions[entry.condition].config
            or not result.complete
            or not result.workload.conformant
            or result.counters.errors
            or result.published
        ):
            raise ValueError("Completed run does not match its frozen campaign identity")
