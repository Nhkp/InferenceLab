"""Offline, read-only campaign verification before any statistical decision."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, JsonValue, StrictBool

from inference_lab.benchmark.campaign import (
    Campaign,
    ConditionName,
    frozen_file,
    load_inputs,
    paired_plan_hash,
)
from inference_lab.benchmark.metrics import aggregate
from inference_lab.benchmark.plan import digest, encoded
from inference_lab.config.io import _read_yaml, load_policy
from inference_lab.config.schemas import (
    ContractModel,
    Environment,
    ExperimentConfig,
    ExperimentResult,
    Identifier,
    MetricName,
    PositiveCount,
    RequestMeasurement,
    Sha256,
    Text,
)


class CalibrationEvidence(ContractModel):
    schema_version: str = Field(pattern=r"^1\.0$")
    synthetic: StrictBool
    policy_sha256: Sha256
    workload_id: Identifier
    baseline_run_sha256: dict[Identifier, Sha256] = Field(min_length=2)
    sample_counts: dict[MetricName, PositiveCount]
    protocol: Text
    volume_justification: Text
    noise_justification: Text


@dataclass
class ConditionArtifacts:
    root: Path
    campaign: Campaign
    condition: ConditionName
    results: list[ExperimentResult]
    metadata: dict[str, dict[str, JsonValue]]
    intervals: dict[str, tuple[int, int]]
    calibration: bytes | None
    result_hashes: set[str]


def dependencies(path: Path) -> dict[str, JsonValue]:
    values: dict[str, JsonValue] = {}
    text = path.read_text()
    if text.strip() == "UNAVAILABLE":
        return {}
    for line in text.splitlines():
        name, separator, version = line.partition("==")
        if not separator or not name or not version or "\n" in version:
            raise ValueError("Package inventory is not a name/version snapshot")
        name = re.sub(r"[-_.]+", "-", name).lower()
        key = "runtime.version" if name == "vllm" else "dependencies." + name
        if key in values and values[key] != version:
            raise ValueError("Conflicting package versions in dependency snapshot")
        values[key] = version
    return values


def read_condition(root: Path, *, role: str) -> ConditionArtifacts:
    campaign = Campaign.model_validate_json((root / "campaign.json").read_bytes())
    if not campaign.complete or campaign.status != "complete":
        raise ValueError("Incomplete campaigns cannot receive a gate verdict")
    condition: ConditionName = "experiment"
    if "experiment" not in campaign.conditions:
        condition = "baseline" if role == "baseline" else "candidate"
    spec = campaign.conditions[condition]
    plan, dataset_path, policy_path = load_inputs(root, spec)
    if (
        paired_plan_hash(plan) != campaign.request_plan_sha256[condition]
        or plan.synthetic != campaign.synthetic
    ):
        raise ValueError("Campaign plan identity differs from its frozen inputs")
    policy = load_policy(policy_path) if policy_path is not None else None
    results = []
    metadata = {}
    intervals = {}
    hashes = set()
    for entry in campaign.schedule:
        if entry.condition != condition:
            continue
        if entry.status != "completed":
            raise ValueError("Condition contains an incomplete run")
        required = {
            "result.json",
            "experiment.yaml",
            "plan.json",
            "dataset.jsonl",
            "environment.json",
            "runtime_manifest.json",
            "pip-freeze.txt",
            "drain.json",
            "raw/requests.jsonl",
            "raw/warmup.jsonl",
            "raw/gpu.jsonl",
        }
        if not required <= entry.artifacts_sha256.keys():
            raise ValueError("Run is missing required hashed measurement evidence")
        paths = {
            name: frozen_file(root, f"{entry.run_id}/{name}", sha)
            for name, sha in entry.artifacts_sha256.items()
        }
        result = ExperimentResult.model_validate_json(paths["result.json"].read_bytes())
        if (
            result.config != spec.config
            or result.campaign_id != campaign.campaign_id
            or result.run_id != entry.run_id
            or result.point != entry.point
            or result.repetition != entry.repetition
            or result.concurrency != entry.concurrency
            or result.synthetic != campaign.synthetic
            or not result.complete
            or result.policy != policy
        ):
            raise ValueError("Run identity/configuration/policy differs from its campaign")
        if result.workload.dataset_sha256 != digest(
            dataset_path.read_bytes()
        ) or result.workload.plan_sha256 != digest(encoded(plan)):
            raise ValueError("Result dataset/plan hashes differ from frozen inputs")
        if (
            result.workload.dataset_name != plan.dataset_name
            or result.workload.dataset_version != plan.dataset_version
            or result.workload.chat_template_sha256 != plan.chat_template_sha256
            or result.workload.template_parameters != plan.template_parameters
        ):
            raise ValueError("Result workload/template metadata differs from its prepared plan")
        if (
            paths["plan.json"].read_bytes() != encoded(plan)
            or paths["dataset.jsonl"].read_bytes() != dataset_path.read_bytes()
        ):
            raise ValueError("Run input copies differ from their campaign snapshot")
        if (
            Environment.model_validate_json(paths["environment.json"].read_bytes())
            != result.environment
        ):
            raise ValueError("Environment snapshot differs from the result")
        if ExperimentConfig.model_validate(_read_yaml(paths["experiment.yaml"])) != result.config:
            raise ValueError("Experiment snapshot differs from the result")
        if any(reference not in paths for reference in result.artifacts.values()):
            raise ValueError("Result references missing artifacts")
        if result.synthetic:
            if json.loads(paths["runtime_manifest.json"].read_text()).get("synthetic") is not True:
                raise ValueError("Synthetic runtime identity differs from its result")
        else:
            actual = json.loads(paths["runtime_manifest.json"].read_text())
            if (
                actual.get("manifest_schema_version") != "1.0"
                or actual.get("status") != "VALIDATED"
                or actual.get("mismatches")
            ):
                raise ValueError("Runtime evidence is unverified or incompatible")
            expected = ExperimentConfig.model_validate(actual["expected"]["experiment"])
            if (
                expected.model != result.config.model
                or expected.runtime != result.config.runtime
                or actual.get("qualification_id") != result.qualification_id
            ):
                raise ValueError("Observed runtime identity differs from the result")
            observed = actual["observed"]
            for section, values in (
                ("model", result.config.model.model_dump()),
                ("runtime", result.config.runtime.model_dump()),
            ):
                for field, value in values.items():
                    if field in ("trust_remote_code", "backend"):
                        continue
                    value = "none" if field == "quantization" and value is None else value
                    observation = observed[section + "." + field]
                    if observation["value"] != value or observation.get("source") == "unavailable":
                        raise ValueError(
                            "Effective runtime differs from its recorded configuration"
                        )
            used = observed["gpu.used"]["value"]
            if not isinstance(used, list) or {row["uuid"] for row in used} != set(
                result.environment.gpu_used_uuids
            ):
                raise ValueError("Runtime used-GPU placement differs from its environment")
            for field in ("attention_backend", "linear_backend", "kv_cache_dtype_resolved"):
                if observed["runtime." + field]["value"] != getattr(result.runtime_manifest, field):
                    raise ValueError("Observed backend snapshot differs from its result")
            if observed["runtime.version"]["value"] != result.environment.vllm_version:
                raise ValueError("Runtime version differs from its environment snapshot")
        requests = [
            RequestMeasurement.model_validate_json(line)
            for line in paths["raw/requests.jsonl"].read_bytes().splitlines()
            if line.strip()
        ]
        if (
            len(requests) != len(plan.requests)
            or len({r.request_id for r in requests}) != len(requests)
            or {r.plan_index for r in requests} != set(range(len(plan.requests)))
        ):
            raise ValueError("Raw requests do not account for each planned attempt exactly once")
        for request in requests:
            planned = plan.requests[request.plan_index]
            if (
                request.dataset_request_id != planned.dataset_request_id
                or request.request_id != f"{result.run_id}-{request.plan_index:06d}"
            ):
                raise ValueError("Raw request identity differs from its prepared plan")
            if request.transport_status == "success" and (
                request.workload_status != "conformant"
                or request.input_tokens != planned.input_tokens
                or request.output_tokens != result.config.generation.max_tokens
            ):
                raise ValueError("Successful request does not conform to its workload")
        if aggregate(requests) != result.metrics:
            raise ValueError("Stored metrics differ from canonical raw-record replay")
        successes = sum(r.transport_status == "success" for r in requests)
        if (len(requests), successes, len(requests) - successes) != (
            result.counters.attempts,
            result.counters.successes,
            result.counters.errors,
        ):
            raise ValueError("Stored counters differ from raw records")
        for prefix in ("input", "output"):
            counts = [
                getattr(r, prefix + "_tokens")
                for r in requests
                if getattr(r, prefix + "_tokens") is not None
            ]
            if (min(counts) if counts else None) != getattr(
                result.workload, prefix + "_tokens_min"
            ) or (max(counts) if counts else None) != getattr(
                result.workload, prefix + "_tokens_max"
            ):
                raise ValueError("Observed token ranges differ from raw usage")
        warmup = [
            RequestMeasurement.model_validate_json(line)
            for line in paths["raw/warmup.jsonl"].read_bytes().splitlines()
            if line.strip()
        ]
        if len(warmup) != len(plan.warmup) or any(
            r.transport_status != "success" or r.workload_status != "conformant" for r in warmup
        ):
            raise ValueError("Warmup was not completed successfully")
        if {r.plan_index for r in warmup} != set(range(len(plan.warmup))):
            raise ValueError("Warmup does not account for every planned request")
        for request in warmup:
            planned = plan.warmup[request.plan_index]
            if (
                request.request_id != f"{result.run_id}-warmup-{request.plan_index:06d}"
                or request.dataset_request_id != planned.dataset_request_id
                or request.input_tokens != planned.input_tokens
                or request.output_tokens != result.config.generation.max_tokens
            ):
                raise ValueError("Warmup identity/usage differs from its plan")
        if any(r.terminal_ns is None or r.request_start_ns is None for r in requests + warmup):
            raise ValueError("Raw attempt boundaries are missing")
        starts = [r.request_start_ns for r in requests if r.request_start_ns is not None]
        terminals = [r.terminal_ns for r in requests if r.terminal_ns is not None]
        if not starts or max(terminals) <= min(starts):
            raise ValueError("No positive-duration measured attempt window")
        first = min(starts)
        last = max(terminals)
        if warmup and max(r.terminal_ns for r in warmup if r.terminal_ns is not None) > first:
            raise ValueError("Warmup overlaps measurement")
        drain = json.loads(paths["drain.json"].read_text())
        if any(
            drain.get(key) != "VALIDATED"
            for key in ("before_warmup", "after_warmup", "after_measurement")
        ):
            raise ValueError("Run lacks completed scheduler idle checks")
        intervals[result.run_id] = (
            min(r.request_start_ns for r in requests + warmup if r.request_start_ns is not None),
            last,
        )
        extra = dependencies(paths["pip-freeze.txt"])
        if not result.synthetic and not extra:
            raise ValueError("Real runs require package inventory")
        if not result.synthetic:
            if extra.get("runtime.version") != result.environment.vllm_version:
                raise ValueError("Package inventory differs from the runtime version")
            for name in ("torch", "triton"):
                if extra.get("dependencies." + name) != getattr(
                    result.environment, name + "_version"
                ):
                    raise ValueError("Package inventory differs from the environment")
            actual = json.loads(paths["runtime_manifest.json"].read_text())
            for name in ("torch", "triton", "vllm"):
                if actual["observed"]["packages." + name]["value"] != getattr(
                    result.environment, name + "_version"
                ):
                    raise ValueError("Runtime packages differ from the environment")
            extra["runtime.engine"] = actual["observed"]["runtime.engine"]["value"]
            if extra["runtime.engine"] is None:
                raise ValueError("Runtime engine is not observed")
        extra.update(
            {
                "protocol.request_plan_sha256": paired_plan_hash(plan),
                "protocol.timeouts": dict(campaign.timeouts),
            }
        )
        if not result.synthetic and result.config.runtime.quantization == "awq":
            # Source proof is opt-in measured evidence, not inferred from the repo name.
            if "provenance.json" in paths:
                provenance = json.loads(paths["provenance.json"].read_text())
                source, quantized = provenance["source_model"], provenance["quantized_model"]
                if (
                    quantized["repo_id"] != result.config.model.repo_id
                    or quantized["revision"] != result.config.model.revision
                    or quantized["base_model"] != source["repo_id"]
                    or source["chat_template_sha256"] != result.workload.chat_template_sha256
                    or any(
                        provenance.get(key) is not True
                        for key in ("same_architecture", "same_vocabulary", "same_chat_template")
                    )
                ):
                    raise ValueError("AWQ provenance does not bind this checkpoint and tokenizer")
                extra["model.source_repo_id"] = source["repo_id"]
                extra["model.source_revision"] = source["revision"]
        elif result.config.runtime.quantization is None:
            extra["model.source_repo_id"] = result.config.model.repo_id
            extra["model.source_revision"] = result.config.model.revision
        metadata[result.run_id] = extra
        results.append(result)
        hashes.add(entry.artifacts_sha256["result.json"])
    calibration = None
    if "calibration" in spec.files:
        calibration = frozen_file(
            root, spec.files["calibration"], spec.sha256["calibration"]
        ).read_bytes()
    return ConditionArtifacts(
        root, campaign, condition, results, metadata, intervals, calibration, hashes
    )
