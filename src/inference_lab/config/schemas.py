"""Contract 1.0 schemas. These validate evidence; they do not run benchmarks or gates."""

import re
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StringConstraints,
    ValidationInfo,
    model_validator,
)

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[Text, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
Count = Annotated[int, Field(ge=0, strict=True)]
PositiveCount = Annotated[int, Field(gt=0, strict=True)]
Number = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Verdict = Literal["PASS", "FAIL", "INCONCLUSIVE", "INVALID"]


def public_url(value: AnyHttpUrl) -> AnyHttpUrl:
    if any(
        part is not None for part in (value.username, value.password, value.query, value.fragment)
    ):
        raise ValueError("HTTP endpoints must not contain credentials, queries, or fragments")
    return value


Endpoint = Annotated[AnyHttpUrl, AfterValidator(public_url)]


def relative_artifact(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value:
        raise ValueError("Artifact references must be relative POSIX paths without traversal")
    return value


ArtifactPath = Annotated[Text, AfterValidator(relative_artifact)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


class ModelConfig(ContractModel):
    repo_id: Text
    revision: Text
    tokenizer_repo_id: Text
    tokenizer_revision: Text
    trust_remote_code: Literal[False] = False

    def require_immutable(self) -> None:
        for revision in (self.revision, self.tokenizer_revision):
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError(
                    "Published model and tokenizer revisions must be immutable commits"
                )


class RuntimeConfig(ContractModel):
    # Required values prevent silently substituting runtime defaults as observed evidence.
    backend: Literal["vllm"]
    dtype: Literal["float16", "float32", "bfloat16"]
    quantization: Literal["awq"] | None
    tensor_parallel_size: PositiveCount
    max_model_len: PositiveCount
    max_num_seqs: PositiveCount
    max_num_batched_tokens: PositiveCount
    gpu_memory_utilization: Annotated[float, Field(gt=0, le=1, strict=True)]
    kv_cache_dtype: Text
    enable_prefix_caching: StrictBool
    enable_chunked_prefill: StrictBool
    enforce_eager: StrictBool

    @model_validator(mode="after")
    def scheduler_budget(self) -> Self:
        if not self.enable_chunked_prefill and self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("Without chunked prefill, the token budget must cover max_model_len")
        return self


class TargetConfig(ContractModel):
    kind: Literal["direct", "gateway"]
    url: Endpoint


class WorkloadConfig(ContractModel):
    dataset: Text
    load_model: Literal["closed_loop"]
    concurrency: Annotated[list[PositiveCount], Field(min_length=1)]
    seed: Count

    @model_validator(mode="after")
    def unique_points(self) -> Self:
        if len(set(self.concurrency)) != len(self.concurrency):
            raise ValueError("Concurrency points must be unique")
        return self


class GenerationConfig(ContractModel):
    stream: StrictBool
    n: Annotated[int, Field(strict=True, ge=1, le=1)]
    temperature: Number
    max_tokens: PositiveCount
    ignore_eos: StrictBool


class WarmupConfig(ContractModel):
    requests_per_point: Count


class MeasurementConfig(ContractModel):
    requests_per_run: PositiveCount
    repetitions: PositiveCount
    purpose: Literal["smoke", "exploratory", "regression"]


class ExperimentConfig(ContractModel):
    schema_version: Literal["1.0"]
    name: Identifier
    model: ModelConfig
    runtime: RuntimeConfig
    target: TargetConfig
    workload: WorkloadConfig
    generation: GenerationConfig
    warmup: WarmupConfig
    measurement: MeasurementConfig
    policy_file: Text | None

    @model_validator(mode="after")
    def valid_context(self, info: ValidationInfo) -> Self:
        if self.generation.max_tokens >= self.runtime.max_model_len:
            raise ValueError("Generation must leave context space for input tokens")
        if self.runtime.quantization == "awq" and self.runtime.dtype != "float16":
            raise ValueError("The qualified AWQ configuration requires float16 activations")
        if self.measurement.purpose == "regression" and self.policy_file is None:
            raise ValueError("Regression requires a policy_file")
        if info.context and info.context.get("published"):
            self.model.require_immutable()
        return self

    def validate_input_length(self, input_tokens: int) -> None:
        """Call with lengths measured after templating; never infer them from a filename."""
        if type(input_tokens) is not int or input_tokens < 0:
            raise ValueError("Input tokens must be a nonnegative integer")
        if input_tokens + self.generation.max_tokens > self.runtime.max_model_len:
            raise ValueError("Input plus requested output exceeds max_model_len")


MetricName = Literal[
    "client_ttft_p50_ms",
    "client_ttft_p95_ms",
    "client_ttft_p99_ms",
    "client_e2e_p50_ms",
    "client_e2e_p95_ms",
    "client_e2e_p99_ms",
    "client_tpot_estimate_p50_ms",
    "client_tpot_estimate_p95_ms",
    "client_tpot_estimate_p99_ms",
    "successful_requests_per_second",
    "successful_output_tokens_per_second",
    "error_rate",
]
METRIC_UNITS = {
    **{
        f"{prefix}_{percentile}_ms": "ms"
        for prefix in ("client_ttft", "client_e2e", "client_tpot_estimate")
        for percentile in ("p50", "p95", "p99")
    },
    "successful_requests_per_second": "requests/s",
    "successful_output_tokens_per_second": "tokens/s",
    "error_rate": "ratio",
}
COMPARISON_DIMENSIONS = frozenset(
    f"{prefix}.{field}"
    for prefix, schema in (
        ("model", ModelConfig),
        ("runtime", RuntimeConfig),
        ("target", TargetConfig),
        ("workload", WorkloadConfig),
        ("generation", GenerationConfig),
        ("warmup", WarmupConfig),
        ("measurement", MeasurementConfig),
    )
    for field in schema.model_fields
) | {
    "runtime.version",
    "runtime.engine",
    "runtime.attention_backend",
    "runtime.linear_backend",
    "runtime.kv_cache_dtype_resolved",
    "hardware.gpu_used_uuids",
    "hardware.topology",
    "hardware.gpu_names",
    "hardware.driver",
    "model.source_repo_id",
    "model.source_revision",
    "workload.dataset_sha256",
    "workload.dataset_version",
    "workload.plan_sha256",
    "workload.chat_template_sha256",
    "workload.template_parameters",
    "workload.input_tokens",
    "workload.output_tokens",
    "metric_contract_version",
    "protocol.warmup",
    "protocol.measurement",
    "protocol.concurrency",
    "protocol.request_plan_sha256",
    "protocol.timeouts",
    "hardware.gpu_used_count",
    "hardware.gpu_indices",
    "hardware.gpu_compute_capability",
    "hardware.gpu_vram_bytes",
    "hardware.cpu",
    "hardware.ram_bytes",
    "environment.python",
    "environment.os",
    "environment.driver_supported_cuda",
    "environment.pytorch_cuda_runtime",
    "environment.cuda_toolkit",
    "client.git_sha",
    "client.git_dirty",
}


class ComparisonConfig(ContractModel):
    kind: Literal["regression", "exploratory"]
    allowed_changes: list[Text]

    @model_validator(mode="after")
    def explicit_dimensions(self) -> Self:
        if len(set(self.allowed_changes)) != len(self.allowed_changes):
            raise ValueError("Allowed changes must be unique")
        if any(
            path not in COMPARISON_DIMENSIONS
            and not re.fullmatch(r"dependencies\.[a-z0-9]+(?:-[a-z0-9]+)*", path)
            for path in self.allowed_changes
        ):
            raise ValueError("Allowed changes must name supported exact paths, without wildcards")
        return self


class SloThreshold(ContractModel):
    max: Number


class RegressionThreshold(ContractModel):
    max_relative_degradation: Number
    noise_margin: Number

    @model_validator(mode="after")
    def valid_margin(self) -> Self:
        if self.noise_margin > self.max_relative_degradation:
            raise ValueError("Noise margin must not make the PASS threshold negative")
        return self


class RegressionPolicy(ContractModel):
    schema_version: Literal["1.0"]
    policy_id: Identifier
    workload_id: Identifier
    calibration_ref: ArtifactPath | None
    required_metrics: Annotated[list[MetricName], Field(min_length=1)]
    min_samples_per_metric: dict[MetricName, PositiveCount]
    min_repetitions: PositiveCount
    slo: dict[MetricName, SloThreshold]
    regression: dict[MetricName, RegressionThreshold]
    comparison: ComparisonConfig

    @model_validator(mode="after")
    def consistent_clauses(self) -> Self:
        required = set(self.required_metrics)
        if len(required) != len(self.required_metrics):
            raise ValueError("Required metrics must be unique")
        if not self.slo and not self.regression:
            raise ValueError("A policy needs at least one SLO or regression clause")
        if (set(self.slo) | set(self.regression)) - required:
            raise ValueError("Every policy clause must be a required metric")
        if required - self.min_samples_per_metric.keys():
            raise ValueError("Every required metric needs a minimum sample count")
        for name, limit in self.slo.items():
            if METRIC_UNITS[name] in ("requests/s", "tokens/s"):
                raise ValueError("SLO max clauses support latencies and error_rate, not throughput")
            if name == "error_rate" and limit.max > 1:
                raise ValueError("error_rate uses a ratio between zero and one")
        return self

    def validate_experiment(self, experiment: ExperimentConfig, *, published: bool = False) -> None:
        if not experiment.generation.stream and any(
            metric.startswith(("client_ttft_", "client_tpot_estimate_"))
            for metric in self.required_metrics
        ):
            raise ValueError("Client TTFT/TPOT policies require streaming")
        if experiment.measurement.repetitions < self.min_repetitions:
            raise ValueError("Experiment has fewer repetitions than its policy requires")
        if published and (self.calibration_ref is None or self.min_repetitions < 3):
            raise ValueError("Published gates need calibration and at least three repetitions")


class MetricValue(ContractModel):
    value: Number | None
    unit: Literal["ms", "requests/s", "tokens/s", "ratio"]
    source: Literal["client", "runtime", "gpu"]
    sample_count: Count
    population: Literal["successful_requests", "all_attempts"]
    reason: Text | None

    @model_validator(mode="after")
    def explicit_missing_value(self) -> Self:
        if self.value is None and self.reason is None:
            raise ValueError("Missing metric values require a reason")
        if self.value is not None and (self.reason is not None or self.sample_count == 0):
            raise ValueError("Measured values need observations and no missing-value reason")
        return self


MissingRequestField = Literal[
    "request_start_ns",
    "first_content_ns",
    "last_content_ns",
    "terminal_ns",
    "input_tokens",
    "output_tokens",
    "token_count_source",
    "finish_reason",
]


class RequestMeasurement(ContractModel):
    request_id: Identifier
    dataset_request_id: Identifier
    plan_index: Count
    transport_status: Literal["success", "error", "timeout", "cancelled", "incomplete"]
    workload_status: Literal["conformant", "nonconformant", "not_checked"]
    request_start_ns: Count | None
    first_content_ns: Count | None
    last_content_ns: Count | None
    terminal_ns: Count | None
    content_events: Count
    input_tokens: Count | None
    output_tokens: Count | None
    token_count_source: Literal["server_usage"] | None
    finish_reason: Text | None
    error: Text | None
    missing_reasons: dict[MissingRequestField, Text]

    @model_validator(mode="after")
    def consistent_stream(self) -> Self:
        nullable = (
            "request_start_ns",
            "first_content_ns",
            "last_content_ns",
            "terminal_ns",
            "input_tokens",
            "output_tokens",
            "token_count_source",
            "finish_reason",
        )
        for field in nullable:
            if (getattr(self, field) is None) != (field in self.missing_reasons):
                raise ValueError(f"A missing reason is required exactly when {field} is null")
        times = [
            t
            for t in (
                self.request_start_ns,
                self.first_content_ns,
                self.last_content_ns,
                self.terminal_ns,
            )
            if t is not None
        ]
        if times != sorted(times):
            raise ValueError("Request timestamps must use one ordered monotone clock")
        if (self.first_content_ns is None) != (self.last_content_ns is None):
            raise ValueError("First and last content timestamps must be present together")
        if (self.content_events == 0) != (self.first_content_ns is None):
            raise ValueError("Content timestamps must agree with observed events")
        if self.transport_status == "success":
            if self.finish_reason not in ("stop", "length"):
                raise ValueError(
                    "Success requires a finish reason supported by the qualified subset"
                )
            if self.error is not None or any(
                getattr(self, field) is None
                for field in (
                    "request_start_ns",
                    "terminal_ns",
                    "input_tokens",
                    "output_tokens",
                    "token_count_source",
                    "finish_reason",
                )
            ):
                raise ValueError("Success requires valid termination, server usage, and no error")
        elif self.error is None:
            raise ValueError("Unsuccessful requests require a normalized error")
        return self


class GPUDevice(ContractModel):
    uuid: Text
    index: Count
    name: Text
    vram_bytes: PositiveCount
    compute_capability: tuple[Count, Count]


class Environment(ContractModel):
    python: Text
    os: Text
    cpu: Text
    ram_bytes: PositiveCount
    gpu_present: list[GPUDevice]
    gpu_visible_uuids: list[Text]
    gpu_used_uuids: list[Text]
    topology: Text | None
    nvidia_driver: Text | None
    driver_supported_cuda: Text | None
    pytorch_cuda_runtime: Text | None
    cuda_toolkit: Text | None
    torch_version: Text | None
    vllm_version: Text | None
    triton_version: Text | None
    git_sha: Text | None
    git_dirty: StrictBool
    missing_reasons: dict[Text, Text]

    @model_validator(mode="after")
    def observed_devices(self) -> Self:
        present = {gpu.uuid for gpu in self.gpu_present}
        visible, used = set(self.gpu_visible_uuids), set(self.gpu_used_uuids)
        if (
            len(present) != len(self.gpu_present)
            or len(visible) != len(self.gpu_visible_uuids)
            or len(used) != len(self.gpu_used_uuids)
        ):
            raise ValueError("GPU identities must be unique")
        if not used <= visible <= present:
            raise ValueError("Used GPUs must be visible and visible GPUs must be present")
        if self.git_sha is not None and not re.fullmatch(r"[0-9a-f]{40}", self.git_sha):
            raise ValueError("Git SHA must be an immutable commit")
        nullable = (
            "topology",
            "nvidia_driver",
            "driver_supported_cuda",
            "pytorch_cuda_runtime",
            "cuda_toolkit",
            "torch_version",
            "vllm_version",
            "triton_version",
            "git_sha",
        )
        if set(self.missing_reasons) != {
            field for field in nullable if getattr(self, field) is None
        }:
            raise ValueError("Environment missing reasons must identify exactly the null fields")
        return self


class WorkloadEvidence(ContractModel):
    dataset_name: Identifier
    dataset_version: Text
    dataset_sha256: Sha256
    plan_sha256: Sha256
    tokenizer_repo_id: Text
    tokenizer_revision: Text
    chat_template_sha256: Sha256
    template_parameters: dict[Text, JsonValue]
    input_tokens_min: Count | None
    input_tokens_max: Count | None
    output_tokens_min: Count | None
    output_tokens_max: Count | None
    conformant: StrictBool
    missing_reasons: dict[Text, Text] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ordered_lengths(self) -> Self:
        fields = ("input_tokens_min", "input_tokens_max", "output_tokens_min", "output_tokens_max")
        if set(self.missing_reasons) != {field for field in fields if getattr(self, field) is None}:
            raise ValueError("Missing observed lengths require explicit reasons")
        if self.conformant and self.missing_reasons:
            raise ValueError("Conformant workloads require observed token lengths")
        for low, high in (
            (self.input_tokens_min, self.input_tokens_max),
            (self.output_tokens_min, self.output_tokens_max),
        ):
            if low is not None and high is not None and low > high:
                raise ValueError("Observed token-length ranges must be ordered")
        return self


class RuntimeEvidence(ContractModel):
    resolved: RuntimeConfig | None
    attention_backend: Text | None
    linear_backend: Text | None
    kv_cache_dtype_resolved: Text | None
    source: Literal["startup_log", "runtime_api", "synthetic"]
    evidence_artifact: ArtifactPath
    missing_reasons: dict[Text, Text]

    @model_validator(mode="after")
    def unverified_values(self) -> Self:
        nullable = ("resolved", "attention_backend", "linear_backend", "kv_cache_dtype_resolved")
        if set(self.missing_reasons) != {
            field for field in nullable if getattr(self, field) is None
        }:
            raise ValueError("Runtime missing reasons must identify exactly the unverified fields")
        return self


class Counters(ContractModel):
    attempts: Count
    successes: Count
    errors: Count

    @model_validator(mode="after")
    def partition_attempts(self) -> Self:
        if self.successes + self.errors != self.attempts:
            raise ValueError("Successes and errors must partition all attempts")
        return self


class ExperimentRun(ContractModel):
    campaign_id: Identifier
    run_id: Identifier
    point: Count
    repetition: Count
    concurrency: PositiveCount
    config: ExperimentConfig

    @model_validator(mode="after")
    def one_point_per_run(self) -> Self:
        if self.point >= len(self.config.workload.concurrency):
            raise ValueError("Point index is outside the experiment")
        if self.config.workload.concurrency[self.point] != self.concurrency:
            raise ValueError("Run concurrency does not match its point")
        if self.repetition >= self.config.measurement.repetitions:
            raise ValueError("Repetition index is outside the experiment")
        return self


class ExperimentResult(ExperimentRun):
    result_schema_version: Literal["1.0"]
    metric_contract_version: Literal["1.0"]
    complete: StrictBool
    published: StrictBool
    synthetic: StrictBool
    qualification_id: Identifier | None
    environment: Environment
    workload: WorkloadEvidence
    runtime_manifest: RuntimeEvidence
    counters: Counters
    metrics: dict[MetricName, MetricValue]
    percentile_method: Literal["linear"]
    limitations: list[Text]
    artifacts: dict[Text, ArtifactPath]
    policy: RegressionPolicy | None = None

    @model_validator(mode="after")
    def consistent_evidence(self) -> Self:
        if set(self.metrics) != set(METRIC_UNITS):
            raise ValueError("All canonical metrics must be explicit, including null values")
        for name, metric in self.metrics.items():
            if metric.unit != METRIC_UNITS[name] or metric.source != "client":
                raise ValueError("Canonical benchmark metrics need their client source and units")
            population = "successful_requests" if metric.unit == "ms" else "all_attempts"
            count = self.counters.successes if metric.unit == "ms" else self.counters.attempts
            if metric.population != population or metric.sample_count > count:
                raise ValueError("Metric population or coverage does not match request counters")
            if name == "error_rate" and metric.value is not None and metric.value > 1:
                raise ValueError("error_rate must be a ratio between zero and one")
        if self.workload.conformant:
            assert self.workload.input_tokens_max is not None
            assert self.workload.output_tokens_max is not None
            self.config.validate_input_length(self.workload.input_tokens_max)
            if self.workload.output_tokens_max > self.config.generation.max_tokens:
                raise ValueError("Observed output exceeds requested generation limit")
        elif not self.limitations:
            raise ValueError("Nonconformant workloads need an explicit limitation")
        if (
            self.workload.conformant
            and self.config.generation.ignore_eos
            and (
                self.workload.output_tokens_min != self.config.generation.max_tokens
                or self.workload.output_tokens_max != self.config.generation.max_tokens
            )
        ):
            raise ValueError("Controlled-length workload does not match observed output lengths")
        if (
            self.workload.tokenizer_repo_id != self.config.model.tokenizer_repo_id
            or self.workload.tokenizer_revision != self.config.model.tokenizer_revision
        ):
            raise ValueError("Workload tokenizer must match the resolved model configuration")
        used = self.environment.gpu_used_uuids
        if used and len(used) != self.config.runtime.tensor_parallel_size:
            raise ValueError("TP must match the number of GPUs actually used")
        if self.config.runtime.dtype == "bfloat16" and any(
            gpu.uuid in used and gpu.compute_capability == (7, 5)
            for gpu in self.environment.gpu_present
        ):
            raise ValueError("T4 execution does not support bfloat16")
        if self.synthetic:
            if (
                used
                or self.runtime_manifest.source != "synthetic"
                or self.qualification_id is not None
            ):
                raise ValueError("Synthetic fixtures cannot claim GPU execution or qualification")
        elif self.runtime_manifest.source == "synthetic":
            raise ValueError("Real runs cannot use synthetic runtime evidence")
        if (
            self.runtime_manifest.resolved is not None
            and self.runtime_manifest.resolved != self.config.runtime
        ):
            raise ValueError(
                "Result runtime differs from observed runtime; keep expected config separate"
            )
        if self.published:
            self.config.model.require_immutable()
            if not self.synthetic and (not used or self.qualification_id is None):
                raise ValueError("Published GPU results need GPU identities and qualification_id")
            if self.config.policy_file is not None and self.policy is None:
                raise ValueError("Published policy-driven results require the policy snapshot")
        if self.policy is not None:
            if self.config.policy_file is None:
                raise ValueError("Policy snapshots require the canonical policy_file reference")
            if self.policy.workload_id != self.workload.dataset_name:
                raise ValueError("Policy workload_id does not match the observed workload")
            self.policy.validate_experiment(
                self.config, published=self.published and not self.synthetic
            )
        if self.complete and self.counters.attempts != self.config.measurement.requests_per_run:
            raise ValueError("A complete run must account for every planned attempt")
        if not {"requests", "gpu"} <= self.artifacts.keys():
            raise ValueError("Results must reference raw request and GPU artifacts")
        return self


class ComparisonDifference(ContractModel):
    path: Text
    baseline: JsonValue
    candidate: JsonValue


class ComparisonReason(ContractModel):
    code: Identifier
    message: Text
    field: Text | None = None


SignedNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class ComparisonClause(ContractModel):
    point: Count
    metric: MetricName
    kind: Literal["slo", "regression"]
    verdict: Verdict
    threshold: Number
    noise_margin: Number | None
    repetitions: list[Count]
    required_repetitions: PositiveCount
    min_samples: PositiveCount
    baseline_values: list[Number | None]
    candidate_values: list[Number | None]
    baseline_sample_counts: list[Count]
    candidate_sample_counts: list[Count]
    relative_changes: list[SignedNumber]
    min_relative_change: SignedNumber | None
    max_relative_change: SignedNumber | None
    violating_run_ids: list[Identifier]
    reason: Text


class ComparisonResult(ContractModel):
    schema_version: Literal["1.0"]
    metric_contract_version: Literal["1.0"]
    baseline_run_ids: list[Identifier]
    candidate_run_ids: list[Identifier]
    comparison: ComparisonConfig | None
    verdict: Verdict
    reasons: list[ComparisonReason]
    differences: list[ComparisonDifference]
    clauses: list[ComparisonClause] = Field(default_factory=list)
    policy_sha256: Sha256 | None = None
    synthetic: StrictBool | None = None
    published: Literal[False] = False
    limitations: list[Text] = Field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2, "INVALID": 2}[self.verdict]

    @model_validator(mode="after")
    def defensible_verdict(self) -> Self:
        if self.verdict != "PASS" and not self.reasons:
            raise ValueError("Non-PASS verdicts require structured reasons")
        if self.verdict != "INVALID" and (
            self.comparison is None or not self.baseline_run_ids or not self.candidate_run_ids
        ):
            raise ValueError("Non-invalid comparisons require declared conditions and run IDs")
        unauthorized = {d.path for d in self.differences} - set(
            self.comparison.allowed_changes if self.comparison is not None else []
        )
        if unauthorized and self.verdict != "INVALID":
            raise ValueError("Unauthorized comparison differences require INVALID")
        if self.verdict in ("PASS", "FAIL") and len(self.baseline_run_ids) != len(
            self.candidate_run_ids
        ):
            raise ValueError("Conclusive comparisons require paired repetitions")
        if self.verdict != "INVALID" and len(
            set(self.baseline_run_ids + self.candidate_run_ids)
        ) != len(self.baseline_run_ids + self.candidate_run_ids):
            raise ValueError("Comparison run identities must be distinct")
        return self
