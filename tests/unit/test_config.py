import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from inference_lab.config.io import load_experiment, load_policy
from inference_lab.config.schemas import (
    ComparisonResult,
    ExperimentConfig,
    ExperimentResult,
    ExperimentRun,
    MetricValue,
    RegressionPolicy,
    RequestMeasurement,
)
from inference_lab.config.settings import Settings

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "experiments/qwen3b-fp16.yaml"
DATA = ROOT / "tests/data"


def example() -> dict:
    return yaml.safe_load(EXAMPLE.read_text())


def test_canonical_roundtrip_and_observed_context_limit(tmp_path: Path) -> None:
    config = load_experiment(EXAMPLE, published=True)
    assert ExperimentConfig.model_validate_json(config.model_dump_json()) == config
    path = tmp_path / "roundtrip.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    assert load_experiment(path) == config
    assert not config.runtime.enable_chunked_prefill
    config.validate_input_length(3968)
    with pytest.raises(ValueError, match="exceeds"):
        config.validate_input_length(3969)
    for length in (-1, True, 1.5):
        with pytest.raises(ValueError):
            config.validate_input_length(length)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (None, "schema_version", "2.0"),
        (None, "schema_version", 1.0),
        (None, "slo", {"client_ttft_ms": 500}),
        ("runtime", "dtype", "fp16"),
        ("runtime", "tensor_parallel_size", 0),
        ("runtime", "tensor_parallel_size", True),
        ("runtime", "gpu_memory_utilization", float("nan")),
        ("runtime", "gpu_memory_utilization", float("inf")),
        ("runtime", "gpu_memory_utilization", 1.01),
        ("runtime", "max_num_batched_tokens", 100),
        ("workload", "concurrency", []),
        ("workload", "concurrency", [1, 1]),
        ("generation", "max_tokens", 4096),
        ("generation", "temperature", float("inf")),
        ("generation", "n", 2),
        ("generation", "n", True),
        ("generation", "stream", "false"),
        ("runtime", "enable_chunked_prefill", "false"),
        ("target", "url", "http://user:secret@localhost/v1"),
        ("target", "url", "http://localhost/v1?api_key=secret"),
    ],
)
def test_invalid_experiment_values(section: str | None, field: str, value: object) -> None:
    data = example()
    (data if section is None else data[section])[field] = value
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(data)


@pytest.mark.parametrize("revision", ["main", "<IMMUTABLE_COMMIT_REQUIRED>", "a" * 39])
def test_mobile_revisions_allowed_for_drafts_only(revision: str) -> None:
    data = example()
    data["model"]["revision"] = revision
    ExperimentConfig.model_validate(data)
    with pytest.raises(ValidationError, match="immutable"):
        ExperimentConfig.model_validate(data, context={"published": True})


@pytest.mark.parametrize(
    "text",
    [
        'schema_version: "1.0"\nschema_version: "2.0"\n',
        "runtime:\n  dtype: float16\n  dtype: bfloat16\n",
        "1: value\n",
        '!!python/object/apply:os.system ["exit 1"]',
    ],
)
def test_yaml_rejects_duplicate_keys_and_unsafe_tags(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises((ValueError, yaml.YAMLError)):
        load_experiment(path)


def test_policy_loading_streaming_and_calibration(tmp_path: Path) -> None:
    policy_path = DATA / "synthetic-policy.json"
    policy = load_policy(policy_path)
    assert policy.calibration_ref is None
    assert RegressionPolicy.model_validate_json(policy.model_dump_json()) == policy
    data = example()
    data["policy_file"] = "tests/data/synthetic-policy.json"
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(data))
    load_experiment(path, project_root=ROOT)
    with pytest.raises(ValueError, match="calibration"):
        load_experiment(path, project_root=ROOT, published=True)
    data["generation"]["stream"] = False
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="streaming"):
        load_experiment(path, project_root=ROOT)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": "2.0"},
        {"comparison": {"kind": "regression", "allowed_changes": ["runtime.*"]}},
        {"comparison": {"kind": "regression", "allowed_changes": ["unknown.field"]}},
        {
            "regression": {
                "client_ttft_p95_ms": {"max_relative_degradation": 0.1, "noise_margin": 0.2}
            }
        },
        {"min_samples_per_metric": {}},
        {"slo": {}, "regression": {}},
        {"slo": {"client_ttft_p95_ms": {"max": float("nan")}}},
    ],
)
def test_invalid_policies(change: dict) -> None:
    data = json.loads((DATA / "synthetic-policy.json").read_text())
    with pytest.raises(ValidationError):
        RegressionPolicy.model_validate(data | change)


def test_null_metric_requires_reason_and_has_no_default_zero() -> None:
    metric = {
        "value": None,
        "unit": "ms",
        "source": "client",
        "sample_count": 0,
        "population": "successful_requests",
        "reason": "Unavailable in synthetic fixture",
    }
    assert MetricValue.model_validate(metric).value is None
    for change in ({"reason": None}, {"value": float("inf")}, {"value": 0.0, "reason": None}):
        with pytest.raises(ValidationError):
            MetricValue.model_validate(metric | change)
    del metric["value"]
    with pytest.raises(ValidationError):
        MetricValue.model_validate(metric)


def test_synthetic_result_roundtrip_and_all_metrics_explicit() -> None:
    result = ExperimentResult.model_validate_json((DATA / "synthetic-result.json").read_text())
    assert result.synthetic and not result.environment.gpu_used_uuids
    assert ExperimentResult.model_validate_json(result.model_dump_json()) == result
    assert all(metric.value is None and metric.reason for metric in result.metrics.values())
    assert (
        result.workload.dataset_sha256
        == hashlib.sha256((DATA / "synthetic-dataset.jsonl").read_bytes()).hexdigest()
    )
    assert (
        result.workload.plan_sha256
        == hashlib.sha256((DATA / "synthetic-plan.json").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "change",
    [
        {"result_schema_version": "2.0"},
        {"metric_contract_version": "2.0"},
        {"repetition": 1},
        {"point": 1},
        {"concurrency": 2},
        {"counters": {"attempts": 1, "successes": 1, "errors": 1}},
        {"metrics": {}},
        {"artifacts": {"requests": "../secret", "gpu": "raw/gpu.jsonl"}},
        {"artifacts": {"requests": "./", "gpu": "raw/gpu.jsonl"}},
        {"authorization": "secret"},
    ],
)
def test_invalid_results(change: dict) -> None:
    data = json.loads((DATA / "synthetic-result.json").read_text())
    with pytest.raises(ValidationError):
        ExperimentResult.model_validate(data | change)


def test_publication_revisions_gpu_usage_and_evidence() -> None:
    data = json.loads((DATA / "synthetic-result.json").read_text())
    data["published"] = True
    ExperimentResult.model_validate(data)
    data["config"]["model"]["revision"] = "main"
    with pytest.raises(ValidationError, match="immutable"):
        ExperimentResult.model_validate(data)
    data = json.loads((DATA / "synthetic-result.json").read_text())
    data["synthetic"] = False
    data["published"] = True
    data["runtime_manifest"]["source"] = "startup_log"
    with pytest.raises(ValidationError, match="GPU identities"):
        ExperimentResult.model_validate(data)
    data["environment"]["gpu_used_uuids"] = ["unobserved-gpu"]
    with pytest.raises(ValidationError, match="visible"):
        ExperimentResult.model_validate(data)


def test_request_timestamps_counts_and_missing_reasons() -> None:
    data = json.loads((DATA / "synthetic-request.json").read_text())
    request = RequestMeasurement.model_validate(data)
    assert RequestMeasurement.model_validate_json(request.model_dump_json()) == request
    for change in (
        {"first_content_ns": 0},
        {"content_events": 0},
        {"terminal_ns": None},
        {"output_tokens": None},
        {"finish_reason": "error"},
        {"error": "unexpected error"},
    ):
        with pytest.raises(ValidationError):
            RequestMeasurement.model_validate(data | change)
    data.update(
        transport_status="timeout",
        workload_status="not_checked",
        error="stream_idle_timeout",
        first_content_ns=None,
        last_content_ns=None,
        content_events=0,
        input_tokens=None,
        output_tokens=None,
        token_count_source=None,
        finish_reason=None,
    )
    data["missing_reasons"] = {
        field: "No server response"
        for field in (
            "first_content_ns",
            "last_content_ns",
            "input_tokens",
            "output_tokens",
            "token_count_source",
            "finish_reason",
        )
    }
    RequestMeasurement.model_validate(data)


def test_nonconformant_lengths_are_preserved_with_a_reason() -> None:
    data = json.loads((DATA / "synthetic-result.json").read_text())
    data["workload"]["output_tokens_max"] = 129
    with pytest.raises(ValidationError, match="output"):
        ExperimentResult.model_validate(data)
    data["workload"]["conformant"] = False
    data["limitations"] = ["Synthetic upstream exceeded the requested output length"]
    result = ExperimentResult.model_validate(data)
    assert result.workload.output_tokens_max == 129 and not result.workload.conformant
    data["limitations"] = []
    with pytest.raises(ValidationError, match="limitation"):
        ExperimentResult.model_validate(data)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("environment", "git_sha"),
        ("runtime_manifest", "attention_backend"),
    ],
)
def test_observation_reasons_cannot_be_missing_or_stale(section: str, field: str) -> None:
    data = json.loads((DATA / "synthetic-result.json").read_text())
    del data[section]["missing_reasons"][field]
    with pytest.raises(ValidationError, match="reasons"):
        ExperimentResult.model_validate(data)
    data = json.loads((DATA / "synthetic-result.json").read_text())
    data[section]["missing_reasons"]["unexpected"] = "Not an unverified field"
    with pytest.raises(ValidationError, match="reasons"):
        ExperimentResult.model_validate(data)


def test_gpu_metadata_checks_used_devices_not_present_device_count() -> None:
    # Metadata fixtures only; no CUDA imports or real GPU claims.
    data = json.loads((DATA / "synthetic-result.json").read_text())
    data["synthetic"] = False
    data["runtime_manifest"]["source"] = "startup_log"
    data["environment"]["gpu_present"] = [
        {
            "uuid": f"fixture-gpu-{i}",
            "index": i,
            "name": "Tesla T4",
            "vram_bytes": 16_000_000_000,
            "compute_capability": [7, 5],
        }
        for i in range(2)
    ]
    data["environment"]["gpu_visible_uuids"] = ["fixture-gpu-0", "fixture-gpu-1"]
    data["environment"]["gpu_used_uuids"] = ["fixture-gpu-0"]
    ExperimentResult.model_validate(data)
    data["config"]["runtime"]["tensor_parallel_size"] = 2
    with pytest.raises(ValidationError, match="actually used"):
        ExperimentResult.model_validate(data)
    data["environment"]["gpu_used_uuids"].append("fixture-gpu-1")
    data["runtime_manifest"]["resolved"]["tensor_parallel_size"] = 2
    ExperimentResult.model_validate(data)
    data["config"]["runtime"]["dtype"] = "bfloat16"
    data["runtime_manifest"]["resolved"]["dtype"] = "bfloat16"
    with pytest.raises(ValidationError, match="T4"):
        ExperimentResult.model_validate(data)


def test_result_metrics_keep_units_sources_population_and_finite_metadata() -> None:
    original = json.loads((DATA / "synthetic-result.json").read_text())
    for change in (
        {"unit": "ratio"},
        {"source": "runtime"},
        {"population": "all_attempts"},
        {"sample_count": 2},
    ):
        data = deepcopy(original)
        data["metrics"]["client_ttft_p95_ms"].update(change)
        with pytest.raises(ValidationError):
            ExperimentResult.model_validate(data)
    data = deepcopy(original)
    data["workload"]["template_parameters"] = {"invalid": float("inf")}
    with pytest.raises(ValidationError):
        ExperimentResult.model_validate(data)


def test_result_policy_is_bound_to_file_and_workload() -> None:
    data = json.loads((DATA / "synthetic-result.json").read_text())
    data["policy"] = json.loads((DATA / "synthetic-policy.json").read_text())
    data["config"]["measurement"]["repetitions"] = 3
    with pytest.raises(ValidationError, match="policy_file"):
        ExperimentResult.model_validate(data)
    data["config"]["policy_file"] = "tests/data/synthetic-policy.json"
    ExperimentResult.model_validate(data)
    data["policy"]["workload_id"] = "another-workload"
    with pytest.raises(ValidationError, match="workload_id"):
        ExperimentResult.model_validate(data)


def test_native_json_schemas_expose_contract_versions() -> None:
    for schema, field in (
        (ExperimentConfig, "schema_version"),
        (ExperimentResult, "result_schema_version"),
        (RegressionPolicy, "schema_version"),
        (ComparisonResult, "metric_contract_version"),
    ):
        document = schema.model_json_schema()
        assert document["properties"][field]["const"] == "1.0"
        assert document["additionalProperties"] is False


@pytest.mark.parametrize(
    ("verdict", "code"), [("PASS", 0), ("FAIL", 1), ("INCONCLUSIVE", 2), ("INVALID", 2)]
)
def test_verdicts_and_dimensions(verdict: str, code: int) -> None:
    data = {
        "schema_version": "1.0",
        "metric_contract_version": "1.0",
        "baseline_run_ids": ["baseline-1"],
        "candidate_run_ids": ["candidate-1"],
        "comparison": {"kind": "regression", "allowed_changes": ["runtime.version"]},
        "verdict": verdict,
        "reasons": [{"code": "synthetic", "message": "Synthetic schema check"}],
        "differences": [
            {"path": "runtime.version", "baseline": "synthetic-a", "candidate": "synthetic-b"}
        ],
    }
    result = ComparisonResult.model_validate(data)
    assert result.exit_code == code
    assert ComparisonResult.model_validate_json(result.model_dump_json()) == result
    if verdict != "PASS":
        with pytest.raises(ValidationError, match="structured reasons"):
            ComparisonResult.model_validate(data | {"reasons": []})
    data["differences"][0]["path"] = "runtime.dtype"
    if verdict == "INVALID":
        ComparisonResult.model_validate(data)
    else:
        with pytest.raises(ValidationError, match="Unauthorized"):
            ComparisonResult.model_validate(data)


def test_run_preserves_campaign_point_and_repetition_identity() -> None:
    config = load_experiment(EXAMPLE)
    run = ExperimentRun(
        campaign_id="campaign-1",
        run_id="run-1",
        point=0,
        repetition=0,
        concurrency=1,
        config=config,
    )
    assert ExperimentRun.model_validate_json(run.model_dump_json()) == run


def test_settings_parse_environment_and_exclude_database_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("VLLM_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:private-password@localhost/lab")
    settings = Settings(_env_file=None)
    assert settings.app_env == "test" and settings.port == 8080
    assert settings.vllm_timeout_seconds == 120
    assert not settings.store_prompt_content
    assert "private-password" not in settings.model_dump_json() + repr(settings)
    assert "database_url" not in settings.model_dump()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("PORT", "0"),
        ("VLLM_TIMEOUT_SECONDS", "nan"),
        ("VLLM_STREAM_IDLE_TIMEOUT_SECONDS", "0"),
        ("DATABASE_URL", "sqlite:///secret"),
        ("DATABASE_URL", "postgresql://user:secret@"),
        ("VLLM_BASE_URL", "http://user:secret@localhost"),
    ],
)
def test_settings_reject_invalid_environment(
    monkeypatch: pytest.MonkeyPatch, key: str, value: str
) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
