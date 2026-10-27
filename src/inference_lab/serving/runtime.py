"""Qualified launch plans and conservative, version-specific startup-log evidence."""

import ast
import csv
import io
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, JsonValue, model_validator

from inference_lab.config.schemas import ContractModel, ExperimentConfig, Text

QUALIFICATION_ID: Literal["kaggle-t4-20261025-c"] = "kaggle-t4-20261025-c"
VERSIONS = {
    "python": "3.12.12",
    "torch": "2.8.0",
    "vllm": "0.10.2",
    "triton": "3.4.0",
    "transformers": "4.55.2",
    "xformers": "0.0.32.post1",
    "pytorch_cuda_runtime": "12.8",
}
MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
AWQ_REVISION = "3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd"


class LaunchPlan(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    qualification_id: Literal["kaggle-t4-20261025-c"] = QUALIFICATION_ID
    experiment: ExperimentConfig
    argv: list[Text]
    environment: dict[Text, Text]
    expected_versions: dict[Text, Text]

    @property
    def command(self) -> str:
        # Only allowlisted launch arguments/environment are constructed here.
        return shlex.join(["env", *(f"{k}={v}" for k, v in self.environment.items()), *self.argv])


def launch_plan(config: ExperimentConfig, *, python: str = sys.executable) -> LaunchPlan:
    """No downloads or process creation. Unsupported assumptions fail explicitly."""
    config.model.require_immutable()
    model, runtime = config.model, config.runtime
    if not config.generation.stream:
        raise ValueError("Direct serving readiness requires streaming generation")
    revision = AWQ_REVISION if runtime.quantization == "awq" else REVISION
    repo = MODEL + ("-AWQ" if runtime.quantization == "awq" else "")
    if (model.repo_id, model.revision, model.tokenizer_repo_id, model.tokenizer_revision) != (
        repo,
        revision,
        repo,
        revision,
    ):
        raise ValueError("This launch profile requires the qualified model/tokenizer revisions")
    if runtime.dtype != "float16" or runtime.tensor_parallel_size not in (1, 2):
        raise ValueError("Qualified serving requires FP16 and TP=1 or TP=2")
    if runtime.quantization == "awq" and runtime.tensor_parallel_size != 1:
        raise ValueError("AWQ was qualified only with TP=1")
    if runtime.enable_chunked_prefill or runtime.enable_prefix_caching or not runtime.enforce_eager:
        raise ValueError(
            "Qualified profile requires eager mode and disables both caches/prefill flags"
        )
    if runtime.kv_cache_dtype != "auto":
        raise ValueError(
            "Only the qualified KV cache setting auto is supported; resolved dtype is unverified"
        )
    target = config.target.url
    if (
        config.target.kind != "direct"
        or target.scheme != "http"
        or target.host != "127.0.0.1"
        or target.path != "/v1/chat/completions"
    ):
        raise ValueError("Direct serving requires http://127.0.0.1:<port>/v1/chat/completions")
    argv = [
        python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(target.port or 80),
        "--model",
        model.repo_id,
        "--revision",
        model.revision,
        "--tokenizer",
        model.tokenizer_repo_id,
        "--tokenizer-revision",
        model.tokenizer_revision,
        "--served-model-name",
        model.repo_id,
    ]
    for flag in (
        "dtype",
        "tensor_parallel_size",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
        "kv_cache_dtype",
    ):
        argv.extend(["--" + flag.replace("_", "-"), str(getattr(runtime, flag))])
    argv.extend(
        [
            "--no-enable-prefix-caching",
            "--no-enable-chunked-prefill",
            "--enforce-eager",
            "--generation-config",
            "vllm",
            "--seed",
            str(config.workload.seed),
            "--no-enable-log-requests",
            "--no-enable-log-outputs",
        ]
    )
    if runtime.quantization is not None:
        argv.extend(["--quantization", runtime.quantization])
    if runtime.tensor_parallel_size == 2:
        argv.extend(["--distributed-executor-backend", "mp"])
    return LaunchPlan(
        experiment=config,
        argv=argv,
        environment={
            "CUDA_VISIBLE_DEVICES": "0,1" if runtime.tensor_parallel_size == 2 else "0",
            "VLLM_USE_V1": "0",
            "VLLM_ATTENTION_BACKEND": "XFORMERS",
        },
        expected_versions=VERSIONS,
    )


def installed_versions(python: str) -> dict[str, str]:
    """Inspect the selected interpreter, not the CPU control-plane environment."""
    code = (
        "import importlib.metadata as m,json,platform,torch; "
        "print(json.dumps({'python':platform.python_version(), "
        "'pytorch_cuda_runtime':torch.version.cuda, "
        "**{k:m.version(k) "
        "for k in ('torch','vllm','triton','transformers','xformers')}}))"
    )
    result = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError("Cannot inspect GPU package versions with the selected Python")
    return dict(json.loads(result.stdout))


class Observation(ContractModel):
    value: JsonValue
    source: Text
    reason: Text | None = None

    @model_validator(mode="after")
    def missing_reason(self) -> "Observation":
        if (self.value is None) != (self.reason is not None):
            raise ValueError(
                "Unknown observations require a reason; known values must not have one"
            )
        return self


def _unknown(reason: str) -> Observation:
    return Observation(value=None, source="unavailable", reason=reason)


def log_observations(log: str) -> dict[str, Observation]:
    """Parse only the qualified 0.10.2 format; never certify from expected config."""
    names = (
        "model.repo_id",
        "model.revision",
        "model.tokenizer_repo_id",
        "model.tokenizer_revision",
        "runtime.dtype",
        "runtime.quantization",
        "runtime.tensor_parallel_size",
        "runtime.max_model_len",
        "runtime.max_num_seqs",
        "runtime.max_num_batched_tokens",
        "runtime.gpu_memory_utilization",
        "runtime.kv_cache_dtype",
        "runtime.enable_prefix_caching",
        "runtime.enable_chunked_prefill",
        "runtime.enforce_eager",
        "runtime.version",
        "runtime.engine",
        "runtime.attention_backend",
        "runtime.linear_backend",
        "runtime.kv_cache_dtype_resolved",
    )
    values = {name: _unknown("Not exposed by the qualified startup log parser") for name in names}
    aliases = {
        "model": "model.repo_id",
        "revision": "model.revision",
        "tokenizer": "model.tokenizer_repo_id",
        "tokenizer_revision": "model.tokenizer_revision",
        "dtype": "runtime.dtype",
        "tensor_parallel_size": "runtime.tensor_parallel_size",
        "max_seq_len": "runtime.max_model_len",
        "kv_cache_dtype": "runtime.kv_cache_dtype",
        "enable_prefix_caching": "runtime.enable_prefix_caching",
        "chunked_prefill_enabled": "runtime.enable_chunked_prefill",
        "enforce_eager": "runtime.enforce_eager",
    }
    conflicts: set[str] = set()

    def record(key: str, value: JsonValue, line: int) -> None:
        prior = values[key]
        if prior.source != "unavailable" and prior.value != value:
            conflicts.add(key)
        values[key] = Observation(value=value, source=f"server.log:{line}")

    for number, raw in enumerate(log.splitlines(), 1):
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
        if match := re.search(r"vLLM API server version (\S+)", line):
            record("runtime.version", match[1], number)
        if "non-default args: " in line:
            # Literal data from a local startup log, never eval executable code.
            try:
                args = ast.literal_eval(line.split("non-default args: ", 1)[1])
            except (ValueError, SyntaxError, RecursionError):
                args = {}
            if isinstance(args, dict):
                for key in ("max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization"):
                    if type(args.get(key)) in (int, float):
                        record("runtime." + key, args[key], number)
        if match := re.search(r"Initializing a (V\d+) LLM engine \(v([^)]*)\) with config:", line):
            record("runtime.engine", match[1], number)
            record("runtime.version", match[2], number)
            for key, canonical in aliases.items():
                if token := re.search(r"\b" + key + r"=('([^']*)'|[^, ]+)", line):
                    value = token[2] if token[2] is not None else token[1]
                    if value in ("True", "False"):
                        parsed: JsonValue = value == "True"
                    elif value.isdecimal():
                        parsed = int(value)
                    else:
                        parsed = value.removeprefix("torch.")
                    record(canonical, parsed, number)
            if token := re.search(r"\bquantization=([^, ]+)", line):
                # Null is a known no-quantization setting, not an absent observation.
                record("runtime.quantization", "none" if token[1] == "None" else token[1], number)
        if match := re.search(r"Using (\S+) backend\.", line):
            record("runtime.attention_backend", match[1].upper(), number)
    for key in conflicts:
        values[key] = _unknown("Conflicting startup observations; use one fresh log per launch")
    return values


def process_identity(pid: int) -> tuple[int, str] | None:
    """Linux parent PID and start ticks; start ticks prevent stale-PID attribution."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return int(fields[1]), fields[19]
    except (OSError, ValueError, IndexError):
        return None


def _belongs_to(pid: int, server_pid: int, start_ticks: str) -> bool:
    seen: set[int] = set()
    while pid > 0 and pid not in seen:
        seen.add(pid)
        identity = process_identity(pid)
        if identity is None:
            return False
        if pid == server_pid:
            return identity[1] == start_ticks
        pid = identity[0]
    return False


def gpu_observations(server_pid: int, start_ticks: str) -> dict[str, Observation]:
    values = {
        "gpu.present": _unknown("nvidia-smi GPU inventory unavailable"),
        "gpu.used": _unknown("Server-owned compute processes could not be verified"),
    }
    try:
        inventory = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        rows: list[JsonValue] = []
        for row in csv.reader(io.StringIO(inventory.stdout), skipinitialspace=True):
            if len(row) != 4:
                raise ValueError("Unexpected GPU inventory format")
            rows.append(
                {"index": int(row[0]), "uuid": row[1], "name": row[2], "memory_mib": int(row[3])}
            )
        values["gpu.present"] = Observation(value=rows, source="nvidia-smi GPU query")
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        used: list[JsonValue] = []
        for row in csv.reader(io.StringIO(processes.stdout), skipinitialspace=True):
            if len(row) != 2:
                raise ValueError("Unexpected compute process format")
            if _belongs_to(int(row[0]), server_pid, start_ticks):
                used.append({"pid": int(row[0]), "uuid": row[1]})
        values["gpu.used"] = (
            Observation(value=used, source="nvidia-smi compute query + /proc ancestry/start ticks")
            if used
            else _unknown("No live server-owned GPU compute process observed")
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return values


class RuntimeManifest(ContractModel):
    manifest_schema_version: Literal["1.0"] = "1.0"
    qualification_id: Literal["kaggle-t4-20261025-c"] = QUALIFICATION_ID
    expected: LaunchPlan
    command: Text
    observed: dict[Text, Observation]
    mismatches: list[Text]
    readiness: dict[str, JsonValue]
    status: Literal["VALIDATED", "FAILED", "UNAVAILABLE", "NOT_TESTED"]
    limitations: list[Text] = Field(default_factory=list)


def runtime_manifest(
    plan: LaunchPlan,
    *,
    log: str,
    gpu: dict[str, Observation],
    smoke: dict[str, Any],
    versions: dict[str, str],
) -> RuntimeManifest:
    observed = log_observations(log) | gpu
    expected = {
        **{
            "model." + k: v
            for k, v in plan.experiment.model.model_dump().items()
            if k != "trust_remote_code"
        },
        **{
            "runtime." + k: v
            for k, v in plan.experiment.runtime.model_dump().items()
            if k != "backend"
        },
        "runtime.version": "0.10.2",
        "runtime.engine": "V0",
        "runtime.attention_backend": "XFORMERS",
    }
    expected["runtime.quantization"] = plan.experiment.runtime.quantization or "none"
    mismatches = [
        key
        for key, value in expected.items()
        if observed[key].source != "unavailable" and observed[key].value != value
    ]
    for name, value in plan.expected_versions.items():
        actual = versions.get(name)
        observed["packages." + name] = (
            Observation(value=actual, source="selected GPU Python metadata/platform/torch.version")
            if actual is not None
            else _unknown("Package inventory unavailable")
        )
        if actual is not None and actual != value:
            mismatches.append("packages." + name)
    used = gpu["gpu.used"].value
    if isinstance(used, list):
        uuids = {row["uuid"] for row in used if isinstance(row, dict)}
        if len(uuids) != plan.experiment.runtime.tensor_parallel_size:
            mismatches.append("gpu.used_count")
        inventory = gpu["gpu.present"].value
        if isinstance(inventory, list):
            selected_indices = {
                int(index) for index in plan.environment["CUDA_VISIBLE_DEVICES"].split(",")
            }
            selected_uuids = {
                row["uuid"]
                for row in inventory
                if isinstance(row, dict) and row["index"] in selected_indices
            }
            if uuids != selected_uuids:
                mismatches.append("gpu.used_placement")
            used_devices = [
                row for row in inventory if isinstance(row, dict) and row["uuid"] in uuids
            ]
            if len(used_devices) != len(uuids) or any(
                row["name"] != "Tesla T4" for row in used_devices
            ):
                mismatches.append("gpu.used_hardware")
    # Unknown effective values do not become known merely because generation worked.
    status: Literal["VALIDATED", "FAILED", "UNAVAILABLE", "NOT_TESTED"]
    if mismatches or smoke.get("status") == "FAILED":
        status = "FAILED"
    elif smoke.get("status") == "VALIDATED":
        required = set(expected) | {"packages." + name for name in VERSIONS} | set(gpu)
        status = "UNAVAILABLE" if any(observed[name].reason for name in required) else "VALIDATED"
    else:
        status = "NOT_TESTED"
    return RuntimeManifest(
        expected=plan,
        command=plan.command,
        observed=observed,
        mismatches=mismatches,
        readiness=smoke,
        status=status,
        limitations=[
            "Status combines smoke, expected-field evidence and GPU ownership; it excludes "
            "unresolved KV dtype/linear kernels and performance.",
            "Unknown observations require review before publication; no inferred KV dtype "
            "or linear kernel.",
            "GPU identities are a server-process observation, not per-request tracing.",
            "Client cancellation closes the connection; engine drain needs a GPU check.",
        ],
    )
