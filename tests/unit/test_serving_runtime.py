import json
import os
import shlex
from pathlib import Path
from unittest.mock import patch

import pytest

from inference_lab.config.io import load_experiment
from inference_lab.config.schemas import ExperimentConfig
from inference_lab.serving.__main__ import _write
from inference_lab.serving.runtime import (
    AWQ_REVISION,
    MODEL,
    VERSIONS,
    Observation,
    _belongs_to,
    gpu_observations,
    installed_versions,
    launch_plan,
    log_observations,
    process_identity,
    runtime_manifest,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "experiments/qwen3b-fp16.yaml"
QUALIFICATION = ROOT / "qualification/kaggle-t4-20261025-c"


def example() -> dict:
    return load_experiment(EXAMPLE).model_dump(mode="json")


def gpu(used: int = 1) -> dict[str, Observation]:
    return {
        "gpu.present": Observation(
            value=[{"index": i, "uuid": f"GPU-{i}", "name": "Tesla T4"} for i in range(2)],
            source="synthetic CPU fixture",
        ),
        "gpu.used": Observation(
            value=[{"pid": i + 10, "uuid": f"GPU-{i}"} for i in range(used)],
            source="synthetic CPU fixture",
        ),
    }


def test_launch_matches_qualified_flags_without_importing_gpu() -> None:
    config = load_experiment(EXAMPLE)
    plan = launch_plan(config, python="/fake gpu/bin/python")
    recorded = shlex.split((QUALIFICATION / "server-command.txt").read_text())
    # Qualified command shape, with only explicit interpreter and served alias changes.
    expected = recorded[3:]
    expected[0] = plan.argv[0]
    expected[expected.index("--served-model-name") + 1] = MODEL
    assert plan.argv == expected
    assert shlex.split(plan.command)[0] == "env"
    assert plan.environment == {
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_USE_V1": "0",
        "VLLM_ATTENTION_BACKEND": "XFORMERS",
    }
    assert all("secret" not in word.lower() for word in shlex.split(plan.command))
    for package in ("torch", "vllm", "triton", "transformers"):
        assert package not in __import__("sys").modules


def test_tp2_and_awq_are_explicit_changes() -> None:
    data = example()
    data["runtime"]["tensor_parallel_size"] = 2
    dual = launch_plan(ExperimentConfig.model_validate(data))
    assert dual.environment["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert dual.argv[-2:] == ["--distributed-executor-backend", "mp"]
    data = example()
    data["runtime"]["quantization"] = "awq"
    data["model"].update(
        repo_id=MODEL + "-AWQ",
        tokenizer_repo_id=MODEL + "-AWQ",
        revision=AWQ_REVISION,
        tokenizer_revision=AWQ_REVISION,
    )
    awq = launch_plan(ExperimentConfig.model_validate(data))
    assert awq.argv[-2:] == ["--quantization", "awq"]
    assert awq.environment["CUDA_VISIBLE_DEVICES"] == "0"


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "revision", "main"),
        ("model", "repo_id", "other/model"),
        ("runtime", "dtype", "bfloat16"),
        ("runtime", "tensor_parallel_size", 3),
        ("runtime", "enable_chunked_prefill", True),
        ("runtime", "enable_prefix_caching", True),
        ("runtime", "enforce_eager", False),
        ("runtime", "kv_cache_dtype", "fp8"),
        ("generation", "stream", False),
        ("target", "kind", "gateway"),
        ("target", "url", "http://example.com/v1/chat/completions"),
    ],
)
def test_unsupported_launch_assumptions_are_rejected(
    section: str, field: str, value: object
) -> None:
    data = example()
    data[section][field] = value
    with pytest.raises(ValueError):
        launch_plan(ExperimentConfig.model_validate(data))


def test_preserved_startup_logs_are_evidence_not_new_gpu_execution() -> None:
    log = (QUALIFICATION / "server-log.txt").read_text()
    observed = log_observations(log)
    assert observed["runtime.engine"].value == "V0"
    assert observed["runtime.dtype"].value == "float16"
    assert observed["runtime.enable_chunked_prefill"].value is False
    assert observed["runtime.max_num_seqs"].value == 32
    assert observed["runtime.gpu_memory_utilization"].value == 0.85
    assert observed["runtime.quantization"].value == "none"
    assert observed["runtime.kv_cache_dtype"].value == "auto"
    assert observed["runtime.kv_cache_dtype_resolved"].value is None
    assert observed["runtime.kv_cache_dtype_resolved"].reason
    assert observed["runtime.linear_backend"].value is None
    assert all(
        item.source.startswith("server.log:") for item in observed.values() if item.reason is None
    )
    log = (QUALIFICATION / "awq-recovery-20261025T204256230401Z/awq/server-log.txt").read_text()
    assert log_observations(log)["runtime.quantization"].value == "awq"
    assert log_observations(log)["model.revision"].value == AWQ_REVISION


def test_empty_and_conflicting_logs_do_not_infer_expected_values() -> None:
    observed = log_observations("")
    assert all(item.value is None and item.reason for item in observed.values())
    log = "vLLM API server version 0.10.2\nvLLM API server version 0.11.0\n"
    assert log_observations(log)["runtime.version"].value is None
    assert "Conflicting" in log_observations(log)["runtime.version"].reason


def test_manifest_detects_actual_changes_and_retains_unknowns() -> None:
    plan = launch_plan(load_experiment(EXAMPLE))
    log = (QUALIFICATION / "server-log.txt").read_text()
    manifest = runtime_manifest(
        plan, log=log, gpu=gpu(), smoke={"status": "VALIDATED"}, versions=VERSIONS
    )
    assert manifest.status == "VALIDATED" and not manifest.mismatches
    assert manifest.observed["runtime.kv_cache_dtype_resolved"].value is None
    changed = log.replace("dtype=torch.float16", "dtype=torch.bfloat16").replace(
        "Using XFormers backend.", "Using FlashAttention backend."
    )
    manifest = runtime_manifest(
        plan,
        log=changed,
        gpu=gpu(2),
        smoke={"status": "VALIDATED"},
        versions=VERSIONS | {"vllm": "0.11.0"},
    )
    assert manifest.status == "FAILED"
    assert set(manifest.mismatches) == {
        "runtime.dtype",
        "runtime.attention_backend",
        "gpu.used_count",
        "gpu.used_placement",
        "packages.vllm",
    }
    manifest = runtime_manifest(
        plan, log="", gpu=gpu(), smoke={"status": "FAILED", "error": "idle_timeout"}, versions={}
    )
    assert manifest.status == "FAILED" and manifest.observed["packages.vllm"].reason
    hardware = gpu()
    hardware["gpu.present"].value[0]["name"] = "Other GPU"
    assert (
        "gpu.used_hardware"
        in runtime_manifest(
            plan, log=log, gpu=hardware, smoke={"status": "VALIDATED"}, versions=VERSIONS
        ).mismatches
    )


def test_pid_start_ticks_prevent_reuse_or_foreign_process_attribution() -> None:
    identity = process_identity(os.getpid())
    assert identity is not None
    assert _belongs_to(os.getpid(), os.getpid(), identity[1])
    assert not _belongs_to(os.getpid(), os.getpid(), "wrong-start")
    with patch(
        "inference_lab.serving.runtime.process_identity",
        side_effect=lambda pid: {20: (10, "worker"), 10: (1, "server")}.get(pid),
    ):
        assert _belongs_to(20, 10, "server")
        assert not _belongs_to(20, 10, "reused")
        assert not _belongs_to(30, 10, "server")


def test_nvidia_absence_is_unknown_not_empty_or_zero() -> None:
    with patch("inference_lab.serving.runtime.subprocess.run", side_effect=FileNotFoundError):
        values = gpu_observations(123, "start")
    assert all(item.value is None and item.reason for item in values.values())


def test_artifacts_are_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "runtime_manifest.json"
    _write(path, {"original": True})
    with pytest.raises(FileExistsError):
        _write(path, {"replacement": True})
    assert json.loads(path.read_text()) == {"original": True}


def test_inventory_uses_selected_python_and_includes_cuda_runtime() -> None:
    from subprocess import CompletedProcess

    with patch(
        "inference_lab.serving.runtime.subprocess.run",
        return_value=CompletedProcess([], 0, json.dumps(VERSIONS), ""),
    ) as run:
        assert installed_versions("/gpu/python") == VERSIONS
    command = run.call_args.args[0]
    assert command[:2] == ["/gpu/python", "-c"] and "torch.version.cuda" in command[2]


def test_gpu_query_excludes_foreign_processes_and_retains_uuid() -> None:
    from subprocess import CompletedProcess

    with (
        patch(
            "inference_lab.serving.runtime.subprocess.run",
            side_effect=[
                CompletedProcess(
                    [], 0, "0, GPU-0, Tesla T4, 15360\n1, GPU-1, Tesla T4, 15360\n", ""
                ),
                CompletedProcess([], 0, "20, GPU-0\n30, GPU-1\n", ""),
            ],
        ),
        patch(
            "inference_lab.serving.runtime.process_identity",
            side_effect=lambda pid: {20: (10, "child"), 10: (1, "server"), 30: (1, "foreign")}.get(
                pid
            ),
        ),
    ):
        observed = gpu_observations(10, "server")
    assert observed["gpu.used"].value == [{"pid": 20, "uuid": "GPU-0"}]
    assert len(observed["gpu.present"].value) == 2


def test_serve_preserves_version_failure_and_refuses_existing_directory(tmp_path: Path) -> None:
    from argparse import Namespace

    from inference_lab.serving.__main__ import _serve

    args = Namespace(experiment=EXAMPLE, python="/gpu/python", output=tmp_path / "attempt")
    with patch(
        "inference_lab.serving.__main__.installed_versions",
        return_value=VERSIONS | {"vllm": "wrong"},
    ):
        with pytest.raises(ValueError, match="versions differ"):
            _serve(args)
    assert (args.output / "launch.json").is_file()
    assert json.loads((args.output / "versions.json").read_text())["vllm"] == "wrong"
    with pytest.raises(FileExistsError):
        _serve(args)


def test_cli_validation_does_not_echo_url_credentials(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    import yaml

    from inference_lab.serving.__main__ import main

    config = example()
    config["target"]["url"] = "http://username:VERY_SECRET@127.0.0.1:8001/v1/chat/completions"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["serving", "command", str(path)])
    assert main() == 1
    assert "SECRET" not in capsys.readouterr().err


async def test_probe_reports_preserved_preflight_failure_without_waiting(tmp_path: Path) -> None:
    from argparse import Namespace

    from inference_lab.serving.__main__ import _probe

    _write(tmp_path / "launch-failure.json", {"status": "FAILED", "error_type": "ValueError"})
    with pytest.raises(ValueError, match="preflight failed"):
        await _probe(Namespace(output=tmp_path, startup_timeout=1))


async def test_healthy_other_port_cannot_pass_before_own_server_startup(tmp_path: Path) -> None:
    from argparse import Namespace
    from unittest.mock import AsyncMock

    from inference_lab.serving.__main__ import _probe

    plan = launch_plan(load_experiment(EXAMPLE))
    _write(tmp_path / "launch.json", plan.model_dump(mode="json"))
    _write(tmp_path / "versions.json", VERSIONS)
    identity = process_identity(os.getpid())
    _write(tmp_path / "server-process.json", {"pid": os.getpid(), "start_ticks": identity[1]})
    (tmp_path / "server.log").write_text(
        "Synthetic fixture: still loading; no startup completion.\n"
    )
    with (
        patch("inference_lab.serving.__main__.readiness", new_callable=AsyncMock) as probe,
        patch("inference_lab.serving.__main__.gpu_observations", return_value=gpu()),
    ):
        assert await _probe(Namespace(output=tmp_path, startup_timeout=0.03)) == 1
        probe.assert_not_called()
    manifest = json.loads((tmp_path / "runtime_manifest.json").read_text())
    assert manifest["readiness"] == {"status": "FAILED", "error": "startup_timeout"}


def test_serve_refuses_occupied_port_before_any_gpu_launch(tmp_path: Path) -> None:
    import socket
    from argparse import Namespace

    import yaml

    from inference_lab.serving.__main__ import _serve

    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        data = example()
        data["target"]["url"] = f"http://127.0.0.1:{occupied.getsockname()[1]}/v1/chat/completions"
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(data))
        args = Namespace(experiment=path, python="/gpu/python", output=tmp_path / "attempt")
        with (
            patch("inference_lab.serving.__main__.installed_versions", return_value=VERSIONS),
            patch("inference_lab.serving.__main__.gpu_observations") as gpu_query,
        ):
            with pytest.raises(OSError):
                _serve(args)
            gpu_query.assert_not_called()
        assert (args.output / "launch-failure.json").is_file()
