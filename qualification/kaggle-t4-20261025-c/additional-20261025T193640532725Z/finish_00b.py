
"""Finish prompt 00B in the existing Kaggle session; never install packages."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import version
from urllib.error import URLError
from urllib.request import urlopen
import zipfile

REPO = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
CAPABILITIES = (
    "single_gpu_fp16_vllm", "streaming",
    "torch_compile", "triton", "tp2", "awq",
)


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def launch_command(path):
    parts = shlex.split([
        line for line in path.read_text().splitlines() if line.strip()
    ][-1])
    settings = {}
    while parts and re.match(r"^[A-Z_][A-Z_0-9]*=", parts[0]):
        key, value = parts.pop(0).split("=", 1)
        settings[key] = value
    assert settings == {
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_USE_V1": "0",
        "VLLM_ATTENTION_BACKEND": "XFORMERS",
    }, settings
    expected = {
        "--model": REPO,
        "--revision": REVISION,
        "--tokenizer-revision": REVISION,
        "--dtype": "float16",
        "--tensor-parallel-size": "1",
        "--port": "8001",
    }
    for flag, value in expected.items():
        assert parts[parts.index(flag) + 1] == value, (flag, value)
    assert "--no-enable-chunked-prefill" in parts
    assert "--no-enable-prefix-caching" in parts
    return settings, parts


def check_stream(result):
    assert result["status"] == "VALIDATED" and result["done_received"]
    assert result["finish_reason"] == "length"
    assert result["token_count_source"] == "server_usage"
    assert result["usage"]["completion_tokens"] == 128
    assert result["usage"]["prompt_tokens"] > 0

    start, first, last, end = (
        result[key] for key in (
            "request_start_ns", "first_content_ns",
            "last_content_ns", "terminal_ns",
        )
    )
    assert all(type(value) is int for value in (start, first, last, end))
    assert start <= first <= last <= end
    assert result["content_events"] >= 2

    expected = {
        "client_ttft_ms": (first - start) / 1e6,
        "client_e2e_ms": (end - start) / 1e6,
        "client_tpot_estimate_ms": (last - first) / 1e6 / 127,
    }
    for metric, value in expected.items():
        assert math.isclose(
            result["metrics"][metric]["value"], value,
            rel_tol=1e-9, abs_tol=1e-9,
        )


GPU_TEST = r"""
import sys
import torch

mode = sys.argv[1]
torch.manual_seed(42)
assert torch.cuda.get_device_capability(0) == (7, 5)

def reference(x, w):
    z = x.float()
    return (
        z * torch.rsqrt((z*z).mean(-1, keepdim=True) + 1e-6)
        * w.float()
    ).to(x.dtype)

if mode == "torch_compile":
    torch._dynamo.config.suppress_errors = False
    compiled = torch.compile(
        reference, backend="inductor", fullgraph=True
    )
    for seed in (7, 42):
        torch.manual_seed(seed)
        for shape in ((4, 128), (16, 2048)):
            x = torch.randn(
                shape, device="cuda", dtype=torch.float16
            )
            w = torch.randn(
                shape[-1], device="cuda", dtype=torch.float16
            )
            torch.testing.assert_close(
                reference(x, w),
                torch.nn.functional.rms_norm(
                    x.float(), (shape[-1],), w.float(), eps=1e-6
                ).half(),
                rtol=2e-3, atol=2e-3,
            )
            # Compilation and a subsequent execution must both work.
            for _ in range(2):
                y = compiled(x, w)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    y, reference(x, w), rtol=2e-3, atol=2e-3
                )
else:
    import triton
    import triton.language as tl

    @triton.jit
    def add(X, Y, Z, N: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        x = tl.load(X+i, i<N, 0)
        y = tl.load(Y+i, i<N, 0)
        tl.store(Z+i, x+y, i<N)

    @triton.jit
    def rms(X, W, Y, N: tl.constexpr, B: tl.constexpr):
        row = tl.program_id(0)
        i = tl.arange(0, B)
        x = tl.load(X+row*N+i, i<N, 0).to(tl.float32)
        w = tl.load(W+i, i<N, 0).to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x*x, 0)/N + 1e-6) * w
        tl.store(Y+row*N+i, y, i<N)

    if mode == "triton_minimal":
        x = torch.randn(
            1025, device="cuda", dtype=torch.float16
        )
        y = torch.randn_like(x)
        z = torch.empty_like(x)
        add[(triton.cdiv(x.numel(), 128),)](
            x, y, z, x.numel(), 128,
            num_warps=4, num_stages=1,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(z, x+y, rtol=0, atol=0)

    elif mode == "triton_rmsnorm":
        for seed in (7, 42):
            torch.manual_seed(seed)
            for shape in ((4, 128), (16, 2048)):
                x = torch.randn(
                    shape, device="cuda", dtype=torch.float16
                )
                w = torch.randn(
                    shape[-1], device="cuda", dtype=torch.float16
                )
                y = torch.empty_like(x)
                rms[(shape[0],)](
                    x, w, y, shape[-1],
                    triton.next_power_of_2(shape[-1]),
                    num_warps=4, num_stages=1,
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    y, reference(x, w), rtol=2e-3, atol=2e-3
                )
    else:
        raise ValueError(mode)

print(
    mode + ": VALIDATED; correctness only, no performance measurement",
    flush=True,
)
"""


def stop_group(pid):
    try:
        os.killpg(pid, signal.SIGTERM)
        time.sleep(3)
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run(command, path, timeout=60, env=None):
    with path.open("a") as out:
        out.write("$ " + shlex.join(map(str, command)) + "\n")
        out.flush()
        process = subprocess.Popen(
            command,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            stop_group(process.pid)
            process.wait(timeout=15)
            raise
    if code:
        raise RuntimeError(f"Exit {code}; see {path}")


def gpu_idle():
    result = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode or result.stdout.strip():
        raise RuntimeError(
            "GPU compute processes remain or cannot be inspected: "
            + result.stdout + result.stderr
        )


def stop_baseline(q):
    pid = int((q / "server-pid.txt").read_text())
    process = Path(f"/proc/{pid}/cmdline")
    if process.exists():
        args = process.read_bytes().split(b"\0")
        if args != [b""]:
            assert b"vllm.entrypoints.openai.api_server" in args, (
                "PID no longer belongs to our server"
            )
            assert REPO.encode() in args and REVISION.encode() in args
            assert os.getpgid(pid) == pid, (
                "Unexpected process group; refusing to kill"
            )
            stop_group(pid)
    gpu_idle()


def server_smoke(stage, command, settings, probe, expected):
    gpu_idle()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 8001))

    write_json(stage / "launch-settings.json", {
        "environment": settings,
        "expected": expected,
    })
    (stage / "server-command.txt").write_text(
        shlex.join(
            [f"{key}={value}" for key, value in settings.items()]
            + command
        ) + "\n"
    )
    env = dict(os.environ, **settings)

    with (stage / "server-log.txt").open("w") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 900
        update = time.monotonic() + 30

        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Server exited {process.returncode}; "
                    f"see {stage / 'server-log.txt'}"
                )
            try:
                with urlopen(
                    "http://127.0.0.1:8001/health", timeout=2
                ) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                pass

            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError("Server startup exceeded 15 minutes")
            if now >= update:
                print(stage.name + ": still starting", flush=True)
                update = now + 30
            time.sleep(2)

        # ponytail: version-specific log checks for vLLM 0.10.2.
        # Replace with structured introspection if the log format changes.
        text = (stage / "server-log.txt").read_text(errors="replace")
        for marker in (
            "Using XFormers backend",
            "dtype=torch.float16",
            "Initializing a V0 LLM engine",
            "chunked_prefill_enabled=False",
            "enable_prefix_caching=False",
            f"tensor_parallel_size={expected['tp']}",
            f"revision={expected['revision']}",
            f"quantization={expected['quantization']}",
        ):
            assert marker in text, (
                "Effective runtime value not verified: " + marker
            )

        write_json(stage / "runtime-manifest.json", {
            "effective_checked_values": expected,
            "evidence": "server-log.txt",
            "visible_devices": settings["CUDA_VISIBLE_DEVICES"],
        })
        run(
            [sys.executable, "-u", str(probe), str(stage)],
            stage / "probe.log", 240, env,
        )
        result = json.loads(
            (stage / "stream-results.jsonl").read_text().splitlines()[-1]
        )
        check_stream(result)
    finally:
        stop_group(process.pid)
        process.wait(timeout=15)


def self_check():
    result = {
        "status": "VALIDATED",
        "done_received": True,
        "finish_reason": "length",
        "token_count_source": "server_usage",
        "usage": {"completion_tokens": 128, "prompt_tokens": 41},
        "request_start_ns": 100,
        "first_content_ns": 1000100,
        "last_content_ns": 128000100,
        "terminal_ns": 129000100,
        "content_events": 2,
        "metrics": {
            "client_ttft_ms": {"value": 1},
            "client_e2e_ms": {"value": 129},
            "client_tpot_estimate_ms": {"value": 1},
        },
    }
    check_stream(result)

    for key, value in (
        ("done_received", False),
        ("content_events", 1),
        ("first_content_ns", None),
    ):
        bad = dict(result, **{key: value})
        try:
            check_stream(bad)
        except (AssertionError, TypeError):
            continue
        raise AssertionError("Invalid stream accepted: " + key)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "server-command.txt"
        command = (
            "CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 "
            "VLLM_ATTENTION_BACKEND=XFORMERS /example/python "
            f"--model {REPO} --revision {REVISION} "
            f"--tokenizer-revision {REVISION} "
            "--dtype float16 --tensor-parallel-size 1 --port 8001 "
            "--no-enable-chunked-prefill --no-enable-prefix-caching"
        )
        path.write_text(command)
        settings, parsed = launch_command(path)
        assert settings["VLLM_ATTENTION_BACKEND"] == "XFORMERS"
        assert parsed[0] == "/example/python"

        path.write_text(command.replace("--dtype float16", "--dtype bfloat16"))
        try:
            launch_command(path)
        except AssertionError:
            pass
        else:
            raise AssertionError("Dtype change was accepted")

    compile(GPU_TEST, "gpu_checks.py", "exec")
    print("Offline self-check passed.", flush=True)


def main(q):
    old = q.with_name("kaggle-t4-20261025-b")
    python = old / ".venv/bin/python"
    assert python.exists(), (
        "Existing qualification environment missing; "
        "no packages will be installed"
    )

    if Path(sys.executable).absolute() != python.absolute():
        os.execv(str(python), [
            str(python), str(Path(__file__).resolve()),
            "--directory", str(q),
        ])

    assert platform.python_version() == "3.12.12"
    for package, pin in {
        "torch": "2.8.0",
        "vllm": "0.10.2",
        "triton": "3.4.0",
        "transformers": "4.55.2",
        "xformers": "0.0.32.post1",
    }.items():
        assert version(package).split("+")[0] == pin, (
            package, version(package)
        )

    settings, command = launch_command(q / "server-command.txt")
    stream = json.loads(
        (q / "stream-results.jsonl").read_text().splitlines()[-1]
    )
    check_stream(stream)

    pin = read_json(q / "model-pin.json")
    assert pin["repo_id"] == REPO and pin["revision"] == REVISION

    log = (q / "server-log.txt").read_text(errors="replace")
    # ponytail: version-specific evidence markers, not a general log parser.
    for marker in (
        "dtype=torch.float16",
        "tensor_parallel_size=1",
        "Using XFormers backend",
        "chunked_prefill_enabled=False",
        "enable_prefix_caching=False",
        "enforce_eager=True",
        "quantization=None",
        f"revision={REVISION}",
    ):
        assert marker in log, marker

    probe = q / "stream-probe.py"
    assert probe.exists()

    batch = q / (
        "additional-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    batch.mkdir()
    shutil.copy2(__file__, batch / "finish_00b.py")

    state = {
        "qualification_id": q.name,
        **{cap: "NOT_TESTED" for cap in CAPABILITIES},
        "reasons": {cap: "Not yet executed" for cap in CAPABILITIES},
        "model": pin,
        "validated_versions": {},
        "evidence": {},
        "qualification_complete": False,
    }
    for name in ("single_gpu_fp16_vllm", "streaming"):
        state[name] = "VALIDATED"
        state["reasons"][name] = (
            "Baseline startup log and successful real streaming generation; "
            "chunked prefill disabled."
        )

    state["baseline_metrics"] = stream["metrics"]
    state["known_failure"] = {
        "qualification_id": old.name,
        "status": "FAILED",
        "scope": "chunked prefill Triton attention kernel",
        "shared_memory_required_bytes": 98304,
        "shared_memory_limit_bytes": 65536,
    }

    if (q / "qualification.json").exists():
        shutil.copy2(
            q / "qualification.json",
            batch / "previous-qualification.json",
        )

    def record(name, status, reason, evidence):
        state[name] = status
        state["reasons"][name] = reason
        state["evidence"][name] = str(evidence.relative_to(q))
        write_json(q / "qualification.json", state)
        print(f"{name}: {status} — {reason}", flush=True)

    try:
        print(
            "Stopping the baseline server before sequential checks.",
            flush=True,
        )
        stop_baseline(q)

        from importlib.metadata import distributions

        inventory_code = """
import torch, json, sys
from pathlib import Path
gpus = [{
    "index": i,
    "name": torch.cuda.get_device_name(i),
    "compute_capability": list(torch.cuda.get_device_capability(i)),
    "vram_bytes": torch.cuda.get_device_properties(i).total_memory,
} for i in range(torch.cuda.device_count())]
Path(sys.argv[1]).write_text(
    json.dumps({"gpus": gpus, "cuda": torch.version.cuda})
)
"""
        run(
            [
                sys.executable, "-c", inventory_code,
                str(batch / "cuda-inventory.json"),
            ],
            batch / "inventory.log",
        )
        inventory = read_json(batch / "cuda-inventory.json")
        gpus = inventory["gpus"]
        assert gpus and all(
            "T4" in gpu["name"]
            and gpu["compute_capability"] == [7, 5]
            for gpu in gpus
        ), gpus

        environment = {
            "qualification_id": q.name,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "os": platform.platform(),
            "cpu_count": os.cpu_count(),
            "ram": [
                line.strip()
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemTotal:")
            ],
            "gpus_visible_to_inventory": gpus,
            "inventory_cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
            "pytorch_cuda_runtime": inventory["cuda"],
            "versions": {
                package: version(package)
                for package in (
                    "torch", "vllm", "triton", "transformers", "xformers"
                )
            },
            "model": pin,
            "baseline_used_gpu_indices": [0],
            "tp2_used_gpu_indices": None,
            "gpu_uuid_driver_and_memory_evidence": str(
                (batch / "gpu-inventory.csv").relative_to(q)
            ),
            "driver_supported_cuda_evidence": "nvidia-smi.txt",
            "cuda_toolkit": None,
            "git_sha": None,
            "git_sha_reason": (
                "Repository is not mounted in the Kaggle notebook; "
                "script SHA256 is preserved instead."
            ),
            "script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }

        for args, file in (
            (["nvidia-smi"], "nvidia-smi.txt"),
            (["nvidia-smi", "topo", "-m"], "nvidia-smi-topo.txt"),
        ):
            run(args, q / file)

        run([
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv",
        ], batch / "gpu-inventory.csv")

        inventory_rows = [
            line
            for line in (batch / "gpu-inventory.csv").read_text().splitlines()
            if re.match(r"^\d+,\s*GPU-", line)
        ]
        environment["gpu_present_count"] = len(inventory_rows)
        environment["gpu_visible_count"] = len(gpus)
        environment["nvidia_driver_version"] = (
            inventory_rows[0].split(",")[-1].strip()
            if inventory_rows else None
        )
        cuda_match = re.search(
            r"CUDA Version:\s*([\d.]+)",
            (q / "nvidia-smi.txt").read_text(),
        )
        environment["driver_supported_cuda"] = (
            cuda_match.group(1) if cuda_match else None
        )

        if shutil.which("nvcc"):
            run(["nvcc", "--version"], batch / "nvcc-version.txt")
            environment["cuda_toolkit"] = str(
                (batch / "nvcc-version.txt").relative_to(q)
            )
        else:
            environment["cuda_toolkit_reason"] = (
                "nvcc unavailable; driver CUDA capability "
                "is not the installed toolkit."
            )

        write_json(q / "environment.json", environment)
        (batch / "pip-freeze.txt").write_text(
            "\n".join(sorted(
                f"{dist.metadata['Name']}=={dist.version}"
                for dist in distributions()
            )) + "\n"
        )
        state["validated_versions"] = {
            "python": platform.python_version(),
            **environment["versions"],
        }
        state["validated_versions_scope"] = (
            "Baseline serving; individual compile/Triton/TP2/AWQ "
            "support is reported separately."
        )

        write_json(q / "runtime-manifest.json", {
            "evidence": "server-log.txt",
            "engine": "V0",
            "attention_backend": "XFORMERS",
            "dtype": "float16",
            "tensor_parallel_size": 1,
            "quantization": None,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enforce_eager": True,
            "model": pin,
            "gpu_used_indices": [0],
            "launch_command_evidence": "server-command.txt",
            "kv_cache_dtype": "auto",
            "kv_cache_resolved_dtype": None,
            "kv_cache_resolved_dtype_reason": (
                "Startup logs record auto; not independently introspected."
            ),
        })

        tests = batch / "gpu_checks.py"
        tests.write_text(GPU_TEST)

        for mode, seconds in (
            ("torch_compile", 600),
            ("triton_minimal", 180),
            ("triton_rmsnorm", 180),
        ):
            path = batch / (mode + ".log")
            try:
                print(
                    f"Running {mode}; timeout {seconds}s; log: {path}",
                    flush=True,
                )
                gpu_idle()
                run(
                    [sys.executable, "-u", str(tests), mode],
                    path, seconds,
                    dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
                )
                record(
                    mode, "VALIDATED",
                    "Executed on T4 and passed numerical correctness checks.",
                    path,
                )
            except Exception as exc:
                record(
                    mode, "FAILED",
                    f"{type(exc).__name__}: {exc}",
                    path,
                )

        state["triton"] = (
            "VALIDATED"
            if state["triton_minimal"] == state["triton_rmsnorm"] == "VALIDATED"
            else "FAILED"
        )
        state["reasons"]["triton"] = (
            "Scoped to minimal add and small RMSNorm; "
            "the chunked-prefill attention kernel in attempt b remains FAILED."
        )

        tp = batch / "tp2"
        tp.mkdir()

        if len(gpus) < 2:
            record(
                "tp2", "UNAVAILABLE",
                "Two visible T4s are required; no hardware fallback.",
                tp,
            )
        else:
            try:
                print(
                    "Starting TP=2 with the same FP16 checkpoint and two T4s.",
                    flush=True,
                )
                tp_command = command.copy()
                tp_command[
                    tp_command.index("--tensor-parallel-size") + 1
                ] = "2"
                tp_command += ["--distributed-executor-backend", "mp"]

                server_smoke(
                    tp, tp_command,
                    dict(settings, CUDA_VISIBLE_DEVICES="0,1"),
                    probe,
                    {
                        "tp": 2,
                        "revision": REVISION,
                        "quantization": "None",
                    },
                )
                environment["tp2_used_gpu_indices"] = [0, 1]
                record(
                    "tp2", "VALIDATED",
                    "Same FP16 checkpoint, two T4s, TP=2, mp executor, "
                    "real streaming generation; no speedup claim.",
                    tp,
                )
            except Exception as exc:
                record(
                    "tp2", "FAILED",
                    f"{type(exc).__name__}: {exc}",
                    tp,
                )

        awq = batch / "awq"
        awq.mkdir()

        from requests.exceptions import RequestException

        try:
            print(
                "Verifying and pinning "
                "Qwen/Qwen2.5-3B-Instruct-AWQ before loading.",
                flush=True,
            )
            gpu_idle()

            from huggingface_hub import HfApi, hf_hub_download
            from transformers import AutoTokenizer

            awq_repo = REPO + "-AWQ"
            info = HfApi().model_info(awq_repo, timeout=60)
            assert re.fullmatch(r"[0-9a-f]{40}", info.sha)

            provenance = (
                info.card_data.to_dict().get("base_model")
                if info.card_data else None
            )
            assert provenance == REPO or provenance == [REPO], (
                "Unverified AWQ source model"
            )

            awq_pin = {
                "repo_id": awq_repo,
                "revision": info.sha,
                "tokenizer_revision": info.sha,
                "base_model": provenance,
            }
            write_json(awq / "model-pin.json", awq_pin)

            config = read_json(Path(hf_hub_download(
                awq_repo, "config.json", revision=info.sha
            )))
            source = read_json(Path(hf_hub_download(
                REPO, "config.json", revision=REVISION
            )))
            quantization = config["quantization_config"]
            assert quantization["quant_method"] == "awq"
            assert quantization["bits"] == 4

            for key in (
                "model_type", "hidden_size", "intermediate_size",
                "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "vocab_size",
            ):
                assert config[key] == source[key], (
                    "Source model mismatch: " + key
                )

            baseline_tokenizer = AutoTokenizer.from_pretrained(
                REPO, revision=REVISION, trust_remote_code=False
            )
            quant_tokenizer = AutoTokenizer.from_pretrained(
                awq_repo, revision=info.sha, trust_remote_code=False
            )
            assert (
                baseline_tokenizer.get_vocab()
                == quant_tokenizer.get_vocab()
            ), "Tokenizer vocabulary differs"
            assert (
                baseline_tokenizer.chat_template
                == quant_tokenizer.chat_template
            ), "Chat template differs"

            write_json(awq / "provenance.json", {
                "source_model": pin,
                "quantized_model": awq_pin,
                "quantization_config": quantization,
                "same_architecture": True,
                "same_vocabulary": True,
                "same_chat_template": True,
                "limitation": (
                    "Publisher identifies the source repo but does not pin "
                    "its original quantization-source commit."
                ),
            })

            awq_command = command.copy()
            for flag in ("--model", "--tokenizer"):
                awq_command[awq_command.index(flag) + 1] = awq_repo
            for flag in ("--revision", "--tokenizer-revision"):
                awq_command[awq_command.index(flag) + 1] = info.sha
            awq_command += ["--quantization", "awq"]

            server_smoke(
                awq, awq_command, settings, probe,
                {
                    "tp": 1,
                    "revision": info.sha,
                    "quantization": "awq",
                },
            )
            record(
                "awq", "VALIDATED",
                "Publisher-linked same-source AWQ, matching tokenizer/template, "
                "effective quantization=awq, FP16 activations, TP=1, "
                "real stream; no performance claim.",
                awq,
            )
        except RequestException as exc:
            record(
                "awq", "UNAVAILABLE",
                f"Model metadata/download unavailable: "
                f"{type(exc).__name__}: {exc}",
                awq,
            )
        except Exception as exc:
            record(
                "awq", "FAILED",
                f"{type(exc).__name__}: {exc}",
                awq,
            )

        write_json(q / "environment.json", environment)

    except Exception as exc:
        state["batch_error"] = f"{type(exc).__name__}: {exc}"
        print(
            "Batch stopped safely: " + state["batch_error"],
            flush=True,
        )
        for cap in CAPABILITIES[2:]:
            if state[cap] == "NOT_TESTED":
                state["reasons"][cap] = (
                    "Batch prerequisite failed: " + state["batch_error"]
                )

    finally:
        state["qualification_complete"] = (
            all(state[cap] != "NOT_TESTED" for cap in CAPABILITIES)
            and state["single_gpu_fp16_vllm"] == "VALIDATED"
        )
        state["limitations"] = [
            "Small correctness and generation smoke checks only; "
            "no SLO, p99, or performance certification.",
            "Chunked prefill enabled failed in attempt b; "
            "baseline c explicitly disables it.",
            "Kernel Lab portfolio implementation and all platform "
            "implementation remain unstarted.",
            "Upstream cancellation/error-path coverage and exact runtime "
            "KV dtype introspection remain untested.",
        ]
        write_json(q / "qualification.json", state)

        if not (old / "qualification.json").exists():
            failed = {
                "qualification_id": old.name,
                **{cap: "NOT_TESTED" for cap in CAPABILITIES},
                "single_gpu_fp16_vllm": "FAILED",
                "streaming": "FAILED",
                "validated_versions": {},
                "model": pin,
                "qualification_complete": False,
                "failure": state["known_failure"],
                "reason": (
                    "Engine crashed in chunked-prefill Triton kernel "
                    "before any content; no completed stream "
                    "or calculable metrics."
                ),
                "remaining_capabilities_reason": (
                    "Separate checks were not run under this failed attempt; "
                    "see qualification c."
                ),
            }
            write_json(old / "qualification.json", failed)

        if (
            not (old / "environment.json").exists()
            and (old / "environment-preflight.json").exists()
        ):
            write_json(old / "environment.json", {
                **read_json(old / "environment-preflight.json"),
                "model": pin,
            })

        if not (old / "qualification-report.md").exists():
            (old / "qualification-report.md").write_text(
                "# Qualification " + old.name
                + "\n\nFAILED: FP16 serving with chunked prefill enabled "
                "crashed on first generation. Triton prefix-prefill requested "
                "98304 bytes shared memory; hardware limit was 65536. "
                "The client received no content and timed out. "
                "Empty stream-sample.txt represents no observed events. "
                "Versions were installed, not certified by this failed attempt. "
                "Remaining capabilities were NOT_TESTED. "
                "See preserved server-log.txt, diagnostics, and "
                "qualification.json. Follow-up c explicitly disables chunked "
                "prefill and keeps the same versions/model/dtype/backend.\n"
            )

        rows = "\n".join(
            f"| {cap} | {state[cap]} | {state['reasons'][cap]} |"
            for cap in CAPABILITIES
        )
        (q / "qualification-report.md").write_text(
            "# Qualification " + q.name + "\n\n"
            "Baseline: pinned Qwen2.5-3B, one T4, FP16, TP=1, "
            "V0/XFORMERS, eager, prefix caching and chunked prefill "
            "disabled.\n\n"
            "| Capability | Status | Evidence / reason |\n"
            "| --- | --- | --- |\n"
            + rows
            + "\n\nFull details, limitations, versions, and artifact "
            "references: `qualification.json`. "
            "The failed chunked-prefill attempt b remains preserved. "
            "No benchmark performance claim is made.\n"
        )

        archive = q.parent.parent / (
            "inferencelab-qualification-" + batch.name + ".zip"
        )
        with zipfile.ZipFile(
            archive, "w", compression=zipfile.ZIP_DEFLATED
        ) as out:
            for directory in (old, q):
                for current, dirs, files in os.walk(directory):
                    dirs[:] = [
                        name for name in dirs
                        if name not in (
                            ".venv", "bootstrap", "__pycache__", ".cache"
                        )
                    ]
                    for name in files:
                        path = Path(current) / name
                        if not path.is_symlink():
                            out.write(
                                path, path.relative_to(q.parent.parent)
                            )

        print("\n" + json.dumps({
            cap: state[cap] for cap in CAPABILITIES
        }, indent=2), flush=True)
        print(
            "Qualification complete:",
            state["qualification_complete"],
            flush=True,
        )
        print("DOWNLOAD THIS ARCHIVE:", archive, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(
            "/kaggle/working/qualification/kaggle-t4-20261025-c"
        ),
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        main(args.directory)
