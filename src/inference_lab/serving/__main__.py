"""`python -m inference_lab.serving`: explicit launch, then a separate readiness probe."""

import argparse
import asyncio
import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from inference_lab.config.io import load_experiment
from inference_lab.config.settings import Settings
from inference_lab.serving.client import StreamError, readiness
from inference_lab.serving.runtime import (
    LaunchPlan,
    gpu_observations,
    installed_versions,
    launch_plan,
    process_identity,
    runtime_manifest,
)


def _write(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _serve(args: argparse.Namespace) -> None:
    plan = launch_plan(load_experiment(args.experiment), python=args.python)
    # A fresh directory prevents mixing logs/manifests from different server lives.
    args.output.mkdir(parents=True, exist_ok=False)
    _write(args.output / "launch.json", plan.model_dump(mode="json"))
    try:
        versions = installed_versions(args.python)
        _write(args.output / "versions.json", versions)
        if versions != plan.expected_versions:
            raise ValueError(
                "GPU interpreter versions differ from qualification; evidence preserved"
            )
        for key, value in os.environ.items():
            if key.startswith("VLLM_") and plan.environment.get(key) != value:
                raise ValueError(
                    "Unexpected VLLM environment override; use a clean GPU launch shell"
                )
        if (
            "CUDA_VISIBLE_DEVICES" in os.environ
            and os.environ["CUDA_VISIBLE_DEVICES"] != plan.environment["CUDA_VISIBLE_DEVICES"]
        ):
            raise ValueError("CUDA_VISIBLE_DEVICES conflicts with the explicit TP launch profile")
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(("127.0.0.1", plan.experiment.target.url.port or 80))
        identity = process_identity(os.getpid())
        if identity is None:
            raise ValueError("This GPU launch profile requires Linux /proc process identity")
        gpu = gpu_observations(os.getpid(), identity[1])
        _write(
            args.output / "gpu-before.json", {key: value.model_dump() for key, value in gpu.items()}
        )
        inventory = gpu["gpu.present"].value
        selected = {int(index) for index in plan.environment["CUDA_VISIBLE_DEVICES"].split(",")}
        if not isinstance(inventory, list):
            raise ValueError("Cannot verify reference T4 hardware with nvidia-smi")
        devices = [row for row in inventory if isinstance(row, dict) and row["index"] in selected]
        if len(devices) != len(selected) or any(row["name"] != "Tesla T4" for row in devices):
            raise ValueError("Selected GPU indices do not identify the qualified Tesla T4 hardware")
        _write(
            args.output / "server-process.json", {"pid": os.getpid(), "start_ticks": identity[1]}
        )
        print(f"Server log: {args.output / 'server.log'}", flush=True)
        with (args.output / "server.log").open("x") as log:
            os.dup2(log.fileno(), sys.stdout.fileno())
            os.dup2(log.fileno(), sys.stderr.fileno())
            os.execvpe(args.python, plan.argv, dict(os.environ) | plan.environment)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        _write(
            args.output / "launch-failure.json",
            {"status": "FAILED", "error_type": type(error).__name__},
        )
        raise


async def _probe(args: argparse.Namespace) -> int:
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        raise ValueError("Startup timeout must be finite and positive")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + args.startup_timeout
    files = ("launch.json", "versions.json", "server-process.json", "server.log")
    try:
        async with asyncio.timeout(args.startup_timeout):
            while not all((args.output / name).is_file() for name in files):
                if (args.output / "launch-failure.json").exists():
                    raise ValueError("Launch preflight failed; inspect preserved launch evidence")
                await asyncio.sleep(0.05)
    except TimeoutError:
        raise ValueError("Launch artifacts unavailable before startup deadline") from None
    plan = LaunchPlan.model_validate_json((args.output / "launch.json").read_text())
    # Refuse hand-edited launch arguments; the manifest must match our explicit plan.
    if not plan.argv or plan != launch_plan(plan.experiment, python=plan.argv[0]):
        raise ValueError("Launch plan differs from the qualified command builder")
    manifest_path = args.output / "runtime_manifest.json"
    if manifest_path.exists():
        raise ValueError("Manifest already exists; preserve it and use a fresh launch directory")
    process = json.loads((args.output / "server-process.json").read_text())
    identity = process_identity(process["pid"])
    smoke: dict[str, Any]
    if identity is None or identity[1] != process["start_ticks"]:
        smoke = {"status": "FAILED", "error": "server_process_missing_or_replaced"}
    else:
        settings = Settings()
        config = plan.experiment
        payload = {
            "model": config.model.repo_id,
            "messages": [{"role": "user", "content": "Count upwards from one."}],
            **config.generation.model_dump(),
            "seed": config.workload.seed,
            "stream_options": {"include_usage": True},
        }
        try:
            # A healthy port may belong to an older server. Require this launch's
            # own startup completion before probing, and recheck its PID afterward.
            async with asyncio.timeout(max(0.001, deadline - loop.time())):
                while "Application startup complete." not in (args.output / "server.log").read_text(
                    errors="replace"
                ):
                    current = process_identity(process["pid"])
                    if current is None or current[1] != process["start_ticks"]:
                        raise StreamError("server_process_missing_or_replaced")
                    await asyncio.sleep(0.05)
            smoke = await readiness(
                str(config.target.url),
                payload,
                startup_timeout=max(0.001, deadline - loop.time()),
                total_timeout=settings.vllm_timeout_seconds,
                idle_timeout=settings.vllm_stream_idle_timeout_seconds,
            )
        except TimeoutError:
            smoke = {"status": "FAILED", "error": "startup_timeout"}
        except StreamError as error:
            smoke = {"status": "FAILED", "error": error.code, "http_status": error.status_code}
        current = process_identity(process["pid"])
        if current is None or current[1] != process["start_ticks"]:
            smoke = {"status": "FAILED", "error": "server_process_missing_or_replaced"}
    gpu = gpu_observations(process["pid"], process["start_ticks"])
    manifest = runtime_manifest(
        plan,
        log=(args.output / "server.log").read_text(errors="replace"),
        gpu=gpu,
        smoke=smoke,
        versions=json.loads((args.output / "versions.json").read_text()),
    )
    _write(manifest_path, manifest.model_dump(mode="json"))
    print(f"Readiness: {manifest.status}; evidence: {manifest_path}")
    return 0 if manifest.status == "VALIDATED" else (1 if manifest.status == "FAILED" else 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("command", "serve"):
        command = commands.add_parser(action)
        command.add_argument("experiment", type=Path)
        command.add_argument("--python", default=sys.executable, help="Qualified GPU interpreter")
        if action == "serve":
            command.add_argument(
                "--output", type=Path, required=True, help="Fresh artifact directory"
            )
    probe = commands.add_parser("probe")
    probe.add_argument(
        "--output", type=Path, required=True, help="Existing launch artifact directory"
    )
    probe.add_argument("--startup-timeout", type=float, default=600)
    args = parser.parse_args()
    try:
        if args.action == "command":
            print(launch_plan(load_experiment(args.experiment), python=args.python).command)
        elif args.action == "serve":
            _serve(args)
        else:
            return asyncio.run(_probe(args))
    except (ValidationError, yaml.YAMLError):
        # Input validation errors can echo credentials; never print their full body.
        print(
            "Invalid serving configuration; check field types and qualified settings.",
            file=sys.stderr,
        )
        return 1
    except (OSError, subprocess.SubprocessError):
        print(
            "Serving I/O or subprocess failed; evidence preserved where available.", file=sys.stderr
        )
        return 1
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
