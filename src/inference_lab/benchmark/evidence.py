"""Co-located runtime evidence; JSON artifacts remain independent of any database."""

import csv
import io
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

from inference_lab.config.schemas import Environment, ExperimentConfig, GPUDevice, RuntimeEvidence
from inference_lab.serving.runtime import RuntimeManifest


def command(argv: list[str]) -> str | None:
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=15, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment(
    config: ExperimentConfig,
    manifest: RuntimeManifest | None,
    output: Path,
) -> Environment:
    def save(name: str, text: str | None) -> None:
        (output / name).write_text(text + "\n" if text is not None else "UNAVAILABLE\n")

    present = []
    visible: list[str] = []
    used: list[str] = []
    nvidia = topology = driver = supported = toolkit = None
    packages: dict[str, Any] = {}
    if manifest is not None:
        nvidia = command(["nvidia-smi"])
        topology = command(["nvidia-smi", "topo", "-m"])
        inventory = command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        save("gpu-inventory.csv", inventory)
        if inventory is None:
            raise ValueError("Co-located GPU inventory is unavailable")
        for row in csv.reader(io.StringIO(inventory), skipinitialspace=True):
            if len(row) != 6:
                raise ValueError("GPU inventory format is unsupported")
            present.append(
                GPUDevice(
                    index=int(row[0]),
                    uuid=row[1],
                    name=row[2],
                    vram_bytes=int(float(row[3]) * 1024**2),
                    compute_capability=(int(row[4].split(".")[0]), int(row[4].split(".")[1])),
                )
            )
            driver = row[5]
        if nvidia and (match := re.search(r"CUDA Version:\s*([\d.]+)", nvidia)):
            supported = match[1]
        toolkit = command(["nvcc", "--version"])
        selected = {
            int(i) for i in manifest.expected.environment["CUDA_VISIBLE_DEVICES"].split(",")
        }
        visible = [device.uuid for device in present if device.index in selected]
        observations = manifest.observed["gpu.used"].value
        if isinstance(observations, list):
            used = sorted({str(row["uuid"]) for row in observations if isinstance(row, dict)})
        packages = {
            key.removeprefix("packages."): value.value
            for key, value in manifest.observed.items()
            if key.startswith("packages.")
        }
        if not used or len(used) != config.runtime.tensor_parallel_size:
            raise ValueError("Runtime manifest does not prove the used GPU identities")
    save("nvidia-smi.txt", nvidia)
    save("nvidia-smi-topo.txt", topology)
    interpreter = manifest.expected.argv[0] if manifest else __import__("sys").executable
    freeze = command(
        [
            interpreter,
            "-c",
            "import importlib.metadata as m; "
            "print(*sorted(d.metadata['Name']+'=='+d.version "
            "for d in m.distributions()),sep=chr(10))",
        ]
    )
    save("pip-freeze.txt", freeze)
    git_sha = command(["git", "rev-parse", "HEAD"])
    dirty = command(["git", "status", "--porcelain"])
    fields: dict[str, Any] = {
        "topology": topology,
        "nvidia_driver": driver,
        "driver_supported_cuda": supported,
        "pytorch_cuda_runtime": packages.get("pytorch_cuda_runtime"),
        "cuda_toolkit": toolkit,
        "torch_version": packages.get("torch"),
        "vllm_version": packages.get("vllm"),
        "triton_version": packages.get("triton"),
        "git_sha": git_sha,
    }
    return Environment(
        python=platform.python_version(),
        os=platform.platform(),
        cpu=platform.processor() or platform.machine(),
        ram_bytes=os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
        gpu_present=present,
        gpu_visible_uuids=visible,
        gpu_used_uuids=used,
        git_dirty=bool(dirty) if dirty is not None else True,
        missing_reasons={
            key: "Not observed in this co-located run"
            for key, value in fields.items()
            if value is None
        },
        **fields,
    )


def runtime_evidence(config: ExperimentConfig, manifest: RuntimeManifest | None) -> RuntimeEvidence:
    fields: dict[str, Any] = {
        "resolved": config.runtime if manifest is not None else None,
        "attention_backend": None,
        "linear_backend": None,
        "kv_cache_dtype_resolved": None,
    }
    if manifest is not None:
        for field in ("attention_backend", "linear_backend", "kv_cache_dtype_resolved"):
            fields[field] = manifest.observed["runtime." + field].value
    return RuntimeEvidence(
        source="startup_log" if manifest is not None else "synthetic",
        evidence_artifact="runtime_manifest.json",
        missing_reasons={
            key: "Not independently observed" for key, value in fields.items() if value is None
        },
        **fields,
    )
