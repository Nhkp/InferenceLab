
import hashlib
import json
import platform
import re
import sys
from importlib.metadata import version
from pathlib import Path

import torch

q = Path(sys.argv[1])
environment = {
    "qualification_id": q.name,
    "python": platform.python_version(),
    "os": platform.platform(),
    "versions": {
        p: version(p)
        for p in ("torch", "vllm", "triton", "transformers")
    },
    "pytorch_cuda_runtime": torch.version.cuda,
    "visible_devices": "0",
    "gpu_visible_count": torch.cuda.device_count(),
    "gpus": [],
}
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    environment["gpus"].append({
        "index": i, "name": p.name,
        "vram_bytes": p.total_memory,
        "compute_capability": [p.major, p.minor],
    })
(q / "environment-preflight.json").write_text(
    json.dumps(environment, indent=2) + "\n"
)
print(json.dumps(environment, indent=2), flush=True)

assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 1, "Expected only GPU 0 visible"
p = torch.cuda.get_device_properties(0)
assert "T4" in p.name and (p.major, p.minor) == (7, 5)

x = torch.ones((32, 32), device="cuda:0", dtype=torch.float16)
torch.testing.assert_close(x @ x, torch.full_like(x, 32))
torch.cuda.synchronize()
print("CUDA FP16 matrix-multiplication smoke: VALIDATED", flush=True)

# Use the real server entrypoint to expose dependency/API incompatibilities.
import subprocess
with (q / "server-import-check.txt").open("a") as log:
    result = subprocess.run(
        [sys.executable, "-m",
         "vllm.entrypoints.openai.api_server", "--help"],
        stdout=log, stderr=subprocess.STDOUT, timeout=180,
    )
assert result.returncode == 0, "See server-import-check.txt"
print("Server entrypoint import check: passed", flush=True)

from huggingface_hub import HfApi
from transformers import AutoTokenizer

repo = "Qwen/Qwen2.5-3B-Instruct"
pin_path = q / "model-pin.json"
if pin_path.exists():
    pin = json.loads(pin_path.read_text())
    assert pin["repo_id"] == repo, "Existing model pin differs"
    revision = pin["revision"]
else:
    revision = HfApi().model_info(repo).sha

assert re.fullmatch(r"[0-9a-f]{40}", revision), "Invalid commit revision"
pin = {
    "repo_id": repo,
    "revision": revision,
    "tokenizer_repo_id": repo,
    "tokenizer_revision": revision,
}
# Preserve the resolved revision before downloading tokenizer files.
pin_path.write_text(json.dumps(pin, indent=2) + "\n")

tokenizer = AutoTokenizer.from_pretrained(
    repo, revision=revision, trust_remote_code=False
)
assert tokenizer.chat_template, "Missing chat template"
pin["chat_template_sha256"] = hashlib.sha256(
    json.dumps(tokenizer.chat_template, ensure_ascii=False,
               sort_keys=True).encode("utf-8")
).hexdigest()
pin_path.write_text(json.dumps(pin, indent=2) + "\n")
print("Pinned model/tokenizer:", json.dumps(pin), flush=True)
print("Serving, streaming, torch.compile, Triton, TP2, AWQ: NOT_TESTED")
