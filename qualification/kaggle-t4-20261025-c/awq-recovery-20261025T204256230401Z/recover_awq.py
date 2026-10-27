"""Recover the blocked AWQ qualification without repeating passed GPU checks."""
import ast
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

q = Path("/kaggle/working/qualification/kaggle-t4-20261025-c")
old = q.with_name("kaggle-t4-20261025-b")
saved = Path("/kaggle/working/finish_00b.py")
source = saved.read_text()
tree = ast.parse(source)
# ponytail: reuse the AWQ block from this exact generated batch. Refuse changed
# structure; extract a named AWQ function if this recovery is reused further.
blocks = [n for n in ast.walk(tree) if isinstance(n, ast.Try) and any(
    isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
    and isinstance(s.value.func, ast.Name) and s.value.func.id == "record"
    and s.value.args and isinstance(s.value.args[0], ast.Constant)
    and s.value.args[0].value == "awq" for s in n.body)]
assert len(blocks) == 1, "Cannot identify the existing AWQ check; stopping."
awq_code = compile(ast.fix_missing_locations(ast.Module(body=blocks[0].body, type_ignores=[])), str(saved), "exec")

batch = q / ("awq-recovery-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
batch.mkdir()
shutil.copy2(saved, batch / "previous-finish_00b.py")
shutil.copy2(q / "qualification.json", batch / "previous-qualification.json")
if (q / "qualification-report.md").exists():
    shutil.copy2(q / "qualification-report.md", batch / "previous-report.md")
shutil.copy2(__file__, batch / "recover_awq.py")

# Fix the shared cleanup function, including all its existing callers.
replacement = r"""def stop_group(pid):
    def identity(pid):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            return int(fields[1]), int(fields[19])
        except (OSError, ValueError, IndexError):
            return None

    # vLLM workers may create new sessions. Snapshot descendants before stopping
    # their parent; start times prevent signalling a subsequently reused PID.
    processes = {int(p.name): identity(int(p.name)) for p in Path("/proc").glob("[0-9]*")}
    owned = {pid} if processes.get(pid) else set()
    while children := {p for p, info in processes.items() if info and info[0] in owned} - owned:
        owned.update(children)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for child in owned:
            current = identity(child)
            if current and current[1] == processes[child][1]:
                try:
                    os.kill(child, sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM and owned:
            time.sleep(3)"""
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stop_group")
lines = source.splitlines(keepends=True)
lines[node.lineno-1:node.end_lineno] = [replacement + "\n"]
saved.write_text("".join(lines))
helpers = runpy.run_path(str(saved))
read_json, write_json = helpers["read_json"], helpers["write_json"]
state = read_json(q / "qualification.json")
assert state["qualification_id"] == q.name
caps = helpers["CAPABILITIES"]
assert all(state[c] == "VALIDATED" for c in caps if c != "awq")
state["awq"] = "NOT_TESTED"
state["reasons"]["awq"] = "Previous attempt was blocked before AWQ execution by a leftover TP=2 worker."
state["qualification_complete"] = False
state["status_correction"] = "Prior AWQ=FAILED represented a failed GPU-idle prerequisite, not an AWQ capability failure."
write_json(q / "qualification.json", state)

awq = batch / "awq"
awq.mkdir()
awq_started = False

def record(name, status, reason, evidence):
    state[name] = status
    state["reasons"][name] = reason
    state["evidence"][name] = str(evidence.relative_to(q))
    write_json(q / "qualification.json", state)
    print(f"{name}: {status} — {reason}", flush=True)

try:
    settings, command = helpers["launch_command"](q / "server-command.txt")
    assert Path(sys.executable).resolve() == Path(command[0]).resolve()
    for package, pin_version in {"torch": "2.8.0", "vllm": "0.10.2", "triton": "3.4.0", "transformers": "4.55.2", "xformers": "0.0.32.post1"}.items():
        assert helpers["version"](package).split("+")[0] == pin_version
    pin = read_json(q / "model-pin.json")
    probe = q / "stream-probe.py"
    tp_path = (q / state["evidence"]["tp2"]).resolve()
    assert tp_path.is_relative_to(q.resolve())
    tp_log = tp_path / "server-log.txt"
    text = tp_log.read_text(errors="replace")
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise BlockingIOError("Cannot inspect active GPU processes: " + result.stderr)
    pids = sorted({int(line.strip()) for line in result.stdout.splitlines() if line.strip()})
    ownership = []
    boot_epoch = time.time() - time.clock_gettime(time.CLOCK_BOOTTIME)
    run_start = datetime.strptime(tp_path.parent.name, "additional-%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc).timestamp()
    for pid in pids:
        proc = Path(f"/proc/{pid}")
        if not proc.exists():
            continue
        fields = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
        started = boot_epoch + int(fields[19]) / os.sysconf("SC_CLK_TCK")
        executable = (proc / "exe").resolve()
        logged = re.search(r"\bpid(?:=|\s+)" + str(pid) + r"\b", text, re.IGNORECASE) is not None
        if not (logged and executable == Path(sys.executable).resolve()
                and run_start - 5 <= started <= tp_log.stat().st_mtime + 5):
            raise BlockingIOError(f"PID {pid} cannot be proven to be our TP=2 worker; no signal sent.")
        ownership.append({"pid": pid, "start_ticks": int(fields[19]), "executable": str(executable), "evidence": str(tp_log.relative_to(q))})
    write_json(batch / "worker-ownership.json", ownership)
    for worker in ownership:
        proc = Path(f"/proc/{worker['pid']}/stat")
        if proc.exists() and int(proc.read_text().rsplit(") ", 1)[1].split()[19]) == worker["start_ticks"]:
            print(f"Stopping verified TP=2 worker {worker['pid']}", flush=True)
            helpers["stop_group"](worker["pid"])
    deadline = time.monotonic() + 15
    while True:
        try:
            helpers["gpu_idle"]()
            break
        except (RuntimeError, BlockingIOError):
            if time.monotonic() >= deadline:
                raise BlockingIOError("GPU processes remain after targeted cleanup; AWQ not started.")
            time.sleep(1)
    print("GPU idle. Running only AWQ; all five passed capabilities are preserved.", flush=True)
    context = dict(helpers, q=q, awq=awq, command=command, settings=settings, probe=probe, pin=pin, record=record)
    awq_started = True
    exec(awq_code, context)
except Exception as exc:
    (batch / "recovery-error.txt").write_text(traceback.format_exc())
    from requests.exceptions import RequestException
    status = "NOT_TESTED" if not awq_started or isinstance(exc, BlockingIOError) else ("UNAVAILABLE" if isinstance(exc, RequestException) else "FAILED")
    record("awq", status, f"{type(exc).__name__}: {exc}", awq)
finally:
    state["qualification_complete"] = all(state[c] != "NOT_TESTED" for c in caps) and state["single_gpu_fp16_vllm"] == "VALIDATED"
    write_json(q / "qualification.json", state)
    rows = "\n".join(f"| {c} | {state[c]} | {state['reasons'][c]} |" for c in caps)
    (q / "qualification-report.md").write_text("# Qualification " + q.name + "\n\n| Capability | Status | Reason |\n| --- | --- | --- |\n" + rows + "\n\nThe earlier AWQ result was a blocked prerequisite, not AWQ execution. Its report and archive remain preserved. Baseline serving requires chunked prefill disabled; the earlier enabled configuration failed. Only correctness and functional smoke checks were performed; no performance claim is made. Full limitations and evidence references are in qualification.json.\n")
    archive = q.parent.parent / ("inferencelab-qualification-" + batch.name + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for directory in (old, q):
            for current, dirs, files in os.walk(directory):
                dirs[:] = [d for d in dirs if d not in (".venv", "bootstrap", "__pycache__", ".cache")]
                for name in files:
                    path = Path(current) / name
                    if not path.is_symlink():
                        out.write(path, path.relative_to(q.parent.parent))
    print(helpers["json"].dumps({c: state[c] for c in caps}, indent=2), flush=True)
    print("Qualification complete:", state["qualification_complete"], flush=True)
    print("DOWNLOAD THIS ARCHIVE:", archive, flush=True)
