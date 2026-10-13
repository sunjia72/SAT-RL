#!/usr/bin/env python3
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

PORTS_TO_FREE = [8080, 10080, 20080]

# ───────────────────────── YAML SAFE UPDATE HELPERS ─────────────────────────────

def safe_load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML dict; if missing/empty/invalid/not-a-dict => {}."""
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def atomic_write_text(path: Path, text: str) -> None:
    """Atomic write via temp + replace (prevents truncation on crash)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _lock_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


class FileLock:
    """
    Simple lockfile using O_EXCL. Helps avoid concurrent writers clobbering YAML.
    """
    def __init__(self, target: Path, timeout_s: float = 30.0, poll_s: float = 0.1):
        self.lockfile = _lock_path(target)
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.fd: Optional[int] = None

    def __enter__(self):
        deadline = time.time() + self.timeout_s
        while True:
            try:
                self.fd = os.open(str(self.lockfile), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode("utf-8"))
                return self
            except FileExistsError:
                if time.time() > deadline:
                    raise TimeoutError(f"Timeout acquiring lock: {self.lockfile}")
                time.sleep(self.poll_s)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.fd is not None:
                os.close(self.fd)
            if self.lockfile.exists():
                self.lockfile.unlink()
        except Exception:
            pass


def patch_yaml_file(path: Path, patch_fn: Callable[[Dict[str, Any]], None], lock: bool = True) -> None:
    """
    Read YAML dict (or {}), apply patch_fn(cfg) in-place, then atomic write back.
    Does NOT truncate original file mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def do_patch():
        cfg = safe_load_yaml(path)
        patch_fn(cfg)
        dumped = yaml.safe_dump(
            cfg,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
        atomic_write_text(path, dumped)

    if lock:
        with FileLock(path):
            do_patch()
    else:
        do_patch()


# ───────────────────────── ORIGINAL FUNCTIONALITY ───────────────────────────────

def free_ports(ports):
    for port in ports:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f"TCP:{port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            pids = [p for p in result.stdout.strip().split("\n") if p]
            for pid in pids:
                subprocess.run(
                    ["kill", "-9", pid],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
        except Exception:
            pass


# Environment setup
env = os.environ.copy()
env.setdefault("PORT", "10080")  # default to 10080 if not provided
port = env["PORT"]

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT = env.get("PROJECT")
HOME = env.get("HOME", str(Path.home()))
if not PROJECT:
    raise RuntimeError("PROJECT not found in environment")

SCRATCH = env.get("SCRATCH", os.path.join(PROJECT, "scratch"))
SIF_PATH = os.path.join(PROJECT, "sandbox", "code_sandbox_server.sif")

# Free up ports if already used
free_ports(PORTS_TO_FREE)

# Get node IP (your original code used hostname; keep it)
try:
    node_ip = subprocess.check_output(["sh", "-c", "hostname"], text=True).strip()
except subprocess.CalledProcessError:
    node_ip = "127.0.0.1"

jobid = env.get("SLURM_JOB_ID")
if not jobid:
    raise RuntimeError("Job ID not found in environment (SLURM_JOB_ID is missing)")

# Path for environment file
params_dir = Path(PROJECT) / "logic" / "output" / "params"


params_dir.mkdir(parents=True, exist_ok=True)

# Construct URL
sandbox_fusion_url = f"http://{node_ip}:{port}/run_code"
yaml_file = params_dir / f"sf_{jobid}.yaml"

# ✅ NEW: patch YAML instead of overwriting it
def patch_sf_yaml(cfg: Dict[str, Any]) -> None:
    tools = cfg.get("tools")
    if not isinstance(tools, list):
        tools = []
        cfg["tools"] = tools

    # find existing SandboxFusionTool entry
    target = None
    for t in tools:
        if isinstance(t, dict) and t.get("class_name") == "verl.tools.sandbox_fusion_tools.SandboxFusionTool":
            target = t
            break

    if target is None:
        target = {}
        tools.append(target)

    target["class_name"] = "verl.tools.sandbox_fusion_tools.SandboxFusionTool"

    # config section: patch keys we own, keep others
    config = target.get("config")
    if not isinstance(config, dict):
        config = {}
        target["config"] = config

    config.setdefault("type", "native")
    config["sandbox_fusion_url"] = sandbox_fusion_url
    config["num_workers"] = os.cpu_count()
    config["enable_global_rate_limit"] = True
    config["rate_limit"] =os.cpu_count()
    config["default_timeout"] = 20
    config["default_language"] = "python"

    # tool_schema: set the schema you need; preserve other top-level keys in target
    target["tool_schema"] = {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "A tool for executing code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The code to execute."}
                },
                "required": ["code"],
            },
        },
    }

# Apply patch atomically (prevents truncation)
patch_yaml_file(yaml_file, patch_fn=patch_sf_yaml, lock=True)

# Build the Apptainer exec command
command = [
    "apptainer", "exec",
    "--env", f"PORT={port}",
    "--env", "OPENBLAS_NUM_THREADS=1",
    "--env", "MALLOC_CONF=background_thread:false",
    "--env", "JEMALLOC_CONF=background_thread:false",
    "--env", "OMP_NUM_THREADS=1",
    "--env", "MKL_NUM_THREADS=1",
    "--bind", f"{HOME}:{HOME}",
    "--bind", f"{SCRATCH}:{SCRATCH}",
    # "--bind", f"{PROJECT}:{PROJECT}",
    SIF_PATH,
    "make", "run-online"
]

# Launch the Apptainer container (same as before)
subprocess.run(command, env=env, cwd=str(BASE_DIR))
