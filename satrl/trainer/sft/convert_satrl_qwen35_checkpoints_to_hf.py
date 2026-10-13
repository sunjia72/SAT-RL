#!/usr/bin/env python3
"""Convert the latest SATRL Qwen3.5 VeRL actor checkpoints to HF folders.

Run this inside the project VeRL Apptainer image. The source checkpoint tree is
read-only input; each output is built in a temporary sibling directory,
validated, and atomically renamed into place.

Example host invocation:

    apptainer exec --nv \
      --bind "$HOME:$HOME" \
      --bind "$SCRATCH:$SCRATCH:ro" \
        --bind "$PROJECT:$PROJECT" \
      --bind /tmp:/tmp \
      "$APPTAINER_IMAGE" \
      env PYTHONDONTWRITEBYTECODE=1 \
          PYTHONPATH="$VERL_DIR" \
      python "$SATRL_ROOT/trainer/sft/convert_satrl_qwen35_checkpoints_to_hf.py" \
        --resume --gpu-smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SATRL_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = SATRL_ROOT.parent
DEFAULT_PROJECT = Path(os.environ.get("PROJECT", ARTIFACT_ROOT))
DEFAULT_SCRATCH = Path(
    os.environ.get("SCRATCH", Path(os.environ.get("TMPDIR", "/tmp")) / "satrl")
)
RUN_MODEL_RE = re.compile(r"Qwen3\.5-(?:4B|9B)(?:-SFT)?(?=-|$)")
STEP_RE = re.compile(r"global_step_(\d+)")
SHARD_RE = re.compile(r"(?P<kind>model|optim|extra_state)_world_size_(?P<world>\d+)_rank_(?P<rank>\d+)\.pt")
BAD_VISUAL_PREFIX = "model.language_model.visual."
GOOD_VISUAL_PREFIX = "model.visual."


@dataclass(frozen=True)
class RunSpec:
    name: str
    experiment_dir: Path
    step: int
    step_dir: Path
    actor_dir: Path
    base_model_dir: Path
    target_dir: Path


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_SCRATCH / "checkpoints",
        help="Root containing top-level SATRL experiment folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_PROJECT / "models" / "satrl_trained",
        help="Destination root for validated HF model folders.",
    )
    parser.add_argument(
        "--base-model-root",
        type=Path,
        default=DEFAULT_PROJECT / "models",
        help="Root containing Qwen3.5-{4B,9B}{,-SFT} base HF folders.",
    )
    parser.add_argument(
        "--verl-dir",
        type=Path,
        default=DEFAULT_PROJECT / "logic" / "verl",
        help="VeRL checkout providing verl.model_merger.",
    )
    parser.add_argument(
        "--repair-script",
        type=Path,
        default=Path(__file__).with_name("repair_qwen35_hf_visual_prefixes.py"),
        help="Qwen3.5 post-export repair utility.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="EXPERIMENT",
        help="Convert only this exact experiment basename; repeat as needed.",
    )
    parser.add_argument("--expected-count", type=int, default=37)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and skip already completed targets; never overwrite them.",
    )
    parser.add_argument(
        "--keep-failed-temp",
        action="store_true",
        help="Keep a failed temporary output for diagnosis instead of deleting it.",
    )
    parser.add_argument(
        "--gpu-smoke",
        action="store_true",
        help="After conversion, load and run one forward pass for every selected model.",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of independent CPU/RAM FSDP merge processes to run concurrently.",
    )

    # Standalone audit/debug modes used by this driver and its GPU workers.
    parser.add_argument("--validate-hf", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--base-model-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--expected-source", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-one", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def exact_rank_set(actor_dir: Path, kind: str, world_size: int) -> set[int]:
    ranks: set[int] = set()
    for path in actor_dir.glob(f"{kind}_world_size_{world_size}_rank_*.pt"):
        match = SHARD_RE.fullmatch(path.name)
        if match and match.group("kind") == kind and int(match.group("world")) == world_size:
            ranks.add(int(match.group("rank")))
    return ranks


def validate_actor(actor_dir: Path) -> int:
    config_path = actor_dir / "fsdp_config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing FSDP config: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        fsdp_config = json.load(handle)
    if fsdp_config.get("FSDP_version") != 2:
        raise ValueError(f"Expected FSDP version 2 in {config_path}: {fsdp_config}")
    world_size = fsdp_config.get("world_size")
    if not isinstance(world_size, int) or world_size < 1:
        raise ValueError(f"Invalid world_size in {config_path}: {world_size!r}")

    expected = set(range(world_size))
    for kind in ("model", "optim", "extra_state"):
        actual = exact_rank_set(actor_dir, kind, world_size)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"Incomplete {kind} shards in {actor_dir}: missing={missing}, extra={extra}")

    required_hf = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "processor_config.json",
    )
    for name in required_hf:
        path = actor_dir / "huggingface" / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty checkpoint HF metadata: {path}")
    return world_size


def discover_runs(args: argparse.Namespace) -> list[RunSpec]:
    root = args.checkpoint_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")
    if not args.base_model_root.is_dir():
        raise FileNotFoundError(f"Base-model root does not exist: {args.base_model_root}")

    selected_names = set(args.only)
    matched_names: set[str] = set()
    runs: list[RunSpec] = []
    for experiment_dir in sorted(root.iterdir(), key=lambda path: path.name):
        if not experiment_dir.is_dir() or "Qwen3.5" not in experiment_dir.name:
            continue
        if selected_names and experiment_dir.name not in selected_names:
            continue
        resolved_experiment = experiment_dir.resolve()
        if not resolved_experiment.is_relative_to(root):
            raise ValueError(f"Experiment path escapes checkpoint root: {experiment_dir} -> {resolved_experiment}")
        matched_names.add(experiment_dir.name)

        model_matches = RUN_MODEL_RE.findall(experiment_dir.name)
        if len(model_matches) != 1:
            raise ValueError(f"Could not unambiguously determine the base model from {experiment_dir.name}")
        base_model_dir = (args.base_model_root / model_matches[0]).resolve()
        if not base_model_dir.is_dir():
            raise FileNotFoundError(f"Base model does not exist for {experiment_dir.name}: {base_model_dir}")

        candidates: list[tuple[int, Path]] = []
        for step_dir in experiment_dir.iterdir():
            match = STEP_RE.fullmatch(step_dir.name)
            if not step_dir.is_dir() or match is None:
                continue
            candidates.append((int(match.group(1)), step_dir))
        if not candidates:
            raise ValueError(f"No global_step_N checkpoint under {experiment_dir}")

        step, step_dir = max(candidates, key=lambda item: item[0])
        actor_dir = step_dir / "actor"
        if not actor_dir.is_dir():
            raise ValueError(f"Numerically greatest checkpoint has no actor directory: {step_dir}")
        resolved_step = step_dir.resolve()
        resolved_actor = actor_dir.resolve()
        if not resolved_step.is_relative_to(resolved_experiment) or not resolved_actor.is_relative_to(resolved_step):
            raise ValueError(f"Selected checkpoint path escapes experiment directory: {actor_dir}")
        validate_actor(actor_dir)
        marker_path = experiment_dir / "latest_checkpointed_iteration.txt"
        if not marker_path.is_file():
            raise ValueError(f"Missing latest checkpoint marker: {marker_path}")
        try:
            marker = int(marker_path.read_text(encoding="utf-8").strip())
        except ValueError as error:
            raise ValueError(f"Invalid latest checkpoint marker: {marker_path}") from error
        if marker != step:
            raise ValueError(f"Latest marker {marker} does not match maximum step {step} in {experiment_dir}")

        runs.append(
            RunSpec(
                name=experiment_dir.name,
                experiment_dir=resolved_experiment,
                step=step,
                step_dir=step_dir.resolve(),
                actor_dir=actor_dir.resolve(),
                base_model_dir=base_model_dir,
                target_dir=(args.output_root / experiment_dir.name).resolve(),
            )
        )

    missing_requested = selected_names - matched_names
    if missing_requested:
        raise ValueError(f"Requested --only experiments were not found: {sorted(missing_requested)}")
    if not runs:
        raise ValueError(f"No top-level Qwen3.5 experiments matched under {root}")
    if not selected_names and len(runs) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} Qwen3.5 experiments, found {len(runs)}")
    return runs


def source_snapshot(runs: Iterable[RunSpec], checkpoint_root: Path) -> tuple[str, dict[str, tuple]]:
    root = checkpoint_root.resolve()
    entries: dict[str, tuple] = {}
    for run in runs:
        for path in sorted(run.experiment_dir.rglob("*")):
            info = path.lstat()
            relative = str(path.relative_to(root))
            link_target = os.readlink(path) if path.is_symlink() else ""
            entries[relative] = (
                stat.S_IFMT(info.st_mode),
                stat.S_IMODE(info.st_mode),
                info.st_size,
                info.st_mtime_ns,
                link_target,
            )
    digest = hashlib.sha256()
    for name, value in sorted(entries.items()):
        digest.update(json.dumps([name, *value], separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest(), entries


def _safetensor_inventory(model_dir: Path) -> tuple[dict[str, tuple[Path, tuple[int, ...], str]], dict]:
    from safetensors import safe_open

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(f"Missing safetensors index: {index_path}")
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid or empty weight_map in {index_path}")

    referenced_names = set(weight_map.values())
    disk_names = {path.name for path in model_dir.glob("*.safetensors") if path.is_file()}
    if disk_names != referenced_names:
        raise ValueError(
            f"Safetensor/index file mismatch in {model_dir}: "
            f"unindexed={sorted(disk_names - referenced_names)}, missing={sorted(referenced_names - disk_names)}"
        )

    dtype_bytes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    inventory: dict[str, tuple[Path, tuple[int, ...], str]] = {}
    tensor_payload_size = 0
    for filename in sorted(referenced_names):
        path = model_dir / filename
        if not path.is_file():
            raise ValueError(f"Indexed safetensor does not exist: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in inventory:
                    raise ValueError(f"Duplicate tensor key {key} in {path} and {inventory[key][0]}")
                tensor_slice = handle.get_slice(key)
                shape = tuple(tensor_slice.get_shape())
                dtype = tensor_slice.get_dtype()
                if dtype not in dtype_bytes:
                    raise ValueError(f"Unknown safetensors dtype {dtype} for {key}")
                elements = 1
                for dimension in shape:
                    elements *= dimension
                tensor_payload_size += elements * dtype_bytes[dtype]
                inventory[key] = (path, shape, dtype)

    if set(inventory) != set(weight_map):
        raise ValueError(
            f"Safetensor/index key mismatch in {model_dir}: "
            f"unindexed={len(set(inventory) - set(weight_map))}, missing={len(set(weight_map) - set(inventory))}"
        )
    for key, filename in weight_map.items():
        if inventory[key][0].name != filename:
            raise ValueError(f"Index maps {key} to {filename}, but it is stored in {inventory[key][0].name}")

    expected_total = sum((model_dir / name).stat().st_size for name in referenced_names)
    recorded_total = index.get("metadata", {}).get("total_size")
    # Standard HF indexes record tensor payload bytes. The local Qwen3.5 repair
    # utility records complete safetensor file bytes after adding MTP tensors.
    # Accept either convention, while rejecting unrelated/stale values.
    if recorded_total is not None and int(recorded_total) not in (expected_total, tensor_payload_size):
        raise ValueError(
            f"Index total_size mismatch in {model_dir}: recorded={recorded_total}, "
            f"file_bytes={expected_total}, tensor_bytes={tensor_payload_size}"
        )
    return inventory, index


def _assert_mtp_matches_base(
    model_dir: Path,
    base_model_dir: Path,
    target_inventory: dict[str, tuple[Path, tuple[int, ...], str]],
    base_inventory: dict[str, tuple[Path, tuple[int, ...], str]],
) -> None:
    import torch
    from safetensors import safe_open

    target_mtp = {key for key in target_inventory if key.startswith("mtp.")}
    base_mtp = {key for key in base_inventory if key.startswith("mtp.")}
    if not base_mtp:
        raise ValueError(f"Base model has no mtp.* tensors: {base_model_dir}")
    if target_mtp != base_mtp:
        raise ValueError(
            f"MTP key mismatch for {model_dir}: missing={sorted(base_mtp-target_mtp)}, "
            f"extra={sorted(target_mtp-base_mtp)}"
        )

    grouped: dict[tuple[Path, Path], list[str]] = {}
    for key in sorted(base_mtp):
        target_path, target_shape, target_dtype = target_inventory[key]
        base_path, base_shape, base_dtype = base_inventory[key]
        if target_shape != base_shape or target_dtype != base_dtype:
            raise ValueError(
                f"MTP metadata mismatch for {key}: target={target_shape}/{target_dtype}, "
                f"base={base_shape}/{base_dtype}"
            )
        grouped.setdefault((target_path, base_path), []).append(key)

    for (target_path, base_path), keys in grouped.items():
        with safe_open(target_path, framework="pt", device="cpu") as target_handle:
            with safe_open(base_path, framework="pt", device="cpu") as base_handle:
                for key in keys:
                    if not torch.equal(target_handle.get_tensor(key), base_handle.get_tensor(key)):
                        raise ValueError(f"Restored MTP tensor differs from base for {key}")


def validate_hf_model(model_dir: Path, base_model_dir: Path, expected_source: Path | None) -> dict[str, int]:
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    from verl.utils.transformers_compat import get_auto_model_for_vision2seq

    model_dir = model_dir.resolve()
    base_model_dir = base_model_dir.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"HF model directory does not exist: {model_dir}")
    if not base_model_dir.is_dir():
        raise FileNotFoundError(f"Base HF model directory does not exist: {base_model_dir}")

    if expected_source is not None:
        pointer = model_dir / "source_checkpoint.txt"
        if not pointer.is_file():
            raise ValueError(f"Missing source checkpoint pointer: {pointer}")
        recorded = pointer.read_text(encoding="utf-8").strip()
        if recorded != str(expected_source.resolve()):
            raise ValueError(f"Source pointer mismatch in {pointer}: {recorded!r}")

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if config.model_type != "qwen3_5" or "Qwen3_5ForConditionalGeneration" not in config.architectures:
        raise ValueError(f"Unexpected HF architecture in {model_dir}: {config.model_type}/{config.architectures}")
    base_config = AutoConfig.from_pretrained(base_model_dir, local_files_only=True, trust_remote_code=False)
    if config.model_type != base_config.model_type:
        raise ValueError(f"Output/base model_type mismatch: {config.model_type} vs {base_config.model_type}")
    for field in ("hidden_size", "num_hidden_layers"):
        output_value = getattr(config.text_config, field)
        base_value = getattr(base_config.text_config, field)
        if output_value != base_value:
            raise ValueError(f"Output/base text_config.{field} mismatch: {output_value} vs {base_value}")

    target_inventory, _ = _safetensor_inventory(model_dir)
    base_inventory, _ = _safetensor_inventory(base_model_dir)
    bad_visual = sorted(key for key in target_inventory if key.startswith(BAD_VISUAL_PREFIX))
    if bad_visual:
        raise ValueError(f"Bad Qwen3.5 visual prefixes remain in {model_dir}: {bad_visual[:5]}")
    visual_count = sum(key.startswith(GOOD_VISUAL_PREFIX) for key in target_inventory)
    language_count = sum(key.startswith("model.language_model.") for key in target_inventory)
    if visual_count == 0 or language_count == 0:
        raise ValueError(f"Missing visual/language tensors in {model_dir}")
    non_bf16 = sorted(key for key, (_, _, dtype) in target_inventory.items() if dtype != "BF16")
    if non_bf16:
        raise ValueError(f"Expected BF16 export; non-BF16 tensors include {non_bf16[:5]}")

    _assert_mtp_matches_base(model_dir, base_model_dir, target_inventory, base_inventory)

    auto_model_class = get_auto_model_for_vision2seq()
    with init_empty_weights():
        empty_model = auto_model_class.from_config(config, dtype=torch.bfloat16, trust_remote_code=False)
    expected_state = {key: tuple(tensor.shape) for key, tensor in empty_model.state_dict().items()}
    del empty_model
    # Qwen3.5's published snapshots carry supplemental mtp.* tensors even
    # though the Transformers model class does not register them in
    # state_dict(). Validate those exactly against the mapped base above, and
    # compare the registered model state independently here.
    expected_keys = {key for key in expected_state if not key.startswith("mtp.")}
    actual_keys = set(target_inventory)
    actual_registered_keys = {key for key in actual_keys if not key.startswith("mtp.")}
    if actual_registered_keys != expected_keys:
        raise ValueError(
            f"HF model-state mismatch in {model_dir}: "
            f"missing={sorted(expected_keys-actual_registered_keys)[:10]}, "
            f"unexpected={sorted(actual_registered_keys-expected_keys)[:10]}"
        )
    shape_mismatches = [
        (key, target_inventory[key][1], expected_state[key])
        for key in sorted(actual_registered_keys)
        if target_inventory[key][1] != expected_state[key]
    ]
    if shape_mismatches:
        raise ValueError(f"HF tensor-shape mismatches in {model_dir}: {shape_mismatches[:5]}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if len(tokenizer) < 1 or processor is None:
        raise ValueError(f"Tokenizer/processor validation failed for {model_dir}")
    return {
        "keys": len(actual_keys),
        "language": language_count,
        "visual": visual_count,
        "mtp": sum(key.startswith("mtp.") for key in actual_keys),
    }


def run_command(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    log("  command: " + " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def convert_one(run: RunSpec, args: argparse.Namespace, env: dict[str, str]) -> None:
    target = run.target_dir
    if target.exists():
        if not args.resume:
            raise FileExistsError(f"Target already exists; refusing to overwrite: {target}")
        counts = validate_hf_model(target, run.base_model_dir, run.actor_dir)
        log(f"[resume] validated existing {run.name}: {counts}")
        return

    temporary = args.output_root.resolve() / f".{run.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    if temporary.exists():
        raise FileExistsError(f"Temporary target unexpectedly exists: {temporary}")

    log(f"[convert] {run.name}")
    log(f"  source: {run.actor_dir}")
    log(f"  base:   {run.base_model_dir}")
    log(f"  target: {target}")
    try:
        merge_command = [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(run.actor_dir),
            "--target_dir",
            str(temporary),
        ]
        run_command(merge_command, cwd=args.verl_dir, env=env)
        repair_command = [
            sys.executable,
            str(args.repair_script),
            str(temporary),
            "--base-model-dir",
            str(run.base_model_dir),
        ]
        run_command(repair_command, cwd=args.repair_script.parent, env=env)
        (temporary / "source_checkpoint.txt").write_text(str(run.actor_dir) + "\n", encoding="utf-8")
        counts = validate_hf_model(temporary, run.base_model_dir, run.actor_dir)
        if target.exists():
            raise FileExistsError(f"Target appeared during conversion; refusing to overwrite: {target}")
        temporary.rename(target)
        log(f"[complete] {run.name}: {counts}")
    except BaseException:
        if temporary.exists():
            if args.keep_failed_temp:
                log(f"[failed] keeping temporary output for diagnosis: {temporary}")
            else:
                log(f"[failed] removing temporary output: {temporary}")
                shutil.rmtree(temporary)
        raise


def convert_worker(run: RunSpec, args: argparse.Namespace, env: dict[str, str]) -> str:
    # The FSDP merger is deliberately CPU-only. Hiding CUDA here avoids each
    # merge/repair worker reserving a GPU context; GPUs remain free for the
    # separate four-way smoke-validation phase.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    worker_env = env.copy()
    worker_env["CUDA_VISIBLE_DEVICES"] = ""
    convert_one(run, args, worker_env)
    return run.name


def smoke_one(model_dir: Path) -> None:
    import torch
    from transformers import AutoTokenizer
    from verl.utils.transformers_compat import get_auto_model_for_vision2seq

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for GPU smoke validation")
    auto_model_class = get_auto_model_for_vision2seq()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = auto_model_class.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    inputs = tokenizer("Hello", return_tensors="pt")
    inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs, use_cache=False)
    logits = output.logits[:, -1, :]
    if logits.ndim != 2 or not torch.isfinite(logits.float()).all():
        raise ValueError(f"Invalid forward-pass logits for {model_dir}: shape={tuple(logits.shape)}")
    log(f"GPU_SMOKE_OK {model_dir} logits={tuple(logits.shape)} dtype={logits.dtype}")


def gpu_smoke_all(runs: list[RunSpec], gpu_ids: list[str]) -> None:
    if not gpu_ids:
        raise ValueError("--gpu-ids did not contain any GPU IDs")
    buckets: list[list[RunSpec]] = [[] for _ in gpu_ids]
    for index, run in enumerate(runs):
        buckets[index % len(gpu_ids)].append(run)

    def worker(gpu_id: str, bucket: list[RunSpec]) -> list[str]:
        completed: list[str] = []
        for run in bucket:
            child_env = os.environ.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = gpu_id
            child_env["PYTHONDONTWRITEBYTECODE"] = "1"
            command = [sys.executable, str(Path(__file__).resolve()), "--smoke-one", str(run.target_dir)]
            result = subprocess.run(command, env=child_env, text=True, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"GPU smoke failed for {run.name} on GPU {gpu_id}\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
            log(result.stdout.strip())
            completed.append(run.name)
        return completed

    log(f"Starting GPU smoke validation on GPUs {gpu_ids}")
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {
            executor.submit(worker, gpu_id, bucket): gpu_id
            for gpu_id, bucket in zip(gpu_ids, buckets)
            if bucket
        }
        completed = 0
        for future in as_completed(futures):
            names = future.result()
            completed += len(names)
            log(f"GPU {futures[future]} smoke-complete models: {len(names)}")
    if completed != len(runs):
        raise RuntimeError(f"GPU smoke completed {completed}/{len(runs)} models")


def print_plan(runs: list[RunSpec]) -> None:
    log(f"Matched {len(runs)} Qwen3.5 experiments:")
    for run in runs:
        log(f"  {run.name}\tstep={run.step}\tbase={run.base_model_dir.name}")


def main() -> None:
    args = parse_args()
    if args.smoke_one is not None:
        smoke_one(args.smoke_one.resolve())
        return
    if args.validate_hf is not None:
        if args.base_model_dir is None:
            raise ValueError("--validate-hf requires --base-model-dir")
        counts = validate_hf_model(args.validate_hf, args.base_model_dir, args.expected_source)
        log(f"HF_VALIDATION_OK {args.validate_hf}: {counts}")
        return

    args.checkpoint_root = args.checkpoint_root.resolve()
    args.output_root = args.output_root.resolve()
    args.base_model_root = args.base_model_root.resolve()
    args.verl_dir = args.verl_dir.resolve()
    args.repair_script = args.repair_script.resolve()
    if not args.verl_dir.is_dir():
        raise FileNotFoundError(f"VeRL checkout does not exist: {args.verl_dir}")
    if not args.repair_script.is_file():
        raise FileNotFoundError(f"Qwen3.5 repair script does not exist: {args.repair_script}")
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")

    runs = discover_runs(args)
    print_plan(runs)
    before_digest, before_snapshot = source_snapshot(runs, args.checkpoint_root)
    log(f"Source metadata manifest before conversion: {before_digest} ({len(before_snapshot)} paths)")

    if args.dry_run:
        for run in runs:
            log(
                f"DRY_RUN {sys.executable} -m verl.model_merger merge --backend fsdp "
                f"--local_dir {run.actor_dir} --target_dir {run.target_dir}"
            )
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.output_root).free
    estimated_bytes = sum(
        path.stat().st_size
        for run in runs
        for path in run.base_model_dir.rglob("*")
        if path.is_file()
    )
    largest_base = max(
        sum(path.stat().st_size for path in run.base_model_dir.rglob("*") if path.is_file()) for run in runs
    )
    required_headroom = estimated_bytes + 2 * largest_base * args.jobs
    log(
        f"Capacity preflight: free={free_bytes} estimated_outputs={estimated_bytes} "
        f"required_with_temp_headroom={required_headroom}"
    )
    if free_bytes < required_headroom:
        raise OSError("Insufficient destination free space for outputs plus temporary headroom")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.verl_dir) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    try:
        if args.jobs == 1:
            for index, run in enumerate(runs, start=1):
                log(f"=== [{index}/{len(runs)}] {run.name} ===")
                convert_one(run, args, env)
        else:
            log(f"Starting {len(runs)} conversions with {args.jobs} independent merge processes")
            executor = ProcessPoolExecutor(max_workers=args.jobs)
            futures = {executor.submit(convert_worker, run, args, env): run for run in runs}
            completed = 0
            try:
                for future in as_completed(futures):
                    name = future.result()
                    completed += 1
                    log(f"PARALLEL_PROGRESS completed={completed}/{len(runs)} latest={name}")
            except BaseException:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
        if args.gpu_smoke:
            gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
            gpu_smoke_all(runs, gpu_ids)
    finally:
        after_digest, after_snapshot = source_snapshot(runs, args.checkpoint_root)
        log(f"Source metadata manifest after conversion:  {after_digest} ({len(after_snapshot)} paths)")
        if before_snapshot != after_snapshot:
            changed = sorted(set(before_snapshot) ^ set(after_snapshot))
            changed.extend(
                name
                for name in sorted(set(before_snapshot) & set(after_snapshot))
                if before_snapshot[name] != after_snapshot[name]
            )
            raise RuntimeError(f"SOURCE CHECKPOINT TREE CHANGED: {changed[:20]}")
        log("SOURCE_CHECKPOINT_METADATA_UNCHANGED")

    log(f"ALL_DONE converted_or_validated={len(runs)} output_root={args.output_root}")


if __name__ == "__main__":
    main()
