#!/usr/bin/env python3
"""Repair Qwen3.5 HF exports produced from verl FSDP checkpoints.

Some VeRL FSDP -> HF conversions for Qwen3.5 can write visual weights as
`model.language_model.visual.*`.  The HF/vLLM layout for
Qwen3_5ForConditionalGeneration expects `model.visual.*`.

The training checkpoint may also omit base-only `mtp.*` tensors even though the
published HF snapshot includes them.  When a base model directory is provided,
this script restores those tensors and syncs tokenizer/processor sidecar files
without overwriting the SFT model config or trained language/visual weights.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


BAD_PREFIX = "model.language_model.visual."
GOOD_PREFIX = "model.visual."
DEFAULT_FILL_PREFIXES = ("mtp.",)
BASE_SIDECAR_FILES = (
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)
STALE_SIDECAR_FILES = (
    "processor_config.json",
    "generation_config.json",
)


def _is_qwen35_conditional_generation(model_dir: Path) -> bool:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return False

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    return (
        config.get("model_type") == "qwen3_5"
        and "Qwen3_5ForConditionalGeneration" in config.get("architectures", [])
    )


def _load_index(model_dir: Path) -> tuple[Path | None, dict | None]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return None, None

    with index_path.open("r", encoding="utf-8") as f:
        return index_path, json.load(f)


def _safetensor_files(model_dir: Path, index: dict | None) -> list[Path]:
    if index is not None:
        return sorted({model_dir / name for name in index.get("weight_map", {}).values()})

    single_file = model_dir / "model.safetensors"
    return [single_file] if single_file.exists() else []


def _collect_safetensor_keys(model_dir: Path, index: dict | None) -> dict[str, Path]:
    key_to_file = {}
    for path in _safetensor_files(model_dir, index):
        if not path.exists():
            raise FileNotFoundError(f"Missing safetensors file: {path}")
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in key_to_file:
                    raise ValueError(f"Duplicate tensor key {key} in {path} and {key_to_file[key]}")
                key_to_file[key] = path
    return key_to_file


def _rewrite_safetensor_file(path: Path) -> int:
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        metadata = f.metadata()
        collisions = [
            key.replace(BAD_PREFIX, GOOD_PREFIX, 1)
            for key in keys
            if key.startswith(BAD_PREFIX) and key.replace(BAD_PREFIX, GOOD_PREFIX, 1) in keys
        ]
        if collisions:
            raise ValueError(f"{path} already contains target keys: {collisions[:5]}")

        rename_count = sum(key.startswith(BAD_PREFIX) for key in keys)
        if rename_count == 0:
            return 0

        tensors = {}
        for key in keys:
            new_key = key.replace(BAD_PREFIX, GOOD_PREFIX, 1) if key.startswith(BAD_PREFIX) else key
            tensors[new_key] = f.get_tensor(key)

    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    save_file(tensors, tmp_path, metadata=metadata)
    tmp_path.replace(path)
    return rename_count


def _write_index(model_dir: Path, index_path: Path, index: dict) -> None:
    weight_files = {name for name in index.get("weight_map", {}).values()}
    index.setdefault("metadata", {})
    index["metadata"]["total_size"] = sum((model_dir / name).stat().st_size for name in weight_files)

    tmp_index = index_path.with_name(f"{index_path.name}.tmp.{os.getpid()}")
    with tmp_index.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_index.replace(index_path)


def _fill_missing_base_tensors(model_dir: Path, base_model_dir: Path, prefixes: tuple[str, ...]) -> int:
    if not prefixes:
        return 0

    _, target_index = _load_index(model_dir)
    _, base_index = _load_index(base_model_dir)
    target_keys = _collect_safetensor_keys(model_dir, target_index)
    base_keys = _collect_safetensor_keys(base_model_dir, base_index)

    missing_keys = [
        key
        for key in sorted(base_keys)
        if key not in target_keys and any(key.startswith(prefix) for prefix in prefixes)
    ]
    if not missing_keys:
        print(f"No base tensors needed from {base_model_dir}.")
        return 0

    tensors = {}
    keys_by_file: dict[Path, list[str]] = defaultdict(list)
    for key in missing_keys:
        keys_by_file[base_keys[key]].append(key)

    for path, keys in keys_by_file.items():
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)

    supplemental_name = "model-missing-from-base.safetensors"
    supplemental_path = model_dir / supplemental_name
    tmp_path = supplemental_path.with_name(f"{supplemental_path.name}.tmp.{os.getpid()}")
    save_file(tensors, tmp_path)
    tmp_path.replace(supplemental_path)

    index_path, index = _load_index(model_dir)
    if index_path is None or index is None:
        index_path = model_dir / "model.safetensors.index.json"
        index = {
            "metadata": {},
            "weight_map": {key: path.name for key, path in target_keys.items()},
        }

    weight_map = index.setdefault("weight_map", {})
    for key in missing_keys:
        if key in weight_map:
            raise ValueError(f"Refusing to overwrite existing index entry for {key}")
        weight_map[key] = supplemental_name
    _write_index(model_dir, index_path, index)

    print(f"Filled {len(missing_keys)} missing base tensors from {base_model_dir}.")
    return len(missing_keys)


def _sync_sidecars_from_base(model_dir: Path, base_model_dir: Path) -> int:
    copied = 0
    for name in BASE_SIDECAR_FILES:
        source = base_model_dir / name
        if source.exists():
            shutil.copy2(source, model_dir / name)
            copied += 1

    for name in STALE_SIDECAR_FILES:
        stale = model_dir / name
        if stale.exists() and not (base_model_dir / name).exists():
            stale.unlink()

    if copied:
        print(f"Synced {copied} tokenizer/processor sidecar files from {base_model_dir}.")
    return copied


def _validate_qwen35_export(model_dir: Path) -> None:
    _, index = _load_index(model_dir)
    key_to_file = _collect_safetensor_keys(model_dir, index)
    keys = set(key_to_file)

    bad_visual = sorted(key for key in keys if key.startswith(BAD_PREFIX))
    if bad_visual:
        raise ValueError(f"Bad visual prefixes remain in {model_dir}: {bad_visual[:5]}")

    visual_count = sum(key.startswith(GOOD_PREFIX) for key in keys)
    language_count = sum(key.startswith("model.language_model.") for key in keys)
    mtp_count = sum(key.startswith("mtp.") for key in keys)

    if visual_count == 0:
        raise ValueError(f"{model_dir} has Qwen3.5 vision config but no {GOOD_PREFIX} tensors.")
    if language_count == 0:
        raise ValueError(f"{model_dir} has no model.language_model.* tensors.")

    print(
        "Validated Qwen3.5 export: "
        f"{language_count} language, {visual_count} visual, {mtp_count} mtp tensors."
    )


def repair_model_dir(
    model_dir: Path,
    base_model_dir: Path | None = None,
    fill_missing_prefixes: tuple[str, ...] = DEFAULT_FILL_PREFIXES,
    sync_sidecars: bool = True,
) -> int:
    if not _is_qwen35_conditional_generation(model_dir):
        print(f"Skipping {model_dir}: not a Qwen3_5ForConditionalGeneration HF folder.")
        return 0

    index_path, index = _load_index(model_dir)
    files = _safetensor_files(model_dir, index)
    if not files:
        raise FileNotFoundError(f"No safetensors weights found under {model_dir}")

    total_renamed = 0
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Missing safetensors shard referenced by index: {path}")
        total_renamed += _rewrite_safetensor_file(path)

    if index_path is not None and index is not None:
        weight_map = index.get("weight_map", {})
        renamed_map = {}
        for key, shard in weight_map.items():
            new_key = key.replace(BAD_PREFIX, GOOD_PREFIX, 1) if key.startswith(BAD_PREFIX) else key
            if new_key in renamed_map:
                raise ValueError(f"Index key collision while renaming {key} -> {new_key}")
            renamed_map[new_key] = shard
        if renamed_map != weight_map:
            index["weight_map"] = renamed_map
            _write_index(model_dir, index_path, index)

    if total_renamed:
        print(f"Repaired {total_renamed} Qwen3.5 visual weight keys in {model_dir}.")
    else:
        print(f"No Qwen3.5 visual-prefix repair needed for {model_dir}.")

    filled = 0
    if base_model_dir is not None:
        if not base_model_dir.exists():
            raise FileNotFoundError(f"Base model directory does not exist: {base_model_dir}")
        if sync_sidecars:
            _sync_sidecars_from_base(model_dir, base_model_dir)
        filled = _fill_missing_base_tensors(model_dir, base_model_dir, fill_missing_prefixes)

    _validate_qwen35_export(model_dir)
    return total_renamed + filled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--base-model-dir", type=Path, default=None)
    parser.add_argument(
        "--fill-missing-prefix",
        action="append",
        default=None,
        help="Tensor-key prefix to fill from --base-model-dir when missing. Defaults to mtp.",
    )
    parser.add_argument("--no-fill-missing", action="store_true")
    parser.add_argument("--no-sync-sidecars", action="store_true")
    args = parser.parse_args()
    fill_prefixes = () if args.no_fill_missing else tuple(args.fill_missing_prefix or DEFAULT_FILL_PREFIXES)
    repair_model_dir(
        args.model_dir,
        base_model_dir=args.base_model_dir,
        fill_missing_prefixes=fill_prefixes,
        sync_sidecars=not args.no_sync_sidecars,
    )


if __name__ == "__main__":
    main()
