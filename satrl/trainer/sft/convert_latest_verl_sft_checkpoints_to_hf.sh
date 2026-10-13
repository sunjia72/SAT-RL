#!/bin/bash
set -euo pipefail

# Convert the latest FSDP checkpoint under each verl SFT experiment into a
# Hugging Face model folder using verl.model_merger.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SATRL_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_ROOT="$(cd -- "$SATRL_ROOT/.." && pwd)"
PROJECT=${PROJECT:-$ARTIFACT_ROOT}
SCRATCH=${SCRATCH:-${TMPDIR:-/tmp}/satrl}
VERL_DIR=${VERL_DIR:-${ARTIFACT_ROOT}/verl}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${SCRATCH}/checkpoints/verl_sft_logic}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ARTIFACT_ROOT}/models}
TARGET_DIR=${TARGET_DIR:-}
BASE_MODEL_ROOT=${BASE_MODEL_ROOT:-${ARTIFACT_ROOT}/models}
BASE_MODEL_DIR=${BASE_MODEL_DIR:-}
APPTAINER_IMAGE=${APPTAINER_IMAGE:-${ARTIFACT_ROOT}/verl.sif}
USE_APPTAINER=${USE_APPTAINER:-1}
DRY_RUN=0
FORCE=0
SKIP_EXISTING=1
MODEL_FILTERS=()
MERGER_ARGS=()
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'EOF'
Usage:
  convert_latest_verl_sft_checkpoints_to_hf.sh [options] [-- extra model_merger args...]

Converts the highest-numbered global_step_* checkpoint for each experiment under
the configured checkpoint root into a Hugging Face folder under the configured
output root.

Options:
  --checkpoint-root DIR    Source verl checkpoint root.
  --output-root DIR        Destination root for converted HF folders.
  --target-dir DIR         Exact destination folder. Requires exactly one matched experiment.
  --models "LIST"         Optional space- or comma-separated experiment/model filters.
  --base-model-root DIR    Root for base HF folders used to restore Qwen3.5 sidecars/MTP.
  --base-model-dir DIR     Exact base HF folder for post-export Qwen3.5 repair.
  --force                 Rebuild output folders even when they already exist.
  --no-skip-existing      Fail instead of skipping existing output folders.
  --dry-run               Print conversions without running model_merger.
  --no-apptainer          Run directly in the current environment.
  -h, --help              Show this help.

Environment overrides:
  PROJECT                 Default: the artifact root
  SCRATCH                 Default: $TMPDIR/satrl (or /tmp/satrl)
  VERL_DIR                Default: $ARTIFACT_ROOT/verl
  CHECKPOINT_ROOT         Default: $SCRATCH/checkpoints/verl_sft_logic
  OUTPUT_ROOT             Default: $ARTIFACT_ROOT/models
  TARGET_DIR              Optional exact output folder.
  BASE_MODEL_ROOT         Default: $ARTIFACT_ROOT/models
  BASE_MODEL_DIR          Optional exact base model folder.
  APPTAINER_IMAGE         Default: $ARTIFACT_ROOT/verl.sif
  USE_APPTAINER=0         Same as --no-apptainer.

Examples:
  bash convert_latest_verl_sft_checkpoints_to_hf.sh --dry-run
  bash convert_latest_verl_sft_checkpoints_to_hf.sh --models "Qwen3-8B Qwen3-14B"
  bash convert_latest_verl_sft_checkpoints_to_hf.sh --force
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-root)
      CHECKPOINT_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --models|--model-list)
      read -r -a parsed_models <<<"${2//,/ }"
      MODEL_FILTERS+=("${parsed_models[@]}")
      shift 2
      ;;
    --base-model-root)
      BASE_MODEL_ROOT="$2"
      shift 2
      ;;
    --base-model-dir)
      BASE_MODEL_DIR="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      SKIP_EXISTING=0
      shift
      ;;
    --no-skip-existing)
      SKIP_EXISTING=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-apptainer)
      USE_APPTAINER=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      MERGER_ARGS+=("$@")
      break
      ;;
    *)
      MODEL_FILTERS+=("$1")
      shift
      ;;
  esac
done

if [[ "${USE_APPTAINER}" == "1" && -z "${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}${_VERL_SFT_CONVERT_IN_CONTAINER:-}" ]]; then
  if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
    echo "Apptainer image not found: ${APPTAINER_IMAGE}" >&2
    echo "Run inside the container with --no-apptainer, or set APPTAINER_IMAGE." >&2
    exit 2
  fi

  exec apptainer exec --nv \
    --bind "${HOME}:${HOME}" \
    --bind "${SCRATCH}:${SCRATCH}" \
    --bind "${PROJECT}:${PROJECT}" \
    --bind /tmp:/tmp \
    "${APPTAINER_IMAGE}" \
    env _VERL_SFT_CONVERT_IN_CONTAINER=1 \
    bash "$0" "${ORIGINAL_ARGS[@]}"
fi

normalize_name() {
  local name="$1"
  name="${name##*/}"
  name="${name%_logic_tool_sft}"
  name="${name%-logic-tool-sft}"
  printf '%s\n' "${name}" | tr '[:upper:]_' '[:lower:]-'
}

matches_filters() {
  local exp_name="$1"
  local exp_norm
  exp_norm="$(normalize_name "${exp_name}")"

  if [[ ${#MODEL_FILTERS[@]} -eq 0 ]]; then
    return 0
  fi

  local filter filter_norm
  for filter in "${MODEL_FILTERS[@]}"; do
    filter_norm="$(normalize_name "${filter}")"
    if [[ "${exp_norm}" == "${filter_norm}" || "${exp_norm}" == "${filter_norm}"-* ]]; then
      return 0
    fi
  done

  return 1
}

base_model_name_for_exp() {
  local name="$1"
  name="${name##*/}"
  name="${name%_logic_tool_sft}"
  name="${name%-logic-tool-sft}"
  name="${name%_sft}"
  name="${name%-sft}"
  printf '%s\n' "${name}"
}

base_model_dir_for_exp() {
  local exp_name="$1"
  if [[ -n "${BASE_MODEL_DIR}" ]]; then
    printf '%s\n' "${BASE_MODEL_DIR}"
    return 0
  fi

  local base_name candidate
  base_name="$(base_model_name_for_exp "${exp_name}")"
  candidate="${BASE_MODEL_ROOT}/${base_name}"
  if [[ -d "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
  fi
}

find_latest_checkpoint() {
  local exp_dir="$1"
  local latest_dir=""
  local latest_step=-1
  local step_dir base step

  while IFS= read -r -d '' step_dir; do
    base="${step_dir##*/}"
    step="${base#global_step_}"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    [[ -f "${step_dir}/fsdp_config.json" ]] || continue
    compgen -G "${step_dir}/model_world_size_*_rank_0.pt" >/dev/null || continue
    [[ -f "${step_dir}/huggingface/config.json" ]] || continue

    if (( step > latest_step )); then
      latest_step="${step}"
      latest_dir="${step_dir}"
    fi
  done < <(find "${exp_dir}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -print0)

  printf '%s\n' "${latest_dir}"
}

target_has_contents() {
  local target="$1"
  [[ -d "${target}" ]] || return 1
  [[ -n "$(find "${target}" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

TMP_TARGETS=()
cleanup_tmp_targets() {
  local target
  for target in "${TMP_TARGETS[@]}"; do
    if [[ -d "${target}" ]]; then
      rm -rf "${target}"
    fi
  done
}
trap cleanup_tmp_targets EXIT

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "Checkpoint root does not exist: ${CHECKPOINT_ROOT}" >&2
  exit 2
fi

if [[ ! -d "${VERL_DIR}" ]]; then
  echo "VeRL directory does not exist: ${VERL_DIR}" >&2
  exit 2
fi

if [[ -n "${TARGET_DIR}" ]]; then
  mkdir -p "$(dirname "${TARGET_DIR}")"
else
  mkdir -p "${OUTPUT_ROOT}"
fi

MATCHED_EXPS=()
while IFS= read -r -d '' exp_dir; do
  exp_name="${exp_dir##*/}"
  matches_filters "${exp_name}" || continue
  MATCHED_EXPS+=("${exp_dir}")
done < <(find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

if [[ "${#MATCHED_EXPS[@]}" -eq 0 ]]; then
  echo "No checkpoint experiments matched under ${CHECKPOINT_ROOT}." >&2
  exit 1
fi

if [[ -n "${TARGET_DIR}" && "${#MATCHED_EXPS[@]}" -ne 1 ]]; then
  echo "--target-dir requires exactly one matched experiment; matched ${#MATCHED_EXPS[@]}." >&2
  exit 2
fi

converted=0
skipped=0
failed=0

for exp_dir in "${MATCHED_EXPS[@]}"; do
  exp_name="${exp_dir##*/}"

  latest_ckpt="$(find_latest_checkpoint "${exp_dir}")"
  if [[ -z "${latest_ckpt}" ]]; then
    echo "Skipping ${exp_name}: no valid FSDP global_step_* checkpoint found." >&2
    skipped=$((skipped + 1))
    continue
  fi

  target_dir="${TARGET_DIR:-${OUTPUT_ROOT}/${exp_name}}"
  if target_has_contents "${target_dir}" && [[ "${FORCE}" != "1" ]]; then
    if [[ "${SKIP_EXISTING}" == "1" ]]; then
      echo "Skipping ${exp_name}: output exists at ${target_dir}. Use --force to rebuild."
      skipped=$((skipped + 1))
      continue
    fi
    echo "Output exists: ${target_dir}. Use --force or omit --no-skip-existing." >&2
    failed=$((failed + 1))
    continue
  fi

  mkdir -p "$(dirname "${target_dir}")"
  tmp_target="${target_dir}.tmp.$$"
  cmd=(
    python -m verl.model_merger merge
    --backend fsdp
    --local_dir "${latest_ckpt}"
    --target_dir "${tmp_target}"
    "${MERGER_ARGS[@]}"
  )

  echo "Converting ${exp_name}"
  echo "  source: ${latest_ckpt}"
  echo "  target: ${target_dir}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    converted=$((converted + 1))
    continue
  fi

  rm -rf "${tmp_target}"
  TMP_TARGETS+=("${tmp_target}")

  (
    cd "${VERL_DIR}"
    export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"
    "${cmd[@]}"
  )

  base_model_dir="$(base_model_dir_for_exp "${exp_name}")"
  repair_cmd=(python "${SCRIPT_DIR}/repair_qwen35_hf_visual_prefixes.py" "${tmp_target}")
  if [[ -n "${base_model_dir}" ]]; then
    repair_cmd+=(--base-model-dir "${base_model_dir}")
  fi
  "${repair_cmd[@]}"

  if [[ "${FORCE}" == "1" ]]; then
    rm -rf "${target_dir}"
  fi
  mv "${tmp_target}" "${target_dir}"
  printf '%s\n' "${latest_ckpt}" >"${target_dir}/source_checkpoint.txt"
  converted=$((converted + 1))
done

echo "Done. converted=${converted} skipped=${skipped} failed=${failed}"
if [[ "${failed}" -gt 0 ]]; then
  exit 1
fi
