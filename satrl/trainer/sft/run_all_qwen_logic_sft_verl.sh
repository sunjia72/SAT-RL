#!/bin/bash
set -euo pipefail

# Batch launcher for Qwen3 / Qwen3.5 VeRL logic SFT runs.
# This intentionally runs models one after another. Each run uses the sibling
# run_qwen_logic_sft_verl.sh launcher so model-specific defaults stay there.
#
# Examples:
#   bash run_all_qwen_logic_sft_verl.sh
#   DRY_RUN=1 bash run_all_qwen_logic_sft_verl.sh LOGGER='["console"]'
#   bash run_all_qwen_logic_sft_verl.sh --models "qwen3-0.6b qwen3-1.7b"
#   CONTINUE_ON_ERROR=1 bash run_all_qwen_logic_sft_verl.sh trainer.save_freq=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/run_qwen_logic_sft_verl.sh}"

DEFAULT_MODELS=(
  # qwen3-0.6b
  # qwen3-1.7b
  # qwen3-4b
  # qwen3-8b
  qwen3-14b
  # qwen3.5-2b
  # qwen3.5-4b
  # qwen3.5-9b
)

MODEL_LIST="${MODEL_LIST:-}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
LOG_OUTPUT="${LOG_OUTPUT:-1}"
LOG_DIR="${LOG_DIR:-${SCRATCH:-${TMPDIR:-/tmp}/satrl}/logs/verl_sft_logic}"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  run_all_qwen_logic_sft_verl.sh [options] [hydra overrides...]

Runs these models sequentially by default:
  qwen3-0.6b      -> Qwen/Qwen3-0.6B
  qwen3-1.7b      -> Qwen/Qwen3-1.7B
  qwen3-4b        -> Qwen/Qwen3-4B
  qwen3-8b        -> Qwen/Qwen3-8B
  qwen3-14b       -> Qwen/Qwen3-14B
  qwen3.5-2b      -> Qwen/Qwen3.5-2B
  qwen3.5-4b      -> Qwen/Qwen3.5-4B
  qwen3.5-9b      -> Qwen/Qwen3.5-9B

Options:
  --models "LIST"          Space- or comma-separated model aliases to run.
  --continue-on-error      Keep running later models after a failure.
  --fail-fast              Stop at the first failure. This is the default.
  --log-dir DIR            Write per-model logs under DIR.
  --no-log                 Do not tee output to log files.
  -h, --help               Show this help.

Environment overrides:
  MODEL_LIST="LIST"        Same as --models.
  CONTINUE_ON_ERROR=1      Same as --continue-on-error.
  LOG_OUTPUT=0             Same as --no-log.
  BASE_SCRIPT=PATH         Use a different single-model launcher.

All remaining arguments are passed to run_qwen_logic_sft_verl.sh, for example:
  GLOBAL_BSZ=32 MICRO_BSZ=1 LOGGER='["console"]' trainer.save_freq=4

Use DRY_RUN=1 to print every torchrun command without starting training.
EOF
}

if [[ -n "${MODEL_LIST}" ]]; then
  read -r -a MODELS <<<"${MODEL_LIST//,/ }"
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models|--model-list)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a model list." >&2
        exit 2
      fi
      read -r -a MODELS <<<"${2//,/ }"
      shift 2
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    --fail-fast)
      CONTINUE_ON_ERROR=0
      shift
      ;;
    --log-dir)
      if [[ $# -lt 2 ]]; then
        echo "--log-dir requires a directory." >&2
        exit 2
      fi
      LOG_DIR="$2"
      LOG_OUTPUT=1
      shift 2
      ;;
    --no-log|--no-logs)
      LOG_OUTPUT=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    MODEL=*|MODEL_PATH=*)
      echo "Do not pass $1 to the batch launcher; use --models to select models." >&2
      exit 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "No models selected." >&2
  exit 2
fi

if [[ ! -x "${BASE_SCRIPT}" ]]; then
  echo "Single-model launcher is not executable: ${BASE_SCRIPT}" >&2
  exit 2
fi

if [[ "${LOG_OUTPUT}" == "1" ]]; then
  mkdir -p "${LOG_DIR}"
fi

started_at="$(date +%Y%m%d_%H%M%S)"
failed_models=()
total="${#MODELS[@]}"

for idx in "${!MODELS[@]}"; do
  model="${MODELS[$idx]}"
  run_number=$((idx + 1))

  printf '\n==== [%s] Starting %s (%s/%s) ====\n' "$(date -Is)" "${model}" "${run_number}" "${total}"

  if [[ "${LOG_OUTPUT}" == "1" ]]; then
    log_model="${model//\//_}"
    log_file="${LOG_DIR}/${started_at}_${run_number}_${log_model}.log"
    printf 'Log file: %s\n' "${log_file}"

    set +e
    "${BASE_SCRIPT}" --model "${model}" "${EXTRA_ARGS[@]}" 2>&1 | tee "${log_file}"
    status="${PIPESTATUS[0]}"
    set -e
  else
    set +e
    "${BASE_SCRIPT}" --model "${model}" "${EXTRA_ARGS[@]}"
    status="$?"
    set -e
  fi

  if [[ "${status}" -ne 0 ]]; then
    failed_models+=("${model}")
    printf '==== [%s] Failed %s with exit code %s ====\n' "$(date -Is)" "${model}" "${status}" >&2

    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit "${status}"
    fi
  else
    printf '==== [%s] Finished %s ====\n' "$(date -Is)" "${model}"
  fi
done

if [[ ${#failed_models[@]} -gt 0 ]]; then
  printf 'Failed models: %s\n' "${failed_models[*]}" >&2
  exit 1
fi

printf '\nAll selected SFT runs finished successfully.\n'
