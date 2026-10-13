#!/bin/bash
set -euo pipefail
set -x

# All-in-one VeRL SFT launcher for Qwen3 / Qwen3.5 logic tool-call SFT.
# Run inside the Apptainer image. The script cds to the local VeRL checkout.
#
# Examples:
#   bash run_qwen_logic_sft_verl.sh --model qwen3-4b
#   bash run_qwen_logic_sft_verl.sh --model Qwen/Qwen3.5-9B LOGGER='["console"]'
#   MODEL=qwen3.5-2b LOGGER='["console"]' bash run_qwen_logic_sft_verl.sh

nproc_per_node=${nproc_per_node:-4}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SATRL_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_ROOT="$(cd -- "$SATRL_ROOT/.." && pwd)"
PROJECT=${PROJECT:-$ARTIFACT_ROOT}
SCRATCH=${SCRATCH:-${TMPDIR:-/tmp}/satrl}
VERL_DIR=${VERL_DIR:-${ARTIFACT_ROOT}/verl}

TRAIN_DATA=${TRAIN_DATA:-${ARTIFACT_ROOT}/data/all_sft_tool_calls_verl/train.parquet}
EVAL_DATA=${EVAL_DATA:-${ARTIFACT_ROOT}/data/all_sft_tool_calls_verl/val.parquet}

MODEL_ARG=${MODEL:-${MODEL_PATH:-qwen3-4b}}
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  run_qwen_logic_sft_verl.sh --model MODEL [hydra overrides...]

Supported MODEL values:
  qwen3-0.6b      -> Qwen/Qwen3-0.6B
  qwen3-1.7b      -> Qwen/Qwen3-1.7B
  qwen3-4b        -> Qwen/Qwen3-4B
  qwen3-8b        -> Qwen/Qwen3-8B
  qwen3-14b       -> Qwen/Qwen3-14B
  qwen3.5-2b      -> Qwen/Qwen3.5-2B
  qwen3.5-4b      -> Qwen/Qwen3.5-4B
  qwen3.5-9b      -> Qwen/Qwen3.5-9B

You may also pass the full Hugging Face id or a local model path, for example:
  --model Qwen/Qwen3.5-2B
  --model /path/to/model

Useful environment overrides:
  nproc_per_node=4
  GLOBAL_BSZ=32
  MICRO_BSZ=1
  CTX_LEN=16384
  MAX_TOKEN_LEN_PER_GPU=16384
  USE_FUSED_KERNELS=true
  FUSED_KERNEL_BACKEND=torch
  ACTIVATION_OFFLOAD=false
  SAVE_FREQ=4
  LR=1e-5
  LOGGER='["console"]'
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  DRY_RUN=1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|-m)
      MODEL_ARG="$2"
      shift 2
      ;;
    --list-models)
      printf '%s\n' \
        qwen3-0.6b \
        qwen3-1.7b \
        qwen3-4b \
        qwen3-8b \
        qwen3-14b \
        qwen3.5-2b \
        qwen3.5-4b \
        qwen3.5-9b
      exit 0
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    MODEL=*|MODEL_PATH=*)
      MODEL_ARG="${1#*=}"
      shift
      ;;
    nproc_per_node=*|MICRO_BSZ=*|GLOBAL_BSZ=*|CTX_LEN=*|MAX_TOKEN_LEN_PER_GPU=*|SP_SIZE=*|LR=*|LOGGER=*|\
USE_FUSED_KERNELS=*|FUSED_KERNEL_BACKEND=*|ACTIVATION_OFFLOAD=*|SAVE_FREQ=*|PYTORCH_CUDA_ALLOC_CONF=*|\
TRAIN_DATA=*|EVAL_DATA=*|save_path=*|experiment_name=*|ALLOW_QWEN35_SP=*|DRY_RUN=*|NUM_WORKERS=*)
      var_name="${1%%=*}"
      var_value="${1#*=}"
      printf -v "${var_name}" '%s' "${var_value}"
      shift
      ;;
    trainer.save_freq=*)
      SAVE_FREQ="${1#*=}"
      shift
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

raw_model="${MODEL_ARG}"
model_base="${raw_model##*/}"
model_key="${model_base,,}"
model_key="${model_key/_/.}"

DEFAULT_MICRO_BSZ=4
DEFAULT_GLOBAL_BSZ=64
DEFAULT_MAX_TOKEN_LEN_PER_GPU=""
DEFAULT_USE_FUSED_KERNELS=false
DEFAULT_ACTIVATION_OFFLOAD=false

case "${model_key}" in
  qwen3-0.6b|qwen3-0-6b)
    CANONICAL_MODEL="Qwen/Qwen3-0.6B"
    MODEL_FAMILY="qwen3"
    DEFAULT_SP_SIZE=1
    ;;
  qwen3-1.7b|qwen3-1-7b)
    CANONICAL_MODEL="Qwen/Qwen3-1.7B"
    MODEL_FAMILY="qwen3"
    DEFAULT_SP_SIZE=1
    ;;
  qwen3-4b)
    CANONICAL_MODEL="Qwen/Qwen3-4B"
    MODEL_FAMILY="qwen3"
    DEFAULT_SP_SIZE=4
    ;;
  qwen3-8b)
    CANONICAL_MODEL="Qwen/Qwen3-8B"
    MODEL_FAMILY="qwen3"
    DEFAULT_SP_SIZE=4
    DEFAULT_MAX_TOKEN_LEN_PER_GPU=8192
    DEFAULT_USE_FUSED_KERNELS=true
    ;;
  qwen3-14b)
    CANONICAL_MODEL="Qwen/Qwen3-14B"
    MODEL_FAMILY="qwen3"
    DEFAULT_SP_SIZE=4
    DEFAULT_MAX_TOKEN_LEN_PER_GPU=8192
    DEFAULT_USE_FUSED_KERNELS=true
    ;;
  qwen3.5-2b|qwen35-2b|qwen3-5-2b)
    CANONICAL_MODEL="Qwen/Qwen3.5-2B"
    MODEL_FAMILY="qwen35"
    DEFAULT_SP_SIZE=1
    ;;
  qwen3.5-4b|qwen35-4b|qwen3-5-4b)
    CANONICAL_MODEL="Qwen/Qwen3.5-4B"
    MODEL_FAMILY="qwen35"
    DEFAULT_SP_SIZE=1
    DEFAULT_USE_FUSED_KERNELS=true
    ;;
  qwen3.5-9b|qwen35-9b|qwen3-5-9b)
    CANONICAL_MODEL="Qwen/Qwen3.5-9B"
    MODEL_FAMILY="qwen35"
    DEFAULT_SP_SIZE=1
    DEFAULT_MAX_TOKEN_LEN_PER_GPU=12288
    DEFAULT_USE_FUSED_KERNELS=true
    ;;
  *)
    echo "Unsupported model '${MODEL_ARG}'." >&2
    usage >&2
    exit 2
    ;;
esac

if [[ "${raw_model}" == */* ]]; then
  MODEL_PATH="${raw_model}"
else
  MODEL_PATH="${CANONICAL_MODEL}"
fi

MODEL_NAME="${CANONICAL_MODEL##*/}"
save_path=${save_path:-${SCRATCH}/checkpoints/verl_sft_logic/${MODEL_NAME}_logic_tool_sft}
experiment_name=${experiment_name:-${MODEL_NAME}_logic_tool_sft}

MICRO_BSZ=${MICRO_BSZ:-${DEFAULT_MICRO_BSZ}}
GLOBAL_BSZ=${GLOBAL_BSZ:-${DEFAULT_GLOBAL_BSZ}}
CTX_LEN=${CTX_LEN:-16384}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-${DEFAULT_MAX_TOKEN_LEN_PER_GPU:-${CTX_LEN}}}
SP_SIZE=${SP_SIZE:-${DEFAULT_SP_SIZE}}
LR=${LR:-1e-5}
LOGGER=${LOGGER:-'["console"]'}
NUM_WORKERS=${NUM_WORKERS:-8}
ALLOW_QWEN35_SP=${ALLOW_QWEN35_SP:-0}
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-${DEFAULT_USE_FUSED_KERNELS}}
FUSED_KERNEL_BACKEND=${FUSED_KERNEL_BACKEND:-torch}
ACTIVATION_OFFLOAD=${ACTIVATION_OFFLOAD:-${DEFAULT_ACTIVATION_OFFLOAD}}
SAVE_FREQ=${SAVE_FREQ:-10}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

case "${USE_FUSED_KERNELS,,}" in
  1|true|yes|on)
    USE_FUSED_KERNELS=true
    ;;
  0|false|no|off)
    USE_FUSED_KERNELS=false
    ;;
esac

case "${ACTIVATION_OFFLOAD,,}" in
  1|true|yes|on)
    ACTIVATION_OFFLOAD=true
    ;;
  0|false|no|off)
    ACTIVATION_OFFLOAD=false
    ;;
esac

export PYTORCH_CUDA_ALLOC_CONF

if [[ "${MODEL_FAMILY}" == "qwen35" && "${SP_SIZE}" != "1" && "${ALLOW_QWEN35_SP}" != "1" ]]; then
  echo "Qwen3.5 SFT requires SP_SIZE=1 with the current VeRL Ulysses path; overriding SP_SIZE=${SP_SIZE} to 1." >&2
  SP_SIZE=1
fi

cd "${VERL_DIR}"
export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"

CMD=(
  torchrun
  --standalone
  --nnodes=1
  --nproc_per_node="${nproc_per_node}"
  -m verl.trainer.sft_trainer
  "data.train_files=${TRAIN_DATA}"
  "data.val_files=${EVAL_DATA}"
  data.messages_key=messages
  data.tools_key=tools
  "data.micro_batch_size_per_gpu=${MICRO_BSZ}"
  "data.train_batch_size=${GLOBAL_BSZ}"
  "data.max_length=${CTX_LEN}"
  "data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
  "data.num_workers=${NUM_WORKERS}"
  data.pad_mode=no_padding
  data.ignore_input_ids_mismatch=True
  "optim.lr=${LR}"
  engine=fsdp
  engine.model_dtype=bf16
  engine.dtype=bfloat16
  "engine.ulysses_sequence_parallel_size=${SP_SIZE}"
  "model.path=${MODEL_PATH}"
  model.use_remove_padding=true
  "model.use_fused_kernels=${USE_FUSED_KERNELS}"
  "model.fused_kernel_options.impl_backend=${FUSED_KERNEL_BACKEND}"
  "model.enable_activation_offload=${ACTIVATION_OFFLOAD}"
  trainer.project_name=verl_sft_logic
  "trainer.experiment_name=${experiment_name}"
  "trainer.default_local_dir=${save_path}"
  trainer.default_hdfs_dir=null
  trainer.total_epochs=1
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.logger=${LOGGER}"
  "${EXTRA_ARGS[@]}"
)

printf 'Resolved model: %s\n' "${MODEL_PATH}"
printf 'Experiment: %s\n' "${experiment_name}"
printf 'SP_SIZE: %s\n' "${SP_SIZE}"
printf 'MAX_TOKEN_LEN_PER_GPU: %s\n' "${MAX_TOKEN_LEN_PER_GPU}"
printf 'Fused kernels: %s (%s backend)\n' "${USE_FUSED_KERNELS}" "${FUSED_KERNEL_BACKEND}"
printf 'Activation offload: %s\n' "${ACTIVATION_OFFLOAD}"
printf 'Checkpoint save frequency: %s\n' "${SAVE_FREQ}"
printf 'PYTORCH_CUDA_ALLOC_CONF: %s\n' "${PYTORCH_CUDA_ALLOC_CONF}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Dry run command:\n'
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

exec "${CMD[@]}"
