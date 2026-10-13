#!/bin/bash
# Evaluate a trained SAT-RL model with the same runtime profile as GRPO training.
set -x
ulimit -n 65535

# === ARGUMENT PARSING ===
# Defaults for the Enigmata symbolic evaluation.
MODE="tools_symbolic"
DATASET_NAME="enigmata_final"
MODEL_PATH="${MODEL_PATH:-}"              # required; override with --model
nnodes=""                              # auto-detect from the job unless overridden with --nnodes
TOOL_FORMAT=""                         # auto-detect unless overridden
FUSED_KERNEL_BACKEND="torch"           # Qwen3.5-9B Triton LCE currently hits CUDA illegal access
RUN_SUFFIX=""
# The submission worker passes 8 explicitly. A one-response fallback preserves
# the behavior of evaluation jobs that were queued before --responses existed.
RESPONSES=1

need_value() {
  local option="$1"
  local count="$2"
  if (( count < 2 )); then
    echo "Missing value for $option" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      need_value "$1" "$#"
      MODE="$2"
      shift 2
      ;;
    --dataset)
      need_value "$1" "$#"
      DATASET_NAME="$2"
      shift 2
      ;;
    --model)
      need_value "$1" "$#"
      MODEL_PATH="$2"
      shift 2
      ;;
    --nnodes)
      need_value "$1" "$#"
      nnodes="$2"
      shift 2
      ;;
    --responses|--val-n)
      need_value "$1" "$#"
      RESPONSES="$2"
      shift 2
      ;;
    --tool-format)
      need_value "$1" "$#"
      TOOL_FORMAT="$2"
      shift 2
      ;;
    --fused-backend)
      need_value "$1" "$#"
      FUSED_KERNEL_BACKEND="$2"
      shift 2
      ;;
    --run-suffix|--job-suffix)
      need_value "$1" "$#"
      RUN_SUFFIX="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--mode MODE] [--dataset NAME] [--model MODEL_PATH] [--nnodes N] [--responses N] [--tool-format FORMAT] [--fused-backend BACKEND] [--run-suffix SUFFIX]"
      echo "Example: $0 --mode tools_symbolic --dataset enigmata_final --model /path/to/model --nnodes 4 --responses 8"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Use --help for usage information."
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "A model path is required; pass --model PATH or set MODEL_PATH." >&2
  exit 2
fi

# === CONFIGURATION ===
ngpu_per_node="${GPUS_PER_NODE:-4}"

if ! [[ "$ngpu_per_node" =~ ^[0-9]+$ ]] || (( ngpu_per_node <= 0 )); then
  echo "Invalid GPUs per node: $ngpu_per_node" >&2
  exit 1
fi

detect_main_node() {
  if [[ -n "${HEAD_NODE:-}" ]]; then
    echo "$HEAD_NODE"
  elif [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1
  elif [[ -n "${SLURM_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol show hostnames "$SLURM_NODELIST" | head -n1
  else
    hostname -s
  fi
}

detect_nnodes() {
  if [[ -n "${SLURM_NNODES:-}" ]]; then
    echo "$SLURM_NNODES"
  elif [[ -n "${SLURM_JOB_NUM_NODES:-}" ]]; then
    echo "$SLURM_JOB_NUM_NODES"
  elif [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'END { print NR }'
  elif [[ -n "${SLURM_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol show hostnames "$SLURM_NODELIST" | awk 'END { print NR }'
  else
    echo 1
  fi
}

MAIN_NODE="$(detect_main_node)"

if [[ -z "$nnodes" ]]; then
  nnodes="$(detect_nnodes)"
fi

if ! [[ "$nnodes" =~ ^[0-9]+$ ]] || (( nnodes <= 0 )); then
  echo "Invalid node count: $nnodes" >&2
  exit 1
fi
if ! [[ "$RESPONSES" =~ ^[0-9]+$ ]] || (( RESPONSES <= 0 )); then
  echo "Invalid response count: $RESPONSES" >&2
  exit 1
fi

if [[ -n "$RUN_SUFFIX" ]]; then
  if ! [[ "$RUN_SUFFIX" =~ ^[_-]?[A-Za-z0-9_.-]+$ ]]; then
    echo "Invalid run suffix: $RUN_SUFFIX" >&2
    exit 1
  fi
  if [[ "$RUN_SUFFIX" != [_-]* ]]; then
    RUN_SUFFIX="_$RUN_SUFFIX"
  fi
fi

reward_nnodes=0
trainer_nnodes=$(( nnodes ))

# Optional bookkeeping.
total_gpus_cluster=$(( nnodes * ngpu_per_node ))
total_gpus_reward=$(( reward_nnodes * ngpu_per_node ))
total_gpus_train=$(( trainer_nnodes * ngpu_per_node ))

export WANDB_MODE="${WANDB_MODE:-online}"

echo "=== Resolved: MODE=$MODE | DATASET_NAME=$DATASET_NAME | MODEL_PATH=$MODEL_PATH | TOTAL_NNODES=$nnodes ==="
echo "=== Split: reward_model.nnodes=$reward_nnodes (GPUs=$total_gpus_reward) | trainer.nnodes=$trainer_nnodes (GPUs=$total_gpus_train) ==="
echo "=== Main node: $MAIN_NODE | WANDB_MODE=${WANDB_MODE:-default} ==="
echo "=== Fused kernel backend: $FUSED_KERNEL_BACKEND ==="

mask_proxy_url() {
  local value="${1:-}"
  if [[ "$value" == *"://"*"@"* ]]; then
    sed -E 's#(://)[^/@]+@#\1***@#' <<< "$value"
  else
    printf '%s\n' "$value"
  fi
}

export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-120}"
WANDB_PROXY_VALUE="${WANDB_PROXY:-${HTTPS_PROXY:-${https_proxy:-}}}"
echo "=== W&B proxy: HTTPS_PROXY=$(mask_proxy_url "$WANDB_PROXY_VALUE") | WANDB_INIT_TIMEOUT=$WANDB_INIT_TIMEOUT ==="

if [[ -z "$TOOL_FORMAT" ]]; then
  case "$MODEL_PATH" in
    *Qwen3.5*|*Qwen3-Coder*|*qwen3.5*|*qwen3-coder*)
      TOOL_FORMAT="qwen3_coder"
      ;;
    *)
      TOOL_FORMAT="hermes"
      ;;
  esac
fi
echo "=== Tool parser format: $TOOL_FORMAT ==="

# === PATHS ===
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SATRL_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_ROOT="$(cd -- "$SATRL_ROOT/.." && pwd)"
PROJECT="${PROJECT:-$ARTIFACT_ROOT}"
SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}/satrl}"
PROJECT_DIR="${VERL_DIR:-$ARTIFACT_ROOT/verl}"
CONFIG_PATH="${CONFIG_PATH:-$SATRL_ROOT/trainer/config}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_NAME="$DATASET_NAME-$MODE-DAPO"
EXP_NAME="$DATASET_NAME-$MODE-DAPO-${MODEL_PATH##*/}${RUN_SUFFIX}"
ROLLOUT_DIR="${ROLLOUT_ROOT:-$ARTIFACT_ROOT/output}/rollout_history/$EXP_NAME/$TIMESTAMP"
VALIDATION_DIR="${VALIDATION_ROOT:-$ARTIFACT_ROOT/output}/validation_history/$EXP_NAME/$TIMESTAMP"
TRAIN_DIR="${DATA_ROOT:-$ARTIFACT_ROOT/data}/$DATASET_NAME/$MODE/train.parquet"
VAL_DIR="${DATA_ROOT:-$ARTIFACT_ROOT/data}/$DATASET_NAME/$MODE/test.parquet"

# === EPOCHS ===
TOTAL_EPOCHS=2

export MLFLOW_TRACKING_URI="sqlite:////tmp/mlruns.db"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

SF_CONFIG_PATH="${SF_CONFIG_PATH:-$ARTIFACT_ROOT/output/params/sf_${SLURM_JOB_ID:-manual}.yaml}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing model directory: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_DIR" ]]; then
  echo "Missing training parquet required by the val-only data module: $TRAIN_DIR" >&2
  exit 1
fi
if [[ ! -f "$VAL_DIR" ]]; then
  echo "Missing evaluation parquet: $VAL_DIR" >&2
  exit 1
fi
if [[ "$MODE" != "no_tools" && ! -s "$SF_CONFIG_PATH" ]]; then
  echo "Missing SandboxFusion tool configuration: $SF_CONFIG_PATH" >&2
  echo "Start SandboxFusion before running symbolic evaluation." >&2
  exit 1
fi

# === HYPERPARAM TEMPLATE ===
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.000

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=4096
max_response_length=8192
actor_lr=1e-6

train_batch_size=128
val_batch_size=128
rollout_n=8
validation_n="$RESPONSES"
validation_do_sample=False
validation_temperature=0
if (( validation_n > 1 )); then
  validation_do_sample=True
  validation_temperature=1.0
fi
ppo_mini_batch_size=128

# Match the current training memory/throughput profile.
ppo_micro_batch_size_per_gpu=4
actor_ppo_max_token_len_per_gpu=16384
infer_log_prob_max_token_len_per_gpu=16384
rollout_log_prob_micro_batch_size_per_gpu=4
ref_log_prob_micro_batch_size_per_gpu=4
USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-True}"
if [[ "${MODEL_PATH,,}" =~ (^|[^[:alnum:]])(2b|4b)([^[:alnum:]]|$) ]]; then
  rollout_tensor_model_parallel_size=1
else
  rollout_tensor_model_parallel_size=2
fi
rollout_gpu_memory_utilization=0.6
rollout_max_model_len=$(( max_prompt_length + max_response_length ))
rollout_max_num_batched_tokens=16384
rollout_max_num_seqs=256
rollout_agent_num_workers=32
reward_num_workers="${REWARD_NUM_WORKERS:-16}"

echo "=== Rollout tensor model parallel size: $rollout_tensor_model_parallel_size ==="
echo "=== Reward workers: $reward_num_workers ==="
echo "=== Validation sampling: n=$validation_n | do_sample=$validation_do_sample | temperature=$validation_temperature ==="

dp_size=$(( trainer_nnodes * ngpu_per_node ))
if (( (train_batch_size * rollout_n) % dp_size != 0 )); then
  echo "Invalid batch config: train_batch_size * rollout_n must be divisible by total training GPUs ($dp_size)." >&2
  exit 1
fi
if (( (ppo_mini_batch_size * rollout_n) % dp_size != 0 )); then
  echo "Invalid batch config: ppo_mini_batch_size * rollout_n must be divisible by total training GPUs ($dp_size)." >&2
  exit 1
fi

local_train_batch=$(( train_batch_size * rollout_n / dp_size ))
local_ppo_mini_batch=$(( ppo_mini_batch_size * rollout_n / dp_size ))
if (( local_train_batch % local_ppo_mini_batch != 0 )); then
  echo "Invalid batch config: local rollout batch ($local_train_batch) must be divisible by local PPO mini-batch ($local_ppo_mini_batch)." >&2
  echo "Use ppo_mini_batch_size <= train_batch_size and make train_batch_size divisible by ppo_mini_batch_size." >&2
  exit 1
fi
if (( local_ppo_mini_batch % ppo_micro_batch_size_per_gpu != 0 )); then
  echo "Invalid batch config: local PPO mini-batch ($local_ppo_mini_batch) must be divisible by ppo_micro_batch_size_per_gpu ($ppo_micro_batch_size_per_gpu)." >&2
  exit 1
fi

# Optional tool-config args (only when MODE != "no_tools").
EXTRA_TOOL_CFG=()
if [[ "$MODE" != "no_tools" ]]; then
  EXTRA_TOOL_CFG+=( "actor_rollout_ref.rollout.multi_turn.tool_config_path=$SF_CONFIG_PATH" )
  EXTRA_TOOL_CFG+=( "actor_rollout_ref.rollout.multi_turn.format=$TOOL_FORMAT" )
fi

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# grpo_sf.yaml uses a cwd-relative Hydra search path, so run from the verl root.
cd "$PROJECT_DIR" || exit 1

# === RUN VALIDATION ONLY ===
python3 -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name='grpo_sf' \
  algorithm.adv_estimator=grpo \
  +ray_kwargs.ray_init.address=auto \
  data.train_files="$TRAIN_DIR" \
  data.val_files="$VAL_DIR" \
  data.train_batch_size=$train_batch_size \
  data.val_batch_size=$val_batch_size \
  data.max_prompt_length=$max_prompt_length \
  data.max_response_length=$max_response_length \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.shuffle=True \
  data.return_raw_chat=True \
  actor_rollout_ref.rollout.n=$rollout_n \
  actor_rollout_ref.rollout.val_kwargs.n=$validation_n \
  actor_rollout_ref.rollout.val_kwargs.do_sample=$validation_do_sample \
  actor_rollout_ref.rollout.val_kwargs.temperature=$validation_temperature \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$rollout_log_prob_micro_batch_size_per_gpu \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$infer_log_prob_max_token_len_per_gpu \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ref_log_prob_micro_batch_size_per_gpu \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$infer_log_prob_max_token_len_per_gpu \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_shm=False \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_ppo_max_token_len_per_gpu \
  actor_rollout_ref.actor.freeze_vision_tower=True \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=$FUSED_KERNEL_BACKEND \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  algorithm.use_kl_in_reward=$use_kl_in_reward \
  algorithm.kl_ctrl.kl_coef=$kl_coef \
  actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
  actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
  actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
  actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.optim.lr=$actor_lr \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tensor_model_parallel_size \
  actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_memory_utilization \
  actor_rollout_ref.rollout.max_model_len=$rollout_max_model_len \
  actor_rollout_ref.rollout.max_num_batched_tokens=$rollout_max_num_batched_tokens \
  actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs \
  actor_rollout_ref.rollout.agent.num_workers=$rollout_agent_num_workers \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  +actor_rollout_ref.rollout.enable_sleep_mode=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=False \
  actor_rollout_ref.rollout.trace.token2text=False \
  reward.num_workers=$reward_num_workers \
  custom_reward_function.path="${REWARD_ROOT:-$ARTIFACT_ROOT/verl/verl/utils/reward_score/satrl}/$MODE.py" \
  custom_reward_function.name=compute_score \
  trainer.rollout_data_dir="$ROLLOUT_DIR" \
  trainer.validation_data_dir="$VALIDATION_DIR" \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="$PROJECT_NAME" \
  +trainer.remove_previous_ckpt_in_save=True \
  trainer.del_local_ckpt_after_load=True \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node=$ngpu_per_node \
  trainer.nnodes=$trainer_nnodes \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.save_freq=2 \
  trainer.test_freq=1 \
  trainer.critic_warmup=0 \
  trainer.val_only=True \
  +engine_kwargs.sglang.attention_backend=flashinfer \
  trainer.default_local_dir="$SCRATCH/checkpoints/$EXP_NAME" \
  actor_rollout_ref.actor.checkpoint.save_contents="['model','extra','optimizer']" \
  "${EXTRA_TOOL_CFG[@]}" \
  "$@"
