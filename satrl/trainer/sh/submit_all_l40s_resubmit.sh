#!/bin/bash
#
# submit_all.sh - submit all dataset × mode × model permutations as sbatch jobs
#
set -euo pipefail

# ---------- Unified cluster params ----------
NNODES="${NNODES:-2}"          # desired training nnodes (effective nnodes for VERL training)
JOBS_PER_SETUP="${JOBS_PER_SETUP:-3}"
GPU="${GPU:-l40s}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"
MEM="${MEM:-400G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
PARTITION="${PARTITION:-}"
DEFAULT_EXCLUDE_NODES="rack12-13"
EXCLUDE="${EXCLUDE:-$DEFAULT_EXCLUDE_NODES}"
MODELS_OVERRIDE="${MODELS_OVERRIDE:-}"

normalize_proxy_env() {
  local https_proxy_value="${HTTPS_PROXY:-${https_proxy:-}}"
  local http_proxy_value="${HTTP_PROXY:-${http_proxy:-$https_proxy_value}}"
  local all_proxy_value="${ALL_PROXY:-${all_proxy:-}}"
  local no_proxy_value="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1}}"

  if [[ -n "$https_proxy_value" ]]; then
    export HTTPS_PROXY="$https_proxy_value"
    export https_proxy="$https_proxy_value"
  fi
  if [[ -n "$http_proxy_value" ]]; then
    export HTTP_PROXY="$http_proxy_value"
    export http_proxy="$http_proxy_value"
  fi
  if [[ -n "$all_proxy_value" ]]; then
    export ALL_PROXY="$all_proxy_value"
    export all_proxy="$all_proxy_value"
  fi
  export NO_PROXY="$no_proxy_value"
  export no_proxy="$no_proxy_value"
  export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-120}"
}

mask_proxy_url() {
  local value="${1:-}"
  if [[ "$value" == *"://"*"@"* ]]; then
    sed -E 's#(://)[^/@]+@#\1***@#' <<< "$value"
  else
    printf '%s\n' "$value"
  fi
}

normalize_proxy_env

normalize_partition() {
  case "${1:-}" in
    ""|"(null)"|"null"|"none"|"None") echo "" ;;
    *) echo "$1" ;;
  esac
}

is_proxy_required_cluster() {
  local cluster="${SLURM_CLUSTER_NAME,,}"
  local host="${HOST_FQDN,,}"
  [[ "$cluster" == *vulcan* || "$cluster" == *rack* || "$host" == *vulcan* || "$host" == *rack* ]]
}

HOST_FQDN="$(hostname -f 2>/dev/null || hostname)"
SLURM_CLUSTER_NAME="$(
  scontrol show config 2>/dev/null \
    | awk -F= '/^[[:space:]]*ClusterName[[:space:]]*=/ {gsub(/[[:space:]]/, "", $2); print tolower($2); exit}' \
    || true
)"
PARTITION="$(normalize_partition "$PARTITION")"
if is_proxy_required_cluster; then
  TIME_LIMIT="8:00:00"
  PARTITION="gpubase_interac"
  export SATRL_START_PROXY="${SATRL_START_PROXY:-1}"
else
  if [[ ",${EXCLUDE}," != *",kn005,"* ]]; then
    EXCLUDE="${EXCLUDE:+${EXCLUDE},}kn005"
  fi
  export SATRL_START_PROXY="${SATRL_START_PROXY:-0}"
fi

# ---------- Parameter spaces ----------
DATASETS=('satbench')
MODES=('tools_symbolic_no_restriction')

# 


# Resolve a default project path at submission time
DEFAULT_PROJECT="${PROJECT:?Set PROJECT to the project root used for the Slurm run}"
DEFAULT_SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}/satrl}"
SATRL_JOB_WORK_ROOT="${SATRL_JOB_WORK_ROOT:-${DEFAULT_SCRATCH}/satrl_jobs}"

# Models to sweep (absolute paths or simple names)
MODELS=(
#  "${DEFAULT_PROJECT}/models/Qwen3.5-2B-SFT"
  # "${DEFAULT_PROJECT}/models/Qwen3.5-4B-SFT"
  # "${DEFAULT_PROJECT}/models/Qwen3-8B-SFT"
  # "${DEFAULT_PROJECT}/models/Qwen3.5-9B"

  "${DEFAULT_PROJECT}/models/Qwen3.5-4B"
  # "${DEFAULT_PROJECT}/models/Qwen3.5-4B-SFT"

  # "${DEFAULT_PROJECT}/models/Qwen3.5-9B-SFT"

)
if [[ -n "$MODELS_OVERRIDE" ]]; then
  read -r -a MODELS <<< "$MODELS_OVERRIDE"
fi

TRAIN_JOB_SH="${DEFAULT_PROJECT}/logic/satrl/trainer/sh/train_job.sh"

if ! [[ "$NNODES" =~ ^[0-9]+$ ]] || (( NNODES <= 0 )); then
  echo "Invalid NNODES: $NNODES" >&2
  exit 1
fi

if ! [[ "$JOBS_PER_SETUP" =~ ^[0-9]+$ ]] || (( JOBS_PER_SETUP <= 0 )); then
  echo "Invalid JOBS_PER_SETUP: $JOBS_PER_SETUP" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_JOB_SH" ]]; then
  echo "Missing train job script: $TRAIN_JOB_SH" >&2
  exit 1
fi

mkdir -p "${DEFAULT_PROJECT}/slurm" "${SATRL_JOB_WORK_ROOT}"

export PROJECT="$DEFAULT_PROJECT"
export SBATCH_TIME="$TIME_LIMIT"
export SBATCH_CPUS_PER_TASK="$CPUS_PER_TASK"
export SBATCH_MEM="$MEM"

echo "Cluster detection: SLURM_CLUSTER_NAME=${SLURM_CLUSTER_NAME:-unknown} | HOST_FQDN=$HOST_FQDN | SATRL_START_PROXY=${SATRL_START_PROXY:-auto}"
echo "Proxy for submitted jobs: HTTPS_PROXY=$(mask_proxy_url "${HTTPS_PROXY:-}") | NO_PROXY=${NO_PROXY:-} | WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-}"

if [[ -n "$PARTITION" ]]; then
  export SBATCH_PARTITION="$PARTITION"
else
  unset SBATCH_PARTITION
fi

if [[ -n "$EXCLUDE" ]]; then
  export SBATCH_EXCLUDE="$EXCLUDE"
else
  unset SBATCH_EXCLUDE
fi

SQUEUE_USER="${USER:-$(id -un)}"
declare -A ACTIVE_JOB_NAMES=()

# The default squeue table truncates long job names. With a custom "%j"
# format and no field width, Slurm prints one full job name per line.
existing_job_names="$(squeue -h -u "$SQUEUE_USER" -o "%j")"
while IFS= read -r active_job_name; do
  [[ -n "$active_job_name" ]] || continue
  ACTIVE_JOB_NAMES["$active_job_name"]=1
done <<< "$existing_job_names"

for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    for model in "${MODELS[@]}"; do
      model_id="$(basename "$model")"
      req_nodes="$NNODES"
      for ((job_idx = 1; job_idx <= JOBS_PER_SETUP; job_idx++)); do
        job_suffix="job${job_idx}"
        job_name="${dataset}_${mode}_${model_id}_${GPU}_trainN${NNODES}_allocN${req_nodes}_${job_suffix}"
        job_workdir="${SATRL_JOB_WORK_ROOT}/${job_name}"

        if [[ -n "${ACTIVE_JOB_NAMES["$job_name"]+x}" ]]; then
          echo "Skipping existing job: ${job_name}"
          continue
        fi

        mkdir -p "${job_workdir}"

        sbatch_args=(
          --job-name="${job_name}"
          --nodes="${req_nodes}"
          --ntasks-per-node=1
          --gres="gpu:${GPU}:${GPUS_PER_NODE}"
          --cpus-per-task="${CPUS_PER_TASK}"
          --mem="${MEM}"
          --output="${DEFAULT_PROJECT}/slurm/%j.out"
          --error="${DEFAULT_PROJECT}/slurm/%j.err"
          --chdir="${job_workdir}"
          --signal=B:USR1@300
          --export=ALL
        )

        if [[ -n "$TIME_LIMIT" ]]; then
          sbatch_args+=(--time="${TIME_LIMIT}")
        fi

        if [[ -n "$PARTITION" ]]; then
          sbatch_args+=(--partition="${PARTITION}")
        fi

        if [[ -n "$EXCLUDE" ]]; then
          sbatch_args+=(--exclude="${EXCLUDE}")
        fi

        sbatch "${sbatch_args[@]}" \
          "${TRAIN_JOB_SH}" \
          "${dataset}" \
          "${mode}" \
          "${model}" \
          "${GPU}" \
          "${req_nodes}" \
          "${GPUS_PER_NODE}" \
          "${job_suffix}"

        ACTIVE_JOB_NAMES["$job_name"]=1
        echo "Submitted job: ${job_name}"
      done
    done
  done
done
