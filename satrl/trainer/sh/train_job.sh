#!/bin/bash
#
# train_job.sh - one training job with infinite resubmission on failure/timeout
#
set -euxo pipefail

########################################
#            ARGUMENT PARSING          #
########################################
usage() {
  echo "Usage: $0 DATASET MODE MODEL [GPU] [NNODES] [GPUS_PER_NODE] [RUN_SUFFIX]"
  echo "Example: $0 zebralogic tools_general /path/to/model l40s 4 4 job1"
  echo
  echo "For normal submission, use:"
  echo "  bash path/to/satrl/trainer/sh/submit_all_l40s_resubmit.sh"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if (( $# < 3 )); then
  usage >&2
  exit 2
fi

DATASET="${1:?DATASET required}"
MODE="${2:?MODE required}"
MODEL="${3:?MODEL required}"
GPU="${4:-l40s}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "train_job.sh must run inside a Slurm allocation. Use submit_all_l40s_resubmit.sh or sbatch this script with arguments." >&2
  usage >&2
  exit 2
fi

# Unified params (come from submit_all.sh)
NNODES="${5:-${NNODES:-4}}"
GPUS_PER_NODE="${6:-${GPUS_PER_NODE:-4}}"
RUN_SUFFIX="${7:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SATRL_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_ROOT="$(cd -- "$SATRL_ROOT/.." && pwd)"
export PROJECT="${PROJECT:-$ARTIFACT_ROOT}"
TRAIN_JOB_SH="${TRAIN_JOB_SH:-$SATRL_ROOT/trainer/sh/train_job.sh}"

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

add_no_proxy_entries() {
  local current="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1}}"
  local entry
  for entry in "$@"; do
    [[ -n "$entry" ]] || continue
    if [[ ",$current," != *",$entry,"* ]]; then
      current="${current:+$current,}$entry"
    fi
  done
  export NO_PROXY="$current"
  export no_proxy="$current"
}

mask_proxy_url() {
  local value="${1:-}"
  if [[ "$value" == *"://"*"@"* ]]; then
    sed -E 's#(://)[^/@]+@#\1***@#' <<< "$value"
  else
    printf '%s\n' "$value"
  fi
}

LOCAL_PROXY_PIDS=()

cleanup_local_proxy() {
  if ((${#LOCAL_PROXY_PIDS[@]} > 0)); then
    kill "${LOCAL_PROXY_PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup_local_proxy EXIT

is_port_listening() {
  local port="$1"
  ss -H -ltn 2>/dev/null | awk -v port=":${port}" '$4 ~ port "$" { found = 1 } END { exit found ? 0 : 1 }'
}

choose_proxy_ports() {
  local seed="${SLURM_JOB_ID:-$$}"
  local start=$((18000 + (seed % 20000)))
  local i

  for i in $(seq 0 200); do
    socks_port=$((start + i * 2))
    http_port=$((socks_port + 1))
    if ! is_port_listening "$socks_port" && ! is_port_listening "$http_port"; then
      return 0
    fi
  done

  echo "ERROR: could not find free localhost proxy ports near ${start}" >&2
  return 1
}

wait_for_proxy_port() {
  local port="$1"
  local name="$2"
  local i

  for i in $(seq 1 30); do
    if (exec 3<>"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.2
  done

  echo "ERROR: ${name} did not start on 127.0.0.1:${port}" >&2
  return 1
}

is_proxy_required_cluster() {
  local cluster="${SLURM_CLUSTER_NAME:-}"
  local host="${HOST_FQDN:-$(hostname -f 2>/dev/null || hostname)}"
  cluster="${cluster,,}"
  host="${host,,}"
  [[ "$cluster" == *vulcan* || "$cluster" == *rack* || "$host" == *vulcan* || "$host" == *rack* ]]
}

should_start_local_proxy() {
  case "${SATRL_START_PROXY:-auto}" in
    1|true|TRUE|yes|YES) return 0 ;;
    0|false|FALSE|no|NO) return 1 ;;
    auto)
      is_proxy_required_cluster
      return $?
      ;;
    *)
      echo "Invalid SATRL_START_PROXY=${SATRL_START_PROXY}; expected auto, 1, or 0." >&2
      return 2
      ;;
  esac
}

start_local_proxy() {
  local proxy_bastion="${PROXY_BASTION:?Set PROXY_BASTION when proxy startup is enabled}"
  local proxy_python="${PROXY_PYTHON:-python}"
  local proxy_log
  local start_proxy_rc

  should_start_local_proxy
  start_proxy_rc=$?
  case "$start_proxy_rc" in
    0) ;;
    1)
      if [[ -n "${HTTPS_PROXY:-${https_proxy:-}}" ]]; then
        echo "[proxy] Using provided proxy endpoint: HTTPS_PROXY=$(mask_proxy_url "${HTTPS_PROXY:-${https_proxy:-}}")"
      else
        echo "[proxy] No proxy endpoint configured; using direct internet."
      fi
      return 0
      ;;
    *) return "$start_proxy_rc" ;;
  esac

  if [[ ! -x "$proxy_python" ]]; then
    echo "ERROR: proxy Python not found or not executable: $proxy_python" >&2
    return 1
  fi
  if ! "$proxy_python" -c 'import pproxy' >/dev/null 2>&1; then
    echo "ERROR: pproxy is not importable with $proxy_python" >&2
    return 1
  fi

  choose_proxy_ports || return 1
  proxy_log="/tmp/pproxy_${USER}_${http_port}.log"

  echo "[proxy] Starting local SOCKS tunnel via ${proxy_bastion} and HTTP proxy on 127.0.0.1:${http_port}"
  ssh -n -D "127.0.0.1:${socks_port}" -q -C -N \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    "$proxy_bastion" &
  LOCAL_PROXY_PIDS+=("$!")
  wait_for_proxy_port "$socks_port" "SOCKS proxy" || return 1

  "$proxy_python" -m pproxy \
    -l "http://127.0.0.1:${http_port}" \
    -r "socks5://127.0.0.1:${socks_port}" \
    >"$proxy_log" 2>&1 &
  LOCAL_PROXY_PIDS+=("$!")
  wait_for_proxy_port "$http_port" "HTTP proxy" || return 1

  export http_proxy="http://127.0.0.1:${http_port}"
  export https_proxy="$http_proxy"
  export all_proxy="$http_proxy"
  export HTTP_PROXY="$http_proxy"
  export HTTPS_PROXY="$http_proxy"
  export ALL_PROXY="$http_proxy"
  export WSS_PROXY="$http_proxy"
  export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"
  export NO_PROXY="${NO_PROXY:-$no_proxy}"
  normalize_proxy_env

  echo "[proxy] Local HTTP proxy ready: HTTPS_PROXY=$(mask_proxy_url "$HTTPS_PROXY") | log=$proxy_log"
}

normalize_proxy_env

# If we’re already inside an allocation, prefer the actual allocation count
# (this prevents mismatches if someone manually changed sbatch flags).
if [[ -n "${SLURM_NNODES:-}" ]]; then
  if [[ "${SLURM_NNODES}" != "${NNODES}" ]]; then
    echo "[warn] SLURM_NNODES=${SLURM_NNODES} differs from requested NNODES=${NNODES}; using SLURM_NNODES."
    NNODES="${SLURM_NNODES}"
  fi
fi
export NNODES GPUS_PER_NODE

########################################
#       INFINITE RESUBMIT SECTION      #
########################################
REQUEUE_STATE_DIR="$PROJECT/logic/output/requeue_state"
mkdir -p "$REQUEUE_STATE_DIR"

REQUEUE_COUNT_FILE="$REQUEUE_STATE_DIR/${SLURM_JOB_ID}.count"
STOP_FILE="$REQUEUE_STATE_DIR/STOP_RESUBMIT"   # global STOP marker

init_requeue_counter() { [[ -f "$REQUEUE_COUNT_FILE" ]] || echo 0 > "$REQUEUE_COUNT_FILE"; }
get_requeue_count() { cat "$REQUEUE_COUNT_FILE" 2>/dev/null || echo 0; }
increment_requeue_count() { echo $(( $(get_requeue_count) + 1 )) > "$REQUEUE_COUNT_FILE"; }

# Keep these in one place so allocation and resubmission stay identical
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
SBATCH_PARTITION_VALUE="${SBATCH_PARTITION:-}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-64}"
SBATCH_MEM="${SBATCH_MEM:-400G}"
DEFAULT_EXCLUDE_NODES="rack12-13"
SBATCH_EXCLUDE_VALUE="${SBATCH_EXCLUDE:-$DEFAULT_EXCLUDE_NODES}"
unset SBATCH_PARTITION
unset SBATCH_EXCLUDE

normalize_partition() {
  case "${1:-}" in
    ""|"(null)"|"null"|"none"|"None") echo "" ;;
    *) echo "$1" ;;
  esac
}

HOST_FQDN="$(hostname -f 2>/dev/null || hostname)"
SLURM_CLUSTER_NAME="$(
  scontrol show config 2>/dev/null \
    | awk -F= '/^[[:space:]]*ClusterName[[:space:]]*=/ {gsub(/[[:space:]]/, "", $2); print tolower($2); exit}' \
    || true
)"
SBATCH_PARTITION_VALUE="$(normalize_partition "$SBATCH_PARTITION_VALUE")"
if [[ "${SLURM_CLUSTER_NAME}" == *vulcan* || "${SLURM_CLUSTER_NAME}" == *rack* || "${HOST_FQDN,,}" == *vulcan* || "${HOST_FQDN,,}" == *rack* ]]; then
  SBATCH_TIME="8:00:00"
  SBATCH_PARTITION_VALUE="gpubase_interac"
fi

resubmit_infinite() {
  if [[ -f "$STOP_FILE" ]]; then
    echo "[resubmit] STOP file present -> NOT resubmitting; exiting."
    exit 0
  fi

  increment_requeue_count
  echo "[resubmit] attempt $(get_requeue_count) for job ${SLURM_JOB_ID:-unknown}"

  sbatch_args=(
    --job-name="${SLURM_JOB_NAME:-resubmitted_job}"
    --nodes="${NNODES}"
    --ntasks-per-node=1
    --gres="gpu:${GPU}:${GPUS_PER_NODE}"
    --cpus-per-task="${SBATCH_CPUS_PER_TASK}"
    --mem="${SBATCH_MEM}"
    --output="$PROJECT/slurm/%j.out"
    --error="$PROJECT/slurm/%j.err"
    --chdir="$PROJECT"
    --signal=B:USR1@300
  )

  if [[ -n "$SBATCH_TIME" ]]; then
    sbatch_args+=(--time="${SBATCH_TIME}")
  fi

  if [[ -n "$SBATCH_PARTITION_VALUE" ]]; then
    sbatch_args+=(--partition="${SBATCH_PARTITION_VALUE}")
  fi

  if [[ -n "$SBATCH_EXCLUDE_VALUE" ]]; then
    sbatch_args+=(--exclude="${SBATCH_EXCLUDE_VALUE}")
  fi

  resubmit_script="$TRAIN_JOB_SH"
  if [[ ! -f "$resubmit_script" ]]; then
    echo "[resubmit] WARN: canonical script missing ($resubmit_script); falling back to $0." >&2
    resubmit_script="$0"
  fi

  train_job_args=("$DATASET" "$MODE" "$MODEL" "$GPU" "$NNODES" "$GPUS_PER_NODE")
  if [[ -n "$RUN_SUFFIX" ]]; then
    train_job_args+=("$RUN_SUFFIX")
  fi

  if ! sbatch "${sbatch_args[@]}" "$resubmit_script" "${train_job_args[@]}"; then
    echo "[resubmit] ERROR: sbatch failed." >&2
    exit 1
  fi

  exit 0
}

on_err() {
  rc=$?
  echo "[trap ERR] batch script failed with exit code $rc — resubmitting new job (unless STOP)."
  resubmit_infinite
}
trap on_err ERR

on_sigusr1() {
  echo "[signal] USR1 (pre-time-limit) caught — resubmitting new job (unless STOP)."
  resubmit_infinite
}
trap on_sigusr1 USR1

on_sigterm() {
  echo "[signal] TERM caught — resubmitting new job (unless STOP)."
  resubmit_infinite
}
trap on_sigterm TERM

init_requeue_counter

########################################
#       PROJECT-SPECIFIC SETUP         #
########################################
PROJECT_DIR="$PROJECT/logic/verl"
SCRIPT_PATH="$PROJECT/logic/satrl/trainer/grpo_trainer/train.sh"
SIF_DIR="$PROJECT/sandbox/verl.sif"
PARAMS_DIR="$PROJECT/logic/output/params"

mkdir -p "$PROJECT/logic/output/slurm_output"
unset ROCR_VISIBLE_DEVICES

SCRIPT_ARGS=(--mode "$MODE" --dataset "$DATASET" --model "$MODEL")
if [[ -n "$RUN_SUFFIX" ]]; then
  SCRIPT_ARGS+=(--run-suffix "$RUN_SUFFIX")
fi
printf -v SCRIPT_ARGS_STR "%q " "${SCRIPT_ARGS[@]}"

########################################
# Sandbox
########################################
bash "$PROJECT/logic/satrl/sh/run_sandbox_interactive.sh"
sleep 5

########################################
# Ray cluster
########################################
export RAY_PORT=64990
export RAY_TMPDIR_BASE="/tmp/ray_${SLURM_JOB_ID}"
export RAY_TMPDIR="${RAY_TMPDIR_BASE}_0"

mapfile -t NODES_ARR < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export HEAD_NODE="${NODES_ARR[0]}"
export HEAD_IP="$(getent hosts "$HEAD_NODE" | head -n1 | awk '{print $1}')"
NO_PROXY_ENTRIES=("${NODES_ARR[@]}" "$HEAD_IP")
for node in "${NODES_ARR[@]}"; do
  node_ip="$(getent hosts "$node" | head -n1 | awk '{print $1}')"
  NO_PROXY_ENTRIES+=("$node_ip")
done
add_no_proxy_entries "${NO_PROXY_ENTRIES[@]}"
echo "[proxy] HTTPS_PROXY=$(mask_proxy_url "${HTTPS_PROXY:-}") | HTTP_PROXY=$(mask_proxy_url "${HTTP_PROXY:-}") | NO_PROXY=${NO_PROXY:-} | WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-}"

set +e
start_local_proxy
proxy_rc=$?
set -e
if (( proxy_rc != 0 )); then
  echo "[proxy] Local proxy setup failed with exit code $proxy_rc. Not resubmitting; fix proxy setup first." >&2
  exit "$proxy_rc"
fi

bash "$PROJECT/logic/satrl/sh/run_ray_container.sh"

# Ray env (sets RAY_ADDRESS etc.)
xtrace_was_on=0
if [[ "$-" == *x* ]]; then
  xtrace_was_on=1
  set +x
fi
source "$PARAMS_DIR/ray_params_${SLURM_JOB_ID}.env"
if (( xtrace_was_on )); then
  set -x
fi
sleep 30
########################################
# Reward server (tools_symbolic only)
########################################


########################################
# MAIN TRAINING EXECUTION
########################################
run_in_verl_container() {
  local apptainer_args=(
    exec --nv
    --bind "$HOME":"$HOME"
    --bind "$SCRATCH":"$SCRATCH"
    --bind "$PROJECT":"$PROJECT"
    --env PROJECT="$PROJECT"
    --env PYTHONPATH="$PROJECT/logic/verl:${PYTHONPATH:-}"
    --env RAY_PORT="$RAY_PORT"
    --env RAY_TMPDIR_BASE="$RAY_TMPDIR_BASE"
    --env RAY_TMPDIR="$RAY_TMPDIR"
    --env HEAD_NODE="$HEAD_NODE"
    --env HEAD_IP="$HEAD_IP"
    --env RAY_ADDRESS="$RAY_ADDRESS"
    --env SLURM_JOB_ID="$SLURM_JOB_ID"
    --env SLURM_NNODES="$SLURM_NNODES"
    --env SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST"
    --env SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
    --env SLURM_MEM_PER_NODE="${SLURM_MEM_PER_NODE:-}"
    --env WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-120}"
    --env SIF_DIR="$SIF_DIR"
    --env SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
  )
  local env_key
  for env_key in \
    HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy \
    WANDB_MODE WANDB_BASE_URL WANDB_ENTITY WANDB_HTTP_TIMEOUT; do
    if [[ -n "${!env_key-}" ]]; then
      apptainer_args+=(--env "$env_key=${!env_key}")
    fi
  done

  apptainer "${apptainer_args[@]}" "$SIF_DIR" "$@"
}

if [[ "${WANDB_CONNECTIVITY_CHECK:-1}" == "1" ]]; then
  echo "[proxy] Running W&B connectivity preflight from driver and Ray worker."
  xtrace_was_on=0
  if [[ "$-" == *x* ]]; then
    xtrace_was_on=1
    set +x
  fi
  set +e
  run_in_verl_container \
    bash -lc "cd '$PROJECT_DIR' && python3 '$PROJECT/logic/satrl/tmp/test_wandb_connectivity.py' --ray-address '$HEAD_IP:$RAY_PORT' --timeout '${WANDB_CONNECTIVITY_TIMEOUT:-20}' --skip-init"
  preflight_rc=$?
  set -e
  if (( xtrace_was_on )); then
    set -x
  fi
  if (( preflight_rc != 0 )); then
    echo "[proxy] W&B connectivity preflight failed with exit code $preflight_rc. Not resubmitting; fix the proxy endpoint/env first." >&2
    exit "$preflight_rc"
  fi
fi

set +e
xtrace_was_on=0
if [[ "$-" == *x* ]]; then
  xtrace_was_on=1
  set +x
fi
run_in_verl_container bash -lc "cd '$PROJECT_DIR' && bash '$SCRIPT_PATH' $SCRIPT_ARGS_STR"
rc=$?
if (( xtrace_was_on )); then
  set -x
fi
set -e

if (( rc != 0 )); then
  echo "[exit] Training failed with exit code $rc — resubmitting new job (unless STOP)."
  resubmit_infinite
fi

echo "[exit] Training finished successfully — no resubmit."
echo "✅ Done. Logs in logic/output/slurm_output/"
