#!/bin/bash
set -euo pipefail

export PROJECT=$PROJECT
export RAY_PORT=64990
export RAY_TMPDIR_BASE="/tmp/ray_${SLURM_JOB_ID}"
export RAY_TMPDIR="${RAY_TMPDIR_BASE}_0"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
if [[ "$GPUS_PER_NODE" =~ ^([0-9]+) ]]; then
  GPUS_PER_NODE="${BASH_REMATCH[1]}"
else
  GPUS_PER_NODE=4
fi
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-32}"
export CUDA_HOME="/usr/local/cuda"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

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

write_export_if_set() {
  local key="$1"
  local value="${!key-}"
  if [[ -n "$value" ]]; then
    printf 'export %s=%q\n' "$key" "$value" >> "$ENV_FILE"
  fi
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

# Gather all allocated hostnames
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE=${NODES[0]}
WORKER_NODES=("${NODES[@]:1}")    # slice from index 1 onward
HEAD_IP=$(getent hosts "$HEAD_NODE" | head -n1 | awk '{print $1}')
NO_PROXY_ENTRIES=("${NODES[@]}" "$HEAD_IP")
for NODE in "${NODES[@]}"; do
  NODE_IP=$(getent hosts "$NODE" | head -n1 | awk '{print $1}')
  NO_PROXY_ENTRIES+=("$NODE_IP")
done
add_no_proxy_entries "${NO_PROXY_ENTRIES[@]}"

# Ensure params directory exists
PARAMS_DIR="$PROJECT/logic/output/params"
mkdir -p "$PARAMS_DIR"

# Write a unique shared env file (per-job)
ENV_FILE="$PARAMS_DIR/ray_params_${SLURM_JOB_ID}.env"
cat > "$ENV_FILE" <<EOF
export PROJECT='${PROJECT}'
export RAY_PORT='${RAY_PORT}'
export RAY_TMPDIR_BASE='${RAY_TMPDIR_BASE}'
export RAY_TMPDIR='${RAY_TMPDIR}'
export GPUS_PER_NODE='${GPUS_PER_NODE}'
export CPUS_PER_TASK='${CPUS_PER_TASK}'
export CUDA_HOME='${CUDA_HOME}'
export CUDA_PATH='${CUDA_PATH}'
export CUDACXX='${CUDACXX}'
export PATH='${CUDA_HOME}/bin':\${PATH:-}
export LD_LIBRARY_PATH='${CUDA_HOME}/lib64':\${LD_LIBRARY_PATH:-}
export HEAD_NODE='${HEAD_NODE}'
export HEAD_IP='${HEAD_IP}'
export VLLM_HOST_IP='$HEAD_IP'
export RAY_ADDRESS=auto
EOF
for proxy_key in \
  HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy \
  WANDB_MODE WANDB_BASE_URL WANDB_ENTITY WANDB_INIT_TIMEOUT WANDB_HTTP_TIMEOUT; do
  write_export_if_set "$proxy_key"
done

echo "Proxy for Ray: HTTPS_PROXY=$(mask_proxy_url "${HTTPS_PROXY:-}") | NO_PROXY=${NO_PROXY:-} | WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-}"

# 1) Launch the head node in the background
echo "▶️  Starting Ray head on $(hostname -s) (this is $HEAD_NODE)"
apptainer exec --nv \
  --bind "$HOME":"$HOME" \
  --bind /tmp:/tmp \
  --bind "$PROJECT":"$PROJECT" \
  $PROJECT/sandbox/verl.sif \
  bash -c "\
    source '$ENV_FILE' && \
    ray stop --force && \
    ray start --head \
      --node-ip-address=${HEAD_IP} \
      --port=\$RAY_PORT \
      --num-gpus=\$GPUS_PER_NODE \
      --temp-dir=\$RAY_TMPDIR && \
    echo '✅ Ray head node is up at ${HEAD_IP}:${RAY_PORT}' && \
    tail -f /dev/null
  " &

# 2) Wait for the head to finish booting
sleep 10

# 3) Launch workers if any
if [ "${#WORKER_NODES[@]}" -eq 0 ]; then
  echo "ℹ️  Only one node allocated. No workers to launch."
else
  i=0
  for NODE in "${WORKER_NODES[@]}"; do
    NODE_IP=$(getent hosts "$NODE" | head -n1 | awk '{print $1}')

    # Append each worker's hostname/IP to the env file
    cat >> "$ENV_FILE" <<EOF
export WORKER_${i}_NODE='${NODE}'
export WORKER_${i}_IP='${NODE_IP}'
EOF

    srun --nodes=1 --ntasks=1 --nodelist="$NODE" --overlap \
      --cpus-per-task="$CPUS_PER_TASK" \
      --cpu-bind=none \
      --gpus-per-node="$GPUS_PER_NODE" \
      apptainer exec --nv \
        --bind "$HOME":"$HOME" \
        --bind /tmp:/tmp \
        --bind "$PROJECT":"$PROJECT" \
        $PROJECT/sandbox/verl.sif \
        bash -c "
          source '$ENV_FILE' &&
          ray start --address=\${HEAD_IP}:\${RAY_PORT} \
            --node-ip-address=\${WORKER_${i}_IP} \
            --num-gpus=\${GPUS_PER_NODE} \
            --temp-dir=\${RAY_TMPDIR} &&
          echo '🚀 Worker joined from \${WORKER_${i}_NODE} at \${WORKER_${i}_IP}' &&
          tail -f /dev/null
        " &

    i=$((i+1))
  done
fi

EXPECTED_GPUS=$(( ${#NODES[@]} * GPUS_PER_NODE ))
echo "Waiting for Ray to register ${EXPECTED_GPUS} GPUs across ${#NODES[@]} node(s)..."
for attempt in {1..60}; do
  AVAILABLE_GPUS="$(
    apptainer exec --nv \
      --bind "$HOME":"$HOME" \
      --bind /tmp:/tmp \
      --bind "$PROJECT":"$PROJECT" \
      "$PROJECT/sandbox/verl.sif" \
      bash -c "source '$ENV_FILE' && python3 - <<'PY'
import os
import ray

ray.init(address=f\"{os.environ['HEAD_IP']}:{os.environ['RAY_PORT']}\", ignore_reinit_error=True, logging_level='ERROR')
print(int(ray.cluster_resources().get('GPU', 0)))
ray.shutdown()
PY" 2>/dev/null || echo 0
  )"
  AVAILABLE_GPUS="${AVAILABLE_GPUS//$'\n'/}"
  if [[ "$AVAILABLE_GPUS" =~ ^[0-9]+$ ]] && (( AVAILABLE_GPUS >= EXPECTED_GPUS )); then
    echo "✅ Ray registered ${AVAILABLE_GPUS}/${EXPECTED_GPUS} GPUs."
    exit 0
  fi
  echo "Ray currently reports ${AVAILABLE_GPUS:-0}/${EXPECTED_GPUS} GPUs; retry ${attempt}/60..."
  sleep 5
done

echo "ERROR: Ray did not register all expected GPUs (${EXPECTED_GPUS}) before timeout." >&2
apptainer exec --nv \
  --bind "$HOME":"$HOME" \
  --bind /tmp:/tmp \
  --bind "$PROJECT":"$PROJECT" \
  "$PROJECT/sandbox/verl.sif" \
  bash -c "source '$ENV_FILE' && ray status --address=\${HEAD_IP}:\${RAY_PORT}" || true
exit 1
