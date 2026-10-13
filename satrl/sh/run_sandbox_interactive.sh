SF_DIR="$PROJECT/logic/SandboxFusion"
RUN_PY="$SF_DIR/run_no_background.py"
LOG_FILE="$SF_DIR/slurm_output/sandbox_interactive_${SLURM_JOB_ID:-manual}.log"
PID_FILE="$SF_DIR/slurm_output/sandbox_interactive_${SLURM_JOB_ID:-manual}.pid"
SF_CONFIG_FILE="$PROJECT/logic/output/params/sf_${SLURM_JOB_ID:-manual}.yaml"
SANDBOX_HEALTH_TIMEOUT_S="${SANDBOX_HEALTH_TIMEOUT_S:-120}"
SANDBOX_HEALTH_INTERVAL_S="${SANDBOX_HEALTH_INTERVAL_S:-2}"
SANDBOX_HEALTH_CURL_TIMEOUT_S="${SANDBOX_HEALTH_CURL_TIMEOUT_S:-10}"
SANDBOX_HEALTH_STABILITY_S="${SANDBOX_HEALTH_STABILITY_S:-3}"

# Ensure output dir exists
mkdir -p "$SF_DIR/slurm_output"

# Module (adjust if your cluster uses a different name/version)
module load apptainer/1.3.5 || true

# Avoid ROCm env leakage on NVIDIA nodes
unset ROCR_VISIBLE_DEVICES || true

# Optional: set/override PORT for the Python runner
export PORT=10080

extract_sandbox_url() {
  local config_file="$1"
  sed -nE "s/^[[:space:]]*sandbox_fusion_url:[[:space:]]*[\"']?([^\"'[:space:]]+)[\"']?.*$/\1/p" "$config_file" | head -n 1
}

log_since_start() {
  local start_line="$1"
  tail -n +"$((start_line + 1))" "$LOG_FILE" 2>/dev/null || true
}

stop_sandbox_launcher() {
  local launcher_pid="$1"
  if kill -0 "$launcher_pid" 2>/dev/null; then
    kill -TERM "-$launcher_pid" 2>/dev/null || kill -TERM "$launcher_pid" 2>/dev/null || true
  fi
}

wait_for_sandbox_ready() {
  local launcher_pid="$1"
  local log_start_line="$2"
  local deadline=$((SECONDS + SANDBOX_HEALTH_TIMEOUT_S))
  local url=""
  local response=""
  local curl_rc=0
  local last_error="waiting for $SF_CONFIG_FILE"
  local payload='{"compile_timeout":5,"run_timeout":5,"code":"print(\"sf_health_ok\")","stdin":null,"memory_limit_MB":256,"language":"python","files":{},"fetch_files":[]}'
  local new_log=""
  local startup_seen=0
  local ready_seen_at=-1

  echo "Waiting for SandboxFusion health check: $SF_CONFIG_FILE"

  while (( SECONDS < deadline )); do
    new_log="$(log_since_start "$log_start_line")"
    if [[ "$new_log" == *"error while attempting to bind"* || "$new_log" == *"address already in use"* ]]; then
      echo "ERROR: SandboxFusion failed to bind its port." >&2
      echo "SandboxFusion log tail ($LOG_FILE):" >&2
      tail -80 "$LOG_FILE" >&2 || true
      return 1
    fi

    startup_seen=0
    if [[ "$new_log" == *"Application startup complete."* ]]; then
      startup_seen=1
    fi

    if [[ -f "$SF_CONFIG_FILE" ]]; then
      url="$(extract_sandbox_url "$SF_CONFIG_FILE")"
    fi

    if [[ -n "$url" ]]; then
      response="$(curl --noproxy "*" -sS --max-time "$SANDBOX_HEALTH_CURL_TIMEOUT_S" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        --data "$payload" \
        "$url" 2>&1)"
      curl_rc=$?

      if (( curl_rc == 0 )) && [[ "$response" == *"sf_health_ok"* ]]; then
        if (( startup_seen )); then
          if (( ready_seen_at < 0 )); then
            ready_seen_at=$SECONDS
            last_error="health check passed; waiting ${SANDBOX_HEALTH_STABILITY_S}s for launcher stability"
          elif (( SECONDS - ready_seen_at >= SANDBOX_HEALTH_STABILITY_S )); then
            echo "SandboxFusion ready: $url"
            return 0
          fi
        else
          last_error="health endpoint responded, waiting for current launcher startup log"
        fi
      else
        ready_seen_at=-1
        last_error="url=$url curl_rc=$curl_rc response=${response:0:500}"
      fi
    fi

    if ! kill -0 "$launcher_pid" 2>/dev/null; then
      echo "ERROR: SandboxFusion launcher exited before health check passed." >&2
      echo "Last health check state: $last_error" >&2
      echo "SandboxFusion log tail ($LOG_FILE):" >&2
      tail -80 "$LOG_FILE" >&2 || true
      return 1
    fi

    sleep "$SANDBOX_HEALTH_INTERVAL_S"
  done

  echo "ERROR: SandboxFusion health check timed out after ${SANDBOX_HEALTH_TIMEOUT_S}s." >&2
  echo "Last health check state: $last_error" >&2
  echo "SandboxFusion log tail ($LOG_FILE):" >&2
  tail -80 "$LOG_FILE" >&2 || true
  return 1
}


# Always run the Python launcher from the SandboxFusion directory
cd "$SF_DIR" || exit 1

# Launch detached from the terminal. The uvicorn server inherits these streams.
if [[ -f "$LOG_FILE" ]]; then
  log_start_line="$(wc -l < "$LOG_FILE")"
else
  log_start_line=0
fi
log_start_line="${log_start_line//[[:space:]]/}"
setsid nohup python "$RUN_PY" >> "$LOG_FILE" 2>&1 < /dev/null &
launcher_pid=$!
echo "$launcher_pid" > "$PID_FILE"
echo "SandboxFusion launcher PID $(cat "$PID_FILE"); logs: $LOG_FILE"

if [[ "${SANDBOX_HEALTH_CHECK:-1}" != "0" ]]; then
  if ! wait_for_sandbox_ready "$launcher_pid" "$log_start_line"; then
    stop_sandbox_launcher "$launcher_pid"
    exit 1
  fi
fi
