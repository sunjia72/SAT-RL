# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray

from verl.tools.base_tool import BaseTool
from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case
from verl.utils.rollout_trace import rollout_trace_op

from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")

RATE_LIMITER_NAME_ENV = "VERL_RATE_LIMITER_NAME"
SANDBOX_LOG_PATH_ENV = "VERL_SANDBOX_FUSION_LOG_PATH"
SANDBOX_CONFIG_PATH_ENV = "VERL_SANDBOX_FUSION_CONFIG_PATH"
SANDBOX_CONFIG_DIR_ENV = "VERL_SANDBOX_FUSION_CONFIG_DIR"
REWARD_LLM_BASE_URL_ENV = "REWARD_LLM_BASE_URL"
REWARD_LLM_MODEL_ENV = "REWARD_LLM_MODEL"
REWARD_LLM_API_KEY_ENV = "REWARD_LLM_API_KEY"
DEFAULT_TEACHER_MODEL = "openai/gpt-oss-20b"
JEMALLOC_BACKGROUND_THREAD_ERROR = "<jemalloc>: arena 0 background thread creation failed (11)"
EMPTY_STDOUT_WARNING = (
    "[ToolWarning] Program finished successfully but produced empty output. "
    "No solver result was printed. Add an explicit print(...)."
)
EMPTY_UNFINISHED_ERROR = (
    "[ToolError] SandboxFusion did not finish and produced empty output. "
    "status={run_status}"
)
FENCED_CODE_BLOCK_RE = re.compile(r"```[ \t]*(?:python|py)?[^\n`]*\n(.*?)```", re.IGNORECASE | re.DOTALL)
FEEDBACK_TAG_RE = re.compile(r"<<<\s*FEEDBACK\s*>>>\s*(.*?)\s*(?=<<<|$)", re.IGNORECASE | re.DOTALL)
TEACHER_MODEL_CACHE: dict[str, str] = {}


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        return value if isinstance(value, str) else str(value)
    except Exception:
        return ""


def _sandbox_error_type(backend_status: str) -> Optional[str]:
    """Map SandboxFusion's backend status to a stable, consumer-facing category."""
    return {
        "api_error": "sandbox_api_error",
        "sandbox_error": "sandbox_error",
        "compile_timeout": "compile_timeout",
        "compile_error": "compile_error",
        "timeout": "execution_timeout",
        "runtime_error": "runtime_error",
        "unknown_failure": "backend_execution_error",
        "unknown_failure_state": "backend_execution_error",
        "unexpected_success_state": "backend_execution_error",
        "unknown_api_state": "backend_execution_error",
    }.get(backend_status)


def _build_execution_status(
    result_status: Any,
    metadata: dict[str, Any],
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    """Build a compact JSON-safe status record without duplicating tool output."""
    backend_status = _safe_str(metadata.get("status") or "unknown")
    execution_succeeded = backend_status in {"success", "wrong_answer"}
    output_usable = execution_succeeded and bool(stdout)
    error_type = _sandbox_error_type(backend_status)
    if execution_succeeded and not output_usable:
        error_type = "empty_output"
    elif not execution_succeeded and error_type is None:
        error_type = "backend_execution_error"

    error_message = ""
    if not execution_succeeded:
        error_message = _safe_str(metadata.get("api_request_error") or metadata.get("compile_stderr") or stderr).strip()

    # These values originate from a JSON HTTP response. Restrict the record to
    # scalar fields so it remains safe to move through DataProto object arrays.
    return {
        "stage": "execute",
        "status": "success" if execution_succeeded and output_usable else "error",
        "error_type": error_type,
        "error_message": error_message[:1000],
        "backend_invoked": True,
        "execution_succeeded": execution_succeeded,
        "output_usable": output_usable,
        "result_status": result_status
        if isinstance(result_status, (bool, int, float, str))
        else _safe_str(result_status),
        "backend_status": backend_status,
        "api_status": _safe_str(metadata.get("api_status")),
        "compile_status": _safe_str(metadata.get("compile_status")),
        "run_status": _safe_str(metadata.get("run_status") or metadata.get("status") or "Unknown"),
        "exit_code": metadata.get("exit_code")
        if isinstance(metadata.get("exit_code"), (type(None), bool, int, float, str))
        else _safe_str(metadata.get("exit_code")),
        "stdout_present": bool(stdout),
        "stderr_present": bool(stderr),
    }


def _project_root() -> str:
    artifact_root = os.environ.get(
        "ARTIFACT_ROOT", str(Path(__file__).resolve().parents[2])
    )
    return os.environ.get("PROJECT", artifact_root)


def _get_slurm_jobid() -> Optional[str]:
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    job_id = _safe_str(job_id).strip()
    return job_id or None


def _load_sf_yaml_config() -> dict[str, Any]:
    config_path = os.environ.get(SANDBOX_CONFIG_PATH_ENV)
    if not config_path:
        job_id = _get_slurm_jobid()
        if not job_id:
            return {}
        config_dir = os.environ.get(SANDBOX_CONFIG_DIR_ENV)
        if not config_dir:
            config_dir = os.path.join(_project_root(), "logic", "output", "params")
        config_path = os.path.join(config_dir, f"sf_{job_id}.yaml")

    try:
        import yaml

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning(f"Failed to load sandbox fusion config {config_path}: {exc}")
        return {}


def _normalize_openai_base_url(base_url: str) -> str:
    base_url = _safe_str(base_url).strip().rstrip("/")
    if base_url and not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def _request_openai_compat(
    url: str,
    payload: Optional[dict[str, Any]] = None,
    timeout_s: float = 30.0,
    api_key_env: str = REWARD_LLM_API_KEY_ENV,
) -> tuple[int, str]:
    body = None
    method = "GET"
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        method = "POST"

    api_key = os.environ.get(api_key_env, "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.getcode(), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def _fetch_first_model_id_openai_compat(base_url: str, timeout_s: float = 30.0) -> Optional[str]:
    base_url = _normalize_openai_base_url(base_url)
    if not base_url:
        return None

    try:
        status, text = _request_openai_compat(f"{base_url}/models", timeout_s=timeout_s)
        if status != 200:
            return None
        data = json.loads(text)
        items = data.get("data", [])
        if isinstance(items, list) and items:
            model_id = _safe_str(items[0].get("id") if isinstance(items[0], dict) else "").strip()
            return model_id or None
    except Exception as exc:
        logger.debug(f"Failed to fetch teacher model id from {base_url}: {exc}")
    return None


def _resolve_teacher_model(base_url: str, sf_config: dict[str, Any]) -> str:
    reward_llm = sf_config.get("reward_llm") if isinstance(sf_config, dict) else {}
    reward_llm = reward_llm if isinstance(reward_llm, dict) else {}
    explicit_model = os.environ.get(REWARD_LLM_MODEL_ENV) or reward_llm.get("model")
    explicit_model = _safe_str(explicit_model).strip()
    if explicit_model:
        return explicit_model

    cache_key = _normalize_openai_base_url(base_url)
    if cache_key in TEACHER_MODEL_CACHE:
        return TEACHER_MODEL_CACHE[cache_key]

    model = _fetch_first_model_id_openai_compat(cache_key) or DEFAULT_TEACHER_MODEL
    TEACHER_MODEL_CACHE[cache_key] = model
    return model


def _build_teacher_feedback_prompt(problem: str, program: str, stdout: str, stderr: str) -> str:
    return (
        "You are a debugging assistant for a tool-call program that tries to solve or encode a "
        "logic/constraint problem.\n\n"
        "You will receive:\n"
        "1) Problem statement\n"
        "2) Candidate program\n"
        "3) Program output (stdout/stderr)\n\n"
        "Your job:\n"
        "- If there are bugs, propose concrete debugging steps.\n"
        "- If the encoding is incomplete or wrong, list missing constraints.\n"
        "- Only rewrite the entire program if it is very short (<= ~30 lines). Avoid delimiters "
        "```python```, and try writing the program in text if you decide to do so.\n\n"
        "Output requirements (STRICT):\n"
        "- Put your final feedback inside exactly one tag block:\n"
        "  <<<FEEDBACK>>>\n"
        "- The text inside <<<FEEDBACK>>> should be <= 300 tokens.\n"
        "- Do NOT include any additional <<<...>>> blocks.\n\n"
        f"=== PROBLEM STATEMENT ===\n{_safe_str(problem).strip()}\n\n"
        f"=== PROGRAM ===\n{_safe_str(program).strip()}\n\n"
        f"=== STDOUT ===\n{_safe_str(stdout)}\n\n"
        f"=== STDERR ===\n{_safe_str(stderr)}\n"
    )


def _extract_feedback_block(text: str) -> str:
    text = _safe_str(text).strip()
    if not text:
        return ""
    match = FEEDBACK_TAG_RE.search(text)
    if match:
        return _safe_str(match.group(1)).strip()
    return text


def _call_teacher_feedback_openai_compat(
    base_url: str,
    model: str,
    prompt: str,
    timeout_s: float = 120.0,
    max_retries: int = 2,
) -> str:
    base_url = _normalize_openai_base_url(base_url)
    if not base_url:
        return ""

    system_msg = (
        "You provide concise, actionable debugging feedback.\n"
        "You MUST output exactly one block: <<<FEEDBACK>>> ...\n"
        "No scores, no grading, no extra tag blocks."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 8192,
    }

    last_error = ""
    for _ in range(max_retries + 1):
        try:
            status, text = _request_openai_compat(f"{base_url}/chat/completions", payload, timeout_s=timeout_s)
            if status != 200:
                return f"(teacher http {status}) {text[:500]}"
            data = json.loads(text)
            content = _safe_str(data["choices"][0]["message"]["content"])
            return _extract_feedback_block(content)
        except Exception as exc:
            last_error = _safe_str(exc)
            logger.debug(f"Teacher feedback request failed: {exc}")

    logger.warning(f"Teacher feedback unavailable: {last_error}")
    return ""


def _strip_jemalloc_error(text: str) -> tuple[str, bool]:
    text = _safe_str(text)
    found = JEMALLOC_BACKGROUND_THREAD_ERROR in text
    if found:
        text = text.replace(JEMALLOC_BACKGROUND_THREAD_ERROR, "")
    return text.strip(), found


def get_rate_limiter_name() -> str:
    name = os.getenv(RATE_LIMITER_NAME_ENV)
    if name:
        return name
    job_id = _get_slurm_jobid()
    if job_id:
        return f"rate-limiter-{job_id}"
    return f"rate-limiter-{uuid4().hex}"


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count


@ray.remote
class ExecutionWorker:
    def __init__(self, enable_global_rate_limit: bool = True, rate_limit: int = 10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit: int):
        name = get_rate_limiter_name()
        try:
            return TokenBucketWorker.options(name=name, lifetime="detached", get_if_exists=True).remote(rate_limit)
        except TypeError:
            try:
                return ray.get_actor(name)
            except ValueError:
                try:
                    return TokenBucketWorker.options(name=name, lifetime="detached").remote(rate_limit)
                except TypeError:
                    return TokenBucketWorker.options(name=name).remote(rate_limit)

    def ping(self):
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        if self.rate_limit_worker is None:
            return fn(*fn_args, **fn_kwargs)

        acquired = False
        try:
            ray.get(self.rate_limit_worker.acquire.remote())
            acquired = True
            return fn(*fn_args, **fn_kwargs)
        except Exception as exc:
            logger.warning(f"Error when executing code: {exc}")
            raise
        finally:
            if acquired:
                try:
                    ray.get(self.rate_limit_worker.release.remote())
                except Exception as exc:
                    logger.warning(f"Error releasing rate limiter token: {exc}")


def init_execution_pool(
    num_workers: int,
    enable_global_rate_limit: bool = True,
    rate_limit: int = 10,
    mode: PoolMode = PoolMode.ThreadMode,
):
    if mode == PoolMode.ThreadMode:
        return ExecutionWorker.options(max_concurrency=num_workers).remote(
            enable_global_rate_limit=enable_global_rate_limit,
            rate_limit=rate_limit,
        )
    raise NotImplementedError("Process mode is not implemented yet")


class SandboxFusionTool(BaseTool):
    """A tool for executing code using a Sandbox Fusion backend."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.num_workers = config.get("num_workers", 10)
        self.rate_limit = config.get("rate_limit", 10)
        self.default_timeout = config.get("default_timeout", 30)
        self.default_language = config.get("default_language", "python")
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.log_path = config.get("log_path") or os.environ.get(SANDBOX_LOG_PATH_ENV)
        if self.log_path is None:
            self.log_path = os.path.join(_project_root(), "sf_log.txt")
        self.execution_pool = None
        self.sandbox_fusion_url = config.get("sandbox_fusion_url", "")
        self.memory_limit_mb = config.get("memory_limit_mb", 1024)
        if self.sandbox_fusion_url == "":
            raise ValueError("sandbox_fusion_url is not set")
        logger.info(f"Init SandboxFusionTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    def _get_execution_pool(self):
        if self.execution_pool is None:
            self.execution_pool = init_execution_pool(
                num_workers=self.num_workers,
                enable_global_rate_limit=self.enable_global_rate_limit,
                rate_limit=self.rate_limit,
                mode=PoolMode.ThreadMode,
            )
        return self.execution_pool

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {}) or {}
        if ground_truth is None:
            ground_truth = create_kwargs.get("ground_truth")

        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
            "enforce_symbolic": create_kwargs.get("enforce_symbolic", False),
            "data_source": create_kwargs.get("data_source", "openai/gsm8k"),
            "enforce_intermediate_checking": bool(
                create_kwargs.get("enforce_intermediate_checking", create_kwargs.get("feedback", False))
            ),
            "question": create_kwargs.get("question"),
        }
        return instance_id, ToolResponse()

    @staticmethod
    def _remove_fences(code: str) -> str:
        if not isinstance(code, str):
            code = str(code)

        fenced_blocks = FENCED_CODE_BLOCK_RE.findall(code)
        if fenced_blocks:
            return "\n\n".join(block.strip() for block in fenced_blocks if block.strip()).strip()

        code = re.sub(r"```[ \t]*[A-Za-z0-9_+-]*", "", code)
        return code.replace("```", "").strip()

    def _maybe_wrap_print(self, code: str) -> str:
        code = self._remove_fences(code)
        lines = code.rstrip().splitlines()
        if not lines:
            return code

        raw_last = lines[-1]
        last = raw_last.strip()
        if not last:
            return code

        non_expression_prefixes = (
            r"^(def|class|for|if|while|with|return|import|from|try|except|finally|async|await|"
            r"global|nonlocal|raise|assert|pass|break|continue)\b"
        )
        if re.match(non_expression_prefixes, last) or last.endswith(":") or re.match(r"^\s*print\s*\(", last):
            return code

        try:
            ast.parse(last, mode="eval")
        except SyntaxError:
            return code

        indent = raw_last[: len(raw_last) - len(raw_last.lstrip())]
        lines[-1] = f"{indent}print({last})"
        return "\n".join(lines)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        auto_strip_fences = parameters.get("auto_strip_fences", True)
        auto_print_last_expr = parameters.get("auto_print_last_expr", True)

        if not isinstance(code, str):
            code = str(code)

        if auto_strip_fences:
            code = self._remove_fences(code)
        if language == "python" and auto_print_last_expr:
            code = self._maybe_wrap_print(code)

        result = await self._get_execution_pool().execute.remote(
            self.execute_code, instance_id, code, timeout, language, True
        )
        if isinstance(result, tuple) and len(result) == 2:
            tool_response, execution_metrics = result
            if not isinstance(tool_response, ToolResponse):
                tool_response = ToolResponse(text=None if tool_response is None else str(tool_response))
            return tool_response, None, execution_metrics

        # Compatibility with custom/older execution pools that still return a
        # ToolResponse (or plain text) rather than the metadata-bearing tuple.
        if isinstance(result, ToolResponse):
            return result, None, None
        return ToolResponse(text=None if result is None else str(result)), None, None

    def _teacher_feedback(self, instance_id: str, code: str, stdout: str, stderr: str) -> str:
        state = self._instance_dict.get(instance_id, {})
        if not state.get("enforce_intermediate_checking", False):
            return ""

        problem = _safe_str(state.get("question")).strip()
        if not problem:
            return ""

        sf_config = _load_sf_yaml_config()
        reward_llm = sf_config.get("reward_llm") if isinstance(sf_config, dict) else {}
        reward_llm = reward_llm if isinstance(reward_llm, dict) else {}
        base_url = os.environ.get(REWARD_LLM_BASE_URL_ENV) or reward_llm.get("base_url")
        base_url = _normalize_openai_base_url(_safe_str(base_url))
        if not base_url:
            return ""

        try:
            model = _resolve_teacher_model(base_url, sf_config)
            prompt = _build_teacher_feedback_prompt(problem=problem, program=code, stdout=stdout, stderr=stderr)
            return _call_teacher_feedback_openai_compat(
                base_url=base_url,
                model=model,
                prompt=prompt,
                timeout_s=120.0,
                max_retries=2,
            )
        except Exception as exc:
            logger.warning(f"Teacher feedback failed: {exc}")
            return ""

    def _append_execution_log(self, instance_id: str, run_status: str, code: str, output: str) -> None:
        if not self.log_path:
            return

        entry = (
            f"---\n"
            f"Instance ID: {instance_id}\n"
            f"Run Status: {run_status}\n"
            f"Code:\n{code}\n"
            f"Output:\n{output}\n\n"
        )
        try:
            log_dir = os.path.dirname(os.path.abspath(self.log_path))
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as exc:
            logger.warning(f"Failed to write Sandbox Fusion log to {self.log_path}: {exc}")

    def execute_code(self, instance_id, code, timeout=30, language="python", return_metadata=False):
        result_status, metadata = _process_single_case(
            0, None, None, self.sandbox_fusion_url, f"{code}", timeout, self.memory_limit_mb, language
        )

        stdout, stdout_jemalloc = _strip_jemalloc_error(metadata.get("stdout"))
        stderr, stderr_jemalloc = _strip_jemalloc_error(metadata.get("stderr"))
        jemalloc_error = stdout_jemalloc or stderr_jemalloc
        run_status = _safe_str(metadata.get("run_status") or metadata.get("status") or "Unknown")

        if result_status in {-1, -2, -3, -4}:
            if not stderr:
                stderr = _safe_str(metadata.get("compile_stderr") or metadata.get("api_request_error"))
            if stderr:
                stderr = f"Error from SandboxFusion: {stderr.strip()}"

        if not stdout and not stderr:
            if run_status == "Finished":
                stderr = EMPTY_STDOUT_WARNING
            else:
                stderr = EMPTY_UNFINISHED_ERROR.format(run_status=run_status)

        response_parts = []
        if stdout:
            response_parts.append(f"stdout: {stdout}")
        if stderr:
            response_parts.append(f"stderr: {stderr}")

        teacher_feedback = self._teacher_feedback(instance_id, code, stdout, stderr)
        if teacher_feedback.strip():
            response_parts.append(f"teacher_feedback: {teacher_feedback.strip()}")

        if jemalloc_error:
            logger.debug(f"Suppressed Sandbox Fusion jemalloc background-thread warning for {instance_id}")

        response = "\n".join(response_parts) + "\n"
        self._append_execution_log(instance_id=instance_id, run_status=run_status, code=code, output=response)
        tool_response = ToolResponse(text=response)
        if not return_metadata:
            return tool_response

        return tool_response, {
            "execution_status": _build_execution_status(
                result_status=result_status,
                metadata=metadata,
                stdout=stdout,
                stderr=stderr,
            )
        }

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
