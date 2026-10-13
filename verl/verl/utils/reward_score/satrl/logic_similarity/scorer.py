"""Reward-time adapter for SATBench Boolean formula equivalence."""

from __future__ import annotations

import re
import json
import atexit
import contextlib
import io
import multiprocessing
import os
import selectors
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from .formula_extractor import extract_formula, extract_raw_formula
from .formula_similarity import FormulaEquivalenceResult, score_extracted_formulas


SUPPORTED_BACKENDS = ("z3", "ortools", "sympy", "pysmt", "pysat")
_IMPORT_ROOTS = {
    "z3": ("z3",),
    "ortools": ("ortools",),
    "sympy": ("sympy",),
    "pysmt": ("pysmt",),
    "pysat": ("pysat",),
}
_MAX_PROGRAM_CHARS = 20000
_EQUIVALENCE_LOG_PATH = os.environ.get("EQUIVALENCE_LOG_PATH")
_VERL_REPO_ROOT = str(Path(__file__).resolve().parents[5])
_DENIED_PROGRAM_PATTERNS = (
    (re.compile(r"\b(?:python|python3)\s+-m\s+pip\b", re.IGNORECASE), "python_m_pip"),
    (re.compile(r"\bpip(?:3)?\s+install\b", re.IGNORECASE), "pip_install"),
    (re.compile(r"\bsubprocess\b"), "subprocess"),
    (re.compile(r"\bos\s*\.\s*system\b"), "os_system"),
    (re.compile(r"\bos\s*\.\s*_exit\b"), "os__exit"),
    (re.compile(r"\bsys\s*\.\s*exit\b"), "sys_exit"),
    (re.compile(r"(?<![\w.])exit\s*\("), "exit"),
    (re.compile(r"(?<![\w.])quit\s*\("), "quit"),
    (re.compile(r"\b__import__\b"), "dunder_import"),
    (re.compile(r"\bsocket\b"), "socket"),
    (re.compile(r"\brequests\b"), "requests"),
)
_WORKER_LOCK = threading.Lock()
_WORKER: "_EquivalenceWorker | None" = None
_LOG_LOCK = threading.Lock()


def compute_logic_equivalent(
    program: str,
    extra_info: dict[str, Any] | None,
    *,
    backend: str | None = None,
    timeout_s: float | None = 20.0,
    solver_timeout_ms: int = 20000,
) -> dict[str, Any]:
    """Extract a proposed formula from ``program`` and compare it to SATBench GT.

    The returned ``logic_equivalent`` value is binary. Failures are represented
    as zero plus a short non-numeric ``error`` value; callers that log only
    numeric reward metrics can ignore that field.
    """
    program = _safe_str(program)
    deny_reason = _denied_program_reason(program)
    if deny_reason:
        result = _zero(f"denied:{deny_reason}")
        _write_equivalence_log(program, extra_info or {}, backend, result)
        return result

    payload = {
        "program": program,
        "extra_info": _jsonable(extra_info or {}),
        "backend": backend,
        "timeout_s": timeout_s,
        "solver_timeout_ms": solver_timeout_ms,
    }
    result = _compute_with_worker(payload, timeout_s)
    _write_equivalence_log(program, extra_info or {}, backend, result)
    return result


class _EquivalenceWorker:
    def __init__(self) -> None:
        child_code = (
            "import sys\n"
            f"sys.path.insert(0, {json.dumps(_VERL_REPO_ROOT)})\n"
            "from verl.utils.reward_score.satrl.logic_similarity.scorer import _worker_main\n"
            "_worker_main()\n"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            _VERL_REPO_ROOT
            if not env.get("PYTHONPATH")
            else _VERL_REPO_ROOT + os.pathsep + env["PYTHONPATH"]
        )
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-B", "-c", child_code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.selector = selectors.DefaultSelector()
        if self.proc.stdout is not None:
            self.selector.register(self.proc.stdout, selectors.EVENT_READ)

    def close(self) -> None:
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        try:
            self.selector.close()
        except Exception:
            pass
        if self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=0.5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def request(self, payload: dict[str, Any], timeout_s: float | None) -> dict[str, Any]:
        if self.proc.poll() is not None:
            return _zero(f"worker_exit:{self.proc.returncode}")
        if self.proc.stdin is None or self.proc.stdout is None:
            return _zero("worker_pipe_missing")

        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
        except Exception as exc:
            return _zero(f"worker_write:{type(exc).__name__}")

        events = self.selector.select(_worker_wall_timeout(timeout_s))
        if not events:
            self.close()
            return _zero("worker_timeout")

        line = self.proc.stdout.readline()
        if not line:
            return _zero(f"worker_exit:{self.proc.poll()}", logs=self._stderr_tail())
        try:
            result = json.loads(line)
        except Exception:
            return _zero("worker_bad_json")
        if isinstance(result, dict):
            return result
        return _zero("worker_bad_result")

    def _stderr_tail(self, max_chars: int = 4000) -> str:
        if self.proc.stderr is None:
            return ""
        if self.proc.poll() is None:
            return ""
        try:
            chunks = []
            while True:
                chunk = self.proc.stderr.readline()
                if not chunk:
                    break
                chunks.append(chunk)
            return "".join(chunks)[-max_chars:]
        except Exception:
            return ""


def _compute_with_worker(payload: dict[str, Any], timeout_s: float | None) -> dict[str, Any]:
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None or _WORKER.proc.poll() is not None:
            if _WORKER is not None:
                _WORKER.close()
            try:
                _WORKER = _EquivalenceWorker()
            except Exception as exc:
                _WORKER = None
                return _zero(f"worker_launch:{type(exc).__name__}")

        result = _WORKER.request(payload, timeout_s)
        if result.get("error", "").startswith(("worker_timeout", "worker_exit", "worker_write")):
            _WORKER.close()
            _WORKER = None
        return result


def _close_worker_at_exit() -> None:
    global _WORKER
    if _WORKER is not None:
        _WORKER.close()
        _WORKER = None


atexit.register(_close_worker_at_exit)


def _worker_main() -> None:
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            result = _compute_request_in_child(payload, payload.get("timeout_s"))
        except BaseException as exc:
            result = _zero(
                f"worker_exception:{type(exc).__name__}",
                logs="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            )
        try:
            print(json.dumps(_jsonable_result(result)), flush=True)
        except BaseException as exc:
            fallback = _zero(
                f"worker_serialize:{type(exc).__name__}",
                logs="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            )
            print(json.dumps(fallback), flush=True)


def _compute_request_in_child(payload: dict[str, Any], timeout_s: float | None) -> dict[str, Any]:
    """Run one equivalence request in a disposable child process.

    The persistent worker keeps imports and IPC alive. Generated/model code is
    executed only in this request child, so sys.exit/os._exit/segfaults/timeouts
    cannot poison later requests.
    """

    ctx_name = "fork" if "fork" in multiprocessing.get_all_start_methods() else multiprocessing.get_start_method()
    ctx = multiprocessing.get_context(ctx_name)
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_request_child_main, args=(child_conn, payload))
    proc.daemon = True
    proc.start()
    child_conn.close()

    wall_timeout = _request_child_wall_timeout(timeout_s)
    try:
        if parent_conn.poll(wall_timeout):
            try:
                result = parent_conn.recv()
            except EOFError:
                result = _zero(f"request_exit:{proc.exitcode}", logs="request child closed pipe without a result")
        else:
            _terminate_process(proc)
            result = _zero("request_timeout", timed_out=1.0, logs=f"request child exceeded {wall_timeout:.3f}s")
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass

    proc.join(timeout=0.2)
    if proc.exitcode not in (0, None) and not result.get("error"):
        result = _zero(f"request_exit:{proc.exitcode}", logs="request child exited non-zero after sending no error")
    if proc.exitcode is None:
        _terminate_process(proc)

    return _jsonable_result(result)


def _request_child_main(conn: Any, payload: dict[str, Any]) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        restore_exit = _patch_process_exit()
        old_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory(prefix="satrl_formula_") as tmp_dir:
                os.chdir(tmp_dir)
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    result = _compute_logic_equivalent_inline(**payload)
        finally:
            try:
                os.chdir(old_cwd)
            except Exception:
                pass
            restore_exit()
        if not result.get("logs"):
            captured = (stdout.getvalue() + stderr.getvalue()).strip()
            if captured:
                result["logs"] = captured[-4000:]
    except BaseException as exc:
        result = _zero(
            f"request_exception:{type(exc).__name__}",
            logs=(
                stdout.getvalue()
                + stderr.getvalue()
                + "\n"
                + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            )[-4000:],
        )
    try:
        conn.send(_jsonable_result(result))
    except BaseException:
        pass
    finally:
        try:
            conn.close()
        except BaseException:
            pass


def _patch_process_exit() -> Any:
    import builtins

    def blocked_exit(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("process exit is blocked during formula extraction")

    original_sys_exit = sys.exit
    original_builtins_exit = getattr(builtins, "exit", None)
    original_builtins_quit = getattr(builtins, "quit", None)
    original_os_exit = getattr(os, "_exit", None)

    sys.exit = blocked_exit
    builtins.exit = blocked_exit
    builtins.quit = blocked_exit
    try:
        os._exit = blocked_exit
    except Exception:
        pass

    def restore() -> None:
        sys.exit = original_sys_exit
        if original_builtins_exit is not None:
            builtins.exit = original_builtins_exit
        if original_builtins_quit is not None:
            builtins.quit = original_builtins_quit
        if original_os_exit is not None:
            os._exit = original_os_exit

    return restore


def _terminate_process(proc: multiprocessing.Process) -> None:
    if proc.exitcode is not None:
        return
    try:
        proc.terminate()
        proc.join(timeout=0.5)
    except Exception:
        pass
    if proc.exitcode is None:
        try:
            proc.kill()
            proc.join(timeout=0.5)
        except Exception:
            pass


def _request_child_wall_timeout(timeout_s: float | None) -> float:
    if timeout_s is None:
        return 20.0
    try:
        return max(3.0, float(timeout_s) + 2.0)
    except Exception:
        return 4.0


def _worker_wall_timeout(timeout_s: float | None) -> float:
    if timeout_s is None:
        return 20.0
    try:
        return max(20.0, float(timeout_s) + 2.0)
    except Exception:
        return 20.0



def _compute_logic_equivalent_inline(
    program: str,
    extra_info: dict[str, Any] | None,
    *,
    backend: str | None = None,
    timeout_s: float | None = 20.0,
    solver_timeout_ms: int = 20000,
) -> dict[str, Any]:
    """Implementation body run inside the persistent scorer worker."""

    program = _safe_str(program).strip()
    if not program:
        return _zero("empty_program")

    extra_info = extra_info or {}
    dims = _coerce_int_list(extra_info.get("dims"))
    if not dims:
        return _zero("missing_dims")

    candidates = _candidate_backends(program, backend or extra_info.get("sampled_tool"))
    if not candidates:
        return _zero("missing_backend")

    last_error = "extraction_failed"
    last_gt_formula = ""
    last_extracted_formula = ""
    last_extracted_assertion_count = 0
    last_extracted_variable_count = 0
    equivalence_check_time = 0.0
    attempt_logs: list[str] = []
    for candidate in candidates:
        try:
            gt_formula, gt_style = _ground_truth_formula(extra_info, dims, candidate)
            last_gt_formula = gt_formula
            pred_style = "one" if candidate == "pysat" else "multi"
            gt = extract_raw_formula(gt_formula)
            pred = extract_formula(program, backend=candidate, run_check=False, quiet=True)
            extracted_formula = _extracted_formula_text(pred)
            extracted_assertions = tuple(getattr(pred, "assertions", ()))
            extracted_variable_count = len(_formula_variables(extracted_assertions))
            last_extracted_formula = extracted_formula
            last_extracted_assertion_count = len(extracted_assertions)
            last_extracted_variable_count = extracted_variable_count
            equivalence_started_at = time.monotonic()
            try:
                result = score_extracted_formulas(
                    gt,
                    pred,
                    dims=dims,
                    gt_var_style=gt_style,
                    pred_var_style=pred_style,
                    backend=candidate,
                    timeout_s=timeout_s,
                    solver_timeout_ms=solver_timeout_ms,
                )
            finally:
                equivalence_check_time += time.monotonic() - equivalence_started_at
            return _result_dict(
                result,
                ground_truth_formula=gt_formula,
                extracted_formula=extracted_formula,
                extracted_assertion_count=len(extracted_assertions),
                extracted_variable_count=extracted_variable_count,
                equivalence_check_time=equivalence_check_time,
                logs="; ".join(attempt_logs),
            )
        except Exception as exc:
            last_error = f"{candidate}:{type(exc).__name__}"
            attempt_logs.append(last_error)

    return _zero(
        last_error,
        ground_truth_formula=last_gt_formula,
        extracted_formula=last_extracted_formula,
        extracted_assertion_count=last_extracted_assertion_count,
        extracted_variable_count=last_extracted_variable_count,
        equivalence_check_time=equivalence_check_time,
        logs="; ".join(attempt_logs),
        backend=candidates[-1] if candidates else None,
    )


def _extracted_formula_text(formula: Any) -> str:
    try:
        assertions = getattr(formula, "assertions", ())
        if not assertions:
            return "true"
        if len(assertions) == 1:
            return _safe_str(assertions[0])
        return " & ".join(f"({assertion})" for assertion in assertions)
    except Exception as exc:
        return f"<formula_render_error:{type(exc).__name__}>"


def _formula_variables(formulas: Sequence[Any]) -> set[str]:
    variables: set[str] = set()

    def visit(formula: Any) -> None:
        if getattr(formula, "op", None) == "var":
            name = _safe_str(getattr(formula, "name", ""))
            if name:
                variables.add(name)
            return
        for child in getattr(formula, "args", ()) or ():
            visit(child)

    for formula in formulas:
        visit(formula)
    return variables


def _write_equivalence_log(
    program: str,
    extra_info: dict[str, Any] | None,
    requested_backend: Any,
    result: dict[str, Any],
) -> None:
    if not _EQUIVALENCE_LOG_PATH:
        return

    extra_info = extra_info or {}
    ground_truth_formula = _safe_str(result.get("ground_truth_formula"))
    if not ground_truth_formula:
        ground_truth_formula = _best_effort_ground_truth_formula(extra_info, requested_backend, result.get("backend"))

    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "requested_backend": _safe_str(requested_backend or extra_info.get("sampled_tool")),
        "backend": _safe_str(result.get("backend")),
        "logic_equivalent": _clamp01(result.get("logic_equivalent", result.get("score"))),
        "error": _safe_str(result.get("error")),
        "logs": _safe_str(result.get("logs")),
        "ground_truth_formula": ground_truth_formula,
        "extracted_formula": _safe_str(result.get("extracted_formula")),
        "extracted_assertion_count": float(result.get("extracted_assertion_count", 0.0)),
        "extracted_variable_count": float(result.get("extracted_variable_count", 0.0)),
        "equivalence_check_time": float(result.get("equivalence_check_time", 0.0)),
        "program": _safe_str(program),
    }
    line = json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"

    try:
        os.makedirs(os.path.dirname(_EQUIVALENCE_LOG_PATH), exist_ok=True)
        with _LOG_LOCK:
            with open(_EQUIVALENCE_LOG_PATH, "a", encoding="utf-8") as log_file:
                try:
                    import fcntl

                    fcntl.flock(log_file.fileno(), fcntl.LOCK_EX)
                except Exception:
                    pass
                try:
                    log_file.write(line)
                    log_file.flush()
                finally:
                    try:
                        import fcntl

                        fcntl.flock(log_file.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
    except Exception:
        pass


def _best_effort_ground_truth_formula(extra_info: dict[str, Any], *backend_values: Any) -> str:
    dims = _coerce_int_list(extra_info.get("dims"))
    if not dims:
        return ""

    backends: list[str] = []
    for value in backend_values + (extra_info.get("sampled_tool"),):
        backend = _normalize_backend(value)
        if backend and backend not in backends:
            backends.append(backend)
    for backend in SUPPORTED_BACKENDS:
        if backend not in backends:
            backends.append(backend)

    for backend in backends:
        try:
            formula, _ = _ground_truth_formula(extra_info, dims, backend)
            if formula:
                return formula
        except Exception:
            pass
    return ""


def _denied_program_reason(program: str) -> str | None:
    if len(_safe_str(program)) > _MAX_PROGRAM_CHARS:
        return "program_too_long"
    for pattern, reason in _DENIED_PROGRAM_PATTERNS:
        if pattern.search(program):
            return reason
    return None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        return {_safe_str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _safe_str(value)


def _jsonable_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return _zero("bad_result_type", logs=f"result type: {type(result).__name__}")
    return {str(key): _jsonable(value) for key, value in result.items()}


def _result_dict(
    result: FormulaEquivalenceResult,
    *,
    ground_truth_formula: str = "",
    extracted_formula: str = "",
    extracted_assertion_count: int = 0,
    extracted_variable_count: int = 0,
    equivalence_check_time: float = 0.0,
    logs: str = "",
) -> dict[str, Any]:
    score = _clamp01(result.score)
    return {
        "score": score,
        "logic_equivalent": score,
        "equivalent": bool(result.equivalent),
        "timed_out": float(bool(result.timed_out)),
        "unknown_checks": float(result.unknown_checks),
        "gt_clause_count": float(result.gt_clause_count),
        "pred_clause_count": float(result.pred_clause_count),
        "error": "",
        "backend": _safe_str(result.backend),
        "ground_truth_formula": _safe_str(ground_truth_formula),
        "extracted_formula": _safe_str(extracted_formula),
        "extracted_assertion_count": float(extracted_assertion_count),
        "extracted_variable_count": float(extracted_variable_count),
        "equivalence_check_time": float(equivalence_check_time),
        "logs": _safe_str(logs),
    }


def _zero(
    error: str,
    *,
    timed_out: float = 0.0,
    ground_truth_formula: str = "",
    extracted_formula: str = "",
    extracted_assertion_count: int = 0,
    extracted_variable_count: int = 0,
    equivalence_check_time: float = 0.0,
    logs: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    return {
        "score": 0.0,
        "logic_equivalent": 0.0,
        "equivalent": False,
        "timed_out": float(timed_out),
        "unknown_checks": 0.0,
        "gt_clause_count": 0.0,
        "pred_clause_count": 0.0,
        "error": error,
        "backend": _safe_str(backend),
        "ground_truth_formula": _safe_str(ground_truth_formula),
        "extracted_formula": _safe_str(extracted_formula),
        "extracted_assertion_count": float(extracted_assertion_count),
        "extracted_variable_count": float(extracted_variable_count),
        "equivalence_check_time": float(equivalence_check_time),
        "logs": _safe_str(error if logs is None else logs),
    }


def _candidate_backends(program: str, preferred: Any) -> list[str]:
    ordered: list[str] = []

    preferred_backend = _normalize_backend(preferred)
    if preferred_backend:
        ordered.append(preferred_backend)

    for backend in SUPPORTED_BACKENDS:
        if backend in ordered:
            continue
        if _program_imports_backend(program, backend):
            ordered.append(backend)

    return ordered


def _normalize_backend(value: Any) -> str | None:
    backend = _safe_str(value).strip().lower()
    return backend if backend in SUPPORTED_BACKENDS else None


def _program_imports_backend(program: str, backend: str) -> bool:
    roots = _IMPORT_ROOTS.get(backend, ())
    if not roots:
        return False
    roots_pat = "|".join(re.escape(root) for root in roots)
    return bool(
        re.search(
            rf"(?:^|[\n\r])\s*(?:import\s+(?:[\w\.]+\s*,\s*)*({roots_pat})(?:\.[\w\.]+)?\b|from\s+({roots_pat})(?:\.[\w\.]+)?\s+import\b)",
            program,
            flags=re.IGNORECASE,
        )
    )


def _ground_truth_formula(extra_info: dict[str, Any], dims: Sequence[int], backend: str) -> tuple[str, str]:
    if backend == "pysat":
        formula = _safe_str(extra_info.get("one_dimension_formula")).strip()
        if formula:
            return formula, "one"
        clauses = _coerce_clauses(extra_info.get("raw_formula"))
        if clauses:
            return _clauses_to_formula(clauses, dims, "one"), "one"

    formula = _safe_str(extra_info.get("multi_dimension_formula")).strip()
    if formula:
        return formula, "multi"

    formula = _safe_str(extra_info.get("readable_formula")).strip()
    if formula:
        return formula, "multi"

    clauses = _coerce_clauses(extra_info.get("raw_formula"))
    if clauses:
        return _clauses_to_formula(clauses, dims, "multi"), "multi"

    raise ValueError("missing ground-truth formula")


def _clauses_to_formula(clauses: Sequence[Sequence[int]], dims: Sequence[int], style: str) -> str:
    if not clauses:
        return "true"

    rendered_clauses = []
    for clause in clauses:
        literals = [_literal_to_formula(int(literal), dims, style) for literal in clause]
        if not literals:
            rendered_clauses.append("false")
        elif len(literals) == 1:
            rendered_clauses.append(literals[0])
        else:
            rendered_clauses.append("(" + " | ".join(literals) + ")")
    return " & ".join(rendered_clauses)


def _literal_to_formula(literal: int, dims: Sequence[int], style: str) -> str:
    name = _var_name(abs(int(literal)), dims, style)
    return f"!{name}" if literal < 0 else name


def _var_name(var_id: int, dims: Sequence[int], style: str) -> str:
    if style == "one":
        return f"x_{int(var_id)}"

    indices = _var_id_to_indices(var_id, dims)
    if len(indices) == 1:
        return f"x({indices[0]},)"
    return "x(" + ", ".join(str(index) for index in indices) + ")"


def _var_id_to_indices(var_id: int, dims: Sequence[int]) -> list[int]:
    offset = int(var_id) - 1
    out: list[int] = []
    for axis, dim in enumerate(dims):
        stride = 1
        for later_dim in dims[axis + 1 :]:
            stride *= int(later_dim)
        out.append(offset // stride)
        offset %= stride
    return out


def _coerce_int_list(value: Any) -> list[int]:
    value = _maybe_to_list(value)
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return []
    try:
        return [int(item) for item in value]
    except Exception:
        return []


def _coerce_clauses(value: Any) -> list[list[int]]:
    value = _maybe_to_list(value)
    if value is None or isinstance(value, (str, bytes)):
        return []

    clauses: list[list[int]] = []
    try:
        for raw_clause in value:
            raw_clause = _maybe_to_list(raw_clause)
            if raw_clause is None or isinstance(raw_clause, (str, bytes)):
                return []
            clauses.append([int(literal) for literal in raw_clause])
    except Exception:
        return []
    return clauses


def _maybe_to_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        return value if isinstance(value, str) else str(value)
    except Exception:
        return ""


def _clamp01(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
