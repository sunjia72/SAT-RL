import importlib
import json
import re
from typing import Any, Optional

from verl.utils.reward_score.satrl.logic_similarity import compute_logic_equivalent

# ============================
# Helpers
# ============================

def _safe_str(s) -> str:
    if s is None:
        return ""
    try:
        return s if isinstance(s, str) else str(s)
    except Exception:
        return ""

def _is_empty_like(x) -> bool:
    if x is None:
        return True
    try:
        if isinstance(x, str):
            return len(x.strip()) == 0
        return len(x) == 0
    except Exception:
        return False

# ============================
# Extractor utilities (answers / code blocks)
# ============================

def normalize_mc_token(text) -> Optional[str]:
    if text is None:
        return None
    s = _safe_str(text).strip()
    if not s:
        return None
    m = re.search(r"^[\(\[\{]?\s*([A-Za-z]|[0-9]+)\s*[\)\]\}]?$", s)
    if not m:
        m = re.search(r"[\(\[\{]?\s*([A-Za-z]|[0-9]+)\s*[\)\]\}]?", s)
        if not m:
            return None
    token = _safe_str(m.group(1))
    if re.fullmatch(r"[A-Za-z]", token):
        return token.lower()
    return str(int(token)) if token.isdigit() else token

_LAST_FENCE_RX = re.compile(
    r"```(?:[\w.+-]+[ \t]*\n)?\s*(.*)```(?!.*```)",
    re.DOTALL
)

def _last_code_block_content(text: str) -> Optional[str]:
    """
    Return content of code block that ends at the *last* triple-backtick.
    """
    s = _safe_str(text)
    if not s:
        return None

    m = _LAST_FENCE_RX.search(s)
    if m:
        content = _safe_str(m.group(1)).strip()
        return content if content else None

    matches = list(re.finditer(r"```(?:[\w.+-]+[ \t]*\n)?\s*(.*?)```", s, re.DOTALL))
    if not matches:
        return None
    content = _safe_str(matches[-1].group(1)).strip()
    return content if content else None

def _last_nonempty_line(s: str) -> Optional[str]:
    s = _safe_str(s)
    for line in reversed(s.splitlines()):
        t = line.strip()
        if t:
            return t
    return None

def extract_solution(solution_str, mode: str = "word") -> Optional[str]:
    """
    IMPORTANT: This function must stay synchronous. Do NOT put any `await` here.
    """
    solution_str = _safe_str(solution_str)
    assert mode in {"mc", "word", "all"}

    search_space = _last_code_block_content(solution_str)
    if search_space is None:
        return None

    if mode == "all":
        return search_space if search_space else None

    line = _last_nonempty_line(search_space)
    if mode == "mc":
        return normalize_mc_token(line)

    # mode == "word"
    if not line:
        return None
    m = re.search(r"([A-Za-z0-9\.\,\-\$]+)\s*$", line)
    if not m:
        return None
    final_answer = m.group(1).replace(",", "").replace("$", "").strip()
    return final_answer.lower() if final_answer else None

# ============================
# Verifier loaders
# ============================

_VERIFIER_MODULE_ALIASES = {
    "folio": "FOLIO",
    "crypto_kka": "crypto_KKA",
    "crypto_kpa": "crypto_KPA",
}


def _to_module_name(data_source: Optional[str]) -> Optional[str]:
    s = _safe_str(data_source).strip()
    if not s:
        return None
    s = s.lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s or None

def _load_verifier(data_source: Optional[str]):
    module_suffix = _to_module_name(data_source)
    if not module_suffix:
        return None, None, None, "empty data_source"
    module_suffix = _VERIFIER_MODULE_ALIASES.get(module_suffix, module_suffix)
    base_pkg = "verl.utils.reward_score.satrl.verifiers"
    fqmn = f"{base_pkg}.{module_suffix}"
    try:
        mod = importlib.import_module(fqmn)
    except ModuleNotFoundError as e:
        return None, None, None, f"module not found: {fqmn} ({e})"
    except Exception as e:
        return None, None, None, f"failed to import {fqmn}: {e}"

    verify_fn = getattr(mod, "verify", None)
    if not callable(verify_fn):
        verify_fn = None

    extractor_fn = None
    for name in ("extract_final_answer", "extract_code_block", "extract_last_code_block"):
        cand = getattr(mod, name, None)
        if callable(cand):
            extractor_fn = cand
            break

    return mod, verify_fn, extractor_fn, None

def _load_synth_verifier(puzzle: Optional[str]):
    puzzle = _safe_str(puzzle).strip().lower()
    if not puzzle:
        return None, None, "empty puzzle"
    fqmn = f"verl.utils.reward_score.satrl.synthetic_puzzle_verifiers.{puzzle}"
    try:
        mod = importlib.import_module(fqmn)
    except ModuleNotFoundError as e:
        return None, None, f"module not found: {fqmn} ({e})"
    except Exception as e:
        return None, None, f"failed to import {fqmn}: {e}"
    verify_fn = getattr(mod, "verify", None)
    if not callable(verify_fn):
        return mod, None, "no verify()"
    return mod, verify_fn, None

# ============================
# Correctness helpers
# ============================

def _wrong_penalty(n: int, R_correct: float, R_honest: float, eps: float, alpha: float) -> float:
    if n is None:
        n = 0
    if n >= 2:
        bound = (n * R_honest - R_correct) / (n - 1) - eps
        return min(-alpha, bound)
    return -alpha

def compute_score_correctness(is_correct, said_idk, correctness_score, idk_score, incorrectness_score=0.0):
    if is_correct:
        return {
            "adjusted_correctness_score": correctness_score,
            "correctness": 1.0,
            "idk": 0.0,
            "incorrectness": 0.0,
        }
    elif said_idk:
        return {
            "adjusted_correctness_score": idk_score,
            "correctness": 0.0,
            "idk": 1.0,
            "incorrectness": 0.0,
        }
    else:
        return {
            "adjusted_correctness_score": incorrectness_score,
            "correctness": 0.0,
            "idk": 0.0,
            "incorrectness": 1.0,
        }

# ============================
# Unified tool-call parsing (single source of truth)
# ============================

_TOOL_CALL_RX = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_RESP_RX = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_XML_CODE_PARAM_RX = re.compile(
    r"<parameter\s*(?:=\s*code|[-_]\s*code|\s+code)\s*>\s*(.*?)\s*</parameter\s*>",
    re.DOTALL | re.IGNORECASE,
)

def _extract_last_tool_call_and_response(solution_str: str) -> tuple[Optional[str], Optional[str]]:
    """
    Returns:
      (last_tool_call_text, first_tool_response_text_after_last_call)
    """
    s = _safe_str(solution_str)
    if not s:
        return None, None
    tool_calls = list(_TOOL_CALL_RX.finditer(s))
    if not tool_calls:
        return None, None
    last_call = tool_calls[-1]
    call_text = _safe_str(last_call.group(1)).strip() or None

    rest = s[last_call.end():]
    m = _TOOL_RESP_RX.search(rest)
    resp_text = _safe_str(m.group(1)).strip() if m else None
    return call_text, (resp_text or None)

def _extract_code_from_tool_call_text(tool_call_text: str) -> Optional[str]:
    """
    Extract the actual program string from supported <tool_call> formats.

    Supported formats:
      - JSON/OpenAI style with an arguments.code field
      - Qwen/Hermes XML style with <parameter=code>...</parameter>
      - Common malformed XML variants like <parameter-code>...</parameter>
    """
    raw = _safe_str(tool_call_text).strip()
    if not raw:
        return None

    m = _XML_CODE_PARAM_RX.search(raw)
    if m:
        code = _safe_str(m.group(1)).strip()
        if code:
            return code

    def _try_json_load(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    obj = _try_json_load(raw)
    if not isinstance(obj, dict):
        return None

    args = obj.get("arguments", None)
    if isinstance(args, dict):
        code = _safe_str(args.get("code", None))
        return code.strip() or None

    if isinstance(args, str):
        args_obj = _try_json_load(args.strip())
        if isinstance(args_obj, dict):
            code = _safe_str(args_obj.get("code", None))
            return code.strip() or None

    fn = obj.get("function", None)
    if isinstance(fn, dict):
        fn_args = fn.get("arguments", None)
        if isinstance(fn_args, dict):
            code = _safe_str(fn_args.get("code", None))
            return code.strip() or None
        if isinstance(fn_args, str):
            fn_args_obj = _try_json_load(fn_args.strip())
            if isinstance(fn_args_obj, dict):
                code = _safe_str(fn_args_obj.get("code", None))
                return code.strip() or None

    return None

def _extract_code_from_openai_tool_call_text(tool_call_text: str) -> Optional[str]:
    return _extract_code_from_tool_call_text(tool_call_text)

def _tool_response_ok(resp_text: Optional[str]) -> bool:
    """
    Backward-compatible success heuristic for saved tool-response text.

    New rollouts should provide structured ``tool_execution_records`` in
    ``extra_info``. This text check remains necessary when rescoring older
    rollouts that predate those records.
    """
    s = _safe_str(resp_text).strip()
    if not s:
        return False
    folded = s.casefold()
    bad_markers = (
        "error from sandboxfusion:",
        "[toolwarning]",
        "[toolerror]",
        "this program has error",
        "no stdout",
        "stderr",
    )
    return not (
        folded.startswith("error when executing tool:")
        or any(marker in folded for marker in bad_markers)
    )


_TOOL_EXECUTION_RECORDS_KEY = "tool_execution_records"
_TOOL_ERROR_TYPES = (
    "invalid_arguments",
    "invalid_tool_name",
    "tool_creation_error",
    "tool_execution_error",
    "tool_release_error",
    "backend_execution_error",
    "empty_output",
)
_BACKEND_ERROR_TYPES = {
    "sandbox_api_error",
    "sandbox_error",
    "compile_timeout",
    "compile_error",
    "execution_timeout",
    "runtime_error",
    "backend_execution_error",
}


def _optional_bool(value: Any) -> Optional[bool]:
    """Coerce common JSON-safe bool representations without guessing."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "success", "ok"}:
            return True
        if normalized in {"false", "0", "no", "error", "failed", "failure"}:
            return False
    return None


def _tool_execution_records(extra_info: Optional[dict]) -> list[dict[str, Any]]:
    """Return ordered, JSON-safe per-call execution records if available."""
    if not isinstance(extra_info, dict):
        return []
    raw = extra_info.get(_TOOL_EXECUTION_RECORDS_KEY)
    if raw is None:
        return []

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    elif not isinstance(raw, (list, tuple, dict)) and hasattr(raw, "tolist"):
        try:
            raw = raw.tolist()
        except Exception:
            return []

    if isinstance(raw, dict):
        nested = raw.get("records")
        raw = nested if isinstance(nested, (list, tuple)) else [raw]
    if not isinstance(raw, (list, tuple)):
        return []

    # Preserve positions so a malformed record cannot shift the status of all
    # subsequent calls. An empty record falls back to response-text detection.
    records = [record if isinstance(record, dict) else {} for record in raw]

    # Parallel calls append records when execution finishes, whereas rendered
    # tool responses retain call order. Restore that order when every record
    # carries the lifecycle indices; leave legacy/unindexed lists untouched.
    try:
        has_lifecycle_indices = records and all(
            record.get("turn_index") is not None and record.get("call_index") is not None for record in records
        )
        if has_lifecycle_indices:
            records.sort(key=lambda record: (int(record["turn_index"]), int(record["call_index"])))
    except (TypeError, ValueError):
        pass
    return records


def _record_field(record: dict[str, Any], key: str) -> Any:
    """Read a promoted status field, with nested tool-metric compatibility."""
    if key in record and record[key] is not None:
        return record[key]
    tool_metrics = record.get("tool_metrics")
    if not isinstance(tool_metrics, dict):
        return None
    execution_status = tool_metrics.get("execution_status")
    if isinstance(execution_status, dict):
        return execution_status.get(key)
    return None


def _structured_tool_response_ok(record: dict[str, Any]) -> Optional[bool]:
    """Return authoritative structured success, or ``None`` if unavailable."""
    if not record:
        return None

    execution_succeeded = _optional_bool(_record_field(record, "execution_succeeded"))
    output_usable = _optional_bool(_record_field(record, "output_usable"))
    if execution_succeeded is False or output_usable is False:
        return False
    if execution_succeeded is True and output_usable is True:
        return True

    # A release failure is operational telemetry, not evidence that the
    # already-rendered tool output was unusable. With no authoritative execute
    # booleans, fall back to the visible response while retaining the release
    # error category.
    stage = _safe_str(_record_field(record, "stage")).strip().casefold()
    error_type = _safe_str(_record_field(record, "error_type")).strip().casefold()
    if stage == "release" or error_type == "tool_release_error":
        return None

    status = _safe_str(_record_field(record, "status")).strip().casefold()
    if status in {"error", "failed", "failure"}:
        return False
    if status in {"success", "ok", "completed"}:
        return True
    if _safe_str(_record_field(record, "error_type")).strip():
        return False
    return None


def _text_tool_error_type(resp_text: Optional[str]) -> str:
    """Best-effort failure category for legacy response-only rollouts."""
    folded = _safe_str(resp_text).strip().casefold()
    if not folded or "no stdout" in folded:
        return "empty_output"
    if "error from sandboxfusion:" in folded or "stderr" in folded or "this program has error" in folded:
        return "backend_execution_error"
    if folded.startswith("error when executing tool:") or "[toolerror]" in folded or "[toolwarning]" in folded:
        return "tool_execution_error"
    return "unclassified"


def _structured_tool_error_type(record: dict[str, Any]) -> str:
    error_type = _safe_str(_record_field(record, "error_type")).strip().casefold()
    if error_type in _TOOL_ERROR_TYPES:
        return error_type
    if error_type in _BACKEND_ERROR_TYPES:
        return "backend_execution_error"

    stage = _safe_str(_record_field(record, "stage")).strip().casefold()
    return {
        "parse_arguments": "invalid_arguments",
        "lookup_tool": "invalid_tool_name",
        "create": "tool_creation_error",
        "execute": "tool_execution_error",
        "release": "tool_release_error",
    }.get(stage, "unclassified")


def _extract_response_backed_tool_events(solution_str: str) -> list[tuple[str, str]]:
    """Return call/response pairs in observed response order."""
    calls = _extract_response_backed_tool_call_texts(solution_str)
    responses = [_safe_str(match.group(1)).strip() for match in _TOOL_RESP_RX.finditer(_safe_str(solution_str))]
    return list(zip(calls, responses, strict=True))


def _tool_execution_outcomes(solution_str: str, extra_info: Optional[dict] = None) -> list[dict[str, Any]]:
    """Resolve visible tool responses using structured status, then text fallback."""
    records = _tool_execution_records(extra_info)
    outcomes: list[dict[str, Any]] = []
    for index, (call_text, resp_text) in enumerate(_extract_response_backed_tool_events(solution_str)):
        record = records[index] if index < len(records) else {}
        response_ok = _structured_tool_response_ok(record)
        if response_ok is None:
            response_ok = _tool_response_ok(resp_text)

        record_status = _safe_str(_record_field(record, "status")).strip().casefold()
        record_error_type = _safe_str(_record_field(record, "error_type")).strip()
        has_structured_error = bool(record_error_type) or record_status in {"error", "failed", "failure"}
        error_type = ""
        if record and (has_structured_error or not response_ok):
            error_type = _structured_tool_error_type(record)
        elif not response_ok:
            error_type = _text_tool_error_type(resp_text)
        outcomes.append(
            {
                "call_text": call_text,
                "resp_text": resp_text,
                "response_ok": bool(response_ok),
                "error_type": error_type or ("unclassified" if not response_ok else ""),
            }
        )
    return outcomes


def _count_tool_responses(solution_str: str, extra_info: Optional[dict] = None) -> tuple[int, int]:
    """
    Return (total_tool_responses, failed_tool_responses).
    """
    outcomes = _tool_execution_outcomes(solution_str, extra_info)
    failed = sum(1 for outcome in outcomes if not outcome["response_ok"])
    return len(outcomes), failed


def _extract_response_backed_tool_call_texts(solution_str: str) -> list[str]:
    """Return one call-source entry for each observed tool response.

    Calls and responses are paired in emission order. This avoids counting an
    extra call tag that the rollout parser did not execute (for example, when
    the model emits more calls than ``max_parallel_calls`` permits). If the
    saved rollout contains a response without a recoverable call tag, an empty
    entry is returned so the execution is counted as unclassified.
    """
    s = _safe_str(solution_str)
    events = [
        *((match.start(), "call", match) for match in _TOOL_CALL_RX.finditer(s)),
        *((match.start(), "response", match) for match in _TOOL_RESP_RX.finditer(s)),
    ]
    events.sort(key=lambda event: event[0])

    pending_calls: list[str] = []
    response_seen_in_group = False
    response_backed_calls: list[str] = []

    for _, event_type, match in events:
        if event_type == "call":
            # A new assistant tool-call group begins after responses. Any
            # leftover calls from the previous group were not executed.
            if response_seen_in_group:
                pending_calls.clear()
                response_seen_in_group = False
            pending_calls.append(_safe_str(match.group(1)).strip())
            continue

        response_seen_in_group = True
        if pending_calls:
            response_backed_calls.append(pending_calls.pop(0))
        else:
            response_backed_calls.append("")

    return response_backed_calls


def _get_last_tool_context(solution_str: str) -> dict[str, Optional[str]]:
    """
    One function used everywhere:
      - last_tool_call_text
      - last_tool_response_text
      - last_program (best-effort)
      - has_tool_call
      - response_ok
    """
    call_text, resp_text = _extract_last_tool_call_and_response(solution_str)
    program = _extract_code_from_tool_call_text(call_text or "")
    if not program:
        program = _safe_str(call_text).strip() or None

    has_tool_call = bool(_safe_str(call_text).strip())
    response_ok = _tool_response_ok(resp_text)

    return {
        "call_text": call_text,
        "resp_text": resp_text,
        "program": program,
        "has_tool_call": "1" if has_tool_call else "0",
        "response_ok": "1" if response_ok else "0",
    }

# ============================
# Solver usage detector
# ============================

_SYMBOLIC_SOLVERS = ("z3", "sympy", "pysmt", "ortools", "pysat")
_TOOL_ALIASES = {
    "z3": ["z3"],
    "sympy": ["sympy"],
    "pysmt": ["pysmt"],
    "ortools": ["ortools"],
    "pysat": ["pysat"],
}

def _build_import_or_from_regex(roots: list[str]) -> re.Pattern:
    roots_pat = "|".join(re.escape(r) for r in roots)
    # matches:
    #   import ROOT
    #   import ROOT.submodule
    #   import x, ROOT, y
    #   from ROOT import ...
    #   from ROOT.submodule import ...
    return re.compile(
        rf"""
        (?:^|[\n\r])\s*
        (?:
            import\s+(?:[\w\.]+\s*,\s*)*({roots_pat})(?:\.[\w\.]+)?\b
            |
            from\s+({roots_pat})(?:\.[\w\.]+)?\s+import\b
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

def _detect_solver_import(program: str, solver: str) -> bool:
    """
    Return whether the program imports the target solver.
    """
    prog = _safe_str(program)
    if not prog.strip():
        return False

    solver = _safe_str(solver).strip().lower()

    if solver in _TOOL_ALIASES:
        roots = _TOOL_ALIASES[solver]
        solver_rx = _build_import_or_from_regex(roots)
        return bool(solver_rx.search(prog))

    return False


def _detect_sampled_solver_import(program: str, sampled_tool: str) -> bool:
    """
    Return whether the program imports the target sampled solver.
    """
    return _detect_solver_import(program, sampled_tool)


def _detect_any_solver_import(program: str) -> Optional[str]:
    """
    Return the first supported symbolic solver imported by the program.
    """
    for solver in _SYMBOLIC_SOLVERS:
        if _detect_solver_import(program, solver):
            return solver
    return None


def _count_solver_tool_calls(solution_str: str) -> dict[str, int]:
    """Count observed tool executions by imported symbolic solver.

    A call importing multiple supported solvers contributes to every matching
    solver counter. An execution whose call source cannot be recovered, whose
    program cannot be parsed, or which contains no supported solver import
    contributes to ``unclassified``.
    """
    counts = {solver: 0 for solver in _SYMBOLIC_SOLVERS}
    unclassified = 0

    for call_text in _extract_response_backed_tool_call_texts(solution_str):
        program = _extract_code_from_tool_call_text(call_text)
        if not program:
            program = _safe_str(call_text).strip()

        detected = [solver for solver in _SYMBOLIC_SOLVERS if _detect_solver_import(program, solver)]
        if not detected:
            unclassified += 1
            continue
        for solver in detected:
            counts[solver] += 1

    counts["unclassified"] = unclassified
    return counts


def _count_successful_solver_tool_calls(tool_outcomes: list[dict[str, Any]]) -> dict[str, int]:
    """Count solver-attributed calls whose execution and output were usable."""
    counts = {solver: 0 for solver in _SYMBOLIC_SOLVERS}
    unclassified = 0

    for outcome in tool_outcomes:
        if not outcome["response_ok"]:
            continue
        call_text = _safe_str(outcome.get("call_text", ""))
        program = _extract_code_from_tool_call_text(call_text)
        if not program:
            program = call_text.strip()

        detected = [solver for solver in _SYMBOLIC_SOLVERS if _detect_solver_import(program, solver)]
        if not detected:
            unclassified += 1
            continue
        for solver in detected:
            counts[solver] += 1

    counts["unclassified"] = unclassified
    return counts


def _count_tool_error_types(tool_outcomes: list[dict[str, Any]]) -> dict[str, int]:
    counts = {error_type: 0 for error_type in _TOOL_ERROR_TYPES}
    counts["unclassified"] = 0
    for outcome in tool_outcomes:
        error_type = _safe_str(outcome.get("error_type", "")).strip().casefold()
        if not error_type and outcome["response_ok"]:
            continue
        counts[error_type if error_type in counts else "unclassified"] += 1
    return counts


def _normalize_solver_usage_requirement(value: str) -> str:
    value = _safe_str(value).strip().lower().replace("-", "_")
    if value in {"any", "any_symbolic", "no_restriction", "unrestricted"}:
        return "any"
    return "sampled"


# ============================
# Main reward
# ============================

async def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    format_score=0.1,
    correctness_score=1.0,     # R_correct
    tool_success_score=0.1,
    solver_usage_score=0.1,
    honest_score=0.0,          # R_honest
    wrong_floor_alpha=0.1,     # alpha
    guess_margin_epsilon=0.02, # epsilon
    reward_honest_idk: bool = True,
    logic_equivalent_timeout_s: float = 20.0,
    logic_equivalent_solver_timeout_ms: int = 20000,
    enable_logic_equivalent: bool = True,
    include_tool_reward_in_score: bool = False,
    include_solver_reward_in_score: bool = True,
    solver_usage_requirement: str = "sampled",
    **kwargs,
):
    """
    Returns dict with:
      - score: total reward
      - plus existing metrics
    """
    solution_str = _safe_str(solution_str)
    tool_outcomes = _tool_execution_outcomes(solution_str, extra_info)
    tool_response_count = len(tool_outcomes)
    failed_tool_response_count = sum(1 for outcome in tool_outcomes if not outcome["response_ok"])
    successful_tool_response_count = tool_response_count - failed_tool_response_count
    tool_error_type_counts = _count_tool_error_types(tool_outcomes)
    solver_tool_call_counts = _count_solver_tool_calls(solution_str)
    successful_solver_tool_call_counts = _count_successful_solver_tool_calls(tool_outcomes)

    # ----------------------------
    # IDK detection
    # ----------------------------
    idk_token = extract_solution(solution_str, mode="word")
    idk_token_2 = extract_solution(solution_str, mode="all")
    said_idk = (
        (idk_token or "").strip().lower() == "idk"
        or (idk_token_2 or "").strip().lower() == "idk"
    )

    # choices length (for wrong-guess penalty)
    n_choices = 0
    try:
        ch = (extra_info or {}).get("possible_choices", (extra_info or {}).get("possible_choice", [])) or []
        n_choices = int(len(ch))
    except Exception:
        n_choices = 0

    if reward_honest_idk:
        incorrectness_score = _wrong_penalty(
            n_choices, correctness_score, honest_score, guess_margin_epsilon, wrong_floor_alpha
        )
        idk_score = honest_score
    else:
        incorrectness_score = 0.0
        idk_score = 0.0
        said_idk = False

    # ----------------------------
    # Format + correctness
    # ----------------------------
    fmt_r = 0.0
    answer_format = (extra_info or {}).get("answer_format", "word")

    is_logic_generator = _safe_str(data_source).lower().startswith("logic_generator/")
    puzzle = _safe_str((extra_info or {}).get("puzzle", "")).strip().lower()
    typ = _safe_str((extra_info or {}).get("type", "")).strip().lower()

    is_correct = False

    if answer_format == "free_response" and is_logic_generator and typ == "solving":
        _mod, synth_verify, err = _load_synth_verifier(puzzle)
        try:
            fmt_r = format_score if _last_code_block_content(solution_str) else 0.0
        except Exception:
            fmt_r = 0.0

        is_correct = False
        if callable(synth_verify):
            try:
                inst_raw = (extra_info or {}).get("instance", "")
                if isinstance(inst_raw, str):
                    try:
                        instance = json.loads(inst_raw)
                    except Exception:
                        instance = inst_raw
                else:
                    instance = inst_raw

                candidate = _last_code_block_content(solution_str) or solution_str
                vr = synth_verify(instance, candidate)
                if isinstance(vr, tuple):
                    is_correct = bool(vr[0])
                else:
                    is_correct = bool(vr)
            except Exception:
                is_correct = False

    elif answer_format == "mc":
        pred_tok = extract_solution(solution_str, mode="mc")
        gt_tok = normalize_mc_token(ground_truth)
        fmt_r = format_score if (pred_tok is not None or said_idk) else 0.0
        is_correct = bool(pred_tok is not None and gt_tok is not None and pred_tok == gt_tok)

    elif answer_format == "free_response":
        task_name = (extra_info or {}).get("task_name", "")
        _mod, verify_fn, extractor_fn, _err = _load_verifier(task_name)
        try:
            if extractor_fn is not None:
                extracted = extractor_fn(solution_str)
                fmt_r = format_score if not _is_empty_like(extracted) else 0.0
            else:
                fmt_r = format_score if _last_code_block_content(solution_str) else 0.0
        except Exception:
            fmt_r = format_score if _last_code_block_content(solution_str) else 0.0

        is_correct = False
        if callable(verify_fn):
            try:
                is_correct = bool(
                    verify_fn(
                        solution_str,
                        ground_truth,
                        _safe_str((extra_info or {}).get("meta", "")),
                    )
                )
            except Exception:
                is_correct = False

    else:
        pred = extract_solution(solution_str, mode="word")
        gt = _safe_str(ground_truth).strip().lower() if ground_truth is not None else None
        fmt_r = format_score if (pred is not None) else 0.0
        is_correct = (gt == pred)

    computed_correctness = compute_score_correctness(
        is_correct, said_idk, correctness_score, idk_score, incorrectness_score
    )

    # ----------------------------
    # Unified: tool_success + solver_usage share same "last tool context"
    # ----------------------------
    if tool_outcomes:
        # Attribute last-call rewards to the last call that actually received
        # a tool response. A trailing call beyond max_parallel_calls was never
        # executed and must not inherit the prior call's structured status.
        last_outcome = tool_outcomes[-1]
        call_text = _safe_str(last_outcome.get("call_text", "")).strip()
        resp_text = _safe_str(last_outcome.get("resp_text", "")).strip()
        last_program = _extract_code_from_tool_call_text(call_text) or call_text
        has_tool_call = bool(call_text)
        response_ok = bool(last_outcome["response_ok"])
        ctx = {
            "call_text": call_text or None,
            "resp_text": resp_text or None,
            "program": last_program or None,
            "has_tool_call": "1" if has_tool_call else "0",
            "response_ok": "1" if response_ok else "0",
        }
    else:
        ctx = _get_last_tool_context(solution_str)
        has_tool_call = ctx["has_tool_call"] == "1"
        response_ok = False
        last_program = ctx["program"] or ""

    # tool_success: require tool call + tool_response OK
    tool_r = tool_success_score if (has_tool_call and response_ok) else 0.0

    # solver_usage: require tool call + response_ok + configured solver import
    solver_r = 0.0
    used_sampled_solver = False
    used_required_solver = False
    detected_solver = ""
    formula_extraction_ok = False
    formula_extraction_error = ""
    formula_extraction_variable_count = 0.0
    formula_extraction_assertion_count = 0.0
    sampled_tool = _safe_str((extra_info or {}).get("sampled_tool", "")).strip().lower()
    solver_usage_requirement = _normalize_solver_usage_requirement(solver_usage_requirement)
    if solver_usage_requirement == "sampled" and not sampled_tool and solver_usage_score > 0.0:
        solver_usage_requirement = "any"
    logic_equivalent_backend = sampled_tool if solver_usage_requirement == "sampled" else None

    if has_tool_call:
        if solver_usage_requirement == "any":
            detected_solver = _detect_any_solver_import(last_program) or ""
            used_required_solver = bool(detected_solver)
            used_sampled_solver = used_required_solver
        else:
            used_sampled_solver = _detect_sampled_solver_import(last_program, sampled_tool)
            used_required_solver = used_sampled_solver
            detected_solver = sampled_tool if used_sampled_solver else ""

        if response_ok and used_required_solver:
            solver_r = min(0.1, solver_usage_score)

    logic_equivalent_r = 0.0
    logic_equivalent_timed_out = 0.0
    equivalence_check_time = 0.0

    logic_result = None
    needs_formula_extraction = has_tool_call and response_ok and used_required_solver
    needs_logic_equivalent = enable_logic_equivalent and has_tool_call and response_ok

    if needs_formula_extraction or needs_logic_equivalent:
        try:
            logic_result = compute_logic_equivalent(
                last_program,
                extra_info or {},
                backend=logic_equivalent_backend,
                timeout_s=logic_equivalent_timeout_s,
                solver_timeout_ms=logic_equivalent_solver_timeout_ms,
            )
        except Exception:
            logic_result = None

    if isinstance(logic_result, dict):
        formula_extraction_error = _safe_str(logic_result.get("error", ""))
        try:
            formula_extraction_variable_count = float(logic_result.get("extracted_variable_count", 0.0))
        except Exception:
            formula_extraction_variable_count = 0.0
        try:
            formula_extraction_assertion_count = float(logic_result.get("extracted_assertion_count", 0.0))
        except Exception:
            formula_extraction_assertion_count = 0.0
        formula_extraction_ok = formula_extraction_variable_count > 0.0

        if enable_logic_equivalent:
            logic_equivalent_r = float(logic_result.get("logic_equivalent", 0.0))
            logic_equivalent_timed_out = float(logic_result.get("timed_out", 0.0))
            equivalence_check_time = float(logic_result.get("equivalence_check_time", 0.0))

    if response_ok and used_required_solver and formula_extraction_ok:
        solver_r = solver_usage_score

    # ----------------------------
    # Total
    # ----------------------------
    tool_r_in_score = tool_r if include_tool_reward_in_score else 0.0
    solver_r_in_score = solver_r if include_solver_reward_in_score else 0.0
    base_total = (
        fmt_r
        + computed_correctness["adjusted_correctness_score"]
        + tool_r_in_score
        + solver_r_in_score
        + logic_equivalent_r
    )
    # Failed tool responses remain available through diagnostic counters, but
    # do not create an additional negative reward.
    total = base_total

    return {
        "score": total,
        "format": fmt_r,
        "correctness": computed_correctness["correctness"],
        "incorrectness": computed_correctness["incorrectness"],
        "idk": computed_correctness["idk"],
        "adjusted_correctness": computed_correctness["adjusted_correctness_score"],
        "tool_success": tool_r,
        "solver_usage": solver_r,
        "logic_equivalent": logic_equivalent_r,
        "tool_response_count": float(tool_response_count),
        "successful_tool_response_count": float(successful_tool_response_count),
        "failed_tool_response_count": float(failed_tool_response_count),
        "invalid_tool_name_count": float(tool_error_type_counts["invalid_tool_name"]),
        "tool_argument_error_count": float(tool_error_type_counts["invalid_arguments"]),
        "tool_creation_error_count": float(tool_error_type_counts["tool_creation_error"]),
        "tool_execution_error_count": float(tool_error_type_counts["tool_execution_error"]),
        "tool_release_error_count": float(tool_error_type_counts["tool_release_error"]),
        "backend_execution_error_count": float(tool_error_type_counts["backend_execution_error"]),
        "empty_tool_output_count": float(tool_error_type_counts["empty_output"]),
        "unclassified_tool_error_count": float(tool_error_type_counts["unclassified"]),
        "z3_tool_call_count": float(solver_tool_call_counts["z3"]),
        "sympy_tool_call_count": float(solver_tool_call_counts["sympy"]),
        "pysmt_tool_call_count": float(solver_tool_call_counts["pysmt"]),
        "ortools_tool_call_count": float(solver_tool_call_counts["ortools"]),
        "pysat_tool_call_count": float(solver_tool_call_counts["pysat"]),
        "unclassified_tool_call_count": float(solver_tool_call_counts["unclassified"]),
        "z3_successful_tool_call_count": float(successful_solver_tool_call_counts["z3"]),
        "sympy_successful_tool_call_count": float(successful_solver_tool_call_counts["sympy"]),
        "pysmt_successful_tool_call_count": float(successful_solver_tool_call_counts["pysmt"]),
        "ortools_successful_tool_call_count": float(successful_solver_tool_call_counts["ortools"]),
        "pysat_successful_tool_call_count": float(successful_solver_tool_call_counts["pysat"]),
        "unclassified_successful_tool_call_count": float(successful_solver_tool_call_counts["unclassified"]),
        "logic_equivalent_timed_out": logic_equivalent_timed_out,
        "logic_equivalent_check_time": equivalence_check_time,
        "formula_extraction_ok": 1.0 if formula_extraction_ok else 0.0,
    }
