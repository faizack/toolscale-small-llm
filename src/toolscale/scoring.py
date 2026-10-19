"""Score one model response and assign a failure category.

Strict correctness = the vendored BFCL checker (identical to BFCL `simple` scoring).
Lenient correctness = the same checker after coercing string-encoded scalars to the
schema type ("10" -> 10, "true" -> True, "[1, 2]" -> [1, 2]). Lenient scoring
isolates *serialisation* errors (e.g. Llama-3.2 emits every number as a string)
from *semantic* errors. It is reported separately, never instead of strict.

Failure taxonomy (RQ2), assigned in this priority order:

  timeout           the serving stack returned nothing within the client timeout (final failure)
  error             request failed (connection reset / server crash) -> retried on resume
  (no parsed tool call, but the raw tokens contain an attempted call:)
  hallucinated_tool   valid JSON call to an unknown name -- Ollama silently drops these
  text_call           valid JSON call written as prose/markdown instead of a tool call
  malformed_call      not valid JSON, e.g. `{"name": air_quality, ...}`
  no_call           no attempted call: plain-text answer or refusal
  multi_call        more than one call (task needs exactly one)
  hallucinated_tool called a name that is not in the visible catalog
  wrong_tool        called a visible tool that is not the gold tool
  missing_arg       gold tool; a required parameter is missing
  extra_arg         gold tool; parameter not in schema / not expected
  wrong_type        gold tool; a value has the wrong type
  wrong_value       gold tool; a value is not among the accepted answers
  missing_optional  gold tool; an optional-but-expected parameter omitted
  correct
"""

from __future__ import annotations

import copy
import json
import re

from .bfcl_checker import simple_function_checker
from .data import Query, checker_description

CATEGORIES = ["correct", "wrong_value", "wrong_type", "missing_arg", "extra_arg",
              "missing_optional", "wrong_tool", "hallucinated_tool", "multi_call",
              "malformed_call", "text_call", "no_call", "timeout", "error"]

_ERR_MAP = {
    "simple_function_checker:missing_required": "missing_arg",
    "simple_function_checker:unexpected_param": "extra_arg",
    "simple_function_checker:missing_optional": "missing_optional",
}


def looks_like_call(text: str, catalog: list[str]) -> bool:
    if not text:
        return False
    if re.search(r'"name"\s*:', text) or "<tool_call>" in text or "[TOOL_CALLS]" in text:
        return True
    return any(re.search(rf"\b{re.escape(n)}\s*\(", text) for n in catalog)


def parse_raw_call(raw: str) -> tuple[str, str | None]:
    """Find the first attempted call in raw generated text.
    Returns ("json", name) if it is valid JSON, ("invalid", name guess) otherwise."""
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw, re.S) or re.search(r"(\{.*\})", raw, re.S)
    if not m:
        g = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", raw)
        return "invalid", (g.group(1) if g else None)
    try:
        obj = json.loads(m.group(1))
        return "json", (obj.get("name") if isinstance(obj, dict) else None)
    except json.JSONDecodeError:
        g = re.search(r'"name"\s*:\s*"?([A-Za-z0-9_.]+)', m.group(1))
        return "invalid", (g.group(1) if g else None)


def _coerce(value, typ: str):
    """Best-effort conversion of a string-encoded value to a BFCL schema type."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    try:
        if typ == "integer" and re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        if typ == "float" and re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", s):
            return float(s)
        if typ == "boolean" and s.lower() in ("true", "false"):
            return s.lower() == "true"
        if typ in ("array", "tuple", "dict") and s[:1] in "[{":
            return json.loads(s)
    except (ValueError, json.JSONDecodeError):
        pass
    return value


def coerce_args(func: dict, args: dict) -> dict:
    props = func["parameters"]["properties"]
    return {k: (_coerce(v, props[k].get("type", "string")) if k in props else v) for k, v in args.items()}


def score(q: Query, resp: dict, catalog: list[str], lenient: bool = False) -> tuple[bool, str, str]:
    """Return (is_correct, category, checker_detail)."""
    if "error" in resp:
        # a hung request (no response within the client timeout) is a final failure, see lab notebook Entry 17
        return False, ("timeout" if "after watchdog restart" in resp["error"] else "error"), resp["error"]
    calls = resp["tool_calls"]
    if not calls:
        raw = resp.get("raw_output") or ""
        txt = raw + "\n" + (resp.get("content") or "")
        if not looks_like_call(txt, catalog):
            return False, "no_call", ""
        kind, name = parse_raw_call(raw or txt)
        if kind == "json" and name is not None and name not in catalog:
            # Ollama silently drops calls to unknown tool names; recovered from the raw tokens
            return False, "hallucinated_tool", f"{name} (dropped by server)"
        if kind == "json" and "<tool_call>" not in raw:
            return False, "text_call", name or ""  # a call written as prose/markdown, not a tool call
        return False, "malformed_call", name or ""
    if len(calls) > 1:
        return False, "multi_call", ""
    name, args = calls[0]["name"], calls[0]["arguments"]
    if name not in catalog:
        return False, "hallucinated_tool", name
    if name != q.gold_name:
        return False, "wrong_tool", name
    if "__unparsed__" in args:
        return False, "malformed_call", "arguments not valid JSON"
    if lenient:
        args = coerce_args(q.func, copy.deepcopy(args))
    try:
        res = simple_function_checker(checker_description(q.func), {name: args}, q.answer)
    except Exception as e:  # values the checker cannot handle
        return False, "wrong_type", f"checker exception: {e!r}"
    if res["valid"]:
        return True, "correct", ""
    et = res.get("error_type", "")
    if et in _ERR_MAP:
        cat = _ERR_MAP[et]
    elif et.startswith("type_error"):
        cat = "wrong_type"
    else:
        cat = "wrong_value"
    return False, cat, json.dumps(res.get("error", []))[:300]
