"""Restricted execution of the pandas code that Claude writes to answer questions.

IMPORTANT: this is defence-in-depth, not a perfect security boundary. It blocks imports,
file/network access, dunder tricks and infinite loops, and only exposes allow-listed
pandas/numpy functions. For a public deployment also run the app in a container
(see Dockerfile) and never put anything sensitive on the server.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import io
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

try:  # plotly is optional here so the module still imports without it
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    px = go = None

MAX_CODE_CHARS = 6000
MAX_OUTPUT_CHARS = 6000
DEFAULT_TIMEOUT_S = 20

SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "int",
    "isinstance", "len", "list", "map", "max", "min", "pow", "print", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "True", "False", "None", "Exception", "ValueError", "KeyError", "TypeError",
]
SAFE_BUILTINS = {n: getattr(builtins, n) for n in SAFE_BUILTIN_NAMES}

BLOCKED_ATTRS = {"eval", "query", "format", "format_map", "show", "style", "mro"}
SAFE_TO_METHODS = {
    "to_string", "to_dict", "to_list", "to_numpy", "to_frame", "to_series", "to_period",
    "to_timestamp", "to_datetime", "to_numeric", "to_pydatetime", "to_markdown", "to_timedelta",
}
BLOCKED_PREFIXES = ("read_", "write_", "save", "load", "dump", "open")

PD_ALLOWED = {
    "DataFrame", "Series", "to_datetime", "to_numeric", "to_timedelta", "concat", "cut", "qcut",
    "crosstab", "pivot_table", "merge", "isna", "isnull", "notna", "notnull", "Timestamp",
    "Timedelta", "date_range", "get_dummies", "unique", "NaT", "NA", "Categorical", "Grouper",
    "DateOffset", "Period", "factorize", "value_counts", "melt", "pivot",
}
NP_ALLOWED = {
    "array", "arange", "linspace", "mean", "median", "std", "var", "sum", "min", "max", "abs",
    "sqrt", "log", "log1p", "log10", "log2", "exp", "round", "floor", "ceil", "where", "nan",
    "inf", "isnan", "isfinite", "isinf", "percentile", "quantile", "corrcoef", "polyfit",
    "polyval", "cumsum", "cumprod", "diff", "unique", "sort", "argsort", "clip", "select",
    "maximum", "minimum", "nanmean", "nansum", "nanmedian", "nanstd", "nanmin", "nanmax",
    "histogram", "digitize", "searchsorted", "sign", "power", "prod", "ones", "zeros", "full",
    "repeat", "tile", "concatenate", "isin", "any", "all", "count_nonzero", "bincount",
    "average", "ptp", "pi", "e", "float64", "int64", "nan_to_num", "cov", "argmax", "argmin",
    "round_", "trunc", "mod", "sin", "cos", "tan", "arctan2", "radians", "degrees",
}


class SandboxError(Exception):
    """Raised for rejected or failing code. The message is shown to Claude so it can retry."""


class _Proxy:
    """Exposes only allow-listed attributes of a module."""

    def __init__(self, module: Any, allowed: set[str], name: str):
        self._module, self._allowed, self._name = module, allowed, name

    def __getattr__(self, item: str) -> Any:
        if item not in self._allowed:
            raise AttributeError(f"'{self._name}.{item}' is not available in the analysis sandbox")
        return getattr(self._module, item)


def _attr_allowed(attr: str) -> bool:
    if attr.startswith("_") or attr in BLOCKED_ATTRS:
        return False
    if attr.startswith("to_") and attr not in SAFE_TO_METHODS:
        return False
    return not attr.startswith(BLOCKED_PREFIXES)


def validate(code: str) -> ast.Module:
    if not code or not code.strip():
        raise SandboxError("No code provided.")
    if len(code) > MAX_CODE_CHARS:
        raise SandboxError(f"Code is too long (max {MAX_CODE_CHARS} characters). Make it shorter.")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SandboxError(f"SyntaxError: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxError("Imports are not allowed. pd, np, px and go are already available.")
        if isinstance(node, (ast.While, ast.With, ast.Global, ast.Nonlocal, ast.ClassDef,
                             ast.AsyncFunctionDef, ast.Await)):
            raise SandboxError(f"'{type(node).__name__}' statements are not allowed. Use for-loops or vectorised pandas.")
        if isinstance(node, ast.Attribute) and not _attr_allowed(node.attr):
            raise SandboxError(f"Attribute '.{node.attr}' is not allowed in the sandbox.")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SandboxError("Names starting with '__' are not allowed.")
    return tree


def _make_tracer(deadline: float):
    def tracer(frame, event, arg):
        if frame.f_code.co_filename != "<analysis>":
            return None  # only trace the user's own code, not pandas internals

        def local(frame, event, arg):
            if time.monotonic() > deadline:
                raise TimeoutError("Execution took too long and was stopped.")
            return local

        if time.monotonic() > deadline:
            raise TimeoutError("Execution took too long and was stopped.")
        return local

    return tracer


def _format_value(v: Any) -> str:
    if isinstance(v, pd.DataFrame):
        return v.to_string(max_rows=40, max_cols=25)
    if isinstance(v, pd.Series):
        return v.to_string(max_rows=40)
    return str(v)


def run_code(code: str, df: pd.DataFrame, timeout: float = DEFAULT_TIMEOUT_S) -> dict:
    """Run analysis code against a copy of df. Returns {"output": str, "fig": Figure|None}.

    The code may assign `result` (anything printable) and/or `fig` (a plotly figure).
    A trailing bare expression is treated as the result.
    """
    tree = validate(code)

    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        tree.body[-1] = ast.copy_location(
            ast.Assign(targets=[ast.Name(id="result", ctx=ast.Store())], value=last.value), last
        )
        ast.fix_missing_locations(tree)

    ns: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "df": df.copy(),
        "pd": _Proxy(pd, PD_ALLOWED, "pd"),
        "np": _Proxy(np, NP_ALLOWED, "np"),
    }
    if px is not None:
        ns["px"], ns["go"] = px, go

    compiled = compile(tree, "<analysis>", "exec")
    buf = io.StringIO()
    old_trace = sys.gettrace()
    sys.settrace(_make_tracer(time.monotonic() + timeout))
    try:
        with contextlib.redirect_stdout(buf):
            exec(compiled, ns)  # noqa: S102 - validated, restricted namespace
    except TimeoutError as e:
        raise SandboxError(str(e)) from e
    except SandboxError:
        raise
    except Exception as e:  # noqa: BLE001
        raise SandboxError(f"{type(e).__name__}: {str(e)[:1200]}") from e
    finally:
        sys.settrace(old_trace)

    parts = []
    printed = buf.getvalue().strip()
    if printed:
        parts.append(printed)
    result = ns.get("result")
    if result is not None:
        parts.append(_format_value(result))
    output = "\n".join(parts) or "(code ran but produced no output; assign the answer to `result`)"
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"

    fig = ns.get("fig")
    if fig is not None and not hasattr(fig, "to_plotly_json"):
        fig = None
    return {"output": output, "fig": fig}
