"""Claude API integration: written insights and natural-language Q&A over the data."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .analyzer import rank_numeric
from .sandbox import SandboxError, run_code

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MODEL_CHOICES = ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"]
MAX_TOOL_ROUNDS = 6


class AIError(Exception):
    """Friendly, user-displayable error."""


INSIGHTS_SYSTEM = """You are a senior data analyst writing for a business reader.
You are given an automatically generated profile of a dataset. Base EVERY statement on the
numbers in the profile; never invent figures. If something cannot be concluded from the profile, say so.

Write in Markdown with exactly these sections:
## Executive Summary
(3-4 sentences: what the data is and the most important takeaways)
## Key Findings
(5-8 bullets, each with specific numbers)
## Data Quality Issues
(missing values, duplicates, outliers, odd columns - and how serious each is)
## Trends & Patterns
(only what the trends/correlations support)
## Recommendations
(numbered, prioritised, concrete, actionable next steps - both for data cleanup and for the business)
## Questions Worth Asking Next
(4-5 specific questions the user could ask about this data)"""

QA_SYSTEM = """You are an expert data analyst inside a web app. The user has uploaded a dataset that is loaded
as a pandas DataFrame named `df`. Answer their questions accurately.

Rules:
- Use the `run_pandas` tool to compute answers. NEVER guess or invent numbers - compute them.
- In the tool, `df`, `pd`, `np`, `px` (plotly express) and `go` (plotly graph_objects) are available. No imports.
  Assign the final answer to `result`. To draw a chart, build a plotly figure and assign it to `fig`.
- File access, network access, while-loops and df.query/df.eval are blocked. Use boolean masks and groupby.
- Keep code short. If a call errors, read the message, fix the code and retry.
- Respect the dataset: column names are case-sensitive. Mind missing values and say if they affect an answer.
- Reply in clear Markdown: lead with the direct answer, then brief supporting numbers or interpretation.
  Do not paste long code in the answer; the app shows the code separately.
- If the question cannot be answered from this data, say so and suggest what would be needed.
- For "why" questions, give what the data supports and flag that correlation is not causation.

DATASET PROFILE:
"""

TOOLS = [
    {
        "name": "run_pandas",
        "description": (
            "Run Python (pandas/numpy/plotly) code against the DataFrame `df` and return the printed output "
            "and the value of `result`. Assign your answer to `result`; optionally assign a plotly figure to `fig`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to execute."}},
            "required": ["code"],
        },
    }
]


@dataclass
class Answer:
    text: str
    figures: list = field(default_factory=list)
    code_steps: list[dict] = field(default_factory=list)  # {"code":..., "output":..., "error":bool}


def make_client(api_key: str):
    import anthropic

    return anthropic.Anthropic(api_key=api_key, timeout=120.0, max_retries=2)


def _friendly(e: Exception) -> AIError:
    import anthropic

    if isinstance(e, anthropic.AuthenticationError):
        return AIError("The API key was rejected. Check that it is correct and active.")
    if isinstance(e, anthropic.PermissionDeniedError):
        return AIError("This API key does not have permission to use that model.")
    if isinstance(e, anthropic.NotFoundError):
        return AIError("Model not found. Pick another model in the sidebar or check the CLAUDE_MODEL setting.")
    if isinstance(e, anthropic.RateLimitError):
        return AIError("Rate limit reached. Wait a moment and try again.")
    if isinstance(e, anthropic.APIConnectionError):
        return AIError("Could not reach the Claude API. Check your internet connection and try again.")
    if isinstance(e, anthropic.APIStatusError):
        return AIError(f"Claude API error ({e.status_code}): {getattr(e, 'message', str(e))}")
    return AIError(f"Unexpected error: {e}")


def _text_of(response) -> str:
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()


def generate_insights(client, model: str, profile_text: str) -> str:
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=2500,
            system=INSIGHTS_SYSTEM,
            messages=[{"role": "user", "content": f"Dataset profile:\n\n{profile_text}\n\nWrite the analysis."}],
        )
    except Exception as e:  # noqa: BLE001
        raise _friendly(e) from e
    return _text_of(resp)


def answer_question(
    client,
    model: str,
    df: pd.DataFrame,
    profile_text: str,
    history: list[dict],
    question: str,
) -> Answer:
    """Tool-use loop: Claude writes pandas code, we run it safely, Claude explains the result."""
    system = [{"type": "text", "text": QA_SYSTEM + profile_text, "cache_control": {"type": "ephemeral"}}]
    messages: list[dict[str, Any]] = [
        {"role": m["role"], "content": m["content"]} for m in history[-8:] if m.get("content")
    ]
    messages.append({"role": "user", "content": question})

    figures, steps = [], []
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            resp = client.messages.create(
                model=model, max_tokens=2000, system=system, tools=TOOLS, messages=messages
            )
            if resp.stop_reason != "tool_use":
                text = _text_of(resp) or "I could not produce an answer for that. Try rephrasing the question."
                return Answer(text, figures, steps)

            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if getattr(block, "type", "") != "tool_use":
                    continue
                code = (block.input or {}).get("code", "")
                try:
                    out = run_code(code, df)
                    steps.append({"code": code, "output": out["output"], "error": False})
                    if out["fig"] is not None:
                        figures.append(out["fig"])
                    content = out["output"] + ("\n[A chart was created and is shown to the user.]" if out["fig"] is not None else "")
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
                except SandboxError as e:
                    steps.append({"code": code, "output": str(e), "error": True})
                    results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": f"Error: {e}", "is_error": True}
                    )
            messages.append({"role": "user", "content": results})
    except AIError:
        raise
    except Exception as e:  # noqa: BLE001
        raise _friendly(e) from e

    return Answer(
        "I ran out of steps before finishing. Try asking a more specific or simpler question.", figures, steps
    )


def default_questions(profile: dict) -> list[str]:
    """Free (no API call) starter questions based on the dataset's columns."""
    kinds = profile["kinds"]
    num = rank_numeric([c for c, k in kinds.items() if k == "numeric"])
    cat = [c for c, k in kinds.items() if k == "categorical"]
    dts = [c for c, k in kinds.items() if k == "datetime"]
    qs = ["Give me a quick summary of this dataset."]
    if num and cat:
        qs.append(f"Which {cat[0]} has the highest total {num[0]}?")
    if num and dts:
        qs.append(f"How has {num[0]} changed over time? Show a chart.")
    if len(num) >= 2:
        qs.append(f"Is there a relationship between {num[0]} and {num[1]}?")
    if profile["outliers"]:
        qs.append(f"Show me the outliers in {next(iter(profile['outliers']))}.")
    if profile["missing"]["missing_cells"]:
        qs.append("Which columns have missing values and how should I handle them?")
    return qs[:5]
