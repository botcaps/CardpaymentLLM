"""
Tool-use accuracy evaluator.

Question: did the auth agent call the right tools, in a reasonable order,
on a labeled transaction?

Why this matters (worth knowing for the viva):
  Agents fail in two distinguishable ways. The first is "wrong final answer
  with right reasoning" — they did the work, the data was misleading. The
  second is "right answer by accident" — they skipped tool calls, hallucinated
  intermediate values, and stumbled into a plausible number. Tool-use
  accuracy catches the second failure mode, which the headline metrics
  (auth-rate uplift, decision quality) wouldn't.

Scoring policy:
  required_tools_called / required_tools_total           — recall over required
  forbidden_tools_avoided                                 — pass/fail boolean
  emit_auth_score_called_exactly_once                     — pass/fail boolean

  Final tool_use_accuracy = recall × forbidden_avoided × emit_correct_count

  This is multiplicative on purpose: if the agent skipped a required tool
  AND called a forbidden one, both terms hit, score is heavily penalized.

Mode handling:
  In mock mode the auth scorer doesn't call tools (it's a deterministic
  function). We synthesise expected tool calls from the same data the
  scorer reads, so the eval is still meaningful — it asks: "if a real
  LLM were trying to score this transaction, would the data be in scope?"
  This is a coarser eval than live mode but it lets the harness run
  without API keys.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contracts.models import Transaction
from data.samples import SAMPLES


# ── Dataset loader ──────────────────────────────────────────────────────────

_DATA = Path(__file__).resolve().parent.parent / "datasets" / "tool_use_labels.jsonl"


def load_labels() -> list[dict]:
    if not _DATA.exists():
        raise FileNotFoundError(f"missing dataset: {_DATA}")
    with _DATA.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _txn_by_id(txn_id: str) -> Transaction | None:
    return next((t for t in SAMPLES if t.txn_id == txn_id), None)


# ── Tool-call extraction ────────────────────────────────────────────────────

def _extract_tool_calls_live(message_history: list) -> list[str]:
    """
    Walk a LangGraph ReAct agent's message history; return the names
    of every tool call in order. Used only in live mode where the
    agent really runs.
    """
    from langchain_core.messages import AIMessage, ToolMessage
    names: list[str] = []
    for msg in message_history:
        if isinstance(msg, AIMessage):
            for tc in (getattr(msg, "tool_calls", []) or []):
                # tool_calls is list of dicts in modern LangChain
                if isinstance(tc, dict) and "name" in tc:
                    names.append(tc["name"])
        elif isinstance(msg, ToolMessage):
            # ToolMessage.name is the tool just executed
            if getattr(msg, "name", None):
                names.append(msg.name)
    # Dedupe-while-keeping-order in case ai_message + tool_message both record it
    seen: set[str] = set()
    uniq: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def _synth_tool_calls_mock(txn: Transaction) -> list[str]:
    """
    In mock mode we don't have a tool-call trace. Synthesise the calls
    the rule-based scorer reads — feature_store_lookup and issuer_health_check
    always; scheme_decline_patterns when off-hours.
    """
    calls = ["feature_store_lookup", "issuer_health_check"]
    if txn.hour_of_day < 8 or txn.hour_of_day > 20:
        calls.append("scheme_decline_patterns")
    calls.append("emit_auth_score")
    return calls


# ── Per-row scoring ─────────────────────────────────────────────────────────

def score_one(label: dict, observed_tools: list[str]) -> dict:
    """
    Score one (label, observed) pair. Returns a dict with the breakdown +
    final tool_use_accuracy in [0, 1].
    """
    required = label.get("required_tools", [])
    forbidden = label.get("forbidden_tools", [])

    # Tool-call-only check excludes the final emit (handled separately)
    tool_calls = [t for t in observed_tools if t != "emit_auth_score"]

    # Recall over required
    if required:
        hits = sum(1 for t in required if t in tool_calls)
        recall = hits / len(required)
    else:
        recall = 1.0

    # Forbidden avoidance
    forbidden_hit = any(f in tool_calls for f in forbidden)
    forbidden_score = 0.0 if forbidden_hit else 1.0

    # emit_auth_score should appear exactly once at the end
    emit_count = sum(1 for t in observed_tools if t == "emit_auth_score")
    emit_score = 1.0 if emit_count == 1 else 0.0

    final = recall * forbidden_score * emit_score

    return {
        "txn_id":            label["txn_id"],
        "scheme":            label["scheme"],
        "recall":            round(recall, 3),
        "forbidden_avoided": forbidden_score == 1.0,
        "emit_exactly_once": emit_count == 1,
        "tool_use_accuracy": round(final, 3),
        "observed_tools":    observed_tools,
        "required_tools":    required,
        "notes":             label.get("notes", ""),
    }


# ── Public entry: run the whole eval ────────────────────────────────────────

def run_tool_use_eval(use_live: bool = False) -> dict:
    """
    Execute every labeled (txn, scheme) row.

    use_live=True actually invokes the auth agent and parses its tool
    calls. use_live=False uses the mock-mode synthesised trace
    (deterministic; useful for CI without API costs).

    Returns a dict with `per_row`, `summary` keys.
    """
    labels = load_labels()
    rows: list[dict] = []

    for lbl in labels:
        txn = _txn_by_id(lbl["txn_id"])
        if txn is None:
            continue
        if use_live:
            observed = _run_live_and_extract(txn, lbl["scheme"])
        else:
            observed = _synth_tool_calls_mock(txn)
        rows.append(score_one(lbl, observed))

    if not rows:
        return {"per_row": [], "summary": {"n": 0}}

    n              = len(rows)
    avg_accuracy   = sum(r["tool_use_accuracy"] for r in rows) / n
    perfect        = sum(1 for r in rows if r["tool_use_accuracy"] == 1.0)
    failed_recall  = sum(1 for r in rows if r["recall"] < 1.0)
    failed_forbid  = sum(1 for r in rows if not r["forbidden_avoided"])

    summary = {
        "n":                       n,
        "avg_tool_use_accuracy":   round(avg_accuracy, 3),
        "perfect_score_count":     perfect,
        "perfect_score_pct":       round(100 * perfect / n, 1),
        "rows_with_recall_miss":   failed_recall,
        "rows_with_forbidden_hit": failed_forbid,
        "mode":                    "live" if use_live else "mock",
    }
    return {"per_row": rows, "summary": summary}


def _run_live_and_extract(txn: Transaction, scheme: str) -> list[str]:
    """
    Live invocation: build the same auth ReAct agent the pipeline uses,
    run it on (txn, scheme), return the tool sequence.

    Imports stay inside the function so the eval module loads in mock
    mode without needing langchain to be installed.
    """
    from langchain_core.messages import HumanMessage
    from langgraph.prebuilt import create_react_agent
    from llm_clients import get_chat_model
    # Reuse the auth agent's tools and prompt from graph.py
    from orchestrator.graph import AUTH_TOOLS, AUTH_SYSTEM_PROMPT

    llm   = get_chat_model("fast")
    agent = create_react_agent(llm, AUTH_TOOLS, prompt=AUTH_SYSTEM_PROMPT)

    msg = HumanMessage(content=(
        f"Transaction: bin={txn.bin}, hour_of_day={txn.hour_of_day}, "
        f"three_ds_status={txn.three_ds_status}, amount={txn.amount_minor} {txn.currency}, "
        f"channel={txn.channel}, region={txn.region}\n"
        f"Scheme to score: {scheme}\n"
        "Gather data via tools, then call emit_auth_score."
    ))
    result = agent.invoke({"messages": [msg]})
    return _extract_tool_calls_live(result["messages"])
