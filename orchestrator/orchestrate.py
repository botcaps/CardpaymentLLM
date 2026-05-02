"""
Orchestrator — async public API wrapper around the LangGraph pipeline.

Keeps the original (Decision | None, Trace) return type so all callers
(dashboard, tests) continue to work unchanged.

HITL note: when invoked through this API (e.g. batch summary runs), any
human-in-the-loop gate is auto-approved.  Interactive HITL is handled by
calling orchestrator.graph.pipeline directly from the dashboard.
"""
from __future__ import annotations
import asyncio
from dataclasses import asdict

from langsmith import traceable
from langgraph.types import Command

from contracts.models import Transaction, Decision
from observability.tracer import Trace
from orchestrator.graph import pipeline


def _build_trace(txn: Transaction, state: dict) -> Trace:
    trace = Trace(txn_id=txn.txn_id)
    trace.candidates         = state.get("candidates") or []
    trace.planner_reasoning  = state.get("planner_reasoning")
    trace.degraded           = state.get("degraded", False)
    trace.auth_scores        = [asdict(a) for a in (state.get("auth_scores") or [])]
    trace.cost_scores        = [asdict(c) for c in (state.get("cost_scores") or [])]
    trace.fraud_scores       = [asdict(f) for f in (state.get("fraud_scores") or [])]
    trace.ranked             = state.get("ranked") or []
    trace.reflection         = state.get("reflection")
    trace.error              = state.get("error")
    trace.guardrail_warnings = state.get("guardrail_warnings") or []
    decision                 = state.get("decision")
    trace.decision           = asdict(decision) if decision is not None else None
    trace.explanation        = state.get("explanation")
    return trace


@traceable(run_type="chain", name="payment-orchestrator")
async def orchestrate(txn: Transaction) -> tuple[Decision | None, Trace]:
    config = {"configurable": {"thread_id": txn.txn_id}}

    def _invoke():
        state = pipeline.invoke({"txn": txn}, config=config)
        # Auto-approve any HITL gate (non-interactive mode)
        graph_state = pipeline.get_state(config)
        if graph_state.next:
            state = pipeline.invoke(Command(resume={"approved": True}), config=config)
        return state

    state    = await asyncio.get_event_loop().run_in_executor(None, _invoke)
    trace    = _build_trace(txn, state)
    decision = state.get("decision")
    return decision, trace
