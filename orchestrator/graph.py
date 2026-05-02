"""
LangGraph orchestration pipeline — all 6 agentic additions wired in.

  1. Fraud Risk Agent       — 3rd parallel ReAct agent scoring p_fraud per scheme
  2. Guardrails             — validate_input, validate_agent_outputs, validate_decision nodes
  3. Human-in-the-loop      — interrupt() gate for transactions > $500
  4. Reflection node        — LLM self-critique of the ranking before compliance
  5. LLM Planner            — Gemini decides which schemes are worth evaluating
  6. Explanation Agent      — plain-English justification attached to the decision

Topology:

  START
    ↓
  validate_input ──(hard_blocked)──→ END
    ↓
  plan_candidates  (LLM planner)
    ↓   fan-out — parallel super-step
  ┌──────────────────────────────────────┐
  │ run_auth_agent   ReAct (Gemini Flash)│
  │ run_cost_agent   ReAct (Gemini Pro)  │
  │ run_fraud_agent  ReAct (Gemini Flash)│
  └──────────────────────────────────────┘
    ↓   fan-in
  validate_agent_outputs ──(hard_blocked)──→ END
    ↓
  aggregate_scores   (w1·p_auth − w2·norm_fee − w3·p_fraud)
    ↓
  reflect_on_ranking  (LLM anomaly detection)
    ↓
  check_hitl_gate  (interrupt() if amount ≥ $500)
    ↓
  run_compliance   (deterministic regulatory gate)
    ↓
  validate_decision ──(hard_blocked)──→ END
    ↓
  generate_explanation  (LLM plain-English justification)
    ↓
  END
"""
from __future__ import annotations
import json
import os
import re
from typing import TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from contracts.models import Transaction, AuthScore, CostScore, FraudScore, Decision
from orchestrator.aggregate import aggregate
from compliance.gate import compliance_gate, NoEligibleSchemeError
from llm_clients import get_config, MODEL_AUTH_AGENT, MODEL_COST_AGENT, MODEL_ORCHESTRATOR


# ── Pipeline state ─────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    txn:                Transaction
    candidates:         list[str]
    planner_reasoning:  str | None
    auth_scores:        list[AuthScore]
    cost_scores:        list[CostScore]
    fraud_scores:       list[FraudScore]
    ranked:             list[dict]
    reflection:         str | None
    decision:           Decision | None
    explanation:        str | None
    error:              str | None
    degraded:           bool
    hard_blocked:       bool
    guardrail_warnings: list[dict]
    # Day 7 addition — counter for the auth agent self-correction loop.
    # Reflexion-style: original emit → critic evaluates → at most one
    # revision attempt. Capped at 1 to bound latency.
    auth_revision_count: int
    auth_critique:       str | None


# ── LLM factory ───────────────────────────────────────────────────────────

def _chat_model(tier: str = "fast"):
    """
    Build a LangChain chat model for the configured provider.

    `tier` is one of: 'fast' | 'smart' | 'judge'. Provider is picked by
    the LLM_PROVIDER env var (see llm_clients.py). This indirection means
    the same agent code runs on OpenAI / Anthropic / Gemini / Llama / Ollama
    by changing one env var.
    """
    from llm_clients import get_chat_model
    return get_chat_model(tier)


# ── LangChain tools: auth agent ────────────────────────────────────────────

@tool
def auth_feature_store_lookup(bin: str, scheme: str) -> str:
    """Return rolling 30/90-day auth rates for a BIN on a specific scheme, plus sample size."""
    from agents.auth_score.tools import feature_store_lookup
    result = feature_store_lookup(bin, scheme)
    return json.dumps(result) if result is not None else "null"


@tool
def auth_issuer_health_check(bin: str, scheme: str) -> str:
    """Return current issuer status (healthy / elevated_declines / unknown) for the bank behind this BIN."""
    from agents.auth_score.tools import issuer_health_check
    return json.dumps(issuer_health_check(bin, scheme))


@tool
def auth_scheme_decline_patterns(bin: str, hour_of_day: int) -> str:
    """Return decline-code distribution bucketed by business hours vs off-hours for this BIN."""
    from agents.auth_score.tools import scheme_decline_patterns
    result = scheme_decline_patterns(bin, hour_of_day)
    return json.dumps(result) if result is not None else "null"


@tool
def emit_auth_score(scheme: str, p_auth: float, confidence: float, reasoning: str) -> str:
    """Emit the final auth score for a scheme. Call exactly once after gathering all data."""
    return json.dumps({
        "scheme": scheme, "p_auth": p_auth,
        "confidence": confidence, "reasoning": reasoning,
    })


AUTH_TOOLS = [
    auth_feature_store_lookup,
    auth_issuer_health_check,
    auth_scheme_decline_patterns,
    emit_auth_score,
]

AUTH_SYSTEM_PROMPT = """\
You are the Auth Score Agent for a payments routing system.
Estimate P(auth | scheme) — the probability the issuer approves this transaction via the given network.

Process:
1. Call auth_feature_store_lookup for the baseline auth rate (BIN + scheme).
2. Call auth_issuer_health_check for the issuer's current status.
3. If hour_of_day < 8 or > 20, or fraud signals matter, call auth_scheme_decline_patterns.
4. Apply adjustments: 3DS lift, issuer incidents, off-hour fraud.
5. Call emit_auth_score once with p_auth (4 dp), confidence, and reasoning.

Adjustment rules:
- Feature store null  → prior p_auth=0.85, confidence=0.40
- 3DS frictionless    → +0.8 pp
- 3DS challenged      → −0.4 pp
- Elevated declines   → −1.5 pp
- Unknown issuer      → −3.0 pp
- Off-hour fraud >18% → −0.6 pp
- Confidence = min(0.95, 0.5 + volume_30d / 500_000)\
"""


# ── LangChain tools: cost agent ────────────────────────────────────────────

@tool
def cost_interchange_lookup(scheme: str, region: str, card_type: str, mcc: str) -> str:
    """Look up interchange in basis points for (scheme, region, card_type, MCC)."""
    from agents.cost.tools import interchange_lookup
    result = interchange_lookup(scheme, region, card_type, mcc)
    return json.dumps(result) if result is not None else "null"


@tool
def cost_scheme_fee_lookup(scheme: str, region: str) -> str:
    """Look up assessment + acquirer fees in bps for (scheme, region)."""
    from agents.cost.tools import scheme_fee_lookup
    result = scheme_fee_lookup(scheme, region)
    return json.dumps(result) if result is not None else "null"


@tool
def emit_cost_score(scheme: str, interchange_bps: float, assessment_bps: float,
                    acquirer_bps: float, reasoning: str) -> str:
    """Emit the final cost in raw bps. Call exactly once after both lookups."""
    return json.dumps({
        "scheme": scheme,
        "interchange_bps": interchange_bps,
        "assessment_bps":  assessment_bps,
        "acquirer_bps":    acquirer_bps,
        "reasoning":       reasoning,
    })


COST_TOOLS = [cost_interchange_lookup, cost_scheme_fee_lookup, emit_cost_score]

COST_SYSTEM_PROMPT = """\
You are the Cost Agent for a payments routing system.
Compute the all-in merchant cost in basis points for the given scheme.

Process:
1. Call cost_interchange_lookup for (scheme, region, card_type, mcc).
2. Call cost_scheme_fee_lookup for (scheme, region).
3. Call emit_cost_score with the raw bps values and a short reasoning paragraph.

Rules:
- Call BOTH lookups before emitting — never estimate from memory.
- If a lookup returns null, that scheme is not supported; do NOT emit.
- Cite region, card type, MCC tier, and regulatory regime in your reasoning.
- Call emit_cost_score exactly once.\
"""


# ── LangChain tools: fraud agent ───────────────────────────────────────────

@tool
def fraud_velocity_check(bin: str, merchant_id: str) -> str:
    """Return transaction velocity for this BIN at this merchant (last 1h and 24h)."""
    from agents.fraud.tools import velocity_check
    return json.dumps(velocity_check(bin, merchant_id))


@tool
def fraud_geo_anomaly_check(issuer_country: str, region: str) -> str:
    """Return geo-risk increment when card issuer country doesn't match the transaction region."""
    from agents.fraud.tools import geo_anomaly_check
    return json.dumps(geo_anomaly_check(issuer_country, region))


@tool
def fraud_device_risk_score(channel: str, three_ds_status: str) -> str:
    """Return the base fraud risk contribution from channel and 3DS status combination."""
    from agents.fraud.tools import device_risk_score
    return json.dumps(device_risk_score(channel, three_ds_status))


@tool
def fraud_scheme_defense(scheme: str) -> str:
    """Return the fraud-detection network strength and notes for this payment scheme."""
    from agents.fraud.tools import scheme_fraud_defense
    return json.dumps(scheme_fraud_defense(scheme))


@tool
def emit_fraud_score(scheme: str, p_fraud: float, confidence: float, reasoning: str) -> str:
    """Emit the final fraud risk score. Call exactly once per scheme after all tools."""
    return json.dumps({
        "scheme": scheme, "p_fraud": p_fraud,
        "confidence": confidence, "reasoning": reasoning,
    })


FRAUD_TOOLS = [
    fraud_velocity_check,
    fraud_geo_anomaly_check,
    fraud_device_risk_score,
    fraud_scheme_defense,
    emit_fraud_score,
]

FRAUD_SYSTEM_PROMPT = """\
You are the Fraud Risk Agent for a payments routing system.
Estimate p_fraud — the probability this transaction is fraudulent if routed via the given scheme.

Process:
1. Call fraud_device_risk_score for base risk from channel + 3DS status.
2. Call fraud_velocity_check to see if this BIN has unusual activity at this merchant.
3. Call fraud_geo_anomaly_check to detect cross-border card usage.
4. Call fraud_scheme_defense to get this scheme's fraud-detection network strength.
5. Call emit_fraud_score once with p_fraud (4 dp), confidence, and reasoning.

Adjustment rules:
- Base from device_risk_score (ecommerce + no 3DS = 0.12, POS = 0.01–0.03)
- Velocity > 5 txns/hr       → +0.15
- Velocity 2–5 txns/hr       → +0.05
- Cross-border geo risk       → add geo_risk_increment from tool result
- Stronger network defense    → multiply base by (1 − (strength − 0.85) × 0.5)
- Confidence = 0.70 (fraud signals are noisier than auth rates)\
"""


# ── Helpers: extract emit results from ReAct message history ───────────────

def _extract_auth_score(messages: list) -> AuthScore | None:
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "emit_auth_score":
            d = json.loads(msg.content)
            return AuthScore(
                scheme=d["scheme"],
                p_auth=round(float(d["p_auth"]), 4),
                confidence=round(float(d["confidence"]), 2),
                reasoning=d["reasoning"],
            )
    return None


def _extract_cost_score(messages: list, txn_amount_minor: int) -> CostScore | None:
    from agents.cost.tools import normalise_fee
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "emit_cost_score":
            d = json.loads(msg.content)
            breakdown = normalise_fee(
                interchange_bps=float(d["interchange_bps"]),
                assessment_bps=float(d["assessment_bps"]),
                acquirer_bps=float(d["acquirer_bps"]),
                txn_amount_minor=txn_amount_minor,
            )
            return CostScore(
                scheme=d["scheme"],
                total_fee_bps=breakdown["total_bps"],
                breakdown=breakdown,
                reasoning=d["reasoning"],
            )
    return None


def _extract_fraud_score(messages: list) -> FraudScore | None:
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "emit_fraud_score":
            d = json.loads(msg.content)
            return FraudScore(
                scheme=d["scheme"],
                p_fraud=round(float(d["p_fraud"]), 4),
                confidence=round(float(d["confidence"]), 2),
                reasoning=d["reasoning"],
            )
    return None


# ── Node 1: validate_input (guardrail) ────────────────────────────────────

def validate_input(state: PipelineState) -> dict:
    from Guardrail import validate_transaction, GuardrailViolation
    base = {
        "candidates": [], "planner_reasoning": None,
        "auth_scores": [], "cost_scores": [], "fraud_scores": [],
        "ranked": [], "reflection": None, "decision": None, "explanation": None,
        "error": None, "degraded": False, "hard_blocked": False, "guardrail_warnings": [],
        "auth_revision_count": 0, "auth_critique": None,
    }
    try:
        warnings = validate_transaction(state["txn"])
        return {**base, "guardrail_warnings": [vars(w) for w in warnings]}
    except GuardrailViolation as e:
        return {**base, "error": f"guardrail: {e.code} — {e.message}", "hard_blocked": True}


# ── Node 2: plan_candidates (LLM planner) ─────────────────────────────────

def plan_candidates(state: PipelineState) -> dict:
    txn      = state["txn"]
    all_caps = list(txn.card_brand_capabilities)
    cfg      = get_config()

    base = {
        "degraded":          len(all_caps) < 2,
        "auth_scores":       [],
        "cost_scores":       [],
        "fraud_scores":      [],
        "ranked":            [],
        "decision":          None,
        "error":             None,
    }

    if not all_caps:
        return {**base, "candidates": [], "error": "no candidate schemes",
                "planner_reasoning": "no schemes on card"}

    if not cfg.use_gemini or len(all_caps) <= 1:
        return {**base, "candidates": all_caps,
                "planner_reasoning": "all schemes selected (mock mode or single candidate)"}

    llm    = _chat_model(MODEL_ORCHESTRATOR)
    prompt = (
        f"You are the Candidate Planner for a payments routing system.\n\n"
        f"Decide which payment schemes to evaluate for this transaction.\n"
        f"Only SKIP a scheme if there is an obvious reason it will fail (e.g. Amex at a "
        f"non-OptBlue merchant, Discover in EU). When in doubt, include all.\n\n"
        f"Transaction:\n"
        f"  card supports: {all_caps}\n"
        f"  merchant: {txn.merchant_id}, MCC: {txn.mcc}\n"
        f"  amount: {txn.amount_minor / 100:.2f} {txn.currency}\n"
        f"  region: {txn.region}, channel: {txn.channel}\n"
        f"  3DS: {txn.three_ds_status}\n\n"
        f"Respond with ONLY a JSON object (no markdown):\n"
        f'  {{"candidates": ["visa", "mastercard"], "reasoning": "one sentence"}}'
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        match = re.search(r"\{.*?\}", response.content, re.DOTALL)
        if match:
            data       = json.loads(match.group())
            candidates = [c for c in data.get("candidates", all_caps) if c in all_caps]
            if not candidates:
                candidates = all_caps
            return {**base, "candidates": candidates,
                    "planner_reasoning": data.get("reasoning", "")}
    except Exception:
        pass

    return {**base, "candidates": all_caps,
            "planner_reasoning": "planner failed; using all candidates"}


# ── Node 3: run_auth_agent (ReAct) ─────────────────────────────────────────

def run_auth_agent(state: PipelineState) -> dict:
    txn = state["txn"]
    cfg = get_config()

    # Day 7: are we on a revision attempt? If so, grab the critique to
    # include in the agent's context. Otherwise this is the first pass.
    critique  = state.get("auth_critique")
    revision  = state.get("auth_revision_count", 0)
    is_retry  = critique is not None and revision == 0

    if not cfg.use_gemini:
        from agents.auth_score.agent import _mock_score_one
        scores = [_mock_score_one(txn, s) for s in state["candidates"]]
        # On retry in mock mode the deterministic scorer would return the
        # same answer (the critic is rule-based, the scorer is rule-based —
        # they don't disagree about the same input). Just pass through and
        # bump the counter so we don't infinite-loop.
        return {
            "auth_scores":         scores,
            "auth_revision_count": revision + (1 if is_retry else 0),
        }

    llm   = _chat_model(MODEL_AUTH_AGENT)
    agent = create_react_agent(llm, AUTH_TOOLS, prompt=AUTH_SYSTEM_PROMPT)
    scores: list[AuthScore] = []

    for scheme in state["candidates"]:
        # On retry, prefix the user message with the critic's feedback so
        # the agent knows what to fix. This is the Reflexion pattern: the
        # critic's feedback joins the message history for the next attempt.
        retry_prefix = ""
        if is_retry:
            retry_prefix = (
                f"PREVIOUS ATTEMPT FAILED CRITIQUE — please revise:\n"
                f"{critique}\n\n"
                f"Re-evaluate carefully. Re-call tools if needed. "
                f"Address the issues above before emitting.\n\n"
            )

        msg = HumanMessage(content=(
            retry_prefix +
            f"Transaction: bin={txn.bin}, hour_of_day={txn.hour_of_day}, "
            f"three_ds_status={txn.three_ds_status}, amount={txn.amount_minor} {txn.currency}, "
            f"channel={txn.channel}, region={txn.region}\n"
            f"Scheme to score: {scheme}\n"
            "Gather data via tools, then call emit_auth_score."
        ))
        result = agent.invoke({"messages": [msg]})
        score  = _extract_auth_score(result["messages"])
        if score is None:
            score = AuthScore(scheme=scheme, p_auth=0.85, confidence=0.30,
                              reasoning="ReAct loop did not emit; conservative prior applied.")
        scores.append(score)

    return {
        "auth_scores":         scores,
        "auth_revision_count": revision + (1 if is_retry else 0),
    }


# ── Node 3b: critique_auth_score (Day 7 — Reflexion-style self-correction) ─
#
# After run_auth_agent emits scores, a critic LLM checks them against the
# supporting data. If the critique flags an issue, the conditional edge
# routes back to run_auth_agent for ONE revision attempt (auth_revision_count
# caps the loop).
#
# Why a separate critic instead of asking the auth agent to self-check?
#   Two LLM passes with different prompts catch different errors. A single
#   agent that's told "double-check yourself" tends to confirm its first
#   answer (sycophancy bias). Splitting the role is the standard fix from
#   the Reflexion paper (Shinn et al. 2023) and from later work on
#   LLM-as-judge reliability.
#
# Why cap at one revision?
#   Empirically, the second revision rarely improves over the first
#   when both share an underlying retrieval blind spot. Capping at 1
#   bounds latency to a predictable 2x and avoids unbounded loops.
#
# Mock-mode behaviour:
#   The critic is rule-based — it checks for: (a) p_auth wildly off the
#   baseline, (b) reasoning doesn't reference issuer health when status
#   is unhealthy, (c) score outside [0.50, 0.99] without justification.
#   This means tests still run end-to-end without API calls.

def critique_auth_score(state: PipelineState) -> dict:
    """
    Evaluate every auth_score against its supporting data. If any score
    looks wrong, return a critique string that the conditional edge
    will use to route back to run_auth_agent for one revision attempt.

    Sets auth_critique = None when scores look good (allows pipeline to
    proceed). Sets auth_critique = "<feedback>" to trigger a retry.
    """
    auth_scores = state.get("auth_scores") or []
    if not auth_scores:
        return {"auth_critique": None}

    txn = state["txn"]
    issues: list[str] = []

    # ─── Rule-based checks (always run, even in live mode — fast & cheap) ─

    from agents.auth_score.tools import feature_store_lookup, issuer_health_check

    for score in auth_scores:
        feats  = feature_store_lookup(txn.bin, score.scheme)
        health = issuer_health_check(txn.bin, score.scheme)

        # Bound check: scores outside plausible range with high confidence
        if score.p_auth < 0.50 or score.p_auth > 0.99:
            issues.append(
                f"{score.scheme}: p_auth={score.p_auth:.4f} is outside the "
                f"plausible range [0.50, 0.99]; verify the baseline."
            )

        # Baseline drift: agent's score wildly off from the feature-store rate.
        # 5pp tolerance is generous given documented adjustments (3DS, health, etc.)
        if feats is not None:
            baseline = feats["rate_30d"]
            drift    = abs(score.p_auth - baseline)
            if drift > 0.05:  # 5pp threshold
                issues.append(
                    f"{score.scheme}: p_auth={score.p_auth:.4f} drifts {drift*100:.1f}pp "
                    f"from the 30d baseline ({baseline:.4f}); justify the adjustment."
                )

        # Reasoning sanity: if issuer is unhealthy, reasoning must mention it
        if health["status"] == "elevated_declines":
            if "elevated" not in score.reasoning.lower() and "decline" not in score.reasoning.lower():
                issues.append(
                    f"{score.scheme}: issuer reports elevated declines but the "
                    f"reasoning doesn't reference issuer health."
                )

    if not issues:
        return {"auth_critique": None}

    # Pack the feedback into a single string the agent will see on retry.
    critique = "Auth-score critique:\n" + "\n".join(f"  • {iss}" for iss in issues)
    return {"auth_critique": critique}


def _route_after_auth_critique(state: PipelineState) -> str:
    """
    Conditional edge after the critic.

    Loops back to run_auth_agent IF (a) the critic flagged issues AND
    (b) we haven't used our one revision attempt yet. Otherwise proceeds
    to validate_agent_outputs.
    """
    if state.get("auth_critique") and state.get("auth_revision_count", 0) < 1:
        return "run_auth_agent"
    return "validate_agent_outputs"


def run_cost_agent(state: PipelineState) -> dict:
    txn = state["txn"]
    cfg = get_config()

    if not cfg.use_gemini:
        from agents.cost.agent import _mock_score_one
        return {
            "cost_scores": [
                s for s in (_mock_score_one(txn, c) for c in state["candidates"])
                if s is not None
            ]
        }

    llm   = _chat_model(MODEL_COST_AGENT)
    agent = create_react_agent(llm, COST_TOOLS, prompt=COST_SYSTEM_PROMPT)
    scores: list[CostScore] = []

    for scheme in state["candidates"]:
        msg = HumanMessage(content=(
            f"Transaction: region={txn.region}, card_type={txn.card_type}, "
            f"mcc={txn.mcc}, amount={txn.amount_minor} {txn.currency}\n"
            f"Scheme to score: {scheme}\n"
            "Call both lookups, then call emit_cost_score."
        ))
        result = agent.invoke({"messages": [msg]})
        score  = _extract_cost_score(result["messages"], txn.amount_minor)
        if score is not None:
            scores.append(score)

    return {"cost_scores": scores}


# ── Node 5: run_fraud_agent (ReAct) ────────────────────────────────────────

def run_fraud_agent(state: PipelineState) -> dict:
    txn = state["txn"]
    cfg = get_config()

    if not cfg.use_gemini:
        from agents.fraud.agent import _mock_score_one
        return {"fraud_scores": [_mock_score_one(txn, s) for s in state["candidates"]]}

    llm   = _chat_model(MODEL_AUTH_AGENT)   # Flash for speed — fraud signals are simpler
    agent = create_react_agent(llm, FRAUD_TOOLS, prompt=FRAUD_SYSTEM_PROMPT)
    scores: list[FraudScore] = []

    for scheme in state["candidates"]:
        msg = HumanMessage(content=(
            f"Transaction: bin={txn.bin}, merchant_id={txn.merchant_id}, "
            f"issuer_country={txn.issuer_country}, region={txn.region}, "
            f"channel={txn.channel}, three_ds_status={txn.three_ds_status}, "
            f"amount={txn.amount_minor} {txn.currency}\n"
            f"Scheme to score: {scheme}\n"
            "Gather fraud signals via tools, then call emit_fraud_score."
        ))
        result = agent.invoke({"messages": [msg]})
        score  = _extract_fraud_score(result["messages"])
        if score is None:
            score = FraudScore(scheme=scheme, p_fraud=0.05, confidence=0.30,
                               reasoning="ReAct loop did not emit; conservative prior applied.")
        scores.append(score)

    return {"fraud_scores": scores}


# ── Node 6: validate_agent_outputs (guardrail) ─────────────────────────────

def validate_agent_outputs(state: PipelineState) -> dict:
    from Guardrail import validate_auth_scores, validate_cost_scores, GuardrailViolation
    candidates = state["candidates"]

    try:
        auth_scores = validate_auth_scores(state["auth_scores"], candidates)
    except GuardrailViolation as e:
        return {"error": f"auth guardrail: {e.code}", "hard_blocked": True}

    try:
        cost_scores = validate_cost_scores(state["cost_scores"], candidates)
    except GuardrailViolation as e:
        return {"error": f"cost guardrail: {e.code}", "hard_blocked": True}

    return {"auth_scores": auth_scores, "cost_scores": cost_scores}


# ── Node 7: aggregate_scores ───────────────────────────────────────────────

def aggregate_scores(state: PipelineState) -> dict:
    if not state.get("auth_scores") or not state.get("cost_scores"):
        return {"ranked": [], "error": "no overlap between auth and cost agent outputs"}
    ranked = aggregate(state["auth_scores"], state["cost_scores"], state.get("fraud_scores"))
    return {"ranked": ranked}


# ── Node 8: reflect_on_ranking (LLM self-critique) ─────────────────────────

def reflect_on_ranking(state: PipelineState) -> dict:
    ranked = state.get("ranked", [])
    if not ranked:
        return {"reflection": "No schemes ranked — pipeline in error state."}

    cfg = get_config()
    if not cfg.use_gemini:
        flags: list[str] = []
        if all(r.get("p_auth", 1) < 0.70 for r in ranked):
            flags.append("all p_auth scores below 0.70 — potential network issue")
        if len(ranked) == 1:
            flags.append("single scheme available — no routing redundancy")
        p_frauds = [r["p_fraud"] for r in ranked if r.get("p_fraud") is not None]
        if p_frauds and max(p_frauds) > 0.15:
            flags.append(f"elevated fraud risk (max p_fraud={max(p_frauds):.2f})")
        fees = [r["total_fee_bps"] for r in ranked]
        if len(fees) >= 2 and max(fees) > 2 * min(fees):
            flags.append(f"fee outlier: {max(fees):.1f} vs {min(fees):.1f} bps")
        return {"reflection": "Anomalies: " + "; ".join(flags) if flags else "No anomalies detected."}

    txn  = state["txn"]
    llm  = _chat_model(MODEL_AUTH_AGENT)
    summary = [
        {"scheme": r["scheme"], "p_auth": r["p_auth"],
         "p_fraud": r.get("p_fraud"), "fee_bps": r["total_fee_bps"],
         "score": r["weighted_score"]}
        for r in ranked
    ]
    prompt = (
        f"Briefly review this payment routing analysis for anomalies.\n\n"
        f"Transaction: {txn.txn_id}, {txn.amount_minor/100:.2f} {txn.currency}, {txn.region}\n"
        f"Rankings:\n{json.dumps(summary, indent=2)}\n\n"
        f"Flag if: any p_auth < 0.70, any p_fraud > 0.15, single scheme, or fee 2× outlier.\n"
        f"If clean, respond: 'No anomalies detected.'\n"
        f"Max 2 sentences."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"reflection": response.content.strip()}


# ── Node 9: check_hitl_gate (human-in-the-loop) ────────────────────────────

HIGH_VALUE_THRESHOLD = 50_000   # $500 in cents

def check_hitl_gate(state: PipelineState) -> dict:
    txn = state["txn"]
    if txn.amount_minor >= HIGH_VALUE_THRESHOLD:
        human_decision = interrupt({
            "type":         "hitl_review",
            "txn_id":       txn.txn_id,
            "amount_minor": txn.amount_minor,
            "currency":     txn.currency,
            "ranked":       state.get("ranked", []),
            "reflection":   state.get("reflection"),
            "message": (
                f"High-value transaction ${txn.amount_minor/100:.2f} {txn.currency} "
                "requires human approval before compliance routing."
            ),
        })
        if human_decision.get("approved") is False:
            return {
                "error": f"Transaction {txn.txn_id} rejected by human reviewer",
                "decision": None,
            }
    return {}


# ── Node 10: run_compliance ────────────────────────────────────────────────

def run_compliance(state: PipelineState) -> dict:
    if state.get("error") or not state.get("ranked"):
        return {}
    try:
        decision          = compliance_gate(state["txn"], state["ranked"])
        decision.degraded = state.get("degraded", False)
        return {"decision": decision}
    except NoEligibleSchemeError as e:
        return {"error": str(e), "decision": None}


# ── Node 11: validate_decision (guardrail) ─────────────────────────────────

def validate_decision_node(state: PipelineState) -> dict:
    from Guardrail import validate_decision as _validate, GuardrailViolation
    decision = state.get("decision")
    if decision is None:
        return {}
    try:
        _validate(decision, state["candidates"])
        return {}
    except GuardrailViolation as e:
        return {"error": f"decision guardrail: {e.code}", "hard_blocked": True, "decision": None}


# ── Node 12: generate_explanation ─────────────────────────────────────────

def generate_explanation(state: PipelineState) -> dict:
    decision = state.get("decision")
    txn      = state["txn"]
    ranked   = state.get("ranked", [])
    cfg      = get_config()

    if decision is None:
        return {"explanation": None}

    if not cfg.use_gemini:
        winner   = next((r for r in ranked if r["scheme"] == decision.scheme), {})
        fraud_str = (
            f", fraud risk {winner['p_fraud']:.1%}" if winner.get("p_fraud") is not None else ""
        )
        rejected_str = (
            f" Rejected: {', '.join(decision.rejected_schemes.keys())}."
            if decision.rejected_schemes else ""
        )
        return {
            "explanation": (
                f"Routed via {decision.scheme.upper()} because it offered the best combination of "
                f"approval probability ({decision.p_auth:.1%}), processing cost "
                f"({decision.fee_bps:.1f} bps){fraud_str}. "
                f"All compliance checks passed.{rejected_str}"
            )
        }

    winner = next((r for r in ranked if r["scheme"] == decision.scheme), None)
    llm    = _chat_model(MODEL_AUTH_AGENT)
    summary = [
        {"scheme": r["scheme"], "p_auth": r["p_auth"],
         "p_fraud": r.get("p_fraud"), "fee_bps": r["total_fee_bps"],
         "score": r["weighted_score"]}
        for r in ranked
    ]
    prompt = (
        f"Write a 2-sentence merchant-friendly explanation of this routing decision.\n\n"
        f"Transaction: {txn.amount_minor/100:.2f} {txn.currency}, {txn.channel}, "
        f"MCC {txn.mcc}, {txn.region}\n"
        f"Chosen: {decision.scheme.upper()} — p_auth={decision.p_auth:.3f}, "
        f"fee={decision.fee_bps:.1f} bps\n"
        f"Rejected: {decision.rejected_schemes or 'none'}\n"
        f"Rankings: {json.dumps(summary)}\n\n"
        f"Start with: 'Routed via {decision.scheme.upper()} because...'\n"
        f"Be factual, no jargon."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    explanation_text = response.content.strip()
    decision.explanation = explanation_text
    return {"explanation": explanation_text}


# ── Conditional edge routing ───────────────────────────────────────────────

def _route_guardrail(state: PipelineState) -> str:
    return END if state.get("hard_blocked") else "plan_candidates"


def _route_after_agent_validation(state: PipelineState) -> str:
    return END if state.get("hard_blocked") else "aggregate_scores"


def _route_after_decision_validation(state: PipelineState) -> str:
    return END if state.get("hard_blocked") else "generate_explanation"


# ── Build & compile ────────────────────────────────────────────────────────

def build_pipeline():
    g = StateGraph(PipelineState)

    g.add_node("validate_input",          validate_input)
    g.add_node("plan_candidates",         plan_candidates)
    g.add_node("run_auth_agent",          run_auth_agent)
    g.add_node("critique_auth_score",     critique_auth_score)   # Day 7
    g.add_node("run_cost_agent",          run_cost_agent)
    g.add_node("run_fraud_agent",         run_fraud_agent)
    g.add_node("validate_agent_outputs",  validate_agent_outputs)
    g.add_node("aggregate_scores",        aggregate_scores)
    g.add_node("reflect_on_ranking",      reflect_on_ranking)
    g.add_node("check_hitl_gate",         check_hitl_gate)
    g.add_node("run_compliance",          run_compliance)
    g.add_node("validate_decision",       validate_decision_node)
    g.add_node("generate_explanation",    generate_explanation)

    g.add_edge(START, "validate_input")
    g.add_conditional_edges("validate_input", _route_guardrail)

    # 3-way parallel fan-out
    g.add_edge("plan_candidates", "run_auth_agent")
    g.add_edge("plan_candidates", "run_cost_agent")
    g.add_edge("plan_candidates", "run_fraud_agent")

    # Day 7 — auth path now goes through critic before joining the fan-in.
    # Cost/fraud bypass the critic and fan in directly.
    g.add_edge("run_auth_agent", "critique_auth_score")
    g.add_conditional_edges(
        "critique_auth_score",
        _route_after_auth_critique,
        # Tell LangGraph the two possible destinations so it can lay out
        # the graph correctly (and so LangSmith renders the loop visibly).
        {
            "run_auth_agent":         "run_auth_agent",
            "validate_agent_outputs": "validate_agent_outputs",
        },
    )
    g.add_edge("run_cost_agent",  "validate_agent_outputs")
    g.add_edge("run_fraud_agent", "validate_agent_outputs")

    g.add_conditional_edges("validate_agent_outputs", _route_after_agent_validation)
    g.add_edge("aggregate_scores",  "reflect_on_ranking")
    g.add_edge("reflect_on_ranking", "check_hitl_gate")
    g.add_edge("check_hitl_gate",   "run_compliance")
    g.add_edge("run_compliance",    "validate_decision")
    g.add_conditional_edges("validate_decision", _route_after_decision_validation)
    g.add_edge("generate_explanation", END)

    return g.compile(checkpointer=MemorySaver())


pipeline = build_pipeline()
