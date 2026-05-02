"""
Auth Score Agent.

Live implementation: Gemini 2.5 Flash with function calling for feature_store_lookup,
issuer_health_check, scheme_decline_patterns. The tool loop runs until the model
calls emit_auth_score, which carries the structured JSON output.

Mock implementation: deterministic stand-in used when GEMINI_API_KEY is
not set or LLM_MODE=mock is forced.
"""
from __future__ import annotations

from langsmith import traceable

from contracts.models import Transaction, AuthScore
from agents.auth_score.tools import (
    feature_store_lookup, issuer_health_check, scheme_decline_patterns,
)
from llm_clients import get_config, gemini_client, MODEL_AUTH_AGENT


# ---------- tool schemas (Gemini function_declarations format) ----------
FUNCTION_DECLARATIONS = [
    {
        "name": "feature_store_lookup",
        "description": "Return rolling 30/90-day auth rates for a BIN on a specific scheme, plus sample size.",
        "parameters": {
            "type": "object",
            "properties": {
                "bin":    {"type": "string", "description": "First 6 digits of the card PAN"},
                "scheme": {"type": "string", "enum": ["visa", "mastercard", "amex", "discover"]},
            },
            "required": ["bin", "scheme"],
        },
    },
    {
        "name": "issuer_health_check",
        "description": "Return current issuer status for the bank behind this BIN. Status can be 'healthy', 'elevated_declines', or 'unknown'.",
        "parameters": {
            "type": "object",
            "properties": {
                "bin":    {"type": "string"},
                "scheme": {"type": "string"},
            },
            "required": ["bin", "scheme"],
        },
    },
    {
        "name": "scheme_decline_patterns",
        "description": "Return the decline-code distribution for this BIN, bucketed by business hours vs off-hours.",
        "parameters": {
            "type": "object",
            "properties": {
                "bin":         {"type": "string"},
                "hour_of_day": {"type": "integer", "minimum": 0, "maximum": 23},
            },
            "required": ["bin", "hour_of_day"],
        },
    },
    {
        "name": "emit_auth_score",
        "description": "Emit the final auth score for one scheme. Call this exactly once per scheme after gathering data.",
        "parameters": {
            "type": "object",
            "properties": {
                "scheme":     {"type": "string"},
                "p_auth":     {"type": "number", "minimum": 0, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reasoning":  {"type": "string"},
            },
            "required": ["scheme", "p_auth", "confidence", "reasoning"],
        },
    },
]


SYSTEM_PROMPT = """You are the Auth Score Agent for a payments routing system.

Your job: for the given transaction and candidate scheme, estimate P(auth | scheme) — the probability the issuer will approve this transaction if routed via that network.

Process:
1. Call feature_store_lookup to get the baseline auth rate for this BIN+scheme.
2. Call issuer_health_check to see if the issuer is currently healthy.
3. If off-hours (hour < 8 or hour > 20) OR fraud signals matter, call scheme_decline_patterns.
4. Reason about adjustments: 3DS lift, issuer incidents, off-hour fraud risk.
5. Call emit_auth_score with the final number, a confidence value, and a one-paragraph reasoning.

Rules:
- If feature_store_lookup returns null, use a conservative prior of 0.85 with confidence 0.4.
- 3DS frictionless adds ~0.8pp; 3DS challenged subtracts ~0.4pp.
- Elevated issuer declines subtract ~1.5pp.
- Elevated off-hour fraud (>18%) subtracts ~0.6pp.
- Confidence scales with sample size: 0.5 + volume_30d/500_000, capped at 0.95.
- Round p_auth to 4 decimals.
- Call emit_auth_score exactly once. Do not emit scores for schemes not asked about."""


# ---------- live implementation ----------

def _run_tool(name: str, inp: dict):
    if name == "feature_store_lookup":
        return feature_store_lookup(inp["bin"], inp["scheme"])
    if name == "issuer_health_check":
        return issuer_health_check(inp["bin"], inp["scheme"])
    if name == "scheme_decline_patterns":
        return scheme_decline_patterns(inp["bin"], inp["hour_of_day"])
    raise ValueError(f"unknown tool: {name}")


@traceable(run_type="llm", name="gemini-auth-score", metadata={"model": MODEL_AUTH_AGENT})
async def _score_one_live(txn: Transaction, scheme: str) -> AuthScore:
    from google.genai import types

    client = gemini_client()

    user_prompt = (
        f"Transaction:\n"
        f"  bin: {txn.bin}\n"
        f"  hour_of_day: {txn.hour_of_day}\n"
        f"  three_ds_status: {txn.three_ds_status}\n"
        f"  amount_minor: {txn.amount_minor} {txn.currency}\n"
        f"  channel: {txn.channel}\n"
        f"  region: {txn.region}\n\n"
        f"Scheme to score: {scheme}\n\n"
        f"Gather data via tools, then call emit_auth_score."
    )

    contents = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]
    tool = types.Tool(function_declarations=FUNCTION_DECLARATIONS)

    for _ in range(8):
        response = await client.aio.models.generate_content(
            model=MODEL_AUTH_AGENT,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[tool],
            ),
        )

        model_content = response.candidates[0].content
        contents.append(model_content)

        function_calls = [p.function_call for p in model_content.parts if p.function_call]

        emit_call = next((fc for fc in function_calls if fc.name == "emit_auth_score"), None)
        if emit_call is not None:
            d = dict(emit_call.args)
            return AuthScore(
                scheme=d["scheme"],
                p_auth=round(float(d["p_auth"]), 4),
                confidence=round(float(d["confidence"]), 2),
                reasoning=d["reasoning"],
            )

        if not function_calls:
            return AuthScore(
                scheme=scheme, p_auth=0.85, confidence=0.40,
                reasoning="Agent produced no structured output; conservative prior applied.",
            )

        response_parts = []
        for fc in function_calls:
            try:
                result = _run_tool(fc.name, dict(fc.args))
            except Exception as e:
                result = {"error": str(e)}
            response_parts.append(
                types.Part(function_response=types.FunctionResponse(
                    name=fc.name,
                    response={"result": result},
                ))
            )
        contents.append(types.Content(role="user", parts=response_parts))

    return AuthScore(
        scheme=scheme, p_auth=0.85, confidence=0.30,
        reasoning="Tool loop exceeded depth cap; conservative prior applied.",
    )


# ---------- mock implementation ----------

def _conservative_prior(scheme: str) -> AuthScore:
    return AuthScore(
        scheme=scheme, p_auth=0.85, confidence=0.40,
        reasoning=f"No historical data for this BIN on {scheme}; applied conservative prior of 0.85.",
    )


def _mock_score_one(txn: Transaction, scheme: str) -> AuthScore:
    feats    = feature_store_lookup(txn.bin, scheme)
    health   = issuer_health_check(txn.bin, scheme)
    declines = scheme_decline_patterns(txn.bin, txn.hour_of_day)

    if feats is None:
        return _conservative_prior(scheme)

    base = feats["rate_30d"]
    adjustments: list[str] = []
    p = base

    if health["status"] == "elevated_declines":
        p -= 0.015
        adjustments.append(f"issuer reports elevated declines ({health['incident']}): -1.5pp")
    elif health["status"] == "unknown":
        p -= 0.03
        adjustments.append("issuer status unknown: -3.0pp")

    if txn.three_ds_status == "authenticated_frictionless":
        p += 0.008
        adjustments.append("3DS frictionless auth: +0.8pp")
    elif txn.three_ds_status == "challenged":
        p -= 0.004
        adjustments.append("3DS challenge abandonment risk: -0.4pp")

    if declines and declines.get("fraud", 0) > 0.18:
        p -= 0.006
        adjustments.append(f"elevated off-hour fraud declines ({declines['fraud']:.0%}): -0.6pp")

    p = max(0.0, min(1.0, p))
    volume = feats.get("volume_30d", 0)
    confidence = min(0.95, 0.5 + volume / 500_000)

    reasoning = (
        f"30-day baseline on {scheme} for this BIN is {base:.1%} "
        f"(n={volume:,}). Adjustments: "
        + ("; ".join(adjustments) if adjustments else "none")
        + f". Final P(auth)={p:.3f}."
    )
    return AuthScore(scheme=scheme, p_auth=round(p, 4),
                     confidence=round(confidence, 2), reasoning=reasoning)


# ---------- dispatch ----------

@traceable(run_type="chain", name="auth-score-agent")
async def auth_score_agent(txn: Transaction, candidates: list[str]) -> list[AuthScore]:
    cfg = get_config()
    if cfg.use_gemini:
        import asyncio
        return await asyncio.gather(*[_score_one_live(txn, s) for s in candidates])
    return [_mock_score_one(txn, s) for s in candidates]
