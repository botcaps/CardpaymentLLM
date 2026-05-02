"""
Cost Agent.

Live implementation: Gemini 2.5 Pro with function calling for interchange_lookup
and scheme_fee_lookup. The loop runs until the model calls emit_cost_score with
the raw bps values; normalise_fee is computed server-side.

Mock implementation: deterministic stand-in used when GEMINI_API_KEY is
not set or LLM_MODE=mock is forced.
"""
from __future__ import annotations

from langsmith import traceable

from contracts.models import Transaction, CostScore
from agents.cost.tools import (
    interchange_lookup, scheme_fee_lookup, normalise_fee,
)
from llm_clients import get_config, gemini_client, MODEL_COST_AGENT


# ---------- tool schemas (Gemini function_declarations format) ----------
FUNCTION_DECLARATIONS = [
    {
        "name": "interchange_lookup",
        "description": "Look up interchange in basis points for (scheme, region, card_type, MCC).",
        "parameters": {
            "type": "object",
            "properties": {
                "scheme":    {"type": "string"},
                "region":    {"type": "string", "enum": ["EU", "US"]},
                "card_type": {"type": "string", "enum": ["credit", "debit"]},
                "mcc":       {"type": "string"},
            },
            "required": ["scheme", "region", "card_type", "mcc"],
        },
    },
    {
        "name": "scheme_fee_lookup",
        "description": "Look up assessment + acquirer fees in bps for (scheme, region).",
        "parameters": {
            "type": "object",
            "properties": {
                "scheme": {"type": "string"},
                "region": {"type": "string"},
            },
            "required": ["scheme", "region"],
        },
    },
    {
        "name": "emit_cost_score",
        "description": (
            "Emit the final cost score for this scheme. "
            "You MUST have called interchange_lookup and scheme_fee_lookup first. "
            "Provide the raw bps values; the server will compute the currency fee."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scheme":           {"type": "string"},
                "interchange_bps":  {"type": "number"},
                "assessment_bps":   {"type": "number"},
                "acquirer_bps":     {"type": "number"},
                "reasoning":        {"type": "string"},
            },
            "required": ["scheme", "interchange_bps", "assessment_bps",
                         "acquirer_bps", "reasoning"],
        },
    },
]


SYSTEM_PROMPT = """You are the Cost Agent for a payments routing system.

Your job: for the given transaction and candidate scheme, compute the all-in merchant cost in basis points.

Process:
1. Call interchange_lookup for (scheme, region, card_type, mcc).
2. Call scheme_fee_lookup for (scheme, region).
3. Call emit_cost_score with the raw bps values and a short reasoning paragraph.

Rules:
- Always call both lookup tools before emitting. Do NOT estimate rates from memory.
- If a tool returns null, that scheme is not supported for this combo — do not emit.
- Reasoning should cite the region, card type, MCC tier, and the regulatory regime (e.g. EU IFR cap, US standard rates).
- Call emit_cost_score exactly once."""


# ---------- live implementation ----------

def _run_tool(name: str, args: dict):
    if name == "interchange_lookup":
        return interchange_lookup(args["scheme"], args["region"],
                                  args["card_type"], args["mcc"])
    if name == "scheme_fee_lookup":
        return scheme_fee_lookup(args["scheme"], args["region"])
    raise ValueError(f"unknown tool: {name}")


@traceable(run_type="llm", name="gemini-cost-score", metadata={"model": MODEL_COST_AGENT})
async def _score_one_live(txn: Transaction, scheme: str) -> CostScore | None:
    from google.genai import types

    client = gemini_client()

    user_prompt = (
        f"Transaction:\n"
        f"  region: {txn.region}\n"
        f"  card_type: {txn.card_type}\n"
        f"  mcc: {txn.mcc}\n"
        f"  amount_minor: {txn.amount_minor} {txn.currency}\n\n"
        f"Scheme to score: {scheme}\n\n"
        f"Gather the rate card and emit_cost_score."
    )

    contents = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]
    tool = types.Tool(function_declarations=FUNCTION_DECLARATIONS)

    for _ in range(8):
        response = await client.aio.models.generate_content(
            model=MODEL_COST_AGENT,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[tool],
            ),
        )

        model_content = response.candidates[0].content
        contents.append(model_content)

        function_calls = [p.function_call for p in model_content.parts if p.function_call]

        emit_call = next((fc for fc in function_calls if fc.name == "emit_cost_score"), None)
        if emit_call is not None:
            d = dict(emit_call.args)
            breakdown = normalise_fee(
                interchange_bps=float(d["interchange_bps"]),
                assessment_bps=float(d["assessment_bps"]),
                acquirer_bps=float(d["acquirer_bps"]),
                txn_amount_minor=txn.amount_minor,
            )
            return CostScore(
                scheme=d["scheme"],
                total_fee_bps=breakdown["total_bps"],
                breakdown=breakdown,
                reasoning=d["reasoning"],
            )

        if not function_calls:
            return None

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

    return None


# ---------- mock implementation ----------

def _mock_score_one(txn: Transaction, scheme: str) -> CostScore | None:
    ic = interchange_lookup(scheme, txn.region, txn.card_type, txn.mcc)
    if ic is None:
        return None
    fees = scheme_fee_lookup(scheme, txn.region)
    if fees is None:
        return None

    breakdown = normalise_fee(
        interchange_bps=ic,
        assessment_bps=fees["assessment_bps"],
        acquirer_bps=fees["acquirer_bps"],
        txn_amount_minor=txn.amount_minor,
    )
    reasoning = (
        f"{scheme} {txn.region} {txn.card_type} under MCC {txn.mcc}: "
        f"interchange {ic:.1f}bps + assessment {fees['assessment_bps']:.1f}bps "
        f"+ acquirer {fees['acquirer_bps']:.1f}bps = {breakdown['total_bps']:.1f}bps "
        f"({breakdown['fee_minor']} minor units on {txn.amount_minor})."
    )
    return CostScore(
        scheme=scheme,
        total_fee_bps=breakdown["total_bps"],
        breakdown=breakdown,
        reasoning=reasoning,
    )


# ---------- dispatch ----------

@traceable(run_type="chain", name="cost-agent")
async def cost_agent(txn: Transaction, candidates: list[str]) -> list[CostScore]:
    cfg = get_config()
    if cfg.use_gemini:
        import asyncio
        results = await asyncio.gather(*[_score_one_live(txn, s) for s in candidates])
        return [r for r in results if r is not None]
    return [r for r in (_mock_score_one(txn, s) for s in candidates) if r is not None]
