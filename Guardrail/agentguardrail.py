"""
Agent Guardrails for the CSO pipeline.

Guardrails sit at three points in the pipeline:
  1. INPUT    — validates and sanitises the Transaction before any agent runs
  2. AGENT    — checks each LLM agent output is within expected bounds
  3. DECISION — final sanity check on the routing choice before it is returned

Guard modes:
  - Hard block → raises GuardrailViolation; pipeline aborts immediately
  - Soft clamp → logs a GuardrailWarning and replaces the value with a safe fallback;
                 pipeline continues

Prompt-injection detection targets every free-text field on the Transaction
that could be crafted by an external actor (merchant, acquirer system, etc.).
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass

from contracts.models import Transaction, AuthScore, CostScore, Decision

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_SCHEMES: frozenset[str] = frozenset({"visa", "mastercard", "amex", "discover"})
KNOWN_CARD_TYPES: frozenset[str] = frozenset({"credit", "debit"})
KNOWN_CHANNELS: frozenset[str] = frozenset({"ecommerce", "pos"})
KNOWN_3DS_STATUSES: frozenset[str] = frozenset(
    {"authenticated_frictionless", "challenged", "none"}
)
KNOWN_REGIONS: frozenset[str] = frozenset({"EU", "US", "APAC", "LATAM", "MEA"})

MAX_AMOUNT_MINOR: int = 10_000_000_00  # $10M in cents — hard ceiling
MAX_FEE_BPS: float = 500.0            # 5 % — anything above is clearly wrong
MAX_REASONING_LEN: int = 2_000        # chars; prevents token-stuffed responses

# Regex for prompt-injection patterns in string fields
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"you\s+are\s+now\s+a",
        r"disregard\s+(your\s+)?(prior|previous|above)",
        r"system\s*:\s*",               # fake system-prompt header
        r"<\s*/?system\s*>",            # XML-style system tags
        r"\[INST\]|\[/INST\]",          # Llama instruction tokens
        r"###\s*(Human|Assistant|System)",  # ChatML-style tags
        r"--\s*ignore",
        r"jailbreak",
        r"DAN\s+mode",
        r"act\s+as\s+(if\s+you\s+are|a)",
    ]
]

# Regex for suspicious structural payloads smuggled in string fields
_INJECTION_STRUCTURAL: list[re.Pattern] = [
    re.compile(p)
    for p in [
        r";\s*(DROP|DELETE|INSERT|UPDATE|SELECT)\s+",  # SQL injection
        r"<script\b",                                   # XSS
        r"\$\{[^}]+\}",                                # template injection
    ]
]


# ---------------------------------------------------------------------------
# Exceptions & warning dataclass
# ---------------------------------------------------------------------------

class GuardrailViolation(Exception):
    """Raised on a hard-block condition; carries a machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def __repr__(self) -> str:
        return f"GuardrailViolation(code={self.code!r}, message={self.message!r})"


@dataclass
class GuardrailWarning:
    field: str
    original: object
    clamped_to: object
    reason: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_injection(value: str, field_name: str) -> None:
    for pat in _INJECTION_PATTERNS + _INJECTION_STRUCTURAL:
        if pat.search(value):
            raise GuardrailViolation(
                code="PROMPT_INJECTION",
                message=(
                    f"Possible prompt-injection in field '{field_name}': "
                    f"matched pattern /{pat.pattern}/"
                ),
            )


def _clamp(value: float, lo: float, hi: float) -> tuple[float, bool]:
    """Return (clamped_value, was_clamped)."""
    clamped = max(lo, min(hi, value))
    return clamped, clamped != value


# ---------------------------------------------------------------------------
# 1. Input guardrail
# ---------------------------------------------------------------------------

def validate_transaction(txn: Transaction) -> list[GuardrailWarning]:
    """
    Hard-block obvious invalid inputs; soft-warn on suspicious-but-usable ones.
    Returns a (possibly empty) list of GuardrailWarnings; raises GuardrailViolation
    on anything that must stop the pipeline.
    """
    warnings: list[GuardrailWarning] = []

    # --- txn_id ---
    if not txn.txn_id or not txn.txn_id.strip():
        raise GuardrailViolation("INVALID_TXN_ID", "txn_id must not be empty")
    if len(txn.txn_id) > 64:
        raise GuardrailViolation("INVALID_TXN_ID", "txn_id exceeds 64 characters")
    if not re.match(r'^[\w\-]+$', txn.txn_id):
        raise GuardrailViolation(
            "INVALID_TXN_ID",
            "txn_id contains characters outside [a-zA-Z0-9_-]",
        )

    # --- BIN ---
    if not re.match(r'^\d{6}$', txn.bin):
        raise GuardrailViolation(
            "INVALID_BIN",
            f"BIN must be exactly 6 digits, got: {txn.bin!r}",
        )

    # --- card_type ---
    if txn.card_type not in KNOWN_CARD_TYPES:
        raise GuardrailViolation(
            "INVALID_CARD_TYPE",
            f"card_type must be one of {sorted(KNOWN_CARD_TYPES)}, got: {txn.card_type!r}",
        )

    # --- card_brand_capabilities ---
    if not txn.card_brand_capabilities:
        raise GuardrailViolation(
            "NO_BRAND_CAPABILITIES",
            "card_brand_capabilities must not be empty",
        )
    unknown_schemes = set(txn.card_brand_capabilities) - KNOWN_SCHEMES
    if unknown_schemes:
        raise GuardrailViolation(
            "UNKNOWN_SCHEME",
            f"card_brand_capabilities contains unknown schemes: {sorted(unknown_schemes)}",
        )

    # --- merchant_id ---
    if not txn.merchant_id or not txn.merchant_id.strip():
        raise GuardrailViolation("INVALID_MERCHANT_ID", "merchant_id must not be empty")
    if len(txn.merchant_id) > 64:
        raise GuardrailViolation("INVALID_MERCHANT_ID", "merchant_id exceeds 64 characters")
    _check_injection(txn.merchant_id, "merchant_id")

    # --- MCC ---
    if not re.match(r'^\d{4}$', txn.mcc):
        raise GuardrailViolation(
            "INVALID_MCC",
            f"MCC must be exactly 4 digits, got: {txn.mcc!r}",
        )

    # --- amount ---
    if txn.amount_minor <= 0:
        raise GuardrailViolation(
            "INVALID_AMOUNT",
            f"amount_minor must be positive, got: {txn.amount_minor}",
        )
    if txn.amount_minor > MAX_AMOUNT_MINOR:
        raise GuardrailViolation(
            "AMOUNT_EXCEEDS_CEILING",
            f"amount_minor {txn.amount_minor} exceeds hard ceiling {MAX_AMOUNT_MINOR}",
        )

    # --- currency ---
    if not re.match(r'^[A-Z]{3}$', txn.currency):
        raise GuardrailViolation(
            "INVALID_CURRENCY",
            f"currency must be a 3-letter ISO code, got: {txn.currency!r}",
        )

    # --- channel ---
    if txn.channel not in KNOWN_CHANNELS:
        raise GuardrailViolation(
            "INVALID_CHANNEL",
            f"channel must be one of {sorted(KNOWN_CHANNELS)}, got: {txn.channel!r}",
        )

    # --- 3DS status ---
    if txn.three_ds_status not in KNOWN_3DS_STATUSES:
        raise GuardrailViolation(
            "INVALID_3DS_STATUS",
            f"three_ds_status must be one of {sorted(KNOWN_3DS_STATUSES)}, "
            f"got: {txn.three_ds_status!r}",
        )

    # --- region ---
    if txn.region not in KNOWN_REGIONS:
        warnings.append(GuardrailWarning(
            field="region",
            original=txn.region,
            clamped_to=txn.region,
            reason=f"Unrecognised region {txn.region!r}; compliance rules may not fire correctly",
        ))

    # --- country codes ---
    for fname in ("issuer_country", "acquirer_country"):
        val = getattr(txn, fname)
        if not re.match(r'^[A-Z]{2}$', val):
            raise GuardrailViolation(
                "INVALID_COUNTRY_CODE",
                f"{fname} must be a 2-letter ISO-3166 code, got: {val!r}",
            )

    # --- hour_of_day ---
    if not (0 <= txn.hour_of_day <= 23):
        raise GuardrailViolation(
            "INVALID_HOUR",
            f"hour_of_day must be 0-23, got: {txn.hour_of_day}",
        )

    # --- token_network consistency ---
    if txn.is_network_token and not txn.token_network:
        raise GuardrailViolation(
            "MISSING_TOKEN_NETWORK",
            "is_network_token=True but token_network is not set",
        )
    if not txn.is_network_token and txn.token_network:
        warnings.append(GuardrailWarning(
            field="token_network",
            original=txn.token_network,
            clamped_to=None,
            reason="token_network set but is_network_token=False; token_network will be ignored",
        ))

    if warnings:
        for w in warnings:
            log.warning("InputGuardrail soft-flag | field=%s reason=%s", w.field, w.reason)

    return warnings


# ---------------------------------------------------------------------------
# 2. Auth-score agent output guardrail
# ---------------------------------------------------------------------------

def validate_auth_scores(
    scores: list[AuthScore],
    candidates: list[str],
) -> list[AuthScore]:
    """
    Validate/clamp auth-score agent outputs. Returns the cleaned list.
    Hard-blocks on missing schemes or extreme structural problems.
    Soft-clamps out-of-range floats.
    """
    if not scores:
        raise GuardrailViolation(
            "NO_AUTH_SCORES",
            "Auth-score agent returned an empty list",
        )

    seen_schemes: set[str] = set()
    cleaned: list[AuthScore] = []

    for score in scores:
        # Scheme must be from the candidate list
        if score.scheme not in candidates:
            log.warning(
                "AuthScoreGuardrail: dropping score for unknown scheme %r (not in candidates)",
                score.scheme,
            )
            continue

        # No duplicates
        if score.scheme in seen_schemes:
            log.warning(
                "AuthScoreGuardrail: dropping duplicate score for scheme %r", score.scheme
            )
            continue
        seen_schemes.add(score.scheme)

        # Clamp p_auth to [0, 1]
        p_auth, clamped = _clamp(score.p_auth, 0.0, 1.0)
        if clamped:
            log.warning(
                "AuthScoreGuardrail: p_auth=%s for scheme %r is out of [0,1]; clamped to %s",
                score.p_auth, score.scheme, p_auth,
            )
            score = AuthScore(
                scheme=score.scheme, p_auth=round(p_auth, 4),
                confidence=score.confidence, reasoning=score.reasoning,
            )

        # Clamp confidence to [0, 1]
        conf, clamped = _clamp(score.confidence, 0.0, 1.0)
        if clamped:
            log.warning(
                "AuthScoreGuardrail: confidence=%s for scheme %r is out of [0,1]; clamped to %s",
                score.confidence, score.scheme, conf,
            )
            score = AuthScore(
                scheme=score.scheme, p_auth=score.p_auth,
                confidence=round(conf, 2), reasoning=score.reasoning,
            )

        # Truncate runaway reasoning strings (token-stuffing / jailbreak output)
        reasoning = score.reasoning
        if len(reasoning) > MAX_REASONING_LEN:
            log.warning(
                "AuthScoreGuardrail: reasoning for %r truncated from %d to %d chars",
                score.scheme, len(reasoning), MAX_REASONING_LEN,
            )
            reasoning = reasoning[:MAX_REASONING_LEN] + "… [truncated by guardrail]"
            score = AuthScore(
                scheme=score.scheme, p_auth=score.p_auth,
                confidence=score.confidence, reasoning=reasoning,
            )

        # Warn on suspiciously extreme p_auth values (not a hard block)
        if score.p_auth < 0.05:
            log.warning(
                "AuthScoreGuardrail: p_auth=%.4f for %r is unusually low; check agent output",
                score.p_auth, score.scheme,
            )
        if score.p_auth > 0.999:
            log.warning(
                "AuthScoreGuardrail: p_auth=%.4f for %r is suspiciously perfect",
                score.p_auth, score.scheme,
            )

        cleaned.append(score)

    if not cleaned:
        raise GuardrailViolation(
            "ALL_AUTH_SCORES_INVALID",
            "All auth-score entries were rejected by the guardrail",
        )

    return cleaned


# ---------------------------------------------------------------------------
# 3. Cost-score agent output guardrail
# ---------------------------------------------------------------------------

def validate_cost_scores(
    scores: list[CostScore],
    candidates: list[str],
) -> list[CostScore]:
    """
    Validate/clamp cost-score agent outputs. Returns the cleaned list.
    """
    if not scores:
        raise GuardrailViolation(
            "NO_COST_SCORES",
            "Cost-score agent returned an empty list",
        )

    seen_schemes: set[str] = set()
    cleaned: list[CostScore] = []

    for score in scores:
        if score.scheme not in candidates:
            log.warning(
                "CostScoreGuardrail: dropping score for unknown scheme %r", score.scheme
            )
            continue

        if score.scheme in seen_schemes:
            log.warning(
                "CostScoreGuardrail: dropping duplicate score for scheme %r", score.scheme
            )
            continue
        seen_schemes.add(score.scheme)

        # fee_bps must be non-negative
        if score.total_fee_bps < 0:
            raise GuardrailViolation(
                "NEGATIVE_FEE",
                f"total_fee_bps={score.total_fee_bps} for scheme {score.scheme!r} is negative",
            )

        # Soft-clamp absurdly high fees
        fee, clamped = _clamp(score.total_fee_bps, 0.0, MAX_FEE_BPS)
        if clamped:
            log.warning(
                "CostScoreGuardrail: total_fee_bps=%.2f for %r exceeds ceiling %.2f; clamped",
                score.total_fee_bps, score.scheme, MAX_FEE_BPS,
            )
            score = CostScore(
                scheme=score.scheme, total_fee_bps=round(fee, 4),
                breakdown=score.breakdown, reasoning=score.reasoning,
            )

        # Breakdown values must all be non-negative
        bad_breakdown = {k: v for k, v in (score.breakdown or {}).items() if v < 0}
        if bad_breakdown:
            log.warning(
                "CostScoreGuardrail: negative breakdown values for %r: %s; zeroing out",
                score.scheme, bad_breakdown,
            )
            clean_breakdown = {k: max(0.0, v) for k, v in (score.breakdown or {}).items()}
            score = CostScore(
                scheme=score.scheme, total_fee_bps=score.total_fee_bps,
                breakdown=clean_breakdown, reasoning=score.reasoning,
            )

        # Truncate reasoning
        reasoning = score.reasoning
        if len(reasoning) > MAX_REASONING_LEN:
            reasoning = reasoning[:MAX_REASONING_LEN] + "… [truncated by guardrail]"
            score = CostScore(
                scheme=score.scheme, total_fee_bps=score.total_fee_bps,
                breakdown=score.breakdown, reasoning=reasoning,
            )

        cleaned.append(score)

    if not cleaned:
        raise GuardrailViolation(
            "ALL_COST_SCORES_INVALID",
            "All cost-score entries were rejected by the guardrail",
        )

    return cleaned


# ---------------------------------------------------------------------------
# 4. Decision guardrail
# ---------------------------------------------------------------------------

def validate_decision(decision: Decision, candidates: list[str]) -> None:
    """
    Final check on the routing decision. Hard-blocks any decision that would
    route to an unchecked scheme or carry nonsensical financial values.
    """
    if decision.scheme not in candidates:
        raise GuardrailViolation(
            "DECISION_UNKNOWN_SCHEME",
            f"Decision scheme {decision.scheme!r} is not in the candidate list {candidates}",
        )

    if not (0.0 <= decision.p_auth <= 1.0):
        raise GuardrailViolation(
            "DECISION_INVALID_P_AUTH",
            f"Decision p_auth={decision.p_auth} is outside [0, 1]",
        )

    if decision.fee_bps < 0:
        raise GuardrailViolation(
            "DECISION_NEGATIVE_FEE",
            f"Decision fee_bps={decision.fee_bps} is negative",
        )

    if decision.fee_bps > MAX_FEE_BPS:
        raise GuardrailViolation(
            "DECISION_FEE_TOO_HIGH",
            f"Decision fee_bps={decision.fee_bps} exceeds ceiling {MAX_FEE_BPS}",
        )

    if not decision.compliance_passed:
        raise GuardrailViolation(
            "COMPLIANCE_NOT_PASSED",
            "Decision has compliance_passed=False; should not reach this point",
        )

    if not decision.txn_id or not decision.txn_id.strip():
        raise GuardrailViolation(
            "DECISION_MISSING_TXN_ID",
            "Decision is missing txn_id",
        )


# ---------------------------------------------------------------------------
# 5. Prompt-injection scanner (standalone, usable in CI or unit tests)
# ---------------------------------------------------------------------------

def scan_for_injection(text: str) -> list[str]:
    """
    Return a list of matched pattern descriptions. Empty list means clean.
    Useful for testing individual fields without raising.
    """
    hits: list[str] = []
    for pat in _INJECTION_PATTERNS + _INJECTION_STRUCTURAL:
        if pat.search(text):
            hits.append(pat.pattern)
    return hits


# ---------------------------------------------------------------------------
# 6. High-level wrapper — drop-in replacement for orchestrate()
# ---------------------------------------------------------------------------

async def orchestrate_with_guardrails(txn: Transaction):
    """
    Wraps the full pipeline with input + output guardrails.

    Returns the same (Decision | None, Trace) tuple as orchestrate(), but:
      - raises GuardrailViolation before any LLM call if the input is invalid
      - patches agent outputs in-place after each stage
      - raises GuardrailViolation if the final decision is structurally invalid
    """
    from dataclasses import asdict
    from orchestrator.orchestrate import (
        _plan_candidates,
        _fallback_auth_scores,
        _fallback_cost_scores,
    )
    from orchestrator.aggregate import aggregate
    from compliance.gate import compliance_gate, NoEligibleSchemeError
    from observability.tracer import Trace
    import asyncio

    # ── Stage 0: input guardrail ──────────────────────────────────────────
    input_warnings = validate_transaction(txn)
    trace = Trace(txn_id=txn.txn_id)
    trace.guardrail_warnings = [vars(w) for w in input_warnings]

    # ── Stage 1: candidate selection ──────────────────────────────────────
    candidates = _plan_candidates(txn)
    trace.candidates = candidates
    trace.degraded = len(candidates) < 2

    if not candidates:
        trace.error = "no candidate schemes after guardrail-validated candidate selection"
        return None, trace

    # ── Stage 2: parallel agent dispatch ─────────────────────────────────
    from agents.auth_score.agent import auth_score_agent
    from agents.cost.agent import cost_agent

    auth_task = asyncio.create_task(auth_score_agent(txn, candidates))
    cost_task = asyncio.create_task(cost_agent(txn, candidates))
    auth_scores, cost_scores = await asyncio.gather(
        auth_task, cost_task, return_exceptions=True,
    )

    if isinstance(auth_scores, Exception):
        trace.error = f"auth agent failed: {auth_scores}"
        auth_scores = await _fallback_auth_scores(candidates)
    if isinstance(cost_scores, Exception):
        trace.error = f"cost agent failed: {cost_scores}"
        cost_scores = await _fallback_cost_scores(txn, candidates)

    # ── Stage 3: agent output guardrails ─────────────────────────────────
    try:
        auth_scores = validate_auth_scores(auth_scores, candidates)
    except GuardrailViolation as exc:
        log.error("Auth-score guardrail hard-block: %s", exc)
        trace.error = f"auth guardrail: {exc.code} — {exc.message}"
        return None, trace

    try:
        cost_scores = validate_cost_scores(cost_scores, candidates)
    except GuardrailViolation as exc:
        log.error("Cost-score guardrail hard-block: %s", exc)
        trace.error = f"cost guardrail: {exc.code} — {exc.message}"
        return None, trace

    trace.auth_scores = [asdict(a) for a in auth_scores]
    trace.cost_scores = [asdict(c) for c in cost_scores]

    # ── Stage 4: aggregation ─────────────────────────────────────────────
    ranked = aggregate(auth_scores, cost_scores)
    trace.ranked = ranked
    if not ranked:
        trace.error = "no overlap between auth and cost agent outputs after guardrail cleaning"
        return None, trace

    # ── Stage 5: compliance gate ─────────────────────────────────────────
    try:
        decision = compliance_gate(txn, ranked)
        decision.degraded = trace.degraded
    except NoEligibleSchemeError as exc:
        trace.error = str(exc)
        trace.decision = {"compliance_passed": False, "rejections": exc.rejections}
        return None, trace

    # ── Stage 6: decision guardrail ──────────────────────────────────────
    try:
        validate_decision(decision, candidates)
    except GuardrailViolation as exc:
        log.error("Decision guardrail hard-block: %s", exc)
        trace.error = f"decision guardrail: {exc.code} — {exc.message}"
        return None, trace

    trace.decision = asdict(decision)
    return decision, trace
