"""
Decision-quality evaluator (Layer 2).

Question: when running on a 200-transaction synthetic set, how does the
full CSO pipeline compare to two deterministic baselines?

Baselines:
  cheapest_first   always pick the scheme with the lowest fee_bps
  highest_auth     always pick the scheme with the highest p_auth
                   (without considering fee or fraud)

Metrics:
  avg_p_auth                      mean p(auth) of the chosen scheme
  avg_fee_bps                     mean cost in basis points
  avg_p_fraud                     mean fraud risk
  expected_revenue_per_100        weighted by p_auth × (1 - fee_bps/10000)
  compliance_violation_count      times the chosen scheme would have failed
                                  the deterministic gate (should always be 0
                                  for CSO, can be > 0 for naive baselines)

Why this matters (worth knowing for the viva):
  Headline argument for the orchestrator: "we get higher auth at acceptable
  cost without compliance risk." This eval produces the numbers to back
  that up. The cheapest_first baseline shows what a fee-only optimiser
  would do (often picks compliance-blocked schemes); the highest_auth
  baseline shows what an auth-only optimiser would do (ignores cost).
  CSO's weighted argmax should sit on the Pareto frontier between them.

  The eval is run on synthetic transactions because the SAMPLES set is too
  small (8 rows) to reach statistical significance. The synth generator
  varies region, MCC, amount, channel, 3DS, card type so the distribution
  spans the regulatory landscape.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import asdict
from typing import Any

from contracts.models import Transaction
from data.interchange import MERCHANT_CONTRACTS, MCC_TIERS

log = logging.getLogger(__name__)


# ── Synthetic transaction generator ─────────────────────────────────────────

_BINS = ["411111", "424242", "555555", "540001", "378282", "601100", "450000", "520000"]
_REGIONS = ["EU", "US"]
_CHANNELS = ["ecommerce", "pos"]
_THREE_DS = ["authenticated_frictionless", "challenged", "none"]
_CARD_TYPES = ["credit", "debit"]


def generate_synthetic(n: int = 200, seed: int = 42) -> list[Transaction]:
    """
    Generate `n` synthetic transactions with varied attributes.

    The mix is biased toward cases that exercise compliance rules (mix of
    EU / US, credit / debit, MCC variety). Reproducible via the seed.
    """
    rng = random.Random(seed)
    txns: list[Transaction] = []

    for i in range(n):
        bin_str = rng.choice(_BINS)
        region  = rng.choice(_REGIONS)

        # Build candidate brand-capabilities mix that's plausible for the BIN.
        # Use deterministic mapping by BIN like the loader does.
        scheme_pool = {
            "411111": ["visa"],
            "424242": ["visa", "mastercard"],
            "555555": ["mastercard"],
            "540001": ["mastercard"],
            "378282": ["amex"],
            "601100": ["discover"],
            "450000": ["visa"],
            "520000": ["visa", "mastercard"],
        }[bin_str]

        # 30% of transactions are dual-brand (more interesting), the rest single
        if len(scheme_pool) > 1 and rng.random() < 0.7:
            caps = scheme_pool
        else:
            caps = [scheme_pool[0]]

        merchant_id = rng.choice(list(MERCHANT_CONTRACTS.keys()))
        mcc         = rng.choice(list(MCC_TIERS.keys()))
        amount      = rng.choice([1500, 4999, 8750, 25_000, 49_999, 125_000])
        channel     = rng.choice(_CHANNELS)
        three_ds    = rng.choice(_THREE_DS)
        card_type   = rng.choice(_CARD_TYPES)
        hour        = rng.randint(0, 23)

        currency = "EUR" if region == "EU" else "USD"
        country  = rng.choice(["FR", "DE", "NL"]) if region == "EU" else "US"

        txns.append(Transaction(
            txn_id           = f"synth_{i:04d}",
            bin              = bin_str,
            card_type        = card_type,
            card_brand_capabilities = caps,
            merchant_id      = merchant_id,
            mcc              = mcc,
            amount_minor     = amount,
            currency         = currency,
            channel          = channel,
            three_ds_status  = three_ds,
            region           = region,
            issuer_country   = country,
            acquirer_country = country,
            hour_of_day      = hour,
        ))

    return txns


# ── Baselines ───────────────────────────────────────────────────────────────

def baseline_cheapest_first(ranked: list[dict]) -> dict | None:
    """Pick the scheme with the lowest total fee, ignoring auth & compliance."""
    if not ranked:
        return None
    return min(ranked, key=lambda r: r["total_fee_bps"])


def baseline_highest_auth(ranked: list[dict]) -> dict | None:
    """Pick the scheme with the highest p_auth, ignoring fee & compliance."""
    if not ranked:
        return None
    return max(ranked, key=lambda r: r["p_auth"])


# ── Compliance check (used to flag baseline violations) ─────────────────────

def _violates_compliance(txn: Transaction, scheme: str, ranked_entry: dict) -> bool:
    """
    Run the deterministic compliance rules. Returns True if the scheme
    would be blocked. Used to count baseline violations.
    """
    from compliance.rules.all_rules import (
        passes_ifr_with_breakdown, passes_durbin, passes_optblue,
        passes_token_lock, merchant_eligible,
    )
    interchange = ranked_entry["interchange_bps"]
    checks = [
        merchant_eligible(txn, scheme),
        passes_token_lock(txn, scheme),
        passes_optblue(txn, scheme),
        passes_ifr_with_breakdown(txn, scheme, interchange),
        passes_durbin(txn, scheme, interchange),
    ]
    return any(not ok for ok, _ in checks)


# ── Per-row scoring ─────────────────────────────────────────────────────────

def _row_metrics(txn: Transaction, ranked: list[dict], chosen: dict | None) -> dict:
    if chosen is None:
        return {
            "txn_id":           txn.txn_id,
            "scheme":           None,
            "p_auth":           None,
            "fee_bps":          None,
            "p_fraud":          None,
            "compliance_ok":    None,
            "expected_revenue": None,
        }
    scheme = chosen["scheme"]
    return {
        "txn_id":           txn.txn_id,
        "scheme":           scheme,
        "p_auth":           round(chosen["p_auth"], 4),
        "fee_bps":          round(chosen["total_fee_bps"], 2),
        "p_fraud":          round(chosen.get("p_fraud") or 0.0, 4),
        "compliance_ok":    not _violates_compliance(txn, scheme, chosen),
        # Toy revenue model: p_auth × (1 - fee_bps/10000) — captures
        # auth-rate × margin trade-off in a single number.
        "expected_revenue": round(
            chosen["p_auth"] * (1 - chosen["total_fee_bps"] / 10_000), 4
        ),
    }


# ── Runner ──────────────────────────────────────────────────────────────────

async def run_decision_quality_eval(n: int = 50, seed: int = 42) -> dict:
    """
    Generate n synthetic transactions, run the CSO pipeline + both
    baselines on each, return aggregate metrics.

    Default n=50 (not 200) for speed during development; bump to 200
    for the final report run.
    """
    from orchestrator.orchestrate import orchestrate

    txns = generate_synthetic(n=n, seed=seed)
    cso_rows = []
    cheap_rows = []
    auth_rows = []
    latencies = []

    for txn in txns:
        try:
            t0 = time.time()
            decision, trace = await orchestrate(txn)
            elapsed = time.time() - t0
            latencies.append(elapsed)
        except Exception as exc:
            log.warning("orchestrate failed for %s: %s", txn.txn_id, exc)
            continue

        ranked = trace.ranked or []

        # CSO chose this entry (or None if no scheme passed compliance)
        cso_chosen = next(
            (r for r in ranked if decision and r["scheme"] == decision.scheme),
            None,
        )
        # Baselines pick from the *ranked* list pre-compliance — that's the
        # fair comparison: same agent outputs, different selection rule.
        cheap_chosen = baseline_cheapest_first(ranked)
        auth_chosen  = baseline_highest_auth(ranked)

        cso_rows.append(_row_metrics(txn, ranked, cso_chosen))
        cheap_rows.append(_row_metrics(txn, ranked, cheap_chosen))
        auth_rows.append(_row_metrics(txn, ranked, auth_chosen))

    return {
        "n":          len(cso_rows),
        "cso":        _summarise(cso_rows),
        "cheapest":   _summarise(cheap_rows),
        "highest_auth": _summarise(auth_rows),
        "latency": {
            "p50": round(_pct(latencies, 0.50), 3),
            "p95": round(_pct(latencies, 0.95), 3),
            "p99": round(_pct(latencies, 0.99), 3),
            "mean": round(sum(latencies) / max(len(latencies), 1), 3),
        },
    }


def _summarise(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["p_auth"] is not None]
    if not valid:
        return {"n_decisions": 0}
    return {
        "n_decisions":               len(valid),
        "n_no_decision":             len(rows) - len(valid),
        "avg_p_auth":                round(sum(r["p_auth"] for r in valid) / len(valid), 4),
        "avg_fee_bps":               round(sum(r["fee_bps"] for r in valid) / len(valid), 2),
        "avg_p_fraud":               round(sum(r["p_fraud"] for r in valid) / len(valid), 4),
        "expected_revenue_per_100":  round(100 * sum(r["expected_revenue"] for r in valid) / len(valid), 3),
        "compliance_violation_pct":  round(100 * sum(1 for r in valid if not r["compliance_ok"]) / len(valid), 1),
    }


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = int(q * (len(s) - 1))
    return s[idx]
