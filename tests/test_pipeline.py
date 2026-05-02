"""
Unit + scenario tests.

Run with:
    cd /home/claude/cso
    python -m pytest tests/ -v
or without pytest:
    python tests/test_pipeline.py
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.models import Transaction
from compliance.rules.all_rules import (
    passes_ifr_with_breakdown, passes_durbin, passes_optblue,
    passes_token_lock, merchant_eligible,
)
from orchestrator.orchestrate import orchestrate
from data.samples import SAMPLES


# ---------- unit tests ----------

def test_ifr_caps_eu_credit():
    txn = SAMPLES[0]  # EU credit
    ok, _ = passes_ifr_with_breakdown(txn, "visa", 30.0)
    assert ok
    ok, reason = passes_ifr_with_breakdown(txn, "visa", 35.0)
    assert not ok and "IFR" in reason


def test_ifr_caps_eu_debit():
    txn = SAMPLES[3]  # EU debit
    ok, _ = passes_ifr_with_breakdown(txn, "mastercard", 20.0)
    assert ok
    ok, _ = passes_ifr_with_breakdown(txn, "mastercard", 25.0)
    assert not ok


def test_ifr_skips_non_eu():
    txn = SAMPLES[2]  # US
    ok, _ = passes_ifr_with_breakdown(txn, "visa", 155.0)
    assert ok


def test_optblue_rejects_unenrolled_merchant():
    txn = SAMPLES[4]  # SmallTown Diner, not OptBlue
    ok, reason = passes_optblue(txn, "amex")
    assert not ok and "OptBlue" in reason


def test_optblue_allows_enrolled_merchant():
    txn = SAMPLES[2]  # Transatlantic Air, on OptBlue
    ok, _ = passes_optblue(txn, "amex")
    assert ok


def test_optblue_noop_for_non_amex():
    txn = SAMPLES[4]
    ok, _ = passes_optblue(txn, "visa")
    assert ok


def test_token_lock_rejects_wrong_network():
    txn = SAMPLES[5]  # tokenised to mastercard
    ok, reason = passes_token_lock(txn, "visa")
    assert not ok and "mastercard" in reason


def test_token_lock_allows_matching_network():
    txn = SAMPLES[5]
    ok, _ = passes_token_lock(txn, "mastercard")
    assert ok


def test_durbin_caps_us_debit():
    txn = SAMPLES[7]  # US debit
    ok, _ = passes_durbin(txn, "visa", 80.0)
    assert ok
    ok, reason = passes_durbin(txn, "visa", 200.0)
    assert not ok and "Durbin" in reason


def test_merchant_eligible_rejects_unknown():
    txn = SAMPLES[0]
    ok, _ = merchant_eligible(txn, "amex")  # mer_42 doesn't accept amex
    assert not ok


# ---------- scenario tests ----------

def _run(txn):
    return asyncio.run(orchestrate(txn))


def test_scenario_dual_brand_picks_higher_auth():
    """Dual-brand card: system picks the scheme with better auth rate."""
    decision, trace = _run(SAMPLES[0])  # bin 424242, visa 92.2% vs mc 93.5%
    assert decision is not None
    assert decision.scheme == "mastercard"
    assert set(trace.candidates) == {"visa", "mastercard"}


def test_scenario_elevated_declines_penalised():
    """Issuer with elevated declines should show in the reasoning."""
    _, trace = _run(SAMPLES[1])  # bin 424242, issuer has elevated_declines
    reasoning_texts = " ".join(a["reasoning"] for a in trace.auth_scores)
    assert "elevated declines" in reasoning_texts


def test_scenario_amex_non_optblue_merchant_rejected():
    """Amex card at non-OptBlue merchant => no decision."""
    decision, trace = _run(SAMPLES[4])
    assert decision is None
    # When no scheme is eligible, the gate raises NoEligibleSchemeError
    # which the orchestrator captures into trace.error. The decision
    # field is None (no scheme survived). Verify the rejection mentions
    # Amex / OptBlue specifically.
    error_text = (trace.error or "").lower()
    decision_text = str(trace.decision or {}).lower()
    assert "optblue" in error_text or "amex" in error_text or "amex" in decision_text


def test_scenario_token_lock_forces_network():
    """Network-tokenised PAN routes only to the token's network."""
    decision, trace = _run(SAMPLES[5])  # token_network=mastercard
    assert decision is not None
    assert decision.scheme == "mastercard"
    # Visa should have been rejected by the token_lock rule
    assert "visa" in decision.rejected_schemes


def test_scenario_single_brand_card_degraded():
    """Card with only one brand => degraded mode but still decides."""
    decision, trace = _run(SAMPLES[6])  # Discover-only
    assert decision is not None
    assert decision.scheme == "discover"
    assert decision.degraded is True


def test_scenario_eu_debit_under_ifr():
    """EU debit transaction: interchange never exceeds 20bps."""
    decision, trace = _run(SAMPLES[3])
    assert decision is not None
    mc_cost = next(c for c in trace.cost_scores if c["scheme"] == "mastercard")
    assert mc_cost["breakdown"]["interchange_bps"] <= 20.0


# ---------- runner ----------

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {t.__name__}  -- {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}  -- {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
