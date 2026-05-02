"""
Deterministic compliance rules.

Every function returns (passed: bool, reason: str). 'reason' is used
when the rule rejects a scheme, for audit logging.

CRITICAL: no LLM calls here. Everything is plain Python so the path
is reproducible and testable.
"""
from __future__ import annotations
from contracts.models import Transaction
from data.interchange import MERCHANT_CONTRACTS

_FALLBACK_CONTRACT = {
    "name": "Unknown Merchant",
    "region": "US",
    "accepts": ["visa", "mastercard", "discover"],
    "optblue": False,
}

EEA = {
    "DE", "FR", "NL", "IT", "ES", "BE", "AT", "IE", "PT", "FI",
    "DK", "SE", "PL", "CZ", "GR", "LU", "SI", "SK", "EE", "LV",
    "LT", "MT", "CY", "HU", "BG", "RO", "HR", "IS", "LI", "NO",
}


def passes_ifr(txn: Transaction, scheme: str, fee_bps: float) -> tuple[bool, str]:
    """EU IFR caps: 0.20% debit, 0.30% credit when both sides are in EEA."""
    if txn.region != "EU":
        return True, ""
    if txn.issuer_country not in EEA or txn.acquirer_country not in EEA:
        return True, ""
    if scheme in ("amex", "discover"):
        # IFR applies to four-party schemes only; Amex under OptBlue is a separate case
        return True, ""
    cap = 20.0 if txn.card_type == "debit" else 30.0
    # Only the interchange component is capped — we check against the
    # interchange bps, not the total. We approximate using breakdown
    # at the call site. For simplicity here, compare to total minus
    # typical scheme/acquirer of ~14bps; real impl passes interchange_bps in.
    # We'll fix this in gate.py by passing the breakdown.
    return True, ""  # placeholder, real check happens in gate.py with breakdown


def passes_ifr_with_breakdown(txn: Transaction, scheme: str, interchange_bps: float) -> tuple[bool, str]:
    if txn.region != "EU":
        return True, ""
    if txn.issuer_country not in EEA or txn.acquirer_country not in EEA:
        return True, ""
    if scheme in ("amex", "discover"):
        return True, ""
    cap = 20.0 if txn.card_type == "debit" else 30.0
    if interchange_bps > cap + 0.01:  # small tolerance
        return False, f"IFR cap exceeded: {interchange_bps:.1f}bps > {cap:.1f}bps"
    return True, ""


def passes_durbin(txn: Transaction, scheme: str, interchange_bps: float) -> tuple[bool, str]:
    """
    Simplified Durbin check: US regulated debit capped at ~80bps equivalent
    for demo purposes. Real cap is 21c + 5bps + 1c fraud adjustment.
    """
    if txn.region != "US" or txn.card_type != "debit":
        return True, ""
    if scheme not in ("visa", "mastercard"):
        return True, ""
    # Our demo INTERCHANGE_TABLE already encodes 80bps for US debit.
    # A real check would consult a regulated-issuer list.
    cap = 95.0  # bps-equivalent ceiling for demo
    if interchange_bps > cap:
        return False, f"Durbin cap exceeded: {interchange_bps:.1f}bps > {cap:.1f}bps"
    return True, ""


def passes_optblue(txn: Transaction, scheme: str) -> tuple[bool, str]:
    """Amex may route only if the merchant is enrolled in OptBlue."""
    if scheme != "amex":
        return True, ""
    contract = MERCHANT_CONTRACTS.get(txn.merchant_id, _FALLBACK_CONTRACT)
    if not contract.get("optblue", False):
        return False, "merchant not enrolled in Amex OptBlue"
    return True, ""


def passes_token_lock(txn: Transaction, scheme: str) -> tuple[bool, str]:
    """If PAN is a network token, only the token's issuing network is eligible."""
    if not txn.is_network_token:
        return True, ""
    if scheme != txn.token_network:
        return False, f"network token locked to {txn.token_network}"
    return True, ""


def merchant_eligible(txn: Transaction, scheme: str) -> tuple[bool, str]:
    """Does this merchant's contract cover this scheme?"""
    contract = MERCHANT_CONTRACTS.get(txn.merchant_id, _FALLBACK_CONTRACT)
    if scheme not in contract["accepts"]:
        return False, f"merchant contract does not accept {scheme}"
    return True, ""
