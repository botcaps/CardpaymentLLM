"""
Compliance gate — the single regulated exit point of the pipeline.
"""
from __future__ import annotations
from contracts.models import Transaction, Decision
from compliance.rules.all_rules import (
    passes_ifr_with_breakdown, passes_durbin, passes_optblue,
    passes_token_lock, merchant_eligible,
)


class NoEligibleSchemeError(Exception):
    def __init__(self, txn_id: str, rejections: dict):
        super().__init__(f"no eligible scheme for {txn_id}: {rejections}")
        self.txn_id = txn_id
        self.rejections = rejections


def compliance_gate(txn: Transaction, ranked: list[dict]) -> Decision:
    """
    ranked: list of dicts in descending order of weighted score, each with:
        scheme, p_auth, total_fee_bps, interchange_bps

    Runs every rule against every scheme (so rejections are fully audited),
    but returns the highest-ranked scheme that passes all rules.
    """
    rejections: dict[str, str] = {}
    winner_entry: dict | None = None

    for entry in ranked:
        scheme = entry["scheme"]
        checks = [
            ("merchant",   merchant_eligible(txn, scheme)),
            ("token_lock", passes_token_lock(txn, scheme)),
            ("optblue",    passes_optblue(txn, scheme)),
            ("ifr",        passes_ifr_with_breakdown(txn, scheme, entry["interchange_bps"])),
            ("durbin",     passes_durbin(txn, scheme, entry["interchange_bps"])),
        ]
        failed = [(name, reason) for name, (ok, reason) in checks if not ok]
        if failed:
            rejections[scheme] = "; ".join(f"{n}: {r}" for n, r in failed)
            continue
        if winner_entry is None:
            winner_entry = entry  # first eligible in rank order wins

    if winner_entry is None:
        raise NoEligibleSchemeError(txn.txn_id, rejections)

    return Decision(
        scheme=winner_entry["scheme"],
        p_auth=winner_entry["p_auth"],
        fee_bps=winner_entry["total_fee_bps"],
        compliance_passed=True,
        txn_id=txn.txn_id,
        rejected_schemes=rejections,
    )
