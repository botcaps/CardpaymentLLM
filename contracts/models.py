"""
Data contracts for the CSO pipeline.

Using plain dataclasses to avoid extra dependencies. In production
you'd probably use Pydantic for JSON schema validation at the edge.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class Transaction:
    txn_id: str
    bin: str
    card_type: str                   # "credit" | "debit"
    card_brand_capabilities: list[str]
    merchant_id: str
    mcc: str
    amount_minor: int                # cents
    currency: str
    channel: str                     # "ecommerce" | "pos"
    three_ds_status: str             # "authenticated_frictionless" | "challenged" | "none"
    region: str                      # "EU" | "US"
    issuer_country: str
    acquirer_country: str
    hour_of_day: int = 14            # local hour
    is_network_token: bool = False
    token_network: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class AuthScore:
    scheme: str
    p_auth: float
    confidence: float
    reasoning: str


@dataclass
class CostScore:
    scheme: str
    total_fee_bps: float
    breakdown: dict
    reasoning: str


@dataclass
class FraudScore:
    scheme: str
    p_fraud: float
    confidence: float
    reasoning: str


@dataclass
class Decision:
    scheme: str
    p_auth: float
    fee_bps: float
    compliance_passed: bool
    txn_id: str
    degraded: bool = False
    rejected_schemes: dict = field(default_factory=dict)  # scheme -> reason
    explanation: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
