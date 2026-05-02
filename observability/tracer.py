"""
Per-transaction decision trace.

In production emit this as structured JSON to your logging pipeline
(e.g. stdout -> Vector -> Loki/ES). Here we accumulate in memory
and pretty-print at the end of each run.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from contracts.models import Transaction, AuthScore, CostScore, FraudScore, Decision
import json


@dataclass
class Trace:
    txn_id: str
    candidates: list[str]            = field(default_factory=list)
    planner_reasoning: str | None    = None
    auth_scores: list[dict]          = field(default_factory=list)
    cost_scores: list[dict]          = field(default_factory=list)
    fraud_scores: list[dict]         = field(default_factory=list)
    ranked: list[dict]               = field(default_factory=list)
    reflection: str | None           = None
    decision: dict | None            = None
    explanation: str | None          = None
    degraded: bool                   = False
    error: str | None                = None
    guardrail_warnings: list[dict]   = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)
