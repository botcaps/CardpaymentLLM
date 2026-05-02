# Card Scheme Orchestrator (CSO) — Capstone

LLM-native multi-agent system for card payment routing. Picks the
optimal payment scheme per transaction by balancing four competing
objectives: approval probability, processing cost, fraud risk, and
regulatory compliance.

---

## What this is for

This is my Agentic AI capstone project. The grading rubric covers key
competencies, and the project is structured so that each rubric concept
is concretely demonstrated in the code:

| Concept | Where it lives |
|---|---|
| Tool-using agent (ReAct) | `agents/auth_score/`, `agents/cost/`, `agents/fraud/` |
| Multi-agent coordination | `orchestrator/graph.py` — LangGraph fan-out/fan-in |
| LLM planner | `orchestrator/graph.py:plan_candidates` |
| Reflection / self-critique | `orchestrator/graph.py:reflect_on_ranking` |
| **Self-correction loop (Reflexion)** | `orchestrator/graph.py:critique_auth_score` + conditional edge |
| Human-in-the-loop | `orchestrator/graph.py:check_hitl_gate` (LangGraph `interrupt()`) |
| Guardrails | `Guardrail/agentguardrail.py` — 3-stage (input / agent / decision) |
| ML model integration | `ml/fraud_model.py` (XGBoost), `ml/training/train_fraud.py` |
| Multi-provider LLM abstraction | `llm_clients.py` |
| Memory / state persistence | LangGraph `MemorySaver` checkpointing |
| **Evaluation rigor** | `evaluation/` — 3-layer harness with custom evaluators |

Bold = meaningful additions over the original prototype.

---

## Stack

LangChain + LangGraph + LangSmith + Streamlit. Pluggable across
six LLM providers — paste your key(s), set `LLM_PROVIDER`, done:

```bash
LLM_PROVIDER=anthropic     ANTHROPIC_API_KEY=sk-ant-...          # Claude
LLM_PROVIDER=azure_openai  AZURE_OPENAI_API_KEY=...              # Azure OpenAI
                           AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
LLM_PROVIDER=openai        OPENAI_API_KEY=sk-...                 # OpenAI direct
LLM_PROVIDER=google        GEMINI_API_KEY=AIza...                # Gemini
LLM_PROVIDER=groq          GROQ_API_KEY=gsk_...                  # Llama via Groq
LLM_PROVIDER=ollama                                              # local Llama
LLM_MODE=mock                                                    # no API at all
```

---

## Quick start

```bash
# 1. Install (Python 3.10–3.12)
pip install -r requirements.txt

# 2. Configure environment (copy and fill in)
cp .env.example .env
# ...edit .env with your provider + key

# 3. Train the fraud model (optional; ~2 minutes, fetches ~145MB via kagglehub)
python -m ml.training.train_fraud

# 4. Run the test suite
LLM_MODE=mock python tests/test_pipeline.py

# 5. Run the evaluation harness
python -m evaluation.run_evals
# Outputs go to evaluation/results/eval_<timestamp>.json + .md

# 6. Launch the dashboard
streamlit run dashboard.py
```

---

## Architecture

The pipeline is a 12-node LangGraph DAG. Every LLM-driven node is
isolated; everything else is plain Python so failures are auditable.

```
Transaction in
     │
     ▼
validate_input ───── (hard block) ──→ END
     │
     ▼
plan_candidates  (LLM picks 2-4 schemes worth evaluating)
     │
     ├──→ run_auth_agent  (ReAct, fast tier)
     │         │
     │         ▼
     │    critique_auth_score ──── (loop back, max 1 retry)
     │         │
     ├──→ run_cost_agent  (ReAct, smart tier)
     │
     └──→ run_fraud_agent  (ReAct + XGBoost model)
              │
              ▼
     validate_agent_outputs ──── (hard block) ──→ END
              │
              ▼
     aggregate_scores  (W1·p_auth − W2·norm_fee − W3·p_fraud)
              │
              ▼
     reflect_on_ranking  (LLM anomaly detection)
              │
              ▼
     check_hitl_gate  (interrupt for amounts ≥ $500)
              │
              ▼
     run_compliance  (deterministic gate)
              │
              ▼
     validate_decision ──── (hard block) ──→ END
              │
              ▼
     generate_explanation
              │
              ▼
            Decision out
```

The compliance gate runs deterministic Python rules (IFR, Durbin,
OptBlue, token_lock, merchant contract). Audit-grade and sub-millisecond.

---

## Components in depth

### Multi-provider LLM (`llm_clients.py`)

`get_chat_model("fast" | "smart" | "judge")` returns a LangChain
`BaseChatModel`. The provider is picked from `LLM_PROVIDER` env var.
Defaults per (provider, tier) are in `TIER_DEFAULTS`. Any default can
be overridden with `OVERRIDE_FAST_MODEL` / `OVERRIDE_SMART_MODEL` /
`OVERRIDE_JUDGE_MODEL`.

The agent code never imports a vendor SDK — it gets a generic
`BaseChatModel` from LangChain's `init_chat_model()`. Same code on
GPT-4o, Claude Sonnet, Gemini Flash, or Llama 3.3.

### Self-correction loop

`critique_auth_score` checks each emitted `p_auth` against the
feature-store baseline. If it drifts more than 5pp, falls outside
[0.50, 0.99], or skips referencing issuer health when the issuer is
unhealthy, the conditional edge routes back to `run_auth_agent` for
one revision. Reflexion pattern. Capped at one retry to bound latency.

### Compliance gate (`compliance/`)

Five deterministic rules encoded in plain Python:
- **merchant** — merchant contract must include the scheme
- **token_lock** — network tokens are locked to the issuing network
- **optblue** — Amex requires explicit OptBlue merchant enrollment
- **ifr** — EU IFR interchange cap (0.20% debit / 0.30% credit)
- **durbin** — US regulated debit ceiling (≤ 95 bps)

The gate is the routing source of truth. All five rules are
unit-testable in isolation — no LLM involved.

### Evaluation (`evaluation/`)

Three layers:

1. **Per-agent quality** — tool-use accuracy (12 hand-labelled rows)
2. **System quality** — CSO vs `cheapest_first` vs `highest_auth`
   on 50 synthetic transactions; latency p50/p95/p99.
3. **Stress** — schema violations + prompt injection + compliance
   impossibilities.

`python -m evaluation.run_evals` produces `eval_<timestamp>.json` +
`eval_<timestamp>.md` in `evaluation/results/`.

---

## Dashboard

`streamlit run dashboard.py` opens the interactive interface:

| Tab | Contents |
|---|---|
| 📊 Overview | KPI metrics, scheme distribution, p_auth vs fee scatter |
| 🔍 Deep Dive | Per-transaction selector with full reasoning trace and interactive HITL |
| 🔐 Auth & Risk | Issuer/scheme heatmap, drift, decline patterns, fraud risk |
| 💰 Cost Analysis | Interchange heatmap, fee breakdowns |
| 🛡️ Compliance | Rule pass/fail matrix, rejection reasons, eligibility funnel |
| 🤖 Agents | Planner decisions, reflection outputs, explanation agent, guardrail warnings |

---

## Project layout

```
cso/
├── orchestrator/        LangGraph pipeline + state
├── agents/
│   ├── auth_score/      ReAct + feature store tools
│   ├── cost/            ReAct + interchange tools
│   └── fraud/           ReAct + XGBoost serving
├── compliance/          deterministic regulatory rules
├── Guardrail/           3-stage input/agent/decision validation
├── contracts/           dataclasses (Transaction, AuthScore, Decision, ...)
├── data/                synthetic feature store + Kaggle loader
├── ml/                  XGBoost training + serving
├── evaluation/          3-layer harness + datasets + runner
├── observability/       Trace dataclass for LangSmith
├── llm_clients.py       multi-provider chat factory
├── dashboard.py         Streamlit UI
└── docs/REPORT.md       technical report
```

---

## Models

Pick one provider via `LLM_PROVIDER`. Default models per tier:

| Tier | Anthropic (Claude) | Azure OpenAI * | OpenAI | Google | Groq (Llama) |
|---|---|---|---|---|---|
| fast  | claude-haiku-4-5-20251001 | gpt-4o-mini | gpt-4o-mini | gemini-2.5-flash | llama-3.3-70b-versatile |
| smart | claude-sonnet-4-6         | gpt-4o      | gpt-4o      | gemini-2.5-flash | llama-3.3-70b-versatile |
| judge | claude-haiku-4-5-20251001 | gpt-4o-mini | gpt-4o-mini | gemini-2.5-flash | llama-3.1-8b-instant |

\* Azure OpenAI values are **deployment names** (the name you gave the
deployment in the Azure portal, not the underlying OpenAI model name).
Override them to match your actual deployments:

```bash
OVERRIDE_FAST_MODEL=my-gpt4o-mini-deployment
OVERRIDE_SMART_MODEL=my-gpt4o-deployment
```

Any tier can be overridden with `OVERRIDE_FAST_MODEL` /
`OVERRIDE_SMART_MODEL` / `OVERRIDE_JUDGE_MODEL`.

---

## Limitations

Honest list, expanded in `docs/REPORT.md` §7:

1. The fraud model trains on Kaggle's PCA features which aren't
   available at inference time; runtime predictions are less accurate
   than the held-out test AUC suggests.
2. Eval datasets are small (12 hand-labelled rows for tool-use accuracy);
   per-row variance is high.
3. State persistence is in-memory only (`MemorySaver`); HITL state
   does not survive process restarts.
