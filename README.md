# Card Scheme Orchestrator (CSO) — Capstone

LLM-native multi-agent system for card payment routing. Picks the
optimal payment scheme per transaction by balancing four competing
objectives: approval probability, processing cost, fraud risk, and
regulatory compliance.

---

## What this is for

This is my Agentic AI + RAG capstone project. The grading rubric
covers two competencies, and the project is structured so that each
rubric concept is concretely demonstrated in the code:

| Concept | Where it lives |
|---|---|
| Tool-using agent (ReAct) | `agents/auth_score/`, `agents/cost/`, `agents/fraud/`, `agents/compliance_rag/` |
| Multi-agent coordination | `orchestrator/graph.py` — LangGraph fan-out/fan-in |
| LLM planner | `orchestrator/graph.py:plan_candidates` |
| Reflection / self-critique | `orchestrator/graph.py:reflect_on_ranking` |
| **Self-correction loop (Reflexion)** | `orchestrator/graph.py:critique_auth_score` + conditional edge |
| Human-in-the-loop | `orchestrator/graph.py:check_hitl_gate` (LangGraph `interrupt()`) |
| Guardrails | `Guardrail/agentguardrail.py` — 3-stage (input / agent / decision) |
| **RAG (chunking + hybrid retrieval + reranker)** | `rag/ingest.py`, `rag/retriever.py` |
| **MCP tool serving** | `rag/mcp_server.py` (FastMCP) |
| ML model integration | `ml/fraud_model.py` (XGBoost), `ml/training/train_fraud.py` |
| Multi-provider LLM abstraction | `llm_clients.py` |
| Memory / state persistence | LangGraph `MemorySaver` checkpointing |
| **Evaluation rigor** | `evaluation/` — 3-layer harness with custom evaluators |

Bold = meaningful additions over the original prototype.

---

## Stack

LangChain + LangGraph + LangSmith + MCP + Streamlit. Pluggable across
five LLM providers — paste any key, the system uses it:

```bash
LLM_PROVIDER=openai     OPENAI_API_KEY=sk-...
LLM_PROVIDER=anthropic  ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=google     GEMINI_API_KEY=AIza...
LLM_PROVIDER=groq       GROQ_API_KEY=gsk_...     # Llama 3.3 70B via Groq
LLM_PROVIDER=ollama                              # local Llama, no key
LLM_MODE=mock                                    # deterministic, no API
```

---

## Quick start

```bash
# 1. Install (Python 3.10–3.12)
pip install -r requirements.txt

# 2. Configure environment (copy and fill in)
cp .env.example .env
# ...edit .env with your provider + key

# 3. Build the RAG corpus (one-time, ~30-60s)
python -m rag.ingest

# 4. Train the fraud model (optional; ~2 minutes, fetches ~145MB via kagglehub)
python -m ml.training.train_fraud

# 5. Run the test suite
LLM_MODE=mock python tests/test_pipeline.py
# Expected: 16 passed, 0 failed

# 6. Run the evaluation harness
python -m evaluation.run_evals
# Outputs go to evaluation/results/eval_<timestamp>.json + .md

# 7. Launch the dashboard
streamlit run dashboard.py
```

---

## Architecture

The pipeline is a 14-node LangGraph DAG. Every LLM-driven node is
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
              ├──→ run_compliance              (deterministic gate)
              │
              └──→ run_compliance_rag_agent    (LLM + MCP-served retriever)
                            │
              ┌─────────────┘
              ▼
     validate_decision ──── (hard block) ──→ END
              │
              ▼
     generate_explanation
              │
              ▼
            Decision out
```

The two parallel paths after `check_hitl_gate` are deliberate:

- **`run_compliance`** is the source of truth for routing. Hardcoded
  Python rules. Audit-grade. Sub-millisecond.
- **`run_compliance_rag_agent`** runs in parallel — it produces a
  citation-backed *explanation* using retrieved regulation passages.
  If it disagrees with the deterministic gate, the disagreement is
  logged for review.

This is the production pattern when one verifier is fast/audit-grade
and the other is slow/explainable: run both, defer routing to the
deterministic one, surface disagreements for human review.

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
GPT-4o, Claude Sonnet, Gemini Pro, or Llama 3.3.

### RAG layer (`rag/`)

- **`rag/corpus.py`** — 5 government-hosted regulation source URLs
  (EU IFR, US Reg II, PSD2 RTS, PSD2 Directive, Reg II FAQ). No PDFs
  in the repo.
- **`rag/ingest.py`** — `WebBaseLoader` → `RecursiveCharacterTextSplitter`
  (size 1000, overlap 150) → `get_embedding_model()` → Chroma.
  CLI: `python -m rag.ingest [--rebuild] [--sources ifr_eu ...]`.
- **`rag/retriever.py`** — `EnsembleRetriever([BM25, Chroma], weights=[0.3, 0.7])`
  → `ContextualCompressionRetriever` wrapping `CrossEncoderReranker`
  (BAAI/bge-reranker-base). CLI: `python -m rag.retriever "EU debit cap"`.

### MCP server (`rag/mcp_server.py`)

FastMCP server exposing 3 tools to MCP clients:
- `retrieve_regulation(query, k, jurisdiction)`
- `get_regulation_metadata(source_id)`
- `list_available_regulations()`

The Compliance RAG agent connects via `MultiServerMCPClient` from
`langchain-mcp-adapters` (subprocess via stdio). Same MCP server
could serve other agents in other frameworks — that's the point.

### Self-correction loop (Day 7)

`critique_auth_score` checks each emitted `p_auth` against the
feature-store baseline; if it drifts more than 5pp, falls outside
[0.50, 0.99], or skips referencing issuer health when the issuer is
unhealthy, the conditional edge routes back to `run_auth_agent` for
one revision. Reflexion pattern. Capped at one retry.

### Evaluation (`evaluation/`)

Three layers:

1. **Per-agent quality** — tool-use accuracy, retrieval P@k,
   RAG faithfulness (LLM-as-judge with mitigated bias).
2. **System quality** — CSO vs `cheapest_first` vs `highest_auth`
   on 50 synthetic transactions; latency p50/p95/p99.
3. **Stress** — schema violations + prompt injection + compliance
   impossibilities.

`python -m evaluation.run_evals` produces `eval_<timestamp>.json` +
`eval_<timestamp>.md` in `evaluation/results/`. The dashboard's
**🧪 Evaluations** tab loads `latest.json` automatically.

---

## Dashboard

`streamlit run dashboard.py` opens the interactive interface with
**9 tabs**:

| Tab | Contents |
|---|---|
| 📊 Overview | KPI metrics, scheme distribution, p_auth vs fee scatter |
| 🔍 Deep Dive | Per-transaction selector with full reasoning trace |
| 🔐 Auth Scores | Issuer/scheme heatmap, drift, decline patterns |
| 💰 Cost Analysis | Interchange heatmap, fee breakdowns |
| 🛡️ Compliance | Rule pass/fail matrix |
| 🗄️ Data Explorer | Raw feature store + JSON traces |
| 🤖 Agentic | Per-step LangGraph trace visualization |
| **📜 Compliance RAG** | Citation viewer — per-scheme verdicts with regulation quotes |
| **🧪 Evaluations** | Eval-harness results tables + plots |

---

## Project layout

```
cso/
├── orchestrator/        LangGraph pipeline + state
├── agents/
│   ├── auth_score/
│   ├── cost/
│   ├── fraud/
│   └── compliance_rag/  Day 5 — uses MCP-served retriever
├── compliance/          deterministic regulatory rules
├── Guardrail/           3-stage input/agent/decision validation
├── contracts/           dataclasses (Transaction, AuthScore, ComplianceVerdict, ...)
├── data/                synthetic feature store + Kaggle loader
├── rag/                 corpus, ingest, retriever, MCP server
├── ml/                  XGBoost training + serving
├── evaluation/          3-layer harness + datasets + runner
├── observability/       Trace dataclass for LangSmith
├── llm_clients.py       multi-provider chat & embedding factory
├── dashboard.py         9-tab Streamlit UI
└── docs/REPORT.md       4-page technical report
```

---

## Models

Pick one provider via `LLM_PROVIDER`. Default models per tier:

| Tier | OpenAI | Anthropic | Google | Groq (Llama) |
|---|---|---|---|---|
| fast | gpt-4o-mini | claude-haiku-4-5 | gemini-2.5-flash | llama-3.3-70b-versatile |
| smart | gpt-4o | claude-sonnet-4-5 | gemini-2.5-pro | llama-3.3-70b-versatile |
| judge | gpt-4o-mini | claude-haiku-4-5 | gemini-2.5-flash | llama-3.1-8b-instant |

Any tier can be overridden with `OVERRIDE_FAST_MODEL` /
`OVERRIDE_SMART_MODEL` / `OVERRIDE_JUDGE_MODEL`.

---

## Limitations

Honest list, expanded in `docs/REPORT.md` §9:

1. The fraud model trains on Kaggle's PCA features which aren't
   available at inference time; runtime predictions are less accurate
   than the held-out test AUC suggests.
2. Eval datasets are small (12 + 15 rows); per-row P@k variance is high.
3. LLM-as-judge faithfulness scores aren't human-validated; standard
   mitigations are documented in `evaluation/evaluators/rag_faithfulness.py`.
4. RAG corpus only covers EU + US (government sources); Visa/Mastercard
   public rules excluded due to layout instability.
5. State persistence is in-memory only (`MemorySaver`); no HITL across
   process restarts. README's Phase 1 spec covers the fix.
