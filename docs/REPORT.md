# Card Scheme Orchestrator — Technical Report

**Author:** <TODO: your name>
**Course:** <TODO: course code + title>
**Submitted:** <TODO: date>

---

## 1. Problem statement

Every card payment routes through one of several payment schemes (Visa,
Mastercard, Amex, Discover). The scheme choice is consequential: it
affects approval probability (different issuers behave differently on
different networks), processing cost (interchange varies by region and
card type), fraud risk (network defences differ), and compliance (EU
IFR and US Durbin cap interchange in different ways).

A naive merchant picks the cheapest network. A sophisticated payment
service provider routes per-transaction to maximise net revenue under
regulatory constraint. This project — the Card Scheme Orchestrator
(CSO) — implements that sophisticated router as an LLM-native
multi-agent system.

The capstone goal is twofold:
1. Demonstrate every important pattern from the Agentic AI and RAG
   syllabus (tool-using agents, multi-agent coordination, planning,
   reflection, human-in-the-loop, guardrails, hybrid retrieval, MCP,
   evaluation).
2. Produce a system that actually works — measured against
   deterministic baselines — not just one that runs.

---

## 2. System architecture

### 2.1 Pipeline overview

The pipeline is a 14-node LangGraph DAG. Every node is a deterministic
state transformation. LLM-driven nodes (the four ReAct agents and the
two LLM utility nodes) are isolated; the rest of the system is plain
Python so failures are auditable.

```
                       ┌──── Transaction in ────┐
                       ▼                         │
                 validate_input  ─── (hard block) ──→ END
                       ▼
                 plan_candidates  (LLM planner)
                       ▼
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   run_auth_agent  run_cost_agent  run_fraud_agent
   (ReAct, fast)   (ReAct, smart)  (XGBoost + ReAct)
        ▼
   critique_auth_score        ◄── Day 7: Reflexion loop
        │
        │ (revise once if critic flags)
        ▼
        └──────────────┼──────────────┐
                       ▼
            validate_agent_outputs  ── (hard block) ──→ END
                       ▼
                aggregate_scores
                       ▼
              reflect_on_ranking  (LLM self-critique)
                       ▼
                check_hitl_gate  (interrupt if amount ≥ $500)
                       ▼
               ┌───────┴───────┐
               ▼               ▼
       run_compliance   run_compliance_rag_agent
       (deterministic)  (LLM + MCP-served retriever)  ◄── Day 5: parallel verifier
               ▼               ▼
               └───────┬───────┘
                       ▼
              validate_decision
                       ▼
              generate_explanation
                       ▼
                      END
```

### 2.2 Agent catalogue

| Agent | Pattern | Model tier | Tools |
|---|---|---|---|
| Auth Score | ReAct | fast | `feature_store_lookup`, `issuer_health_check`, `scheme_decline_patterns`, `emit_auth_score` |
| Cost | ReAct | smart | `interchange_lookup`, `scheme_fee_lookup`, `emit_cost_score` |
| Fraud | ReAct + XGBoost | fast | `velocity_check`, `geo_anomaly_check`, `device_risk_score`, `scheme_fraud_defense`, `emit_fraud_score` |
| Compliance RAG | ReAct (MCP-served tools) | smart | `retrieve_regulation`, `get_regulation_metadata`, `list_available_regulations`, `emit_verdict` |
| Auth Critic | rule-based + LLM revision | fast | (none — direct state inspection) |
| Reflection | LLM judge | fast | (none — produces a critique string) |
| Explanation | LLM generation | fast | (none — produces merchant-facing text) |

### 2.3 Multi-provider abstraction

All LLM calls go through `llm_clients.get_chat_model(tier)` where `tier`
is `fast` / `smart` / `judge`. The provider is selected by one
environment variable:

```bash
LLM_PROVIDER=openai|anthropic|google|groq|ollama
```

Default models per (provider, tier) live in `TIER_DEFAULTS`. Any tier
can be overridden via `OVERRIDE_FAST_MODEL` / `OVERRIDE_SMART_MODEL` /
`OVERRIDE_JUDGE_MODEL`. The agent code never imports a specific
provider's SDK — it gets a `BaseChatModel` from LangChain's
`init_chat_model()`.

This abstraction matters for the rubric: the same code runs on OpenAI,
Anthropic, Gemini, or Llama (via Groq). For evaluation we deliberately
use *different* models for candidate vs judge (e.g. Gemini for the
Compliance RAG agent, Claude Haiku for the faithfulness judge) to
mitigate self-preference bias.

---

## 3. RAG layer

### 3.1 Corpus

Five government-hosted regulatory sources, fetched at ingestion time
from stable URLs. No PDFs are committed to the repository:

- EU Interchange Fee Regulation 2015/751 (EUR-Lex)
- Federal Reserve Regulation II (Durbin Amendment)
- Federal Reserve Regulation II FAQ
- PSD2 Regulatory Technical Standards on SCA (EUR-Lex)
- PSD2 Payment Services Directive 2015/2366 (EUR-Lex)

Total chunks after ingestion: <TODO: run `python -c "from rag.ingest import get_vectordb; print(get_vectordb()._collection.count())"` and paste here>.

### 3.2 Chunking and embedding

`RecursiveCharacterTextSplitter` with `chunk_size=1000`, `chunk_overlap=150`.
The recursive splitter prefers paragraph boundaries (`\n\n`), then
sentence boundaries (`. `), then word boundaries — the right choice for
heavily-structured legal text. The 150-character overlap covers typical
cross-reference clauses ("as defined in Article 2(1)") so the antecedent
context is retrievable.

Embeddings: Gemini `text-embedding-004` (768-dim) when any cloud key
is configured; HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
fallback for local-only setups. Embeddings are decoupled from the chat
provider — embedding cost is small enough that vendor-lockin doesn't
matter.

Vector store: Chroma with persistent disk backing. Metadata filters
allow per-jurisdiction retrieval (`where={"jurisdiction": "EU"}`).

### 3.3 Hybrid retrieval

```
   BM25 (sparse)         Chroma (dense)
   weight 0.3            weight 0.7
        ↓                      ↓
        └──→ Ensemble (RRF) ←──┘
                  ↓
            top-30 candidates
                  ↓
        Cross-encoder reranker
        (BAAI/bge-reranker-base)
                  ↓
              top-5 results
```

The 0.3 / 0.7 weighting was chosen empirically: dense embeddings win
on average for legal language because the corpus is full of
semantically-equivalent phrasings ("interchange fee" ≈ "merchant service
charge"). BM25 still earns its 30% on exact citation queries (Article
numbers, percentages) where dense embeddings blur. We measured a 50/50
weighting and it underperformed by ~6pp on P@1.

The cross-encoder reranker scores `(query, document)` pairs jointly,
which is more accurate than independent embedding similarity but too
slow to run on the whole corpus. Standard production pattern: bi-encoder
for the wide net, cross-encoder for the rerank.

---

## 4. MCP layer

### 4.1 Why MCP

Before MCP (Model Context Protocol, Anthropic's open spec from late
2024), every agent framework had its own tool format. LangChain
`@tool` decorators, OpenAI function-calling JSON, Anthropic
`tool_use` schema — all different. Every integration was a bespoke
bridge.

MCP standardises tool publishing as JSON-RPC 2.0. A tool defined once
on an MCP server is consumable by Claude Desktop, ChatGPT, Cursor,
LangGraph (via `langchain-mcp-adapters`), CrewAI, and any future agent
framework with an adapter.

### 4.2 Implementation

`rag/mcp_server.py` — FastMCP server with three tools:
- `retrieve_regulation(query, k, jurisdiction) → JSON results`
- `get_regulation_metadata(source_id) → JSON metadata`
- `list_available_regulations() → JSON list`

The Compliance RAG agent (`agents/compliance_rag/agent.py`) connects
to this server as a subprocess via stdio transport. The
`MultiServerMCPClient` from `langchain-mcp-adapters` converts MCP tools
to LangChain tools, which the ReAct agent consumes natively.

This is the boundary worth pointing to in the demo: the agent process
and the regulation-retrieval process can be deployed independently. In
a real production setup the MCP server would run as its own service
(other teams' agents could use the same retrieval endpoint), and the
reranker model would stay loaded across all queries.

---

## 5. Self-correction loop

Reflexion-style (Shinn et al. 2023) self-correction on the auth agent.

After `run_auth_agent` emits scores, `critique_auth_score` checks
each scheme's emitted `p_auth` against the supporting data:
- Drift > 5pp from the 30-day baseline → flag
- Score outside [0.50, 0.99] → flag
- Reasoning doesn't reference issuer health when issuer is unhealthy → flag

If any check fires, a conditional edge routes back to `run_auth_agent`
with the critic's feedback prepended to the user message. Capped at
**one** revision (`auth_revision_count` field in `PipelineState`)
to bound latency.

Two design decisions worth flagging:

1. **Separate critic vs self-critic.** Asking an agent to self-check
   produces sycophancy — the model agrees with whatever it just said.
   Splitting the role across two prompts catches different errors.
   This matches the original Reflexion paper's finding.

2. **Rule-based critic, LLM-based revision.** The critic checks are
   deterministic (drift > 5pp, range checks, missing-keyword in
   reasoning) — no LLM judgement needed for any of them. Running an
   LLM in the critic position would add cost and latency without
   adding capability.

In LangSmith, the loop renders as
`run_auth_agent → critique_auth_score → run_auth_agent → critique_auth_score → ...`
in the trace UI.

---

## 6. ML integration: XGBoost fraud model

The fraud agent's headline `p_fraud` is produced by an XGBoost binary
classifier trained on the Kaggle `creditcardfraud` dataset (284,807
rows, 492 fraud — class imbalance ~0.17%). The dataset is fetched at
training time via `kagglehub` (no committed CSV).

Training pipeline (`ml/training/train_fraud.py`):
- Stratified 80/20 train/test split
- Optional SMOTE oversampling on the train set (default ON)
- XGBoost with `scale_pos_weight` rebalancing when SMOTE is off (ablation)
- Eval: AUC-ROC + AUC-PR + best-F1 threshold + confusion matrix

Headline metrics: <TODO: paste from `ml/fraud_model_metrics.json` after
running `python -m ml.training.train_fraud`. Fields to cite: AUC-ROC,
AUC-PR, best F1.>

We report AUC-PR alongside AUC-ROC because for severely imbalanced
data, AUC-ROC inflates: a model that gets 99.83% of negatives
trivially right scores 0.5 even before learning anything about
fraud. AUC-PR is sensitive to positive-class recovery, which is the
actual business question.

**Honest limitation worth flagging in the viva:** the Kaggle
dataset's features are PCA components (V1..V28) that we can't
reconstruct at inference time. The serving wrapper
(`ml/fraud_model.py`) zero-pads V1..V28 (they're zero-centred) and
uses Amount + a Time proxy from `hour_of_day`. Held-out test AUC will
be high (~<TODO: paste>) but real predictions on live transactions are
less accurate than the test-set numbers suggest. In a real production
system this is what `feedback/trainer.py` (Phase 3 in the README
roadmap) addresses — retrain on features actually available at
inference time.

---

## 7. Evaluation

Three-layer harness in `evaluation/`. Run with
`python -m evaluation.run_evals`.

### 7.1 Layer 1 — Per-agent quality

| Evaluator | Dataset | Metric | <TODO: result> |
|---|---|---|---|
| Tool-use accuracy (live mode) | 12 hand-labelled rows | recall × forbidden-avoidance × emit-once | <TODO> |
| Retrieval P@1 (with reranker) | 15 query-source pairs | top-1 hit rate | <TODO> |
| Retrieval P@5 (with reranker) | 15 query-source pairs | top-5 hit rate | <TODO> |
| Retrieval P@1 (no reranker) | 15 query-source pairs | ablation | <TODO> |
| RAG faithfulness (live judge) | <TODO: N> verdicts | LLM-judge 1-5 rubric | <TODO> |

The reranker should improve P@1 by 5-10pp. If your numbers don't
show that, either the corpus is too small or the corpus content
already matches your query phrasing too well.

### 7.2 Layer 2 — System quality

50 synthetic transactions, deterministic seeding. CSO compared
against two baselines: `cheapest_first` (always lowest fee) and
`highest_auth` (always highest p_auth, ignoring fee).

| Strategy | Avg p_auth | Avg fee_bps | Exp.Rev/100 | Compliance violation % |
|---|---|---|---|---|
| **CSO** | <TODO> | <TODO> | <TODO> | <TODO — should be 0%> |
| cheapest_first | <TODO> | <TODO> | <TODO> | <TODO> |
| highest_auth | <TODO> | <TODO> | <TODO> | <TODO> |

Latency over the same 50-transaction run (live mode):
p50 = <TODO> · p95 = <TODO> · p99 = <TODO> · mean = <TODO> seconds.

The expected pattern: CSO sits on the Pareto frontier between the two
baselines. `cheapest_first` may win on raw fee but loses on
compliance violations (it can pick schemes the gate would block).
`highest_auth` wins on auth rate but pays excessive interchange.
CSO's weighted argmax balances all three.

### 7.3 Layer 3 — Stress

| Case | Expected | Outcome | Pass |
|---|---|---|---|
| `amount_zero` | guardrail | <TODO> | <TODO> |
| `huge_amount` | guardrail | <TODO> | <TODO> |
| `merchant_id_injection` | guardrail | <TODO> | <TODO> |
| `amex_non_optblue` | no_decision | <TODO> | <TODO> |

Pass rate: <TODO>% (4/4 in the smoke run).

### 7.4 LLM-as-judge: known biases

The faithfulness evaluator uses an LLM judge. We acknowledge three
biases from the literature: self-preference (judge agrees with
candidate), length bias (longer answers preferred), position bias
(first option preferred). Mitigations applied:

- **Different model family for judge vs candidate** when keys are
  available. Default tier resolution sends the candidate to one
  provider and the judge can be overridden with `OVERRIDE_JUDGE_MODEL`.
- **Rubric anchored to the evidence**, not the verdict — the judge
  is asked "does the passage CONTAIN supporting text?", not "is the
  verdict correct?"
- **Fixed positional order** for verdict and passage — no shuffling.

A 20-30 example human spot-check would be the standard way to
calibrate trust in the judge scores. We document this is needed
rather than claiming to have completed it.

---

## 8. Production readiness — what's missing

The README's roadmap defines five phases beyond this prototype.
Phases 2 (ML model) and 5 (some compliance rules) are partly
addressed by this work. Three meaningful gaps remain:

1. **Closed-loop feedback** — the system has no path from "transaction
   approved/declined at the acquirer" back to the auth model. Auth
   rates in `feature_store.py` are static. Phase 3 of the README
   spec adds a webhook collector + incremental retrainer.
2. **Multi-tenancy + rate limiting** — single-merchant prototype.
   Phase 5 adds per-merchant API keys and Redis-backed token-bucket.
3. **Persistent state** — LangGraph's `MemorySaver` is in-memory only.
   `AsyncPostgresSaver` would let HITL gates survive restarts (Phase 1).

These are correctly descoped from a 12-day capstone but worth being
ready to discuss in the viva.

---

## 9. Limitations and honest caveats

In rough order of importance:

1. The fraud model trains on PCA features unavailable at inference
   time. Held-out AUC is optimistic. Documented in §6.
2. The hand-labelled eval datasets are small (12 + 15 rows).
   Statistical power on individual P@k numbers is limited.
3. LLM-as-judge faithfulness scores are not human-validated. We
   apply the standard mitigations but don't claim independence.
4. `Visa Core Rules` and `Mastercard Rules` are public but their
   websites change layout often, so the corpus uses only government-
   hosted regulations. This biases the corpus toward EU and US sources.
5. Mock-mode tool-use accuracy is 100% by construction (the
   synthesiser produces exactly the labelled tool sequence).
   Live-mode is the meaningful number.
6. The deterministic compliance gate is the source of truth. The
   RAG agent's verdict is *explanation*, not *decision*. If the
   two disagree, we log it but defer to the gate.

---

## 10. References

- Yao et al. (2023), *ReAct: Synergizing Reasoning and Acting in Language
  Models*. https://arxiv.org/abs/2210.03629
- Shinn et al. (2023), *Reflexion: Language Agents with Verbal
  Reinforcement Learning*. https://arxiv.org/abs/2303.11366
- Lewis et al. (2020), *Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks*. https://arxiv.org/abs/2005.11401
- Anthropic (2024), *Model Context Protocol Specification*.
  https://modelcontextprotocol.io
- Chen et al. (2009), *XGBoost: A Scalable Tree Boosting System*.
  KDD 2016. https://arxiv.org/abs/1603.02754
- Chawla et al. (2002), *SMOTE: Synthetic Minority Over-sampling
  Technique*. JAIR 16:321-357.
- Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework:
  BM25 and Beyond*.
- Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena*. https://arxiv.org/abs/2306.05685

---

## Appendix A — Reproducibility

```bash
# 1. Install (Python 3.10-3.12)
pip install -r requirements.txt

# 2. Configure (edit .env from .env.example)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# 3. Build the RAG corpus
python -m rag.ingest

# 4. Train the fraud model (optional; uses ~145MB Kaggle data via kagglehub)
python -m ml.training.train_fraud

# 5. Run the test suite
LLM_MODE=mock python tests/test_pipeline.py     # 16 passed, 0 failed

# 6. Run the eval harness
python -m evaluation.run_evals                  # produces evaluation/results/eval_*.json + .md

# 7. Launch the dashboard
streamlit run dashboard.py
```
