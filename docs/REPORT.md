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
1. Demonstrate the core patterns from the Agentic AI syllabus
   (tool-using agents, multi-agent coordination, planning, reflection,
   human-in-the-loop, guardrails, evaluation).
2. Produce a system that actually works — measured against
   deterministic baselines — not just one that runs.

---

## 2. System architecture

### 2.1 Pipeline overview

The pipeline is a 12-node LangGraph DAG. Every node is a deterministic
state transformation. LLM-driven nodes (the three ReAct agents and the
three LLM utility nodes) are isolated; the rest of the system is plain
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
   critique_auth_score        ◄── Reflexion self-correction loop
        │
        │ (revise once if critic flags)
        ▼
        └──────────────┼──────────────┘
                       ▼
            validate_agent_outputs  ── (hard block) ──→ END
                       ▼
                aggregate_scores
                       ▼
              reflect_on_ranking  (LLM self-critique)
                       ▼
                check_hitl_gate  (interrupt if amount ≥ $500)
                       ▼
               run_compliance  (deterministic Python gate)
                       ▼
              validate_decision  ── (hard block) ──→ END
                       ▼
              generate_explanation  (LLM merchant-facing text)
                       ▼
                      END
```

### 2.2 Agent catalogue

| Agent | Pattern | Model tier | Tools |
|---|---|---|---|
| Auth Score | ReAct | fast | `feature_store_lookup`, `issuer_health_check`, `scheme_decline_patterns`, `emit_auth_score` |
| Cost | ReAct | smart | `interchange_lookup`, `scheme_fee_lookup`, `emit_cost_score` |
| Fraud | ReAct + XGBoost | fast | `velocity_check`, `geo_anomaly_check`, `device_risk_score`, `scheme_fraud_defense`, `emit_fraud_score` |
| Auth Critic | rule-based + LLM revision | fast | (none — direct state inspection) |
| Reflection | LLM judge | fast | (none — produces a critique string) |
| Explanation | LLM generation | fast | (none — produces merchant-facing text) |

### 2.3 Multi-provider abstraction

All LLM calls go through `llm_clients.get_chat_model(tier)` where `tier`
is `fast` / `smart` / `judge`. The provider is selected by one
environment variable:

```bash
LLM_PROVIDER=anthropic | azure_openai | openai | google | groq | ollama
```

Default models per (provider, tier):

| Tier | Anthropic | Azure OpenAI * | OpenAI | Google |
|---|---|---|---|---|
| fast  | claude-haiku-4-5-20251001 | gpt-4o-mini | gpt-4o-mini | gemini-2.5-flash |
| smart | claude-sonnet-4-6         | gpt-4o      | gpt-4o      | gemini-2.5-flash |
| judge | claude-haiku-4-5-20251001 | gpt-4o-mini | gpt-4o-mini | gemini-2.5-flash |

\* Azure deployment names — override with `OVERRIDE_FAST_MODEL` /
`OVERRIDE_SMART_MODEL` to match names you created in the Azure portal.

Any tier can be overridden via `OVERRIDE_FAST_MODEL` / `OVERRIDE_SMART_MODEL` /
`OVERRIDE_JUDGE_MODEL`. The agent code never imports a specific
provider's SDK — it gets a `BaseChatModel` from LangChain's
`init_chat_model()`.

This abstraction matters for the rubric: the same code runs on
Anthropic Claude, Azure OpenAI, OpenAI, Gemini, or Llama by changing
one env var. For evaluation we deliberately use *different* models for
candidate vs judge to mitigate self-preference bias.

---

## 3. Self-correction loop

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

## 4. ML integration: XGBoost fraud model

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

## 5. Evaluation

Three-layer harness in `evaluation/`. Run with
`python -m evaluation.run_evals`.

### 5.1 Layer 1 — Per-agent quality

| Evaluator | Dataset | Metric | Result |
|---|---|---|---|
| Tool-use accuracy (live mode) | 12 hand-labelled rows | recall × forbidden-avoidance × emit-once | <TODO> |

### 5.2 Layer 2 — System quality

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

### 5.3 Layer 3 — Stress

| Case | Expected | Outcome | Pass |
|---|---|---|---|
| `amount_zero` | guardrail | <TODO> | <TODO> |
| `huge_amount` | guardrail | <TODO> | <TODO> |
| `merchant_id_injection` | guardrail | <TODO> | <TODO> |
| `amex_non_optblue` | no_decision | <TODO> | <TODO> |

Pass rate: <TODO>% (4/4 in the smoke run).

---

## 6. Production readiness — what's missing

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

## 7. Limitations and honest caveats

In rough order of importance:

1. The fraud model trains on PCA features unavailable at inference
   time. Held-out AUC is optimistic. Documented in §4.
2. The hand-labelled tool-use eval dataset is small (12 rows).
   Statistical power on individual accuracy numbers is limited.
3. Mock-mode tool-use accuracy is 100% by construction (the
   synthesiser produces exactly the labelled tool sequence).
   Live-mode is the meaningful number.
4. The deterministic compliance gate is the source of truth for
   routing decisions. It encodes the five compliance rules in plain
   Python and is fully auditable without an LLM.

---

## 8. References

- Yao et al. (2023), *ReAct: Synergizing Reasoning and Acting in Language
  Models*. https://arxiv.org/abs/2210.03629
- Shinn et al. (2023), *Reflexion: Language Agents with Verbal
  Reinforcement Learning*. https://arxiv.org/abs/2303.11366
- Chen et al. (2016), *XGBoost: A Scalable Tree Boosting System*.
  KDD 2016. https://arxiv.org/abs/1603.02754
- Chawla et al. (2002), *SMOTE: Synthetic Minority Over-sampling
  Technique*. JAIR 16:321-357.

---

## Appendix A — Reproducibility

```bash
# 1. Install (Python 3.10-3.12)
pip install -r requirements.txt

# 2. Configure (edit .env from .env.example)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# 3. Train the fraud model (optional; uses ~145MB Kaggle data via kagglehub)
python -m ml.training.train_fraud

# 4. Run the test suite
LLM_MODE=mock python tests/test_pipeline.py

# 5. Run the eval harness
python -m evaluation.run_evals       # produces evaluation/results/eval_*.json + .md

# 6. Launch the dashboard
streamlit run dashboard.py
```
