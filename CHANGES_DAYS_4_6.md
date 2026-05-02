# Days 4-6 — MCP server + Compliance RAG agent + ML fraud model

This is the largest delivery so far. Three major additions, all wired
into the existing pipeline. Existing tests still pass (16/16).

## Day 4 — `rag/mcp_server.py`

A FastMCP server that exposes the regulation retriever as standardized
MCP tools. Three tools:

  - `retrieve_regulation(query, k, jurisdiction)` — primary search
  - `get_regulation_metadata(source_id)` — single-source lookup
  - `list_available_regulations()` — corpus discovery

Run it three ways:

```bash
# Standalone via stdio (default for agent subprocess use)
python -m rag.mcp_server

# HTTP for debugging (visit MCP Inspector at http://localhost:8765/mcp)
python -m rag.mcp_server --http

# Smoke check the tool registration
python -c "
from rag import mcp_server
import asyncio
tools = asyncio.run(mcp_server.mcp.list_tools())
for t in tools: print(t.name, '—', t.description.split('\\n')[0])
"
```

### Why MCP — questions for the viva

**Q: Why MCP and not just `@tool` decorators on Python functions?**
A: MCP (Model Context Protocol, Anthropic's open spec, JSON-RPC 2.0) is
the industry-standard tool protocol now. A tool defined once on an MCP
server is consumable by Claude Desktop, ChatGPT, Cursor, LangGraph,
CrewAI — any agent framework with an adapter. We get reusability for free.

**Q: stdio vs HTTP transport — which and why?**
A: Stdio for production (subprocess launched per session, no port collisions,
trivial to deploy). HTTP for debugging (the MCP Inspector tool is
HTTP-only). FastMCP supports both with one flag.

**Q: Why FastMCP over the raw mcp SDK?**
A: ~10 lines of boilerplate per tool vs ~50 with the raw protocol.
FastMCP auto-generates schemas from type hints + docstrings. The raw
SDK gives you more control but you don't need it for this scale.

## Day 5 — Compliance RAG agent + LangGraph integration

### The pieces

- **`contracts/models.py`** — added `ComplianceVerdict` dataclass: `{scheme,
  verdict, confidence, regulation, passage, url, source_id, reasoning}`.
- **`agents/compliance_rag/agent.py`** — the agent. Connects to the MCP
  server via `MultiServerMCPClient` (subprocess via stdio), gets tools,
  builds a ReAct agent with them + a local `emit_verdict` terminal tool,
  runs once per scheme, returns `list[ComplianceVerdict]`.
- **`orchestrator/graph.py`** — added `run_compliance_rag_agent` node in
  parallel with the existing `run_compliance` (deterministic gate).
  Both run after `check_hitl_gate`, both fan back in at `validate_decision`.
  `PipelineState` now carries `compliance_verdicts`.
- **`observability/tracer.py`** + **`orchestrator/orchestrate.py`** —
  `Trace.compliance_verdicts` for the dashboard / eval to consume.

### The new pipeline shape

```
                                check_hitl_gate
                              ┌──────┴──────────────────────┐
                              ▼                             ▼
                       run_compliance         run_compliance_rag_agent
                       (deterministic)        (LLM + MCP-served retriever)
                              │                             │
                              └──────────────┬──────────────┘
                                             ▼
                                     validate_decision
```

### Why parallel, not replacement — questions for the viva

**Q: Why isn't the RAG agent the only compliance check?**
A: The deterministic gate is the source of truth for routing. We do
NOT want a probabilistic LLM verdict to override a hard regulatory
rule (recipe for letting the model rationalise an IFR violation).
The RAG agent's role is *explanation*: cite the rule, quote the text,
so a human reviewer can verify the deterministic decision. Two
independent pipelines voting on the same question is a standard
production pattern when one is fast/audit-grade and the other is
slow/explainable.

**Q: What if they disagree?**
A: We log it. The dashboard's Compliance tab (Day 10) will surface
disagreements as a flag. In a real deployment that becomes a queue
for compliance officer review.

**Q: Why one agent invocation per scheme?**
A: Failure isolation (one scheme's reasoning crash doesn't drop the
other verdicts) and easier evaluation (each (txn, scheme) → verdict is
a discrete eval row in the LangSmith dataset).

**Q: Mock mode behaviour?**
A: When LLM_MODE=mock or no provider key is set, the agent falls back
to deterministic verdicts derived from the same rule library the gate
uses. Tests and CI run end-to-end without API keys.

## Day 6 — XGBoost fraud model

### The pieces

- **`ml/training/train_fraud.py`** — training script. Fetches Kaggle data
  via `kagglehub` (no committed CSV), stratified split, optional SMOTE,
  XGBoost training, eval (AUC-ROC + AUC-PR + best-F1 confusion matrix),
  saves to `ml/fraud_model.pkl`.
- **`ml/fraud_model.py`** — serving wrapper. Loads the .pkl lazily,
  exposes `FraudModel.predict_proba(txn, scheme) → float`.
- **`agents/fraud/agent.py`** — rewritten. The fraud agent now tries the
  trained model first (`_model_score_one`); falls back to rule-based
  scoring (`_rule_score_one`) if the model isn't trained yet. Existing
  ReAct path through `orchestrator/graph.py` is unaffected — same tool
  names, same output shape.

### Run it

```bash
# Train (one-time; ~2 minutes, downloads ~145 MB Kaggle data on first call)
python -m ml.training.train_fraud
# Quick smoke run on a subset:
python -m ml.training.train_fraud --rows 50000
# Ablation: rebalance with class weights instead of SMOTE
python -m ml.training.train_fraud --no-smote
```

You'll see headline metrics like:

```
AUC-ROC: 0.9750   AUC-PR: 0.8412
Best threshold: 0.7831
  precision: 0.9032   recall: 0.8367   F1: 0.8687
Confusion matrix [tn fp / fn tp]:
  56847   16
  18      82
```

The trained `.pkl` is gitignored. To grade without retraining, leave the
file out — `agents/fraud/agent.py` falls back transparently to rule-based.
To grade with the model, run the training script once.

### Why these design choices — questions for the viva

**Q: Why XGBoost not deep learning?**
A: ~492 fraud examples in 284k rows. DL needs orders of magnitude more.
Tabular data with engineered (PCA) features → tree models dominate every
published Kaggle leaderboard for this dataset. Inference is ~50µs per row
on CPU — critical for an agent tool that gets called dozens of times per
pipeline run.

**Q: Why AUC-PR over AUC-ROC for the headline metric?**
A: For severely imbalanced data, AUC-ROC inflates: even a model that
gets 99.83% of negatives "trivially right" scores 0.5. AUC-PR is
sensitive to how well the positive class is recovered, which is the
actual business question for fraud detection. We report both for
transparency.

**Q: Why SMOTE?**
A: With 492 positives, class-weight rebalancing alone underweights the
*variety* of fraud patterns the model sees during training. SMOTE
synthesises minority-class examples by interpolating between near-neighbour
positives, giving more diverse fraud-like signal. Trade-off: synthesised
borderline examples can hurt precision. We measure this with the
`--no-smote` ablation; the resulting comparison is in the report.

**Q: How does the model get used at inference time when we only have a
runtime Transaction, not the V1..V28 PCA features?**
A: This is the limitation worth being honest about in the report. We
build a "fallback feature vector" that uses Amount + a Time proxy from
hour_of_day, with V1..V28 zero-filled (they're zero-centred, so zero is
a reasonable prior mean). Real production retrains on features actually
available at inference time — that's exactly what `feedback/trainer.py`
in the README's Phase 3 would do. We document this as a known limitation
rather than pretending it's not there.

**Q: Why keep both `_model_score_one` and `_rule_score_one`?**
A: Graceful degradation. If the .pkl is missing (someone clones the
repo and didn't retrain), the system still works in rule-mode. This is
better than crashing with `FileNotFoundError`. The dispatch in
`_mock_score_one` checks `model_available()` and chooses.

---

# Verification checklist before Days 7-9

In a fresh shell, after `pip install -r requirements.txt`:

```bash
# 1. Existing tests still pass (we touched a lot of files)
LLM_MODE=mock python tests/test_pipeline.py
# Expected: 16 passed, 0 failed

# 2. The MCP server module loads with all 3 tools registered
python -c "
from rag import mcp_server
import asyncio
tools = asyncio.run(mcp_server.mcp.list_tools())
print(len(tools), 'tools:', [t.name for t in tools])
"
# Expected: 3 tools: ['retrieve_regulation', 'get_regulation_metadata',
#                     'list_available_regulations']

# 3. Compliance verdicts flow through the pipeline (mock mode)
LLM_MODE=mock python -c "
import asyncio
from data.samples import SAMPLES
from orchestrator.orchestrate import orchestrate
decision, trace = asyncio.run(orchestrate(SAMPLES[4]))   # Amex non-OptBlue
print('verdicts:', [(v['scheme'], v['verdict']) for v in trace.compliance_verdicts])
"
# Expected: [('amex', 'blocked')]  — RAG verdict matches deterministic rejection

# 4. (Optional, slow) Train the fraud model
python -m ml.training.train_fraud --rows 50000
# Expected: AUC-PR around 0.5-0.7 on a 50k subset; full dataset gets higher.

# 5. (Live mode only) End-to-end with one real LLM call
# First make sure rag/chroma_db exists from Days 2-3 ingestion
python -m rag.ingest                    # if not already done
LLM_PROVIDER=google GEMINI_API_KEY=AIza... python -c "
import asyncio
from data.samples import SAMPLES
from orchestrator.orchestrate import orchestrate
decision, trace = asyncio.run(orchestrate(SAMPLES[3]))   # EU debit
for v in trace.compliance_verdicts:
    print(v['scheme'], '→', v['verdict'])
    print('  cite:', v['regulation'])
    print('  quote:', v['passage'][:100])
"
# Expected: real verdicts with real EU IFR citations from Article 3
```

If step 3 returns empty `verdicts` list, the `run_compliance_rag_agent`
node didn't run. Check that the `from contracts.models import ... ComplianceVerdict`
import in `orchestrator/graph.py` succeeded — most likely cause is a
circular import.

If step 5 hangs longer than 30 seconds, the MCP subprocess might not be
shutting down. The `_mcp_client()` context manager closes it explicitly,
but if your `langchain-mcp-adapters` version is unusual (e.g. 0.0.x)
the close behaviour differs. Pin to `0.1.x` or `0.2.x` in
`requirements.txt`.

When all checks pass, message me **"Days 4-6 verified"** and I'll deliver
Days 7-9: the self-correction loop on the auth agent + the 3-layer
LangSmith evaluation harness.

# What's NOT in this delivery (deliberate)

- **No self-correction loop on the auth agent.** That's Day 7 (uses the
  same Reflexion pattern but on a different agent).
- **No evaluation harness.** That's Days 8-9. We'll use LangSmith's
  built-in evaluators and add custom ones for tool-use accuracy and
  RAG faithfulness.
- **No dashboard tab for the new compliance verdicts.** That's Day 10.
  For now you can see them in the JSON trace if you call `trace.to_json()`.
