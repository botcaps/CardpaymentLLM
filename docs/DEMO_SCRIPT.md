# Demo Video Script — Card Scheme Orchestrator

**Total length: ~6 minutes.** Aim for under 7. Graders skim past the
8-minute mark.

**Recording setup:**
- Screen recorder (OBS or QuickTime) at 1080p
- Microphone, not laptop built-in. A cheap $30 USB mic is fine.
- Have these tabs/windows ready BEFORE you hit record:
  - VS Code with the project open (or any editor)
  - Browser with `localhost:8501` and Streamlit already loaded
  - Terminal with the project root as cwd
  - LangSmith UI logged in (if you have an account) — optional
  - This script open on a second monitor or printed

**Before recording:**
- Run `python -m rag.ingest` once so the corpus is built
- Run `python -m ml.training.train_fraud --rows 50000` once so the model exists
- Run `python -m evaluation.run_evals --quick` once so `latest.json` exists
- Refresh the dashboard so the Evaluations tab has data

---

## Script

### [0:00–0:30] Opening — what is this and why does it matter

> "Hi, I'm <YOUR NAME>. This is my Agentic AI and RAG capstone, the
> Card Scheme Orchestrator. The problem: every card payment routes
> through one of several networks — Visa, Mastercard, Amex, Discover —
> and the choice affects approval rate, processing cost, fraud risk,
> and regulatory compliance. CSO is a multi-agent system that picks
> the optimal scheme per transaction by balancing all four."

**What's on screen:** `docs/architecture.png` (the Mermaid render)
filling the screen. Cursor doesn't move; just narrate over the diagram.

### [0:30–1:30] The pipeline — point at the diagram

> "Pipeline is built on LangGraph. Transactions enter on the left.
> Three agents run in parallel after a planning step — auth, cost, and
> fraud — each is a ReAct loop with its own tools.
>
> The auth agent has a self-correction loop after it — that's the
> Reflexion pattern. A critic node checks the emitted score against the
> baseline; if it drifts too much, it loops back for one revision.
>
> Then aggregation, reflection, an HITL interrupt for high-value
> transactions, and the compliance step. The compliance step itself is
> parallel: a deterministic Python gate is the source of truth for
> routing, and a Compliance RAG agent runs in parallel to produce a
> citation-backed explanation. The RAG agent uses MCP — Model Context
> Protocol — to access a regulation retriever. I'll show that next."

**What's on screen:** zoom into specific parts of the architecture
diagram as you describe each. Don't switch screens yet.

### [1:30–2:30] The codebase tour

> "Quick look at the layout."

Open the file tree in VS Code. Walk through:

> "`orchestrator/graph.py` — the LangGraph DAG, fourteen nodes.
> `agents/` — four agents, each with its own tools. `compliance_rag`
> is the new one I added.
> `rag/` — corpus list, ingestion pipeline, hybrid retriever,
> MCP server.
> `ml/` — XGBoost training plus serving wrapper.
> `evaluation/` — three-layer harness with custom evaluators.
> `llm_clients.py` — multi-provider abstraction. The whole system
> runs on OpenAI, Anthropic, Gemini, Llama via Groq, or local Ollama
> by changing one env var."

Open `llm_clients.py`, scroll to `TIER_DEFAULTS`. Stay 5 seconds.

> "Tier system: agents ask for `fast`, `smart`, or `judge`. Defaults
> per provider are in this table. Code never imports a vendor SDK
> directly."

### [2:30–3:30] RAG layer — the headline feature

Switch to terminal:

```bash
python -m rag.retriever "EU debit interchange cap"
```

Wait for the output. Read aloud from the top result:

> "I'm querying the regulation corpus. This is hybrid retrieval —
> BM25 plus dense embeddings plus a cross-encoder reranker. The top
> result is from the EU Interchange Fee Regulation, Article 3, which
> is exactly the right rule. The retriever returns five chunks ranked
> by relevance, each with source metadata."

Then:

```bash
python -m rag.mcp_server --http &
```

(or just describe — actually launching it as a subprocess works but
is fragile in a recording)

> "The same retriever is exposed as an MCP server. The Compliance RAG
> agent connects to it as a subprocess, gets the tools, and uses them
> in a ReAct loop. I went with MCP rather than direct LangChain tools
> because it's the standard tool protocol now — the same server could
> serve a Claude Desktop client or a CrewAI agent without changes."

### [3:30–4:30] Live demo — run a transaction end-to-end

Switch to the dashboard, already loaded at `localhost:8501`.

Click **🔍 Deep Dive** tab. Pick `txn_0001` (vanilla EU dual-brand).

> "Here's a vanilla EU dual-brand transaction. The pipeline ranked
> Mastercard above Visa because the issuer's auth rate is 1.3 points
> higher on Mastercard. Both passed the deterministic compliance gate."

Click **📜 Compliance RAG** tab. The same transaction.

> "Same transaction in the Compliance RAG view. The deterministic gate
> picked Mastercard. The RAG agent's verdicts for both schemes are
> 'compliant' — green agreement badge — and you can see the cited
> regulation, the verbatim passage from the EU IFR, and the source
> URL. Clicking the URL takes you to EUR-Lex."

Now switch to **txn_0005** (Amex at non-OptBlue merchant).

> "Now an Amex card at a merchant that isn't OptBlue-enrolled. The
> deterministic gate rejected it — no eligible scheme. The RAG agent
> agrees: blocked, with the specific contract clause. Both verifiers
> aligned, no human review needed."

### [4:30–5:30] Evaluation tab — proof it actually works

Click **🧪 Evaluations** tab.

> "The evaluation harness has three layers. Layer one is per-agent
> quality — tool-use accuracy, retrieval precision at k, RAG
> faithfulness with an LLM judge. Layer two is system quality — CSO
> versus two baselines on fifty synthetic transactions. Layer three
> is stress tests — schema violations, prompt injection, compliance
> impossibilities."

Point at each panel as you describe:

> "Tool-use accuracy on twelve hand-labeled cases is <YOUR NUMBER>.
> Retrieval P@1 with the cross-encoder reranker is <YOUR NUMBER>;
> without reranker it's <YOUR NUMBER>, so the reranker is worth its
> compute. Faithfulness scores from the live LLM judge average
> <YOUR NUMBER> out of five.
>
> Layer two — CSO beats the cheapest-first baseline on expected
> revenue per hundred transactions, and crucially has zero compliance
> violations where cheapest-first has <YOUR NUMBER>%.
>
> Layer three — four out of four stress cases caught."

### [5:30–6:00] Closing — what's intentionally not done + ask

> "Things this project deliberately doesn't do, all documented in the
> report's limitations section: no closed-loop feedback yet, no
> multi-tenancy, in-memory state only. The roadmap in the README
> covers each of those.
>
> Code is at <YOUR REPO LINK>. Technical report is in `docs/REPORT.md`,
> four pages with citations. Thanks for watching."

End recording.

---

## Production tips

- **Take two passes per segment.** First pass to find the words; second
  pass to deliver them cleanly. Cut the first pass.
- **Don't read the script aloud verbatim.** Use it to remember the
  *shape* of what you want to say. Improvising sounds more natural.
- **Speak slightly slower than feels natural.** Recordings always feel
  fast in playback. Aim for 140 words/minute.
- **Highlight your cursor** in the screen recorder settings if it's an
  option. Graders watching at 2x speed lose track of where you're
  pointing otherwise.
- **No music.** Music dates demo videos and distracts from explanation.
- **Render at 1080p, export as MP4 H.264.** Universal playback.
- **Final length target: 5:45–6:30.** Anything over 7 minutes loses
  attention.

## Common viva questions to be ready for after the demo

In rough order of likelihood:

1. *"Walk me through the self-correction loop. What happens when the
   critic flags?"* — Show `orchestrator/graph.py:critique_auth_score`,
   trace the conditional edge.
2. *"Why is the RAG agent in parallel with the deterministic gate
   rather than after it?"* — See REPORT.md §2.1 — independence of
   verifiers, deterministic is source of truth.
3. *"How do you measure RAG faithfulness, and what are its biases?"* —
   See REPORT.md §7.4 and the docstring of `rag_faithfulness.py`.
   You should be able to name the three biases and the three mitigations.
4. *"Why did you pick BM25 + dense weights of 0.3/0.7?"* — Empirical
   tuning. Dense wins on average for legal language; BM25 wins on
   exact citations. We tested 50/50 and it was worse.
5. *"Why MCP over direct LangChain tools?"* — Decoupling. The MCP
   server can serve any framework; tools become reusable across
   projects.
6. *"What's the failure mode of your fraud model in production?"* —
   PCA features unavailable at inference time. Documented limitation.
   Real fix is feedback-driven retraining.
7. *"Show me a transaction where the deterministic gate and the RAG
   agent disagree."* — Have one ready in the dashboard. If you don't
   see one in your test runs, that itself is interesting and worth
   discussing.
