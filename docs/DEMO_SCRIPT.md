# Demo Video Script — Card Scheme Orchestrator

**Total length: ~5 minutes.** Aim for under 6. Graders skim past the
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
- Run `python -m ml.training.train_fraud --rows 50000` once so the model exists
- Run `python -m evaluation.run_evals --quick` once so `latest.json` exists
- Refresh the dashboard so the Evaluations tab has data

---

## Script

### [0:00–0:30] Opening — what is this and why does it matter

> "Hi, I'm <YOUR NAME>. This is my Agentic AI capstone, the Card Scheme
> Orchestrator. The problem: every card payment routes through one of
> several networks — Visa, Mastercard, Amex, Discover — and the choice
> affects approval rate, processing cost, fraud risk, and regulatory
> compliance. CSO is a multi-agent system that picks the optimal scheme
> per transaction by balancing all four."

**What's on screen:** `docs/architecture.png` (the Mermaid render)
filling the screen. Cursor doesn't move; just narrate over the diagram.

### [0:30–1:30] The pipeline — point at the diagram

> "Pipeline is built on LangGraph. Transactions enter at the top.
> A planning step decides which schemes are worth evaluating for this
> transaction.
>
> Then three agents run in parallel — auth, cost, and fraud — each a
> ReAct loop with its own tools. The auth agent has a self-correction
> loop after it: that's the Reflexion pattern. A critic node checks the
> emitted score against the baseline data; if it drifts too much, it
> loops back for one revision.
>
> After aggregation and an LLM reflection pass, there's an HITL
> interrupt for high-value transactions — anything over $500 pauses
> for human approval. Then a deterministic compliance gate filters
> by IFR, Durbin, OptBlue, and merchant contract rules. Finally an
> explanation agent writes the merchant-facing justification."

**What's on screen:** zoom into specific parts of the architecture
diagram as you describe each.

### [1:30–2:30] The codebase tour

> "Quick look at the layout."

Open the file tree in VS Code. Walk through:

> "`orchestrator/graph.py` — the LangGraph DAG, twelve nodes.
> `agents/` — three scoring agents, each with its own tools.
> `compliance/` — the deterministic gate and rule set.
> `ml/` — XGBoost training plus serving wrapper.
> `evaluation/` — three-layer harness with custom evaluators.
> `llm_clients.py` — multi-provider abstraction. The whole system
> runs on OpenAI, Anthropic, Gemini, Llama via Groq, or local Ollama
> by changing one env var."

Open `llm_clients.py`, scroll to `TIER_DEFAULTS`. Stay 5 seconds.

> "Tier system: agents ask for `fast`, `smart`, or `judge`. Defaults
> per provider are in this table. Code never imports a vendor SDK
> directly."

### [2:30–3:30] Live demo — run a transaction end-to-end

Switch to the dashboard, already loaded at `localhost:8501`.

Click **🔍 Deep Dive** tab. Pick `txn_0001` (vanilla EU dual-brand).

> "Here's a vanilla EU dual-brand transaction. The pipeline ranked
> Mastercard above Visa because the issuer's auth rate is 1.3 points
> higher on Mastercard. Both passed the deterministic compliance gate."

Expand the Rejected schemes section if any exist, then show the
weighted ranking chart.

> "This chart decomposes the score: the blue bar is p_auth weighted
> at 1.0, the red bar is the fee penalty at 0.15, and the orange bar
> is the fraud penalty at 0.30. The diamond is the final weighted
> score the planner uses to route."

Switch to **txn_0005** or pick a high-value transaction.

> "Now a transaction over the $500 threshold. In Live run mode this
> triggers the Human-in-the-Loop gate — the pipeline pauses and
> waits for an Approve or Reject before the compliance step runs."

### [3:30–4:30] Evaluation tab — proof it actually works

Click **🧪 Evaluations** tab (or describe from `evaluation/results/latest.md`).

> "The evaluation harness has three layers. Layer one is per-agent
> quality — tool-use accuracy on twelve hand-labelled cases, measuring
> whether the auth agent calls the right tools in the right order.
> Layer two is system quality — CSO versus two baselines on fifty
> synthetic transactions. Layer three is stress tests — schema
> violations, prompt injection, compliance impossibilities."

Point at each panel as you describe:

> "Tool-use accuracy on the twelve hand-labeled cases is <YOUR NUMBER>.
>
> Layer two — CSO beats the cheapest-first baseline on expected
> revenue per hundred transactions, and crucially has zero compliance
> violations where cheapest-first has <YOUR NUMBER>%.
>
> Layer three — four out of four stress cases caught."

### [4:30–5:00] Closing — what's intentionally not done + ask

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
- **Final length target: 4:45–5:30.** Anything over 7 minutes loses
  attention.

## Common viva questions to be ready for after the demo

In rough order of likelihood:

1. *"Walk me through the self-correction loop. What happens when the
   critic flags?"* — Show `orchestrator/graph.py:critique_auth_score`,
   trace the conditional edge.
2. *"Why are three agents running in parallel rather than sequentially?"*
   — Independent state writes (auth_scores, cost_scores, fraud_scores),
   no data dependency, so LangGraph super-steps them for free.
3. *"What's the failure mode of your fraud model in production?"* —
   PCA features unavailable at inference time. Documented limitation.
   Real fix is feedback-driven retraining on features actually available.
4. *"How does the HITL gate work technically?"* — `langgraph.types.interrupt()`
   suspends the graph and serialises state to MemorySaver. The dashboard
   resumes it via `pipeline.invoke(Command(resume=...))`.
5. *"Why is the compliance gate deterministic rather than LLM-driven?"* —
   Auditability. Regulatory rules are precise and don't need LLM
   interpretation. The gate is plain Python that can be unit-tested
   exhaustively. LLMs are used where ambiguity requires reasoning.
6. *"How does multi-provider support work?"* — `init_chat_model("provider:model")`
   from LangChain 0.3. One env var switches everything. Show TIER_DEFAULTS
   in `llm_clients.py`.
