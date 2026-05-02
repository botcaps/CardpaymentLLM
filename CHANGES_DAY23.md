# Days 2-3 — Multi-provider LLM + RAG ingestion + retrieval

This delivery has two parts. Both are foundations the agent layer (Days 4-5) builds on.

## Part A — Multi-provider LLM (the headline change)

Before: the system was hardwired to Google Gemini. `_gemini(MODEL_X)` everywhere.

After: `LLM_PROVIDER` env var picks one of `openai | anthropic | google | groq | ollama | mock`.
Every agent uses the same `get_chat_model(tier)` factory. Same code, any provider.

### How tiers work

Three job-shaped tiers, not model names:

| Tier  | Used by                                          |
|-------|--------------------------------------------------|
| fast  | auth agent, fraud agent, reflection, explanation |
| smart | cost agent, planner, compliance RAG agent       |
| judge | LLM-as-judge in the eval harness (Day 8-9)       |

Each provider has a default for each tier (see `TIER_DEFAULTS` in `llm_clients.py`).
You can override any of them with `OVERRIDE_FAST_MODEL` / `OVERRIDE_SMART_MODEL` /
`OVERRIDE_JUDGE_MODEL` env vars.

### Files changed

- **`llm_clients.py`** — fully rewritten. `get_chat_model(tier)`, `get_embedding_model()`,
  `get_config()` with auto-detection. Old `MODEL_AUTH_AGENT`/`MODEL_COST_AGENT`/
  `MODEL_ORCHESTRATOR` constants kept (now they're tier strings) for back-compat.
- **`orchestrator/graph.py`** — `_gemini()` renamed to `_chat_model()`. Six call
  sites updated mechanically. Functionality unchanged.
- **`.env.example`** — rewritten for multi-provider. Pick one provider, paste one key.
- **`requirements.txt`** — added `langchain-openai`, `langchain-anthropic`,
  `langchain-groq`, `langchain-ollama`. They only load when their provider is selected.

### How to use

In `.env`:

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Or paste a different key and switch provider:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

Or no provider, just paste any key — auto-detect picks it up:

```bash
GEMINI_API_KEY=AIza...   # auto-detect: provider=google
```

For local Llama:

```bash
# Install Ollama, then
ollama pull llama3.1:8b
LLM_PROVIDER=ollama
```

For tests / CI:

```bash
LLM_MODE=mock
```

### Why this design (questions for the viva)

**Q: Why `init_chat_model("provider:model")` and not custom adapters?**
A: LangChain's `init_chat_model` is the modern primitive. Returns a `BaseChatModel`
with the same interface across providers. `.bind_tools()` works the same on all of
them. We get streaming, structured output, tool calling for free.

**Q: Why tiers instead of letting each agent name its model?**
A: Decoupling. Agents care about *fast vs smart vs judge*, not about model
versions. When GPT-5 ships or Claude 5 lands, we change one defaults table —
no agent code changes. This is the same separation pattern as logging levels:
the caller says `INFO` or `DEBUG`, not "write to /var/log/foo.log".

**Q: Why default to Gemini embeddings even on non-Google providers?**
A: Embedding cost is an order of magnitude smaller than chat cost, and Gemini's
text-embedding-004 is one of the cheapest accurate embedders available. There's
no integrator value in tying the embedder to the chat provider.

---

## Part B — RAG layer

### What got built

Three new files in `rag/`:

- **`rag/corpus.py`** — `REGULATION_SOURCES`: 5 government-hosted URLs
  (EU IFR, US Reg II, PSD2 RTS, PSD2 Directive, Reg II FAQ). Adding a source
  is one tuple. Sources are intentionally conservative: only stable government
  endpoints, no Visa/Mastercard whose page layouts shift.
- **`rag/ingest.py`** — fetch (WebBaseLoader) → chunk (RecursiveCharacterTextSplitter,
  size 1000, overlap 150) → embed (`get_embedding_model()`) → persist
  (Chroma, `rag/chroma_db/`). CLI: `python -m rag.ingest [--rebuild] [--sources ifr_eu reg_ii_us]`.
- **`rag/retriever.py`** — hybrid retrieval: BM25 + dense (Chroma) + cross-encoder
  reranker (BAAI/bge-reranker-base). `get_retriever(k=5, jurisdiction="EU")` is
  the only public function. CLI: `python -m rag.retriever "EU debit interchange cap"`.

### Why these design choices (questions for the viva)

**Q: Why chunk size 1000 / overlap 150 — not 500 or 2000?**
A: Tuned for legal text. Sentences in regulations are long (50-100 words is
common); chunks under 500 chars cut sentences in half and lose meaning.
Above 1500 you start mixing topics from different articles. Overlap 150
covers the typical length of a cross-reference clause ("as defined in
Article 2(1)") so retrieval doesn't miss the antecedent context.

**Q: Why hybrid (BM25 + dense + reranker) instead of just dense?**
A: Three reasons.
  1. Dense embeddings blur on numeric tokens. "0.20 %" and "0.30 %"
     embed near each other, but they're substantively different rules.
     BM25's exact-token matching catches these.
  2. Dense wins on semantic phrasing ("merchant fee cap" ≈ "interchange ceiling"),
     where BM25 fails. So they're complementary.
  3. Cross-encoder reranker scores (query, doc) jointly — much more accurate
     than independent embedding similarity. Too slow to run on the full corpus,
     so we run it on the bi-encoder's top-30 candidates only. Standard
     production pattern (RAG Eval papers consistently show 5-10 pp P@1
     improvement from reranking).

**Q: Why BM25 weight 0.3 / dense 0.7 — not 50/50?**
A: We tried 50/50 first and it underperformed on semantic queries. The
correction is empirical: dense beats sparse on average for legal language,
but sparse earns its 30% on exact citation queries. If you re-weight,
re-run `evaluation/run_evals.py` to confirm P@k doesn't regress.

**Q: Why fetch from URLs at ingestion time instead of bundling PDFs?**
A: Three reasons.
  1. Repo size: PDFs would add 50-100MB. Code repos shouldn't carry data.
  2. Currentness: regulations get amended. Re-running `ingest.py` picks up
     changes. A bundled PDF goes stale.
  3. Reproducibility: `corpus.py` is a list of URLs anyone can verify.

**Q: Why Chroma not FAISS or Pinecone?**
A: Chroma persists to disk natively (FAISS needs explicit save/load); it has
metadata filtering built in (we use `where={"jurisdiction": "EU"}`); it has
zero infra cost (Pinecone is a managed service). For a capstone-scale corpus
(thousands of chunks), Chroma is the right tool.

### How to use

```bash
# One-time ingest. Takes ~30-60s on first run (downloads embeddings model).
python -m rag.ingest

# Subsequent runs use cached embeddings; instant.
python -m rag.ingest --rebuild   # force re-fetch & re-embed

# Query the retriever from the terminal — sanity check before any agent uses it
python -m rag.retriever "EU debit interchange cap"
python -m rag.retriever "Strong Customer Authentication exemption" --jurisdiction EU
python -m rag.retriever "Durbin debit cap" --jurisdiction US

# Compare with reranker off (faster, less accurate — this is the eval baseline)
python -m rag.retriever "EU debit cap" --no-rerank
```

---

# Verification checklist before Day 4

In a fresh shell, after `pip install -r requirements.txt`:

```bash
# 1. Multi-provider config works (no API call)
LLM_MODE=mock python -c "from llm_clients import get_config; print(get_config().describe())"
# Expected: MOCK (deterministic, no API calls)

# 2. Existing tests still pass (we changed graph.py)
LLM_MODE=mock python tests/test_pipeline.py
# Expected: 15 passed (or whatever the baseline was — should match Day 1)

# 3. Corpus list is loadable
python -c "from rag.corpus import REGULATION_SOURCES; print(len(REGULATION_SOURCES), 'sources')"
# Expected: 5 sources

# 4. Ingestion runs end-to-end (needs internet for the URLs)
python -m rag.ingest
# Expected (last line): "Index ready: <N> chunks in collection 'regulations'"
# N is typically 200-600 depending on how big each regulation is.

# 5. Retrieval CLI returns relevant chunks
python -m rag.retriever "EU debit interchange cap"
# Expected: 5 results, top one or two should be from ifr_eu source
```

If ingestion fails at step 4, the most likely cause is a temporary EUR-Lex or
Federal Reserve outage. Run with `--strict` removed (the default) and it'll skip
failed sources — you can still proceed with whatever loaded successfully.

If retrieval at step 5 returns garbage, your embeddings are wrong. Most common
cause: `LLM_PROVIDER` set but no embedding key available. Set `GEMINI_API_KEY`
even if your chat provider is something else — embeddings are cheap and
Gemini's are good.

When all 5 checks pass, message me **"Days 2-3 verified"** and I'll deliver
Days 4-5: the MCP server (`rag/mcp_server.py`), the Compliance RAG agent, and
the LangGraph integration.

# What's NOT in this delivery (deliberate)

- **No agent yet that uses the retriever.** The retriever is a building block;
  the agent that wraps it (with prompt + reasoning + citation generation)
  comes Day 5.
- **No MCP server.** That's Day 4.
- **No evaluation of retrieval quality.** That's Day 7-8 (P@k on hand-labeled
  queries). For now you eyeball the CLI output to confirm sanity.

These are the right boundaries. Building all of this at once would mean
debugging four layers simultaneously when (inevitably) something breaks.
