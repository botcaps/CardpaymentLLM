# Day 1 — Foundation cleanup

Four changes. None add new features. All make every later day safer.

## 1. Fixed hardcoded API key in `orchestrator/graph.py`

**Bug:** the original code had `os.environ["AIzaSyAyL-LXiaYFyzR_WuLzQxuXkvo61JTWEQ0"]`
which used a real API-key string as the *name* of the env var to look up. This
would fail at runtime with `KeyError` whenever LIVE mode was used.

**Fix:** the new `_gemini()` reads `GEMINI_API_KEY` (with `GOOGLE_API_KEY` as a
fallback) and raises a clear error if neither is set. Mirrors the convention
already in `llm_clients.gemini_client()`.

**Action for you:** if `AIzaSyAyL-LXiaYFyzR_WuLzQxuXkvo61JTWEQ0` was ever a real
key tied to your Google account, **rotate it now** at
https://aistudio.google.com/apikey. Even if it was a placeholder, treat it as
compromised since it's been in your repo.

## 2. `data/kaggle_loader.py` → `kagglehub`

**Before:** wrote the CSV to `data/kaggle_cache/creditcard.csv` (144 MB inside
the repo).

**After:** calls `kagglehub.dataset_download("mlg-ulb/creditcardfraud")`, which
caches to `~/.cache/kagglehub/...`. The repo never contains the CSV.

The public function `get_or_download_csv()` still returns a string path, so
`kaggle_features.py` and the dashboard work unchanged.

I also deleted `data/kaggle_cache/` from disk — you should not commit it.

## 3. New `.gitignore`

The repo had no `.gitignore` at all. The new one keeps out:

- `__pycache__/`, `*.pyc`, `.venv/` etc. (standard Python)
- `.env`, `.kaggle/`, `*.key` (secrets)
- `data/kaggle_cache/` (datasets fetched at runtime)
- `rag/chroma_db/` (vector store, rebuildable)
- `ml/*.pkl` (trained models — see note in file)
- `evaluation/results/*.json|png|html` (regeneratable)

## 4. Updated `requirements.txt`

Pinned versions for the entire planned stack — LangChain, LangGraph, LangSmith,
MCP, RAG (Chroma + sentence-transformers), ML (XGBoost + SMOTE), dashboard,
testing.

The MCP entry is the one to watch:

```
mcp>=1.2.0,<2.0.0
langchain-mcp-adapters>=0.1.0,<0.3.0
```

`langchain-mcp-adapters` 0.2.x had API changes from 0.1.x. The pin window
covers both because the public API we'll use (`MultiServerMCPClient`,
`get_tools()`) is stable across that range.

## 5. New `.env.example`

Reminds graders / users which env vars matter:
- `GEMINI_API_KEY`, `LLM_MODE`
- `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`
- (Kaggle uses `~/.kaggle/kaggle.json`, not an env var)

---

# What to verify before Day 2

Run these three checks; if any fail, fix before moving on.

```bash
cd cso_v2

# 1. Install dependencies
pip install -r requirements.txt

# 2. Imports cleanly (mock mode — no API key needed)
python -c "from orchestrator.orchestrate import orchestrate; print('imports OK')"

# 3. Existing tests still pass
python tests/test_pipeline.py
```

Expected output of the last command: `15 passed, 0 failed` (or close — the
guardrail-related scenario tests need the existing Guardrail module, which
is unchanged).

If `pip install` fails on `chromadb` or `sentence-transformers` (these are
heavy installs), it's almost always a Python-version mismatch. Confirm
Python 3.10–3.12; 3.13 still has gaps for some of these.

If `langchain-mcp-adapters` fails to install, that's the one I'd most expect
trouble from. If it does, pin `mcp>=1.2.0,<1.5.0` and retry — sometimes the
newer mcp core breaks adapter compatibility briefly between releases.

Once all three checks pass, message me with "Day 1 verified" and we'll move
to Day 2 (RAG ingestion: WebBaseLoader + chunking + ChromaDB).
