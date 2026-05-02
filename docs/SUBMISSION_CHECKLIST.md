# Submission Checklist

A grader's first 5 minutes with your project decides 30% of your mark.
This checklist makes those 5 minutes go well.

## Before submitting — run these in order

```bash
# 1. Fresh install in a clean venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Confirm tests pass (no API key needed)
LLM_MODE=mock python tests/test_pipeline.py
# REQUIRED: 16 passed, 0 failed

# 3. Build the RAG corpus (requires internet, ~60s)
python -m rag.ingest
# REQUIRED: "Index ready: <N> chunks in collection 'regulations'"

# 4. Train the fraud model (requires Kaggle credentials in ~/.kaggle/kaggle.json)
python -m ml.training.train_fraud
# REQUIRED: ml/fraud_model.pkl created, AUC-PR > 0.5

# 5. Run all eval layers in mock mode first (no API spend)
LLM_MODE=mock python -m evaluation.run_evals --quick
# REQUIRED: report renders all 5 sections, no Python errors

# 6. THE BIG ONE — full live-mode eval
# This costs maybe $0.50–$2 in API calls depending on provider
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... \
  python -m evaluation.run_evals --live-tools
# REQUIRED: faithfulness has live-mode scores (some 4s and 5s)
#           tool-use accuracy is between 0.7 and 1.0 (real LLM behaviour)
#           CSO numerically beats both baselines on Exp.Rev/100

# 7. Manually confirm the dashboard works
streamlit run dashboard.py
# Click through every tab, especially the new ones (Compliance RAG, Evaluations).
# Make sure the Evaluations tab shows real numbers (it reads
# evaluation/results/latest.json — make sure that file exists from step 6).
```

## Update the report with real numbers

Open `docs/REPORT.md`. Find every `<TODO: ...>` placeholder. For each,
paste the actual number from your live eval run.

There are roughly 18 placeholders. Don't skip any. A grader who sees
`<TODO>` in your final report deducts marks immediately.

The most important numbers to fill in:
- §3.1 — total chunks in the Chroma index
- §6 — fraud model AUC-ROC and AUC-PR (from `ml/fraud_model_metrics.json`)
- §7.1 — Layer 1 evaluator scores (from `evaluation/results/latest.json`)
- §7.2 — Layer 2 baseline comparison numbers
- §7.3 — Layer 3 stress test outcomes

## Render the architecture diagram

1. Open `docs/architecture.mmd` — copy the entire file
2. Paste into https://mermaid.live
3. Click **Actions → PNG** (or SVG for vector)
4. Save as `docs/architecture.png`
5. The README and REPORT both reference this file

## Record the demo video

Use `docs/DEMO_SCRIPT.md`. Aim for 5:45–6:30. Export as 1080p MP4.
Save as `docs/demo.mp4` or upload to YouTube/Drive and put the link
in the README's Quick Start section.

## Final repo cleanup

```bash
# Confirm .gitignore is doing its job
git status
# These should NOT be staged:
#   data/kaggle_cache/
#   rag/chroma_db/
#   ml/fraud_model.pkl
#   evaluation/results/eval_*.json (or .md, except 'latest.json' if you want graders to see one without running)
#   __pycache__/, .venv/, .env

# IMPORTANT: rotate any API key you ever pasted into the code.
# Even if it's been deleted from the codebase, treat it as compromised.
# - aistudio.google.com/apikey         (Gemini)
# - platform.openai.com/api-keys       (OpenAI)
# - console.anthropic.com/             (Anthropic)
# - console.groq.com/keys              (Groq)
```

## What graders look for in the first 5 minutes

In rough priority order — make sure each one is obvious:

1. **README leads with capstone framing.** Yours does — the "What this
   is for" section maps every rubric concept to a file path. Graders
   should not have to hunt.
2. **Architecture diagram.** A grader who opens REPORT.md should see
   the diagram on page 1 or 2. If you skipped rendering the .mmd to
   PNG, it shows as raw text and looks unfinished.
3. **Demo video, ≤ 7 minutes.** Most graders watch this before reading
   the code. If it's missing or 15 minutes long, you lose marks even
   if everything else is perfect.
4. **`python tests/test_pipeline.py` passes** in a fresh clone.
   The grader will run this. If it fails, mark drops by 10+ pp.
5. **`requirements.txt` installs cleanly** on Python 3.10–3.12.
6. **A real evaluation report exists.** `evaluation/results/latest.md`
   should be a real file with real numbers, not a placeholder.
7. **`docs/REPORT.md` has no `<TODO>` placeholders left.**
8. **The repo doesn't contain secrets.** Run `git log -p | grep -i 'sk-\|AIza\|gsk_'`. If anything matches, the key has been rotated, right?
9. **The repo doesn't contain bulk data.** No CSVs, no .pkl over a few
   MB, no PDFs. `du -sh .git/` should be reasonable (under 50MB).
10. **Tests aren't all `assert True`.** Yours aren't (16 real scenarios)
    but make sure none drifted to no-ops.

## Things you'll be tempted to do that aren't worth it

- Adding a 10th dashboard tab. The current 9 cover the rubric. More tabs dilute focus.
- Training the fraud model on the full 284k rows for 0.001 better AUC. The 50k subset is fine for the report.
- Hand-validating LLM-as-judge faithfulness. Acknowledge the limitation in the report and the viva. Spending 8 hours grading 30 examples to claim "verified" buys you maybe 2 marks; spending those 8 hours polishing the demo video buys you 5.
- Adding a 5th LLM provider. Four (OpenAI / Anthropic / Google / Groq) is already more than the rubric requires.
- Writing a 10-page report. Four pages of dense, well-cited content beats ten pages of padding.

## What to actually spend time on instead

In rough order of mark-per-hour ROI:

1. **Polish the demo video.** Re-record any segment where you stumbled. 30 minutes here matters more than 8 hours of code polish.
2. **Fill in every `<TODO>` in REPORT.md.** A complete report with real numbers beats a half-filled report with extra features.
3. **Have the viva questions ready.** Read `docs/REPORT.md` §9 (limitations) carefully — that's where most viva probes go.
4. **Render the architecture diagram cleanly.** Mermaid → PNG. Pasting raw .mmd source into your report looks unfinished.
5. **One end-to-end test on a fresh clone.** Open a new shell, clone your own repo, run all the steps above. If anything breaks, fix it now.

## What the rubric (probably) weighs

Based on the original prompt — equal weight on depth / demo / rigor:

- Code quality + agentic pattern depth: ~33%
- Working demo: ~33%
- Documentation + evaluation rigor: ~33%

Implications:
- A polished demo with mediocre code beats brilliant code with no demo.
- A 4-page rigorous report with real numbers beats a 12-page essay.
- An eval harness that prints "RUN ME" and produces real numbers beats
  hand-curated screenshots that grader can't reproduce.

---

You've got this. Submit, then go to bed early.
