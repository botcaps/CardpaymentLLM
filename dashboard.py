"""
CSO Multi-Agent Pipeline — Dashboard

Run:
    streamlit run dashboard.py
"""
from __future__ import annotations
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from data.samples import get_samples
from data.feature_store import AUTH_RATE_HISTORY, ISSUER_HEALTH, BIN_TABLE, DECLINE_PATTERNS
from data.interchange import INTERCHANGE_TABLE, SCHEME_FEES, MERCHANT_CONTRACTS
from orchestrator.orchestrate import orchestrate, _build_trace
from orchestrator.graph import pipeline, HIGH_VALUE_THRESHOLD
from langgraph.types import Command
from llm_clients import get_config
from compliance.rules.all_rules import (
    passes_ifr_with_breakdown, passes_durbin, passes_optblue,
    passes_token_lock, merchant_eligible,
)

# ─── PAGE CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSO Pipeline",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
.section-header {
    font-size: 1.05rem; font-weight: 600; color: #1e40af;
    border-bottom: 2px solid #bfdbfe; padding-bottom: 4px; margin-bottom: 12px;
}
div[data-testid="stExpander"] details { border: 1px solid #e2e8f0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

SCHEME_COLORS = {
    "visa": "#1d4ed8", "mastercard": "#dc2626",
    "amex": "#0369a1",  "discover": "#ea580c",
}
MCC_NAMES = {
    "5411": "Grocery", "5812": "Restaurant",
    "4511": "Airlines", "7011": "Lodging",
    "5999": "Misc Retail", "7995": "Gambling",
}
RULES = ["merchant", "token_lock", "optblue", "ifr", "durbin"]


# ─── DATA LOADING ───────────────────────────────────────────────────
def _run_txn_in_thread(txn):
    try:
        decision, trace = asyncio.run(orchestrate(txn))
        return {"txn": txn, "decision": decision, "trace": trace}
    except Exception as exc:
        from observability.tracer import Trace
        return {"txn": txn, "decision": None,
                "trace": Trace(txn_id=txn.txn_id, error=f"Pipeline error: {exc}")}


@st.cache_data(show_spinner=False)
def run_pipeline(source: str = "kaggle", csv_path: str | None = None, n: int = 10):
    samples = get_samples(source=source, csv_path=csv_path, n=n)
    with ThreadPoolExecutor(max_workers=2) as executor:
        out = list(executor.map(_run_txn_in_thread, samples))
    return out


def build_summary_df(results):
    rows = []
    for r in results:
        txn, dec, trace = r["txn"], r["decision"], r["trace"]
        contract = MERCHANT_CONTRACTS.get(txn.merchant_id, {})
        p_fraud = None
        if dec and trace.fraud_scores:
            fs = next((f for f in trace.fraud_scores if f.get("scheme") == dec.scheme), None)
            if fs:
                p_fraud = fs.get("p_fraud")
        rows.append({
            "txn_id":     txn.txn_id,
            "merchant":   contract.get("name", txn.merchant_id),
            "region":     txn.region,
            "card_type":  txn.card_type,
            "bin":        txn.bin,
            "amount":     txn.amount_minor / 100,
            "currency":   txn.currency,
            "channel":    txn.channel,
            "3ds":        txn.three_ds_status.replace("authenticated_", ""),
            "candidates": len(trace.candidates),
            "decided":    dec is not None,
            "scheme":     dec.scheme if dec else "—",
            "p_auth":     dec.p_auth if dec else None,
            "fee_bps":    dec.fee_bps if dec else None,
            "p_fraud":    p_fraud,
            "hitl":       txn.amount_minor >= HIGH_VALUE_THRESHOLD,
            "degraded":   dec.degraded if dec else False,
            "rejections": len(dec.rejected_schemes) if dec else len(trace.candidates),
            "error":      trace.error,
            "hour":       txn.hour_of_day,
            "mcc":        txn.mcc,
        })
    return pd.DataFrame(rows)


def _run_live_with_hitl(txn):
    config   = {"configurable": {"thread_id": f"live_{txn.txn_id}"}}
    resume_k = f"hitl_resume_{txn.txn_id}"
    if resume_k in st.session_state:
        human_decision = st.session_state.pop(resume_k)
        state = pipeline.invoke(Command(resume=human_decision), config=config)
        return state, False
    state       = pipeline.invoke({"txn": txn}, config=config)
    graph_state = pipeline.get_state(config)
    if graph_state.next:
        for task in graph_state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                st.session_state[f"hitl_data_{txn.txn_id}"] = task.interrupts[0].value
                break
        return state, True
    return state, False


# ─── SIDEBAR ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💳 CSO Pipeline")
    cfg = get_config()
    mode_color = "🟢" if cfg.mode == "live" else "🟡"
    st.info(f"{mode_color} **Mode:** {cfg.describe()}")

    if st.button("🔄 Re-run pipeline", use_container_width=True, type="primary"):
        run_pipeline.clear()
        st.rerun()

    st.divider()
    st.caption("Data Source")
    data_source = st.radio(
        "Transaction source",
        ["Kaggle (real data)", "Mock samples"],
        key="data_source",
    )
    kaggle_n = 30
    if data_source == "Kaggle (real data)":
        kaggle_n = st.slider("Transactions", min_value=10, max_value=100, value=10, step=10)


# ─── LOAD DATA ──────────────────────────────────────────────────────
if data_source == "Kaggle (real data)":
    try:
        from data.kaggle_loader import get_or_download_csv
        with st.spinner("Fetching dataset from Kaggle…"):
            kaggle_csv_path = get_or_download_csv()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    feat_key = "kaggle_features_loaded"
    if feat_key not in st.session_state:
        with st.spinner("Loading feature tables…"):
            from data.kaggle_features import load_kaggle_feature_tables_cached
            from data.feature_store import inject_kaggle_features
            from data.fraud_store import inject_kaggle_fraud_features
            tables = load_kaggle_feature_tables_cached(kaggle_csv_path)
            inject_kaggle_features(tables)
            inject_kaggle_fraud_features(tables)
            run_pipeline.clear()
            st.session_state[feat_key] = True

    with st.spinner(f"Running pipeline on {kaggle_n} transactions…"):
        results = run_pipeline("kaggle", kaggle_csv_path, kaggle_n)
else:
    with st.spinner("Running pipeline on mock transactions…"):
        results = run_pipeline("mock")

df       = build_summary_df(results)
decided  = df[df["decided"]]
rejected = df[~df["decided"]]


# ─── TABS ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🔍 Deep Dive",
    "🔐 Auth & Risk",
    "💰 Cost",
    "🛡️ Compliance",
    "🤖 Agents",
])


# ════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Pipeline Overview")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Transactions",  len(df))
    k2.metric("Decided",       len(decided),
              delta=f"{len(decided)/len(df):.0%}" if len(df) else "—")
    k3.metric("Rejected",      len(rejected),
              delta=f"-{len(rejected)}" if len(rejected) else "0",
              delta_color="inverse")
    k4.metric("Avg p(auth)",   f"{decided['p_auth'].mean():.3f}"  if len(decided) else "—")
    k5.metric("Avg fee (bps)", f"{decided['fee_bps'].mean():.1f}" if len(decided) else "—")
    k6.metric("Degraded runs", int(decided["degraded"].sum()) if len(decided) else 0,
              delta_color="inverse")

    st.divider()
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<div class="section-header">Scheme Distribution</div>',
                    unsafe_allow_html=True)
        if len(decided):
            sc = decided["scheme"].value_counts().reset_index()
            sc.columns = ["scheme", "count"]
            fig = px.pie(sc, values="count", names="scheme",
                         color="scheme", color_discrete_map=SCHEME_COLORS,
                         hole=0.45, height=300)
            fig.update_traces(textfont_size=13, textinfo="label+percent")
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Auth Probability vs Fee (bps)</div>',
                    unsafe_allow_html=True)
        if len(decided):
            fig = px.scatter(
                decided, x="fee_bps", y="p_auth",
                color="scheme", symbol="region", size="amount",
                hover_data=["txn_id", "merchant", "card_type"],
                color_discrete_map=SCHEME_COLORS,
                labels={"fee_bps": "Total Fee (bps)", "p_auth": "P(auth)"},
                height=300,
            )
            fig.update_layout(margin=dict(t=20, b=40), legend_title_text="Scheme")
            fig.add_hline(y=decided["p_auth"].mean(), line_dash="dot",
                          line_color="gray", annotation_text="avg p(auth)")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">By Region</div>',
                    unsafe_allow_html=True)
        fig = px.histogram(df, x="region", color="decided", barmode="group", height=240,
                           color_discrete_map={True: "#22c55e", False: "#ef4444"},
                           labels={"decided": "Decided"})
        fig.update_layout(margin=dict(t=10, b=30), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">By Card Type</div>',
                    unsafe_allow_html=True)
        if len(decided):
            fig = px.histogram(decided, x="card_type", color="scheme",
                               barmode="group", height=240,
                               color_discrete_map=SCHEME_COLORS)
            fig.update_layout(margin=dict(t=10, b=30), legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-header">All Transactions</div>',
                unsafe_allow_html=True)
    disp = df[[
        "txn_id", "merchant", "region", "card_type", "amount", "currency",
        "channel", "candidates", "scheme", "p_auth", "fee_bps", "degraded", "error",
    ]].copy()
    disp["p_auth"]  = disp["p_auth"].apply(lambda x: f"{x:.3f}" if x else "—")
    disp["fee_bps"] = disp["fee_bps"].apply(lambda x: f"{x:.1f}" if x else "—")
    disp["amount"]  = disp.apply(lambda r: f"{r['amount']:.2f} {r['currency']}", axis=1)
    disp = disp.drop(columns=["currency"])
    st.dataframe(disp, hide_index=True, use_container_width=True,
                 column_config={
                     "txn_id":     st.column_config.TextColumn("TXN ID"),
                     "degraded":   st.column_config.CheckboxColumn("Degraded"),
                     "candidates": st.column_config.NumberColumn("# Candidates"),
                 })


# ════════════════════════════════════════════════════════════════════
#  TAB 2 — TRANSACTION DEEP DIVE
# ════════════════════════════════════════════════════════════════════
with tab2:
    labels = [
        f"{r['txn'].txn_id}  ·  "
        f"{MERCHANT_CONTRACTS.get(r['txn'].merchant_id, {}).get('name', r['txn'].merchant_id)}"
        f"  ({r['txn'].region}, {r['txn'].card_type})"
        for r in results
    ]
    choice = st.selectbox("Select transaction", labels, key="txn_select")
    idx    = labels.index(choice)
    r      = results[idx]
    txn    = r["txn"]

    is_high_value = txn.amount_minor >= HIGH_VALUE_THRESHOLD
    if is_high_value:
        st.info(
            f"This transaction ({txn.amount_minor/100:.2f} {txn.currency}) exceeds the "
            f"{HIGH_VALUE_THRESHOLD/100:.0f} threshold and triggers the **Human-in-the-Loop gate**. "
            "Switch to **Live run** mode to approve or reject interactively."
        )

    view_mode = st.radio(
        "View mode",
        ["Cached result (auto-approved HITL)", "Live run (interactive HITL)"],
        horizontal=True, key="view_mode",
    )

    if view_mode == "Live run (interactive HITL)":
        prev_txn = st.session_state.get("_dd_prev_txn")
        if prev_txn != txn.txn_id:
            for k in [f"hitl_data_{txn.txn_id}", f"hitl_data_{prev_txn}",
                      f"_live_state_{prev_txn}"]:
                st.session_state.pop(k, None)
            st.session_state["_dd_prev_txn"] = txn.txn_id

        live_state_key = f"_live_state_{txn.txn_id}"
        if live_state_key not in st.session_state or st.button("🔄 Re-run live", key="rerun_live"):
            st.session_state.pop(live_state_key, None)
            with st.spinner("Running pipeline…"):
                live_state, interrupted = _run_live_with_hitl(txn)
            st.session_state[live_state_key] = (live_state, interrupted)

        live_state, interrupted = st.session_state.get(live_state_key, ({}, False))

        if interrupted:
            idata = st.session_state.get(f"hitl_data_{txn.txn_id}", {})
            st.warning(idata.get("message", "High-value transaction requires approval."))
            st.subheader("Ranking available for review")
            iranked = idata.get("ranked", [])
            if iranked:
                st.dataframe(pd.DataFrame(iranked)[[
                    c for c in ["scheme", "p_auth", "p_fraud", "total_fee_bps", "weighted_score"]
                    if c in pd.DataFrame(iranked).columns
                ]], hide_index=True, use_container_width=True)
            if idata.get("reflection"):
                st.info(f"**Reflection:** {idata['reflection']}")
            approve_col, reject_col, _ = st.columns([1, 1, 3])
            if approve_col.button("✅ Approve", type="primary"):
                st.session_state[f"hitl_resume_{txn.txn_id}"] = {"approved": True}
                st.session_state.pop(live_state_key, None)
                st.rerun()
            if reject_col.button("❌ Reject"):
                st.session_state[f"hitl_resume_{txn.txn_id}"] = {"approved": False}
                st.session_state.pop(live_state_key, None)
                st.rerun()
            st.stop()

        decision = live_state.get("decision")
        trace    = _build_trace(txn, live_state)
    else:
        decision = r["decision"]
        trace    = r["trace"]

    st.divider()
    ctx_col, dec_col = st.columns([3, 2])

    with ctx_col:
        st.markdown('<div class="section-header">Transaction Context</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TXN ID",  txn.txn_id)
        c2.metric("BIN",     txn.bin)
        c3.metric("Amount",  f"{txn.amount_minor/100:.2f} {txn.currency}")
        c4.metric("Hour",    f"{txn.hour_of_day}:00")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Merchant", MERCHANT_CONTRACTS.get(txn.merchant_id, {}).get("name", txn.merchant_id))
        c6.metric("Region",   txn.region)
        c7.metric("Card",     txn.card_type)
        c8.metric("Channel",  txn.channel)

        st.markdown(
            f"**3DS:** `{txn.three_ds_status}`  |  "
            f"**MCC:** `{txn.mcc}` ({MCC_NAMES.get(txn.mcc, 'unknown')})  |  "
            f"**Issuer:** `{txn.issuer_country}`  |  "
            f"**Acquirer:** `{txn.acquirer_country}`"
        )
        if txn.is_network_token:
            st.warning(f"Network token locked to: **{txn.token_network}**")

    with dec_col:
        st.markdown('<div class="section-header">Pipeline Decision</div>',
                    unsafe_allow_html=True)
        if decision:
            color = "#f59e0b" if decision.degraded else "#22c55e"
            st.markdown(
                f"<div style='background:{color};padding:16px;border-radius:10px;color:white'>"
                f"<div style='font-size:1.8rem;font-weight:800'>✅ {decision.scheme.upper()}</div>"
                f"<div style='margin-top:8px'>p(auth) = <b>{decision.p_auth:.4f}</b> &nbsp;|&nbsp; "
                f"fee = <b>{decision.fee_bps:.1f} bps</b></div>"
                f"{'<div style=\"margin-top:4px\">⚠ Degraded mode</div>' if decision.degraded else ''}"
                f"</div>", unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background:#ef4444;padding:16px;border-radius:10px;color:white'>"
                f"<div style='font-size:1.4rem;font-weight:800'>❌ REJECTED</div>"
                f"<div style='margin-top:8px'>{trace.error or 'No eligible scheme'}</div>"
                f"</div>", unsafe_allow_html=True,
            )
        st.markdown(f"**Candidates:** {', '.join(f'`{c}`' for c in trace.candidates)}")
        if decision and decision.rejected_schemes:
            with st.expander(f"⚠ {len(decision.rejected_schemes)} scheme(s) rejected"):
                for s, reason in decision.rejected_schemes.items():
                    st.markdown(f"- **{s}**: {reason}")

    st.divider()
    auth_col, cost_col = st.columns(2)

    with auth_col:
        st.markdown('<div class="section-header">Auth Scores</div>', unsafe_allow_html=True)
        if trace.auth_scores:
            auth_df = pd.DataFrame(trace.auth_scores)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=auth_df["scheme"], y=auth_df["p_auth"], name="p(auth)",
                marker_color=[SCHEME_COLORS.get(s, "#6b7280") for s in auth_df["scheme"]],
                text=[f"{v:.4f}" for v in auth_df["p_auth"]], textposition="outside",
            ))
            fig.add_trace(go.Bar(
                x=auth_df["scheme"], y=auth_df["confidence"], name="confidence",
                marker_color=[SCHEME_COLORS.get(s, "#6b7280") for s in auth_df["scheme"]],
                marker_opacity=0.4,
                text=[f"{v:.2f}" for v in auth_df["confidence"]], textposition="outside",
            ))
            fig.update_layout(barmode="group", height=280,
                              yaxis=dict(range=[0, 1.15], title="Score"),
                              xaxis_title="", margin=dict(t=10, b=10),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Reasoning"):
                for a in trace.auth_scores:
                    st.markdown(f"**{a['scheme']}** — {a['reasoning']}")
        else:
            st.info("No auth scores produced.")

    with cost_col:
        st.markdown('<div class="section-header">Cost Breakdown</div>', unsafe_allow_html=True)
        if trace.cost_scores:
            cost_rows = [{
                "scheme":       c["scheme"],
                "interchange":  c["breakdown"]["interchange_bps"],
                "assessment":   c["breakdown"]["assessment_bps"],
                "acquirer":     c["breakdown"]["acquirer_bps"],
                "total":        c["total_fee_bps"],
            } for c in trace.cost_scores]
            cost_df = pd.DataFrame(cost_rows)
            fig = go.Figure()
            for comp, color in [("interchange", "#6366f1"),
                                 ("assessment",  "#f59e0b"),
                                 ("acquirer",    "#10b981")]:
                fig.add_trace(go.Bar(x=cost_df["scheme"], y=cost_df[comp],
                                     name=comp.title(), marker_color=color))
            fig.update_layout(barmode="stack", height=280,
                              yaxis_title="Basis Points", xaxis_title="",
                              margin=dict(t=10, b=10),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Reasoning"):
                for c in trace.cost_scores:
                    st.markdown(f"**{c['scheme']}** — {c['reasoning']}")
        else:
            st.info("No cost scores produced.")

    if trace.fraud_scores:
        st.divider()
        st.markdown('<div class="section-header">Fraud Risk</div>', unsafe_allow_html=True)
        fraud_df = pd.DataFrame(trace.fraud_scores)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fraud_df["scheme"], y=fraud_df["p_fraud"], name="p(fraud)",
            marker_color=["#ef4444" if v > 0.10 else "#f59e0b" if v > 0.05 else "#22c55e"
                          for v in fraud_df["p_fraud"]],
            text=[f"{v:.4f}" for v in fraud_df["p_fraud"]], textposition="outside",
        ))
        fig.update_layout(height=240, yaxis=dict(range=[0, 0.4], title="p(fraud)"),
                          xaxis_title="", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Reasoning"):
            for f in trace.fraud_scores:
                st.markdown(f"**{f['scheme']}** — {f['reasoning']}")

    if trace.ranked:
        st.divider()
        has_fraud = any(r.get("p_fraud") is not None for r in trace.ranked)
        formula   = "1.0×p_auth − 0.15×norm_fee" + (" − 0.30×p_fraud" if has_fraud else "")
        st.markdown(f'<div class="section-header">Weighted Ranking ({formula})</div>',
                    unsafe_allow_html=True)
        rank_df = pd.DataFrame(trace.ranked)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=rank_df["scheme"], y=rank_df["p_auth"], name="1.0×p(auth)",
            marker_color=[SCHEME_COLORS.get(s, "#6b7280") for s in rank_df["scheme"]],
        ))
        norm_fee = (rank_df["total_fee_bps"] / 200.0).clip(upper=1.0) * 0.15
        fig.add_trace(go.Bar(
            x=rank_df["scheme"], y=-norm_fee,
            name="−0.15×norm_fee", marker_color="#f87171",
        ))
        if has_fraud and "p_fraud" in rank_df.columns:
            fig.add_trace(go.Bar(
                x=rank_df["scheme"], y=-(rank_df["p_fraud"].fillna(0) * 0.30),
                name="−0.30×p_fraud", marker_color="#f97316",
            ))
        fig.add_trace(go.Scatter(
            x=rank_df["scheme"], y=rank_df["weighted_score"],
            mode="markers+text",
            marker=dict(size=14, color="#1d4ed8", symbol="diamond"),
            text=[f"{v:.4f}" for v in rank_df["weighted_score"]],
            textposition="top center", name="Weighted Score",
        ))
        fig.update_layout(barmode="relative", height=320,
                          yaxis_title="Score", xaxis_title="",
                          margin=dict(t=20, b=20),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  TAB 3 — AUTH & RISK
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Auth & Risk Analysis")

    st.markdown('<div class="section-header">Historical Auth Rates by BIN × Scheme</div>',
                unsafe_allow_html=True)
    hist_rows = []
    for (bin_, scheme), data in AUTH_RATE_HISTORY.items():
        issuer_info = BIN_TABLE.get(bin_, ("?", bin_, "?", "?", []))
        hist_rows.append({
            "bin": bin_, "issuer": issuer_info[1], "scheme": scheme,
            "rate_30d": data["rate_30d"], "rate_90d": data["rate_90d"],
            "volume_30d": data["volume_30d"],
        })
    hist_df = pd.DataFrame(hist_rows)

    pivot = hist_df.pivot_table(index="issuer", columns="scheme", values="rate_30d")
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn", zmin=0.85, zmax=1.0,
                    text_auto=".3f", labels=dict(color="Auth Rate 30d"), height=350)
    fig.update_layout(margin=dict(t=10, b=10), coloraxis_colorbar_title="Rate")
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">30d vs 90d Auth Rate Drift</div>',
                    unsafe_allow_html=True)
        hist_df["drift"] = (hist_df["rate_30d"] - hist_df["rate_90d"]) * 100
        hist_df["label"] = hist_df["issuer"] + " / " + hist_df["scheme"]
        fig = px.bar(hist_df.sort_values("drift"),
                     x="drift", y="label", orientation="h",
                     color="drift", color_continuous_scale="RdYlGn",
                     labels={"drift": "Drift (pp)", "label": ""}, height=350)
        fig.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False, yaxis_title="")
        fig.add_vline(x=0, line_color="gray", line_dash="dot")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Volume vs Auth Rate</div>',
                    unsafe_allow_html=True)
        fig = px.scatter(hist_df, x="volume_30d", y="rate_30d",
                         color="scheme", size="volume_30d",
                         hover_data=["issuer", "rate_90d"],
                         color_discrete_map=SCHEME_COLORS,
                         labels={"volume_30d": "Volume (30d)", "rate_30d": "Auth Rate 30d"},
                         height=350)
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-header">Issuer Health Status</div>',
                unsafe_allow_html=True)
    health_rows = []
    for issuer_id, info in ISSUER_HEALTH.items():
        name = next((v[1] for v in BIN_TABLE.values() if v[0] == issuer_id), issuer_id)
        icon = {"healthy": "🟢", "elevated_declines": "🔴"}.get(info["status"], "🟡")
        health_rows.append({
            "Issuer": name,
            "Status": icon + "  " + info["status"],
            "Incident": info["incident"] or "—",
        })
    st.dataframe(pd.DataFrame(health_rows), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-header">Fraud Risk — All Transactions</div>',
                unsafe_allow_html=True)
    all_fraud = []
    for r in results:
        txn = r["txn"]
        for f in r["trace"].fraud_scores:
            all_fraud.append({
                "txn_id": txn.txn_id, "scheme": f["scheme"],
                "p_fraud": f["p_fraud"], "confidence": f["confidence"],
                "channel": txn.channel, "region": txn.region,
            })
    if all_fraud:
        fraud_all_df = pd.DataFrame(all_fraud)
        fig = px.scatter(
            fraud_all_df, x="txn_id", y="p_fraud",
            color="scheme", symbol="channel",
            color_discrete_map=SCHEME_COLORS,
            labels={"p_fraud": "p(fraud)", "txn_id": "Transaction"}, height=320,
        )
        fig.add_hline(y=0.10, line_dash="dot", line_color="#ef4444",
                      annotation_text="risk threshold 0.10")
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            fraud_all_df, hide_index=True, use_container_width=True,
            column_config={
                "p_fraud": st.column_config.ProgressColumn(
                    "p(fraud)", min_value=0.0, max_value=0.5, format="%.4f"),
            },
        )

    st.divider()
    st.markdown('<div class="section-header">Decline Patterns (Off-hours vs Business)</div>',
                unsafe_allow_html=True)
    dp_rows = []
    for (bin_, bucket), data in DECLINE_PATTERNS.items():
        issuer_info = BIN_TABLE.get(bin_, ("?", bin_, "?", "?", []))
        for code, rate in data.items():
            dp_rows.append({"bin": bin_, "issuer": issuer_info[1],
                            "bucket": bucket, "code": code, "rate": rate})
    if dp_rows:
        dp_df = pd.DataFrame(dp_rows)
        fig = px.bar(dp_df, x="issuer", y="rate", color="code",
                     facet_col="bucket", barmode="stack",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     labels={"rate": "Decline Share", "code": "Decline Code"}, height=350)
        fig.update_layout(margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  TAB 4 — COST ANALYSIS
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Cost Analysis")

    st.markdown('<div class="section-header">Interchange Rate Table (bps)</div>',
                unsafe_allow_html=True)
    ic_rows = []
    for (scheme, region, card_type, tier), bps in INTERCHANGE_TABLE.items():
        ic_rows.append({"scheme": scheme, "region": region,
                        "card_type": card_type, "tier": tier, "bps": bps})
    ic_df = pd.DataFrame(ic_rows)
    ic_df["key"] = ic_df["region"] + " / " + ic_df["card_type"] + " / " + ic_df["tier"]
    ic_pivot = ic_df.pivot_table(index="key", columns="scheme", values="bps")
    fig = px.imshow(ic_pivot, text_auto=".1f", color_continuous_scale="Blues",
                    labels=dict(color="bps"), height=420)
    fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown('<div class="section-header">Scheme Fees by Region</div>',
                    unsafe_allow_html=True)
        fee_rows = []
        for (scheme, region), fees in SCHEME_FEES.items():
            fee_rows.append({
                "scheme": scheme, "region": region,
                "assessment": fees["assessment_bps"],
                "acquirer":   fees["acquirer_bps"],
            })
        fee_df = pd.DataFrame(fee_rows)
        fee_melt = fee_df.melt(id_vars=["scheme", "region"],
                                value_vars=["assessment", "acquirer"],
                                var_name="component", value_name="bps")
        fig = px.bar(fee_melt, x="scheme", y="bps", color="component",
                     facet_col="region", barmode="stack",
                     color_discrete_sequence=["#f59e0b", "#10b981"],
                     labels={"bps": "Basis Points"}, height=320)
        fig.update_layout(margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_c2:
        st.markdown('<div class="section-header">Pipeline Fee vs Transaction Amount</div>',
                    unsafe_allow_html=True)
        cost_pipeline = []
        for r in results:
            txn = r["txn"]
            for c in r["trace"].cost_scores:
                cost_pipeline.append({
                    "txn_id": txn.txn_id, "scheme": c["scheme"],
                    "amount": txn.amount_minor / 100,
                    "fee_bps": c["total_fee_bps"],
                    "fee_abs": round(txn.amount_minor / 100 * c["total_fee_bps"] / 10000, 4),
                    "region": txn.region,
                })
        if cost_pipeline:
            cp_df = pd.DataFrame(cost_pipeline)
            fig = px.scatter(cp_df, x="amount", y="fee_bps",
                             color="scheme", symbol="region",
                             hover_data=["txn_id", "fee_abs"],
                             color_discrete_map=SCHEME_COLORS,
                             labels={"amount": "Amount", "fee_bps": "Total Fee (bps)"},
                             height=320)
            fig.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-header">Full Cost Breakdown — All Transactions</div>',
                unsafe_allow_html=True)
    all_cost = []
    for r in results:
        txn = r["txn"]
        merchant = MERCHANT_CONTRACTS.get(txn.merchant_id, {}).get("name", txn.merchant_id)
        for c in r["trace"].cost_scores:
            all_cost.append({
                "txn_id":      txn.txn_id,
                "merchant":    merchant,
                "scheme":      c["scheme"],
                "interchange": round(c["breakdown"]["interchange_bps"], 2),
                "assessment":  round(c["breakdown"]["assessment_bps"], 2),
                "acquirer":    round(c["breakdown"]["acquirer_bps"], 2),
                "total_bps":   round(c["total_fee_bps"], 2),
                "fee (€)":     round(c["breakdown"]["fee_minor"] / 100, 4),
            })
    if all_cost:
        st.dataframe(pd.DataFrame(all_cost), hide_index=True, use_container_width=True,
                     column_config={
                         "total_bps": st.column_config.ProgressColumn(
                             "Total (bps)", min_value=0, max_value=300, format="%.1f"),
                     })


# ════════════════════════════════════════════════════════════════════
#  TAB 5 — COMPLIANCE
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Compliance Gate")

    with st.expander("Compliance rules"):
        st.markdown("""
| Rule | Description |
|------|-------------|
| **merchant** | Merchant contract must include the scheme |
| **token_lock** | Network tokens are locked to the issuing network |
| **optblue** | Amex requires explicit OptBlue merchant enrollment |
| **ifr** | EU IFR interchange cap (0.20% debit / 0.30% credit, both sides EEA) |
| **durbin** | US regulated debit interchange ceiling (≤ 95 bps) |
        """)

    st.markdown('<div class="section-header">Rule Pass / Fail Matrix</div>',
                unsafe_allow_html=True)
    matrix_rows = []
    for r in results:
        txn, trace, decision = r["txn"], r["trace"], r["decision"]
        for entry in trace.ranked:
            scheme = entry["scheme"]
            checks = {
                "merchant":   merchant_eligible(txn, scheme)[0],
                "token_lock": passes_token_lock(txn, scheme)[0],
                "optblue":    passes_optblue(txn, scheme)[0],
                "ifr":        passes_ifr_with_breakdown(txn, scheme, entry["interchange_bps"])[0],
                "durbin":     passes_durbin(txn, scheme, entry["interchange_bps"])[0],
            }
            chosen = decision and decision.scheme == scheme
            matrix_rows.append({
                "txn_id": txn.txn_id, "scheme": scheme,
                "chosen": "✅ chosen" if chosen else "",
                **{k: "✅" if v else "❌" for k, v in checks.items()},
            })
    if matrix_rows:
        st.dataframe(pd.DataFrame(matrix_rows), hide_index=True, use_container_width=True)

    # Rejection summary + rule frequency
    st.divider()
    st.markdown('<div class="section-header">Rejection Details</div>',
                unsafe_allow_html=True)
    rej_rows = []
    for r in results:
        decision, txn = r["decision"], r["txn"]
        if decision and decision.rejected_schemes:
            for scheme, reason in decision.rejected_schemes.items():
                rej_rows.append({"txn_id": txn.txn_id, "scheme": scheme, "reason": reason})
        elif not decision and r["trace"].error:
            rej_rows.append({"txn_id": txn.txn_id, "scheme": "ALL", "reason": r["trace"].error})

    if rej_rows:
        st.dataframe(pd.DataFrame(rej_rows), hide_index=True, use_container_width=True)
        rule_hits = {rule: 0 for rule in RULES}
        for row in rej_rows:
            for rule in RULES:
                if rule in row["reason"]:
                    rule_hits[rule] += 1
        if any(rule_hits.values()):
            fig = px.bar(x=list(rule_hits.keys()), y=list(rule_hits.values()),
                         labels={"x": "Rule", "y": "Rejection Count"},
                         color=list(rule_hits.keys()),
                         color_discrete_sequence=px.colors.qualitative.Set1, height=280)
            fig.update_layout(margin=dict(t=20, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No rejections across all transactions.")

    # Eligibility funnel
    st.divider()
    st.markdown('<div class="section-header">Eligibility Funnel per Transaction</div>',
                unsafe_allow_html=True)
    funnel_rows = [{
        "txn_id":     r["txn"].txn_id,
        "candidates": len(r["trace"].candidates),
        "scored":     len(r["trace"].ranked),
        "decided":    1 if r["decision"] else 0,
    } for r in results]
    funnel_df = pd.DataFrame(funnel_rows)
    fig = go.Figure()
    for col, color, label in [
        ("candidates", "#6366f1", "Candidates"),
        ("scored",     "#f59e0b", "Scored"),
        ("decided",    "#22c55e", "Decided"),
    ]:
        fig.add_trace(go.Bar(x=funnel_df["txn_id"], y=funnel_df[col],
                             name=label, marker_color=color))
    fig.update_layout(barmode="group", height=280, yaxis_title="Count",
                      margin=dict(t=10, b=10), yaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)



# ════════════════════════════════════════════════════════════════════
#  TAB 6 — AGENT INSIGHTS
# ════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Agent Insights")

    # Planner decisions
    st.markdown('<div class="section-header">LLM Planner Decisions</div>',
                unsafe_allow_html=True)
    plan_rows = [{
        "txn_id":    r["txn"].txn_id,
        "card_caps": ", ".join(r["txn"].card_brand_capabilities),
        "selected":  ", ".join(r["trace"].candidates),
        "reasoning": r["trace"].planner_reasoning or "—",
    } for r in results]
    st.dataframe(pd.DataFrame(plan_rows), hide_index=True, use_container_width=True)

    st.divider()

    # Reflection
    st.markdown('<div class="section-header">Reflection Outputs</div>',
                unsafe_allow_html=True)
    for r in results:
        refl   = r["trace"].reflection or "—"
        amount = f"{r['txn'].amount_minor/100:.2f} {r['txn'].currency}"
        flag   = refl not in ("No anomalies detected.", "—")
        icon   = "🔴" if flag else "🟢"
        with st.expander(f"{icon} {r['txn'].txn_id}  ·  {amount}"):
            st.write(refl)

    st.divider()

    # Explanation
    st.markdown('<div class="section-header">Explanation Agent Outputs</div>',
                unsafe_allow_html=True)
    for r in results:
        expl = r["trace"].explanation
        dec  = r["decision"]
        if expl:
            badge = f"**{dec.scheme.upper()}**" if dec else "**REJECTED**"
            with st.expander(f"{r['txn'].txn_id} → {badge}"):
                st.write(expl)

    st.divider()

    # HITL gate summary
    st.markdown('<div class="section-header">Human-in-the-Loop Gate</div>',
                unsafe_allow_html=True)
    hitl_rows = [{
        "txn_id":    r["txn"].txn_id,
        "amount":    f"{r['txn'].amount_minor/100:.2f} {r['txn'].currency}",
        "triggered": r["txn"].amount_minor >= HIGH_VALUE_THRESHOLD,
        "outcome":   r["decision"].scheme.upper() if r["decision"] else "REJECTED",
    } for r in results]
    hitl_df = pd.DataFrame(hitl_rows)
    st.dataframe(hitl_df, hide_index=True, use_container_width=True,
                 column_config={"triggered": st.column_config.CheckboxColumn("HITL triggered")})
    hitl_count = hitl_df["triggered"].sum()
    if hitl_count:
        st.info(
            f"{hitl_count} transaction(s) exceeded the ${HIGH_VALUE_THRESHOLD/100:.0f} threshold. "
            "Use **Live run** in the Deep Dive tab to experience the approval flow."
        )

    st.divider()

    # Guardrail warnings
    st.markdown('<div class="section-header">Guardrail Warnings</div>',
                unsafe_allow_html=True)
    gw_rows = []
    for r in results:
        for w in r["trace"].guardrail_warnings:
            gw_rows.append({
                "txn_id": r["txn"].txn_id,
                "field":  w.get("field", "?"),
                "reason": w.get("reason", str(w)),
            })
    if gw_rows:
        st.dataframe(pd.DataFrame(gw_rows), hide_index=True, use_container_width=True)
    else:
        st.success("No guardrail warnings across all transactions.")
