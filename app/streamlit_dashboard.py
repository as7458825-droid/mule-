"""Streamlit dashboard for MuleShield-AI.

Launch with:
    streamlit run app/streamlit_dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make src importable when running from app/
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.explain import SAMPLE_ACCOUNT_ID, SAMPLE_FEATURES, SAMPLE_RISK_SCORE, SAMPLE_RULES, generate_explanation
from src.features import build_features, label_accounts
from src.data_loader import load_paysim
from src.network import build_graph, detect_rings
from src.rules import apply_rules

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MuleShield-AI Dashboard",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0B3D91"
RED = "#C00000"
GREEN = "#70AD47"
ORANGE = "#ED7D31"
YELLOW = "#FFC000"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("MuleShield-AI")
st.sidebar.caption("AI-Powered Mule Account & Suspicious Transaction Detection")
st.sidebar.markdown("**Team:** Ayush Kumar (Lead), Abhinav, Vinay, Anubhav")
st.sidebar.markdown("**Institute:** Rajkiya Engineering College, Kannauj")
st.sidebar.markdown("**Hackathon:** PSB Cybersecurity, Fraud & AI — 2026")

data_dir = st.sidebar.text_input("Data directory", value="data")
sample_n = st.sidebar.slider("Sample size (rows)", min_value=10_000, max_value=500_000,
                              value=100_000, step=10_000)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading PaySim sample ...")
def load_data(data_dir: str, sample_n: int) -> pd.DataFrame:
    return load_paysim(data_dir, sample_n=sample_n)

@st.cache_data(show_spinner="Engineering 19 features ...")
def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    feats = build_features(df)
    labels = label_accounts(df, feats)
    rule_flags = apply_rules(df)
    return feats, labels, rule_flags

try:
    df = load_data(data_dir, sample_n)
    feats, labels, rule_flags = make_features(df)
    st.sidebar.success(f"Loaded {len(df):,} rows, {int(labels.sum())} mule accounts")
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.info("Place the PaySim CSV in `data/raw/` (see data/README.md).")
    st.stop()

# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Accounts", f"{len(feats):,}")
col2.metric("Mule Accounts", f"{int(labels.sum()):,}")
col3.metric("Fraud Ratio", f"{100*labels.mean():.3f}%")
col4.metric("Avg Rule Hits", f"{rule_flags['rule_hit_count'].mean():.2f}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_overview, tab_rules, tab_account, tab_network, tab_about = st.tabs(
    ["Overview", "Rule Engine", "Account Lookup", "Mule Network", "About"]
)

# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Risk Score Distribution (synthetic)")

    # Mock risk scores derived from rule hits + features (for demo)
    rng = np.random.default_rng(42)
    risk_scores = (
        rule_flags["rule_hit_count"] * 15
        + feats["F198_night_ratio"] * 20
        + feats["F259_rapid_movement"] * 25
        + rng.normal(0, 5, len(feats))
    ).clip(0, 100)

    bins = [0, 25, 50, 75, 100]
    labels_g = ["LOW (0-25)", "MEDIUM (26-50)", "HIGH (51-75)", "CRITICAL (76-100)"]
    risk_bucket = pd.cut(risk_scores, bins=bins, labels=labels_g, include_lowest=True)
    counts = risk_bucket.value_counts().reindex(labels_g, fill_value=0)

    fig = go.Figure(data=[
        go.Bar(x=counts.index, y=counts.values,
               marker_color=[GREEN, YELLOW, ORANGE, RED],
               text=counts.values, textposition="auto")
    ])
    fig.update_layout(
        title="Accounts by Risk Band",
        yaxis_title="Number of accounts",
        xaxis_title="Risk band",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("MuleShield Sample Output — Account AC-2847")

    explanation = generate_explanation(
        SAMPLE_ACCOUNT_ID, SAMPLE_RISK_SCORE, SAMPLE_FEATURES, SAMPLE_RULES, gov_ticket_match=True
    )
    st.code(explanation, language="markdown")

# ---------------------------------------------------------------------------
# Tab 2: Rule Engine
# ---------------------------------------------------------------------------
with tab_rules:
    st.subheader("7-Rule Mule Detection Engine")
    st.caption("Each rule fires per account; rule_hit_count is used as a feature for the ML model.")

    rules_summary = pd.DataFrame({
        "Rule": [
            "R1_dormant_activation", "R2_smurfing", "R3_odd_hour", "R4_new_beneficiary",
            "R5_new_account_high_activity", "R6_rapid_movement", "R7_cross_channel_anomaly",
        ],
        "Trigger": [
            "Dormant 6+ months, sudden high-value credits",
            "Many small credits → 1 large debit",
            "High-value TXN between 1 AM and 5 AM",
            "First transfer to new payee > threshold",
            "Account < 30 days, volume > average",
            "Funds received and forwarded within 24h",
            "Activity across 3+ channels (UPI/NEFT/ATM)",
        ],
        "Hits": [int(rule_flags[c].sum()) for c in [
            "R1_dormant_activation", "R2_smurfing", "R3_odd_hour", "R4_new_beneficiary",
            "R5_new_account_high_activity", "R6_rapid_movement", "R7_cross_channel",
        ]],
    })
    st.dataframe(rules_summary, use_container_width=True, hide_index=True)

    st.bar_chart(rule_flags["rule_hit_count"].value_counts().sort_index())

# ---------------------------------------------------------------------------
# Tab 3: Account Lookup
# ---------------------------------------------------------------------------
with tab_account:
    st.subheader("Single-Account Risk Lookup")
    account = st.selectbox("Choose an account", feats.index.tolist()[:500])
    if account:
        c1, c2, c3 = st.columns(3)
        c1.metric("Account", account)
        c2.metric("Total TXN", int(feats.loc[account, "F_total_txn"]))
        c3.metric("Total Amount", f"{feats.loc[account, 'F_total_amount']:,.0f}")
        st.dataframe(feats.loc[[account]].T, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Mule Network
# ---------------------------------------------------------------------------
with tab_network:
    st.subheader("Mule Ring Detection — NetworkX")
    st.caption("Louvain community detection on the transaction graph surfaces tightly connected mule rings.")

    if st.button("Detect rings (may take ~10s)"):
        with st.spinner("Building graph + detecting communities ..."):
            G = build_graph(df)
            rings = detect_rings(G, top_n=10)
        st.session_state["rings"] = rings

    if "rings" in st.session_state:
        rings = st.session_state["rings"]
        st.success(f"Detected {len(rings)} rings (>= 3 accounts)")
        ring_df = pd.DataFrame([{
            "Community": r["community_id"],
            "Accounts": r["n_accounts"],
            "Edges": r["n_edges"],
            "Total Flow": f"{r['total_flow']:,.0f}",
            "Top Hub": r["top_hub"],
        } for r in rings])
        st.dataframe(ring_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 5: About
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("""
    ## MuleShield-AI
    **AI-Powered Hybrid Mule Account & Suspicious Transaction Detection System**

    Built for **PSB's Cybersecurity, Fraud & AI Hackathon 2026** (BOI + IIT Hyderabad).

    ### Architecture
    - **L1 Data Ingestion** — PaySim (or hackathon CSV) → cleaned transactions
    - **L2 Rule Engine** — 7 expert rules for transparent mule pattern detection
    - **L3 ML Models** — XGBoost (primary) + Random Forest + Isolation Forest
    - **L4 GenAI Explanation** — plain-English alerts via Claude / GPT / Gemini
    - **L5 Streamlit Dashboard** — this app

    ### Innovation
    - **Complicit vs Witting** mule classification (fairness-first)
    - **5-step Prevention Simulator** (freeze → STR → audit log)
    - **NetworkX link-chaining** to expose entire mule rings
    - **19-feature engineering** aligned to Indian banking context (UPI, NEFT, IMPS, ATM)

    ### Team
    | # | Name | Role |
    |---|---|---|
    | 1 | Ayush Kumar | Team Lead — ML & GenAI |
    | 2 | Abhinav Sikarwar | Data Engineer |
    | 3 | Vinay Kushwaha | Dashboard Developer |
    | 4 | Anubhav Upadhyay | ML Support — Graph |

    **Institute:** Rajkiya Engineering College, Kannauj (UP)
    **Program:** B.Tech
    """)
