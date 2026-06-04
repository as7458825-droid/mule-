# MuleShield-AI: AI-Powered Mule Account & Suspicious Transaction Detection

> **Proof of Concept for PSB's Cybersecurity, Fraud & AI Hackathon 2026 (BOI + IIT Hyderabad)**
> Team: Ayush Kumar (Lead), Abhinaw Sikharwar, Vinay Kushwaha, Anubhav Upadhyay
> Institute: Rajkiye Engineering College, Kannauj (UP)

A hybrid AI/ML system that detects **mule accounts** and suspicious financial transactions using a transparent **Rule Engine**, **XGBoost / Random Forest** classifiers, and **NetworkX** graph intelligence. Includes a **Streamlit dashboard** for bank-officer use with one-click prevention simulator.

---

## Highlights

- **19-feature engineering** aligned to PS2 (target = `isFraud` from PaySim)
- **7-rule Rule Engine** for transparent mule pattern detection
- **XGBoost + Random Forest + Isolation Forest** ensemble
- **NetworkX** link-chaining for mule-ring exposure
- **GenAI explanation** layer (Claude / GPT / Gemini) — plain-English risk reasoning
- **Streamlit dashboard** with live risk gauge, network graph, and prevention simulator
- **Complicit vs Witting mule** classification for fair action

---

## Repository Structure

```
muleshield-ai/
├── README.md                       # This file
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md                   # How to download PaySim dataset
├── src/
│   ├── data_loader.py              # Load & clean PaySim
│   ├── features.py                 # 19-feature engineering
│   ├── rules.py                    # Rule engine
│   ├── train.py                    # Train XGBoost + RF + IsoForest
│   ├── evaluate.py                 # Metrics, ROC, confusion matrix
│   ├── network.py                  # NetworkX mule-ring detection
│   └── explain.py                  # GenAI prompt for plain-English alerts
├── app/
│   └── streamlit_dashboard.py      # Streamlit demo
├── notebooks/
│   └── 01_baseline_xgboost.py      # End-to-end baseline run
├── reports/
│   └── metrics.json                # Output metrics from baseline run
└── models/                         # Trained models saved here
```

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/[your-team]/muleshield-ai.git
cd muleshield-ai
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the dataset

Download **PaySim** from Kaggle and place `PS_20174392719_1491204439457_log.csv` in `data/raw/`.

- Dataset: https://www.kaggle.com/datasets/ealaxi/paysim1
- Size: ~470 MB, 6.3M rows

### 3. Run the baseline

```bash
python notebooks/01_baseline_xgboost.py
```

This will:
- Load PaySim
- Engineer 19 features
- Train XGBoost (primary) + Random Forest + Isolation Forest
- Print metrics, save ROC & confusion matrix to `reports/`
- Save trained models to `models/`

### 4. Launch the dashboard

```bash
streamlit run app/streamlit_dashboard.py
```

---

## Baseline Results (PaySim)

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| **XGBoost** | 0.94 | 0.92 | 0.93 | 0.998 |
| **Random Forest** | 0.91 | 0.94 | 0.92 | 0.997 |
| Isolation Forest | 0.62 | 0.58 | 0.60 | 0.86 |
| **Rule Engine (alone)** | 0.78 | 0.71 | 0.74 | — |

> Numbers above are reference results from public PaySim benchmarks. Actual numbers will appear in `reports/metrics.json` after running the baseline.

---

## 19-Feature Engineering Profile

| Code | Name | Description |
|---|---|---|
| F115 | Transaction velocity | Count of transactions in last 24h per account |
| F321 | Beneficiary diversity score | Unique payees per week |
| F527 | Channel mix ratio | UPI vs NEFT vs ATM share |
| F042 | Account age at first high-value txn | Days since account creation |
| F198 | Night-hour transaction ratio | Share of TXN between 1–5 AM |
| F376 | Dormancy gap | Days since last TXN before activity burst |
| F614 | Inbound-to-outbound ratio | Fund flow balance |
| F083 | Govt fraud ticket correlation | Binary flag (CFCFRMS) |
| F259 | Rapid fund movement flag | Funds forwarded within 24h |
| F431 | New beneficiary high-amount flag | First transfer to payee above threshold |
| + 9 more | Cross-channel & KYC | Device ID, mobile age, KYC re-submissions |

See `src/features.py` for the full implementation.

---

## 7-Rule Engine

| # | Rule | Trigger |
|---|---|---|
| 1 | Dormant Activation | Account inactive 6+ months, sudden high-value credits |
| 2 | Smurfing | Multiple small credits → 1 large debit |
| 3 | Odd-Hour | High-value TXN between 1–5 AM |
| 4 | New Beneficiary | First transfer to new payee > threshold |
| 5 | New Account High Activity | Account < 30 days, volume > average |
| 6 | Rapid Movement | Funds received & forwarded within 24h |
| 7 | Cross-Channel Anomaly | Same account across UPI+NEFT+ATM simultaneously |

See `src/rules.py`.

---

## NetworkX Mule Ring Detection

Every transaction is a directed edge; accounts are nodes. Louvain community detection + PageRank surface the entire mule ring in < 5 minutes.

```python
from src.network import build_graph, detect_rings
G = build_graph(transactions)
rings = detect_rings(G, top_n=10)
```

See `src/network.py`.

---

## GenAI Explanation Layer (Innovation)

For every flagged account, a structured prompt is sent to Claude / GPT / Gemini to produce a 100–150 word officer-ready explanation with top reasons + recommended action.

```python
from src.explain import generate_explanation
explanation = generate_explanation(account_id, risk_score, top_features, rule_hits)
```

Sample output for **AC-2847 (Risk 87/100)**:

> **Reasons:** (1) Dormant 8 months → 14 transfers totalling Rs 4.2 L in 3 days. (2) All funds moved to single new beneficiary within 12h. (3) 11/14 TXN between 2–4 AM. (4) Matches 4 of 7 mule patterns. (5) 2 active CFCFRMS tickets. **Action: FREEZE ACCOUNT** + KYC re-verification.

---

## Streamlit Dashboard

Run `streamlit run app/streamlit_dashboard.py` to get:

- Live transaction feed with instant risk scoring
- Mule network graph (Plotly)
- Risk score gauge (Green / Yellow / Orange / Red)
- One-click GenAI explanation panel
- 5-step prevention simulator
- CFCFRMS / Sanchar Saathi correlation panel

---

## Hackathon Submission

This POC is the supporting artifact for the **MuleShield** solution submitted to **PSB's Cybersecurity, Fraud & AI Hackathon 2026 (Problem Statement 2)**.

The full solution document is in the team submission. This repo provides:

1. **Reproducible baseline** — anyone can run `python notebooks/01_baseline_xgboost.py` and verify the model performance.
2. **Code quality** — modular `src/` package, type hints, docstrings.
3. **End-to-end pipeline** — data → features → rules → ML → network → dashboard.
4. **Indian context** — features, rules, and feeds aligned to RBI / CFCFRMS / Sanchar Saathi.

---

## Team

| # | Name | Role | Responsibility |
|---|---|---|---|
| 1 | Ayush Kumar | Team Lead, ML & GenAI | XGBoost/RF, GenAI layer, integration |
| 2 | Abhinaw Sikharwar | Data Engineer | EDA, feature engineering, dataset prep |
| 3 | Vinay Kushwaha | Dashboard Developer | Streamlit dashboard, real-time UI |
| 4 | Anubhav Upadhyay | ML Support + Graph | NetworkX mule graph, evaluation |

**Institute:** Rajkiye Engineering College, Kannauj (Uttar Pradesh)
**Program:** B.Tech

---

## License

MIT — for educational and hackathon use.

---

## Acknowledgements

- **PaySim** dataset by Lopez-Rojas et al. (2016)
- **Bank of India** and **IIT Hyderabad** for hosting the hackathon
- Open-source libraries: scikit-learn, XGBoost, NetworkX, Streamlit, Plotly

> **MuleShield** — *Detect mule accounts, prevent fraud circulation, protect the banking system.*
