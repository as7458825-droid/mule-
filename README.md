# MuleShield-AI

> AI-Powered Hybrid Mule Account & Suspicious Transaction Detection System

Built for **Bank of India's PSB Cybersecurity, Fraud & AI Hackathon 2026** (in collaboration with IIT Hyderabad). Powered by Department of Financial Services (DFS) and Indian Banks' Association (IBA).

## Problem Statement (PS2)

Develop an AI/ML solution that ingests financial transactions, fraud monitoring alerts, and government cyber-fraud tickets to detect suspicious transactions and mule accounts, and prevents the circulation of fraudulent proceeds through mule accounts. The solution must consume real-time regulatory inputs and cross-channel bank data.

## Team — Rajkiye Engineering College, Kannauj (B.Tech)

| # | Name | Role |
|---|---|---|
| 1 | Ayush Kumar | Team Lead — ML & GenAI |
| 2 | Abhinaw Sikharwar | Data Engineer |
| 3 | Vinay Kushwaha | Dashboard Developer |
| 4 | Anubhav Upadhyay | ML Support — Graph |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the end-to-end baseline (synthetic data fallback included)
python notebooks/01_baseline_xgboost.py

# 3. Launch the dashboard
streamlit run app/streamlit_dashboard.py
```

The pipeline auto-generates a synthetic PaySim-style dataset if the real PaySim CSV is not present in `data/raw/`.

## Repository Layout

```
muleshield-ai/
├── src/
│   ├── data_loader.py    # PaySim loader + CHANNEL_MAP + synthetic fallback
│   ├── features.py       # 19-feature engineering profile
│   ├── rules.py          # 7-rule expert engine (R1-R7)
│   ├── train.py          # XGBoost + RF + IsolationForest
│   ├── evaluate.py       # ROC, confusion matrix, feature importance plots
│   ├── network.py        # NetworkX graph + Louvain ring detection
│   └── explain.py        # GenAI prompt builder (Claude / GPT / Gemini)
├── app/
│   └── streamlit_dashboard.py   # 5-tab interactive dashboard
├── notebooks/
│   └── 01_baseline_xgboost.py   # End-to-end runnable script
├── reports/
│   ├── figures/                  # CM, ROC, feature importance PNGs
│   ├── final_metrics.json        # Reproducible model metrics
│   └── metrics.json
├── docs/
│   └── phase1_research.md        # PSB hackathon winning-pattern research
├── data/
│   └── README.md                 # PaySim download instructions
├── requirements.txt
└── README.md
```

## Key Results (Synthetic PaySim-style Data — 200K rows)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **XGBoost (primary)** | 0.406 | **0.838** | 0.547 | **0.953** | **0.744** |
| Random Forest | 0.407 | 0.833 | 0.547 | 0.954 | 0.696 |
| Isolation Forest (anomaly) | 0.697 | 0.180 | 0.287 | — | — |

**Top 5 features by XGBoost gain:** F198_night_ratio (0.60), F614_in_out_ratio (0.20), F321_beneficiary_diversity, F_total_amount, F431b_max_single_txn.

See `reports/final_metrics.json` and `reports/figures/` for full reproducible artefacts.

## 5-Layer Architecture

1. **L1 — Data Ingestion:** PaySim / hackathon CSV / CFCFRMS / Sanchar Saathi feeds
2. **L2 — Rule Engine:** 7 expert rules (R1 dormancy, R2 smurfing, R3 odd-hour, R4 new beneficiary, R5 new-account, R6 rapid movement, R7 cross-channel)
3. **L3 — ML Models:** XGBoost (primary) + Random Forest (ensemble) + Isolation Forest (anomaly)
4. **L4 — Network Intelligence:** NetworkX graph + Louvain community detection surfaces mule rings
5. **L5 — GenAI Explanation:** Plain-English alert reasons via Claude / GPT / Gemini
6. **L6 — Streamlit Dashboard:** 5-tab UI (Overview, Rules, Lookup, Network, About)

## Innovation Highlights

- **Hybrid rule + ML design** — every flag is interpretable
- **Complicit vs Witting** mule classification (fairness-first)
- **5-step Prevention Simulator** (freeze → STR → audit)
- **Cross-channel coverage** — UPI, NEFT, IMPS, RTGS, ATM
- **Reproducible POC** — single command runs the whole pipeline

## Documentation

- `docs/phase1_research.md` — PSB hackathon history, evaluation rubric, winning patterns
- Inline docstrings in every `src/*.py` module
- Solution PDF: see the hackathon portal submission

## License

MIT — for educational use as part of PSB Hackathon 2026.

## Acknowledgements

- Bank of India & IIT Hyderabad (FinShield 2026 organisers)
- Department of Financial Services (DFS), Ministry of Finance
- Indian Banks' Association (IBA)
