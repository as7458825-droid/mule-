# PSB Hackathon Series — Phase 1 Selection Patterns
*Research compiled for MuleShield-AI submission — PSB Cybersecurity, Fraud & AI Hackathon 2026 (BOI + IIT Hyderabad)*

**Sources:** financialservices.gov.in, iith.ac.in, finshield.iith.ac.in, boihackathon.cse.iith.ac.in, passionateinmarketing.com, campuskatta.com, news.edunovations.com, pnb.bank.in, ptu.ac.in

---

## 1. The 2025 PSB Series at a Glance

DFS + IBA coordinated **12 parallel hackathons** — one per Public Sector Bank, each with a partner IIT/university:

| Bank | Partner | Final Date | Theme |
|---|---|---|---|
| Bank of Baroda | IIT Kanpur | 3 Oct 2025 | API Security |
| **Bank of India** | **IIT Hyderabad** | **3 Sep 2025** | **FinTech + Cyber** |
| Bank of Maharashtra | COEP Pune | 10 Aug 2025 | Log Analysis / SOC |
| Canara Bank | IISc Bengaluru | 31 Jul 2025 | FinTech |
| Central Bank of India | MANIT Bhopal | 25 Jul 2025 | FinTech |
| Indian Bank | IIT Madras | 7 May 2025 | FinTech |
| Indian Overseas Bank | B.S. Abdur Rahman | 9 May 2025 | FinTech |
| Punjab & Sind Bank | PTU Jalandhar | 18 Sep 2025 | Financial Literacy |
| Punjab National Bank | IIT Kanpur | 19 Jul 2025 | Cyber |
| State Bank of India | IIT Guwahati | 20 Aug 2025 | Location Tracking |
| UCO Bank | IIEST Shibpur | 17 Sep 2025 | FinTech |
| Union Bank of India | K.J. Somaiya | 18 Mar 2025 | FinTech |

**Funnel stats (BOI / FinShield 2025 — the most relevant prior edition):**
- **661 applications** received
- **72 teams** shortlisted for Phase 2 (≈11%)
- **18 teams** made it to the final round (≈2.7%)
- 3 winners + 1 special jury per problem statement

---

## 2. Phase 1: What Submissions Get Selected?

Across all 12 PSB hackathons, Phase 1 is **idea + approach only** — no code is judged. From official rules, winning summaries, and the 200_OK / ALATDS / Pathfinders case studies:

### 2.1 What the judges look for (consensus rubric)

| # | Criterion | Weight (approx.) | Evidence |
|---|---|---|---|
| 1 | **Problem clarity** — restate PS2, identify the actual gap | 10% | 661 → 72; submissions that misread the problem were filtered |
| 2 | **Solution novelty** — "unique & plagiarism strictly prohibited" | 20% | 200_OK scored well on "unphishable tokens + zk-SNARKs" — non-obvious stack |
| 3 | **Technical feasibility** — clear architecture, not hand-wavy | 20% | Jigyasa (1st, PS2-2025) used keystroke + motion = real measurable signals |
| 4 | **Data plan** — which datasets, what features, what sample size | 10% | 2026 PS2 explicitly mentions 19-feature profile on hackathon dataset |
| 5 | **AI/ML substance** — specific algorithms, metrics, baselines | 15% | 200_OK declared mTLS + zk-SNARK, Pathfinders declared dual-task ensemble |
| 6 | **Real-world banking impact** — "scalability, accuracy, ethical" | 10% | Pathfinders judged on "potential for real-world implementation" |
| 7 | **Banking regulatory awareness** — RBI / SEBI / DFS context | 5% | 200_OK won because of "phased implementation roadmap" for BOB |
| 8 | **Presentation quality** — PDF ≤ 1 MB, diagrams, structure | 10% | Hard gate — file size, format violations = auto-reject |

### 2.2 What gets you REJECTED in Phase 1

- ❌ **No diagrams** — judges read 600+ PDFs; plain text = skimmed and dropped
- ❌ **Vague methodology** — "we will use AI/ML" with no algorithm named
- ❌ **No dataset** — "we will collect data later" = red flag
- ❌ **Misses the problem statement keywords** — 2026 PS2 explicitly says *"ingest financial transactions AND fraud monitoring alerts AND govt cyber fraud tickets"*; solutions ignoring any of these 3 inputs lose marks
- ❌ **Generic dashboard claims** — "real-time monitoring with charts" without specifying what the model outputs
- ❌ **No metrics plan** — "high accuracy" with no F1/PR-AUC/recall-at-FPR target
- ❌ **No team roles** — judging 4 generic "developers" vs 4 specialized roles (ML, Graph, Dashboard, GenAI) — second wins
- ❌ **PDF > 1 MB or wrong format** — auto-reject per FinShield rules

---

## 3. Phase 1 vs Phase 2 — Difference

| | Phase 1 (May 7 – Jun 15) | Phase 2 (Jul 1 – Aug 17) |
|---|---|---|
| **What is judged** | Idea + approach PDF (≤1 MB) | Working prototype + progress reports |
| **Code?** | No | Yes — must run |
| **Deliverable** | Solution Approach PDF | Progress report (Aug 17) → final presentation (Aug 27-28, IITH) |
| **Selection** | Top 72 of 661 (~11%) | Top 18 of 72 (~25%) |
| **Win** | — | Top 3 + jury prize (₹5L/₹3L/₹2L/₹1L) |

**Key insight:** Phase 1 selects for *promise + clarity*. Phase 2 selects for *working code + banking relevance*.

---

## 4. What 2025 Winners Did Differently (PS2 — closest to our 2026 PS2)

| Team | Institute | What stood out in their Phase-1 PDF |
|---|---|---|
| **Jigyasa (1st, ₹5L)** | IIIT Kottayam | Defined "behavioural biometric" = keystroke dynamics + motion sensor. Cited prior art (CMU 2018 dataset). Showed ROC curve on synthetic data. |
| **Vajra (2nd, ₹3L)** | Amrita Vishwa Vidyapeetham | Combined passwordless auth + transaction signing. Architecture diagram in Phase-1 itself. |
| **Team Kavach (3rd, ₹2L)** | IIIT Kottayam | Biometric + encryption pipeline. Emphasised "no PII stored on server". |
| **Mnemonics (Jury prize, ₹1L)** | — | Clever UX layer (recovery flow for fraud victims). Judges loved the human-impact angle. |

**Common thread:** every winner had a **named specific technique** (keystroke dynamics, motion sensor, mTLS, zk-SNARK, CTGAN, etc.) — NOT "AI/ML" hand-waving.

---

## 5. How MuleShield-AI Maps to the Winning Pattern

| Winning pattern | MuleShield-AI equivalent |
|---|---|
| Named specific technique | XGBoost + Random Forest + Isolation Forest (named) + NetworkX Louvain rings + Claude/GPT GenAI |
| Architecture diagram in PDF | 5 figures (architecture, ring, perf, gauge, prevention) |
| Dataset + features named | 19-feature profile on hackathon dataset (feature 3924) + PaySim benchmark |
| Metrics plan | ROC AUC, F1, Precision@K, ring recall — declared upfront |
| Banking context | UPI / NEFT / IMPS / ATM channels, CFCFRMS / Sanchar Saathi ticketing |
| Real-world impact | 5-step Prevention Simulator (freeze → STR → audit) — directly deployable |
| Team specialization | 4 distinct roles: ML, Data Eng, Dashboard, ML+Graph |
| ≤1 MB PDF | 616 KB (verified) |
| Plagiarism-safe | Custom rule engine + custom feature engineering — no copy-pasted templates |

---

## 6. Strategic Advice for the 2026 PS2

**Probability of Phase-1 shortlist (estimate):**

| Submission quality | Probability |
|---|---|
| Bare-minimum 5-page PDF | ~20% |
| Solid 8-page PDF, no GitHub | ~45% |
| **Solid 12-page PDF with diagrams + GitHub POC (MuleShield today)** | **~65%** |
| Same + working Streamlit demo URL | ~75% |
| Same + 1-min demo video | ~80% |

**Why these odds are realistic:**
- 661 → 72 = ~11% base rate
- Diagrams + named models + dataset + 19-feature + GenAI + rings = covers 7 of 8 rubric items above
- BOI's 2025 PS2 winners all came from **IIT/IIIT** (IIPE, IIIT Kottayam ×2, Amrita) — REC Kannauj will stand out positively as a non-premium-tier institute
- No plagiarism risk (custom rule engine)

**Top 3 things to do in the next 7 days:**
1. **Today:** Submit PDF (already done)
2. **This week:** Push GitHub repo (covered)
3. **If time:** Deploy Streamlit to share.streamlit.io (free, 5 min)

---

## 7. Key Citations

- [BOI FinShield 2025 winners — campuskatta.com](https://campuskatta.com/grand-finale-of-finshield-hackathon-2025-concludes-at-iit-hyderabad/)
- [BOI FinShield 2025 conclave — passionateinmarketing.com](https://www.passionateinmarketing.com/dfs-secretary-awards-hackathon-winners-in-boi-finshield-conclave/)
- [BOI 2026 PSB Cyber hackathon portal](https://boihackathon.cse.iith.ac.in/hackathon2026/hackathonReg/newReg/)
- [DFS PSB Hackathon Series 2025 — government listing](https://www.financialservices.gov.in/beta/en/psb-hackathon)
- [Bank of Baroda — 200_OK winner journey](https://www.financialservices.gov.in/beta/sites/default/files/2025-10/BOB-Winning-Team-Hackathon-Journey-Summary.pdf)
- [Bank of Maharashtra — ALATDS winner summary](https://www.financialservices.gov.in/beta/sites/default/files/2025-10/Bank-of-Maharashtra.pdf)
- [SBI — Team Pathfinders winner summary](https://www.financialservices.gov.in/beta/sites/default/files/2025-10/State-Bank-of-India.pdf)
- [PNB 2026 PSB Cyber hackathon page](https://pnb.bank.in/psbs-cybersecurity-hackathon.html)
- [PTU Punjab & Sind 2025 — SAFE evaluation rubric](https://ptu.ac.in/psbs-hackathon-series-2025/)
