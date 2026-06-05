"""01_baseline_xgboost.py — End-to-end baseline run.

Pipeline:
  1. Load PaySim
  2. Engineer 19 features + apply 7 rules
  3. Train XGBoost + Random Forest + Isolation Forest
  4. Evaluate and save metrics
  5. Render ROC + feature importance plots

Run with:
    python notebooks/01_baseline_xgboost.py
"""
from __future__ import annotations

import logging
from pathlib import Path

# Add repo root to path so we can import src
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_paysim
from src.features import build_features, label_accounts
from src.rules import apply_rules
from src.train import (
    train_xgboost, train_random_forest, train_isolation_forest,
    evaluate_classifier, save_models, save_metrics, save_final_summary,
    split_features,
)
from src.evaluate import run as run_evaluate
from src.network import build_graph, detect_rings, render_ring_html

LOG = logging.getLogger("muleshield.baseline")


def main(sample_n: int = 50_000) -> None:
    import time
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    t_total = time.time()

    LOG.info("Step 1: Loading PaySim sample (%d rows) ...", sample_n)
    df = load_paysim("data", sample_n=sample_n)

    LOG.info("Step 2: Building 19 features + 7 rules ...")
    feats = build_features(df)
    labels = label_accounts(df, feats)
    rule_flags = apply_rules(df)
    feats_full = feats.join(rule_flags, how="left").fillna(0)
    LOG.info("Feature matrix: %s", feats_full.shape)

    LOG.info("Step 3: Train/test split ...")
    X_train, X_test, y_train, y_test = split_features(feats_full, labels)

    LOG.info("Step 4: Training models ...")
    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(int((y_train == 0).sum()), 1)
    spw = n_neg / n_pos
    xgb = train_xgboost(X_train, y_train, scale_pos_weight=spw)
    rf = train_random_forest(X_train, y_train)
    iso = train_isolation_forest(X_train)

    LOG.info("Step 5: Evaluating ...")
    metrics = []
    metrics.append(evaluate_classifier(xgb, X_test, y_test, "xgboost"))
    metrics.append(evaluate_classifier(rf, X_test, y_test, "random_forest"))

    # IsolationForest special-case evaluation
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    y_iso = np.where(iso.predict(X_test) == -1, 1, 0)
    iso_m = {
        "model": "isolation_forest",
        "precision": float(precision_score(y_test, y_iso, zero_division=0)),
        "recall": float(recall_score(y_test, y_iso, zero_division=0)),
        "f1": float(f1_score(y_test, y_iso, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_iso).tolist(),
    }
    LOG.info("[isolation_forest] precision=%.3f recall=%.3f f1=%.3f",
             iso_m["precision"], iso_m["recall"], iso_m["f1"])
    metrics.append(iso_m)

    save_models({"xgboost": xgb, "random_forest": rf, "isolation_forest": iso},
                Path("models"))
    save_metrics(metrics, Path("reports/metrics.json"))
    save_final_summary(
        metrics, xgb, Path("reports/final_metrics.json"),
        df=df, feats=feats_full, labels=labels,
        runtime_sec=round(time.time() - t_total, 1),
    )

    LOG.info("Step 6: Plotting figures ...")
    run_evaluate(models_dir=Path("models"), data_dir=Path("data"),
                 reports_dir=Path("reports"), sample_n=sample_n)

    LOG.info("Step 7: Detecting mule rings ...")
    G = build_graph(df)
    rings = detect_rings(G, top_n=10)
    render_ring_html(rings, Path("reports/rings.html"))

    LOG.info("Total runtime: %.1f seconds. See reports/ for outputs.",
             time.time() - t_total)


if __name__ == "__main__":
    main()
