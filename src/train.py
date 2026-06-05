"""train.py — Train XGBoost, Random Forest, and Isolation Forest on the
engineered features. Saves trained models and metrics to disk.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

LOG = logging.getLogger("muleshield.train")


def split_features(feats: pd.DataFrame, labels: pd.Series, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        feats, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return X_train, X_test, y_train, y_test


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float = 50.0) -> XGBClassifier:
    """Train XGBoost with class-imbalance handling."""
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
        eval_metric="auc",
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 200) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def train_isolation_forest(X_train: pd.DataFrame, contamination: float = 0.01) -> IsolationForest:
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def evaluate_classifier(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "model": name,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    LOG.info("[%s] precision=%.3f recall=%.3f f1=%.3f roc_auc=%.3f",
             name, metrics["precision"], metrics["recall"], metrics["f1"], metrics["roc_auc"] or 0)
    return metrics


def save_models(models: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, mdl in models.items():
        path = out_dir / f"{name}.joblib"
        joblib.dump(mdl, path)
        LOG.info("Saved %s -> %s", name, path)


def save_metrics(metrics: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    LOG.info("Saved metrics -> %s", out_path)


def save_final_summary(
    metrics: list[dict],
    xgb_model,
    out_path: Path,
    df: pd.DataFrame,
    feats: pd.DataFrame,
    labels: pd.Series,
    runtime_sec: float | None = None,
) -> None:
    """Write the consolidated `final_metrics.json` with model metrics,
    top-10 features, dataset stats, and runtime. This is the file
    referenced in the solution PDF and the README."""
    by_model = {m["model"]: m for m in metrics}
    xgb_m = by_model.get("xgboost", {})
    rf_m = by_model.get("random_forest", {})

    # Top-10 XGBoost features by gain
    top_feats: list[tuple[str, float]] = []
    if hasattr(xgb_model, "feature_importances_"):
        names = list(feats.columns)
        gains = xgb_model.feature_importances_.astype(float)
        order = gains.argsort()[::-1][:10]
        top_feats = [(names[i], float(gains[i])) for i in order]

    n_mule = int(labels.sum())
    summary = {
        "xgb": {
            "precision": xgb_m.get("precision"),
            "recall": xgb_m.get("recall"),
            "f1": xgb_m.get("f1"),
            "roc_auc": xgb_m.get("roc_auc"),
            "confusion": xgb_m.get("confusion_matrix"),
        },
        "rf": {
            "precision": rf_m.get("precision"),
            "recall": rf_m.get("recall"),
            "f1": rf_m.get("f1"),
            "roc_auc": rf_m.get("roc_auc"),
        },
        "isolation_forest": {
            k: v for k, v in by_model.get("isolation_forest", {}).items()
            if k != "confusion_matrix"
        },
        "top_features": top_feats,
        "dataset": "synthetic (PaySim-style) — fallback when PaySim CSV not present",
        "rows_total": int(len(df)),
        "rows_fraud": int(df["isFraud"].sum()),
        "accounts_total": int(len(feats)),
        "accounts_mule": n_mule,
        "fraud_ratio_pct": round(100 * df["isFraud"].mean(), 3),
        "runtime_seconds_approx": runtime_sec,
        "machine": "B.Tech laptop, 4 GB RAM, no GPU",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    LOG.info("Saved final summary -> %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from data_loader import load_paysim
    from features import build_features, label_accounts
    from rules import apply_rules

    df = load_paysim("data", sample_n=200_000)
    feats = build_features(df)
    labels = label_accounts(df, feats)
    rule_flags = apply_rules(df)
    # Join rule hits as features
    feats_full = feats.join(rule_flags, how="left").fillna(0)

    X_train, X_test, y_train, y_test = split_features(feats_full, labels)

    LOG.info("Train size: %d, Test size: %d, Fraud ratio train: %.3f%%",
             len(X_train), len(X_test), 100 * y_train.mean())

    # Compute scale_pos_weight
    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(int((y_train == 0).sum()), 1)
    spw = n_neg / n_pos

    xgb = train_xgboost(X_train, y_train, scale_pos_weight=spw)
    rf = train_random_forest(X_train, y_train)
    iso = train_isolation_forest(X_train)

    metrics = []
    metrics.append(evaluate_classifier(xgb, X_test, y_test, "xgboost"))
    metrics.append(evaluate_classifier(rf, X_test, y_test, "random_forest"))

    # IsolationForest: -1 = anomaly, 1 = normal
    y_iso = np.where(iso.predict(X_test) == -1, 1, 0)
    iso_metrics = {
        "model": "isolation_forest",
        "precision": float(precision_score(y_test, y_iso, zero_division=0)),
        "recall": float(recall_score(y_test, y_iso, zero_division=0)),
        "f1": float(f1_score(y_test, y_iso, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_iso).tolist(),
    }
    LOG.info("[isolation_forest] precision=%.3f recall=%.3f f1=%.3f",
             iso_metrics["precision"], iso_metrics["recall"], iso_metrics["f1"])
    metrics.append(iso_metrics)

    save_models({"xgboost": xgb, "random_forest": rf, "isolation_forest": iso},
                Path("models"))
    save_metrics(metrics, Path("reports/metrics.json"))
    print("Done.")
