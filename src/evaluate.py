"""evaluate.py — Generate confusion matrix, ROC curve, and feature importance
plots. Saves them to reports/figures/.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

LOG = logging.getLogger("muleshield.evaluate")


def plot_confusion_matrix(model, X_test, y_test, name: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}")
    out_path = out_dir / f"cm_{name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", out_path)


def plot_roc(models: dict, X_test, y_test, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, mdl in models.items():
        if hasattr(mdl, "predict_proba"):
            RocCurveDisplay.from_estimator(mdl, X_test, y_test, ax=ax, name=name)
    ax.set_title("ROC Curves — Classifier Comparison")
    ax.grid(alpha=0.3)
    out_path = out_dir / "roc_comparison.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", out_path)


def plot_feature_importance(model, feature_names, name: str, out_dir: Path, top_n: int = 15) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(order)), importances[order][::-1], color="#0B3D91")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features — {name}")
    fig.tight_layout()
    out_path = out_dir / f"feature_importance_{name}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", out_path)


def run(models_dir: Path = Path("models"),
        data_dir: Path = Path("data"),
        reports_dir: Path = Path("reports"),
        sample_n: int = 200_000) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.data_loader import load_paysim
    from src.features import build_features, label_accounts
    from src.rules import apply_rules

    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_paysim(data_dir, sample_n=sample_n)
    feats = build_features(df)
    labels = label_accounts(df, feats)
    rule_flags = apply_rules(df)
    feats_full = feats.join(rule_flags, how="left").fillna(0)

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(feats_full, labels, test_size=0.2,
                                            random_state=42, stratify=labels)

    models = {
        n: joblib.load(models_dir / f"{n}.joblib")
        for n in ("xgboost", "random_forest")
        if (models_dir / f"{n}.joblib").exists()
    }

    for name, mdl in models.items():
        plot_confusion_matrix(mdl, X_test, y_test, name, fig_dir)
        plot_feature_importance(mdl, feats_full.columns.tolist(), name, fig_dir)

    if len(models) > 1:
        plot_roc(models, X_test, y_test, fig_dir)


if __name__ == "__main__":
    run()
