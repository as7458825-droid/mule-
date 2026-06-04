"""rules.py — 7-rule expert rule engine for mule detection.

Each rule returns a boolean Series indexed by account (nameOrig).
The `apply_rules` function combines all rules into a per-account hit
count, which the model uses as an additional feature.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOG = logging.getLogger("muleshield.rules")


def rule_dormant_activation(df: pd.DataFrame) -> pd.Series:
    """R1: Account inactive 6+ months (180 days) suddenly receives credits."""
    grouped = df.groupby("nameOrig")
    max_gap = grouped["step"].apply(lambda s: s.sort_values().diff().max() or 0)
    high_value = df[df["amount"] > df["amount"].quantile(0.95)].groupby("nameOrig").size()
    flagged = (max_gap > 180) & high_value.reindex(max_gap.index, fill_value=0).gt(0)
    return flagged.fillna(False).rename("R1_dormant_activation")


def rule_smurfing(df: pd.DataFrame) -> pd.Series:
    """R2: Many small credits followed by one large debit."""
    grouped = df.groupby("nameOrig")
    n_credits = grouped.apply(lambda g: (g["amount"] < g["amount"].quantile(0.5)).sum(),
                              include_groups=False)
    n_large_debits = grouped.apply(
        lambda g: ((g["amount"] > g["amount"].quantile(0.95)) & (g["type"].isin(["CASH_OUT", "TRANSFER"]))).sum(),
        include_groups=False,
    )
    flagged = (n_credits > 5) & (n_large_debits > 0)
    return flagged.fillna(False).rename("R2_smurfing")


def rule_odd_hour(df: pd.DataFrame) -> pd.Series:
    """R3: High-value TXN between 1 AM and 5 AM (PaySim hour 1-5)."""
    night = df[(df["hour"] >= 1) & (df["hour"] <= 5) & (df["amount"] > df["amount"].quantile(0.90))]
    counts = night.groupby("nameOrig").size()
    flagged = counts > 2
    return flagged.reindex(df["nameOrig"].unique(), fill_value=False).rename("R3_odd_hour")


def rule_new_beneficiary(df: pd.DataFrame) -> pd.Series:
    """R4: First transfer to a new beneficiary above threshold."""
    threshold = df["amount"].quantile(0.95)
    df_sorted = df.sort_values(["nameOrig", "step"])
    first_seen = df_sorted.groupby(["nameOrig", "nameDest"]).first()
    flagged = (
        first_seen.reset_index()
        .groupby("nameOrig")
        .apply(lambda g: ((g["amount"] > threshold)).sum() > 2, include_groups=False)
    )
    return flagged.reindex(df["nameOrig"].unique(), fill_value=False).rename("R4_new_beneficiary")


def rule_new_account_high_activity(df: pd.DataFrame) -> pd.Series:
    """R5: Account < 30 days, volume > average."""
    avg_volume = df.groupby("nameOrig")["amount"].sum().mean()
    flagged = (df.groupby("nameOrig")["step"].max() <= 30) & (
        df.groupby("nameOrig")["amount"].sum() > avg_volume
    )
    return flagged.reindex(df["nameOrig"].unique(), fill_value=False).rename("R5_new_account_high_activity")


def rule_rapid_movement(df: pd.DataFrame) -> pd.Series:
    """R6: Funds received and forwarded within 24h, repeatedly."""
    df_sorted = df.sort_values(["nameOrig", "step"])
    diffs = df_sorted.groupby("nameOrig")["step"].diff()
    df_sorted = df_sorted.assign(_diff=diffs)
    fast_pairs = df_sorted[df_sorted["_diff"] <= 24]
    flagged = fast_pairs.groupby("nameOrig").size() > 3
    return flagged.reindex(df["nameOrig"].unique(), fill_value=False).rename("R6_rapid_movement")


def rule_cross_channel_anomaly(df: pd.DataFrame) -> pd.Series:
    """R7: Activity across UPI + NEFT + ATM simultaneously (>= 3 channels)."""
    n_channels = df.groupby("nameOrig")["channel"].nunique()
    flagged = n_channels >= 3
    return flagged.reindex(df["nameOrig"].unique(), fill_value=False).rename("R7_cross_channel")


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Run all 7 rules and return a per-account boolean DataFrame."""
    accounts = df["nameOrig"].unique()
    out = pd.DataFrame(index=accounts)
    out.index.name = "nameOrig"

    rules = [
        rule_dormant_activation,
        rule_smurfing,
        rule_odd_hour,
        rule_new_beneficiary,
        rule_new_account_high_activity,
        rule_rapid_movement,
        rule_cross_channel_anomaly,
    ]
    for fn in rules:
        try:
            res = fn(df)
            out[res.name] = res.reindex(accounts, fill_value=False)
        except Exception as e:
            LOG.warning("Rule %s failed: %s", fn.__name__, e)
            out[fn.__name__] = False

    out["rule_hit_count"] = out.sum(axis=1)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_loader import load_paysim
    df = load_paysim("data", sample_n=50_000)
    flags = apply_rules(df)
    print(flags.head())
    print("Hit-count distribution:", flags["rule_hit_count"].value_counts().sort_index())
